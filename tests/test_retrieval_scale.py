"""Scale/perf guardrails for the `chirp ask` retrieval path (AC-5, AC-6).

These assert that repeated queries against an unchanged index do not redo
O(corpus) work, and that a burst of note saves does not trigger one full-corpus
BM25 rebuild per save. They use the real index/bm25 code with a fake embed
client — no daemon, no real model.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from config.settings import ChirpSettings
from notes_chat import bm25 as bm25_module
from notes_chat import retrieval as retrieval_module
from notes_chat.index import IndexManager
from notes_chat.retrieval import _search_bm25, retrieve_context

_NOTE_BODY = (
    "# Project Sync\n\nThis note has enough content about budgets and timelines "
    "to be chunked and indexed for hybrid retrieval scale testing purposes."
)


class _FakeEmbedClient:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_sync(self, inputs, model="default"):
        return [[0.1] * self.dim for _ in inputs]


def _make_config(tmp_path: Path) -> ChirpSettings:
    config = ChirpSettings()
    config.directories.notes_root = tmp_path / "notes"
    config.directories.notes_root.mkdir(parents=True, exist_ok=True)
    config.notes_chat.index_dir = tmp_path / ".notes_index"
    return config


def _seed_note(notes_root: Path, slug: str) -> Path:
    note_dir = notes_root / slug
    note_dir.mkdir(parents=True, exist_ok=True)
    note_file = note_dir / "notes.md"
    note_file.write_text(_NOTE_BODY)
    return note_file


class TestBm25ModelCache:
    def test_unchanged_index_is_not_re_tokenized(self, tmp_path):
        bm25_file = tmp_path / "bm25.json"
        bm25_file.write_text(
            json.dumps(
                {
                    "doc_ids": ["d1", "d2"],
                    "corpus": ["budget timeline project", "roadmap planning sync"],
                }
            )
        )
        bm25_module._MODEL_CACHE.pop(str(bm25_file), None)

        with patch.object(bm25_module, "BM25Okapi", wraps=bm25_module.BM25Okapi) as spy:
            _search_bm25(bm25_file, "budget", k=5)
            _search_bm25(bm25_file, "roadmap", k=5)
            _search_bm25(bm25_file, "project", k=5)

        # Built once, then cached by (mtime, size) for later queries.
        assert spy.call_count == 1


class TestFreshnessShortCircuit:
    def test_unchanged_tree_skips_index_rebuild_scan(self, tmp_path):
        config = _make_config(tmp_path)
        _seed_note(config.directories.notes_root, "sync-2025-01-15")

        retrieval_module._FRESHNESS_CACHE.pop(str(config.notes_chat.index_dir), None)

        IndexManager(config, llm_client=_FakeEmbedClient()).build_index()

        with (
            patch("notes_chat.retrieval._get_query_embedding", return_value=[0.1] * 8),
            patch("notes_chat.retrieval.IndexManager.build_index") as mock_build,
        ):
            mock_build.return_value = {"success": True}
            retrieve_context(config, "budget")
            # Second query: tree unchanged, so build_index is skipped.
            retrieve_context(config, "timeline")

        assert mock_build.call_count == 1

    def test_in_place_edit_invalidates_freshness(self, tmp_path):
        """M1: editing `<slug>/notes.md` in place must invalidate the signature.

        The dir mtime is unchanged by an in-place file write, so a signature
        keyed only on (dir mtime, child count) would never notice — a long-lived
        process would serve stale chunks. The signature folds each notes.md
        (mtime, size), so it changes here.
        """
        import os
        import time

        config = _make_config(tmp_path)
        note_file = _seed_note(config.directories.notes_root, "sync-2025-01-15")

        sig_before = retrieval_module._notes_tree_signature(config)
        retrieval_module._record_index_freshness(config)
        assert retrieval_module._index_is_fresh(config)

        # Bump mtime too, so the edit is observable even if the size matches.
        note_file.write_text("# Sync\n\nCompletely new body content after an edit.")
        future = time.time() + 5
        os.utime(note_file, (future, future))

        sig_after = retrieval_module._notes_tree_signature(config)
        assert sig_after != sig_before
        assert not retrieval_module._index_is_fresh(config)


class TestBm25RebuildCadence:
    def test_burst_of_saves_does_not_full_rebuild_per_save(self, tmp_path):
        """N auto-index saves => N cheap appends, never N full-corpus rebuilds.

        Uses the REAL Chroma collection (now that the chunk-id collision is
        fixed, distinct notes produce distinct ids): each save adds the note's
        chunks and appends them to the lexicon. ``append_bm25_index`` is called
        once per save and ``rebuild_bm25_index`` is never called.
        """
        config = _make_config(tmp_path)
        notes_root = config.directories.notes_root
        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        with (
            patch("notes_chat.bm25.rebuild_bm25_index") as full_rebuild,
            patch(
                "notes_chat.bm25.append_bm25_index",
                wraps=bm25_module.append_bm25_index,
            ) as append_spy,
        ):
            for i in range(5):
                note_file = _seed_note(notes_root, f"note-{i}-2025-01-1{i}")
                assert manager._add_to_index(note_file)
                manager.append_bm25_for_file(str(note_file))

        full_rebuild.assert_not_called()
        assert append_spy.call_count == 5

        with (config.notes_chat.index_dir / "bm25.json").open() as f:
            data = json.load(f)
        assert len(data["doc_ids"]) == 5
        assert len(set(data["doc_ids"])) == 5
