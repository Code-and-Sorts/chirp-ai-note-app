import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import tomli_w

from config.settings import ChirpSettings
from llm.exceptions import LLMTransportError
from notes_chat.index import IndexManager

_NOTE_BODY = (
    "# Test Meeting\n\nThis is some longer content that should be sufficient "
    "for chunking and indexing purposes. It contains enough text to pass the "
    "minimum length requirements."
)


class _FakeEmbedClient:
    """Fake LLMClient.embed_sync: one vector per input, in order."""

    def __init__(self, dim: int = 384, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail

    def embed_sync(self, inputs, model="default"):
        if self.fail:
            raise LLMTransportError("embed daemon unavailable")
        return [[0.1] * self.dim for _ in inputs]


def _make_config(tmp_path):
    config = ChirpSettings()
    config.directories.notes_root = tmp_path
    config.notes_chat.index_dir = tmp_path / ".notes_index"
    return config


def _seed_note(tmp_path: Path) -> Path:
    note_dir = tmp_path / "test-2026-04-20"
    note_dir.mkdir()
    note_file = note_dir / "notes.md"
    note_file.write_text(_NOTE_BODY)
    return note_file


def _seed_named_note(tmp_path: Path, slug: str, body: str | None = None) -> Path:
    note_dir = tmp_path / slug
    note_dir.mkdir()
    note_file = note_dir / "notes.md"
    note_file.write_text(body if body is not None else _NOTE_BODY)
    return note_file


class TestIndexManifest:
    def test_signature_calculation(self, tmp_path):
        """Test file signature calculation."""
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)

        manager = IndexManager(config)
        files = manager._scan_notes_files()

        assert str(note_file) in files
        assert "mtime" in files[str(note_file)]
        assert "size" in files[str(note_file)]
        assert files[str(note_file)]["size"] > 0

    def test_metadata_extraction(self, tmp_path):
        """Test extraction of metadata from notes files."""
        config = _make_config(tmp_path)

        content = """# Weekly Standup Meeting

**Duration:** 45m
**Participants:** Alice, Bob, Charlie

## Summary
Test meeting content
"""

        note_dir = tmp_path / "weekly-standup-2025-01-15"
        note_dir.mkdir()
        note_file = note_dir / "notes.md"
        note_file.write_text(content)

        manager = IndexManager(config)
        meta = manager._extract_metadata(note_file, content)

        assert meta is not None
        assert meta.title == "Weekly Standup Meeting"
        assert meta.duration == 45
        assert "Alice" in meta.participants
        assert "Bob" in meta.participants

    def test_idempotent_skip(self, tmp_path):
        """Test that unchanged files are skipped."""
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        result1 = manager.build_index()
        assert result1["success"]
        assert result1["files_processed"] == 1

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["files_processed"] == 0
        assert "up to date" in result2["message"]

    def test_force_rebuild(self, tmp_path):
        """Test --force rebuild behavior."""
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        result1 = manager.build_index()
        assert result1["success"]

        result2 = manager.build_index(force=True)
        assert result2["success"]
        assert result2["files_processed"] == 1

    def test_file_removal_detection(self, tmp_path):
        """Test detection of removed files."""
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        result1 = manager.build_index()
        assert result1["success"]

        note_file.unlink()

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["removed"] == 1

    def test_file_modification_detection(self, tmp_path):
        """Test detection of modified files."""
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        result1 = manager.build_index()
        assert result1["success"]

        import time

        time.sleep(0.1)
        note_file.write_text("# Test Meeting\n\nModified content")

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["modified"] == 1

    def test_embedding_failure_handling(self, tmp_path):
        """A failing embed leaves build_index successful but indexes nothing."""
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient(fail=True))
        result = manager.build_index()

        assert result["success"]
        assert result["files_processed"] == 0  # embed failed -> nothing indexed

    def test_meeting_date_from_meta_not_file_mtime(self, tmp_path):
        """Stored date is the meeting date even when notes.md was touched today.

        Regression for AC-2: `_extract_metadata` used to regex the date out of
        `file_path.name` (always "notes.md" → never matched → fell back to the
        file's mtime). The canonical date is `meta.toml`'s `date`.
        """
        config = _make_config(tmp_path)
        note_dir = tmp_path / "weekly-standup-2025-01-15"
        note_dir.mkdir()
        note_file = note_dir / "notes.md"
        note_file.write_text(_NOTE_BODY)
        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump({"title": "Weekly Standup", "date": "2025-01-15"}, fh)

        # Touch notes.md to today so an mtime-derived date would be wrong.
        today = datetime.now().timestamp()
        os.utime(note_file, (today, today))

        manager = IndexManager(config)
        meta = manager._extract_metadata(note_file, _NOTE_BODY)

        assert meta is not None
        assert meta.date.date() == datetime(2025, 1, 15).date()
        assert meta.date.tzinfo is None

    def test_meeting_date_filter_round_trip(self, tmp_path):
        """`on:<meeting-date>` matches the note; `on:<mtime-day>` does not."""
        config = _make_config(tmp_path)
        note_dir = tmp_path / "review-2025-01-15"
        note_dir.mkdir()
        note_file = note_dir / "notes.md"
        note_file.write_text(_NOTE_BODY)
        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump({"title": "Review", "date": "2025-01-15"}, fh)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())
        assert manager.build_index()["success"]

        stored = manager.collection.get(include=["metadatas"])
        dates = {m["date"][:10] for m in stored["metadatas"]}
        assert dates == {"2025-01-15"}

    def test_two_notes_both_indexed_no_chunk_id_collision(self, tmp_path):
        """Two distinct notes both land in Chroma — chunk ids must be unique.

        Regression for the chunk-id collision found during 8.2 review: ids were
        ``f"{file_path.stem}_NNN"`` and the stem is always "notes", so every
        note's chunks collided on ``notes_000`` and Chroma's add silently kept
        only one note's chunks. Ids are now slug-prefixed (parent dir name).
        This FAILS on the old scheme (num docs == 1) and PASSES now.
        """
        config = _make_config(tmp_path)
        _seed_named_note(
            tmp_path,
            "alpha-sync-2025-01-15",
            "# Alpha Sync\n\nLong enough body about alpha budgets and timelines "
            "for chunking and indexing this distinct note.",
        )
        _seed_named_note(
            tmp_path,
            "beta-review-2025-02-10",
            "# Beta Review\n\nLong enough body about beta roadmap and planning "
            "for chunking and indexing this distinct note.",
        )

        manager = IndexManager(config, llm_client=_FakeEmbedClient())
        result = manager.build_index()
        assert result["success"]
        assert result["files_processed"] == 2

        stored = manager.collection.get(include=["metadatas"])
        ids = stored["ids"]

        assert len(ids) >= 2
        assert len(ids) == len(set(ids))
        assert any(chunk_id.startswith("alpha-sync-2025-01-15_") for chunk_id in ids)
        assert any(chunk_id.startswith("beta-review-2025-02-10_") for chunk_id in ids)

        slugs_from_paths = {Path(m["path"]).parent.name for m in stored["metadatas"]}
        assert slugs_from_paths == {"alpha-sync-2025-01-15", "beta-review-2025-02-10"}

    def test_manifest_records_only_successful_files(self, tmp_path):
        """A file that fails to embed is absent from the manifest and retried.

        Regression for AC-9: the manifest used to be written wholesale, so a
        failed embed was recorded as "indexed" and never retried.
        """
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient(fail=True))
        result = manager.build_index()

        assert result["success"]
        assert result["failed"] == 1
        assert str(note_file) not in manager._load_manifest()

        # A later run with a working embed must retry the previously-failed file.
        manager_ok = IndexManager(config, llm_client=_FakeEmbedClient())
        retry = manager_ok.build_index()
        assert retry["files_processed"] == 1
        assert str(note_file) in manager_ok._load_manifest()

    def test_force_rebuild_interrupted_leaves_prior_index_intact(self, tmp_path):
        """An interrupted --force keeps the previous manifest/collection usable.

        Regression for AC-8: `_reset_index` unlinked the manifest/bm25 up front,
        so a crash mid-rebuild left "No search index found".
        """
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())
        assert manager.build_index()["success"]
        manifest_before = manager._load_manifest()
        assert manifest_before

        # Fault-inject an interruption mid-rebuild: blow up inside the add loop.
        boom = IndexManager(config, llm_client=_FakeEmbedClient())
        original_add = boom._add_to_index

        def exploding_add(path):
            original_add(path)
            raise KeyboardInterrupt("ctrl-c mid rebuild")

        boom._add_to_index = exploding_add
        try:
            boom.build_index(force=True)
        except KeyboardInterrupt:
            # simulated ctrl-c mid-rebuild
            pass

        survivor = IndexManager(config)
        assert survivor.manifest_file.exists()
        assert survivor._load_manifest() == manifest_before
        assert survivor.collection.get()["ids"]

    def test_embed_model_change_is_detected(self, tmp_path):
        """A changed embed fingerprint is detected on an incremental build.

        Regression for AC-10: new-dim/new-model vectors silently mismatched the
        old ones. The collection now stores an embed fingerprint and an
        incremental build raises an actionable error on mismatch.
        """
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())
        manager._resolved_embed_alias = lambda: "bge-small"
        assert manager.build_index(force=True)["success"]

        # Fingerprint records "bge-small"; a later build with a different alias
        # must detect the change.
        reopened = IndexManager(config, llm_client=_FakeEmbedClient())
        reopened._resolved_embed_alias = lambda: "nomic-embed"
        note_dir = tmp_path / "test-2026-04-20"
        (note_dir / "notes.md").write_text(_NOTE_BODY + "\n\nmore content here today")

        result = reopened.build_index()
        assert result["success"] is False
        assert "embed model changed" in result["error"].lower()

    def test_incremental_build_stamps_fingerprint(self, tmp_path):
        """A first incremental build stamps the embed fingerprint (AC-10).

        Without this, an index first populated incrementally (or via auto-index)
        carries no fingerprint and a later model change is undetectable.
        """
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())
        manager._resolved_embed_alias = lambda: "bge-small"
        assert manager.build_index()["success"]

        reopened = IndexManager(config)
        alias, dim = reopened._stored_fingerprint()
        assert alias == "bge-small"
        assert dim == 384


class TestSingleNoteIndexing:
    def test_add_note_incremental_with_guard(self, tmp_path):
        """AC-6 / L1: guarded incremental add appends BM25 and never rebuilds."""
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)
        manager = IndexManager(config)

        with (
            patch.object(manager, "_ensure_embed_fingerprint") as ensure_fp,
            patch.object(manager, "_add_to_index", return_value=True) as add_idx,
            patch.object(manager, "_save_manifest") as save_manifest,
            patch.object(manager, "append_bm25_for_file") as append_bm25,
            patch.object(manager, "_rebuild_bm25") as rebuild_bm25,
            patch.object(manager, "_stamp_fingerprint_if_missing") as stamp_fp,
        ):
            result = manager.add_note(
                note_file, guard_embed_fingerprint=True, incremental_bm25=True
            )

        assert result is True
        ensure_fp.assert_called_once()
        add_idx.assert_called_once_with(note_file)
        save_manifest.assert_called_once()
        append_bm25.assert_called_once_with(str(note_file))
        stamp_fp.assert_called_once()
        rebuild_bm25.assert_not_called()

    def test_add_note_defaults_full_rebuild_no_guard(self, tmp_path):
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)
        manager = IndexManager(config)

        with (
            patch.object(manager, "_ensure_embed_fingerprint") as ensure_fp,
            patch.object(manager, "_add_to_index", return_value=True),
            patch.object(manager, "_save_manifest"),
            patch.object(manager, "append_bm25_for_file") as append_bm25,
            patch.object(manager, "_rebuild_bm25") as rebuild_bm25,
        ):
            result = manager.add_note(note_file)

        assert result is True
        ensure_fp.assert_not_called()
        rebuild_bm25.assert_called_once()
        append_bm25.assert_not_called()

    def test_add_note_returns_false_when_index_fails(self, tmp_path):
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)
        manager = IndexManager(config)

        with (
            patch.object(manager, "_add_to_index", return_value=False),
            patch.object(manager, "_save_manifest") as save_manifest,
            patch.object(manager, "_rebuild_bm25") as rebuild_bm25,
            patch.object(manager, "append_bm25_for_file") as append_bm25,
        ):
            result = manager.add_note(note_file)

        assert result is False
        save_manifest.assert_not_called()
        rebuild_bm25.assert_not_called()
        append_bm25.assert_not_called()

    def test_add_note_propagates_embed_model_change(self, tmp_path):
        from chirp.exceptions import EmbedModelChanged

        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)
        manager = IndexManager(config)

        with (
            patch.object(
                manager,
                "_ensure_embed_fingerprint",
                side_effect=EmbedModelChanged("changed"),
            ),
            patch.object(manager, "_add_to_index") as add_idx,
            pytest.raises(EmbedModelChanged),
        ):
            manager.add_note(note_file, guard_embed_fingerprint=True)

        add_idx.assert_not_called()

    def test_remove_note(self, tmp_path):
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)
        manager = IndexManager(config)

        with (
            patch.object(manager, "_remove_from_index") as remove_idx,
            patch.object(manager, "_save_manifest") as save_manifest,
            patch.object(manager, "_rebuild_bm25") as rebuild_bm25,
        ):
            manager.remove_note(note_file)

        remove_idx.assert_called_once_with(str(note_file))
        save_manifest.assert_called_once()
        rebuild_bm25.assert_called_once()
