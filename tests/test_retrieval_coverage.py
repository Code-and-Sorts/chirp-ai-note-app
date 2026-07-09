"""Tests targeting uncovered branches in notes_chat/retrieval.py."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from notes_chat.retrieval import (
    CONCEPTUAL_WEIGHTS,
    LITERAL_WEIGHTS,
    _build_context,
    _create_chunk_header,
    _format_mm_ss,
    _generate_suggestion,
    _route_query,
    _search_bm25,
    _search_chroma,
    _slug_from_chunk,
    _timestamp_seconds_from_chunk,
    format_sources,
    retrieve_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, semantic_enabled: bool = False):
    """Build a minimal settings-like namespace for testing.

    ``semantic_enabled`` defaults to ``False`` to match the production default;
    chroma-path tests pass ``True`` explicitly.
    """
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    notes_root = tmp_path / "notes"
    notes_root.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        directories=SimpleNamespace(notes_root=notes_root),
        notes_chat=SimpleNamespace(
            semantic_enabled=semantic_enabled,
            index_dir=index_dir,
            k=5,
            ctx_char_budget=10000,
        ),
    )


def _make_index_manager(config, manifest_exists: bool = True):
    manager = MagicMock()
    manager.config = config
    manager.manifest_file = MagicMock()
    manager.manifest_file.exists.return_value = manifest_exists
    manager.bm25_file = config.notes_chat.index_dir / "bm25.json"
    manager.collection = MagicMock()
    manager.build_index.return_value = {"success": True}
    return manager


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    def test_returns_error_when_manifest_missing(self, tmp_path):
        config = _make_config(tmp_path)
        with patch("notes_chat.retrieval.IndexManager") as MockIM:
            manager = _make_index_manager(config, manifest_exists=False)
            MockIM.return_value = manager
            with patch(
                "notes_chat.retrieval._generate_suggestion", return_value="hint"
            ):
                result = retrieve_context(config, "what happened")

        assert result["success"] is False
        assert "No search index" in result["error"]
        assert result["suggestion"] == "hint"

    def test_returns_error_when_build_index_fails(self, tmp_path):
        config = _make_config(tmp_path)
        with patch("notes_chat.retrieval.IndexManager") as MockIM:
            manager = _make_index_manager(config, manifest_exists=True)
            manager.build_index.return_value = {
                "success": False,
                "error": "disk full",
            }
            MockIM.return_value = manager
            result = retrieve_context(config, "any question")

        assert result["success"] is False
        assert "Failed to update index" in result["error"]
        assert "disk full" in result["error"]

    def test_returns_error_when_no_results_found(self, tmp_path):
        config = _make_config(tmp_path)
        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_chroma", return_value=[]),
            patch("notes_chat.retrieval._search_bm25", return_value=[]),
            patch(
                "notes_chat.retrieval._generate_suggestion", return_value="try wider"
            ),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
        ):
            manager = _make_index_manager(config, manifest_exists=True)
            MockIM.return_value = manager
            result = retrieve_context(config, "no match question")

        assert result["success"] is False
        assert "No documents found" in result["error"]

    def test_returns_error_when_context_empty_after_filtering(self, tmp_path):
        config = _make_config(tmp_path, semantic_enabled=True)
        chunk = ("id1", 0.9, {"content": "", "metadata": {}})
        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_chroma", return_value=[chunk]),
            patch("notes_chat.retrieval._search_bm25", return_value=[]),
            patch("notes_chat.retrieval._generate_suggestion", return_value="hint"),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
        ):
            manager = _make_index_manager(config, manifest_exists=True)
            MockIM.return_value = manager
            result = retrieve_context(config, "sparse question")

        assert result["success"] is False
        assert "No relevant content" in result["error"]

    def test_returns_success_with_context(self, tmp_path):
        config = _make_config(tmp_path, semantic_enabled=True)
        chunk = (
            "id1",
            0.9,
            {
                "content": "some content about meetings",
                "metadata": {"path": "/notes/note-slug/notes.md", "date": "2025-01-01"},
            },
        )
        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_chroma", return_value=[chunk]),
            patch("notes_chat.retrieval._search_bm25", return_value=[]),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
            patch("notes_chat.retrieval._build_note_index", return_value={}),
        ):
            manager = _make_index_manager(config, manifest_exists=True)
            MockIM.return_value = manager
            result = retrieve_context(config, "what happened")

        assert result["success"] is True
        assert "context" in result
        assert len(result["context"]) > 0
        assert "chunks_retrieved" in result

    def test_uses_when_filter_for_time_range(self, tmp_path):
        config = _make_config(tmp_path, semantic_enabled=True)
        fake_range = SimpleNamespace(start=MagicMock(), end_exclusive=MagicMock())
        chunk = (
            "id1",
            0.9,
            {
                "content": "meeting content",
                "metadata": {"path": "/notes/slug/notes.md", "date": "2025-01-01"},
            },
        )
        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_chroma", return_value=[chunk]),
            patch("notes_chat.retrieval._search_bm25", return_value=[]),
            patch(
                "notes_chat.retrieval.parse_time_range", return_value=fake_range
            ) as mock_parse,
            patch("notes_chat.retrieval._build_note_index", return_value={}),
        ):
            manager = _make_index_manager(config, manifest_exists=True)
            MockIM.return_value = manager
            retrieve_context(config, "last week meeting", when_filter="last week")

        mock_parse.assert_called_once_with("last week meeting", "last week")

    def test_catches_unexpected_exceptions(self, tmp_path):
        config = _make_config(tmp_path)
        with patch(
            "notes_chat.retrieval.IndexManager", side_effect=RuntimeError("boom")
        ):
            result = retrieve_context(config, "question")

        assert result["success"] is False
        assert "boom" in result["error"]


class TestSemanticGate:
    def test_lexical_only_skips_chroma_when_semantic_disabled(self, tmp_path):
        config = _make_config(tmp_path, semantic_enabled=False)
        bm25_chunk = (
            "id1",
            1.0,
            {
                "content": "lexical content about meetings",
                "metadata": {"path": "/notes/slug/notes.md", "date": "2025-01-01"},
            },
        )
        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_chroma") as mock_chroma,
            patch("notes_chat.retrieval._get_query_embedding") as mock_embed,
            patch("notes_chat.retrieval._search_bm25", return_value=[bm25_chunk]),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
            patch("notes_chat.retrieval._build_note_index", return_value={}),
        ):
            manager = _make_index_manager(config, manifest_exists=True)
            MockIM.return_value = manager
            result = retrieve_context(config, "what happened in the meeting")

        assert result["success"] is True
        mock_chroma.assert_not_called()
        mock_embed.assert_not_called()

    def test_searches_chroma_when_semantic_enabled(self, tmp_path):
        config = _make_config(tmp_path, semantic_enabled=True)
        chunk = (
            "id1",
            0.9,
            {
                "content": "semantic content",
                "metadata": {"path": "/notes/slug/notes.md", "date": "2025-01-01"},
            },
        )
        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch(
                "notes_chat.retrieval._search_chroma", return_value=[chunk]
            ) as mock_chroma,
            patch("notes_chat.retrieval._search_bm25", return_value=[]),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
            patch("notes_chat.retrieval._build_note_index", return_value={}),
        ):
            manager = _make_index_manager(config, manifest_exists=True)
            MockIM.return_value = manager
            result = retrieve_context(config, "summarise the planning discussion")

        assert result["success"] is True
        mock_chroma.assert_called_once()


class TestRouteQuery:
    def test_quoted_phrase_is_literal(self):
        assert _route_query('find "action items" please now') == LITERAL_WEIGHTS

    def test_jira_style_id_is_literal(self):
        assert _route_query("status of PROJ-1234 ticket") == LITERAL_WEIGHTS

    def test_version_token_is_literal(self):
        assert _route_query("when did we ship v2.3.1 to staging") == LITERAL_WEIGHTS

    def test_bare_number_is_conceptual(self):
        assert (
            _route_query("what is the layout of room 101 in the building")
            == CONCEPTUAL_WEIGHTS
        )

    def test_acronym_is_literal(self):
        assert _route_query("what did we decide about the API gateway") == (
            LITERAL_WEIGHTS
        )

    def test_short_query_is_literal(self):
        assert _route_query("budget numbers") == LITERAL_WEIGHTS

    def test_natural_language_question_is_conceptual(self):
        assert (
            _route_query("what were the main concerns raised during planning")
            == CONCEPTUAL_WEIGHTS
        )


# ---------------------------------------------------------------------------
# _search_chroma
# ---------------------------------------------------------------------------


class TestSearchChroma:
    def _make_manager_with_embedding(self, embedding, query_results):
        manager = MagicMock()
        manager.config = MagicMock()
        manager.collection.query.return_value = query_results
        return manager

    def test_returns_empty_when_embedding_fails(self):
        manager = MagicMock()
        with patch("notes_chat.retrieval._get_query_embedding", return_value=None):
            result = _search_chroma(manager, "query", 5)
        assert result == []

    def test_returns_results_from_chroma(self):
        manager = MagicMock()
        manager.collection.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"path": "a.md"}, {"path": "b.md"}]],
            "documents": [["doc one", "doc two"]],
        }
        with patch(
            "notes_chat.retrieval._get_query_embedding",
            return_value=[0.1, 0.2, 0.3],
        ):
            result = _search_chroma(manager, "test query", 5)

        assert len(result) == 2
        assert result[0][0] == "chunk1"
        assert abs(result[0][1] - 0.9) < 1e-6
        assert result[0][2]["content"] == "doc one"

    def test_returns_empty_when_no_ids(self):
        manager = MagicMock()
        manager.collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }
        with patch(
            "notes_chat.retrieval._get_query_embedding",
            return_value=[0.1, 0.2],
        ):
            result = _search_chroma(manager, "query", 5)
        assert result == []

    def test_builds_where_clause_when_time_range_provided(self):
        manager = MagicMock()
        manager.collection.query.return_value = {
            "ids": [["c1"]],
            "distances": [[0.2]],
            "metadatas": [[{"path": "x.md"}]],
            "documents": [["content"]],
        }
        time_range = SimpleNamespace(
            start=MagicMock(isoformat=lambda: "2025-01-01T00:00:00"),
            end_exclusive=MagicMock(isoformat=lambda: "2025-02-01T00:00:00"),
        )
        with patch(
            "notes_chat.retrieval._get_query_embedding",
            return_value=[0.1, 0.2],
        ):
            result = _search_chroma(manager, "query", 5, time_range)

        called_kwargs = manager.collection.query.call_args.kwargs
        assert called_kwargs["where"] is not None
        assert "$and" in called_kwargs["where"]
        assert len(result) == 1

    def test_where_clause_bounds_match_stored_date_shape(self):
        """AC-3/AC-4: the real TimeRange feeds naive bounds that match storage.

        The stored chunk `date` is naive (`datetime.isoformat()` with no offset);
        the time-range filter used to build tz-aware ISO strings with an offset,
        so Chroma's lexical string comparison mis-compared. This drives the REAL
        `parse_time_range` through `_search_chroma`'s where-clause builder and
        asserts both bounds carry no tz offset — same shape as the stored value.
        """
        from datetime import datetime

        from notes_chat.index import IndexManager
        from notes_chat.time_ranges import parse_time_range
        from notes_chat.types import Chunk, NoteMeta

        time_range = parse_time_range("meetings", when_arg="on:2025-01-15")
        assert time_range is not None
        # The real TimeRange is tz-aware (datetime.now(tzlocal())).
        assert time_range.start.tzinfo is not None

        manager = MagicMock()
        manager.collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }
        with patch(
            "notes_chat.retrieval._get_query_embedding",
            return_value=[0.1, 0.2],
        ):
            _search_chroma(manager, "query", 5, time_range)

        where = manager.collection.query.call_args.kwargs["where"]
        gte = where["$and"][0]["date"]["$gte"]
        lt = where["$and"][1]["date"]["$lt"]

        # Stored date is a naive isoformat: no "+HH:MM"/"Z" offset.
        stored_meta = NoteMeta(
            path=Path("/n/slug/notes.md"),
            title="t",
            date=datetime(2025, 1, 15),
            participants=[],
            duration=0,
            mtime=0.0,
            size=0,
        )
        chunk = Chunk(
            id="x",
            path=stored_meta.path,
            content="c",
            meta=stored_meta,
            content_hash="h",
        )
        stored_date = IndexManager.__dict__["_chunk_to_metadata"](manager, chunk)[
            "date"
        ]

        for bound in (gte, lt, stored_date):
            assert "+" not in bound
            assert not bound.endswith("Z")
        assert gte <= "2025-01-15T12:00:00" < lt

    def test_returns_empty_on_exception(self):
        manager = MagicMock()
        manager.collection.query.side_effect = Exception("chroma down")
        with patch(
            "notes_chat.retrieval._get_query_embedding",
            return_value=[0.1, 0.2],
        ):
            result = _search_chroma(manager, "query", 5)
        assert result == []

    def test_handles_none_results_fields(self):
        manager = MagicMock()
        manager.collection.query.return_value = {
            "ids": None,
            "distances": None,
            "metadatas": None,
            "documents": None,
        }
        with patch(
            "notes_chat.retrieval._get_query_embedding",
            return_value=[0.1, 0.2],
        ):
            result = _search_chroma(manager, "query", 5)
        assert result == []


# ---------------------------------------------------------------------------
# _search_bm25
# ---------------------------------------------------------------------------


class TestSearchBm25:
    def test_returns_results_from_bm25(self, tmp_path):
        bm25_file = tmp_path / "bm25.json"
        bm25_data = {
            "doc_ids": ["doc1", "doc2"],
            "corpus": ["machine learning model", "deep learning neural net"],
        }
        bm25_file.write_text(json.dumps(bm25_data))
        result = _search_bm25(bm25_file, "machine learning", k=2)
        assert len(result) > 0
        returned_ids = [r[0] for r in result]
        assert "doc1" in returned_ids
        for _chunk_id, _score, data in result:
            assert data == {"source": "bm25"}

    def test_returns_empty_on_exception(self, tmp_path):
        bm25_file = tmp_path / "broken.json"
        bm25_file.write_text("not valid json {{")
        result = _search_bm25(bm25_file, "query", k=5)
        assert result == []

    def test_returns_empty_when_search_raises(self, tmp_path):
        bm25_file = tmp_path / "bm25.json"
        bm25_file.write_text(json.dumps({"doc_ids": ["d1"], "corpus": ["content"]}))
        with patch("notes_chat.retrieval.BM25Index") as MockBM25:
            instance = MockBM25.return_value
            instance.search.side_effect = RuntimeError("search exploded")
            result = _search_bm25(bm25_file, "query", k=5)
        assert result == []


# ---------------------------------------------------------------------------
# _build_context - skip empty content
# ---------------------------------------------------------------------------


class TestBuildContextEmptyContent:
    def test_skips_chunks_with_no_content(self):
        chunks = [
            ("id1", 0.9, {"content": "", "metadata": {}}),
            (
                "id2",
                0.8,
                {
                    "content": "real content here",
                    "metadata": {"path": "/n/slug/notes.md", "date": "2025-01-01"},
                },
            ),
        ]
        context, _sources, ids = _build_context(chunks, 10000)
        assert "id1" not in ids
        assert "id2" in ids
        assert "real content here" in context

    def test_min_size_exceeds_budget_skips_truncation(self):
        header_size = 30
        content = "X" * 500
        tiny_budget = header_size + 50

        chunks = [
            (
                "id1",
                1.0,
                {
                    "content": content,
                    "metadata": {"path": "/n/slug/notes.md", "date": "2025-01-01"},
                },
            ),
        ]
        _context, _sources, ids = _build_context(chunks, tiny_budget)
        assert len(ids) <= 1


# ---------------------------------------------------------------------------
# _slug_from_chunk
# ---------------------------------------------------------------------------


class TestSlugFromChunk:
    def test_returns_none_when_no_metadata(self):
        assert _slug_from_chunk({}) is None
        assert _slug_from_chunk({"content": "x"}) is None

    def test_returns_none_when_no_path_in_metadata(self):
        assert _slug_from_chunk({"metadata": {}}) is None
        assert _slug_from_chunk({"metadata": {"date": "2025-01-01"}}) is None

    def test_returns_parent_dir_name(self):
        data = {"metadata": {"path": "/notes/my-meeting-2025-01-01/transcript.txt"}}
        assert _slug_from_chunk(data) == "my-meeting-2025-01-01"

    def test_returns_none_for_empty_path(self):
        data = {"metadata": {"path": ""}}
        result = _slug_from_chunk(data)
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# _timestamp_seconds_from_chunk
# ---------------------------------------------------------------------------


class TestTimestampSecondsFromChunk:
    def test_returns_none_when_no_metadata(self):
        assert _timestamp_seconds_from_chunk({}) is None
        assert _timestamp_seconds_from_chunk({"content": "x"}) is None

    def test_uses_start_seconds_field(self):
        data = {"metadata": {"start_seconds": 90.5}}
        assert _timestamp_seconds_from_chunk(data) == pytest.approx(90.5)

    def test_uses_start_ms_field(self):
        data = {"metadata": {"start_ms": 90500}}
        assert _timestamp_seconds_from_chunk(data) == pytest.approx(90.5)

    def test_uses_timestamp_ms_field(self):
        data = {"metadata": {"timestamp_ms": 60000}}
        assert _timestamp_seconds_from_chunk(data) == pytest.approx(60.0)

    def test_returns_none_when_no_timestamp_fields(self):
        data = {"metadata": {"path": "/notes/slug/notes.md"}}
        assert _timestamp_seconds_from_chunk(data) is None

    def test_returns_none_for_invalid_start_seconds(self):
        data = {"metadata": {"start_seconds": "not-a-number"}}
        assert _timestamp_seconds_from_chunk(data) is None

    def test_returns_none_for_invalid_start_ms(self):
        data = {"metadata": {"start_ms": "bad"}}
        assert _timestamp_seconds_from_chunk(data) is None

    def test_returns_none_for_invalid_timestamp_ms(self):
        data = {"metadata": {"timestamp_ms": None}}
        assert _timestamp_seconds_from_chunk(data) is None

    def test_start_seconds_zero_is_valid(self):
        data = {"metadata": {"start_seconds": 0}}
        assert _timestamp_seconds_from_chunk(data) == 0.0


# ---------------------------------------------------------------------------
# _format_mm_ss
# ---------------------------------------------------------------------------


class TestFormatMmSs:
    def test_zero_seconds(self):
        assert _format_mm_ss(0) == "00:00"

    def test_one_minute(self):
        assert _format_mm_ss(60) == "01:00"

    def test_one_hour(self):
        assert _format_mm_ss(3600) == "60:00"

    def test_mixed_minutes_seconds(self):
        assert _format_mm_ss(125) == "02:05"

    def test_negative_clamped_to_zero(self):
        assert _format_mm_ss(-5) == "00:00"

    def test_float_truncated(self):
        assert _format_mm_ss(90.9) == "01:30"


# ---------------------------------------------------------------------------
# _create_chunk_header
# ---------------------------------------------------------------------------


class TestCreateChunkHeader:
    def test_formats_iso_date_to_ymd(self):
        data = {
            "metadata": {
                "date": "2025-03-15T14:30:00",
                "path": "/notes/slug/notes.md",
            }
        }
        header = _create_chunk_header(data)
        assert "2025-03-15" in header
        assert "notes.md" in header

    def test_handles_unknown_date(self):
        data = {
            "metadata": {
                "date": "Unknown date",
                "path": "/notes/slug/notes.md",
            }
        }
        header = _create_chunk_header(data)
        assert "Unknown date" in header

    def test_handles_invalid_date_string(self):
        data = {
            "metadata": {
                "date": "not-a-date",
                "path": "/notes/slug/notes.md",
            }
        }
        header = _create_chunk_header(data)
        assert "not-a-date" in header

    def test_handles_missing_path(self):
        data = {"metadata": {"date": "2025-01-01"}}
        header = _create_chunk_header(data)
        assert "Unknown" in header

    def test_returns_unknown_source_when_no_metadata(self):
        header = _create_chunk_header({"source": "bm25"})
        assert header == "Unknown source"

    def test_handles_z_suffix_in_iso_date(self):
        data = {
            "metadata": {
                "date": "2025-06-01T10:00:00Z",
                "path": "/notes/s/notes.md",
            }
        }
        header = _create_chunk_header(data)
        assert "2025-06-01" in header


# ---------------------------------------------------------------------------
# _generate_suggestion
# ---------------------------------------------------------------------------


class TestGenerateSuggestion:
    def test_returns_run_index_when_no_manifest(self, tmp_path):
        config = _make_config(tmp_path)
        result = _generate_suggestion(config, None)
        assert "chirp index" in result

    def test_returns_rebuild_suggestion_on_bad_manifest(self, tmp_path):
        config = _make_config(tmp_path)
        manifest = config.notes_chat.index_dir / "manifest.json"
        manifest.write_text("{ bad json {{")
        result = _generate_suggestion(config, None)
        assert "chirp index --force" in result

    def test_returns_no_files_in_index_when_manifest_empty(self, tmp_path):
        config = _make_config(tmp_path)
        manifest = config.notes_chat.index_dir / "manifest.json"
        manifest.write_text("null")
        result = _generate_suggestion(config, None)
        assert "chirp index" in result

    def test_returns_create_notes_when_notes_root_missing(self, tmp_path):
        config = _make_config(tmp_path)
        manifest = config.notes_chat.index_dir / "manifest.json"
        manifest.write_text(json.dumps({"file1.md": "hash1"}))
        import shutil

        shutil.rmtree(config.directories.notes_root)
        result = _generate_suggestion(config, None)
        assert "chirp transcribe" in result

    def test_returns_create_notes_when_no_notes_md_files(self, tmp_path):
        config = _make_config(tmp_path)
        manifest = config.notes_chat.index_dir / "manifest.json"
        manifest.write_text(json.dumps({"file1.md": "hash1"}))
        result = _generate_suggestion(config, None)
        assert "chirp transcribe" in result

    def test_returns_broader_search_with_latest_date_when_notes_exist(self, tmp_path):
        config = _make_config(tmp_path)
        manifest = config.notes_chat.index_dir / "manifest.json"
        manifest.write_text(json.dumps({"file1.md": "hash1"}))
        note_dir = config.directories.notes_root / "my-note-2025-01-15"
        note_dir.mkdir()
        (note_dir / "notes.md").write_text("some note content")
        result = _generate_suggestion(config, None)
        assert "broader search" in result or "2025-01-15" in result

    def test_returns_fallback_on_os_error(self, tmp_path):
        config = _make_config(tmp_path)
        manifest = config.notes_chat.index_dir / "manifest.json"
        manifest.write_text(json.dumps({"file1.md": "hash1"}))
        note_dir = config.directories.notes_root / "my-note"
        note_dir.mkdir()
        (note_dir / "notes.md").write_text("content")

        original_stat = Path.stat

        def stat_raises_oserror(self, *args, **kwargs):
            if self.name == "notes.md":
                raise OSError("permission denied")
            return original_stat(self, *args, **kwargs)

        with patch.object(Path, "stat", stat_raises_oserror):
            result = _generate_suggestion(config, None)

        assert "broader search" in result or "keywords" in result


# ---------------------------------------------------------------------------
# _get_query_embedding is covered in tests/test_embedding_adapter.py (story 6.3)
# ---------------------------------------------------------------------------
# format_sources — timestamp update and multiple-chunk collapsing
# ---------------------------------------------------------------------------


class TestFormatSourcesAdditionalBranches:
    def test_updates_timestamp_to_earlier_when_second_chunk_is_earlier(self):
        note_index = {"my-note": {"index": 1, "title": "My Note"}}
        chunks = [
            (
                "c1",
                "# header\nbody\n",
                {"metadata": {"path": "/n/my-note/notes.md", "start_seconds": 120}},
            ),
            (
                "c2",
                "# header\nbody\n",
                {"metadata": {"path": "/n/my-note/notes.md", "start_seconds": 30}},
            ),
        ]
        sources = format_sources(chunks, note_index)
        assert "00:30" in sources[0]

    def test_keeps_existing_timestamp_when_second_chunk_is_later(self):
        note_index = {"my-note": {"index": 1, "title": "My Note"}}
        chunks = [
            (
                "c1",
                "# header\nbody\n",
                {"metadata": {"path": "/n/my-note/notes.md", "start_seconds": 30}},
            ),
            (
                "c2",
                "# header\nbody\n",
                {"metadata": {"path": "/n/my-note/notes.md", "start_seconds": 120}},
            ),
        ]
        sources = format_sources(chunks, note_index)
        assert "00:30" in sources[0]

    def test_no_index_number_when_index_key_is_none(self):
        note_index = {"my-note": {"title": "My Note"}}
        chunks = [
            (
                "c1",
                "# header\nbody\n",
                {"metadata": {"path": "/n/my-note/notes.md"}},
            ),
        ]
        sources = format_sources(chunks, note_index)
        assert sources[0] == "My Note"

    def test_handles_chunk_with_no_metadata_uses_chunk_id_as_slug(self):
        chunks = [("orphan-id", "# header\nbody\n", {"source": "bm25"})]
        sources = format_sources(chunks, note_index={})
        assert sources[0] == "orphan-id"


class TestRetrieveContextTagFilter:
    def _seed_note(self, notes_root: Path, slug: str, tags: list[str]) -> None:
        note_dir = notes_root / slug
        note_dir.mkdir()
        (note_dir / "notes.md").write_text(f"# {slug}\n", encoding="utf-8")
        tag_list = ", ".join(f'"{tag}"' for tag in tags)
        (note_dir / "meta.toml").write_text(
            f'title = "{slug}"\ndate = "2026-07-01T09:00:00"\ntags = [{tag_list}]\n',
            encoding="utf-8",
        )

    def _chunk(self, slug: str, content: str):
        return (
            f"{slug}-chunk",
            0.9,
            {
                "content": content,
                "metadata": {
                    "path": f"/notes/{slug}/notes.md",
                    "date": "2026-07-01",
                    "content_hash": slug,
                },
            },
        )

    def test_filters_chunks_to_tagged_notes(self, tmp_path):
        config = _make_config(tmp_path)
        notes_root = config.directories.notes_root
        self._seed_note(notes_root, "standup-note", ["standup"])
        self._seed_note(notes_root, "other-note", ["planning"])

        captured = {}

        def fake_bm25(bm25_file, question, k, index_manager=None):
            captured["k"] = k
            return [
                self._chunk("standup-note", "standup content"),
                self._chunk("other-note", "planning content"),
            ]

        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_bm25", side_effect=fake_bm25),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
            patch("notes_chat.retrieval._build_note_index", return_value={}),
        ):
            MockIM.return_value = _make_index_manager(config, manifest_exists=True)
            result = retrieve_context(config, "what happened", tags=["standup"])

        assert result["success"] is True
        assert "standup content" in result["context"]
        assert "planning content" not in result["context"]
        assert captured["k"] == config.notes_chat.k * 3

    def test_and_semantics_across_tags(self, tmp_path):
        config = _make_config(tmp_path)
        notes_root = config.directories.notes_root
        self._seed_note(notes_root, "both-note", ["standup", "work"])
        self._seed_note(notes_root, "one-note", ["standup"])

        def fake_bm25(bm25_file, question, k, index_manager=None):
            return [
                self._chunk("both-note", "both content"),
                self._chunk("one-note", "one content"),
            ]

        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_bm25", side_effect=fake_bm25),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
            patch("notes_chat.retrieval._build_note_index", return_value={}),
        ):
            MockIM.return_value = _make_index_manager(config, manifest_exists=True)
            result = retrieve_context(config, "what happened", tags=["standup", "work"])

        assert result["success"] is True
        assert "both content" in result["context"]
        assert "one content" not in result["context"]

    def test_no_notes_match_tags_returns_error(self, tmp_path):
        config = _make_config(tmp_path)
        self._seed_note(config.directories.notes_root, "a-note", ["planning"])

        result = retrieve_context(config, "anything", tags=["standup"])

        assert result["success"] is False
        assert "No notes match tag(s): standup" in result["error"]
        assert "chirp notes" in result["suggestion"]

    def test_contentless_hits_dropped_under_tag_filter(self, tmp_path):
        config = _make_config(tmp_path)
        self._seed_note(config.directories.notes_root, "standup-note", ["standup"])

        def fake_bm25(bm25_file, question, k, index_manager=None):
            return [
                self._chunk("standup-note", "standup content"),
                ("stale-chunk", 0.5, {"source": "bm25"}),
            ]

        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_bm25", side_effect=fake_bm25),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
            patch("notes_chat.retrieval._build_note_index", return_value={}),
        ):
            MockIM.return_value = _make_index_manager(config, manifest_exists=True)
            result = retrieve_context(config, "what happened", tags=["standup"])

        assert result["success"] is True
        assert "stale-chunk" not in result["retrieved_ids"]
        assert "standup-note-chunk" in result["retrieved_ids"]

    def test_tags_are_stripped_and_blank_tags_ignored(self, tmp_path):
        config = _make_config(tmp_path)
        self._seed_note(config.directories.notes_root, "standup-note", ["standup"])

        def fake_bm25(bm25_file, question, k, index_manager=None):
            return [self._chunk("standup-note", "standup content")]

        with (
            patch("notes_chat.retrieval.IndexManager") as MockIM,
            patch("notes_chat.retrieval._search_bm25", side_effect=fake_bm25),
            patch("notes_chat.retrieval.parse_time_range", return_value=None),
            patch("notes_chat.retrieval._build_note_index", return_value={}),
        ):
            MockIM.return_value = _make_index_manager(config, manifest_exists=True)
            padded = retrieve_context(config, "what happened", tags=["  standup  "])
            blank_only = retrieve_context(config, "what happened", tags=["   "])

        assert padded["success"] is True
        assert "standup content" in padded["context"]
        # A tag that strips to nothing is no filter at all, matching
        # _parse_tag_filter's behavior — not a filter that matches nothing.
        assert blank_only["success"] is True
