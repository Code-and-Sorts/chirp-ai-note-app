import pytest

from notes_chat.retrieval import _build_context, _merge_and_dedupe


class TestRetrievalMerge:
    def test_semantic_bm25_merge(self):
        """Test merging of semantic and BM25 results."""
        chroma_results = [
            (
                "chunk1",
                0.9,
                {
                    "content": "project planning",
                    "metadata": {
                        "path": "file1.md",
                        "content_hash": "hash_planning",
                    },
                },
            ),
            (
                "chunk2",
                0.8,
                {
                    "content": "team discussion",
                    "metadata": {
                        "path": "file2.md",
                        "content_hash": "hash_discussion",
                    },
                },
            ),
        ]

        bm25_results = [
            ("chunk4", 2.5, {"source": "bm25"}),
            ("chunk3", 1.8, {"source": "bm25"}),
        ]

        merged = _merge_and_dedupe(chroma_results, bm25_results)

        chunk_ids = [chunk_id for chunk_id, _, _ in merged]
        assert len(merged) == 4
        assert "chunk1" in chunk_ids
        assert "chunk2" in chunk_ids
        assert "chunk3" in chunk_ids
        assert "chunk4" in chunk_ids

        scores = [score for _, score, _ in merged]
        assert scores == sorted(scores, reverse=True)

    def test_dedupe_by_path_and_content_hash(self):
        """Test deduplication by path and content hash."""
        chroma_results = [
            (
                "chunk1_001",
                0.9,
                {
                    "content": "content1",
                    "metadata": {
                        "path": "file1.md",
                        "content_hash": "abc123",
                    },
                },
            ),
            (
                "chunk1_002",
                0.85,
                {
                    "content": "content1",
                    "metadata": {
                        "path": "file1.md",
                        "content_hash": "abc123",
                    },
                },
            ),
        ]

        bm25_results = []

        merged = _merge_and_dedupe(chroma_results, bm25_results)

        assert len(merged) == 1
        assert merged[0][0] == "chunk1_001"

    def test_round_robin_budgeting(self):
        """Test round-robin character budget allocation."""
        chunks = [
            (
                "chunk1",
                1.0,
                {
                    "content": "A" * 100,
                    "metadata": {"path": "file1.md", "date": "2025-01-15"},
                },
            ),
            (
                "chunk2",
                0.9,
                {
                    "content": "B" * 200,
                    "metadata": {"path": "file2.md", "date": "2025-01-15"},
                },
            ),
            (
                "chunk3",
                0.8,
                {
                    "content": "C" * 500,
                    "metadata": {"path": "file3.md", "date": "2025-01-15"},
                },
            ),
        ]

        context, sources, retrieved_ids = _build_context(chunks, 400)

        assert len(retrieved_ids) <= 3
        assert "chunk1" in retrieved_ids
        assert len(context) <= 400
        assert len(sources) == len(retrieved_ids)

    def test_context_truncation(self):
        """Test that content is properly truncated when budget is exceeded."""
        chunks = [
            (
                "chunk1",
                1.0,
                {
                    "content": "X" * 1000,
                    "metadata": {"path": "file1.md", "date": "2025-01-15"},
                },
            ),
        ]

        context, _sources, retrieved_ids = _build_context(chunks, 200)

        assert len(context) <= 200
        assert "..." in context
        assert len(retrieved_ids) == 1

    def test_empty_chunks_handling(self):
        """Test handling of empty chunk lists."""
        context, sources, retrieved_ids = _build_context([], 1000)

        assert context == ""
        assert sources == []
        assert retrieved_ids == []

    def test_merge_score_ordering(self):
        """RRF interleaves by rank, so a high-magnitude BM25 hit cannot dominate.

        The old raw-score sort buried the top semantic chunk under any BM25 hit
        whose unbounded score exceeded 1.0. Under RRF the fused score is a sum of
        1/(k+rank) terms, so the two rank-0 chunks (one per source) tie at the
        top and neither source's raw magnitude matters.
        """
        chroma_results = [
            (
                "chroma1",
                0.95,
                {
                    "content": "high score chroma",
                    "metadata": {
                        "path": "file1.md",
                        "content_hash": "hash_high",
                    },
                },
            ),
            (
                "chroma2",
                0.3,
                {
                    "content": "low score chroma",
                    "metadata": {
                        "path": "file2.md",
                        "content_hash": "hash_low",
                    },
                },
            ),
        ]

        bm25_results = [
            ("bm25_1", 1.5, {"source": "bm25"}),
            ("bm25_2", 0.8, {"source": "bm25"}),
        ]

        merged = _merge_and_dedupe(chroma_results, bm25_results)

        scores = [score for _, score, _ in merged]
        assert scores == sorted(scores, reverse=True)

        # Both rank-0 chunks share the top RRF score; the top semantic chunk is
        # NOT buried under the high-magnitude BM25 hit.
        rank_zero = {chunk_id for chunk_id, _, _ in merged[:2]}
        assert rank_zero == {"chroma1", "bm25_1"}
        assert merged[0][1] == merged[1][1]
        # The raw BM25 magnitude (1.5) is replaced by the fused score.
        assert merged[0][1] < 1.0

    def test_top_semantic_survives_high_magnitude_bm25(self):
        """A top-ranked semantic chunk reaches the top of the fused list."""
        chroma_results = [
            (
                "sem_top",
                0.42,
                {
                    "content": "the semantically relevant answer",
                    "metadata": {"path": "a.md", "content_hash": "h_sem"},
                },
            ),
        ]
        # BM25 raw scores dwarf the semantic score; under the old sort they'd
        # all rank above sem_top.
        bm25_results = [
            ("lex1", 12.0, {"source": "bm25"}),
            ("lex2", 9.0, {"source": "bm25"}),
            ("lex3", 7.0, {"source": "bm25"}),
        ]

        merged = _merge_and_dedupe(chroma_results, bm25_results)
        top_ids = {chunk_id for chunk_id, _, _ in merged[:1]}
        # sem_top is rank 0 in its list and ties lex1 (rank 0) at the top.
        assert "sem_top" in {cid for cid, _, _ in merged[:2]}
        assert top_ids <= {"sem_top", "lex1"}

    def test_dedupe_across_sources_keeps_best_rank(self):
        """A chunk found by BOTH retrievers collapses and sums its RRF terms.

        Drives the REAL `_search_bm25` hydration: bm25 raw results carry no
        metadata, so without hydration the same physical chunk would key by bare
        chunk_id (bm25) vs `path::content_hash` (chroma) and NOT dedupe — getting
        1/60 twice instead of a summed 2/60 and wasting a top-k slot. With
        hydration both sources carry `path::content_hash` and collapse.
        """
        import json
        from unittest.mock import MagicMock

        from notes_chat.retrieval import _search_bm25

        chunk_id = "team-sync-2025-01-15_000"
        shared_meta = {"path": "/n/team-sync-2025-01-15/notes.md", "content_hash": "h1"}

        # The chunk lives in Chroma; the same id is also a BM25 hit.
        manager = MagicMock()
        manager.collection.get.return_value = {
            "ids": [chunk_id],
            "documents": ["budget timeline discussion"],
            "metadatas": [shared_meta],
        }

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            bm25_file = Path(tmp) / "bm25.json"
            bm25_file.write_text(
                json.dumps(
                    {"doc_ids": [chunk_id], "corpus": ["budget timeline discussion"]}
                )
            )
            from notes_chat import bm25 as bm25_module

            bm25_module._MODEL_CACHE.pop(str(bm25_file), None)
            bm25_results = _search_bm25(bm25_file, "budget", k=5, index_manager=manager)

        # Hydration gave the bm25 hit the chroma metadata + content.
        assert len(bm25_results) == 1
        assert bm25_results[0][2]["metadata"] == shared_meta
        assert bm25_results[0][2]["content"] == "budget timeline discussion"

        # The same chunk from the chroma side, same (path, content_hash).
        chroma_results = [
            (
                chunk_id,
                0.9,
                {"content": "budget timeline discussion", "metadata": shared_meta},
            ),
        ]

        merged = _merge_and_dedupe(chroma_results, bm25_results)

        assert len(merged) == 1  # collapsed, not duplicated
        _id, score, data = merged[0]
        # BOTH rank-0 contributions are summed: 1/60 + 1/60.
        assert score == pytest.approx(2.0 / 60.0)
        assert data.get("content")  # survives into _build_context

    def test_dedupe_without_hydration_does_not_collapse(self):
        """Without the index_manager (no hydration), the bm25 hit keys by id.

        This is the pre-fix behavior the H1 hydration corrects: a metadata-less
        bm25 result can't dedupe against the chroma hit. Kept as a guard so a
        regression that drops hydration is visible.
        """
        chunk_id = "team-sync-2025-01-15_000"
        shared_meta = {"path": "/n/team-sync-2025-01-15/notes.md", "content_hash": "h1"}
        chroma_results = [
            (chunk_id, 0.9, {"content": "x", "metadata": shared_meta}),
        ]
        bm25_results = [(chunk_id, 5.0, {"source": "bm25"})]  # no metadata

        merged = _merge_and_dedupe(chroma_results, bm25_results)
        # Two distinct signatures (path::hash vs bare id) => not collapsed.
        assert len(merged) == 2

    def test_bm25_only_hit_surfaces_content_after_hydration(self):
        """M3: a lexical-only hit (not in chroma results) reaches the context.

        Before hydration a BM25-only hit had no `content`, so `_build_context`
        dropped it — the lexical half of "hybrid" could never surface unique
        content. After hydration the hit carries its document and survives.
        """
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock

        from notes_chat.retrieval import _search_bm25

        chunk_id = "jira-1234-note-2025-02-01_000"
        meta = {
            "path": "/n/jira-1234-note-2025-02-01/notes.md",
            "content_hash": "hx",
            "date": "2025-02-01",
        }
        manager = MagicMock()
        manager.collection.get.return_value = {
            "ids": [chunk_id],
            "documents": ["JIRA-1234 must ship before the release cutoff."],
            "metadatas": [meta],
        }

        with tempfile.TemporaryDirectory() as tmp:
            bm25_file = Path(tmp) / "bm25.json"
            bm25_file.write_text(
                json.dumps(
                    {
                        "doc_ids": [chunk_id],
                        "corpus": ["jira 1234 must ship before the release cutoff"],
                    }
                )
            )
            from notes_chat import bm25 as bm25_module

            bm25_module._MODEL_CACHE.pop(str(bm25_file), None)
            bm25_results = _search_bm25(
                bm25_file, "jira 1234", k=5, index_manager=manager
            )

        # Only the lexical source has this chunk; it must still enter the context.
        context, _sources, ids = _build_context(bm25_results, char_budget=10000)
        assert chunk_id in ids
        assert "JIRA-1234" in context

    def test_chunk_header_creation(self):
        """Headers carry date + filename; sources fall back to slug when no config."""
        chunks = [
            (
                "chunk1",
                1.0,
                {
                    "content": "content",
                    "metadata": {
                        "path": "/path/to/team-meeting-2025-01-15/notes.md",
                        "date": "2025-01-15T10:30:00",
                        "title": "Team Meeting",
                    },
                },
            ),
        ]

        context, sources, _retrieved_ids = _build_context(chunks, 1000)

        assert "2025-01-15 · notes.md" in context
        # No `config` was provided so the slug is the fallback label.
        assert sources[0] == "team-meeting-2025-01-15"
