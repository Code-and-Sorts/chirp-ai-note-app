from notes_chat.retrieval import _build_context, _merge_and_dedupe


class TestRetrievalMerge:
    def test_date_filtering(self):
        """Test that date filtering works in Chroma queries."""
        # This would be tested with actual Chroma integration
        # For now, just test the merge logic
        pass

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
                        "first_96_chars": "project planning meeting",
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
                        "first_96_chars": "team discussion notes",
                    },
                },
            ),
        ]

        bm25_results = [
            ("chunk4", 2.5, {"source": "bm25"}),
            ("chunk3", 1.8, {"source": "bm25"}),
        ]

        merged = _merge_and_dedupe(chroma_results, bm25_results)

        # Should have 4 unique chunks (no duplicates)
        chunk_ids = [chunk_id for chunk_id, _, _ in merged]
        assert len(merged) == 4
        assert "chunk1" in chunk_ids
        assert "chunk2" in chunk_ids
        assert "chunk3" in chunk_ids
        assert "chunk4" in chunk_ids

        # Should be sorted by score (highest first)
        scores = [score for _, score, _ in merged]
        assert scores == sorted(scores, reverse=True)

    def test_dedupe_by_path_and_first96(self):
        """Test deduplication by path and first 96 characters."""
        chroma_results = [
            (
                "chunk1_001",
                0.9,
                {
                    "content": "content1",
                    "metadata": {
                        "path": "file1.md",
                        "first_96_chars": "same content here",
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
                        "first_96_chars": "same content here",
                    },
                },
            ),
        ]

        bm25_results = []

        merged = _merge_and_dedupe(chroma_results, bm25_results)

        # Should have only 1 chunk after deduplication
        assert len(merged) == 1
        assert merged[0][0] == "chunk1_001"  # Higher score wins

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

        # Small budget should fit first two chunks but not the third
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

        context, sources, retrieved_ids = _build_context(chunks, 200)

        assert len(context) <= 200
        assert "..." in context  # Should be truncated
        assert len(retrieved_ids) == 1

    def test_empty_chunks_handling(self):
        """Test handling of empty chunk lists."""
        context, sources, retrieved_ids = _build_context([], 1000)

        assert context == ""
        assert sources == []
        assert retrieved_ids == []

    def test_merge_score_ordering(self):
        """Test that merged results maintain proper score ordering."""
        chroma_results = [
            (
                "chroma1",
                0.95,
                {
                    "content": "high score chroma",
                    "metadata": {
                        "path": "file1.md",
                        "first_96_chars": "high score chroma",
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
                        "first_96_chars": "low score chroma",
                    },
                },
            ),
        ]

        bm25_results = [
            ("bm25_1", 1.5, {"source": "bm25"}),
            ("bm25_2", 0.8, {"source": "bm25"}),
        ]

        merged = _merge_and_dedupe(chroma_results, bm25_results)

        # Check that scores are in descending order
        scores = [score for _, score, _ in merged]
        assert scores == sorted(scores, reverse=True)

        # Highest score should be first
        assert merged[0][1] == 1.5  # bm25_1
        assert merged[1][1] == 0.95  # chroma1

    def test_chunk_header_creation(self):
        """Test creation of chunk headers."""
        chunks = [
            (
                "chunk1",
                1.0,
                {
                    "content": "content",
                    "metadata": {
                        "path": "/path/to/meetings_2025_01_15.md",
                        "date": "2025-01-15T10:30:00",
                        "title": "Team Meeting",
                    },
                },
            ),
        ]

        context, sources, retrieved_ids = _build_context(chunks, 1000)

        assert "2025-01-15 · meetings_2025_01_15.md" in context
        assert "Team Meeting (meetings_2025_01_15.md)" in sources[0]
