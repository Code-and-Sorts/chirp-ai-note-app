import json
from unittest.mock import Mock

from notes_chat.bm25 import (
    BM25Index,
    _tokenize_document,
    append_bm25_index,
    rebuild_bm25_index,
)


class TestBM25:
    def test_tokenization_consistency(self):
        """Test that tokenization is consistent and stable."""
        text1 = "This is a test document with some words!"
        text2 = "This is a test document with some words!"

        tokens1 = _tokenize_document(text1)
        tokens2 = _tokenize_document(text2)

        assert tokens1 == tokens2
        assert len(tokens1) > 0
        assert all(len(token) > 1 for token in tokens1)
        assert all(token.islower() for token in tokens1)

    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly."""
        text = "Hello, world! This is a test... isn't it?"
        tokens = _tokenize_document(text)

        expected = ["hello", "world", "this", "is", "test", "isn", "it"]
        assert tokens == expected

    def test_bm25_search_ranking(self, tmp_path):
        """Test that BM25 returns relevant results in ranked order."""
        bm25_file = tmp_path / "test_bm25.json"

        bm25_data = {
            "doc_ids": ["doc1", "doc2", "doc3"],
            "corpus": [
                "machine learning algorithms neural networks",
                "deep learning artificial intelligence",
                "natural language processing text analysis",
            ],
        }

        with bm25_file.open("w") as f:
            json.dump(bm25_data, f)

        index = BM25Index(bm25_file)
        results = index.search("machine learning", k=3)

        assert len(results) > 0
        assert results[0][0] == "doc1"

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_tie_break_stability(self, tmp_path):
        """Test that tie-breaking is stable and consistent."""
        bm25_file = tmp_path / "test_bm25.json"

        bm25_data = {
            "doc_ids": ["doc1", "doc2", "doc3"],
            "corpus": [
                "the quick brown fox",
                "the quick brown fox",
                "the quick brown fox",
            ],
        }

        with bm25_file.open("w") as f:
            json.dump(bm25_data, f)

        index = BM25Index(bm25_file)
        results1 = index.search("quick fox", k=3)
        results2 = index.search("quick fox", k=3)

        assert results1 == results2
        assert len(results1) == 3

    def test_top_k_mapping(self, tmp_path):
        """Test that top-k results map back to correct chunk IDs."""
        bm25_file = tmp_path / "test_bm25.json"

        bm25_data = {
            "doc_ids": ["chunk_001", "chunk_002", "chunk_003"],
            "corpus": [
                "project management meeting agenda",
                "technical review code quality",
                "unrelated document about something else",
            ],
        }

        with bm25_file.open("w") as f:
            json.dump(bm25_data, f)

        index = BM25Index(bm25_file)
        results = index.search("project management", k=3)

        assert len(results) == 3
        assert results[0][0] == "chunk_001"

        all_chunk_ids = [doc_id for doc_id, _ in results]
        assert "chunk_001" in all_chunk_ids
        assert "chunk_002" in all_chunk_ids
        assert "chunk_003" in all_chunk_ids

    def test_empty_query_handling(self, tmp_path):
        """Test handling of empty or whitespace queries."""
        bm25_file = tmp_path / "test_bm25.json"

        bm25_data = {"doc_ids": ["doc1"], "corpus": ["some content here"]}

        with bm25_file.open("w") as f:
            json.dump(bm25_data, f)

        index = BM25Index(bm25_file)

        assert index.search("", k=5) == []
        assert index.search("   ", k=5) == []
        assert index.search("a", k=5) == []

    def test_missing_file_handling(self, tmp_path):
        """Test handling when BM25 file doesn't exist."""
        bm25_file = tmp_path / "nonexistent.json"

        index = BM25Index(bm25_file)
        results = index.search("test query", k=5)

        assert results == []
        assert index.bm25 is None

    def test_rebuild_from_chroma(self, tmp_path):
        """Test rebuilding BM25 index from Chroma collection."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["chunk1", "chunk2"],
            "documents": [
                "Meeting about project planning and timelines",
                "Technical discussion on architecture decisions",
            ],
        }

        bm25_file = tmp_path / "rebuilt_bm25.json"
        rebuild_bm25_index(mock_collection, bm25_file)

        assert bm25_file.exists()

        with bm25_file.open() as f:
            data = json.load(f)

        assert data["doc_ids"] == ["chunk1", "chunk2"]
        assert len(data["corpus"]) == 2
        assert "project" in data["corpus"][0]
        assert "technical" in data["corpus"][1]


class TestBM25Append:
    def test_append_adds_chunks(self, tmp_path):
        bm25_file = tmp_path / "bm25.json"
        append_bm25_index(
            bm25_file,
            ["alpha_000", "alpha_001"],
            ["first chunk text", "second chunk text"],
            stale_id_prefix="alpha_",
        )
        with bm25_file.open() as f:
            data = json.load(f)
        assert set(data["doc_ids"]) == {"alpha_000", "alpha_001"}

    def test_append_preserves_other_notes(self, tmp_path):
        bm25_file = tmp_path / "bm25.json"
        bm25_file.write_text(
            json.dumps({"doc_ids": ["beta_000"], "corpus": ["beta content here"]})
        )
        append_bm25_index(
            bm25_file, ["alpha_000"], ["alpha content"], stale_id_prefix="alpha_"
        )
        with bm25_file.open() as f:
            data = json.load(f)
        # Other notes are untouched; this note is added.
        assert set(data["doc_ids"]) == {"beta_000", "alpha_000"}

    def test_append_purges_ghost_ids_on_shrink(self, tmp_path):
        """L3: a note shrinking from 3 chunks to 2 must not leave a ghost id."""
        bm25_file = tmp_path / "bm25.json"
        # First save: 3 chunks for note "alpha".
        append_bm25_index(
            bm25_file,
            ["alpha_000", "alpha_001", "alpha_002"],
            ["chunk a", "chunk b", "chunk c"],
            stale_id_prefix="alpha_",
        )
        # Re-save after an edit that produces only 2 chunks.
        append_bm25_index(
            bm25_file,
            ["alpha_000", "alpha_001"],
            ["chunk a edited", "chunk b edited"],
            stale_id_prefix="alpha_",
        )
        with bm25_file.open() as f:
            data = json.load(f)
        # alpha_002 (the vanished chunk) is purged — no ghost.
        assert set(data["doc_ids"]) == {"alpha_000", "alpha_001"}
        assert "alpha_002" not in data["doc_ids"]

    def test_append_without_prefix_keeps_ghosts(self, tmp_path):
        """Without a stale prefix, old behavior holds (ghost left until rebuild)."""
        bm25_file = tmp_path / "bm25.json"
        append_bm25_index(
            bm25_file,
            ["alpha_000", "alpha_001", "alpha_002"],
            ["chunk a", "chunk b", "chunk c"],
        )
        append_bm25_index(bm25_file, ["alpha_000"], ["chunk a edited"])
        with bm25_file.open() as f:
            data = json.load(f)
        # alpha_001/alpha_002 linger (self-heals on full rebuild).
        assert "alpha_002" in data["doc_ids"]
