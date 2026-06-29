import json

from notes_chat.bm25 import (
    _MODEL_CACHE,
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

    def test_rebuild_writes_self_sufficient_store(self, tmp_path):
        """Rebuild writes corpus + raw documents + metadata (no Chroma needed)."""
        doc_ids = ["chunk1", "chunk2"]
        documents = [
            "Meeting about project planning and timelines",
            "Technical discussion on architecture decisions",
        ]
        metadatas = [
            {"path": "/n/a/notes.md", "content_hash": "h1"},
            {"path": "/n/b/notes.md", "content_hash": "h2"},
        ]

        bm25_file = tmp_path / "rebuilt_bm25.json"
        rebuild_bm25_index(doc_ids, documents, metadatas, bm25_file)

        assert bm25_file.exists()

        with bm25_file.open() as f:
            data = json.load(f)

        assert data["doc_ids"] == ["chunk1", "chunk2"]
        assert len(data["corpus"]) == 2
        assert "project" in data["corpus"][0]
        assert "technical" in data["corpus"][1]
        assert data["documents"] == documents
        assert data["metadatas"] == metadatas

    def test_rebuild_skips_write_when_empty(self, tmp_path):
        """No doc ids => no file written (nothing to index)."""
        bm25_file = tmp_path / "empty_bm25.json"
        rebuild_bm25_index([], [], [], bm25_file)
        assert not bm25_file.exists()

    def test_index_exposes_documents_and_metadata(self, tmp_path):
        """BM25Index round-trips documents/metadatas for in-store hydration."""
        bm25_file = tmp_path / "bm25.json"
        doc_ids = ["doc1", "doc2"]
        documents = ["alpha budget planning", "beta roadmap review"]
        metadatas = [
            {"path": "/n/alpha/notes.md", "content_hash": "ha"},
            {"path": "/n/beta/notes.md", "content_hash": "hb"},
        ]
        rebuild_bm25_index(doc_ids, documents, metadatas, bm25_file)

        index = BM25Index(bm25_file)
        assert index.documents == documents
        assert index.metadatas == metadatas
        assert index.hydrate("doc1") == (documents[0], metadatas[0])
        assert index.hydrate("missing") is None

    def test_old_schema_hydrate_returns_none(self, tmp_path):
        """An old 2-key bm25.json still searches but hydrates nothing (rebuild cue).

        Existing installs carry ``{doc_ids, corpus}`` only; until the next full
        rebuild repopulates documents/metadatas, hydration must degrade to None
        rather than crash so a lexical hit is still surfaced (without content).
        """
        bm25_file = tmp_path / "bm25.json"
        bm25_file.write_text(
            json.dumps({"doc_ids": ["doc1"], "corpus": ["alpha budget planning"]})
        )
        _MODEL_CACHE.pop(str(bm25_file), None)

        index = BM25Index(bm25_file)
        assert index.search("budget", k=5)
        assert index.documents == []
        assert index.hydrate("doc1") is None


class TestBM25Append:
    def _meta(self, *slugs):
        return [{"path": f"/n/{s}/notes.md", "content_hash": s} for s in slugs]

    def test_append_adds_chunks(self, tmp_path):
        bm25_file = tmp_path / "bm25.json"
        append_bm25_index(
            bm25_file,
            ["alpha_000", "alpha_001"],
            ["first chunk text", "second chunk text"],
            self._meta("a0", "a1"),
            stale_id_prefix="alpha_",
        )
        with bm25_file.open() as f:
            data = json.load(f)
        assert set(data["doc_ids"]) == {"alpha_000", "alpha_001"}
        assert data["documents"] == ["first chunk text", "second chunk text"]
        assert data["metadatas"] == self._meta("a0", "a1")

    def test_append_preserves_other_notes(self, tmp_path):
        bm25_file = tmp_path / "bm25.json"
        rebuild_bm25_index(
            ["beta_000"], ["beta content here"], self._meta("b0"), bm25_file
        )
        append_bm25_index(
            bm25_file,
            ["alpha_000"],
            ["alpha content"],
            self._meta("a0"),
            stale_id_prefix="alpha_",
        )
        with bm25_file.open() as f:
            data = json.load(f)
        assert set(data["doc_ids"]) == {"beta_000", "alpha_000"}
        index = BM25Index(bm25_file)
        assert index.hydrate("beta_000") == ("beta content here", self._meta("b0")[0])

    def test_append_preserves_old_schema_entries(self, tmp_path):
        """Appending to a 2-key file keeps the existing ids (padded content)."""
        bm25_file = tmp_path / "bm25.json"
        bm25_file.write_text(
            json.dumps({"doc_ids": ["beta_000"], "corpus": ["beta content here"]})
        )
        append_bm25_index(
            bm25_file,
            ["alpha_000"],
            ["alpha content"],
            self._meta("a0"),
            stale_id_prefix="alpha_",
        )
        with bm25_file.open() as f:
            data = json.load(f)
        assert set(data["doc_ids"]) == {"beta_000", "alpha_000"}
        assert len(data["documents"]) == 2
        assert len(data["metadatas"]) == 2

    def test_append_purges_ghost_ids_on_shrink(self, tmp_path):
        """L3: a note shrinking from 3 chunks to 2 must not leave a ghost id."""
        bm25_file = tmp_path / "bm25.json"
        append_bm25_index(
            bm25_file,
            ["alpha_000", "alpha_001", "alpha_002"],
            ["chunk a", "chunk b", "chunk c"],
            self._meta("a0", "a1", "a2"),
            stale_id_prefix="alpha_",
        )
        append_bm25_index(
            bm25_file,
            ["alpha_000", "alpha_001"],
            ["chunk a edited", "chunk b edited"],
            self._meta("a0", "a1"),
            stale_id_prefix="alpha_",
        )
        with bm25_file.open() as f:
            data = json.load(f)
        assert set(data["doc_ids"]) == {"alpha_000", "alpha_001"}
        assert "alpha_002" not in data["doc_ids"]

    def test_append_without_prefix_keeps_ghosts(self, tmp_path):
        """Without a stale prefix, old behavior holds (ghost left until rebuild)."""
        bm25_file = tmp_path / "bm25.json"
        append_bm25_index(
            bm25_file,
            ["alpha_000", "alpha_001", "alpha_002"],
            ["chunk a", "chunk b", "chunk c"],
            self._meta("a0", "a1", "a2"),
        )
        append_bm25_index(
            bm25_file, ["alpha_000"], ["chunk a edited"], self._meta("a0")
        )
        with bm25_file.open() as f:
            data = json.load(f)
        # Ghosts linger by design without a prefix; a full rebuild purges them.
        assert "alpha_002" in data["doc_ids"]
