import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, bm25_file: Path):
        self.bm25_file = bm25_file
        self.bm25 = None
        self.doc_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._vocabulary: dict[str, int] | None = None
        self.load()

    def load(self):
        """Load BM25 index from file."""
        if not self.bm25_file.exists():
            return

        try:
            with open(self.bm25_file) as f:
                data = json.load(f)

            self.doc_ids = data.get("doc_ids", [])

            if data.get("corpus"):
                corpus = [doc.split() for doc in data["corpus"]]
                self._tokenized_corpus = corpus
                self.bm25 = BM25Okapi(corpus)
        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            AttributeError,
            TypeError,
        ):
            # Cache miss / corruption / schema drift — caller will rebuild the index.
            # AttributeError and TypeError cover schema-drifted JSON (e.g., [] instead
            # of dict, or null values in corpus).
            pass

    def vocabulary(self) -> dict[str, int]:
        """Return token → distinct-document count for the loaded corpus."""
        if self._vocabulary is not None:
            return self._vocabulary

        counts: dict[str, int] = {}
        for tokens in self._tokenized_corpus:
            for token in set(tokens):
                counts[token] = counts.get(token, 0) + 1
        self._vocabulary = counts
        return counts

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search using BM25."""
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            if i < len(self.doc_ids):
                results.append((self.doc_ids[i], float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [token for token in text.split() if len(token) > 1]


def rebuild_bm25_index(chroma_collection, bm25_file: Path):
    """Rebuild BM25 index from Chroma collection."""
    try:
        results = chroma_collection.get()

        if not results["ids"]:
            return

        doc_ids = results["ids"]
        documents = results["documents"]

        tokenized_corpus = []
        for doc in documents:
            tokens = _tokenize_document(doc)
            tokenized_corpus.append(tokens)

        bm25_data = {
            "doc_ids": doc_ids,
            "corpus": [" ".join(tokens) for tokens in tokenized_corpus],
        }

        bm25_file.parent.mkdir(parents=True, exist_ok=True)
        with open(bm25_file, "w") as f:
            json.dump(bm25_data, f, indent=2)

    except Exception as e:  # noqa: BLE001 - chromadb or IO; many failure modes
        logger.warning("Failed to rebuild BM25 index: %s", e)


def _tokenize_document(text: str) -> list[str]:
    """Tokenize document for BM25 indexing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [token for token in text.split() if len(token) > 1]
