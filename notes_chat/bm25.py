import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, bm25_file: Path):
        self.bm25_file = bm25_file
        self.bm25 = None
        self.doc_ids: list[str] = []
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
                self.bm25 = BM25Okapi(corpus)
        except Exception:
            pass

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

    except Exception as e:
        print(f"Failed to rebuild BM25 index: {e}")


def _tokenize_document(text: str) -> list[str]:
    """Tokenize document for BM25 indexing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [token for token in text.split() if len(token) > 1]
