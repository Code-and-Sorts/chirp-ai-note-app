import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Process-local BM25 model cache keyed by bm25.json path, invalidated on
# (mtime, size). `chirp ask` builds a BM25Index per query; without this each
# query re-tokenizes and rebuilds BM25Okapi — O(corpus) on a ~5k-note corpus.
_MODEL_CACHE: dict[
    str, tuple[tuple[float, int], "BM25Okapi", list[str], list[list[str]]]
] = {}


def _file_fingerprint(bm25_file: Path) -> tuple[float, int] | None:
    try:
        stat = bm25_file.stat()
    except OSError:
        return None
    return (stat.st_mtime, stat.st_size)


class BM25Index:
    def __init__(self, bm25_file: Path):
        self.bm25_file = bm25_file
        self.bm25 = None
        self.doc_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._vocabulary: dict[str, int] | None = None
        self.load()

    def load(self):
        """Load the BM25 model, reusing the cached one when the file is unchanged."""
        if not self.bm25_file.exists():
            return

        fingerprint = _file_fingerprint(self.bm25_file)
        cache_key = str(self.bm25_file)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None and fingerprint is not None and cached[0] == fingerprint:
            _, self.bm25, self.doc_ids, self._tokenized_corpus = cached
            return

        try:
            with self.bm25_file.open() as f:
                data = json.load(f)

            self.doc_ids = data.get("doc_ids", [])

            if data.get("corpus"):
                corpus = [doc.split() for doc in data["corpus"]]
                self._tokenized_corpus = corpus
                self.bm25 = BM25Okapi(corpus)
                if fingerprint is not None:
                    _MODEL_CACHE[cache_key] = (
                        fingerprint,
                        self.bm25,
                        self.doc_ids,
                        corpus,
                    )
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
        with bm25_file.open("w") as f:
            json.dump(bm25_data, f, indent=2)

        # Drop the cached model: the rewrite may land within the filesystem's
        # mtime resolution, so the fingerprint alone wouldn't invalidate it.
        _MODEL_CACHE.pop(str(bm25_file), None)

    except Exception as e:  # noqa: BLE001 - chromadb or IO; many failure modes
        logger.warning("Failed to rebuild BM25 index: %s", e)


def append_bm25_index(
    bm25_file: Path,
    doc_ids: list[str],
    documents: list[str],
    stale_id_prefix: str | None = None,
) -> None:
    """Merge a single note's chunks into ``bm25.json`` without a full re-tokenize.

    Auto-indexing one saved note used to trigger a whole-corpus rebuild
    (``rebuild_bm25_index`` pulls the entire Chroma collection and re-tokenizes
    it). A burst of N saves then cost N full rebuilds. Appending just the
    changed chunks — replacing any existing entries for the same ids so a
    re-save updates in place — keeps a save O(note) instead of O(corpus).

    ``stale_id_prefix`` (the note's ``{slug}_`` id prefix) drops any existing
    entry that belongs to this note but is no longer present — i.e. ghost ids
    left when a note shrinks from N chunks to fewer. Without it, vanished chunks
    would linger in the lexicon until the next full rebuild.
    """
    existing_ids: list[str] = []
    existing_corpus: list[str] = []
    if bm25_file.exists():
        try:
            with bm25_file.open() as f:
                data = json.load(f)
            existing_ids = data.get("doc_ids", []) or []
            existing_corpus = data.get("corpus", []) or []
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            existing_ids = []
            existing_corpus = []

    incoming = {
        doc_id: " ".join(_tokenize_document(doc))
        for doc_id, doc in zip(doc_ids, documents, strict=False)
    }

    merged_ids: list[str] = []
    merged_corpus: list[str] = []
    for doc_id, corpus_entry in zip(existing_ids, existing_corpus, strict=False):
        if doc_id in incoming:
            continue
        # Drop ghost ids: a chunk that vanished when this note was re-chunked.
        if stale_id_prefix is not None and doc_id.startswith(stale_id_prefix):
            continue
        merged_ids.append(doc_id)
        merged_corpus.append(corpus_entry)
    for doc_id, corpus_entry in incoming.items():
        merged_ids.append(doc_id)
        merged_corpus.append(corpus_entry)

    bm25_file.parent.mkdir(parents=True, exist_ok=True)
    with bm25_file.open("w") as f:
        json.dump({"doc_ids": merged_ids, "corpus": merged_corpus}, f, indent=2)

    _MODEL_CACHE.pop(str(bm25_file), None)


def _tokenize_document(text: str) -> list[str]:
    """Tokenize document for BM25 indexing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [token for token in text.split() if len(token) > 1]
