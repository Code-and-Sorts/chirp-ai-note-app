import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import ChirpSettings
from llm.client import LLMClient
from llm.exceptions import LLMError
from notes_chat.bm25 import BM25Index
from notes_chat.index import IndexManager
from notes_chat.time_ranges import parse_time_range

logger = logging.getLogger(__name__)

# RRF smoothing constant. 60 is the canonical value from the original RRF paper
# (and .docs/hybrid-retrieval.md); it damps low-rank items without starving
# either source.
RRF_K = 60

LEXICAL_ONLY = (1.0, 0.0)
LITERAL_WEIGHTS = (1.0, 0.3)
CONCEPTUAL_WEIGHTS = (1.0, 1.0)

# Retrieval returns a fixed top-k, so a query that truly matches one note still
# comes back padded with weak/zero-score filler. RRF fuses by rank (not score),
# discarding the raw-score gap, so that filler would otherwise ride into the
# context and the `sources` line. Drop hits that scored <= 0 (matched nothing)
# or below this fraction of the top hit's score, applied to raw scores BEFORE
# the merge. The top hit is always kept (top >= top * ratio).
RELEVANCE_FLOOR_RATIO = 0.2

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")


def _looks_like_id(token: str) -> bool:
    """True for mixed digit+non-digit tokens (jira-123, v2.3.1, 2026-06-28).

    Excludes bare numbers (room 101) so ordinary counts don't route as literal.
    """
    return any(ch.isdigit() for ch in token) and not token.isdigit()


def _as_naive(value: datetime) -> datetime:
    """Drop tzinfo (converting to local wall-clock) so date strings sort lexically.

    Stored chunk dates are naive local (see ``index.py`` / ``_resolve_created_at``).
    Chroma compares the ``date`` metadata filter as a plain string, so the
    where-clause bounds must have the same naive shape or the comparison is wrong.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


# Process-local freshness fingerprints keyed by index_dir, skipping a wasted
# per-query build_index on an unchanged corpus. Folds each <slug>/notes.md's
# (mtime, size) so an IN-PLACE edit (which leaves notes_root's dir mtime
# untouched) invalidates it too; costs one stat() per note.
_FRESHNESS_CACHE: dict[str, int] = {}


def _notes_tree_signature(config: ChirpSettings) -> int | None:
    notes_root = config.directories.notes_root
    if not notes_root.exists():
        return None
    try:
        components: list[tuple[str, float, int]] = []
        for note_file in sorted(notes_root.glob("*/notes.md")):
            stat = note_file.stat()
            components.append((str(note_file), stat.st_mtime, stat.st_size))
    except OSError:
        return None
    return hash(tuple(components))


def _index_is_fresh(config: ChirpSettings) -> bool:
    signature = _notes_tree_signature(config)
    if signature is None:
        return False
    return _FRESHNESS_CACHE.get(str(config.notes_chat.index_dir)) == signature


def _record_index_freshness(config: ChirpSettings) -> None:
    signature = _notes_tree_signature(config)
    if signature is not None:
        _FRESHNESS_CACHE[str(config.notes_chat.index_dir)] = signature


def retrieve_context(
    config: ChirpSettings, question: str, when_filter: str | None = None
) -> dict[str, Any]:
    """Retrieve context for answering a question using hybrid search."""
    try:
        index_manager = IndexManager(config)

        if not index_manager.manifest_file.exists():
            suggestion = _generate_suggestion(config, None)
            return {
                "success": False,
                "error": "No search index found",
                "suggestion": suggestion,
            }

        if not _index_is_fresh(config):
            index_result = index_manager.build_index(force=False)
            if not index_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to update index: {index_result.get('error')}",
                }
            _record_index_freshness(config)

        time_range = None
        if when_filter:
            time_range = parse_time_range(question, when_filter)
        else:
            time_range = parse_time_range(question)

        bm25_results = _search_bm25(
            index_manager.bm25_file,
            question,
            config.notes_chat.k,
            index_manager=index_manager,
        )

        chroma_results: list[tuple[str, float, dict[str, Any]]] = []
        weights = LEXICAL_ONLY
        if config.notes_chat.semantic_enabled:
            weights = _route_query(question)
            chroma_results = _search_chroma(
                index_manager, question, config.notes_chat.k, time_range
            )

        if not chroma_results and not bm25_results:
            suggestion = _generate_suggestion(config, time_range)
            return {
                "success": False,
                "error": "No documents found matching your query",
                "suggestion": suggestion,
            }

        merged_chunks = _merge_and_dedupe(chroma_results, bm25_results, weights)

        context, sources, retrieved_ids = _build_context(
            merged_chunks, config.notes_chat.ctx_char_budget, config
        )

        if not context.strip():
            suggestion = _generate_suggestion(config, time_range)
            return {
                "success": False,
                "error": "No relevant content found after filtering",
                "suggestion": suggestion,
            }

        return {
            "success": True,
            "context": context,
            "sources": sources,
            "retrieved_ids": retrieved_ids,
            "chunks_retrieved": len(merged_chunks),
        }

    except Exception as e:  # noqa: BLE001 - retrieve_context orchestrates many subsystems
        logger.debug("retrieve_context failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


def _search_chroma(
    index_manager: IndexManager, query: str, k: int, time_range: Any | None = None
) -> list[tuple[str, float, dict[str, Any]]]:
    """Search Chroma vector database."""
    try:
        where_clause = None
        if time_range:
            start_iso = _as_naive(time_range.start).isoformat()
            end_iso = _as_naive(time_range.end_exclusive).isoformat()
            where_clause = {
                "$and": [{"date": {"$gte": start_iso}}, {"date": {"$lt": end_iso}}]
            }

        query_embedding = _get_query_embedding(index_manager.config, query)
        if not query_embedding:
            return []

        results = index_manager.collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=k,
            where=where_clause,  # type: ignore[arg-type]
        )

        ids_list = results.get("ids") or [[]]
        distances_list = results.get("distances") or [[]]
        metadatas_list = results.get("metadatas") or [[]]
        documents_list = results.get("documents") or [[]]

        ids = ids_list[0] if ids_list else []
        distances = distances_list[0] if distances_list else []
        metadatas = metadatas_list[0] if metadatas_list else []
        documents = documents_list[0] if documents_list else []

        if not ids:
            return []

        chroma_results = []
        for i, chunk_id in enumerate(ids):
            if i < len(distances) and i < len(metadatas) and i < len(documents):
                score = 1.0 - distances[i]
                metadata = metadatas[i]
                document = documents[i]

                chroma_results.append(
                    (chunk_id, score, {"content": document, "metadata": metadata})
                )

        return _apply_relevance_floor(chroma_results)

    except Exception as e:  # noqa: BLE001 - chromadb raises various internal exceptions
        logger.debug("Chroma search failed: %s", e, exc_info=True)
        return []


def _search_bm25(
    bm25_file: Path,
    query: str,
    k: int,
    index_manager: IndexManager | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Search BM25, hydrating each hit with its content + metadata from the store.

    The lexical store (``bm25.json``) carries each chunk's content and metadata,
    so hits hydrate from it directly — no Chroma round-trip, which is what lets
    lexical-only retrieval answer with the vector half disabled. A bare
    ``{"source": "bm25"}`` payload carries no ``metadata`` and no ``content``,
    which breaks two things: (1) ``_merge_and_dedupe`` would key the hit by bare
    chunk_id instead of ``path::content_hash``, so a chunk found by BOTH
    retrievers fails to dedupe and its RRF contributions are never summed;
    (2) ``_build_context`` drops contentless chunks, so the lexical half could
    never surface a BM25-only hit. Hydrating from the store gives both sources
    the same shape. ``index_manager`` is accepted for call-site compatibility but
    no longer consulted — the BM25 store is self-sufficient.
    """
    try:
        bm25_index = BM25Index(bm25_file)
        bm25_results = bm25_index.search(query, k)
        if not bm25_results:
            return []

        results: list[tuple[str, float, dict[str, Any]]] = []
        for chunk_id, score in bm25_results:
            hydrated = bm25_index.hydrate(chunk_id)
            if hydrated is None:
                # Keep a contentless old-schema/stale hit so the match isn't
                # dropped, though without content it can't enter context.
                data: dict[str, Any] = {"source": "bm25"}
            else:
                content, metadata = hydrated
                data = {"content": content, "metadata": metadata, "source": "bm25"}
            results.append((chunk_id, score, data))
        return _apply_relevance_floor(results)

    except Exception as e:  # noqa: BLE001 - BM25 can raise many types on corrupt index
        logger.debug("BM25 search failed: %s", e, exc_info=True)
        return []


def _signature(chunk_id: str, data: dict[str, Any]) -> str:
    """Dedupe key: ``(path, content_hash)`` when present, else the chunk id."""
    metadata = data.get("metadata")
    if metadata:
        path = metadata.get("path", "")
        content_hash = metadata.get("content_hash", "")
        return f"{path}::{content_hash}"
    return chunk_id


def _apply_relevance_floor(
    results: list[tuple[str, float, dict[str, Any]]],
    ratio: float = RELEVANCE_FLOOR_RATIO,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Drop weak/zero-score filler from a single retriever's ranked hits.

    Keeps only hits scoring above 0 AND at least ``ratio`` of the top hit's
    score, so the strongest match is always retained while padding that matched
    nothing (or barely anything) is cut before it can pollute the context and
    the ``sources`` line.
    """
    if not results:
        return results
    top_score = max(score for _id, score, _data in results)
    if top_score <= 0:
        # Degenerate/tiny corpus: BM25's IDF can push even a genuine match to
        # <= 0 (e.g. a term present in most of a handful of notes), so there's
        # no positive top to measure filler against. Keep everything rather than
        # drop a real, low-scored hit and answer "no documents found".
        return results
    threshold = top_score * ratio
    return [hit for hit in results if hit[1] > 0 and hit[1] >= threshold]


def _route_query(question: str) -> tuple[float, float]:
    """Pick per-source RRF weights ``(bm25, chroma)`` from the query's shape.

    Pure and deterministic — no I/O. Literal signals (quoted spans, id-shaped
    tokens, ALL-CAPS acronyms, very short queries) favour BM25 because lexical
    retrieval wins on exact spans the small embedding model can't disambiguate.
    Anything else is treated as conceptual, where paraphrase matching helps.
    """
    if '"' in question:
        return LITERAL_WEIGHTS

    tokens = question.split()
    if len(tokens) <= 2:
        return LITERAL_WEIGHTS

    for token in tokens:
        if _looks_like_id(token):
            return LITERAL_WEIGHTS

    if _ACRONYM_RE.search(question):
        return LITERAL_WEIGHTS

    return CONCEPTUAL_WEIGHTS


def _merge_and_dedupe(
    chroma_results: list[tuple[str, float, dict[str, Any]]],
    bm25_results: list[tuple[str, float, dict[str, Any]]],
    weights: tuple[float, float] = (1.0, 1.0),
) -> list[tuple[str, float, dict[str, Any]]]:
    """Fuse Chroma and BM25 results with Reciprocal Rank Fusion.

    Each input list is already rank-ordered. A chunk's fused score is
    ``Σ weight_i · 1/(RRF_K + rank_i)`` across the lists it appears in (rank is
    0-based), so its contribution is its *rank* in each source scaled by that
    source's weight, not the source-specific raw magnitude — this keeps an
    unbounded BM25 score from starving semantic hits. ``weights`` is
    ``(bm25_weight, chroma_weight)`` from the query router; the default
    ``(1.0, 1.0)`` reproduces unweighted fusion exactly. Results are deduped by
    ``(path, content_hash)``, keeping the best-ranked representative, and ordered
    by fused score. The fused score replaces the raw score for ordering only.
    """
    bm25_weight, chroma_weight = weights
    source_weights = {"chroma": chroma_weight, "bm25": bm25_weight}
    fused: dict[str, float] = {}
    best: dict[str, tuple[str, dict[str, Any]]] = {}
    best_rank: dict[str, int] = {}

    for source, results in (("chroma", chroma_results), ("bm25", bm25_results)):
        weight_source = source_weights[source]
        for rank, (chunk_id, _raw_score, data) in enumerate(results):
            data["source"] = data.get("source", source)
            signature = _signature(chunk_id, data)
            fused[signature] = fused.get(signature, 0.0) + weight_source * (
                1.0 / (RRF_K + rank)
            )
            if signature not in best_rank or rank < best_rank[signature]:
                best_rank[signature] = rank
                best[signature] = (chunk_id, data)

    merged = [
        (best[signature][0], score, best[signature][1])
        for signature, score in fused.items()
    ]
    merged.sort(key=lambda item: item[1], reverse=True)
    return merged


def _build_context(
    chunks: list[tuple[str, float, dict[str, Any]]],
    char_budget: int,
    config: ChirpSettings | None = None,
) -> tuple[str, list[str], list[str]]:
    """Build context string with round-robin truncation."""
    if not chunks:
        return "", [], []

    context_parts = []
    retrieved_ids = []

    chunks_to_include = []
    remaining_budget = char_budget

    for chunk_id, _score, data in chunks:
        content = data.get("content", "")
        if not content:
            continue

        header = _create_chunk_header(data)
        full_content = f"{header}\n{content}\n"

        if len(full_content) <= remaining_budget:
            chunks_to_include.append((chunk_id, full_content, data))
            remaining_budget -= len(full_content)
        else:
            min_size = len(header) + 100
            if min_size <= remaining_budget:
                truncated_content = (
                    content[: remaining_budget - len(header) - 20] + "..."
                )
                full_content = f"{header}\n{truncated_content}\n"
                chunks_to_include.append((chunk_id, full_content, data))
            break

    note_index = _build_note_index(config) if config is not None else {}

    sources = format_sources(chunks_to_include, note_index)
    for chunk_id, content, _data in chunks_to_include:
        context_parts.append(content)
        retrieved_ids.append(chunk_id)

    context = "\n".join(context_parts)
    return context, sources, retrieved_ids


def _build_note_index(config: ChirpSettings) -> dict[str, dict[str, Any]]:
    """Map slug → ``{"index": N, "title": ...}`` matching ``chirp notes``.

    The index is 1-based newest-first; the title falls back to the slug
    when ``meta.toml`` did not record one.
    """
    from utils.file_utils import list_notes

    notes_root = config.directories.notes_root
    records = [r for r in list_notes(notes_root) if r.notes is not None]
    return {
        record.slug: {"index": idx, "title": record.title or record.slug}
        for idx, record in enumerate(reversed(records), start=1)
    }


def format_sources(
    chunks_to_include: list[tuple[str, str, dict[str, Any]]],
    note_index: dict[str, dict[str, Any]],
) -> list[str]:
    """Format `sources:` line entries as ``note #N (Title · mm:ss)``.

    - `#N` matches the index `chirp notes` prints (newest-first, 1-based).
    - The note title (from ``meta.toml`` or, failing that, the slug)
      always sits inside the parentheses.
    - ``mm:ss`` is appended after a `·` separator only when the chunk
      metadata carries a timestamp (``start_ms``, ``start_seconds``, or
      ``timestamp_ms``).
    - Chunks from the same note collapse into one entry, keeping the
      earliest timestamp.
    """
    by_slug: dict[str, dict[str, Any]] = {}
    fallback_order: list[str] = []
    for chunk_id, _content, data in chunks_to_include:
        slug = _slug_from_chunk(data) or chunk_id
        timestamp = _timestamp_seconds_from_chunk(data)
        entry = by_slug.get(slug)
        if entry is None:
            entry = {"slug": slug, "timestamp": timestamp}
            by_slug[slug] = entry
            fallback_order.append(slug)
        elif timestamp is not None:
            current = entry.get("timestamp")
            if current is None or timestamp < current:
                entry["timestamp"] = timestamp

    sources: list[str] = []
    for slug in fallback_order:
        entry = by_slug[slug]
        info = note_index.get(slug)
        if info is None:
            sources.append(slug)
            continue

        title = info.get("title") or slug
        index = info.get("index")
        timestamp = entry.get("timestamp")
        suffix = f" · {_format_mm_ss(timestamp)}" if timestamp is not None else ""
        if index is None:
            sources.append(f"{title}{suffix}")
        else:
            sources.append(f"note #{index} ({title}{suffix})")
    return sources


def _slug_from_chunk(data: dict[str, Any]) -> str | None:
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if not metadata:
        return None
    path = metadata.get("path")
    if not path:
        return None
    return Path(path).parent.name or None


def _timestamp_seconds_from_chunk(data: dict[str, Any]) -> float | None:
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if not metadata:
        return None
    if (value := metadata.get("start_seconds")) is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    for key in ("start_ms", "timestamp_ms"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value) / 1000.0
        except (TypeError, ValueError):
            continue
    return None


def _format_mm_ss(seconds: float) -> str:
    total = int(max(seconds, 0))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _create_chunk_header(data: dict[str, Any]) -> str:
    """Create a header for a chunk."""
    if "metadata" in data:
        metadata = data["metadata"]
        date_str = metadata.get("date", "Unknown date")
        if date_str != "Unknown date":
            try:
                date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                date_str = date_obj.strftime("%Y-%m-%d")
            except (ValueError, AttributeError) as exc:
                logger.debug("could not normalize source date %r: %s", date_str, exc)

        path = metadata.get("path", "Unknown")
        filename = Path(path).name if path != "Unknown" else "Unknown"

        return f"{date_str} · {filename}"
    return "Unknown source"


def _generate_suggestion(config: ChirpSettings, time_range: Any | None) -> str:
    """Generate a helpful suggestion when no results are found."""
    try:
        index_dir = config.notes_chat.index_dir
        manifest_file = index_dir / "manifest.json"

        if not manifest_file.exists():
            return "Index not found. Run 'chirp index' to build the search index first."

        try:
            with manifest_file.open() as f:
                manifest = json.load(f)
                if not manifest:
                    return "No files in search index. Run 'chirp index' to build the index."
        except Exception:  # noqa: BLE001 - corrupt manifest: any read/parse error maps to the same actionable hint
            return "Index appears corrupted. Run 'chirp index --force' to rebuild."

        if not config.directories.notes_root.exists():
            return "No notes directory found. Try running 'chirp transcribe' to create some notes first."

        note_files = list(config.directories.notes_root.glob("*/notes.md"))
        if not note_files:
            return "No notes found. Try running 'chirp transcribe' to create some notes first."

        latest_note = max(note_files, key=lambda f: f.stat().st_mtime)
        latest_date = datetime.fromtimestamp(latest_note.stat().st_mtime)

        return f"Try a broader search or check notes from {latest_date.strftime('%Y-%m-%d')}"

    except OSError:
        return "Try a broader search or different keywords"


def _get_query_embedding(
    _config: ChirpSettings,
    query: str,
    client: LLMClient | None = None,
) -> list[float] | None:
    """Embed the query via chirpd. Returns None on empty input or LLM failure.

    None signals callers (``_search_chroma``) to skip vector search and fall
    back to BM25-only retrieval — the same contract the pre-chirpd path had.
    """
    if not query.strip():
        return None
    try:
        vectors = (client or LLMClient()).embed_sync(inputs=[query], model="default")
    except LLMError as exc:
        logger.debug("Query embedding failed: %s", exc)
        return None
    return vectors[0] if vectors else None
