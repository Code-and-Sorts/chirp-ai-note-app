import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from config.settings import ChirpSettings
from notes_chat.bm25 import BM25Index
from notes_chat.index import IndexManager
from notes_chat.time_ranges import parse_time_range

logger = logging.getLogger(__name__)


def retrieve_context(
    config: ChirpSettings, question: str, when_filter: Optional[str] = None
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

        index_result = index_manager.build_index(force=False)
        if not index_result["success"]:
            return {
                "success": False,
                "error": f"Failed to update index: {index_result.get('error')}",
            }

        time_range = None
        if when_filter:
            time_range = parse_time_range(question, when_filter)
        else:
            time_range = parse_time_range(question)

        chroma_results = _search_chroma(
            index_manager, question, config.notes_chat.k, time_range
        )
        bm25_results = _search_bm25(
            index_manager.bm25_file, question, config.notes_chat.k
        )

        if not chroma_results and not bm25_results:
            suggestion = _generate_suggestion(config, time_range)
            return {
                "success": False,
                "error": "No documents found matching your query",
                "suggestion": suggestion,
            }

        merged_chunks = _merge_and_dedupe(chroma_results, bm25_results)

        context, sources, retrieved_ids = _build_context(
            merged_chunks, config.notes_chat.ctx_char_budget
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

    except Exception as e:
        return {"success": False, "error": str(e)}


def _search_chroma(
    index_manager: IndexManager, query: str, k: int, time_range: Optional[Any] = None
) -> list[tuple[str, float, dict[str, Any]]]:
    """Search Chroma vector database."""
    try:
        where_clause = None
        if time_range:
            start_iso = time_range.start.isoformat()
            end_iso = time_range.end_exclusive.isoformat()
            where_clause = {
                "$and": [{"date": {"$gte": start_iso}}, {"date": {"$lt": end_iso}}]
            }

        query_embedding = _get_query_embedding(index_manager.config, query)
        if not query_embedding:
            return []

        results = index_manager.collection.query(
            query_embeddings=[query_embedding],  # type: ignore
            n_results=k,
            where=where_clause,  # type: ignore
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

        return chroma_results

    except Exception as e:
        logger.debug("Chroma search failed: %s", e, exc_info=True)
        return []


def _search_bm25(
    bm25_file: Path, query: str, k: int
) -> list[tuple[str, float, dict[str, Any]]]:
    """Search BM25 index."""
    try:
        bm25_index = BM25Index(bm25_file)
        bm25_results = bm25_index.search(query, k)

        return [
            (chunk_id, score, {"source": "bm25"}) for chunk_id, score in bm25_results
        ]

    except Exception as e:
        logger.debug("BM25 search failed: %s", e, exc_info=True)
        return []


def _merge_and_dedupe(
    chroma_results: list[tuple[str, float, dict[str, Any]]],
    bm25_results: list[tuple[str, float, dict[str, Any]]],
) -> list[tuple[str, float, dict[str, Any]]]:
    """Merge Chroma and BM25 results, deduplicating by (path, content_hash)."""
    seen_signatures: set[str] = set()
    merged = []

    all_results = []

    for chunk_id, score, data in chroma_results:
        data["source"] = "chroma"
        all_results.append((chunk_id, score, data))

    all_results.extend(bm25_results)

    all_results.sort(key=lambda x: x[1], reverse=True)

    for chunk_id, score, data in all_results:
        if "metadata" in data:
            path = data["metadata"].get("path", "")
            content_hash = data["metadata"].get("content_hash", "")
            signature = f"{path}::{content_hash}"
        else:
            signature = chunk_id

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            merged.append((chunk_id, score, data))

    return merged


def _build_context(
    chunks: list[tuple[str, float, dict[str, Any]]], char_budget: int
) -> tuple[str, list[str], list[str]]:
    """Build context string with round-robin truncation."""
    if not chunks:
        return "", [], []

    context_parts = []
    sources = []
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

    seen_sources = set()
    for chunk_id, content, data in chunks_to_include:
        context_parts.append(content)
        retrieved_ids.append(chunk_id)

        if "metadata" in data:
            path = data["metadata"].get("path", "Unknown")
            title = data["metadata"].get("title", "Unknown")
            source_text = f"{title} ({Path(path).name})"
            if source_text not in seen_sources:
                seen_sources.add(source_text)
                sources.append(source_text)
        else:
            source_text = f"Document {chunk_id}"
            if source_text not in seen_sources:
                seen_sources.add(source_text)
                sources.append(source_text)

    context = "\n".join(context_parts)
    return context, sources, retrieved_ids


def _create_chunk_header(data: dict[str, Any]) -> str:
    """Create a header for a chunk."""
    if "metadata" in data:
        metadata = data["metadata"]
        date_str = metadata.get("date", "Unknown date")
        if date_str != "Unknown date":
            try:
                date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                date_str = date_obj.strftime("%Y-%m-%d")
            except:
                pass

        path = metadata.get("path", "Unknown")
        filename = Path(path).name if path != "Unknown" else "Unknown"

        return f"{date_str} · {filename}"
    else:
        return "Unknown source"


def _generate_suggestion(config: ChirpSettings, time_range: Optional[Any]) -> str:
    """Generate a helpful suggestion when no results are found."""
    try:
        index_dir = config.notes_chat.index_dir
        manifest_file = index_dir / "manifest.json"

        if not manifest_file.exists():
            return "Index not found. Run 'chirp notes index' to build the search index first."

        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
                if not manifest:
                    return "No files in search index. Run 'chirp notes index' to build the index."
        except:
            return (
                "Index appears corrupted. Run 'chirp notes index --force' to rebuild."
            )

        if not config.directories.notes.exists():
            return "No notes directory found. Try running 'chirp process' to create some notes first."

        note_files = list(config.directories.notes.glob("*.md"))
        if not note_files:
            return "No notes found. Try running 'chirp process' to create some notes first."

        latest_note = max(note_files, key=lambda f: f.stat().st_mtime)
        latest_date = datetime.fromtimestamp(latest_note.stat().st_mtime)

        return f"Try a broader search or check notes from {latest_date.strftime('%Y-%m-%d')}"

    except Exception:
        return "Try a broader search or different keywords"


def _get_query_embedding(config: ChirpSettings, query: str) -> Optional[list[float]]:
    """Get embedding for query text using the same model as indexing."""
    try:
        response = requests.post(
            f"{config.models.ollama_url}/api/embeddings",
            json={"model": config.notes_chat.emb_model, "prompt": query},
            timeout=30,
        )
        if response.status_code != 200:
            return None
        result = response.json()
        embedding = result.get("embedding")
        if isinstance(embedding, list) and all(
            isinstance(x, (int, float)) for x in embedding
        ):
            return embedding
        return None
    except Exception:
        return None
