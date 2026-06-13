import hashlib
import json
import logging

from notes_chat.config import get_notes_config

logger = logging.getLogger(__name__)


def get_cached_answer(question: str, retrieved_ids: list[str]) -> str | None:
    """Get cached answer if available."""
    try:
        config = get_notes_config()
        cache_key = _generate_cache_key(question, retrieved_ids)
        cache_file = config.notes_chat.index_dir / "cache" / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        with cache_file.open() as f:
            data = json.load(f)

        answer = data.get("answer")
        return answer if isinstance(answer, str) else None

    except Exception as exc:  # noqa: BLE001 - get_notes_config or chromadb can raise many types
        logger.debug("get_cached_answer failed: %s", exc)
        return None


def cache_answer(question: str, retrieved_ids: list[str], answer: str) -> bool:
    """Cache an answer for future use."""
    try:
        config = get_notes_config()
        cache_key = _generate_cache_key(question, retrieved_ids)
        cache_file = config.notes_chat.index_dir / "cache" / f"{cache_key}.json"

        cache_file.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "question": question,
            "retrieved_ids": sorted(retrieved_ids),
            "answer": answer,
            "cache_key": cache_key,
        }

        with cache_file.open("w") as f:
            json.dump(cache_data, f, indent=2)

        return True

    except Exception as exc:  # noqa: BLE001 - get_notes_config or IO can raise many types
        logger.debug("cache_answer failed: %s", exc)
        return False


def _generate_cache_key(question: str, retrieved_ids: list[str]) -> str:
    """Generate a cache key from question and retrieved IDs."""
    sorted_ids = sorted(retrieved_ids)

    hash_input = f"{question}::{','.join(sorted_ids)}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def clear_cache() -> bool:
    """Clear the entire cache directory."""
    try:
        config = get_notes_config()
        cache_dir = config.notes_chat.index_dir / "cache"

        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.json"):
                cache_file.unlink()

        return True

    except Exception as exc:  # noqa: BLE001 - get_notes_config or IO can raise many types
        logger.debug("clear_cache failed: %s", exc)
        return False
