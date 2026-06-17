import hashlib
import json
import logging
import time

from notes_chat.config import PROMPT_VERSION, get_notes_config

logger = logging.getLogger(__name__)

# Answers older than the TTL are treated as misses (and pruned). Bounds staleness
# even when the inputs that key the cache haven't changed.
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

# Hard cap on cached answer files; the oldest are evicted past this so the cache
# dir can't grow without bound (it previously wrote one file per key, forever).
CACHE_MAX_ENTRIES = 500


def _resolved_models() -> tuple[str, str]:
    """Resolve the active chat + embed aliases (best effort) for cache keying.

    A `chirp models default <alias>` switch must invalidate cached answers, so
    the resolved models are part of the key. Falls back to ``"unknown"`` when the
    registry is absent so caching still works (just not model-discriminated).
    """
    try:
        from llm.registry import read_registry

        registry = read_registry()
        return (
            registry.default_chat or "unknown",
            registry.default_embed or "unknown",
        )
    except Exception as exc:  # noqa: BLE001 - registry IO/parse: keying is best-effort
        logger.debug("Could not resolve models for cache key: %s", exc)
        return ("unknown", "unknown")


def get_cached_answer(question: str, retrieved_ids: list[str]) -> str | None:
    """Get cached answer if available and not expired."""
    try:
        config = get_notes_config()
        cache_key = _generate_cache_key(question, retrieved_ids)
        cache_file = config.notes_chat.index_dir / "cache" / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        if time.time() - cache_file.stat().st_mtime > CACHE_TTL_SECONDS:
            cache_file.unlink(missing_ok=True)
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
        cache_dir = config.notes_chat.index_dir / "cache"
        cache_key = _generate_cache_key(question, retrieved_ids)
        cache_file = cache_dir / f"{cache_key}.json"

        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "question": question,
            "retrieved_ids": sorted(retrieved_ids),
            "answer": answer,
            "cache_key": cache_key,
        }

        with cache_file.open("w") as f:
            json.dump(cache_data, f, indent=2)

        _evict_if_needed(cache_dir)
        return True

    except Exception as exc:  # noqa: BLE001 - get_notes_config or IO can raise many types
        logger.debug("cache_answer failed: %s", exc)
        return False


def _evict_if_needed(cache_dir) -> None:
    """Drop the oldest entries once the cache exceeds ``CACHE_MAX_ENTRIES``."""
    overflow = len(list(cache_dir.glob("*.json"))) - CACHE_MAX_ENTRIES
    if overflow <= 0:
        # Under the cap: nothing to evict. (A bare negative slice would delete
        # almost everything — `entries[:-498]` keeps only the newest 2.)
        return
    entries = sorted(cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for stale in entries[:overflow]:
        stale.unlink(missing_ok=True)


def _generate_cache_key(question: str, retrieved_ids: list[str]) -> str:
    """Key a cached answer by everything that determines the answer.

    Beyond the question and retrieved ids, the resolved chat model, resolved
    embed model, and a prompt-version token are folded in so a model switch or a
    prompt edit produces a different key (no stale hit from the old model/prompt).
    """
    sorted_ids = sorted(retrieved_ids)
    chat_model, embed_model = _resolved_models()

    hash_input = (
        f"{question}::{','.join(sorted_ids)}"
        f"::chat={chat_model}::embed={embed_model}::prompt={PROMPT_VERSION}"
    )
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
