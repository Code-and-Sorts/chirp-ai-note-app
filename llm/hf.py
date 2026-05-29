"""HuggingFace boundary for the chirp model registry.

Every outbound ``huggingface_hub`` call lives in this module: repo validation,
snapshot download with progress feedback, and role inference from repo
metadata. Consumers (``chirp models add``, ``chirp models pull``) get a small
typed API and a single mock seam at the ``huggingface_hub`` import boundary.

The daemon-side ``local_files_only=True`` lookup at ``model.load`` time is
the one allowed exception and lives in ``chirpd/backend.py``.
"""
# TODO(convention-check): forbid huggingface_hub outside llm/hf.py once the
# tests/test_conventions.py AST-grep scaffolding lands (architecture
# §Enforcement, tracked in EPIC-CHIRPD-CORE).

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from tqdm import tqdm as _BaseTqdm

_EMBED_TAGS = frozenset({"sentence-transformers", "feature-extraction"})
_CHAT_TAGS = frozenset({"text-generation", "conversational"})
_EMBED_NAME_MARKERS = (
    "bge-",
    "e5-",
    "gte-",
    "mpnet",
    "minilm",
    "sentence-transformers",
)
_CHAT_ARCH_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")

_ROLE_AMBIGUOUS_HINT = "pass --role chat or --role embed explicitly"


class HfError(Exception):
    """Base class for HuggingFace boundary errors."""


class HfRepoNotFound(HfError):
    """Repo does not exist on HuggingFace (or is private without auth)."""

    def __init__(self, repo_id: str) -> None:
        super().__init__(f"HuggingFace repo not found: {repo_id}")
        self.repo_id = repo_id


class HfNetworkError(HfError):
    """Transient connectivity or 5xx response from the HuggingFace API."""

    def __init__(self, repo_id: str, *, original: Exception) -> None:
        status = _status_code_from(original)
        suffix = f" (status {status})" if status is not None else ""
        super().__init__(f"network error fetching {repo_id}{suffix}: {original}")
        self.repo_id = repo_id
        self.original = original


def _status_code_from(err: Exception) -> int | None:
    response = getattr(err, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


class HfDownloadFailed(HfError):
    """Local filesystem failure during snapshot_download (disk full, perm)."""

    def __init__(self, repo_id: str, *, original: Exception) -> None:
        super().__init__(f"download failed for {repo_id}: {original}")
        self.repo_id = repo_id
        self.original = original


class RoleInferenceAmbiguous(HfError):
    """Heuristic could not pick chat vs embed; caller must supply --role."""

    def __init__(self, repo_id: str, *, hint: str = _ROLE_AMBIGUOUS_HINT) -> None:
        super().__init__(f"could not infer role for {repo_id}: {hint}")
        self.repo_id = repo_id
        self.hint = hint


class HfUnexpectedError(HfError):
    """Catch-all for unmapped failures from the HuggingFace boundary.

    Any exception that escapes the named-error mappings — a new error class
    introduced by a future ``huggingface_hub`` release, an
    :class:`HFValidationError`, or a surprise Python error from inside
    third-party code — is re-raised as this type so callers' ``except
    HfError`` blocks stay complete.
    """

    def __init__(self, repo_id: str, *, original: Exception) -> None:
        super().__init__(f"unexpected HuggingFace error for {repo_id}: {original}")
        self.repo_id = repo_id
        self.original = original


@dataclass
class HfRepoMetadata:
    repo_id: str
    tags: list[str]
    siblings_count: int
    architectures: list[str]
    sha: str | None


@dataclass
class DownloadResult:
    repo_id: str
    local_path: Path
    bytes_downloaded: int
    cache_hit: bool


class ProgressCallback(Protocol):
    def on_start(self, total_bytes: int | None) -> None: ...
    def on_progress(self, bytes_downloaded: int, total_bytes: int | None) -> None: ...
    def on_done(self) -> None: ...


def validate_repo(repo_id: str) -> HfRepoMetadata:
    """Confirm ``repo_id`` exists on HF and capture the metadata role inference needs."""
    try:
        info = HfApi().repo_info(repo_id=repo_id)
    except RepositoryNotFoundError as err:
        raise HfRepoNotFound(repo_id) from err
    except HfHubHTTPError as err:
        raise HfNetworkError(repo_id, original=err) from err
    except Exception as err:
        raise HfUnexpectedError(repo_id, original=err) from err

    architectures = _fetch_architectures(repo_id)
    tags = list(getattr(info, "tags", None) or [])
    siblings = list(getattr(info, "siblings", None) or [])
    return HfRepoMetadata(
        repo_id=repo_id,
        tags=tags,
        siblings_count=len(siblings),
        architectures=architectures,
        sha=getattr(info, "sha", None),
    )


def _fetch_architectures(repo_id: str) -> list[str]:
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except EntryNotFoundError:
        return []
    except RepositoryNotFoundError as err:
        raise HfRepoNotFound(repo_id) from err
    except HfHubHTTPError as err:
        raise HfNetworkError(repo_id, original=err) from err
    except Exception as err:
        raise HfUnexpectedError(repo_id, original=err) from err

    try:
        with open(config_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []

    raw = data.get("architectures") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(arch) for arch in raw]


def infer_role(metadata: HfRepoMetadata) -> Literal["chat", "embed"]:
    """Classify ``metadata`` as chat or embed using a 5-step priority heuristic.

    Order matters — first match wins:

    1. ``sentence-transformers`` / ``feature-extraction`` tag → ``embed``.
    2. Repo basename contains ``bge-`` / ``e5-`` / ``gte-`` / ``mpnet`` /
       ``minilm`` / ``sentence-transformers`` → ``embed``.
    3. ``text-generation`` / ``conversational`` tag → ``chat``.
    4. ``config.json`` architecture suffix ``ForCausalLM`` /
       ``ForConditionalGeneration`` → ``chat``.
    5. Otherwise raise :class:`RoleInferenceAmbiguous`.

    The bias toward ``RoleInferenceAmbiguous`` is intentional: a wrong role
    at registry time surfaces as confusing daemon behavior much later. Bare
    ``BertModel`` architectures fall through to step 5 because the same
    architecture serves both sentence-transformer embeddings and downstream
    classification — without a ``pooling_mode_*`` signal we can't tell.
    """
    tag_set = {tag.lower() for tag in metadata.tags}

    if tag_set & _EMBED_TAGS:
        return "embed"

    basename = metadata.repo_id.rsplit("/", 1)[-1].lower()
    if any(marker in basename for marker in _EMBED_NAME_MARKERS):
        return "embed"

    if tag_set & _CHAT_TAGS:
        return "chat"

    for arch in metadata.architectures:
        if arch.endswith(_CHAT_ARCH_SUFFIXES):
            return "chat"

    raise RoleInferenceAmbiguous(repo_id=metadata.repo_id)


def download_model(
    repo_id: str, *, progress: ProgressCallback | None = None
) -> DownloadResult:
    """Snapshot-download ``repo_id`` into the standard HF cache.

    A custom ``tqdm_class`` (a :class:`tqdm.tqdm` subclass) is always passed to
    ``snapshot_download`` so we can drive the ``ProgressCallback`` and total the
    downloaded bytes. ``huggingface_hub`` (pin 1.8) constructs that class twice:
    once as a shared byte-progress bar (``unit="B"``) whose total grows as files
    are discovered, and once as the ``thread_map`` file-count bar. Subclassing
    real ``tqdm`` is what makes the file-count usage work — ``thread_map``
    requires the class-level lock protocol and iterates the bar to drive the
    downloads, neither of which a hand-rolled stand-in provides.

    Only the byte bar feeds the callback and the byte total; the file-count bar
    is ignored. ``DownloadResult.cache_hit`` is ``True`` when no bytes were
    transferred — the byte bar is always constructed even on a warm cache, so
    its construction is not itself a cache-miss signal. On a warm cache
    ``on_start``/``on_done`` still fire (with a zero byte total) because
    ``huggingface_hub`` reports the snapshot total incrementally, not upfront.

    On failure the typed exception is the only signal; no callbacks fire after
    it is raised.
    """
    state: dict[str, Any] = {"started": False, "bytes": 0}
    tqdm_cls = _build_tqdm_adapter(state, progress)

    try:
        local_path = snapshot_download(repo_id=repo_id, tqdm_class=tqdm_cls)
    except RepositoryNotFoundError as err:
        raise HfRepoNotFound(repo_id) from err
    except HfHubHTTPError as err:
        raise HfNetworkError(repo_id, original=err) from err
    except OSError as err:
        raise HfDownloadFailed(repo_id, original=err) from err
    except Exception as err:
        raise HfUnexpectedError(repo_id, original=err) from err

    if progress is not None and state["started"]:
        progress.on_done()

    bytes_downloaded = int(state["bytes"])
    return DownloadResult(
        repo_id=repo_id,
        local_path=Path(local_path),
        bytes_downloaded=bytes_downloaded,
        cache_hit=bytes_downloaded == 0,
    )


def _build_tqdm_adapter(
    state: dict[str, Any], progress: ProgressCallback | None
) -> type[_BaseTqdm]:
    class _CallbackTqdm(_BaseTqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            is_byte_bar = kwargs.get("unit") == "B"
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            if is_byte_bar and state.get("byte_bar") is None:
                state["byte_bar"] = self
                state["started"] = True
                if progress is not None:
                    progress.on_start(self.total or None)

        def update(self, n: float | None = 1) -> None:
            super().update(n)
            if state.get("byte_bar") is self and n:
                state["bytes"] += int(n)
                if progress is not None:
                    progress.on_progress(state["bytes"], self.total or None)

    return _CallbackTqdm
