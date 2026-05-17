"""LLMBackend protocol and concrete implementations (MLX + Fake).

The Protocol defines the surface daemon state interacts with for model load /
unload / inference. ``MLXBackend`` is the production implementation that wraps
``mlx_lm`` and ``huggingface_hub``; ``FakeBackend`` is the unit-test double
exercised throughout ``tests/chirpd`` and ``tests/llm``.

Inference methods (``stream_generate``, ``embed``) are stubbed in this story
and filled in by story 3.6.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, runtime_checkable

from llm.exceptions import LLMModelLoadFailed

_logger = logging.getLogger("chirpd.backend")

ModelRole = Literal["chat", "embed"]


@runtime_checkable
class LLMBackend(Protocol):
    """Inference-backend protocol used by ``DaemonState``."""

    async def load(self, repo: str, role: ModelRole) -> Any: ...

    async def unload(self, handle: Any) -> None: ...

    def stream_generate(
        self,
        handle: Any,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        should_stop: asyncio.Event,
    ) -> AsyncIterator[str]: ...

    async def embed(
        self,
        handle: Any,
        inputs: list[str],
    ) -> list[list[float]]: ...


class MLXBackend:
    """Production backend wrapping ``mlx_lm`` + ``huggingface_hub``."""

    async def load(
        self, repo: str, role: ModelRole
    ) -> Any:  # pragma: no cover — opt-in @slow @integration per AC-27
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError as err:
            raise LLMModelLoadFailed(
                "huggingface_hub is not installed; chirpd requires "
                "Apple Silicon dependencies",
                details={"error": str(err)},
            ) from err

        try:
            local_path = await asyncio.to_thread(
                snapshot_download, repo, local_files_only=True
            )
        except LocalEntryNotFoundError as err:
            raise LLMModelLoadFailed(
                f"weights not in HF cache for {repo!r}; "
                f"run `chirp models pull <alias>` to download",
                details={"repo": repo, "role": role},
            ) from err

        try:
            from mlx_lm import load as mlx_load
        except ImportError as err:
            raise LLMModelLoadFailed(
                "mlx_lm is not installed; chirpd requires Apple Silicon dependencies",
                details={"error": str(err)},
            ) from err

        try:
            loaded = await asyncio.to_thread(mlx_load, local_path)
        except Exception as err:  # noqa: BLE001 — wrap any mlx error
            raise LLMModelLoadFailed(
                f"mlx_lm.load failed for {repo!r}: {err}",
                details={
                    "repo": repo,
                    "role": role,
                    "local_path": str(local_path),
                    "exception_type": type(err).__name__,
                },
            ) from err

        return {
            "repo": repo,
            "role": role,
            "local_path": str(local_path),
            "model": loaded[0],
            "tokenizer": loaded[1],
        }

    async def unload(self, handle: Any) -> None:
        if isinstance(handle, dict):
            handle.pop("model", None)
            handle.pop("tokenizer", None)
        await asyncio.to_thread(gc.collect)

    def stream_generate(
        self,
        handle: Any,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        should_stop: asyncio.Event,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("MLXBackend.stream_generate lands in story 3.6")

    async def embed(
        self,
        handle: Any,
        inputs: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError("MLXBackend.embed lands in story 3.6")


class FakeBackend:
    """In-process deterministic backend for unit / integration tests."""

    def __init__(self, load_delay_s: float = 0.0) -> None:
        self.load_delay_s = load_delay_s
        self.load_calls: list[tuple[str, ModelRole]] = []
        self.unload_calls: list[Any] = []
        self.load_raises: BaseException | None = None

    async def load(self, repo: str, role: ModelRole) -> Any:
        self.load_calls.append((repo, role))
        if self.load_delay_s > 0:
            await asyncio.sleep(self.load_delay_s)
        if self.load_raises is not None:
            raise self.load_raises
        return {"repo": repo, "role": role, "loaded": True}

    async def unload(self, handle: Any) -> None:
        self.unload_calls.append(handle)
        if isinstance(handle, dict):
            handle.clear()

    def stream_generate(
        self,
        handle: Any,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        should_stop: asyncio.Event,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("FakeBackend.stream_generate lands in story 3.6")

    async def embed(
        self,
        handle: Any,
        inputs: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError("FakeBackend.embed lands in story 3.6")
