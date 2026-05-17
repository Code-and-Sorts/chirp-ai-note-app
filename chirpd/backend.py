"""LLMBackend protocol and concrete implementations (MLX + Fake).

The Protocol defines the surface daemon state interacts with for model load /
unload / inference. ``MLXBackend`` is the production implementation that wraps
``mlx_lm`` and ``huggingface_hub``; ``FakeBackend`` is the unit-test double
exercised throughout ``tests/chirpd`` and ``tests/llm``.
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
        usage_out: dict[str, int],
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

    async def stream_generate(  # pragma: no cover — exercised via opt-in @slow tests
        self,
        handle: Any,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        should_stop: asyncio.Event,
        usage_out: dict[str, int],
    ) -> AsyncIterator[str]:
        from llm.exceptions import LLMGenerationFailed

        tokenizer = handle["tokenizer"]
        model = handle["model"]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        usage_out["prompt_tokens"] = len(tokenizer.encode(prompt))

        try:
            from mlx_lm import stream_generate as mlx_stream_generate
        except ImportError as err:
            raise LLMGenerationFailed(
                "mlx_lm is not installed; chirpd requires Apple Silicon dependencies",
                details={"error": str(err)},
            ) from err

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        SENTINEL_DONE = "done"
        SENTINEL_ERROR = "error"
        SENTINEL_TOKEN = "token"

        def _produce() -> None:
            try:
                for piece in mlx_stream_generate(model, tokenizer, prompt, **options):
                    if should_stop.is_set():
                        break
                    text = _extract_token_text(piece)
                    if text is None:
                        continue
                    loop.call_soon_threadsafe(queue.put_nowait, (SENTINEL_TOKEN, text))
                loop.call_soon_threadsafe(queue.put_nowait, (SENTINEL_DONE, None))
            except Exception as exc:  # noqa: BLE001 — surface to consumer
                loop.call_soon_threadsafe(queue.put_nowait, (SENTINEL_ERROR, exc))

        worker = asyncio.create_task(asyncio.to_thread(_produce))
        try:
            while True:
                kind, payload = await queue.get()
                if kind == SENTINEL_TOKEN:
                    yield payload
                    continue
                if kind == SENTINEL_ERROR:
                    raise LLMGenerationFailed(
                        f"mlx_lm.stream_generate failed: {payload}",
                        details={"exception_type": type(payload).__name__},
                    ) from payload
                return
        finally:
            should_stop.set()
            await worker

    async def embed(  # pragma: no cover — exercised via opt-in @slow tests
        self,
        handle: Any,
        inputs: list[str],
    ) -> list[list[float]]:
        model = handle["model"]
        tokenizer = handle["tokenizer"]

        def _run() -> list[list[float]]:
            results: list[list[float]] = []
            for text in inputs:
                token_ids = tokenizer.encode(text)
                vector = _invoke_embed(model, token_ids, handle.get("repo"))
                results.append(_vector_to_floats(vector))
            return results

        return await asyncio.to_thread(_run)


def _invoke_embed(model: Any, token_ids: Any, repo: Any) -> Any:  # pragma: no cover
    from llm.exceptions import LLMGenerationFailed

    embed_callable = getattr(model, "embed", None)
    if callable(embed_callable):
        return embed_callable(token_ids)
    embed_tokens = getattr(model, "embed_tokens", None)
    if callable(embed_tokens):
        return embed_tokens(token_ids)
    raise LLMGenerationFailed(
        "loaded model does not expose an 'embed' or 'embed_tokens' callable",
        details={"repo": repo, "model_type": type(model).__name__},
    )


def _extract_token_text(piece: Any) -> str | None:
    if isinstance(piece, str):
        return piece
    text = getattr(piece, "text", None)
    if isinstance(text, str):
        return text
    return None


def _vector_to_floats(vector: Any) -> list[float]:
    tolist = getattr(vector, "tolist", None)
    if callable(tolist):
        as_list = tolist()
    else:
        as_list = list(vector)
    return [float(x) for x in as_list]


class FakeBackend:
    """In-process deterministic backend for unit / integration tests."""

    def __init__(
        self,
        load_delay_s: float = 0.0,
        chat_tokens: list[str] | None = None,
        generation_delay_s: float = 0.0,
        embed_dim: int = 4,
        stream_raises: BaseException | None = None,
        stream_raises_after: int = 0,
    ) -> None:
        self.load_delay_s = load_delay_s
        self.chat_tokens = (
            chat_tokens if chat_tokens is not None else ["hello", " world"]
        )
        self.generation_delay_s = generation_delay_s
        self.embed_dim = embed_dim
        self.stream_raises = stream_raises
        self.stream_raises_after = stream_raises_after
        self.load_calls: list[tuple[str, ModelRole]] = []
        self.unload_calls: list[Any] = []
        self.load_raises: BaseException | None = None
        self.last_prompt: str | None = None
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_options: dict[str, Any] | None = None
        self.embed_calls: list[list[str]] = []

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

    async def stream_generate(
        self,
        handle: Any,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        should_stop: asyncio.Event,
        usage_out: dict[str, int],
    ) -> AsyncIterator[str]:
        self.last_messages = list(messages)
        self.last_options = dict(options)
        self.last_prompt = _render_fake_chat_template(messages)
        usage_out["prompt_tokens"] = len(self.last_prompt.split())
        emitted = 0
        for token in self.chat_tokens:
            if should_stop.is_set():
                return
            if self.generation_delay_s > 0:
                await asyncio.sleep(self.generation_delay_s)
            if should_stop.is_set():
                return
            if self.stream_raises is not None and emitted >= self.stream_raises_after:
                raise self.stream_raises
            yield token
            emitted += 1

    async def embed(
        self,
        handle: Any,
        inputs: list[str],
    ) -> list[list[float]]:
        self.embed_calls.append(list(inputs))
        return [_fake_embed_vector(text, self.embed_dim) for text in inputs]


def _render_fake_chat_template(messages: list[dict[str, Any]]) -> str:
    parts = [f"<{m.get('role', 'user')}>{m.get('content', '')}" for m in messages]
    parts.append("<assistant>")
    return "\n".join(parts)


def _fake_embed_vector(text: str, dim: int) -> list[float]:
    seed = (len(text), hash(text) % 100)
    if dim <= 0:
        return []
    if dim == 1:
        return [float(seed[0])]
    base = [float(seed[0]), float(seed[1])]
    if dim <= 2:
        return base[:dim]
    return base + [0.0] * (dim - 2)
