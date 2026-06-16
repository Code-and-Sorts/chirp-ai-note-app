"""LLMBackend protocol and concrete implementations (MLX + Fake).

The Protocol defines the surface daemon state interacts with for model load /
unload / inference. ``MLXBackend`` is the production implementation that wraps
``mlx_lm`` and ``huggingface_hub``; ``FakeBackend`` is the unit-test double
exercised throughout ``tests/chirpd`` and ``tests/llm``.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, runtime_checkable

from llm.exceptions import LLMModelLoadFailed

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

        if role == "embed":
            return await self._load_embed(repo, role, local_path)
        return await self._load_chat(repo, role, local_path)

    async def _load_chat(  # pragma: no cover — opt-in @slow @integration
        self, repo: str, role: ModelRole, local_path: str
    ) -> dict[str, Any]:
        try:
            from mlx_lm import load as mlx_load
        except ImportError as err:
            raise LLMModelLoadFailed(
                "mlx_lm is not installed; chirpd requires Apple Silicon dependencies",
                details={"error": str(err)},
            ) from err

        try:
            loaded = await asyncio.to_thread(mlx_load, local_path)
        except Exception as err:
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

    async def _load_embed(  # pragma: no cover — opt-in @slow @integration
        self, repo: str, role: ModelRole, local_path: str
    ) -> dict[str, Any]:
        try:
            from mlx_embeddings import load as mlx_embeddings_load
        except ImportError as err:
            raise LLMModelLoadFailed(
                "mlx_embeddings is not installed; chirpd requires "
                "Apple Silicon dependencies",
                details={"error": str(err)},
            ) from err

        try:
            # mlx_embeddings.load returns (model, tokenizer); for text models the
            # second element is a TokenizerWrapper. It is stored under "processor"
            # to match generate()'s parameter name and to read uniformly in embed().
            loaded = await asyncio.to_thread(mlx_embeddings_load, local_path)
        except Exception as err:
            raise LLMModelLoadFailed(
                f"mlx_embeddings.load failed for {repo!r}: {err}",
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
            "processor": loaded[1],
        }

    async def unload(self, handle: Any) -> None:
        if isinstance(handle, dict):
            handle.pop("model", None)
            handle.pop("tokenizer", None)
            handle.pop("processor", None)

        def _collect_and_release() -> None:
            gc.collect()
            # gc only drops the Python references; MLX parks freed Metal
            # buffers in its allocator cache, so without clearing it the
            # daemon's resident memory never shrinks after an idle unload.
            # Best-effort: MLX is gated to darwin+arm64, so off that platform
            # there is no cache to clear and the import simply won't resolve.
            try:
                import mlx.core as mx
            except ImportError:
                return
            mx.clear_cache()

        await asyncio.to_thread(_collect_and_release)

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
        prompt = _apply_chat_template_no_thinking(tokenizer, messages)
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
        worker_stop = threading.Event()

        def _produce() -> None:
            try:
                stopped_early = False
                for piece in mlx_stream_generate(model, tokenizer, prompt, **options):
                    if should_stop.is_set() or worker_stop.is_set():
                        stopped_early = True
                        break
                    text = _extract_token_text(piece)
                    if text is None:
                        continue
                    loop.call_soon_threadsafe(queue.put_nowait, (SENTINEL_TOKEN, text))
                if not stopped_early:
                    # Natural exhaustion (not a cancel-driven break): mark the full
                    # answer as complete so the dispatcher emits done, not cancel.
                    usage_out["completed"] = 1
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
            # Tell the worker to halt without mutating the dispatcher's
            # should_stop — otherwise normal completion would look like a cancel.
            worker_stop.set()
            await worker

    async def embed(  # pragma: no cover — exercised via opt-in @slow tests
        self,
        handle: Any,
        inputs: list[str],
    ) -> list[list[float]]:
        from llm.exceptions import LLMGenerationFailed

        if not inputs:
            return []

        model = handle["model"]
        processor = handle["processor"]

        try:
            from mlx_embeddings import generate as mlx_embeddings_generate
        except ImportError as err:
            raise LLMGenerationFailed(
                "mlx_embeddings is not installed; chirpd requires "
                "Apple Silicon dependencies",
                details={"error": str(err)},
            ) from err

        def _run() -> list[list[float]]:
            output = mlx_embeddings_generate(model, processor, texts=inputs)
            text_embeds = output.text_embeds
            return [_vector_to_floats(vector) for vector in text_embeds]

        try:
            return await asyncio.to_thread(_run)
        except Exception as err:
            raise LLMGenerationFailed(
                f"mlx_embeddings.generate failed for {handle.get('repo')!r}: {err}",
                details={
                    "repo": handle.get("repo"),
                    "exception_type": type(err).__name__,
                },
            ) from err


def _extract_token_text(piece: Any) -> str | None:
    if isinstance(piece, str):
        return piece
    text = getattr(piece, "text", None)
    if isinstance(text, str):
        return text
    return None


def _apply_chat_template_no_thinking(  # pragma: no cover — opt-in @slow path
    tokenizer: Any, messages: list[dict[str, Any]]
) -> str:
    # Reasoning-model templates (Qwen3, DeepSeek-R1, …) emit <think>…</think>
    # blocks ahead of the answer unless the template variable is unset. chirp's
    # grounded-answer flow never wants that on stdout, so we always ask the
    # template for non-thinking output and silently fall back if the tokenizer
    # rejects the kwarg.
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return str(rendered)


def _vector_to_floats(vector: Any) -> list[float]:
    tolist = getattr(vector, "tolist", None)
    as_list = tolist() if callable(tolist) else list(vector)
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
        scheduled_raise = self.load_raises
        if scheduled_raise is not None:
            raise scheduled_raise
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
        for emitted, token in enumerate(self.chat_tokens):
            if should_stop.is_set():
                return
            if self.generation_delay_s > 0:
                await asyncio.sleep(self.generation_delay_s)
            if should_stop.is_set():
                return
            scheduled_raise = self.stream_raises
            if scheduled_raise is not None and emitted >= self.stream_raises_after:
                raise scheduled_raise
            yield token
        # Reached only on natural exhaustion (no early return on should_stop):
        # signals the dispatcher the full answer streamed, so a cancel that
        # raced in at the end is graceful (done), not MODEL_CANCELLED.
        usage_out["completed"] = 1

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
