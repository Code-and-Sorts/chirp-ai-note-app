"""Tests for :class:`chirpd.backend.FakeBackend`."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from chirpd.backend import FakeBackend, LLMBackend


async def test_fake_backend_load_unload_round_trip() -> None:
    backend = FakeBackend()
    handle = await backend.load("mlx-community/foo", "chat")
    assert handle["repo"] == "mlx-community/foo"
    assert handle["role"] == "chat"
    assert handle["loaded"] is True

    await backend.unload(handle)
    assert backend.unload_calls == [handle]


def test_fake_backend_implements_protocol() -> None:
    backend = FakeBackend()
    assert isinstance(backend, LLMBackend)


async def test_fake_backend_load_delay_respected() -> None:
    backend = FakeBackend(load_delay_s=0.05)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await backend.load("mlx-community/foo", "chat")
    elapsed = loop.time() - started
    assert elapsed >= 0.04


async def test_fake_backend_stream_generate_yields_scripted_tokens() -> None:
    backend = FakeBackend(chat_tokens=["one", "two", "three"])
    handle = await backend.load("mlx-community/foo", "chat")
    usage: dict[str, int] = {}
    tokens: list[str] = []
    async for token in backend.stream_generate(
        handle,
        [{"role": "user", "content": "hi"}],
        {},
        asyncio.Event(),
        usage,
    ):
        tokens.append(token)
    assert tokens == ["one", "two", "three"]
    assert backend.last_prompt is not None
    assert "hi" in backend.last_prompt
    assert usage["prompt_tokens"] > 0


async def test_fake_backend_stream_generate_respects_should_stop() -> None:
    backend = FakeBackend(chat_tokens=["a", "b", "c", "d"], generation_delay_s=0.02)
    handle = await backend.load("mlx-community/foo", "chat")
    should_stop = asyncio.Event()

    async def _cancel_after() -> None:
        await asyncio.sleep(0.03)
        should_stop.set()

    cancel_task = asyncio.create_task(_cancel_after())
    tokens: list[str] = []
    async for token in backend.stream_generate(
        handle,
        [{"role": "user", "content": "hi"}],
        {},
        should_stop,
        {},
    ):
        tokens.append(token)
    await cancel_task
    assert len(tokens) < 4


async def test_fake_backend_embed_returns_vector_per_input() -> None:
    backend = FakeBackend(embed_dim=3)
    handle = await backend.load("mlx-community/foo", "embed")
    vectors = await backend.embed(handle, ["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == 3 for v in vectors)
    # Ordering preserved: different input → different leading value (length-based).
    assert vectors[0][0] == 3.0
    assert vectors[1][0] == 3.0
    assert vectors[2][0] == 5.0


async def test_fake_backend_stream_generate_raises_after_n_tokens() -> None:
    backend = FakeBackend(
        chat_tokens=["a", "b", "c"],
        stream_raises=RuntimeError("kaboom"),
        stream_raises_after=2,
    )
    handle = await backend.load("mlx-community/foo", "chat")
    tokens: list[str] = []
    with pytest.raises(RuntimeError):
        async for token in backend.stream_generate(
            handle, [{"role": "user", "content": "x"}], {}, asyncio.Event(), {}
        ):
            tokens.append(token)
    assert tokens == ["a", "b"]


async def test_fake_backend_stream_generate_raises_pre_first_token() -> None:
    backend = FakeBackend(stream_raises=RuntimeError("kaboom"))
    handle = await backend.load("mlx-community/foo", "chat")
    with pytest.raises(RuntimeError):
        async for _ in backend.stream_generate(
            handle, [{"role": "user", "content": "x"}], {}, asyncio.Event(), {}
        ):
            pass


async def test_mlx_backend_unload_clears_handle() -> None:
    from chirpd.backend import MLXBackend

    backend = MLXBackend()
    handle = {"model": object(), "tokenizer": object(), "repo": "x"}
    await backend.unload(handle)
    assert "model" not in handle
    assert "tokenizer" not in handle


def test_apply_chat_template_no_thinking_passes_kwarg_when_supported() -> None:
    from chirpd.backend import _apply_chat_template_no_thinking

    calls: list[dict] = []

    class _AcceptsKwarg:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return f"prompt-from-{len(messages)}-messages"

    out = _apply_chat_template_no_thinking(
        _AcceptsKwarg(), [{"role": "user", "content": "hi"}]
    )
    assert out == "prompt-from-1-messages"
    assert calls[0]["enable_thinking"] is False
    assert calls[0]["add_generation_prompt"] is True
    assert calls[0]["tokenize"] is False


def test_apply_chat_template_no_thinking_falls_back_on_typeerror() -> None:
    from chirpd.backend import _apply_chat_template_no_thinking

    seen_kwargs: list[dict] = []

    class _RejectsKwarg:
        def apply_chat_template(self, messages, **kwargs):
            seen_kwargs.append(kwargs)
            if "enable_thinking" in kwargs:
                raise TypeError("got unexpected keyword 'enable_thinking'")
            return "fallback-prompt"

    out = _apply_chat_template_no_thinking(
        _RejectsKwarg(), [{"role": "user", "content": "hi"}]
    )
    assert out == "fallback-prompt"
    # First call attempts with enable_thinking; fallback call omits it.
    assert "enable_thinking" in seen_kwargs[0]
    assert "enable_thinking" not in seen_kwargs[1]


async def test_mlx_backend_unload_clears_handle_and_survives_missing_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """unload() drops the heavy handle refs and treats MLX as best-effort.

    MLX is gated to darwin+arm64, so on other CI platforms `import mlx.core`
    raises ImportError — unload must swallow it (there is no cache to clear)
    rather than failing the model.unload op.
    """
    from chirpd.backend import MLXBackend

    # A None entry in sys.modules makes `import mlx.core` raise ImportError,
    # simulating a non-darwin/arm64 host where the dep isn't installed.
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)

    handle = {"repo": "r", "role": "chat", "model": object(), "tokenizer": object()}
    await MLXBackend().unload(handle)  # must not raise

    assert "model" not in handle
    assert "tokenizer" not in handle


async def test_mlx_backend_unload_clears_cache_when_mlx_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MLX is importable, unload() clears its Metal allocator cache."""
    from chirpd.backend import MLXBackend

    cleared: list[bool] = []
    fake_mlx_core = types.SimpleNamespace(clear_cache=lambda: cleared.append(True))
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mlx_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    await MLXBackend().unload({"model": object()})

    assert cleared == [True]
