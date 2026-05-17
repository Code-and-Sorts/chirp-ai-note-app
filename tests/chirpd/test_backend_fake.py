"""Tests for :class:`chirpd.backend.FakeBackend`."""

from __future__ import annotations

import asyncio

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


async def test_fake_backend_stream_generate_stub_raises() -> None:
    backend = FakeBackend()
    handle = await backend.load("mlx-community/foo", "chat")
    with pytest.raises(NotImplementedError):
        backend.stream_generate(handle, [], {}, asyncio.Event())


async def test_fake_backend_embed_stub_raises() -> None:
    backend = FakeBackend()
    handle = await backend.load("mlx-community/foo", "embed")
    with pytest.raises(NotImplementedError):
        await backend.embed(handle, ["hello"])


async def test_mlx_backend_unload_clears_handle() -> None:
    from chirpd.backend import MLXBackend

    backend = MLXBackend()
    handle = {"model": object(), "tokenizer": object(), "repo": "x"}
    await backend.unload(handle)
    assert "model" not in handle
    assert "tokenizer" not in handle
