"""Tests for :class:`chirpd.state.DaemonState`."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from chirpd.backend import FakeBackend
from chirpd.state import DaemonState
from llm.registry import Registry, RegistryEntry


def _registry(**entries: RegistryEntry) -> Registry:
    return Registry(schema_version=1, models=dict(entries))


def _chat_entry(repo: str = "mlx-community/gemma-4-4b-it-4bit") -> RegistryEntry:
    return RegistryEntry(hf_repo=repo, role="chat")


def _embed_entry(repo: str = "mlx-community/nomic-embed") -> RegistryEntry:
    return RegistryEntry(hf_repo=repo, role="embed")


async def test_load_chat_model_schedules_idle_unload() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=0.1)

    loaded = await state.load("gemma", "chat")
    assert loaded.idle_unload_task is not None

    await asyncio.sleep(0.25)
    assert state.get("gemma") is None
    assert backend.unload_calls, "FakeBackend.unload should have been called"


async def test_load_embed_model_never_schedules_idle_unload() -> None:
    backend = FakeBackend()
    registry = _registry(nomic=_embed_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=0.05)

    loaded = await state.load("nomic", "embed")
    assert loaded.idle_unload_task is None

    await asyncio.sleep(0.15)
    assert state.get("nomic") is not None


async def test_keep_alive_minus_one_pins_model() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=0.05)

    loaded = await state.load("gemma", "chat")
    state.schedule_idle_unload(loaded, keep_alive=-1)
    assert loaded.idle_unload_task is None

    await asyncio.sleep(0.15)
    assert state.get("gemma") is not None


async def test_keep_alive_zero_unloads_immediately() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    loaded = await state.load("gemma", "chat")
    state.schedule_idle_unload(loaded, keep_alive=0)

    for _ in range(20):
        if state.get("gemma") is None:
            break
        await asyncio.sleep(0.01)
    assert state.get("gemma") is None


async def test_idle_unload_resets_on_activity() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=10.0)

    loaded = await state.load("gemma", "chat")
    first_task = loaded.idle_unload_task
    assert first_task is not None

    state.schedule_idle_unload(loaded, keep_alive=None)
    second_task = loaded.idle_unload_task
    assert second_task is not None
    assert second_task is not first_task
    for _ in range(10):
        if first_task.done():
            break
        await asyncio.sleep(0)
    assert first_task.done()


async def test_concurrent_load_of_same_alias_serializes() -> None:
    registry = _registry(gemma=_chat_entry())
    inside_backend_load = asyncio.Event()
    release_backend_load = asyncio.Event()

    from chirpd.backend import ModelRole

    class _BlockingBackend(FakeBackend):
        async def load(self, repo: str, role: ModelRole) -> Any:
            self.load_calls.append((repo, role))
            inside_backend_load.set()
            await release_backend_load.wait()
            return {"repo": repo, "role": role, "loaded": True}

    backend = _BlockingBackend()
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    first = asyncio.create_task(state.load("gemma", "chat"))
    await inside_backend_load.wait()

    second = asyncio.create_task(state.load("gemma", "chat"))
    for _ in range(20):
        await asyncio.sleep(0)
        if not second.done():
            break
    assert not second.done(), "second load should be blocked on per-model lock"
    assert state._registry_locks["gemma"].locked(), "lock must be held during load"

    release_backend_load.set()
    results = await asyncio.gather(first, second)
    assert results[0] is results[1]
    assert len(backend.load_calls) == 1


async def test_status_reports_loaded_models_with_rss() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    await state.load("gemma", "chat")
    status = state.status()
    assert status["rss_bytes"] > 0
    assert status["daemon_version"]
    assert status["pid"] > 0
    aliases = [model["alias"] for model in status["models"]]
    assert "gemma" in aliases


async def test_list_models_returns_registered_and_loaded_states() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry(), nomic=_embed_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    await state.load("gemma", "chat")
    listed = state.list_models()
    by_alias = {item["alias"]: item for item in listed}
    assert by_alias["gemma"]["loaded"] is True
    assert by_alias["nomic"]["loaded"] is False
    assert by_alias["nomic"]["last_used_seconds_ago"] is None


async def test_unload_unknown_alias_is_noop() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    await state.unload("never-loaded")
    assert backend.unload_calls == []


async def test_idle_task_skips_unload_when_activity_advanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=0.05)

    loaded = await state.load("gemma", "chat")
    loaded.last_used = datetime.now(UTC) + timedelta(seconds=10)

    await asyncio.sleep(0.15)
    assert state.get("gemma") is not None


async def test_load_default_alias_resolves_to_configured_default() -> None:
    backend = FakeBackend()
    registry = Registry(
        schema_version=1,
        default_chat="gemma",
        models={"gemma": _chat_entry()},
    )
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    loaded = await state.load("default", "chat")
    assert loaded.alias == "gemma"


async def test_list_models_includes_loaded_not_in_registry() -> None:
    backend = FakeBackend()
    registry = Registry(schema_version=1, models={})
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    await state.load("mlx-community/raw-repo", "chat")
    listed = state.list_models()
    aliases = [item["alias"] for item in listed]
    assert "mlx-community/raw-repo" in aliases


async def test_state_exposes_registry_version_and_timeout() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=42.0)

    assert state.registry is registry
    assert state.daemon_version
    assert state.idle_timeout_s == 42.0


async def test_touch_updates_last_used() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    loaded = await state.load("gemma", "chat")
    before = loaded.last_used
    await asyncio.sleep(0.01)
    state.touch(loaded)
    assert loaded.last_used > before


async def test_schedule_idle_unload_on_embed_is_noop() -> None:
    backend = FakeBackend()
    registry = _registry(nomic=_embed_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    loaded = await state.load("nomic", "embed")
    state.schedule_idle_unload(loaded, keep_alive=None)
    assert loaded.idle_unload_task is None
