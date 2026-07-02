"""Tests for :class:`chirpd.state.DaemonState`."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from chirpd.backend import FakeBackend
from chirpd.state import DaemonState
from llm.exceptions import LLMModelCapacityExceeded, LLMModelNotFound
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
        async def load(
            self, repo: str, role: ModelRole, revision: str | None = None
        ) -> Any:
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


async def test_load_sees_alias_added_after_startup() -> None:
    backend = FakeBackend()
    current = _registry()
    state = DaemonState(
        backend=backend,
        registry=current,
        idle_timeout_s=60.0,
        registry_reader=lambda: current,
    )

    with pytest.raises(LLMModelNotFound):
        await state.load("gemma", "chat")

    current = _registry(gemma=_chat_entry())
    loaded = await state.load("gemma", "chat")
    assert loaded.alias == "gemma"


async def test_list_models_reflects_registry_changes_after_startup() -> None:
    backend = FakeBackend()
    current = _registry()
    state = DaemonState(
        backend=backend,
        registry=current,
        idle_timeout_s=60.0,
        registry_reader=lambda: current,
    )

    assert state.list_models() == []

    current = _registry(gemma=_chat_entry())
    aliases = [item["alias"] for item in state.list_models()]
    assert aliases == ["gemma"]


async def test_registry_reader_absent_keeps_in_memory_registry() -> None:
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)

    state.list_models()
    assert state.registry is registry


def _two_alias_registry(default_alias: str) -> Registry:
    return Registry(
        schema_version=1,
        default_chat=default_alias,
        models={
            "gemma": _chat_entry("mlx-community/gemma-4-4b-it-4bit"),
            "other": _chat_entry("mlx-community/other-4bit"),
        },
    )


async def test_resolved_snapshot_survives_mid_request_default_change() -> None:
    # The chat dispatcher resolves once (for the 'loading' event) and then loads.
    # If models.toml flips 'default' between those steps, the load must still use
    # the alias that was announced — one registry snapshot per request.
    backend = FakeBackend()
    current = _two_alias_registry("gemma")
    state = DaemonState(
        backend=backend,
        registry=current,
        idle_timeout_s=60.0,
        registry_reader=lambda: current,
    )

    entry, alias = state.resolve("default", "chat")
    assert alias == "gemma"

    current = _two_alias_registry("other")  # default flips mid-request

    loaded = await state.load("default", "chat", resolved=(entry, alias))
    assert loaded.alias == "gemma"
    assert backend.load_calls[-1] == ("mlx-community/gemma-4-4b-it-4bit", "chat")


async def test_load_without_snapshot_reresolves_current_default() -> None:
    # Without a snapshot (e.g. the embed path), load does its own single read and
    # resolves against the live registry.
    backend = FakeBackend()
    current = _two_alias_registry("gemma")
    state = DaemonState(
        backend=backend,
        registry=current,
        idle_timeout_s=60.0,
        registry_reader=lambda: current,
    )

    current = _two_alias_registry("other")

    loaded = await state.load("default", "chat")
    assert loaded.alias == "other"


def test_register_cancellation_refuses_duplicate_in_flight() -> None:
    backend = FakeBackend()
    state = DaemonState(backend=backend, registry=_registry(), idle_timeout_s=60.0)

    first = asyncio.Event()
    second = asyncio.Event()
    assert state.register_cancellation("r-aaaaaaaaaaaa", first) is True
    assert state.register_cancellation("r-aaaaaaaaaaaa", second) is False
    assert state.get_cancellation("r-aaaaaaaaaaaa") is first


def test_clear_cancellation_is_identity_safe() -> None:
    backend = FakeBackend()
    state = DaemonState(backend=backend, registry=_registry(), idle_timeout_s=60.0)

    original = asyncio.Event()
    rejected_dup = asyncio.Event()
    state.register_cancellation("r-bbbbbbbbbbbb", original)

    # A rejected duplicate's finally must not evict the original's live event.
    state.clear_cancellation("r-bbbbbbbbbbbb", rejected_dup)
    assert state.get_cancellation("r-bbbbbbbbbbbb") is original

    state.clear_cancellation("r-bbbbbbbbbbbb", original)
    assert state.get_cancellation("r-bbbbbbbbbbbb") is None


async def test_resident_cap_evicts_lru_non_in_flight_model() -> None:
    backend = FakeBackend()
    registry = _registry(
        gemma=_chat_entry("mlx-community/gemma"),
        other=_chat_entry("mlx-community/other"),
    )
    state = DaemonState(
        backend=backend,
        registry=registry,
        idle_timeout_s=60.0,
        max_resident_chat=1,
    )

    first = await state.load("gemma", "chat")
    first.last_used = datetime.now(UTC) - timedelta(seconds=10)
    await state.load("other", "chat")

    assert state.get("gemma") is None, "LRU chat model should be evicted under cap"
    assert state.get("other") is not None
    assert backend.unload_calls, "eviction must unload the evicted model"


async def test_concurrent_distinct_alias_loads_respect_cap() -> None:
    # H1 regression: without a state-wide load gate, two concurrent distinct-alias
    # loads both see resident < cap mid-load and end up resident together (OOM).
    backend = FakeBackend(load_delay_s=0.05)
    registry = _registry(
        gemma=_chat_entry("mlx-community/gemma"),
        other=_chat_entry("mlx-community/other"),
    )
    state = DaemonState(
        backend=backend,
        registry=registry,
        idle_timeout_s=60.0,
        max_resident_chat=1,
    )

    await asyncio.gather(
        state.load("gemma", "chat"),
        state.load("other", "chat"),
    )

    resident_chat = [
        alias for alias in ("gemma", "other") if state.get(alias) is not None
    ]
    assert len(resident_chat) == 1, (
        f"cap=1 must hold under concurrent distinct-alias loads; "
        f"resident chat models: {resident_chat}"
    )


async def test_resident_cap_never_evicts_in_flight_model() -> None:
    backend = FakeBackend()
    registry = _registry(
        gemma=_chat_entry("mlx-community/gemma"),
        other=_chat_entry("mlx-community/other"),
    )
    state = DaemonState(
        backend=backend,
        registry=registry,
        idle_timeout_s=60.0,
        max_resident_chat=1,
    )

    in_flight = await state.load("gemma", "chat")
    async with in_flight.lock:
        with pytest.raises(LLMModelCapacityExceeded):
            await state.load("other", "chat")
    assert state.get("gemma") is not None
    assert state.get("other") is None


async def test_resident_cap_is_per_role_embed_not_evicted_by_chat() -> None:
    backend = FakeBackend()
    registry = _registry(
        gemma=_chat_entry("mlx-community/gemma"),
        other=_chat_entry("mlx-community/other"),
        nomic=_embed_entry("mlx-community/nomic"),
    )
    state = DaemonState(
        backend=backend,
        registry=registry,
        idle_timeout_s=60.0,
        max_resident_chat=1,
        max_resident_embed=1,
    )

    await state.load("nomic", "embed")
    await state.load("gemma", "chat")
    await state.load("other", "chat")

    assert state.get("nomic") is not None, "embed pin survives a chat eviction"
    assert state.get("other") is not None


async def test_embed_model_not_idle_unloaded_on_timer() -> None:
    backend = FakeBackend()
    registry = _registry(nomic=_embed_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=0.05)

    await state.load("nomic", "embed")
    await asyncio.sleep(0.15)
    assert state.get("nomic") is not None, "embed pin: no plain idle-timer unload"


async def test_status_reports_resident_counts_caps_and_eviction() -> None:
    backend = FakeBackend()
    registry = _registry(
        gemma=_chat_entry("mlx-community/gemma"),
        other=_chat_entry("mlx-community/other"),
    )
    state = DaemonState(
        backend=backend,
        registry=registry,
        idle_timeout_s=60.0,
        max_resident_chat=1,
    )

    await state.load("gemma", "chat")
    await state.load("other", "chat")

    status = state.status()
    assert status["resident_caps"]["chat"] == 1
    assert status["resident_counts"]["chat"] == 1
    assert status["last_eviction"] is not None
    assert status["last_eviction"]["alias"] == "gemma"
    assert status["last_eviction"]["reason"] == "resident_cap"


async def test_delayed_unload_skips_and_reschedules_while_lock_held() -> None:
    # AC-6: a _delayed_unload whose sleep elapses mid-generation (lock held) must
    # reschedule, not unload the model out from under the request.
    backend = FakeBackend()
    registry = _registry(gemma=_chat_entry())
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=0.05)

    loaded = await state.load("gemma", "chat")

    async with loaded.lock:
        await asyncio.sleep(0.2)
        assert state.get("gemma") is not None, "must not unload an in-flight model"
        assert backend.unload_calls == []
    assert state.get("gemma") is not None
