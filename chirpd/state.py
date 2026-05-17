"""Daemon-wide state container: loaded models, locks, idle-unload scheduler."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import psutil

from chirpd.backend import LLMBackend, ModelRole
from llm.exceptions import LLMError
from llm.protocol import OP_MODEL_LOAD, OP_MODEL_UNLOAD, package_version
from llm.registry import Registry, resolve_alias

_logger = logging.getLogger("chirpd.state")

DEFAULT_IDLE_TIMEOUT_S: float = 300.0


@dataclass
class LoadedModel:
    alias: str
    role: ModelRole
    handle: Any
    last_used: datetime
    lock: asyncio.Lock
    loaded_at: datetime
    idle_unload_task: asyncio.Task[None] | None = field(default=None)


class DaemonState:
    """Tracks loaded models and orchestrates lifecycle transitions."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: Registry,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._idle_timeout_s = idle_timeout_s
        self._models: dict[str, LoadedModel] = {}
        self._registry_locks: dict[str, asyncio.Lock] = {}
        self._cancellation_events: dict[str, asyncio.Event] = {}
        self._start_monotonic = time.monotonic()
        self._daemon_version = package_version()
        self._proc = psutil.Process(os.getpid())

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def backend(self) -> LLMBackend:
        return self._backend

    @property
    def daemon_version(self) -> str:
        return self._daemon_version

    @property
    def idle_timeout_s(self) -> float:
        return self._idle_timeout_s

    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_monotonic

    def get(self, alias: str) -> LoadedModel | None:
        return self._models.get(alias)

    def resolve_canonical_alias(self, identifier: str, role: ModelRole) -> str:
        resolve_alias(self._registry, identifier, role)
        if identifier == "default":
            return _alias_for_default(self._registry, role)
        return identifier

    async def load(
        self,
        identifier: str,
        role: ModelRole = "chat",
    ) -> LoadedModel:
        entry = resolve_alias(self._registry, identifier, role)
        alias = (
            identifier
            if identifier != "default"
            else _alias_for_default(self._registry, role)
        )

        lock = self._registry_locks.setdefault(alias, asyncio.Lock())
        async with lock:
            existing = self._models.get(alias)
            if existing is not None:
                existing.last_used = datetime.now(UTC)
                if existing.role == "chat":
                    self.schedule_idle_unload(existing, keep_alive=None)
                return existing

            handle = await self._backend.load(entry.hf_repo, entry.role)
            now = datetime.now(UTC)
            model = LoadedModel(
                alias=alias,
                role=entry.role,
                handle=handle,
                last_used=now,
                loaded_at=now,
                lock=lock,
            )
            self._models[alias] = model
            if model.role == "chat":
                self.schedule_idle_unload(model, keep_alive=None)
            _logger.info(
                "model loaded",
                extra={"model": alias, "op": OP_MODEL_LOAD},
            )
            return model

    async def unload(self, alias: str) -> None:
        model = self._models.get(alias)
        if model is None:
            return
        async with model.lock:
            still_present = self._models.get(alias)
            if still_present is None:  # pragma: no cover — lost-race defensive guard
                return
            await self._cancel_idle_task(still_present)
            try:
                await self._backend.unload(still_present.handle)
            finally:
                self._models.pop(alias, None)
        _logger.info(
            "model unloaded",
            extra={"model": alias, "op": OP_MODEL_UNLOAD},
        )

    def list_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        now = datetime.now(UTC)
        for alias, entry in self._registry.models.items():
            seen.add(alias)
            loaded = self._models.get(alias)
            out.append(
                {
                    "alias": alias,
                    "role": entry.role,
                    "hf_repo": entry.hf_repo,
                    "loaded": loaded is not None,
                    "last_used_seconds_ago": (
                        (now - loaded.last_used).total_seconds()
                        if loaded is not None
                        else None
                    ),
                }
            )
        for alias, loaded in self._models.items():
            if alias in seen:
                continue
            out.append(
                {
                    "alias": alias,
                    "role": loaded.role,
                    "hf_repo": _hf_repo_for_loaded(loaded),
                    "loaded": True,
                    "last_used_seconds_ago": (now - loaded.last_used).total_seconds(),
                }
            )
        return out

    def status(self) -> dict[str, Any]:
        rss_bytes = self._proc.memory_info().rss
        now = datetime.now(UTC)
        loaded_payload: list[dict[str, Any]] = []
        for alias, model in self._models.items():
            idle_elapsed = (now - model.last_used).total_seconds()
            countdown = (
                max(self._idle_timeout_s - idle_elapsed, 0.0)
                if model.role == "chat"
                else None
            )
            loaded_payload.append(
                {
                    "alias": alias,
                    "role": model.role,
                    "loaded_at": model.loaded_at.isoformat(),
                    "last_used": model.last_used.isoformat(),
                    "idle_countdown_seconds": countdown,
                }
            )
        return {
            "pid": self._proc.pid,
            "uptime_seconds": self.uptime_seconds(),
            "daemon_version": self._daemon_version,
            "rss_bytes": rss_bytes,
            "idle_timeout_seconds": self._idle_timeout_s,
            "models": loaded_payload,
        }

    def touch(self, model: LoadedModel) -> None:
        model.last_used = datetime.now(UTC)

    def register_cancellation(self, request_id: str, event: asyncio.Event) -> None:
        self._cancellation_events[request_id] = event

    def clear_cancellation(self, request_id: str) -> None:
        self._cancellation_events.pop(request_id, None)

    def get_cancellation(self, request_id: str) -> asyncio.Event | None:
        return self._cancellation_events.get(request_id)

    def schedule_idle_unload(
        self,
        model: LoadedModel,
        keep_alive: int | None,
    ) -> None:
        if model.role != "chat":
            return

        existing = model.idle_unload_task
        if existing is not None and not existing.done():
            existing.cancel()
        model.idle_unload_task = None

        if keep_alive == -1:
            return

        if keep_alive == 0:
            model.idle_unload_task = asyncio.create_task(self._immediate_unload(model))
            return

        delay = float(keep_alive) if keep_alive is not None else self._idle_timeout_s
        model.idle_unload_task = asyncio.create_task(self._delayed_unload(model, delay))

    async def _delayed_unload(self, model: LoadedModel, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        # Belt-and-braces: reschedules always cancel the prior task first, so a
        # task that survives the sleep should mean its delay elapsed without
        # activity. This guard catches any future scheduling slip.
        now = datetime.now(UTC)
        if (now - model.last_used).total_seconds() + 1e-6 < delay:
            return
        await self.unload(model.alias)

    async def _immediate_unload(self, model: LoadedModel) -> None:
        await self.unload(model.alias)

    async def _cancel_idle_task(self, model: LoadedModel) -> None:
        task = model.idle_unload_task
        model.idle_unload_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _alias_for_default(registry: Registry, role: ModelRole) -> str:
    default = registry.default_chat if role == "chat" else registry.default_embed
    if default is None:
        raise LLMError(f"no default {role!r} model registered")
    return default


def _hf_repo_for_loaded(model: LoadedModel) -> str | None:
    handle = model.handle
    if isinstance(handle, dict):
        repo = handle.get("repo")
        if isinstance(repo, str):
            return repo
    return None
