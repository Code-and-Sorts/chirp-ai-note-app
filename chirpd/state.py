"""Daemon-wide state container: loaded models, locks, idle-unload scheduler."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import psutil

from chirpd.backend import LLMBackend, ModelRole
from llm.exceptions import LLMError, LLMModelCapacityExceeded
from llm.protocol import OP_MODEL_LOAD, OP_MODEL_UNLOAD, package_version
from llm.registry import Registry, RegistryEntry, resolve_alias

_logger = logging.getLogger("chirpd.state")

DEFAULT_IDLE_TIMEOUT_S: float = 300.0
# Generous per-role resident cap: a typical session holds one chat model and
# one pinned embed model. A new load that would exceed a role's cap evicts the
# LRU non-in-flight model of that role; if every resident model of that role is
# locked (in-flight), the load fails with a typed, actionable error rather than
# risking OOM.
DEFAULT_MAX_RESIDENT_CHAT: int = 1
DEFAULT_MAX_RESIDENT_EMBED: int = 1


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
        *,
        registry_reader: Callable[[], Registry] | None = None,
        max_resident_chat: int = DEFAULT_MAX_RESIDENT_CHAT,
        max_resident_embed: int = DEFAULT_MAX_RESIDENT_EMBED,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._registry_reader = registry_reader
        self._idle_timeout_s = idle_timeout_s
        self._max_resident: dict[ModelRole, int] = {
            "chat": max_resident_chat,
            "embed": max_resident_embed,
        }
        self._models: dict[str, LoadedModel] = {}
        self._registry_locks: dict[str, asyncio.Lock] = {}
        # Serializes cap-check + eviction + the multi-GB backend load across
        # DISTINCT aliases so two concurrent loads can't both pass a cap check and
        # then both load (the OOM the cap exists to prevent). Loading two large
        # models at once is itself the memory risk, so serializing the load — not
        # just the check — is the correct, deadlock-free choice (acquired only
        # AFTER the per-alias lock; eviction takes the victim's per-alias lock,
        # never the gate, so there is no lock cycle).
        self._load_gate = asyncio.Lock()
        self._cancellation_events: dict[str, asyncio.Event] = {}
        self._start_monotonic = time.monotonic()
        self._daemon_version = package_version()
        self._proc = psutil.Process(os.getpid())
        self._last_request_at: datetime | None = None
        self._last_eviction: dict[str, Any] | None = None

    @property
    def registry(self) -> Registry:
        return self._registry

    def _refresh_registry(self) -> Registry:
        """Re-read ``models.toml`` so aliases added since startup are visible.

        Per-op reads keep the daemon honest without a file watcher (hot-reload
        is out of scope). Tests that pass no reader keep their in-memory
        registry untouched.
        """
        if self._registry_reader is not None:
            self._registry = self._registry_reader()
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

    def mark_request(self) -> None:
        """Stamp the wall-clock time of the latest user-facing inference op.

        ``chirp daemon status`` reports this so a user can tell "never reached"
        (``None``) from "served a request a while ago". Only the work ops (chat,
        embed) call this — ``health`` / ``model.status`` must not, or the status
        command would always report itself as the last request.
        """
        self._last_request_at = datetime.now(UTC)

    def get(self, alias: str) -> LoadedModel | None:
        return self._models.get(alias)

    def resolve(self, identifier: str, role: ModelRole) -> tuple[RegistryEntry, str]:
        """Re-read the registry once and resolve ``identifier`` to (entry, alias).

        A request that both announces a model (``loading`` event) and loads it
        must resolve once and reuse the result. Resolving twice would read the
        registry twice, so a ``models.toml`` edit landing between the two reads
        could make the announced alias diverge from the one actually generated.
        """
        registry = self._refresh_registry()
        entry = resolve_alias(registry, identifier, role)
        alias = (
            identifier
            if identifier != "default"
            else _alias_for_default(registry, role)
        )
        return entry, alias

    def resolve_canonical_alias(self, identifier: str, role: ModelRole) -> str:
        return self.resolve(identifier, role)[1]

    async def load(
        self,
        identifier: str,
        role: ModelRole = "chat",
        *,
        resolved: tuple[RegistryEntry, str] | None = None,
    ) -> LoadedModel:
        entry, alias = (
            resolved if resolved is not None else self.resolve(identifier, role)
        )

        lock = self._registry_locks.setdefault(alias, asyncio.Lock())
        async with lock:
            existing = self._models.get(alias)
            if existing is not None:
                existing.last_used = datetime.now(UTC)
                if existing.role == "chat":
                    self.schedule_idle_unload(existing, keep_alive=None)
                return existing

            # Hold the state-wide gate across cap-check → eviction → backend load
            # → insertion so a concurrent distinct-alias loader observes this load
            # (its eviction or its inserted model) and cannot independently pass
            # the same cap check. Re-check residency under the gate in case it was
            # filled while we waited.
            async with self._load_gate:
                existing = self._models.get(alias)
                if existing is not None:
                    existing.last_used = datetime.now(UTC)
                    if existing.role == "chat":
                        self.schedule_idle_unload(existing, keep_alive=None)
                    return existing

                await self._enforce_resident_cap(entry.role, incoming_alias=alias)
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

    async def _enforce_resident_cap(
        self, role: ModelRole, *, incoming_alias: str
    ) -> None:
        """Evict the LRU non-in-flight model of ``role`` if a load would exceed the cap.

        Deliberate pressure/cap action — never a plain idle timer, so the embed
        pin (FR6) is respected: an embed model is only evicted to make room, and
        only when it is not in flight. A model holding its ``LoadedModel.lock``
        (mid-generation / mid-embed) is never evicted. When every resident model
        of ``role`` is locked, the load fails with a typed, actionable error.
        """
        cap = self._max_resident.get(role)
        if cap is None or cap <= 0:
            return
        resident = [
            model
            for alias, model in self._models.items()
            if model.role == role and alias != incoming_alias
        ]
        if len(resident) < cap:
            return

        evictable = sorted(
            (model for model in resident if not model.lock.locked()),
            key=lambda model: model.last_used,
        )
        overflow = len(resident) - cap + 1
        if len(evictable) < overflow:
            raise LLMModelCapacityExceeded(
                f"cannot load another {role!r} model: the resident set is full "
                f"({len(resident)}/{cap}) and every resident {role!r} model is "
                "in flight; retry after an in-flight request completes",
                details={
                    "role": role,
                    "resident": len(resident),
                    "cap": cap,
                    "in_flight": len(resident) - len(evictable),
                },
            )

        for victim in evictable[:overflow]:
            self._last_eviction = {
                "alias": victim.alias,
                "role": victim.role,
                "reason": "resident_cap",
                "to_load": incoming_alias,
                "at": datetime.now(UTC).isoformat(),
            }
            _logger.info(
                "evicting model under resident cap",
                extra={
                    "model": victim.alias,
                    "op": OP_MODEL_UNLOAD,
                    "reason": "resident_cap",
                },
            )
            await self.unload(victim.alias)

    def list_models(self) -> list[dict[str, Any]]:
        registry = self._refresh_registry()
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        now = datetime.now(UTC)
        for alias, entry in registry.models.items():
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
        resident_counts = {"chat": 0, "embed": 0}
        for model in self._models.values():
            resident_counts[model.role] = resident_counts.get(model.role, 0) + 1
        return {
            "pid": self._proc.pid,
            "uptime_seconds": self.uptime_seconds(),
            "daemon_version": self._daemon_version,
            "rss_bytes": rss_bytes,
            "idle_timeout_seconds": self._idle_timeout_s,
            "last_request_at": (
                self._last_request_at.isoformat()
                if self._last_request_at is not None
                else None
            ),
            "resident_counts": resident_counts,
            "resident_caps": dict(self._max_resident),
            "last_eviction": self._last_eviction,
            "models": loaded_payload,
        }

    def touch(self, model: LoadedModel) -> None:
        model.last_used = datetime.now(UTC)

    def register_cancellation(self, request_id: str, event: asyncio.Event) -> bool:
        """Register ``event`` under ``request_id`` unless one is already in flight.

        Returns ``True`` when the slot was free and the event was stored,
        ``False`` when ``request_id`` already maps to a live event. Refusing the
        overwrite keeps a duplicate/reused id from clobbering the prior
        request's ``should_stop`` (which would leave it uncancellable and let a
        cancel cross-fire onto the wrong request).
        """
        if request_id in self._cancellation_events:
            return False
        self._cancellation_events[request_id] = event
        return True

    def clear_cancellation(self, request_id: str, event: asyncio.Event) -> None:
        """Pop ``request_id`` only if it still maps to ``event``.

        Identity-guarded so a rejected duplicate's ``finally`` (or a same-id
        request that registered after this one cleared) cannot evict another
        request's live cancellation event.
        """
        if self._cancellation_events.get(request_id) is event:
            del self._cancellation_events[request_id]

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
        # A generation longer than the idle timeout holds model.lock past the
        # sleep. Never unload a model out from under an in-flight request:
        # reschedule instead so the model stays resident and the dispatcher's
        # post-completion reschedule then governs the real idle countdown.
        if model.lock.locked():
            self.schedule_idle_unload(model, keep_alive=None)
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
