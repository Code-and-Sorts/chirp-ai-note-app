"""Op dispatcher; ``hello`` lives in :mod:`chirpd.server` because mismatch must stop the server."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from chirpd.state import DaemonState
from llm import error_codes
from llm.exceptions import LLMError, LLMModelError
from llm.protocol import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_LOADING,
    EVENT_READY,
    EVENT_STATUS,
    OP_HEALTH,
    OP_MODEL_LIST,
    OP_MODEL_LOAD,
    OP_MODEL_STATUS,
    OP_MODEL_UNLOAD,
    encode_event,
    package_version,
)

_logger = logging.getLogger("chirpd.dispatcher")


class Dispatcher:
    """Routes validated request envelopes to per-op coroutines."""

    def __init__(self, state: DaemonState | None = None) -> None:
        self._start_monotonic = time.monotonic()
        self._daemon_version = package_version()
        self._state = state

    @property
    def daemon_version(self) -> str:
        return self._daemon_version

    @property
    def state(self) -> DaemonState | None:
        return self._state

    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_monotonic

    async def dispatch(
        self,
        envelope: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        request_id = envelope.get("id")
        op = envelope.get("op")
        try:
            if op == OP_HEALTH:
                await self._handle_health(request_id, writer)
                return
            if op == OP_MODEL_LIST:
                await self._handle_model_list(request_id, writer)
                return
            if op == OP_MODEL_LOAD:
                await self._handle_model_load(request_id, envelope, writer)
                return
            if op == OP_MODEL_UNLOAD:
                await self._handle_model_unload(request_id, envelope, writer)
                return
            if op == OP_MODEL_STATUS:
                await self._handle_model_status(request_id, writer)
                return
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": f"unknown or unsupported op {op!r}",
                    "details": {"op": op},
                },
            )
        except LLMModelError as exc:
            await _write_error_for_typed(writer, request_id, op, exc)
        except Exception as exc:  # noqa: BLE001 — architecture § Exception Construction
            _logger.exception(
                "dispatch failed",
                extra={
                    "req_id": str(request_id) if request_id is not None else "",
                    "op": str(op) if op is not None else "",
                    "err_code": error_codes.MODEL_GENERATION_FAILED,
                    "err_type": type(exc).__name__,
                },
            )
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.MODEL_GENERATION_FAILED,
                    "message": str(exc),
                    "details": {"exception_type": type(exc).__name__},
                },
            )

    async def _handle_health(
        self,
        request_id: Any,
        writer: asyncio.StreamWriter,
    ) -> None:
        await _write_event(
            writer,
            {
                "id": request_id,
                "event": EVENT_READY,
                "status": "ok",
                "uptime_seconds": self.uptime_seconds(),
                "version": self._daemon_version,
            },
        )
        await _write_event(
            writer,
            {"id": request_id, "event": EVENT_DONE},
        )

    async def _handle_model_list(
        self,
        request_id: Any,
        writer: asyncio.StreamWriter,
    ) -> None:
        state = self._require_state()
        await _write_event(
            writer,
            {
                "id": request_id,
                "event": EVENT_STATUS,
                "models": state.list_models(),
            },
        )
        await _write_event(writer, {"id": request_id, "event": EVENT_DONE})

    async def _handle_model_status(
        self,
        request_id: Any,
        writer: asyncio.StreamWriter,
    ) -> None:
        state = self._require_state()
        payload = state.status()
        event_payload: dict[str, Any] = {
            "id": request_id,
            "event": EVENT_STATUS,
        }
        event_payload.update(payload)
        await _write_event(writer, event_payload)
        await _write_event(writer, {"id": request_id, "event": EVENT_DONE})

    async def _handle_model_load(
        self,
        request_id: Any,
        envelope: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        state = self._require_state()
        identifier = envelope.get("model")
        role = envelope.get("role", "chat")
        if not isinstance(identifier, str):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "model.load requires string 'model' field",
                    "details": {"model": identifier},
                },
            )
            return
        if role not in ("chat", "embed"):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": f"model.load 'role' must be 'chat' or 'embed', got {role!r}",
                    "details": {"role": role},
                },
            )
            return

        await _write_event(
            writer,
            {
                "id": request_id,
                "event": EVENT_LOADING,
                "model": identifier,
            },
        )
        loaded = await state.load(identifier, role)
        await _write_event(
            writer,
            {
                "id": request_id,
                "event": EVENT_READY,
                "model": loaded.alias,
                "role": loaded.role,
            },
        )
        await _write_event(writer, {"id": request_id, "event": EVENT_DONE})

    async def _handle_model_unload(
        self,
        request_id: Any,
        envelope: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        state = self._require_state()
        alias = envelope.get("model")
        if not isinstance(alias, str):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "model.unload requires string 'model' field",
                    "details": {"model": alias},
                },
            )
            return
        await state.unload(alias)
        await _write_event(writer, {"id": request_id, "event": EVENT_DONE})

    def _require_state(self) -> DaemonState:
        if self._state is None:
            raise LLMError(
                "dispatcher invoked without a DaemonState",
            )
        return self._state


async def _write_event(writer: asyncio.StreamWriter, envelope: dict[str, Any]) -> None:
    writer.write(encode_event(envelope))
    await writer.drain()


async def _write_error_for_typed(
    writer: asyncio.StreamWriter,
    request_id: Any,
    op: Any,
    exc: LLMModelError,
) -> None:
    code = exc.code or error_codes.MODEL_GENERATION_FAILED
    _logger.info(
        "model op failed",
        extra={
            "req_id": str(request_id) if request_id is not None else "",
            "op": str(op) if op is not None else "",
            "err_code": code,
            "err_type": type(exc).__name__,
        },
    )
    await _write_event(
        writer,
        {
            "id": request_id,
            "event": EVENT_ERROR,
            "code": code,
            "message": exc.message,
            "details": exc.details,
        },
    )
