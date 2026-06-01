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
    EVENT_DELTA,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_LOADING,
    EVENT_READY,
    EVENT_STATUS,
    OP_CANCEL,
    OP_CHAT,
    OP_EMBED,
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
            if op == OP_CHAT:
                await self._handle_chat(request_id, envelope, writer)
                return
            if op == OP_EMBED:
                await self._handle_embed(request_id, envelope, writer)
                return
            if op == OP_CANCEL:
                await self._handle_cancel(request_id, envelope, writer)
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

    async def _handle_chat(
        self,
        request_id: Any,
        envelope: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        state = self._require_state()
        identifier = envelope.get("model", "default")
        messages = envelope.get("messages")
        options = envelope.get("options") or {}
        keep_alive = envelope.get("keep_alive")

        if not isinstance(identifier, str):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "chat requires string 'model' field",
                    "details": {"model": identifier},
                },
            )
            return
        if not isinstance(messages, list) or not messages:
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "chat requires non-empty 'messages' list",
                    "details": {"messages": messages},
                },
            )
            return
        if not isinstance(options, dict):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "chat 'options' must be an object",
                    "details": {"options": options},
                },
            )
            return
        if not isinstance(request_id, str):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "chat requires a string request id",
                    "details": {"id": request_id},
                },
            )
            return

        entry, alias = state.resolve(identifier, "chat")
        already_loaded = state.get(alias) is not None
        if not already_loaded:
            await _write_event(
                writer,
                {"id": request_id, "event": EVENT_LOADING, "model": alias},
            )

        loaded = await state.load(identifier, "chat", resolved=(entry, alias))
        should_stop = asyncio.Event()
        state.register_cancellation(request_id, should_stop)
        start_ns = time.perf_counter_ns()
        usage_out: dict[str, int] = {"prompt_tokens": 0}
        completion_tokens = 0
        connection_closed = False

        try:
            async with loaded.lock:
                try:
                    async for token in state.backend.stream_generate(
                        loaded.handle, messages, options, should_stop, usage_out
                    ):
                        if should_stop.is_set():
                            break
                        completion_tokens += 1
                        try:
                            await _write_event(
                                writer,
                                {
                                    "id": request_id,
                                    "event": EVENT_DELTA,
                                    "text": token,
                                },
                            )
                        except (BrokenPipeError, ConnectionResetError):
                            connection_closed = True
                            should_stop.set()
                            break
                except LLMModelError as exc:
                    await _write_error_for_typed(writer, request_id, OP_CHAT, exc)
                    return
                except Exception as exc:  # noqa: BLE001
                    _logger.exception(
                        "chat generation failed",
                        extra={
                            "req_id": request_id,
                            "op": OP_CHAT,
                            "model": alias,
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
                    return

            duration_ms = max(int((time.perf_counter_ns() - start_ns) / 1_000_000), 0)
            state.touch(loaded)

            if connection_closed:
                return

            if should_stop.is_set():
                await _write_event(
                    writer,
                    {
                        "id": request_id,
                        "event": EVENT_ERROR,
                        "code": error_codes.MODEL_CANCELLED,
                        "message": "cancelled by client",
                        "details": {},
                    },
                )
                _logger.info(
                    "chat cancelled",
                    extra={
                        "req_id": request_id,
                        "op": OP_CHAT,
                        "model": alias,
                        "duration_ms": duration_ms,
                        "tokens": completion_tokens,
                    },
                )
                return

            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_DONE,
                    "usage": {
                        "prompt_tokens": usage_out.get("prompt_tokens", 0),
                        "completion_tokens": completion_tokens,
                        "ms": duration_ms,
                    },
                },
            )
            _logger.info(
                "chat completed",
                extra={
                    "req_id": request_id,
                    "op": OP_CHAT,
                    "model": alias,
                    "duration_ms": duration_ms,
                    "tokens": completion_tokens,
                },
            )
        finally:
            state.clear_cancellation(request_id)
            if not connection_closed:
                state.schedule_idle_unload(
                    loaded, keep_alive=_coerce_keep_alive(keep_alive)
                )

    async def _handle_embed(
        self,
        request_id: Any,
        envelope: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        state = self._require_state()
        identifier = envelope.get("model", "default")
        inputs = envelope.get("inputs")

        if not isinstance(identifier, str):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "embed requires string 'model' field",
                    "details": {"model": identifier},
                },
            )
            return
        if (
            not isinstance(inputs, list)
            or not inputs
            or not all(isinstance(x, str) for x in inputs)
        ):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "embed requires non-empty 'inputs' list of strings",
                    "details": {"inputs_type": type(inputs).__name__},
                },
            )
            return

        loaded = await state.load(identifier, "embed")
        async with loaded.lock:
            try:
                vectors = await state.backend.embed(loaded.handle, inputs)
            except LLMModelError as exc:
                await _write_error_for_typed(writer, request_id, OP_EMBED, exc)
                return
            except Exception as exc:  # noqa: BLE001
                _logger.exception(
                    "embed failed",
                    extra={
                        "req_id": str(request_id) if request_id is not None else "",
                        "op": OP_EMBED,
                        "model": loaded.alias,
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
                return

        state.touch(loaded)
        await _write_event(
            writer,
            {
                "id": request_id,
                "event": EVENT_DONE,
                "vectors": vectors,
            },
        )

    async def _handle_cancel(
        self,
        request_id: Any,
        envelope: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        state = self._require_state()
        target_id = envelope.get("target_id")
        if not isinstance(target_id, str):
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "cancel requires string 'target_id' field",
                    "details": {"target_id": target_id},
                },
            )
            return

        event = state.get_cancellation(target_id)
        if event is None:
            await _write_event(
                writer,
                {
                    "id": request_id,
                    "event": EVENT_ERROR,
                    "code": error_codes.MODEL_NOT_FOUND,
                    "message": f"no in-flight request with id {target_id!r}",
                    "details": {"target_id": target_id},
                },
            )
            return

        event.set()
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


def _coerce_keep_alive(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


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
