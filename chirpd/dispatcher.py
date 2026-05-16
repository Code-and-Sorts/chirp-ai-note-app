"""Op dispatcher; ``hello`` lives in :mod:`chirpd.server` because mismatch must stop the server."""

from __future__ import annotations

import asyncio
import logging
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from llm import error_codes
from llm.protocol import (
    DISTRIBUTION_NAME,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_READY,
    OP_HEALTH,
    encode_event,
)

_logger = logging.getLogger("chirpd.dispatcher")


def _daemon_version() -> str:
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "0.0.0"


class Dispatcher:
    """Routes validated request envelopes to per-op coroutines."""

    def __init__(self) -> None:
        self._start_monotonic = time.monotonic()
        self._daemon_version = _daemon_version()

    @property
    def daemon_version(self) -> str:
        return self._daemon_version

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


async def _write_event(writer: asyncio.StreamWriter, envelope: dict[str, Any]) -> None:
    writer.write(encode_event(envelope))
    await writer.drain()
