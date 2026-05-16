"""Unit tests for chirpd.dispatcher.Dispatcher."""

from __future__ import annotations

import asyncio
import json

import pytest

from chirpd import dispatcher as dispatcher_module
from chirpd.dispatcher import Dispatcher
from llm import error_codes
from llm.protocol import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_READY,
    OP_HEALTH,
)


class _RecordingWriter:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.payloads.append(data)

    async def drain(self) -> None:
        pass

    def events(self) -> list[dict[str, object]]:
        decoded: list[dict[str, object]] = []
        for chunk in self.payloads:
            for line in chunk.splitlines():
                if line:
                    decoded.append(json.loads(line))
        return decoded


async def test_dispatch_health_emits_ready_then_done() -> None:
    dispatcher = Dispatcher()
    writer = _RecordingWriter()
    await dispatcher.dispatch(
        {"id": "r-aaaaaaaaaaaa", "op": OP_HEALTH},
        writer,  # type: ignore[arg-type]
    )
    events = writer.events()
    assert events[0]["event"] == EVENT_READY
    assert events[0]["status"] == "ok"
    assert events[0]["version"] == dispatcher.daemon_version
    assert events[0]["id"] == "r-aaaaaaaaaaaa"
    assert events[1]["event"] == EVENT_DONE


async def test_dispatch_unknown_op_emits_protocol_malformed() -> None:
    dispatcher = Dispatcher()
    writer = _RecordingWriter()
    await dispatcher.dispatch(
        {"id": "r-aaaaaaaaaaaa", "op": "totally-bogus"},
        writer,  # type: ignore[arg-type]
    )
    events = writer.events()
    assert len(events) == 1
    assert events[0]["event"] == EVENT_ERROR
    assert events[0]["code"] == error_codes.PROTOCOL_MALFORMED


async def test_dispatch_exception_boundary_converts_to_model_generation_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = Dispatcher()
    writer = _RecordingWriter()

    async def _boom(*_: object, **__: object) -> None:
        raise RuntimeError("forced failure")

    monkeypatch.setattr(dispatcher, "_handle_health", _boom)
    await dispatcher.dispatch(
        {"id": "r-aaaaaaaaaaaa", "op": OP_HEALTH},
        writer,  # type: ignore[arg-type]
    )
    events = writer.events()
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.MODEL_GENERATION_FAILED
    assert events[-1]["details"]["exception_type"] == "RuntimeError"  # type: ignore[index]


async def test_uptime_seconds_is_non_negative() -> None:
    dispatcher = Dispatcher()
    await asyncio.sleep(0.001)
    assert dispatcher.uptime_seconds() >= 0


def test_daemon_version_falls_back_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import PackageNotFoundError

    def _missing(_: str) -> str:
        raise PackageNotFoundError("chirp-notes-ai")

    monkeypatch.setattr(dispatcher_module, "version", _missing)
    assert dispatcher_module._daemon_version() == "0.0.0"
