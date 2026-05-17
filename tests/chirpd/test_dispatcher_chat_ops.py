"""End-to-end socket tests for the chat / cancel dispatcher ops."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio

from chirpd.backend import FakeBackend
from chirpd.dispatcher import Dispatcher
from chirpd.server import serve
from chirpd.state import DaemonState
from llm import error_codes
from llm.protocol import (
    EVENT_DELTA,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_LOADING,
    OP_CANCEL,
    OP_CHAT,
    OP_HELLO,
    new_request_id,
    package_version,
)
from llm.registry import Registry, RegistryEntry


@pytest.fixture
def chat_socket_path() -> Iterator[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="cdchat-", dir="/tmp"))
    path = tmp_dir / "s"
    yield path
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


@pytest.fixture
def chat_registry() -> Registry:
    return Registry(
        schema_version=1,
        default_chat="gemma",
        default_embed="nomic",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
            "phi": RegistryEntry(hf_repo="mlx-community/phi-4-mini-4bit", role="chat"),
            "nomic": RegistryEntry(hf_repo="mlx-community/nomic-embed", role="embed"),
        },
    )


@pytest_asyncio.fixture
async def chat_server(
    chat_socket_path: Path,
    chat_registry: Registry,
) -> AsyncIterator[tuple[FakeBackend, DaemonState, Dispatcher]]:
    backend = FakeBackend(chat_tokens=["hello", " ", "world"])
    state = DaemonState(backend=backend, registry=chat_registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(chat_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not chat_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        yield backend, state, dispatcher
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


async def _do_hello(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> dict:
    envelope = {
        "id": new_request_id(),
        "op": OP_HELLO,
        "client_version": package_version(),
    }
    writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    parsed: dict = json.loads(line.decode("utf-8"))
    return parsed


async def _send(writer: asyncio.StreamWriter, envelope: dict) -> None:
    writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
    await writer.drain()


async def _read_events_until_done(
    reader: asyncio.StreamReader, timeout: float = 5.0
) -> list[dict]:
    events: list[dict] = []
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            break
        event = json.loads(line.decode("utf-8"))
        events.append(event)
        if event.get("event") in (EVENT_DONE, EVENT_ERROR):
            break
    return events


async def _connect(
    socket_path: Path,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    hello = await _do_hello(reader, writer)
    assert hello["event"] == "ready"
    return reader, writer


async def _close(writer: asyncio.StreamWriter) -> None:
    if not writer.is_closing():
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass


def _chat_envelope(
    request_id: str | None = None,
    model: str = "gemma",
    messages: list[dict] | None = None,
    options: dict | None = None,
    keep_alive: int | None = None,
) -> dict:
    return {
        "id": request_id or new_request_id(),
        "op": OP_CHAT,
        "model": model,
        "messages": messages or [{"role": "user", "content": "hi"}],
        "options": options or {},
        "keep_alive": keep_alive,
    }


async def test_chat_streams_deltas_then_done(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = ["a", "b", "c"]
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    deltas = [e for e in events if e["event"] == EVENT_DELTA]
    assert [e["text"] for e in deltas] == ["a", "b", "c"]
    assert events[-1]["event"] == EVENT_DONE
    usage = events[-1]["usage"]
    assert usage["completion_tokens"] == 3
    assert usage["prompt_tokens"] > 0
    assert isinstance(usage["ms"], int)


async def test_chat_applies_chat_template_before_generation(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(
            writer,
            _chat_envelope(
                messages=[
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "ping"},
                ]
            ),
        )
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert backend.last_prompt is not None
    assert "system" in backend.last_prompt
    assert "be brief" in backend.last_prompt
    assert "ping" in backend.last_prompt
    assert backend.last_prompt.endswith("<assistant>")


async def test_chat_lazy_loads_model_on_first_request(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    _, state, _ = chat_server
    assert state.get("gemma") is None
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert state.get("gemma") is not None


async def test_chat_emits_loading_when_load_triggered(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[0]["event"] == EVENT_LOADING
    assert events[0]["model"] == "gemma"


async def test_chat_does_not_emit_loading_when_model_already_resident(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    _, state, _ = chat_server
    await state.load("gemma", "chat")
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    event_names = [e["event"] for e in events]
    assert EVENT_LOADING not in event_names


async def test_chat_serializes_concurrent_requests_against_same_model(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = ["x", "y", "z"]
    backend.generation_delay_s = 0.02

    async def _drive() -> list[str]:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(writer, _chat_envelope())
            order: list[str] = []
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not line:
                    break
                event = json.loads(line.decode("utf-8"))
                if event["event"] == EVENT_DELTA:
                    order.append(event["text"])
                if event["event"] in (EVENT_DONE, EVENT_ERROR):
                    break
            return order
        finally:
            await _close(writer)

    a, b = await asyncio.gather(_drive(), _drive())
    assert a == ["x", "y", "z"]
    assert b == ["x", "y", "z"]


async def test_chat_parallel_with_embed_against_different_models(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = ["one", "two", "three"]
    backend.generation_delay_s = 0.02

    async def _do_chat() -> list[dict]:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(writer, _chat_envelope())
            return await _read_events_until_done(reader)
        finally:
            await _close(writer)

    async def _do_embed() -> list[dict]:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(
                writer,
                {
                    "id": new_request_id(),
                    "op": "embed",
                    "model": "nomic",
                    "inputs": ["hello"],
                },
            )
            return await _read_events_until_done(reader)
        finally:
            await _close(writer)

    chat_events, embed_events = await asyncio.gather(_do_chat(), _do_embed())
    assert chat_events[-1]["event"] == EVENT_DONE
    assert embed_events[-1]["event"] == EVENT_DONE
    assert "vectors" in embed_events[-1]


async def test_cancel_halts_generation_within_200ms(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = [f"t{i}" for i in range(100)]
    backend.generation_delay_s = 0.05

    chat_request_id = new_request_id()

    async def _drive_chat() -> tuple[list[dict], float]:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(
                writer,
                _chat_envelope(request_id=chat_request_id),
            )
            events: list[dict] = []
            cancel_observed_at = 0.0
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not line:
                    break
                event = json.loads(line.decode("utf-8"))
                events.append(event)
                if event["event"] == EVENT_ERROR or event["event"] == EVENT_DONE:
                    cancel_observed_at = time.monotonic()
                    break
            return events, cancel_observed_at
        finally:
            await _close(writer)

    async def _drive_cancel() -> float:
        await asyncio.sleep(0.2)
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(
                writer,
                {
                    "id": new_request_id(),
                    "op": OP_CANCEL,
                    "target_id": chat_request_id,
                },
            )
            sent_at = time.monotonic()
            await _read_events_until_done(reader)
            return sent_at
        finally:
            await _close(writer)

    (events, cancel_observed_at), cancel_sent_at = await asyncio.gather(
        _drive_chat(), _drive_cancel()
    )
    elapsed_ms = (cancel_observed_at - cancel_sent_at) * 1000.0
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.MODEL_CANCELLED
    assert elapsed_ms <= 250.0, f"cancel took {elapsed_ms:.1f}ms"


async def test_cancel_unknown_target_emits_error(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(
            writer,
            {
                "id": new_request_id(),
                "op": OP_CANCEL,
                "target_id": "r-ffffffffffff",
            },
        )
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.MODEL_NOT_FOUND


async def test_chat_idle_unload_scheduled_after_completion(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    _, state, _ = chat_server
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    loaded = state.get("gemma")
    assert loaded is not None
    assert loaded.idle_unload_task is not None
    assert not loaded.idle_unload_task.done()


async def test_chat_with_keep_alive_minus_one_pins_model(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    _, state, _ = chat_server
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope(keep_alive=-1))
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    loaded = state.get("gemma")
    assert loaded is not None
    assert loaded.idle_unload_task is None


async def test_chat_propagates_generation_failure(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = ["one", "two", "three", "four"]
    backend.stream_raises = RuntimeError("mlx blew up")
    backend.stream_raises_after = 2
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    deltas = [e for e in events if e["event"] == EVENT_DELTA]
    assert [e["text"] for e in deltas] == ["one", "two"]
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.MODEL_GENERATION_FAILED


async def test_connection_close_during_chat_triggers_cancel(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, state, _ = chat_server
    backend.chat_tokens = [f"t{i}" for i in range(50)]
    backend.generation_delay_s = 0.02

    request_id = new_request_id()
    reader, writer = await _connect(chat_socket_path)
    await _send(writer, _chat_envelope(request_id=request_id))
    delta_count = 0
    while delta_count < 2:
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        event = json.loads(line)
        if event["event"] == EVENT_DELTA:
            delta_count += 1
    await _close(writer)

    deadline = asyncio.get_running_loop().time() + 2.0
    while asyncio.get_running_loop().time() < deadline:
        if state.get_cancellation(request_id) is None:
            break
        await asyncio.sleep(0.02)
    assert state.get_cancellation(request_id) is None


async def test_chat_missing_messages_field_emits_protocol_malformed(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(
            writer,
            {
                "id": new_request_id(),
                "op": OP_CHAT,
                "model": "gemma",
            },
        )
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.PROTOCOL_MALFORMED


async def test_cancel_missing_target_id_emits_protocol_malformed(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(
            writer,
            {"id": new_request_id(), "op": OP_CANCEL},
        )
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.PROTOCOL_MALFORMED
