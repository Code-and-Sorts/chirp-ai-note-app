"""End-to-end socket tests for the model.* dispatcher ops."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio

from chirpd.backend import FakeBackend
from chirpd.dispatcher import Dispatcher
from chirpd.server import serve
from chirpd.state import DaemonState
from llm import error_codes
from llm.exceptions import LLMModelLoadFailed
from llm.protocol import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_LOADING,
    EVENT_READY,
    EVENT_STATUS,
    OP_HELLO,
    OP_MODEL_LIST,
    OP_MODEL_LOAD,
    OP_MODEL_STATUS,
    OP_MODEL_UNLOAD,
    PROTOCOL_VERSION,
    new_request_id,
)
from llm.registry import Registry, RegistryEntry


@pytest.fixture
def model_socket_path() -> Iterator[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="cd-", dir="/tmp"))
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
def model_registry() -> Registry:
    return Registry(
        schema_version=1,
        default_chat="gemma",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
            "nomic": RegistryEntry(hf_repo="mlx-community/nomic-embed", role="embed"),
        },
    )


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest_asyncio.fixture
async def model_server(
    model_socket_path: Path,
    fake_backend: FakeBackend,
    model_registry: Registry,
) -> AsyncIterator[tuple[asyncio.Task[None], Dispatcher, DaemonState]]:
    state = DaemonState(
        backend=fake_backend, registry=model_registry, idle_timeout_s=60.0
    )
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(model_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not model_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        yield task, dispatcher, state
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


async def _do_hello(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> dict:
    from llm.protocol import package_version

    envelope = {
        "id": new_request_id(),
        "op": OP_HELLO,
        "client_version": package_version(),
        "protocol_version": PROTOCOL_VERSION,
    }
    writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    parsed: dict = json.loads(line.decode("utf-8"))
    return parsed


async def _send(writer: asyncio.StreamWriter, envelope: dict) -> None:
    writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
    await writer.drain()


async def _read_until_done(reader: asyncio.StreamReader) -> list[dict]:
    events: list[dict] = []
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        if not line:
            break
        event = json.loads(line.decode("utf-8"))
        events.append(event)
        if event.get("event") in (EVENT_DONE, EVENT_ERROR):
            break
    return events


async def _connect_and_request(socket_path: Path, envelope: dict) -> list[dict]:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        hello = await _do_hello(reader, writer)
        assert hello["event"] == EVENT_READY
        await _send(writer, envelope)
        return await _read_until_done(reader)
    finally:
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass


async def test_model_load_emits_loading_then_ready_then_done(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
) -> None:
    envelope = {
        "id": new_request_id(),
        "op": OP_MODEL_LOAD,
        "model": "gemma",
        "role": "chat",
    }
    events = await _connect_and_request(model_socket_path, envelope)
    event_names = [e["event"] for e in events]
    assert event_names == [EVENT_LOADING, EVENT_READY, EVENT_DONE]
    assert events[0]["model"] == "gemma"
    assert events[1]["role"] == "chat"


async def test_model_load_missing_alias_emits_model_not_found(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
) -> None:
    envelope = {
        "id": new_request_id(),
        "op": OP_MODEL_LOAD,
        "model": "ghost-alias",
        "role": "chat",
    }
    events = await _connect_and_request(model_socket_path, envelope)
    error_event = events[-1]
    assert error_event["event"] == EVENT_ERROR
    assert error_event["code"] == error_codes.MODEL_NOT_FOUND


async def test_model_load_missing_weights_emits_model_load_failed(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
    fake_backend: FakeBackend,
) -> None:
    fake_backend.load_raises = LLMModelLoadFailed(
        "weights not in HF cache for 'mlx-community/gemma-4-4b-it-4bit'; "
        "run `chirp models pull <alias>` to download",
        details={"repo": "mlx-community/gemma-4-4b-it-4bit"},
    )
    envelope = {
        "id": new_request_id(),
        "op": OP_MODEL_LOAD,
        "model": "gemma",
        "role": "chat",
    }
    events = await _connect_and_request(model_socket_path, envelope)
    error_event = events[-1]
    assert error_event["event"] == EVENT_ERROR
    assert error_event["code"] == error_codes.MODEL_LOAD_FAILED
    assert "chirp models pull" in error_event["message"]


async def test_model_unload_idempotent(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
) -> None:
    envelope = {
        "id": new_request_id(),
        "op": OP_MODEL_UNLOAD,
        "model": "never-loaded",
    }
    events = await _connect_and_request(model_socket_path, envelope)
    assert events[-1]["event"] == EVENT_DONE


async def test_model_list_returns_registered_models(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
) -> None:
    envelope = {"id": new_request_id(), "op": OP_MODEL_LIST}
    events = await _connect_and_request(model_socket_path, envelope)
    status_events = [e for e in events if e["event"] == EVENT_STATUS]
    assert status_events
    aliases = {item["alias"] for item in status_events[0]["models"]}
    assert {"gemma", "nomic"}.issubset(aliases)


async def test_model_status_includes_rss_and_uptime(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
) -> None:
    envelope = {"id": new_request_id(), "op": OP_MODEL_STATUS}
    events = await _connect_and_request(model_socket_path, envelope)
    status_event = next(e for e in events if e["event"] == EVENT_STATUS)
    assert "pid" in status_event
    assert "uptime_seconds" in status_event
    assert "daemon_version" in status_event
    assert "rss_bytes" in status_event


async def test_model_load_without_model_field_emits_protocol_malformed(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
) -> None:
    envelope = {"id": new_request_id(), "op": OP_MODEL_LOAD}
    events = await _connect_and_request(model_socket_path, envelope)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.PROTOCOL_MALFORMED


async def test_model_load_with_invalid_role_emits_protocol_malformed(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
) -> None:
    envelope = {
        "id": new_request_id(),
        "op": OP_MODEL_LOAD,
        "model": "gemma",
        "role": "vision",
    }
    events = await _connect_and_request(model_socket_path, envelope)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.PROTOCOL_MALFORMED


async def test_model_unload_without_model_field_emits_protocol_malformed(
    model_server: tuple[asyncio.Task[None], Dispatcher, DaemonState],
    model_socket_path: Path,
) -> None:
    envelope = {"id": new_request_id(), "op": OP_MODEL_UNLOAD}
    events = await _connect_and_request(model_socket_path, envelope)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.PROTOCOL_MALFORMED


async def test_dispatcher_without_state_raises_internal_error() -> None:
    from llm.exceptions import LLMError

    dispatcher = Dispatcher(state=None)
    with pytest.raises(LLMError):
        dispatcher._require_state()
