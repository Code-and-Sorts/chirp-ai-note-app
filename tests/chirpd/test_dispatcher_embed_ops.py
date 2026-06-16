"""End-to-end socket tests for the embed dispatcher op."""

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
from llm.protocol import (
    EVENT_DONE,
    EVENT_ERROR,
    OP_CHAT,
    OP_EMBED,
    OP_HELLO,
    PROTOCOL_VERSION,
    new_request_id,
    package_version,
)
from llm.registry import Registry, RegistryEntry


@pytest.fixture
def embed_socket_path() -> Iterator[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="cdembed-", dir="/tmp"))
    path = tmp_dir / "s"
    yield path
    try:
        if path.exists():
            path.unlink()
    except OSError:
        # Best-effort cleanup: the temp file may already be removed.
        pass
    try:
        tmp_dir.rmdir()
    except OSError:
        # Best-effort cleanup: the temp directory may be missing or non-empty.
        pass


@pytest.fixture
def embed_registry() -> Registry:
    return Registry(
        schema_version=1,
        default_chat="gemma",
        default_embed="nomic",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
            "nomic": RegistryEntry(hf_repo="mlx-community/nomic-embed", role="embed"),
        },
    )


@pytest_asyncio.fixture
async def embed_server(
    embed_socket_path: Path,
    embed_registry: Registry,
) -> AsyncIterator[tuple[FakeBackend, DaemonState]]:
    backend = FakeBackend()
    state = DaemonState(backend=backend, registry=embed_registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(embed_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not embed_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        yield backend, state
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            # Best-effort teardown: the task is already being cancelled.
            pass


async def _do_hello(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> dict:
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


async def _read_events_until_done(reader: asyncio.StreamReader) -> list[dict]:
    events: list[dict] = []
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
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
            # Best-effort teardown: the writer/peer may already be gone.
            pass


async def test_embed_returns_one_vector_per_input(
    embed_server: tuple[FakeBackend, DaemonState],
    embed_socket_path: Path,
) -> None:
    reader, writer = await _connect(embed_socket_path)
    try:
        await _send(
            writer,
            {
                "id": new_request_id(),
                "op": OP_EMBED,
                "model": "nomic",
                "inputs": ["alpha", "beta", "gamma"],
            },
        )
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    done = events[-1]
    assert done["event"] == EVENT_DONE
    vectors = done["vectors"]
    assert len(vectors) == 3
    assert all(isinstance(v, list) for v in vectors)


async def test_embed_against_embed_model_does_not_schedule_idle_unload(
    embed_server: tuple[FakeBackend, DaemonState],
    embed_socket_path: Path,
) -> None:
    _, state = embed_server
    reader, writer = await _connect(embed_socket_path)
    try:
        await _send(
            writer,
            {
                "id": new_request_id(),
                "op": OP_EMBED,
                "model": "nomic",
                "inputs": ["x"],
            },
        )
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    loaded = state.get("nomic")
    assert loaded is not None
    assert loaded.idle_unload_task is None


async def test_embed_missing_inputs_emits_protocol_malformed(
    embed_server: tuple[FakeBackend, DaemonState],
    embed_socket_path: Path,
) -> None:
    reader, writer = await _connect(embed_socket_path)
    try:
        await _send(
            writer,
            {"id": new_request_id(), "op": OP_EMBED, "model": "nomic"},
        )
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.PROTOCOL_MALFORMED


async def test_embed_parallel_with_chat(
    embed_server: tuple[FakeBackend, DaemonState],
    embed_socket_path: Path,
) -> None:
    backend, _ = embed_server
    backend.chat_tokens = ["a", "b"]
    backend.generation_delay_s = 0.02

    async def _do_embed() -> list[dict]:
        reader, writer = await _connect(embed_socket_path)
        try:
            await _send(
                writer,
                {
                    "id": new_request_id(),
                    "op": OP_EMBED,
                    "model": "nomic",
                    "inputs": ["x"],
                },
            )
            return await _read_events_until_done(reader)
        finally:
            await _close(writer)

    async def _do_chat() -> list[dict]:
        reader, writer = await _connect(embed_socket_path)
        try:
            await _send(
                writer,
                {
                    "id": new_request_id(),
                    "op": OP_CHAT,
                    "model": "gemma",
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {},
                    "keep_alive": None,
                },
            )
            return await _read_events_until_done(reader)
        finally:
            await _close(writer)

    chat_events, embed_events = await asyncio.gather(_do_chat(), _do_embed())
    assert chat_events[-1]["event"] == EVENT_DONE
    assert embed_events[-1]["event"] == EVENT_DONE
