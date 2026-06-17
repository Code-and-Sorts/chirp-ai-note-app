"""Shared fixtures for the ``llm.client`` test suite."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from chirpd.dispatcher import Dispatcher
from chirpd.server import serve
from llm.protocol import (
    EVENT_DONE,
    EVENT_READY,
    EVENT_VERSION_MISMATCH,
    OP_HELLO,
    encode_event,
)

HandlerFn = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]
FakeDaemonFactory = Callable[..., Awaitable["FakeDaemon"]]


@pytest.fixture
def temp_socket_path() -> Iterator[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="llmc-", dir="/tmp"))
    path = tmp_dir / "s"
    try:
        yield path
    finally:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            # best-effort cleanup
            pass
        try:
            tmp_dir.rmdir()
        except OSError:
            # best-effort cleanup
            pass


@pytest_asyncio.fixture
async def in_process_daemon(
    temp_socket_path: Path,
) -> AsyncIterator[asyncio.Task[None]]:
    dispatcher = Dispatcher()
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    await _wait_for_socket(temp_socket_path, task)
    try:
        yield task
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            # best-effort teardown
            pass


async def _wait_for_socket(
    socket_path: Path, task: asyncio.Task[None], timeout: float = 2.0
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if task.done():
            task.result()
            raise RuntimeError(f"serve() exited before binding socket {socket_path}")
        if socket_path.exists():
            return
        await asyncio.sleep(0.02)
    raise RuntimeError(f"socket {socket_path} did not appear within {timeout}s")


class FakeDaemon:
    """Scripted asyncio.start_unix_server stand-in for the chirpd daemon."""

    def __init__(self, socket_path: Path, handler: HandlerFn) -> None:
        self.socket_path = socket_path
        self._handler = handler
        self._server: asyncio.base_events.Server | None = None
        self.connections_seen = 0

    async def start(self) -> None:
        async def _wrapped(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            self.connections_seen += 1
            try:
                await self._handler(reader, writer)
            finally:
                if not writer.is_closing():
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        # best-effort teardown
                        pass

        self._server = await asyncio.start_unix_server(
            _wrapped, path=str(self.socket_path)
        )

    def close_listener(self) -> None:
        if self._server is not None:
            self._server.close()
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except OSError:
            # best-effort cleanup
            pass

    async def stop(self) -> None:
        self.close_listener()
        if self._server is None:
            return
        try:
            await self._server.wait_closed()
        except Exception:  # noqa: BLE001
            # best-effort teardown
            pass


@pytest_asyncio.fixture
async def fake_daemon_factory(
    temp_socket_path: Path,
) -> AsyncIterator[FakeDaemonFactory]:
    started: list[FakeDaemon] = []

    async def _factory(
        handler: HandlerFn, socket_path: Path | None = None
    ) -> FakeDaemon:
        daemon = FakeDaemon(socket_path or temp_socket_path, handler)
        await daemon.start()
        started.append(daemon)
        return daemon

    try:
        yield _factory
    finally:
        for daemon in started:
            await daemon.stop()


async def read_request(reader: asyncio.StreamReader) -> dict[str, Any]:
    line = await reader.readuntil(b"\n")
    parsed = json.loads(line.decode("utf-8"))
    assert isinstance(parsed, dict)
    return parsed


async def write_event(writer: asyncio.StreamWriter, envelope: dict[str, Any]) -> None:
    writer.write(encode_event(envelope))
    await writer.drain()


def make_ready_then_done_handler(
    daemon_version: str, ready_extra: dict[str, Any] | None = None
) -> HandlerFn:
    """Handler that replies ``ready`` to hello then ``ready+done`` to any op."""

    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        assert hello["op"] == OP_HELLO
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": daemon_version,
            },
        )
        try:
            request = await read_request(reader)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return
        ready_payload: dict[str, Any] = {
            "id": request.get("id"),
            "event": EVENT_READY,
            "status": "ok",
            "uptime_seconds": 1.0,
            "version": daemon_version,
        }
        if ready_extra:
            ready_payload.update(ready_extra)
        await write_event(writer, ready_payload)
        await write_event(writer, {"id": request.get("id"), "event": EVENT_DONE})

    return _handler


def make_version_mismatch_then_exit_handler(
    daemon_version: str,
    on_mismatch: Callable[[], Awaitable[None]] | None = None,
) -> HandlerFn:
    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        assert hello["op"] == OP_HELLO
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_VERSION_MISMATCH,
                "daemon_version": daemon_version,
            },
        )
        if on_mismatch is not None:
            await on_mismatch()

    return _handler
