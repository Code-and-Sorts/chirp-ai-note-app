"""Test fixtures for the chirpd integration suite."""

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

SendFn = Callable[[dict[str, Any]], Awaitable[None]]
ReadFn = Callable[[], Awaitable[dict[str, Any]]]


@pytest.fixture
def socket_path() -> Iterator[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="cd-", dir="/tmp"))
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


@pytest.fixture
def dispatcher() -> Dispatcher:
    return Dispatcher()


@pytest_asyncio.fixture
async def running_server(
    socket_path: Path, dispatcher: Dispatcher
) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(serve(socket_path, dispatcher))
    await _wait_for_socket(socket_path, task)
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


@pytest_asyncio.fixture
async def client_connection(
    running_server: asyncio.Task[None], socket_path: Path
) -> AsyncIterator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        yield reader, writer
    finally:
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                # best-effort teardown
                pass


@pytest.fixture
def send_envelope() -> Callable[
    [asyncio.StreamWriter, dict[str, Any]], Awaitable[None]
]:
    async def _send(writer: asyncio.StreamWriter, envelope: dict[str, Any]) -> None:
        writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
        await writer.drain()

    return _send


@pytest.fixture
def read_envelope() -> Callable[[asyncio.StreamReader], Awaitable[dict[str, Any]]]:
    async def _read(reader: asyncio.StreamReader) -> dict[str, Any]:
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        if not line:
            raise EOFError("connection closed before envelope received")
        parsed = json.loads(line.decode("utf-8"))
        assert isinstance(parsed, dict)
        return parsed

    return _read


@pytest.fixture(autouse=True)
def _isolate_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("CHIRP_DAEMON_SOCKET", raising=False)
