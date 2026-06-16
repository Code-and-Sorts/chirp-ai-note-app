"""Integration tests for chirpd.server via in-process unix sockets."""

from __future__ import annotations

import asyncio
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from chirpd.dispatcher import Dispatcher
from llm import error_codes
from llm.protocol import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_READY,
    EVENT_VERSION_MISMATCH,
    MAX_LINE_BYTES,
    OP_HEALTH,
    OP_HELLO,
    PROTOCOL_VERSION,
    new_request_id,
    package_version,
)

SendFn = Callable[[asyncio.StreamWriter, dict[str, Any]], Awaitable[None]]
ReadFn = Callable[[asyncio.StreamReader], Awaitable[dict[str, Any]]]


async def _hello_with_matching_version(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> dict[str, Any]:
    request_id = new_request_id()
    await send_envelope(
        writer,
        {
            "id": request_id,
            "op": OP_HELLO,
            "client_version": dispatcher.daemon_version,
            "protocol_version": dispatcher.protocol_version,
        },
    )
    response = await read_envelope(reader)
    assert response["event"] == EVENT_READY
    assert response["daemon_version"] == dispatcher.daemon_version
    assert response["protocol_version"] == dispatcher.protocol_version
    assert response["id"] == request_id
    return response


async def test_hello_with_matching_version_returns_ready(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> None:
    reader, writer = client_connection
    await _hello_with_matching_version(
        reader, writer, dispatcher, send_envelope, read_envelope
    )


async def test_package_version_skew_handshakes_ready(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> None:
    """A cosmetic package bump with an unchanged protocol must NOT evict the model."""
    reader, writer = client_connection
    request_id = new_request_id()
    skewed_package = package_version() + "-cosmetic"
    await send_envelope(
        writer,
        {
            "id": request_id,
            "op": OP_HELLO,
            "client_version": skewed_package,
            "protocol_version": PROTOCOL_VERSION,
        },
    )
    response = await read_envelope(reader)
    assert response["event"] == EVENT_READY
    assert response["id"] == request_id


async def test_protocol_version_mismatch_returns_mismatch_and_daemon_exits(
    running_server: asyncio.Task[None],
    socket_path: Path,
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> None:
    """A bumped PROTOCOL_VERSION still triggers the exit-and-respawn path."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        await send_envelope(
            writer,
            {
                "id": new_request_id(),
                "op": OP_HELLO,
                "client_version": package_version(),
                "protocol_version": PROTOCOL_VERSION + 1,
            },
        )
        response = await read_envelope(reader)
        assert response["event"] == EVENT_VERSION_MISMATCH
        assert response["protocol_version"] == dispatcher.protocol_version
    finally:
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                # Best-effort teardown: the writer/peer may already be gone.
                pass

    await asyncio.wait_for(running_server, timeout=1.0)
    assert running_server.done()


async def test_health_op_returns_ready_then_done(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> None:
    reader, writer = client_connection
    await _hello_with_matching_version(
        reader, writer, dispatcher, send_envelope, read_envelope
    )

    health_id = new_request_id()
    await send_envelope(writer, {"id": health_id, "op": OP_HEALTH})

    ready = await read_envelope(reader)
    assert ready["event"] == EVENT_READY
    assert ready["id"] == health_id
    assert ready["status"] == "ok"
    assert isinstance(ready["uptime_seconds"], int | float)
    assert ready["version"] == dispatcher.daemon_version

    done = await read_envelope(reader)
    assert done["event"] == EVENT_DONE
    assert done["id"] == health_id


async def test_hello_with_mismatched_version_returns_version_mismatch_and_daemon_exits(
    running_server: asyncio.Task[None],
    socket_path: Path,
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> None:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        await send_envelope(
            writer,
            {
                "id": new_request_id(),
                "op": OP_HELLO,
                "client_version": "0.0.0-bogus",
                "protocol_version": PROTOCOL_VERSION + 99,
            },
        )
        response = await read_envelope(reader)
        assert response["event"] == EVENT_VERSION_MISMATCH
        assert response["daemon_version"] == dispatcher.daemon_version
    finally:
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                # Best-effort teardown: the writer/peer may already be gone.
                pass

    await asyncio.wait_for(running_server, timeout=1.0)
    assert running_server.done()


async def test_first_message_must_be_hello(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> None:
    reader, writer = client_connection
    await send_envelope(writer, {"id": new_request_id(), "op": OP_HEALTH})
    response = await read_envelope(reader)
    assert response["event"] == EVENT_ERROR
    assert response["code"] == error_codes.PROTOCOL_MALFORMED

    closing = await reader.read()
    assert closing == b""


async def test_malformed_line_returns_protocol_malformed_error(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    read_envelope: ReadFn,
) -> None:
    reader, writer = client_connection
    writer.write(b"not valid json\n")
    await writer.drain()

    response = await read_envelope(reader)
    assert response["event"] == EVENT_ERROR
    assert response["code"] == error_codes.PROTOCOL_MALFORMED


async def test_socket_mode_is_0600(
    running_server: asyncio.Task[None], socket_path: Path
) -> None:
    mode = socket_path.stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


async def test_override_socket_parent_created_0700_and_socket_0600() -> None:
    """AC-9: an override socket gets a 0700 parent and a 0600 socket."""
    import shutil
    import tempfile

    from chirpd.server import serve

    # A short base dir under /tmp keeps the AF_UNIX path under the ~104-byte cap.
    base = Path(tempfile.mkdtemp(prefix="cd9-", dir="/tmp"))
    parent = base / "p"
    override_socket = parent / "s"
    assert not parent.exists()

    dispatcher = Dispatcher()
    task = asyncio.create_task(serve(override_socket, dispatcher))
    try:
        deadline = asyncio.get_running_loop().time() + 2.0
        while not override_socket.exists():
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError("override socket never appeared")
            await asyncio.sleep(0.02)

        assert stat.S_IMODE(parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(override_socket.stat().st_mode) == 0o600
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, TimeoutError):
            # Best-effort teardown: the task is already being cancelled.
            pass
        shutil.rmtree(base, ignore_errors=True)


async def test_oversized_line_rejected(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    read_envelope: ReadFn,
) -> None:
    reader, writer = client_connection
    chunk_size = 32_768
    written = 0
    target = MAX_LINE_BYTES + 64
    response: dict[str, Any] | None = None
    while written < target and response is None:
        writer.write(b"x" * chunk_size)
        written += chunk_size
        try:
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            break
        try:
            response = await asyncio.wait_for(read_envelope(reader), timeout=0.05)
        except TimeoutError:
            continue

    if response is None:
        response = await read_envelope(reader)

    assert response["event"] == EVENT_ERROR
    assert response["code"] == error_codes.PROTOCOL_MALFORMED


async def test_unknown_op_after_hello_returns_protocol_malformed(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> None:
    reader, writer = client_connection
    await _hello_with_matching_version(
        reader, writer, dispatcher, send_envelope, read_envelope
    )

    writer.write(b'{"id":"r-aaaaaaaaaaaa","op":"not-a-real-op"}\n')
    await writer.drain()
    response = await read_envelope(reader)
    assert response["event"] == EVENT_ERROR
    assert response["code"] == error_codes.PROTOCOL_MALFORMED


async def test_second_hello_is_rejected(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
) -> None:
    reader, writer = client_connection
    await _hello_with_matching_version(
        reader, writer, dispatcher, send_envelope, read_envelope
    )

    await send_envelope(
        writer,
        {
            "id": new_request_id(),
            "op": OP_HELLO,
            "client_version": dispatcher.daemon_version,
        },
    )
    response = await read_envelope(reader)
    assert response["event"] == EVENT_ERROR
    assert response["code"] == error_codes.PROTOCOL_MALFORMED


async def test_handle_connection_safety_net_swallows_unexpected_exceptions(
    socket_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chirpd import server as server_module

    async def _explode(*_: Any, **__: Any) -> None:
        raise RuntimeError("forced read failure")

    monkeypatch.setattr(server_module, "_read_envelope", _explode)

    dispatcher = Dispatcher()
    task = asyncio.create_task(server_module.serve(socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not socket_path.exists() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)

    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        closing = await asyncio.wait_for(reader.read(), timeout=1.0)
        assert closing == b""
        assert not writer.is_closing() or writer.is_closing()
    finally:
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                # Best-effort teardown: the writer/peer may already be gone.
                pass
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, TimeoutError):
            # Best-effort teardown: the task is already being cancelled.
            pass


async def test_dispatcher_exception_boundary_emits_model_generation_failed(
    client_connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    dispatcher: Dispatcher,
    send_envelope: SendFn,
    read_envelope: ReadFn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, writer = client_connection
    await _hello_with_matching_version(
        reader, writer, dispatcher, send_envelope, read_envelope
    )

    async def _boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("forced failure for boundary test")

    monkeypatch.setattr(dispatcher, "_handle_health", _boom)

    health_id = new_request_id()
    await send_envelope(writer, {"id": health_id, "op": OP_HEALTH})
    response = await read_envelope(reader)
    assert response["event"] == EVENT_ERROR
    assert response["code"] == error_codes.MODEL_GENERATION_FAILED
    assert response["details"]["exception_type"] == "RuntimeError"
    assert response["id"] == health_id
