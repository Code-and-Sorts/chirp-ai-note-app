"""Unix-domain NDJSON socket server for chirpd.

``hello`` is handled here (not in :class:`Dispatcher`) because a
version-mismatch must shut the asyncio server down so the daemon can exit.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Final

from chirpd import paths
from chirpd.dispatcher import Dispatcher
from llm import error_codes
from llm.exceptions import LLMMalformedResponse
from llm.protocol import (
    EVENT_ERROR,
    EVENT_READY,
    EVENT_VERSION_MISMATCH,
    MAX_LINE_BYTES,
    OP_HELLO,
    decode_line,
    encode_event,
    validate_request,
)

_logger = logging.getLogger("chirpd.server")

_LINE_READ_LIMIT: Final[int] = MAX_LINE_BYTES + 1


async def serve(socket_path: Path, dispatcher: Dispatcher) -> None:
    """Start a unix-domain NDJSON server at ``socket_path`` and serve forever."""
    _ensure_socket_parent(socket_path)
    _unlink_stale_socket(socket_path)
    shutdown_event = asyncio.Event()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await _handle_connection(reader, writer, dispatcher, shutdown_event)

    server = await _bind_with_restricted_umask(handle, socket_path)
    # The umask above already keeps the bind from ever exposing group/other
    # bits; the explicit chmod pins the mode to exactly 0600 (NFR-S2/SC-9)
    # regardless of the caller's prior umask.
    socket_path.chmod(0o600)
    _logger.info("chirpd listening", extra={"op": "serve"})

    serve_forever_task = asyncio.create_task(server.serve_forever())
    shutdown_waiter = asyncio.create_task(shutdown_event.wait())
    try:
        await asyncio.wait(
            {serve_forever_task, shutdown_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:  # pragma: no cover — cancel path covered via _run
        pass
    finally:
        shutdown_waiter.cancel()
        if not serve_forever_task.done():
            serve_forever_task.cancel()
        for task in (serve_forever_task, shutdown_waiter):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await _close_server(server)
        _unlink_stale_socket(socket_path)


async def _close_server(server: asyncio.base_events.Server) -> None:
    server.close()
    try:
        await server.wait_closed()
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass


def _ensure_socket_parent(socket_path: Path) -> None:
    # A CHIRP_DAEMON_SOCKET override may point at a parent that does not yet
    # exist (or is not owner-only). Create and tighten it to 0700 so the socket
    # is never reachable through a world-traversable directory — matching how
    # ensure_runtime_dirs() treats APP_SUPPORT_DIR for the default path. On the
    # default path the chmod is a harmless idempotent no-op (the dir is already
    # 0700); keeping it unconditional means the override path needs no special
    # casing and a hand-loosened default dir is re-tightened on startup.
    parent = socket_path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=paths.RUNTIME_DIR_MODE)
    try:
        parent.chmod(paths.RUNTIME_DIR_MODE)
    except OSError:  # pragma: no cover — best-effort on an unowned parent
        pass


async def _bind_with_restricted_umask(
    handle: Any, socket_path: Path
) -> asyncio.base_events.Server:
    # Bind under a 0177 umask so the socket inode is created owner-only from the
    # first instant, closing the sub-millisecond window between bind and the
    # follow-up chmod(0600) during which a default umask could leave it
    # group/other-reachable. os.umask is process-global and not thread-safe, but
    # chirpd binds exactly once at startup on the single event loop before any
    # work runs (chirpd/__main__.py: serve() is the only umask mutator), so the
    # save/restore here cannot race another thread's umask.
    previous_umask = os.umask(0o177)
    try:
        return await asyncio.start_unix_server(
            handle, path=str(socket_path), limit=_LINE_READ_LIMIT
        )
    finally:
        os.umask(previous_umask)


def _unlink_stale_socket(socket_path: Path) -> None:
    # Guard against accidentally deleting a non-socket file at the configured
    # path — CHIRP_DAEMON_SOCKET can point anywhere, so refuse to unlink
    # anything that isn't actually a unix socket.
    try:
        if socket_path.is_socket():
            socket_path.unlink()
    except FileNotFoundError:
        pass


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    dispatcher: Dispatcher,
    shutdown_event: asyncio.Event,
) -> None:
    try:
        envelope = await _read_envelope(reader, writer)
        if envelope is None:
            return
        if envelope.get("op") != OP_HELLO:
            await _write_event(
                writer,
                {
                    "id": envelope.get("id"),
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "first message on a connection must be 'hello'",
                    "details": {"op": envelope.get("op")},
                },
            )
            return

        if not await _handle_hello(envelope, writer, dispatcher, shutdown_event):
            return

        next_envelope = await _read_envelope(reader, writer)
        if next_envelope is None:
            return
        if next_envelope.get("op") == OP_HELLO:
            await _write_event(
                writer,
                {
                    "id": next_envelope.get("id"),
                    "event": EVENT_ERROR,
                    "code": error_codes.PROTOCOL_MALFORMED,
                    "message": "'hello' may only be sent as the first message",
                    "details": {"op": OP_HELLO},
                },
            )
            return
        await dispatcher.dispatch(next_envelope, writer)
    except ConnectionResetError:  # pragma: no cover — peer-abort defensive branch
        _logger.info("connection reset by peer")
    except Exception as exc:
        _logger.exception(
            "unhandled connection error",
            extra={"err_type": type(exc).__name__},
        )
    finally:
        await _safe_close(writer)


async def _read_envelope(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> dict[str, Any] | None:
    try:
        line = await reader.readuntil(b"\n")
    except asyncio.IncompleteReadError:
        return None
    except asyncio.LimitOverrunError:
        await _write_event(
            writer,
            {
                "id": None,
                "event": EVENT_ERROR,
                "code": error_codes.PROTOCOL_MALFORMED,
                "message": "line exceeds MAX_LINE_BYTES",
                "details": {"limit": MAX_LINE_BYTES},
            },
        )
        return None

    try:
        envelope = decode_line(line)
        validate_request(envelope)
    except LLMMalformedResponse as err:
        await _write_event(
            writer,
            {
                "id": None,
                "event": EVENT_ERROR,
                "code": err.code,
                "message": err.message,
                "details": err.details,
            },
        )
        return None
    return envelope


async def _handle_hello(
    envelope: dict[str, Any],
    writer: asyncio.StreamWriter,
    dispatcher: Dispatcher,
    shutdown_event: asyncio.Event,
) -> bool:
    request_id = envelope.get("id")
    client_protocol_version = envelope.get("protocol_version")
    daemon_protocol_version = dispatcher.protocol_version
    daemon_version = dispatcher.daemon_version

    # The handshake gates on the wire-format PROTOCOL_VERSION, not the package
    # version: a cosmetic package bump with an unchanged protocol keeps the warm
    # daemon (and its multi-GB model) resident. ``daemon_version`` is still
    # reported for human-facing diagnostics in ``chirp daemon status``.
    if client_protocol_version == daemon_protocol_version:
        await _write_event(
            writer,
            {
                "id": request_id,
                "event": EVENT_READY,
                "daemon_version": daemon_version,
                "protocol_version": daemon_protocol_version,
            },
        )
        return True

    _logger.info(
        "protocol version mismatch; daemon will exit",
        extra={
            "req_id": str(request_id) if request_id is not None else "",
            "op": OP_HELLO,
            "err_code": error_codes.PROTOCOL_VERSION_MISMATCH,
        },
    )
    await _write_event(
        writer,
        {
            "id": request_id,
            "event": EVENT_VERSION_MISMATCH,
            "daemon_version": daemon_version,
            "protocol_version": daemon_protocol_version,
        },
    )
    shutdown_event.set()
    return False


async def _write_event(writer: asyncio.StreamWriter, envelope: dict[str, Any]) -> None:
    try:
        writer.write(encode_event(envelope))
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):  # pragma: no cover
        pass


async def _safe_close(writer: asyncio.StreamWriter) -> None:
    try:
        if not writer.is_closing():
            writer.close()
        await writer.wait_closed()
    except (ConnectionResetError, BrokenPipeError, OSError):  # pragma: no cover
        pass
