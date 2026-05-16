"""Async unix-socket client for the chirpd LLM daemon.

The client owns connection lifetime (one request per connection per the
architecture's IPC contract), performs the ``hello`` handshake on every
connect, lazy-spawns ``chirpd`` when no daemon is running, and transparently
respawns once on version mismatch before surfacing the failure.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator, Coroutine
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, TypeVar

from chirpd.paths import SOCKET_PATH
from config.settings import get_daemon_socket_override, get_settings
from llm.error_codes import CODE_TO_EXCEPTION
from llm.exceptions import (
    LLMConnectionLost,
    LLMDaemonSpawnFailed,
    LLMDaemonUnreachable,
    LLMError,
    LLMMalformedResponse,
    LLMVersionMismatch,
)
from llm.protocol import (
    DISTRIBUTION_NAME,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_READY,
    EVENT_VERSION_MISMATCH,
    MAX_LINE_BYTES,
    OP_HEALTH,
    OP_HELLO,
    decode_line,
    encode_request,
    new_request_id,
)

DEFAULT_SOCKET_PATH: Final[Path] = SOCKET_PATH
DEFAULT_SPAWN_TIMEOUT_S: Final[float] = 3.0
VERSION_MISMATCH_RESPAWN_WAIT_S: Final[float] = 1.0

_SPAWN_POLL_INTERVAL_S: Final[float] = 0.05
_LINE_READ_LIMIT: Final[int] = MAX_LINE_BYTES + 1

_logger = logging.getLogger("llm.client")

T = TypeVar("T")


class _VersionMismatchMidStream(Exception):
    """Internal signal: daemon emitted version_mismatch after handshake."""


class _HandshakeVersionMismatch(Exception):
    """Internal signal: daemon's hello returned version_mismatch."""

    def __init__(self, daemon_version: Any) -> None:
        super().__init__("daemon version mismatch on handshake")
        self.daemon_version = daemon_version


def _client_version() -> str:
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:  # pragma: no cover — package always installed in tests
        return "0.0.0"


def resolve_socket_path() -> Path:
    """Return the socket path per env → config → default precedence."""
    override = get_daemon_socket_override()
    if override is not None:
        return override
    try:
        configured = get_settings().llm.daemon_socket
    except Exception:  # noqa: BLE001 — config errors must not block client construction
        configured = None
    if configured is not None:
        return configured
    return DEFAULT_SOCKET_PATH


class LLMClient:
    """High-level async client for the chirpd daemon."""

    def __init__(
        self,
        socket_path: Path | None = None,
        spawn_timeout_s: float = DEFAULT_SPAWN_TIMEOUT_S,
        retry_on_version_mismatch: bool = True,
    ) -> None:
        self.socket_path: Path = socket_path or resolve_socket_path()
        self.spawn_timeout_s = spawn_timeout_s
        self.retry_on_version_mismatch = retry_on_version_mismatch
        self._client_version = _client_version()

    async def health(self) -> dict[str, Any]:
        envelope = {"id": new_request_id(), "op": OP_HEALTH}
        ready_payload: dict[str, Any] | None = None
        async for event in self._request(envelope):
            if event.get("event") == EVENT_READY:
                ready_payload = event
        if ready_payload is None:
            raise LLMMalformedResponse(
                "health op completed without emitting a ready event",
            )
        return ready_payload

    def health_sync(self) -> dict[str, Any]:
        return _run_sync(self.health())

    async def model_list(self) -> list[dict[str, Any]]:
        raise NotImplementedError("model_list lands in story 3.5")

    async def model_load(self, alias: str) -> dict[str, Any]:
        raise NotImplementedError("model_load lands in story 3.5")

    async def model_unload(self, alias: str) -> dict[str, Any]:
        raise NotImplementedError("model_unload lands in story 3.5")

    async def model_status(self, alias: str) -> dict[str, Any]:
        raise NotImplementedError("model_status lands in story 3.5")

    async def chat(self, *args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError("chat lands in story 3.6")

    async def embed(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        raise NotImplementedError("embed lands in story 3.6")

    async def cancel(self, request_id: str) -> dict[str, Any]:
        raise NotImplementedError("cancel lands in story 3.6")

    async def _request(self, envelope: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        attempted_respawn = False
        while True:
            try:
                reader, writer = await self._connect_with_handshake()
            except _HandshakeVersionMismatch as err:
                if attempted_respawn or not self.retry_on_version_mismatch:
                    raise LLMVersionMismatch(
                        "daemon protocol version does not match client",
                        details={
                            "client_version": self._client_version,
                            "daemon_version": err.daemon_version,
                        },
                    ) from err
                attempted_respawn = True
                await self._wait_for_socket_gone(VERSION_MISMATCH_RESPAWN_WAIT_S)
                continue

            try:
                try:
                    async for event in self._run_request_on_connection(
                        reader, writer, envelope
                    ):
                        yield event
                    return
                except _VersionMismatchMidStream as err:
                    if attempted_respawn or not self.retry_on_version_mismatch:
                        raise LLMVersionMismatch(
                            "daemon reported version mismatch after handshake",
                            details={"client_version": self._client_version},
                        ) from err
            finally:
                await _close_writer(writer)

            attempted_respawn = True
            await self._wait_for_socket_gone(VERSION_MISMATCH_RESPAWN_WAIT_S)

    async def _connect_with_handshake(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await self._open_connection()
        try:
            hello_event = await self._do_hello(reader, writer)
        except BaseException:
            await _close_writer(writer)
            raise

        event_name = hello_event.get("event")
        if event_name == EVENT_READY:
            return reader, writer

        await _close_writer(writer)

        if event_name == EVENT_VERSION_MISMATCH:
            raise _HandshakeVersionMismatch(hello_event.get("daemon_version"))

        raise LLMMalformedResponse(
            f"unexpected handshake event {event_name!r}",
            details={"event": event_name},
        )

    async def _open_connection(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            return await asyncio.open_unix_connection(
                str(self.socket_path), limit=_LINE_READ_LIMIT
            )
        except (FileNotFoundError, ConnectionRefusedError):
            pass

        await self._spawn_daemon()
        try:
            return await asyncio.open_unix_connection(
                str(self.socket_path), limit=_LINE_READ_LIMIT
            )
        except (
            ConnectionRefusedError
        ) as err:  # pragma: no cover — spawn-success-then-refusal is racy
            raise LLMDaemonUnreachable(
                f"daemon socket at {self.socket_path} refused connection",
                details={"socket_path": str(self.socket_path)},
            ) from err
        except (
            FileNotFoundError
        ) as err:  # pragma: no cover — spawn polled the socket already
            raise LLMDaemonSpawnFailed(
                f"daemon socket at {self.socket_path} did not appear after spawn",
                details={"socket_path": str(self.socket_path)},
            ) from err

    async def _spawn_daemon(self) -> None:
        try:
            proc = subprocess.Popen(
                ["chirpd"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as err:
            raise LLMDaemonSpawnFailed(
                "chirpd executable not found on PATH",
                details={"error": str(err)},
            ) from err
        except OSError as err:
            raise LLMDaemonSpawnFailed(
                f"failed to spawn chirpd: {err}",
                details={"error": str(err)},
            ) from err

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.spawn_timeout_s
        while loop.time() < deadline:
            try:
                _, writer = await asyncio.open_unix_connection(
                    str(self.socket_path), limit=_LINE_READ_LIMIT
                )
            except (FileNotFoundError, ConnectionRefusedError):
                await asyncio.sleep(_SPAWN_POLL_INTERVAL_S)
                continue
            writer.close()
            try:
                await writer.wait_closed()
            except (
                ConnectionResetError,
                BrokenPipeError,
                OSError,
            ):  # pragma: no cover — cleanup of probe socket
                pass
            return

        _reap_failed_spawn(proc)
        raise LLMDaemonSpawnFailed(
            f"chirpd did not accept connections within {self.spawn_timeout_s}s",
            details={
                "socket_path": str(self.socket_path),
                "spawn_timeout_s": self.spawn_timeout_s,
            },
        )

    async def _wait_for_socket_gone(self, timeout_s: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            try:
                _, writer = await asyncio.open_unix_connection(
                    str(self.socket_path), limit=_LINE_READ_LIMIT
                )
            except (FileNotFoundError, ConnectionRefusedError):
                return
            writer.close()
            try:
                await writer.wait_closed()
            except (
                ConnectionResetError,
                BrokenPipeError,
                OSError,
            ):  # pragma: no cover — cleanup of probe socket
                pass
            await asyncio.sleep(_SPAWN_POLL_INTERVAL_S)
        raise LLMDaemonUnreachable(
            "previous chirpd did not vacate socket within "
            f"{timeout_s}s after version mismatch",
            details={"socket_path": str(self.socket_path), "timeout_s": timeout_s},
        )

    async def _do_hello(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> dict[str, Any]:
        hello_envelope = {
            "id": new_request_id(),
            "op": OP_HELLO,
            "client_version": self._client_version,
        }
        try:
            writer.write(encode_request(hello_envelope))
            await writer.drain()
        except (
            BrokenPipeError,
            ConnectionResetError,
        ) as err:  # pragma: no cover — defensive against immediate peer-RST
            raise LLMConnectionLost(
                f"connection broken while sending hello: {err}",
            ) from err
        return await _read_event(reader)

    async def _run_request_on_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        envelope: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            writer.write(encode_request(envelope))
            await writer.drain()
        except (
            BrokenPipeError,
            ConnectionResetError,
        ) as err:  # pragma: no cover — defensive against immediate peer-RST
            raise LLMConnectionLost(
                f"connection broken while sending request: {err}",
            ) from err

        while True:
            event = await _read_event(reader)
            event_name = event.get("event")
            if event_name == EVENT_ERROR:
                raise _exception_for_error_event(event)
            if event_name == EVENT_VERSION_MISMATCH:
                raise _VersionMismatchMidStream()
            yield event
            if event_name == EVENT_DONE:
                return


async def _read_event(reader: asyncio.StreamReader) -> dict[str, Any]:
    try:
        line = await reader.readuntil(b"\n")
    except asyncio.IncompleteReadError as err:
        raise LLMConnectionLost(
            "daemon closed connection before sending a complete event",
            details={"partial_bytes": len(err.partial)},
        ) from err
    except (
        BrokenPipeError,
        ConnectionResetError,
    ) as err:  # pragma: no cover — IncompleteReadError covers the common close path
        raise LLMConnectionLost(
            f"connection broken while reading event: {err}",
        ) from err
    except (
        asyncio.LimitOverrunError
    ) as err:  # pragma: no cover — exercised via daemon-side cap test
        raise LLMMalformedResponse(
            "event line exceeds MAX_LINE_BYTES",
            details={"limit": MAX_LINE_BYTES},
        ) from err
    return decode_line(line)


def _exception_for_error_event(event: dict[str, Any]) -> LLMError:
    code = event.get("code")
    message = event.get("message") or "unknown error from daemon"
    details = event.get("details") or {}
    exc_cls = CODE_TO_EXCEPTION.get(code) if isinstance(code, str) else None
    if exc_cls is None:
        return LLMMalformedResponse(
            f"daemon emitted error event with unknown code {code!r}",
            details={"code": code, "message": message, "raw_details": details},
        )
    return exc_cls(message, details=details)


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        if not writer.is_closing():
            writer.close()
        await writer.wait_closed()
    except (
        ConnectionResetError,
        BrokenPipeError,
        OSError,
    ):  # pragma: no cover — best-effort teardown
        pass


def _reap_failed_spawn(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        _logger.debug("failed to reap chirpd spawn", exc_info=True)


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise LLMError(
        "sync wrapper cannot be called from a running event loop",
    )
