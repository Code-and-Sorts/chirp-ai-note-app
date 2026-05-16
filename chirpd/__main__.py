"""chirpd entry point — Apple-Silicon check, single-instance, asyncio serve."""

from __future__ import annotations

import asyncio
import logging
import platform
import signal
import sys
from pathlib import Path

from chirpd.dispatcher import Dispatcher
from chirpd.lifecycle import (
    SOCKET_PATH as DEFAULT_SOCKET_PATH,
)
from chirpd.lifecycle import (
    acquire_single_instance_lock,
    ensure_runtime_dirs,
    release_lock,
)
from chirpd.logging_setup import configure_logging
from chirpd.server import serve
from config.settings import get_daemon_socket_override

_REQUIRED_MACHINE = "arm64"


def _resolve_socket_path() -> Path:
    return get_daemon_socket_override() or DEFAULT_SOCKET_PATH


def main() -> int:
    machine = platform.machine()
    if machine != _REQUIRED_MACHINE:
        print(
            f"chirpd requires Apple Silicon (arm64); detected: {machine}",
            file=sys.stderr,
        )
        return 2

    configure_logging()
    ensure_runtime_dirs()

    lock_handle = acquire_single_instance_lock()
    if lock_handle is None:
        return 0

    socket_path = _resolve_socket_path()
    dispatcher = Dispatcher()
    logger = logging.getLogger("chirpd")
    logger.info("chirpd starting", extra={"op": "startup"})

    try:
        asyncio.run(_run(socket_path, dispatcher))
    except KeyboardInterrupt:
        pass
    finally:
        release_lock(lock_handle)
        logger.info("chirpd stopped", extra={"op": "shutdown"})
    return 0


async def _run(socket_path: Path, dispatcher: Dispatcher) -> None:
    loop = asyncio.get_running_loop()
    serve_task = asyncio.create_task(serve(socket_path, dispatcher))

    def _request_stop() -> None:  # pragma: no cover — signal-driven
        if not serve_task.done():
            serve_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:  # pragma: no cover — non-Unix loops
            pass

    try:
        await serve_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
