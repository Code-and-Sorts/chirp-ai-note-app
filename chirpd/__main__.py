"""chirpd entry point — Apple-Silicon check, single-instance, asyncio serve."""

from __future__ import annotations

import asyncio
import logging
import platform
import signal
import sys
from pathlib import Path

from chirpd.backend import MLXBackend
from chirpd.dispatcher import Dispatcher
from chirpd.lifecycle import (
    ensure_runtime_dirs,
    single_instance_lock,
)
from chirpd.logging_setup import configure_logging
from chirpd.paths import SOCKET_PATH as DEFAULT_SOCKET_PATH
from chirpd.paths import lock_path_for_socket
from chirpd.server import serve
from chirpd.state import DaemonState
from config.settings import (
    get_daemon_socket_override,
    resolve_idle_timeout_seconds,
    resolve_max_resident_chat,
    resolve_max_resident_embed,
)
from llm.registry import read_registry

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

    configure_logging(to_stderr=sys.stdout.isatty())
    ensure_runtime_dirs()

    logger = logging.getLogger("chirpd")
    socket_path = _resolve_socket_path()
    with single_instance_lock(lock_path_for_socket(socket_path)) as acquired:
        if not acquired:
            return 0
        try:
            backend = MLXBackend()
            registry = read_registry()
            state = DaemonState(
                backend=backend,
                registry=registry,
                idle_timeout_s=resolve_idle_timeout_seconds(),
                registry_reader=read_registry,
                max_resident_chat=resolve_max_resident_chat(),
                max_resident_embed=resolve_max_resident_embed(),
            )
            dispatcher = Dispatcher(state=state)
            logger.info("chirpd starting", extra={"op": "startup"})

            try:
                asyncio.run(_run(socket_path, dispatcher))
            except KeyboardInterrupt:
                pass
        finally:
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
