"""Runtime-directory creation and single-instance flock enforcement."""

from __future__ import annotations

import fcntl
import logging
from pathlib import Path
from typing import IO, Final

APP_SUPPORT_DIR: Final[Path] = Path.home() / "Library" / "Application Support" / "chirp"
LOG_DIR: Final[Path] = Path.home() / "Library" / "Logs" / "chirp"
LOCK_PATH: Final[Path] = APP_SUPPORT_DIR / "chirpd.lock"
SOCKET_PATH: Final[Path] = APP_SUPPORT_DIR / "chirpd.sock"

_logger = logging.getLogger("chirpd")


def ensure_runtime_dirs() -> None:
    APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)


def acquire_single_instance_lock(lock_path: Path = LOCK_PATH) -> IO[bytes] | None:
    """Acquire an exclusive non-blocking flock on ``lock_path``.

    Returns the open file handle on success; the caller must keep it alive for
    the process lifetime (closing it releases the lock). Returns ``None`` when
    another process already holds the lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = open(lock_path, "ab+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        _logger.info("another chirpd already running; exiting")
        return None
    return handle


def release_lock(handle: IO[bytes]) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    handle.close()
