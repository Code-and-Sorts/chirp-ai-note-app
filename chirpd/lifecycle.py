"""Runtime-directory creation and single-instance flock enforcement."""

from __future__ import annotations

import fcntl
import logging
from pathlib import Path
from typing import IO

from chirpd import paths

_logger = logging.getLogger("chirpd")


def ensure_runtime_dirs() -> None:
    paths.APP_SUPPORT_DIR.mkdir(
        parents=True, exist_ok=True, mode=paths.RUNTIME_DIR_MODE
    )
    paths.LOG_DIR.mkdir(parents=True, exist_ok=True, mode=paths.RUNTIME_DIR_MODE)


def acquire_single_instance_lock(
    lock_path: Path | None = None,
) -> IO[bytes] | None:
    """Acquire an exclusive non-blocking flock on ``lock_path``.

    Returns the open file handle on success; the caller must keep it alive for
    the process lifetime (closing it releases the lock). Returns ``None`` when
    another process already holds the lock.
    """
    target = lock_path if lock_path is not None else paths.LOCK_PATH
    target.parent.mkdir(parents=True, exist_ok=True, mode=paths.RUNTIME_DIR_MODE)
    handle = open(target, "ab+")
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
