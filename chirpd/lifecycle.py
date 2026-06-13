"""Runtime-directory creation and single-instance flock enforcement."""

from __future__ import annotations

import contextlib
import fcntl
import logging
from collections.abc import Iterator
from pathlib import Path

from chirpd import paths

_logger = logging.getLogger("chirpd")


def ensure_runtime_dirs() -> None:
    paths.APP_SUPPORT_DIR.mkdir(
        parents=True, exist_ok=True, mode=paths.RUNTIME_DIR_MODE
    )
    paths.LOG_DIR.mkdir(parents=True, exist_ok=True, mode=paths.RUNTIME_DIR_MODE)


@contextlib.contextmanager
def single_instance_lock(
    lock_path: Path | None = None,
) -> Iterator[bool]:
    """Hold an exclusive non-blocking flock on ``lock_path`` for the block.

    Yields ``True`` if the lock was acquired (caller is the single instance),
    ``False`` if another process already holds it. The lock is released and the
    backing file descriptor is closed on exit, regardless of how the block ends.
    """
    target = lock_path if lock_path is not None else paths.LOCK_PATH
    target.parent.mkdir(parents=True, exist_ok=True, mode=paths.RUNTIME_DIR_MODE)
    with target.open("ab+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _logger.info("another chirpd already running; exiting")
            yield False
            return
        try:
            yield True
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
