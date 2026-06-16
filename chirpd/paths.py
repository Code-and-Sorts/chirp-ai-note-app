"""Runtime filesystem layout for the chirpd daemon."""

from __future__ import annotations

from pathlib import Path
from typing import Final

APP_SUPPORT_DIR: Final[Path] = Path.home() / "Library" / "Application Support" / "chirp"
LOG_DIR: Final[Path] = Path.home() / "Library" / "Logs" / "chirp"
LOCK_PATH: Final[Path] = APP_SUPPORT_DIR / "chirpd.lock"
SOCKET_PATH: Final[Path] = APP_SUPPORT_DIR / "chirpd.sock"
MODELS_TOML_PATH: Final[Path] = APP_SUPPORT_DIR / "models.toml"
LOG_FILE: Final[Path] = LOG_DIR / "chirpd.log"
RUNTIME_DIR_MODE: Final[int] = 0o700


def lock_path_for_socket(socket_path: Path) -> Path:
    """Derive the single-instance lock path for ``socket_path``.

    One socket maps to one lock so ``CHIRP_DAEMON_SOCKET`` truly isolates an
    instance: a distinct override socket yields a distinct lock and a second
    daemon can run alongside the primary. The default socket maps back to
    :data:`LOCK_PATH` exactly, so the no-override path is byte-compatible with
    existing installs and the LaunchAgent.
    """
    if socket_path == SOCKET_PATH:
        return LOCK_PATH
    return socket_path.with_name(socket_path.name + ".lock")
