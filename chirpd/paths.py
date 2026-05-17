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
