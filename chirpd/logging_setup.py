"""logfmt root-logger configuration for the chirpd daemon."""

from __future__ import annotations

import logging
import logging.handlers
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from chirpd import paths

LOG_MAX_BYTES: Final[int] = 10_485_760
LOG_BACKUP_COUNT: Final[int] = 1

_REQUIRED_KEYS: Final[tuple[str, ...]] = ("ts", "level", "component", "msg")
_OPTIONAL_KEYS: Final[tuple[str, ...]] = (
    "req_id",
    "op",
    "model",
    "duration_ms",
    "tokens",
    "err_code",
    "err_type",
)

_SAFE_VALUE_CHARS: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:/+@"
)


def _quote(value: object) -> str:
    text = str(value)
    if text == "":
        return '""'
    if all(ch in _SAFE_VALUE_CHARS for ch in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class LogfmtFormatter(logging.Formatter):
    """Format records as logfmt key=value lines with a closed key allow-list."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        ts = (
            timestamp.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{timestamp.microsecond // 1000:03d}Z"
        )

        required_values: dict[str, str] = {
            "ts": ts,
            "level": record.levelname.lower(),
            "component": _quote(record.name),
            "msg": _quote(record.getMessage()),
        }
        parts: list[str] = [f"{key}={required_values[key]}" for key in _REQUIRED_KEYS]
        for key in _OPTIONAL_KEYS:
            if hasattr(record, key):
                parts.append(f"{key}={_quote(getattr(record, key))}")
        if record.exc_info and record.exc_info[0] is not None:
            parts.append(f"err_type={_quote(record.exc_info[0].__name__)}")
        return " ".join(parts)


def configure_logging() -> None:
    """Install the rotating file handler with the logfmt formatter at INFO."""
    paths.LOG_DIR.mkdir(parents=True, exist_ok=True, mode=paths.RUNTIME_DIR_MODE)
    handler = logging.handlers.RotatingFileHandler(
        paths.LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(LogfmtFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for existing in list(root.handlers):
        if (
            isinstance(existing, logging.handlers.RotatingFileHandler)
            and Path(existing.baseFilename) == paths.LOG_FILE
        ):
            root.removeHandler(existing)
            existing.close()
    root.addHandler(handler)
