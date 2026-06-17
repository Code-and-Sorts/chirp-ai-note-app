"""logfmt logging for the chirpd daemon: configuration, rotation, redaction.

Public surface:

- ``configure_logging(*, log_dir=None, level="INFO", to_stderr=False)`` — idempotent
  entrypoint called once at daemon startup. Installs a rotating file handler
  (10 MB, one prior generation) writing logfmt lines to ``<log_dir>/chirpd.log``.
- ``log_op_event(logger, level, msg, *, req_id, op, ...)`` — the **only** sanctioned
  way to emit op-related log lines from ``chirpd/``. It rejects any keyword not in
  the documented metadata set, which is how the "no user content in logs" rule
  (NFR-S5) is enforced at the call site.
- ``logfmt_escape(value)`` — the single place quoting/escaping logic lives.
- ``LogfmtFormatter`` — formats records as ``key=value`` lines; renders any record
  field whose key looks like user content as ``<redacted>``.

Redaction rule (NFR-S5, hard constraint): no user prompt text, chat messages,
note content, embed input text, or transcript text ever reaches the log. Logged
metadata is limited to op name, model alias, request id, token counts, durations,
operation results (``result=spawned|stopped|...``), and error classifications. A
forbidden-looking field that slips in via ``extra=``
is rendered ``<redacted>`` by the formatter and triggers exactly one
``err_type=RedactionViolation`` warning line (emitted by a logger-level filter, so
it fires once per record regardless of how many handlers are attached). Example::

    ts=2026-05-15T14:32:01.234Z level=info component=chirpd msg="chat ok" req_id=r-7c4a op=chat model=gemma-4-4b-it-4bit duration_ms=612 tokens=412
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024
LOG_BACKUP_COUNT: Final[int] = 1

LOG_FILE_NAME: Final[str] = "chirpd.log"

_DARWIN_LOG_SUBPATH: Final[Path] = Path("Library/Logs/chirp")
_FALLBACK_LOG_SUBPATH: Final[Path] = Path(".cache/chirp")
DEFAULT_LOG_DIR_DARWIN: Final[Path] = Path.home() / _DARWIN_LOG_SUBPATH
DEFAULT_LOG_DIR_FALLBACK: Final[Path] = Path.home() / _FALLBACK_LOG_SUBPATH

LOGFMT_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("ts", "level", "component", "msg")
LOGFMT_OPTIONAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "req_id",
        "op",
        "model",
        "duration_ms",
        "tokens",
        "err_code",
        "err_type",
        "result",
        "chirpd_path",
    }
)

FORBIDDEN_EXTRA_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"prompt|message|content|text|note|transcript", re.IGNORECASE
)

_MAX_MSG_LEN: Final[int] = 200
_LOGGER_NAMES: Final[tuple[str, ...]] = ("chirpd", "chirp.llm")
_NOISY_THIRD_PARTY: Final[tuple[str, ...]] = ("huggingface_hub", "urllib3")

_SAFE_VALUE_CHARS: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:/+@"
)

_STANDARD_LOGRECORD_ATTRS: Final[frozenset[str]] = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | frozenset({"message", "asctime", "taskName"})

_redaction_guard = threading.local()


def logfmt_escape(value: object) -> str:
    """Render ``value`` as a logfmt token, double-quoting only when needed.

    Bare when every character is in the safe set; otherwise wrapped in double
    quotes with ``\\`` and ``"`` backslash-escaped and newlines rendered ``\\n``.
    """
    text = str(value)
    if text == "":
        return '""'
    if all(ch in _SAFE_VALUE_CHARS for ch in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _forbidden_keys(record: logging.LogRecord) -> list[str]:
    """Return sorted ``extra=`` keys on ``record`` that look like user content."""
    extras = set(record.__dict__) - _STANDARD_LOGRECORD_ATTRS - LOGFMT_OPTIONAL_FIELDS
    return sorted(k for k in extras if FORBIDDEN_EXTRA_KEY_PATTERN.search(k))


class LogfmtFormatter(logging.Formatter):
    """Format records as logfmt lines: required fields first, then metadata.

    Side-effect-free (safe to call more than once per record, as
    ``RotatingFileHandler`` does). Forbidden-looking ``extra=`` fields render as
    ``<redacted>``; the accompanying violation warning is emitted by
    :class:`_RedactionViolationFilter`, not here.
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        ts = (
            timestamp.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{timestamp.microsecond // 1000:03d}Z"
        )
        required = {
            "ts": ts,
            "level": record.levelname.lower(),
            "component": logfmt_escape(record.name),
            "msg": logfmt_escape(record.getMessage()),
        }
        parts: list[str] = [f"{key}={required[key]}" for key in LOGFMT_REQUIRED_FIELDS]
        parts.extend(
            f"{key}={logfmt_escape(getattr(record, key))}"
            for key in LOGFMT_OPTIONAL_FIELDS
            if hasattr(record, key)
        )
        if record.exc_info and record.exc_info[0] is not None:
            parts.append(f"err_type={logfmt_escape(record.exc_info[0].__name__)}")
        parts.extend(f"{key}=<redacted>" for key in _forbidden_keys(record))
        return " ".join(parts)


class _RedactionViolationFilter(logging.Filter):
    """Emit one ``RedactionViolation`` warning per record carrying forbidden keys.

    Attached to the managed handlers (not the loggers) so that records from the
    whole ``chirpd.*`` subtree — which propagate to the parent loggers' handlers
    but do **not** run the parent loggers' filters — are covered. A per-record
    sentinel makes emission fire exactly once even when several handlers are
    installed; a thread-local guard additionally prevents the violation line
    itself from re-triggering the check.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_chirp_redaction_checked", False):
            return True
        record._chirp_redaction_checked = True
        if getattr(_redaction_guard, "active", False):
            return True
        keys = _forbidden_keys(record)
        if keys:
            _redaction_guard.active = True
            try:
                logging.getLogger(record.name).warning(
                    "redaction violation: keys=%s",
                    ",".join(keys),
                    extra={"err_type": "RedactionViolation"},
                )
            finally:
                _redaction_guard.active = False
        return True


def _default_log_dir() -> Path:
    """Resolve the platform default log directory at call time (lazy)."""
    subpath = _DARWIN_LOG_SUBPATH if sys.platform == "darwin" else _FALLBACK_LOG_SUBPATH
    return Path.home() / subpath


def resolve_log_path(log_dir: Path | None = None) -> Path:
    """Return the full path to ``chirpd.log`` — the canonical log location.

    Single source of truth shared between the writer (``configure_logging``) and
    any reader (``chirp daemon logs``): ``~/Library/Logs/chirp/chirpd.log`` on
    Darwin, ``~/.cache/chirp/chirpd.log`` elsewhere. Pass ``log_dir`` to override
    the directory (tests, non-default installs).
    """
    target_dir = log_dir if log_dir is not None else _default_log_dir()
    return target_dir / LOG_FILE_NAME


def configure_logging(
    *,
    log_dir: Path | None = None,
    level: str = "INFO",
    to_stderr: bool = False,
) -> None:
    """Install the logfmt rotating file handler on the chirp loggers.

    Idempotent: re-invoking replaces the handlers and filters this module
    previously installed rather than stacking duplicates. Creates ``<log_dir>/``
    lazily and raises :class:`OSError` (chained) if the log file cannot be opened
    — it never silently degrades to stderr-only. Raises :class:`ValueError` on an
    unknown ``level``.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unknown log level {level!r}")

    log_file = resolve_log_path(log_dir)
    target_dir = log_file.parent

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError as err:
        raise OSError(f"cannot open chirpd log file at {log_file}: {err}") from err

    file_handler.setFormatter(LogfmtFormatter())
    handlers: list[logging.Handler] = [_mark_managed(file_handler)]
    if to_stderr:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(LogfmtFormatter())
        handlers.append(_mark_managed(stream_handler))

    violation_filter = _RedactionViolationFilter()
    for handler in handlers:
        handler.addFilter(violation_filter)

    _remove_managed_handlers()
    for name in _LOGGER_NAMES:
        chirp_logger = logging.getLogger(name)
        for handler in handlers:
            chirp_logger.addHandler(handler)
        chirp_logger.setLevel(numeric_level)
        chirp_logger.propagate = False

    for noisy in _NOISY_THIRD_PARTY:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _mark_managed(handler: logging.Handler) -> logging.Handler:
    handler._chirp_managed = True  # type: ignore[attr-defined]
    return handler


def _remove_managed_handlers() -> None:
    detached: set[logging.Handler] = set()
    for name in _LOGGER_NAMES:
        chirp_logger = logging.getLogger(name)
        for handler in list(chirp_logger.handlers):
            if getattr(handler, "_chirp_managed", False):
                chirp_logger.removeHandler(handler)
                detached.add(handler)
    for handler in detached:
        handler.close()


def log_op_event(
    logger: logging.Logger,
    level: int,
    msg: str,
    *,
    req_id: str,
    op: str,
    model: str | None = None,
    duration_ms: int | None = None,
    tokens: int | None = None,
    err_code: str | None = None,
    err_type: str | None = None,
    result: str | None = None,
    chirpd_path: str | None = None,
    **extra: object,
) -> None:
    """Emit an op-level log line — the only sanctioned op emitter in ``chirpd/``.

    Rejects any keyword not in :data:`LOGFMT_OPTIONAL_FIELDS` (this is where the
    "no user content" rule lives — a caller cannot pass ``messages=`` or
    ``prompt=`` without a :class:`ValueError`). ``msg`` is truncated to 200 chars
    so a full prompt can never become the message body.
    """
    unknown = [key for key in extra if key not in LOGFMT_OPTIONAL_FIELDS]
    if unknown:
        raise ValueError(
            f"log_op_event received disallowed field(s) {unknown!r}; "
            f"only {sorted(LOGFMT_OPTIONAL_FIELDS)} are permitted"
        )

    if len(msg) > _MAX_MSG_LEN:
        msg = msg[:_MAX_MSG_LEN] + "…"

    fields: dict[str, object] = {"req_id": req_id, "op": op}
    optional = {
        "model": model,
        "duration_ms": duration_ms,
        "tokens": tokens,
        "err_code": err_code,
        "err_type": err_type,
        "result": result,
        "chirpd_path": chirpd_path,
    }
    fields.update({key: value for key, value in optional.items() if value is not None})
    logger.log(level, msg, extra=fields)
