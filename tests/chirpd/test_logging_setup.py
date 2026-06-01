"""Tests for :mod:`chirpd.logging_setup` — logfmt format, rotation, redaction."""

from __future__ import annotations

import importlib
import logging
import logging.handlers
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from chirpd import logging_setup
from chirpd.logging_setup import (
    LogfmtFormatter,
    configure_logging,
    log_op_event,
    logfmt_escape,
)

CHIRP_LOGGER = "chirpd"


@pytest.fixture(autouse=True)
def _reset_chirp_loggers() -> Iterator[None]:
    """Detach managed handlers and reset chirp loggers between tests."""
    yield
    logging_setup._remove_managed_handlers()
    for name in logging_setup._LOGGER_NAMES:
        lg = logging.getLogger(name)
        lg.setLevel(logging.NOTSET)
        lg.propagate = True


def _read_log(log_dir: Path) -> str:
    return (log_dir / "chirpd.log").read_text(encoding="utf-8")


def _fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.split(" "):
        if "=" in token:
            key, _, value = token.partition("=")
            out.setdefault(key, value)
    return out


# --- AC-15 ----------------------------------------------------------------


def test_configure_logging_creates_log_directory(tmp_path: Path) -> None:
    log_dir = tmp_path / "fresh" / "logs"
    configure_logging(log_dir=log_dir)
    logging.getLogger(CHIRP_LOGGER).info("hello")
    assert log_dir.is_dir()
    assert (log_dir / "chirpd.log").exists()


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    configure_logging(log_dir=tmp_path)
    logging.getLogger(CHIRP_LOGGER).info("only-once-marker")
    assert _read_log(tmp_path).count("only-once-marker") == 1
    assert len(logging.getLogger(CHIRP_LOGGER).handlers) == 1


def test_rotation_at_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging_setup, "LOG_MAX_BYTES", 1024)
    configure_logging(log_dir=tmp_path)
    logger = logging.getLogger(CHIRP_LOGGER)
    for i in range(400):
        logger.info("rotation filler line number %03d padding-padding", i)
    assert (tmp_path / "chirpd.log.1").exists()
    assert not (tmp_path / "chirpd.log.2").exists()


def test_logfmt_required_fields_order(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    logging.getLogger(CHIRP_LOGGER).info("started")
    first_line = _read_log(tmp_path).splitlines()[0]
    leading_keys = [token.split("=", 1)[0] for token in first_line.split(" ")[:4]]
    assert leading_keys == ["ts", "level", "component", "msg"]


def test_logfmt_escape_quotes_spaces_and_specials() -> None:
    assert logfmt_escape("hello world") == '"hello world"'
    assert logfmt_escape("a=b") == '"a=b"'
    assert logfmt_escape('say "hi"') == '"say \\"hi\\""'
    assert logfmt_escape("line\nbreak") == '"line\\nbreak"'
    assert logfmt_escape("back\\slash") == '"back\\\\slash"'


def test_logfmt_escape_passthrough_simple_value() -> None:
    assert logfmt_escape("gemma-4-4b-it-4bit") == "gemma-4-4b-it-4bit"


def test_logfmt_escape_empty_string() -> None:
    assert logfmt_escape("") == '""'


def test_default_log_dir_darwin_uses_library_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(logging_setup.sys, "platform", "darwin")
    assert logging_setup._default_log_dir().parts[-2:] == ("Logs", "chirp")


def test_timestamp_is_utc_with_milliseconds(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    logging.getLogger(CHIRP_LOGGER).info("tick")
    ts = _fields(_read_log(tmp_path).splitlines()[0])["ts"]
    assert ts.endswith("Z")
    fractional = ts[:-1].split(".")[1]
    assert len(fractional) == 3


def test_to_stderr_flag_adds_stream_handler(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, to_stderr=True)
    handlers = logging.getLogger(CHIRP_LOGGER).handlers
    assert len(handlers) == 2
    assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers)
    assert any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in handlers
    )


def test_import_performs_no_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    importlib.reload(logging_setup)
    assert not (tmp_path / "Library" / "Logs" / "chirp").exists()
    assert not (tmp_path / ".cache" / "chirp").exists()


def test_non_darwin_falls_back_to_xdg_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(logging_setup.sys, "platform", "linux")
    configure_logging()
    logging.getLogger(CHIRP_LOGGER).info("xdg")
    assert (tmp_path / ".cache" / "chirp" / "chirpd.log").exists()


def test_configure_logging_rejects_unknown_level(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        configure_logging(log_dir=tmp_path, level="WANR")


def test_oserror_on_unwritable_directory(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o000)
    try:
        with pytest.raises(OSError):
            configure_logging(log_dir=locked)
    finally:
        os.chmod(locked, 0o700)


# --- AC-16 (redaction discipline) -----------------------------------------


@pytest.mark.parametrize(
    "bad_key", ["prompt", "messages", "content", "text", "note", "transcript"]
)
def test_log_op_event_rejects_unknown_keys(bad_key: str) -> None:
    with pytest.raises(ValueError):
        log_op_event(
            logging.getLogger(CHIRP_LOGGER),
            logging.INFO,
            "chat ok",
            req_id="r-1",
            op="chat",
            **{bad_key: "should not be allowed"},  # type: ignore[arg-type]
        )


def test_log_op_event_truncates_long_msg(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    long_msg = "x" * 500
    log_op_event(
        logging.getLogger(CHIRP_LOGGER),
        logging.INFO,
        long_msg,
        req_id="r-1",
        op="chat",
    )
    text = _read_log(tmp_path)
    assert "x" * 200 + "…" in text
    assert "x" * 201 not in text


def test_formatter_redacts_forbidden_extra_keys_defensively(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    logging.getLogger(CHIRP_LOGGER).info("test", extra={"prompt": "SECRET_CANARY"})
    text = _read_log(tmp_path)
    assert "SECRET_CANARY" not in text
    assert "prompt=<redacted>" in text
    assert text.count("err_type=RedactionViolation") == 1


def test_redaction_violation_emitted_once_with_two_handlers(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, to_stderr=True)
    logging.getLogger(CHIRP_LOGGER).info("test", extra={"content": "SECRET"})
    assert _read_log(tmp_path).count("RedactionViolation") == 1


def test_redaction_violation_fires_for_child_logger(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    logging.getLogger("chirpd.dispatcher").info("op", extra={"prompt": "SECRET"})
    text = _read_log(tmp_path)
    assert "SECRET" not in text
    assert "prompt=<redacted>" in text
    assert text.count("err_type=RedactionViolation") == 1
    assert "component=chirpd.dispatcher" in text


def test_canary_value_in_op_metadata_passes_through(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    log_op_event(
        logging.getLogger(CHIRP_LOGGER),
        logging.INFO,
        "loaded",
        req_id="r-1",
        op="model.load",
        model="my-model-SECRET_CANARY",
    )
    assert "model=my-model-SECRET_CANARY" in _read_log(tmp_path)


def test_full_request_payload_never_appears_in_log(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    log_op_event(
        logging.getLogger(CHIRP_LOGGER),
        logging.INFO,
        "chat completed",
        req_id="r-7c4a",
        op="chat",
        model="gemma-4-4b-it-4bit",
        duration_ms=612,
        tokens=412,
    )
    assert "REDACTION_CANARY_42" not in _read_log(tmp_path)


def test_formatter_emits_exc_type() -> None:
    formatter = LogfmtFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "chirpd", logging.ERROR, "t", 1, "failed", (), sys.exc_info()
        )
    assert "err_type=ValueError" in formatter.format(record)
