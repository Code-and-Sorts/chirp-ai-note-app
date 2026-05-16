"""Tests for chirpd.logging_setup formatter and rotating handler."""

from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

import pytest

from chirpd.logging_setup import (
    LOG_FILE,
    LogfmtFormatter,
    _quote,
    configure_logging,
)


def _format_record(**kwargs: object) -> str:
    record = logging.LogRecord(
        name=str(kwargs.get("component", "chirpd")),
        level=logging.INFO,
        pathname="test",
        lineno=1,
        msg=str(kwargs.get("msg", "hello")),
        args=(),
        exc_info=None,
    )
    for key, value in kwargs.items():
        if key in {"component", "msg"}:
            continue
        setattr(record, key, value)
    return LogfmtFormatter().format(record)


def test_quote_passes_safe_strings_through() -> None:
    assert _quote("simple_value-1") == "simple_value-1"


def test_quote_escapes_spaces_and_quotes() -> None:
    assert _quote('hello "world"') == '"hello \\"world\\""'


def test_quote_empty_string_renders_as_pair_of_quotes() -> None:
    assert _quote("") == '""'


def test_formatter_includes_err_type_when_exc_info_present() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="chirpd",
            level=logging.ERROR,
            pathname="t",
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    line = LogfmtFormatter().format(record)
    assert "err_type=ValueError" in line


def test_configure_logging_replaces_prior_rotating_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_dir = tmp_path / "logs2"
    monkeypatch.setattr("chirpd.logging_setup.LOG_DIR", log_dir)
    monkeypatch.setattr("chirpd.logging_setup.LOG_FILE", log_dir / "chirpd.log")
    configure_logging()
    configure_logging()
    try:
        root = logging.getLogger()
        matching = [
            h
            for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
            and Path(h.baseFilename) == log_dir / "chirpd.log"
        ]
        assert len(matching) == 1
    finally:
        for h in list(logging.getLogger().handlers):
            if (
                isinstance(h, logging.handlers.RotatingFileHandler)
                and Path(h.baseFilename) == log_dir / "chirpd.log"
            ):
                logging.getLogger().removeHandler(h)
                h.close()


def test_logfmt_required_keys() -> None:
    line = _format_record(msg="started")
    parts = dict(token.split("=", 1) for token in line.split(" ", 3))
    assert "ts" in parts
    assert parts["level"] == "info"
    assert parts["component"] == "chirpd"
    assert parts["msg"] == "started"


def test_logfmt_timestamp_format() -> None:
    line = _format_record()
    match = re.search(r"ts=(\S+)", line)
    assert match is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", match.group(1))


def test_logfmt_quotes_strings_with_spaces() -> None:
    line = _format_record(msg="hello world")
    assert 'msg="hello world"' in line


def test_logfmt_includes_optional_extras() -> None:
    line = _format_record(req_id="r-abc123def456", op="chat", duration_ms=42)
    assert "req_id=r-abc123def456" in line
    assert "op=chat" in line
    assert "duration_ms=42" in line


def test_log_redacts_no_user_content() -> None:
    forbidden = ("prompt", "messages", "content", "text", "transcript", "notes")
    line = _format_record(
        msg="generation complete",
        prompt="secret prompt",
        messages="secret messages",
        content="secret content",
        text="secret text",
        transcript="secret transcript",
        notes="secret notes",
    )
    for key in forbidden:
        assert f"{key}=" not in line
    assert "secret" not in line


def test_configure_logging_installs_rotating_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("chirpd.logging_setup.LOG_DIR", log_dir)
    monkeypatch.setattr("chirpd.logging_setup.LOG_FILE", log_dir / "chirpd.log")

    configure_logging()
    try:
        root = logging.getLogger()
        rotating = [
            h
            for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
            and Path(h.baseFilename) == log_dir / "chirpd.log"
        ]
        assert rotating, "expected a rotating file handler"
        handler = rotating[0]
        assert handler.maxBytes == 10_485_760
        assert handler.backupCount >= 1
        assert isinstance(handler.formatter, LogfmtFormatter)
        assert root.level == logging.INFO
    finally:
        for h in list(logging.getLogger().handlers):
            if (
                isinstance(h, logging.handlers.RotatingFileHandler)
                and Path(h.baseFilename) == log_dir / "chirpd.log"
            ):
                logging.getLogger().removeHandler(h)
                h.close()


def test_configure_logging_creates_log_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_dir = tmp_path / "fresh-logs"
    monkeypatch.setattr("chirpd.logging_setup.LOG_DIR", log_dir)
    monkeypatch.setattr("chirpd.logging_setup.LOG_FILE", log_dir / "chirpd.log")
    configure_logging()
    try:
        assert log_dir.exists()
    finally:
        for h in list(logging.getLogger().handlers):
            if (
                isinstance(h, logging.handlers.RotatingFileHandler)
                and Path(h.baseFilename) == log_dir / "chirpd.log"
            ):
                logging.getLogger().removeHandler(h)
                h.close()


def test_log_file_default_path_is_under_library_logs() -> None:
    assert LOG_FILE.parts[-3:] == ("Logs", "chirp", "chirpd.log")
