"""Tests for ``chirp daemon logs`` (story 5.5).

The reader-side counterpart to ``chirpd.logging_setup``: ``configure_logging`` is
neutralized and ``resolve_log_path`` is redirected at a ``tmp_path`` file so no
test touches the real ``~/Library/Logs/chirp/chirpd.log``. The follow-loop tests
drive ``_follow_log_file`` (or the ``logs`` command body) on a background thread,
observe a patched ``sys.stdout`` ``StringIO`` live, and stop the loop by making
the patched ``time.sleep`` raise ``KeyboardInterrupt`` — the same exit path a
real ``^C`` takes (AC-7).
"""

from __future__ import annotations

import io
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from chirpd.logging_setup import LOG_FILE_NAME, resolve_log_path
from llm.cli import daemon as daemon_module

runner = CliRunner()

# Captured before any test patches ``time.sleep``, so the test-side poll/settle
# helpers keep sleeping for real even while the follow loop's sleep is rigged.
_REAL_SLEEP = time.sleep

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _silence_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``logs`` from configuring the real chirpd log handler."""
    monkeypatch.setattr(daemon_module, "configure_logging", MagicMock())


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the follow/wait cadences so threaded tests finish fast (AC-12)."""
    monkeypatch.setattr(daemon_module, "_FOLLOW_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(daemon_module, "_WAIT_FILE_POLL_INTERVAL_SECONDS", 0.01)


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``resolve_log_path`` at a tmp file; return that path (may not exist)."""
    path = tmp_path / LOG_FILE_NAME
    monkeypatch.setattr(daemon_module, "resolve_log_path", lambda *a, **k: path)
    return path


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        _REAL_SLEEP(0.005)
    return predicate()


def _patch_stoppable_sleep(
    monkeypatch: pytest.MonkeyPatch, stop: threading.Event
) -> None:
    """Make the follow loop's ``time.sleep`` raise ``KeyboardInterrupt`` on stop."""

    def fake_sleep(_seconds: float) -> None:
        _REAL_SLEEP(0.005)
        if stop.is_set():
            raise KeyboardInterrupt

    monkeypatch.setattr(daemon_module.time, "sleep", fake_sleep)


def _run_in_thread(
    target: Callable[[], Any],
) -> tuple[threading.Thread, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def run() -> None:
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 — surfaced to the assertion
            captured["exc"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, captured


# --- resolve_log_path (AC-2) ------------------------------------------------


def test_resolve_log_path_appends_log_file_name(tmp_path: Path) -> None:
    assert resolve_log_path(tmp_path) == tmp_path / LOG_FILE_NAME


# --- default / -n behavior (AC-3, AC-4) -------------------------------------


def test_logs_default_cats_full_file(log_path: Path) -> None:
    content = "".join(f"line{i}\n" for i in range(1, 6))
    log_path.write_text(content, encoding="utf-8")

    result = runner.invoke(daemon_module.daemon_app, ["logs"])

    assert result.exit_code == 0
    assert result.stdout == content


def test_logs_with_n_shows_last_n_lines(log_path: Path) -> None:
    lines = [f"line{i}\n" for i in range(1, 101)]
    log_path.write_text("".join(lines), encoding="utf-8")

    result = runner.invoke(daemon_module.daemon_app, ["logs", "-n", "10"])

    assert result.exit_code == 0
    assert result.stdout == "".join(lines[-10:])


def test_logs_with_n_larger_than_file_shows_all(log_path: Path) -> None:
    content = "a\nb\nc\n"
    log_path.write_text(content, encoding="utf-8")

    result = runner.invoke(daemon_module.daemon_app, ["logs", "-n", "100"])

    assert result.exit_code == 0
    assert result.stdout == content


def test_logs_handles_partial_final_line(log_path: Path) -> None:
    log_path.write_text("line1\nline2 without newline", encoding="utf-8")

    result = runner.invoke(daemon_module.daemon_app, ["logs"])

    assert result.exit_code == 0
    assert "line1" in result.stdout
    assert "line2 without newline" in result.stdout


# --- missing file (AC-8) ----------------------------------------------------


def test_logs_missing_file_default_prints_notice(log_path: Path) -> None:
    result = runner.invoke(daemon_module.daemon_app, ["logs"])

    assert result.exit_code == 0
    assert "no log file" in result.stderr
    assert str(log_path) in result.stderr
    assert result.stdout == ""


def test_logs_missing_file_with_follow_waits(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = threading.Event()
    _patch_stoppable_sleep(monkeypatch, stop)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(daemon_module.sys, "stdout", out)
    monkeypatch.setattr(daemon_module.sys, "stderr", err)

    thread, captured = _run_in_thread(
        lambda: daemon_module.logs(follow=True, lines=None)
    )
    _REAL_SLEEP(0.05)  # let the loop announce the wait before the file appears
    log_path.write_text("first\n", encoding="utf-8")

    assert _wait_until(lambda: "first" in out.getvalue())
    stop.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "waiting for log file" in err.getvalue()
    assert "first" in out.getvalue()
    assert "exc" not in captured


# --- follow (AC-5, AC-6, AC-7) ----------------------------------------------


def test_logs_follow_picks_up_appended_lines(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path.write_text("line1\n", encoding="utf-8")
    stop = threading.Event()
    _patch_stoppable_sleep(monkeypatch, stop)
    out = io.StringIO()
    monkeypatch.setattr(daemon_module.sys, "stdout", out)

    thread, captured = _run_in_thread(
        lambda: daemon_module._follow_log_file(log_path, start_at_end=True)
    )
    _REAL_SLEEP(0.05)  # let the loop open + seek to EOF before we append
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("line2\n")
        fh.flush()

    assert _wait_until(lambda: "line2" in out.getvalue())
    stop.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "line2" in out.getvalue()
    assert "line1" not in out.getvalue()  # started at EOF, so prior content skipped
    assert "exc" not in captured


def test_logs_follow_with_n_dumps_then_follows(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path.write_text("".join(f"line{i}\n" for i in range(1, 6)), encoding="utf-8")
    stop = threading.Event()
    _patch_stoppable_sleep(monkeypatch, stop)
    out = io.StringIO()
    monkeypatch.setattr(daemon_module.sys, "stdout", out)

    thread, captured = _run_in_thread(lambda: daemon_module.logs(follow=True, lines=3))
    assert _wait_until(lambda: "line5" in out.getvalue())
    assert out.getvalue().startswith("line3\nline4\nline5\n")  # last 3 first

    _REAL_SLEEP(0.05)  # let the follow loop seek to EOF after the dump
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("line6\n")
        fh.flush()

    assert _wait_until(lambda: "line6" in out.getvalue())
    stop.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "exc" not in captured


def test_logs_follow_across_rotation(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The load-bearing AC-11 test: the follow loop must re-open by name when the
    # inode changes under it (chirpd.log → chirpd.log.1 + fresh chirpd.log).
    log_path.write_text("a1\na2\na3\n", encoding="utf-8")
    stop = threading.Event()
    _patch_stoppable_sleep(monkeypatch, stop)
    out = io.StringIO()
    monkeypatch.setattr(daemon_module.sys, "stdout", out)

    thread, captured = _run_in_thread(
        lambda: daemon_module._follow_log_file(log_path, start_at_end=True)
    )
    _REAL_SLEEP(0.05)  # let the loop seek to EOF
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("a4\n")
        fh.flush()
    assert _wait_until(lambda: "a4" in out.getvalue())

    rotated = log_path.with_name(log_path.name + ".1")
    log_path.rename(rotated)
    log_path.write_text("b1\n", encoding="utf-8")

    assert _wait_until(lambda: "b1" in out.getvalue())
    stop.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert "exc" not in captured


def test_logs_ctrl_c_exits_cleanly(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path.write_text("x\n", encoding="utf-8")

    def raise_keyboard_interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(daemon_module.time, "sleep", raise_keyboard_interrupt)

    result = runner.invoke(daemon_module.daemon_app, ["logs", "-f"])

    assert result.exit_code == 0
    assert "Traceback" not in result.stdout


# --- exit codes (AC-14) -----------------------------------------------------


def test_logs_ioerror_exits_1(log_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path.write_text("x\n", encoding="utf-8")

    def boom(_path: Path) -> None:
        raise OSError("disk error")

    monkeypatch.setattr(daemon_module, "_cat_log_file", boom)

    result = runner.invoke(daemon_module.daemon_app, ["logs"])

    assert result.exit_code == 1
    assert "couldn't read log file" in result.stderr


def test_logs_follow_ioerror_exits_1(log_path: Path) -> None:
    # A path that exists but cannot be opened for reading (a directory) makes the
    # follow loop's open() raise OSError — it must surface as the friendly exit-1
    # message, not an uncaught traceback.
    log_path.mkdir()

    result = runner.invoke(daemon_module.daemon_app, ["logs", "-f"])

    assert result.exit_code == 1
    assert "couldn't read log file" in result.stderr


def test_logs_negative_n_is_usage_error(log_path: Path) -> None:
    # AC-14: negative -n is a Typer usage error (exit 2), not a crash.
    log_path.write_text("a\nb\nc\n", encoding="utf-8")

    result = runner.invoke(daemon_module.daemon_app, ["logs", "-n", "-5"])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)


# --- diagnostic logging (AC-13) ---------------------------------------------


def test_logs_emits_diagnostic_log_event(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path.write_text("hi\n", encoding="utf-8")
    log_mock = MagicMock()
    monkeypatch.setattr(daemon_module, "log_op_event", log_mock)

    result = runner.invoke(daemon_module.daemon_app, ["logs"])

    assert result.exit_code == 0
    assert log_mock.call_count == 1
    _, kwargs = log_mock.call_args
    assert kwargs["op"] == "daemon_logs"
    assert kwargs["result"] == "cat"
    assert kwargs["duration_ms"] == 0


def test_logs_diagnostic_result_missing_when_no_file(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_mock = MagicMock()
    monkeypatch.setattr(daemon_module, "log_op_event", log_mock)

    result = runner.invoke(daemon_module.daemon_app, ["logs"])

    assert result.exit_code == 0
    assert log_mock.call_args.kwargs["result"] == "missing"


# --- help discipline (AC-10, AC-15) -----------------------------------------


def test_logs_help_text(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(daemon_module.daemon_app, ["logs", "--help"])

    help_text = _ANSI_RE.sub("", result.stdout)
    assert result.exit_code == 0
    assert "--follow" in help_text
    assert "--lines" in help_text
    assert "Tail with -f" in help_text
    assert "json" not in help_text.lower()  # AC-10: no JSON mention
