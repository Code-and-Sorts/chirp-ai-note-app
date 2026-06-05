"""Tests for ``chirp daemon status`` (story 5.2).

The daemon boundary is mocked end-to-end: :class:`LLMClient` is replaced with a
``MagicMock`` whose ``health_sync`` / ``model_status_sync`` return canned op
payloads, and ``configure_logging`` is neutralized so no test touches the real
log file. The suite runs with no MLX, network, or daemon subprocess.
"""

from __future__ import annotations

import io
import json
import re
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from llm.cli import daemon as daemon_module
from llm.exceptions import LLMDaemonUnreachable, LLMProtocolError

runner = CliRunner()

CHAT_ALIAS = "gemma-4-4b-it-4bit"
EMBED_ALIAS = "bge-small-en-v1.5"
LAST_USED = "2026-05-15T14:32:01.234+00:00"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _silence_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``status`` from opening the real chirpd log file during tests."""
    monkeypatch.setattr(daemon_module, "configure_logging", MagicMock())


@pytest.fixture
def force_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make stdout look like a terminal so ``status`` takes the table path."""
    monkeypatch.setattr(
        daemon_module, "stdout_console", Console(force_terminal=True, width=120)
    )


def _health_payload() -> dict[str, object]:
    return {"status": "ok", "uptime_seconds": 723.42, "version": "0.7.0"}


def _status_payload(models: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "pid": 12345,
        "uptime_seconds": 723.42,
        "daemon_version": "0.7.0",
        "rss_bytes": 4558481792,
        "idle_timeout_seconds": 300.0,
        "last_request_at": LAST_USED,
        "models": models if models is not None else [_chat_model()],
    }


def _chat_model() -> dict[str, object]:
    return {
        "alias": CHAT_ALIAS,
        "role": "chat",
        "loaded_at": LAST_USED,
        "last_used": LAST_USED,
        "idle_countdown_seconds": 166.8,
    }


def _embed_model() -> dict[str, object]:
    return {
        "alias": EMBED_ALIAS,
        "role": "embed",
        "loaded_at": LAST_USED,
        "last_used": LAST_USED,
        "idle_countdown_seconds": None,
    }


def _mock_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: list[dict[str, object]] | None = None,
    health_error: Exception | None = None,
    lazy_spawned: bool = False,
    respawned: bool = False,
) -> MagicMock:
    client = MagicMock()
    client.daemon_lazy_spawned = lazy_spawned
    client.daemon_respawned = respawned
    if health_error is not None:
        client.health_sync.side_effect = health_error
    else:
        client.health_sync.return_value = _health_payload()
    client.model_status_sync.return_value = _status_payload(models)
    monkeypatch.setattr(daemon_module, "LLMClient", MagicMock(return_value=client))
    return client


# --- command: running -------------------------------------------------------


def test_status_renders_table_when_daemon_running(
    monkeypatch: pytest.MonkeyPatch, force_tty: None
) -> None:
    _mock_client(monkeypatch)

    result = runner.invoke(daemon_module.daemon_app, ["status"])

    assert result.exit_code == 0
    assert CHAT_ALIAS in result.stdout
    assert "chat" in result.stdout
    assert "GB" in result.stdout  # total RSS rendered human-readable


def test_status_emits_valid_json_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_client(monkeypatch, models=[_chat_model(), _embed_model()])

    result = runner.invoke(daemon_module.daemon_app, ["status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == set(daemon_module._JSON_STATUS_KEYS)
    assert payload["running"] is True
    assert payload["pid"] == 12345
    assert payload["version"] == "0.7.0"
    assert payload["total_rss_bytes"] == 4558481792
    assert payload["last_request_at"] == LAST_USED
    assert len(payload["loaded_models"]) == 2
    chat = next(m for m in payload["loaded_models"] if m["role"] == "chat")
    assert set(chat) == {
        "alias",
        "role",
        "rss_bytes",
        "last_used",
        "idle_countdown_seconds",
    }
    assert chat["rss_bytes"] is None  # per-model RSS not reported by the daemon


def test_status_json_when_stdout_not_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Piped stdout (non-TTY, no --json) still emits JSON, not a table."""
    _mock_client(monkeypatch)

    result = runner.invoke(daemon_module.daemon_app, ["status"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["running"] is True


# --- command: unreachable ---------------------------------------------------


def test_status_renders_stopped_when_daemon_unreachable(
    monkeypatch: pytest.MonkeyPatch, force_tty: None
) -> None:
    _mock_client(monkeypatch, health_error=LLMDaemonUnreachable("no socket"))

    result = runner.invoke(daemon_module.daemon_app, ["status"])

    assert result.exit_code == 3
    assert "stopped" in result.stdout
    assert "chirp daemon start" in result.stderr


def test_status_json_renders_null_fields_when_daemon_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_client(monkeypatch, health_error=LLMDaemonUnreachable("no socket"))

    result = runner.invoke(daemon_module.daemon_app, ["status", "--json"])

    assert result.exit_code == 3
    assert json.loads(result.stdout) == {
        "running": False,
        "pid": None,
        "uptime_seconds": None,
        "version": None,
        "loaded_models": [],
        "last_request_at": None,
        "total_rss_bytes": None,
    }


def test_status_other_llm_error_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_client(monkeypatch, health_error=LLMProtocolError("malformed status"))

    result = runner.invoke(daemon_module.daemon_app, ["status"])

    assert result.exit_code == 1
    assert "could not read daemon status" in result.stderr


# --- command: spawn / respawn notices ---------------------------------------


def test_status_notes_lazy_spawn_to_stderr_on_tty(
    monkeypatch: pytest.MonkeyPatch, force_tty: None
) -> None:
    _mock_client(monkeypatch, lazy_spawned=True)

    result = runner.invoke(daemon_module.daemon_app, ["status"])

    assert result.exit_code == 0
    assert "started a new instance" in result.stderr
    assert "started a new instance" not in result.stdout


def test_status_suppresses_lazy_spawn_notice_in_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_client(monkeypatch, lazy_spawned=True)

    result = runner.invoke(daemon_module.daemon_app, ["status", "--json"])

    assert result.exit_code == 0
    assert "started a new instance" not in result.stderr


def test_status_notes_version_mismatch_resolved(
    monkeypatch: pytest.MonkeyPatch, force_tty: None
) -> None:
    _mock_client(monkeypatch, lazy_spawned=True, respawned=True)

    result = runner.invoke(daemon_module.daemon_app, ["status"])

    assert result.exit_code == 0
    assert "restarted to match this CLI's version" in result.stderr
    assert "started a new instance" not in result.stderr


# --- formatters -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "<1 KB"),
        (512, "<1 KB"),
        (1024, "1.0 KB"),
        (1024**2, "1.0 MB"),
        (1024**3, "1.0 GB"),
        (int(2.5 * 1024**3), "2.5 GB"),
    ],
)
def test_format_bytes(value: int, expected: str) -> None:
    assert daemon_module._format_bytes(value) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.5, "0m 0s"),
        (12, "0m 12s"),
        (90, "1m 30s"),
        (3600, "1h 0m 00s"),
        (3661, "1h 1m 01s"),
        (90061, "1d 1h 1m"),
        (86400 * 5 + 3700, "5d 1h 1m"),
    ],
)
def test_format_uptime(seconds: float, expected: str) -> None:
    assert daemon_module._format_uptime(seconds) == expected


def test_format_relative_time_handles_none() -> None:
    assert daemon_module._format_relative_time(None, datetime.now(UTC)) == "—"


def test_format_relative_time_handles_just_now() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    recent = (now - timedelta(seconds=0.3)).isoformat()
    assert daemon_module._format_relative_time(recent, now) == "just now"


@pytest.mark.parametrize(
    ("ago_seconds", "expected"),
    [(134, "2m 14s ago"), (242, "4m 02s ago"), (3725, "1h 02m ago")],
)
def test_format_relative_time_renders_ago(ago_seconds: int, expected: str) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    then = (now - timedelta(seconds=ago_seconds)).isoformat()
    assert daemon_module._format_relative_time(then, now) == expected


# --- table rendering helpers ------------------------------------------------


def _render_to_string(payload: dict[str, object]) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    daemon_module._render_status_table(payload, console)
    return buffer.getvalue()


def _running_payload(models: list[dict[str, object]]) -> dict[str, object]:
    return {
        "running": True,
        "pid": 12345,
        "uptime_seconds": 723.42,
        "version": "0.7.0",
        "loaded_models": models,
        "last_request_at": LAST_USED,
        "total_rss_bytes": 4558481792,
    }


def test_idle_countdown_renders_pinned_for_embed_role() -> None:
    output = _render_to_string(
        _running_payload([daemon_module._normalize_model(_embed_model())])
    )
    assert "pinned" in output


def test_idle_countdown_renders_countdown_for_chat_role() -> None:
    output = _render_to_string(
        _running_payload([daemon_module._normalize_model(_chat_model())])
    )
    assert "2m 46s" in output


def test_render_status_table_omits_loaded_models_table_when_empty() -> None:
    output = _render_to_string(_running_payload([]))
    assert "Loaded models" not in output


# --- diagnostic logging -----------------------------------------------------


def test_diagnostic_log_line_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_client(monkeypatch)
    log_mock = MagicMock()
    monkeypatch.setattr(daemon_module, "log_op_event", log_mock)

    result = runner.invoke(daemon_module.daemon_app, ["status", "--json"])

    assert result.exit_code == 0
    assert log_mock.call_count == 1
    _, kwargs = log_mock.call_args
    assert kwargs["op"] == "status"
    assert kwargs["duration_ms"] >= 0


# --- help discipline --------------------------------------------------------


def test_status_help_mentions_loaded_models_and_idle() -> None:
    result = runner.invoke(daemon_module.daemon_app, ["status", "--help"])

    # When Rich colorizes Typer's help (CI forces color, local non-TTY does not),
    # the ``--json`` option is split by ANSI codes between the dashes — strip them
    # so the assertion checks the rendered text, not its styling.
    help_text = _ANSI_RE.sub("", result.stdout)
    assert result.exit_code == 0
    assert "--json" in help_text
    assert "loaded models" in help_text
    assert "idle" in help_text
