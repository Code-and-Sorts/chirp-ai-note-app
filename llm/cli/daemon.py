"""``chirp daemon`` Typer subcommands.

This module is the eventual home of the seven ``chirp daemon`` subcommands:
``status`` (5.2), ``start`` / ``stop`` / ``restart`` (5.3), ``enable`` /
``disable`` (5.4), and ``logs`` (5.5). Story 5.6 wires ``daemon_app`` into the
top-level CLI; until then the subapp is reachable only through its Typer
instance (``python -m`` / ``CliRunner``).

``status`` issues ``health`` then ``model.status`` via :class:`llm.client.LLMClient`
and renders the merged snapshot — a Rich table on a TTY, a single JSON document
when ``--json`` is passed or stdout is piped. It has no side effect beyond the
read ops it issues (which may lazy-spawn a daemon, the one side effect any chirp
command that needs the daemon performs).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import IO, Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from chirpd.logging_setup import configure_logging, log_op_event
from llm.cli._console import console, stdout_console
from llm.client import LLMClient
from llm.exceptions import LLMDaemonUnreachable, LLMError
from llm.protocol import new_request_id

daemon_app = typer.Typer(help="Daemon lifecycle and diagnostics", no_args_is_help=True)


@daemon_app.callback()
def daemon_main() -> None:
    """Daemon lifecycle and diagnostics.

    This callback keeps ``daemon`` a multi-command group so Typer does not
    auto-promote a lone subcommand to the group root. ``status`` (5.2) lives
    here; ``start`` / ``stop`` / ``restart`` (5.3), ``enable`` / ``disable``
    (5.4), and ``logs`` (5.5) join it.
    """


_logger = logging.getLogger("chirp.llm.cli.daemon")

_JSON_STATUS_KEYS = (
    "running",
    "pid",
    "uptime_seconds",
    "version",
    "loaded_models",
    "last_request_at",
    "total_rss_bytes",
)


@daemon_app.command("status")
def status(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON to stdout."
    ),
) -> None:
    """Show daemon status: PID, uptime, version, loaded models, idle countdowns."""
    use_json = _should_use_json(json_output)
    _ensure_logging()
    req_id = new_request_id()
    client = LLMClient()
    start_ns = time.perf_counter_ns()

    try:
        payload = _collect_status(client)
    except LLMDaemonUnreachable as err:
        _log_status(req_id, start_ns, daemon_running=False)
        stopped = _stopped_payload()
        if use_json:
            _render_status_json(stopped, sys.stdout)
        else:
            _render_status_table(stopped, stdout_console)
            console.print(
                "the daemon is not running and could not be spawned. "
                "try: chirp daemon start",
                style="red",
                markup=False,
                soft_wrap=True,
            )
        raise typer.Exit(code=3) from err
    except LLMError as err:
        _log_status(req_id, start_ns, daemon_running=False)
        code = getattr(err, "code", None)
        suffix = f" ({code})" if code else ""
        console.print(
            f"Error: could not read daemon status: {err}{suffix}.",
            style="red",
            markup=False,
            soft_wrap=True,
        )
        raise typer.Exit(code=1) from err

    _log_status(req_id, start_ns, daemon_running=True)

    if not use_json:
        if getattr(client, "daemon_respawned", False):
            console.print(
                "daemon was restarted to match this CLI's version.",
                markup=False,
                soft_wrap=True,
            )
        elif getattr(client, "daemon_lazy_spawned", False):
            console.print(
                "daemon was not running — started a new instance.",
                markup=False,
                soft_wrap=True,
            )

    if use_json:
        _render_status_json(payload, sys.stdout)
    else:
        _render_status_table(payload, stdout_console)


def _collect_status(client: LLMClient) -> dict[str, Any]:
    """Merge ``health`` + ``model.status`` into the locked status payload.

    ``health`` carries uptime and version; ``model.status`` carries pid, total
    RSS, the loaded-model list, and last-request time. Per-model RSS is not yet
    reported by the daemon, so it normalizes to ``None`` (rendered ``—`` / null).
    """
    health = client.health_sync()
    models = client.model_status_sync()
    return {
        "running": True,
        "pid": models.get("pid"),
        "uptime_seconds": health.get("uptime_seconds"),
        "version": health.get("version"),
        "loaded_models": [_normalize_model(m) for m in models.get("models", [])],
        "last_request_at": models.get("last_request_at"),
        "total_rss_bytes": models.get("rss_bytes"),
    }


def _normalize_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "alias": model.get("alias"),
        "role": model.get("role"),
        "rss_bytes": model.get("rss_bytes"),
        "last_used": model.get("last_used"),
        "idle_countdown_seconds": model.get("idle_countdown_seconds"),
    }


def _stopped_payload() -> dict[str, Any]:
    return {
        "running": False,
        "pid": None,
        "uptime_seconds": None,
        "version": None,
        "loaded_models": [],
        "last_request_at": None,
        "total_rss_bytes": None,
    }


def _render_status_table(payload: dict[str, Any], target: Console) -> None:
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")

    if not payload.get("running"):
        table.add_row("daemon", "[red]stopped[/red]")
        target.print(table)
        return

    now = datetime.now(UTC)
    uptime = payload.get("uptime_seconds")
    total_rss = payload.get("total_rss_bytes")
    table.add_row("daemon", "running")
    table.add_row("pid", _or_dash(payload.get("pid")))
    table.add_row("uptime", _format_uptime(uptime) if uptime is not None else "—")
    table.add_row("version", _or_dash(payload.get("version")))
    table.add_row(
        "total RSS", _format_bytes(total_rss) if total_rss is not None else "—"
    )
    table.add_row(
        "last request", _format_relative_time(payload.get("last_request_at"), now)
    )
    target.print(table)

    models = payload.get("loaded_models") or []
    if not models:
        return

    model_table = Table(title="Loaded models", box=box.SIMPLE)
    for column in ("alias", "role", "RSS", "last used", "idle in"):
        model_table.add_column(column)
    for model in models:
        rss = model.get("rss_bytes")
        model_table.add_row(
            _or_dash(model.get("alias")),
            _or_dash(model.get("role")),
            _format_bytes(rss) if rss is not None else "—",
            _format_relative_time(model.get("last_used"), now),
            _format_idle_in(model.get("role"), model.get("idle_countdown_seconds")),
        )
    target.print(model_table)


def _render_status_json(payload: dict[str, Any], stream: IO[str]) -> None:
    stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _should_use_json(json_flag: bool) -> bool:
    """JSON when ``--json`` is passed or stdout is not an interactive terminal."""
    return json_flag or not stdout_console.is_terminal


def _format_bytes(n: int) -> str:
    if n < 1024:
        return "<1 KB"
    value = float(n)
    units = ("KB", "MB", "GB", "TB")
    for unit in units:
        value /= 1024
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} {units[-1]}"  # pragma: no cover — loop always returns


def _format_uptime(seconds: float) -> str:
    total = int(seconds)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m {secs:02d}s"
    return f"{minutes}m {secs}s"


def _format_relative_time(iso_timestamp: str | None, now: datetime) -> str:
    if iso_timestamp is None:
        return "—"
    then = datetime.fromisoformat(iso_timestamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    delta = (now - then).total_seconds()
    if delta < 1:
        return "just now"
    total = int(delta)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m ago"
    return f"{minutes}m {secs:02d}s ago"


def _format_idle_in(role: Any, countdown: float | None) -> str:
    if role == "embed":
        return "pinned"
    if countdown is None:
        return "—"
    return _format_uptime(countdown)


def _or_dash(value: Any) -> str:
    return "—" if value is None else str(value)


def _ensure_logging() -> None:
    """Best-effort: a diagnostic command must not die because it can't log."""
    try:
        configure_logging(to_stderr=False)
    except OSError:
        pass


def _log_status(req_id: str, start_ns: int, *, daemon_running: bool) -> None:
    # ``log_op_event`` enforces a strict field allowlist (NFR-S5); ``daemon_running``
    # is not in it, so the reachability bit rides in the (content-free) message.
    duration_ms = max(int((time.perf_counter_ns() - start_ns) / 1_000_000), 0)
    log_op_event(
        _logger,
        logging.INFO,
        f"daemon status: {'running' if daemon_running else 'stopped'}",
        req_id=req_id,
        op="status",
        duration_ms=duration_ms,
    )
