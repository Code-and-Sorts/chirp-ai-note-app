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

``start`` / ``stop`` / ``restart`` manage the daemon process explicitly. ``start``
spawns ``chirpd`` via :func:`subprocess.Popen` and polls the socket; ``stop`` and
``restart`` shut the daemon down.

``enable`` / ``disable`` (5.4) install and remove the ``com.chirp.chirpd``
LaunchAgent so the daemon auto-starts at login. These commands are thin wrappers:
the real work — plist generation, ``launchctl load``/``unload``, the typed error
surface — lives in :mod:`chirpd.launchd`, which ``chirp init`` also imports.

**Shutdown mechanism (story 5.3 task 1).** CHIRPD-CORE ships a *signal-based*
shutdown, not a wire ``shutdown`` op: ``chirpd/__main__.py`` installs ``SIGTERM`` /
``SIGINT`` handlers that cancel the asyncio serve task and exit cleanly (option
(b) in the story). ``stop`` therefore reads the daemon ``pid`` from ``model.status``
and sends ``SIGTERM`` (escalating to ``SIGKILL`` only if the socket does not vacate
within the timeout). There is no ``LLMClient.shutdown()`` method to use. Because
``stop`` must never *spawn* a daemon just to ask it to die, it probes with
``model_status_sync(spawn_if_absent=False)`` (AC-4); the daemon ``pid`` comes from
``model.status``, not ``health`` (``health`` carries only uptime/version).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket as socketlib
import subprocess
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from chirpd.launchd import (
    LAUNCH_AGENT_PLIST_PATH,
    LaunchAgentAlreadyInstalled,
    LaunchAgentError,
    LaunchAgentNotInstalled,
    LaunchctlFailed,
    install_launch_agent,
    installed_chirpd_path,
    is_launch_agent_installed,
    uninstall_launch_agent,
)
from chirpd.logging_setup import configure_logging, log_op_event, resolve_log_path
from config.settings import CHIRP_DAEMON_SOCKET_ENV
from llm.cli._console import console, stdout_console
from llm.client import LLMClient, resolve_socket_path
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

# The locked top-level JSON schema (epic §3 decision 10 / AC-4), in emit order.
# Source of truth for ``_stopped_payload`` and asserted by the CLI tests.
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
    payload: dict[str, Any] = dict.fromkeys(_JSON_STATUS_KEYS, None)
    payload["running"] = False
    payload["loaded_models"] = []
    return payload


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
        # A read-only home or un-creatable log dir must not break a read-only
        # diagnostic command — degrade to no logging rather than failing.
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


# --- lifecycle: start / stop / restart (story 5.3) --------------------------

_SOCKET_POLL_INTERVAL_S = 0.05
_START_TIMEOUT_S = 5.0
_STOP_TIMEOUT_S = 5.0
_RESTART_SETTLE_S = 0.1
_LOG_HINT = "~/Library/Logs/chirp/chirpd.log"

_ERR_NOT_ON_PATH = "DAEMON_NOT_ON_PATH"
_ERR_START_TIMEOUT = "DAEMON_START_TIMEOUT"
_ERR_UNREACHABLE = "DAEMON_UNREACHABLE"
_ERR_STOP_TIMEOUT = "DAEMON_STOP_TIMEOUT"
_ERR_SPAWN_FAILED = "DAEMON_SPAWN_FAILED"


@daemon_app.command("start")
def start(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON to stdout."
    ),
) -> None:
    """Spawn the daemon process. No-op if already running."""
    use_json = _should_use_json(json_output)
    _ensure_logging()
    start_ns = time.perf_counter_ns()
    socket_path = resolve_socket_path()

    result = _attempt_start(socket_path)

    if result["ok"]:
        outcome = "already_running" if result["already"] else "spawned"
        _log_lifecycle("daemon_start", start_ns, result=outcome)
        if use_json:
            _emit_json(
                {
                    "action": "start",
                    "running": True,
                    "pid": result["pid"],
                    "spawned": result["spawned"],
                }
            )
        elif result["already"]:
            console.print("daemon is already running", markup=False, soft_wrap=True)
        else:
            stdout_console.print(
                f"daemon started (pid={_format_pid(result['pid'])})",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
        return

    error = result["error"]
    _log_lifecycle("daemon_start", start_ns, result="failed")
    if use_json:
        _emit_json(
            {
                "action": "start",
                "running": False,
                "pid": None,
                "spawned": result["spawned"],
                "error": error,
            }
        )
    else:
        console.print(error["message"], style="red", markup=False, soft_wrap=True)
    raise typer.Exit(code=_start_exit_code(error))


@daemon_app.command("stop")
def stop(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON to stdout."
    ),
) -> None:
    """Stop the daemon process. No-op if not running."""
    use_json = _should_use_json(json_output)
    _ensure_logging()
    start_ns = time.perf_counter_ns()
    socket_path = resolve_socket_path()

    result = _attempt_stop(socket_path)

    if result["ok"]:
        outcome = "stopped" if result["was_running"] else "not_running"
        _log_lifecycle("daemon_stop", start_ns, result=outcome)
        if use_json:
            _emit_json(
                {
                    "action": "stop",
                    "running": False,
                    "was_running": result["was_running"],
                    "killed": False,
                }
            )
        elif result["was_running"]:
            stdout_console.print("daemon stopped", markup=False, soft_wrap=True)
        else:
            console.print("daemon is not running", markup=False, soft_wrap=True)
        return

    error = result["error"]
    _log_lifecycle(
        "daemon_stop", start_ns, result="killed" if result["killed"] else "failed"
    )
    if use_json:
        _emit_json(
            {
                "action": "stop",
                "running": result["running"],
                "was_running": True,
                "killed": result["killed"],
                "error": error,
            }
        )
    else:
        console.print(error["message"], style="red", markup=False, soft_wrap=True)
    raise typer.Exit(code=1)


@daemon_app.command("restart")
def restart(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON to stdout."
    ),
) -> None:
    """Stop then start the daemon. Starts a fresh instance if not running."""
    use_json = _should_use_json(json_output)
    _ensure_logging()
    start_ns = time.perf_counter_ns()
    socket_path = resolve_socket_path()

    stop_result = _attempt_stop(socket_path)
    old_pid = stop_result["pid"]

    if not stop_result["ok"]:
        error = stop_result["error"]
        _log_lifecycle("daemon_restart", start_ns, result="stop_failed")
        if use_json:
            _emit_json(
                {
                    "action": "restart",
                    "running": stop_result["running"],
                    "old_pid": old_pid,
                    "new_pid": None,
                    "error": error,
                }
            )
        else:
            console.print(error["message"], style="red", markup=False, soft_wrap=True)
        raise typer.Exit(code=1)

    was_running = stop_result["was_running"]
    if was_running:
        # Give the asyncio server time to unlink the socket and release the flock
        # before a fresh chirpd races to bind the same path (see Dev notes).
        time.sleep(_RESTART_SETTLE_S)
    elif not use_json:
        console.print(
            "daemon is not running — starting fresh instance...",
            markup=False,
            soft_wrap=True,
        )

    start_result = _attempt_start(socket_path)

    if start_result["ok"]:
        _log_lifecycle("daemon_restart", start_ns, result="ok")
        if use_json:
            _emit_json(
                {
                    "action": "restart",
                    "running": True,
                    "old_pid": old_pid,
                    "new_pid": start_result["pid"],
                }
            )
        else:
            stdout_console.print(
                f"daemon restarted (pid={_format_pid(start_result['pid'])})",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
        return

    error = start_result["error"]
    _log_lifecycle("daemon_restart", start_ns, result="start_failed")
    if was_running:
        message = f"daemon was stopped but did not restart — check {_LOG_HINT}"
        exit_code = 1
    else:
        message = error["message"]
        exit_code = _start_exit_code(error)
    if use_json:
        _emit_json(
            {
                "action": "restart",
                "running": False,
                "old_pid": old_pid,
                "new_pid": None,
                "error": {"code": error["code"], "message": message},
            }
        )
    else:
        console.print(message, style="red", markup=False, soft_wrap=True)
    raise typer.Exit(code=exit_code)


def _attempt_start(socket_path: Path) -> dict[str, Any]:
    """Bring the daemon up. Returns an outcome dict; never prints or exits.

    Keys: ``ok``, ``pid``, ``spawned``, ``already``, ``error`` (a
    ``{"code", "message"}`` dict on failure, else ``None``).
    """
    running, pid = _probe_running(socket_path)
    if running:
        return {
            "ok": True,
            "pid": pid,
            "spawned": False,
            "already": True,
            "error": None,
        }

    chirpd_path = shutil.which("chirpd")
    if chirpd_path is None:
        return {
            "ok": False,
            "pid": None,
            "spawned": False,
            "already": False,
            "error": {
                "code": _ERR_NOT_ON_PATH,
                "message": "chirpd binary not found on PATH — is chirp installed correctly?",
            },
        }

    try:
        _spawn_chirpd(chirpd_path, socket_path)
    except OSError as exc:
        return {
            "ok": False,
            "pid": None,
            "spawned": False,
            "already": False,
            "error": {
                "code": _ERR_SPAWN_FAILED,
                "message": f"failed to spawn chirpd ({chirpd_path}): {exc}",
            },
        }

    if not _wait_for_socket(socket_path, present=True, timeout=_START_TIMEOUT_S):
        return {
            "ok": False,
            "pid": None,
            "spawned": True,
            "already": False,
            "error": {
                "code": _ERR_START_TIMEOUT,
                "message": f"daemon failed to start within {int(_START_TIMEOUT_S)} seconds — check {_LOG_HINT}",
            },
        }

    running, pid = _probe_running(socket_path)
    if not running:
        return {
            "ok": False,
            "pid": None,
            "spawned": True,
            "already": False,
            "error": {
                "code": _ERR_UNREACHABLE,
                "message": f"daemon started but is not reachable — check {_LOG_HINT}",
            },
        }
    return {"ok": True, "pid": pid, "spawned": True, "already": False, "error": None}


def _attempt_stop(socket_path: Path) -> dict[str, Any]:
    """Bring the daemon down via SIGTERM (SIGKILL escalation). Never prints/exits.

    Keys: ``ok``, ``was_running``, ``pid`` (the pid found, for ``restart`` to
    report as ``old_pid``), ``killed`` (SIGKILL had to escalate), ``running``
    (the daemon's true post-attempt state), ``error``.

    The pid read from ``model.status`` is trusted for the lifetime of this call.
    In the (alpha-acceptable) worst case the daemon exits and the OS recycles its
    pid before the 5 s escalation fires, so SIGKILL could land on an unrelated
    same-user process; the window is small and stop is operator-initiated.
    """
    running, pid = _probe_running(socket_path)
    if not running:
        return {
            "ok": True,
            "was_running": False,
            "pid": None,
            "killed": False,
            "running": False,
            "error": None,
        }

    if pid is not None:
        _signal_pid(pid, signal.SIGTERM)

    if _wait_for_socket(socket_path, present=False, timeout=_STOP_TIMEOUT_S):
        return {
            "ok": True,
            "was_running": True,
            "pid": pid,
            "killed": False,
            "running": False,
            "error": None,
        }

    timeout_message = f"daemon did not stop within {int(_STOP_TIMEOUT_S)} seconds"
    if pid is not None and _pid_alive(pid):
        killed = _signal_pid(pid, signal.SIGKILL)
        suffix = " — sent SIGKILL" if killed else ""
        return {
            "ok": False,
            "was_running": True,
            "pid": pid,
            "killed": killed,
            # SIGKILL delivery is not synchronous with the socket closing, so
            # report the actually-observed state rather than assuming "down".
            "running": _socket_accepting(socket_path),
            "error": {
                "code": _ERR_STOP_TIMEOUT,
                "message": f"{timeout_message}{suffix}",
            },
        }

    # No live pid to signal: the daemon may still be holding the socket (we just
    # have no pid to kill) or it may have exited just past the poll deadline.
    # Re-probe instead of inferring state — and never claim a SIGKILL we did not
    # send.
    if not _socket_accepting(socket_path):
        return {
            "ok": True,
            "was_running": True,
            "pid": pid,
            "killed": False,
            "running": False,
            "error": None,
        }
    return {
        "ok": False,
        "was_running": True,
        "pid": pid,
        "killed": False,
        "running": True,
        "error": {"code": _ERR_STOP_TIMEOUT, "message": timeout_message},
    }


def _probe_running(socket_path: Path) -> tuple[bool, int | None]:
    """Return ``(running, pid)`` without ever lazy-spawning the daemon (AC-4).

    ``pid`` rides on the ``model.status`` payload; ``spawn_if_absent=False`` makes
    an absent daemon raise :class:`LLMDaemonUnreachable` rather than be spawned.
    """
    try:
        payload = LLMClient(socket_path=socket_path).model_status_sync(
            spawn_if_absent=False
        )
    except LLMDaemonUnreachable:
        return False, None
    pid = payload.get("pid")
    return True, int(pid) if pid is not None else None


def _spawn_chirpd(chirpd_path: str, socket_path: Path) -> None:
    """Detach a new ``chirpd`` bound to ``socket_path`` (no parent stdio inheritance)."""
    env = os.environ.copy()
    env[CHIRP_DAEMON_SOCKET_ENV] = str(socket_path)
    subprocess.Popen(
        [chirpd_path],
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_socket(socket_path: Path, *, present: bool, timeout: float) -> bool:
    """Poll until the socket's accept state matches ``present``, or ``timeout``.

    ``present=True`` waits for the socket to start accepting connections (start);
    ``present=False`` waits for it to stop (stop). Returns True if the transition
    occurred within ``timeout`` seconds, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        if _socket_accepting(socket_path) == present:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_SOCKET_POLL_INTERVAL_S)


def _socket_accepting(socket_path: Path) -> bool:
    """True if a unix socket at ``socket_path`` accepts a connection right now."""
    probe = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    try:
        probe.connect(str(socket_path))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _signal_pid(pid: int, sig: int) -> bool:
    """Send ``sig`` to ``pid``; True if delivered, False if the process was gone."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover — daemon is always same-user
        return False
    return True


def _pid_alive(pid: int) -> bool:
    return _signal_pid(pid, 0)


def _format_pid(pid: int | None) -> str:
    """Render a pid for user-facing text — ``model.status`` may omit it."""
    return str(pid) if pid is not None else "unknown"


def _start_exit_code(error: dict[str, Any]) -> int:
    # AC-10: a post-spawn unreachable daemon is the rare exit-3 case; all other
    # start failures (missing binary, spawn timeout) are operational exit-1.
    return 3 if error["code"] == _ERR_UNREACHABLE else 1


def _emit_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _log_lifecycle(
    op: str, start_ns: int, *, result: str, chirpd_path: str | None = None
) -> None:
    duration_ms = max(int((time.perf_counter_ns() - start_ns) / 1_000_000), 0)
    log_op_event(
        _logger,
        logging.INFO,
        f"{op}: {result}",
        req_id=new_request_id(),
        op=op,
        duration_ms=duration_ms,
        result=result,
        chirpd_path=chirpd_path,
    )


# --- LaunchAgent: enable / disable (story 5.4) ------------------------------


@daemon_app.command("enable")
def enable(
    force: bool = typer.Option(
        False, "--force", help="Reinstall even if a LaunchAgent is already present."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON to stdout."
    ),
) -> None:
    """Install a LaunchAgent so chirpd starts at login and restarts on crash."""
    use_json = _should_use_json(json_output)
    _ensure_logging()
    start_ns = time.perf_counter_ns()

    try:
        plist_path = install_launch_agent(force=force)
    except LaunchAgentAlreadyInstalled as err:
        _log_lifecycle(
            "daemon_enable",
            start_ns,
            result="already_installed",
            chirpd_path=shutil.which("chirpd"),
        )
        message = f"LaunchAgent already installed at {err}. Pass --force to reinstall."
        _emit_enable_failure(err, message, use_json)
        raise typer.Exit(code=1) from err
    except LaunchctlFailed as err:
        _log_lifecycle("daemon_enable", start_ns, result="launchctl_failed")
        _emit_enable_failure(err, _format_launchctl_error(err), use_json)
        raise typer.Exit(code=1) from err
    except LaunchAgentError as err:
        result = "missing_binary" if "not found on PATH" in str(err) else "error"
        _log_lifecycle("daemon_enable", start_ns, result=result)
        _emit_enable_failure(err, str(err), use_json)
        raise typer.Exit(code=1) from err

    # Report the path actually baked into the plist (AC-8), not a re-resolution.
    chirpd_path = installed_chirpd_path(plist_path) or shutil.which("chirpd")
    _log_lifecycle(
        "daemon_enable", start_ns, result="installed", chirpd_path=chirpd_path
    )
    if use_json:
        _emit_json(
            {
                "action": "enable",
                "installed": True,
                "plist_path": str(plist_path),
                "chirpd_path": chirpd_path,
            }
        )
    else:
        stdout_console.print(
            f"LaunchAgent installed at {plist_path}\nDaemon will auto-start at login.",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


@daemon_app.command("disable")
def disable(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON to stdout."
    ),
) -> None:
    """Remove the LaunchAgent. Daemon will no longer auto-start at login."""
    use_json = _should_use_json(json_output)
    _ensure_logging()
    start_ns = time.perf_counter_ns()

    try:
        uninstall_launch_agent()
    except LaunchAgentNotInstalled:
        # Idempotent: disabling a clean machine is a no-op, not an error (exit 0).
        _log_lifecycle("daemon_disable", start_ns, result="not_installed")
        if use_json:
            _emit_json(
                {"action": "disable", "installed": False, "was_installed": False}
            )
        else:
            console.print(
                "LaunchAgent is not installed; nothing to do.",
                markup=False,
                soft_wrap=True,
            )
        return
    except LaunchctlFailed as err:
        _log_lifecycle("daemon_disable", start_ns, result="launchctl_failed")
        if use_json:
            _emit_json(_disable_error_payload(err))
        else:
            console.print(
                _format_launchctl_error(err), style="red", markup=False, soft_wrap=True
            )
        raise typer.Exit(code=1) from err
    except LaunchAgentError as err:
        _log_lifecycle("daemon_disable", start_ns, result="error")
        if use_json:
            _emit_json(_disable_error_payload(err))
        else:
            console.print(str(err), style="red", markup=False, soft_wrap=True)
        raise typer.Exit(code=1) from err

    _log_lifecycle("daemon_disable", start_ns, result="removed")
    if use_json:
        _emit_json({"action": "disable", "installed": False, "was_installed": True})
    else:
        stdout_console.print(
            "LaunchAgent removed.\nDaemon will no longer auto-start at login.",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


def _emit_enable_failure(err: LaunchAgentError, message: str, use_json: bool) -> None:
    """Render an ``enable`` failure: JSON error object or a red stderr message."""
    if use_json:
        _emit_json(
            {
                "action": "enable",
                "installed": _safe_is_installed(),
                "error": _error_object(err),
            }
        )
    else:
        console.print(message, style="red", markup=False, soft_wrap=True)


def _disable_error_payload(err: LaunchAgentError) -> dict[str, Any]:
    return {
        "action": "disable",
        "installed": _safe_is_installed(),
        "error": _error_object(err),
    }


def _error_object(err: LaunchAgentError) -> dict[str, Any]:
    return {
        "code": type(err).__name__,
        "message": str(err),
        "launchctl_stderr": getattr(err, "stderr", None),
    }


def _format_launchctl_error(err: LaunchctlFailed) -> str:
    """Multi-line CLI rendering that preserves launchctl's own stderr verbatim."""
    invocation = " ".join(err.command[:2]) if len(err.command) > 1 else err.command[0]
    lines = [f"{invocation} failed (exit {err.returncode}):"]
    stderr = (err.stderr or "").strip()
    lines.extend(f"  {line}" for line in stderr.splitlines())
    lines.append(f"plist at: {LAUNCH_AGENT_PLIST_PATH}")
    return "\n".join(lines)


def _safe_is_installed() -> bool:
    """Best-effort current install state for JSON error payloads."""
    try:
        return is_launch_agent_installed()
    except (OSError, LaunchAgentError):
        return False


# --- logs (story 5.5) -------------------------------------------------------

# Read from module scope so tests can shrink them; production values keep the
# follow loop responsive (200 ms) without burning CPU and poll for a not-yet-
# existing log file once a second (AC-12).
_FOLLOW_POLL_INTERVAL_SECONDS = 0.2
_WAIT_FILE_POLL_INTERVAL_SECONDS = 1.0


@daemon_app.command("logs")
def logs(
    follow: bool = typer.Option(
        False,
        "-f",
        "--follow",
        help="Follow the log file for new output. Re-opens across rotation.",
    ),
    lines: int | None = typer.Option(
        None,
        "-n",
        "--lines",
        min=1,
        help="Show only the last N lines (or last N before following with -f).",
    ),
) -> None:
    """Print the daemon log file. Tail with -f; show last N with -n."""
    _ensure_logging()
    path = resolve_log_path()
    _log_daemon_logs(_logs_result(follow=follow, lines=lines, path=path))

    try:
        if follow:
            # A fresh install may not have a log file yet; the follow loop waits
            # for it to appear, so dump the tail only when there is something to.
            # The follow loop then re-opens and seeks to a freshly-measured EOF,
            # so a line the daemon writes in the microsecond gap between the dump
            # and the seek is in neither — the same benign race `tail -n N -f`
            # carries. Accepted: the alternative (threading the dump's end offset
            # into the follow loop) tangles with the rotation re-open path.
            if lines is not None and path.exists():
                _tail_last_n_lines(path, lines)
            _follow_log_file(path, start_at_end=True)
            return

        if not path.exists():
            print(
                f"no log file at {path} yet. logs are written when the daemon runs.",
                file=sys.stderr,
            )
            return

        if lines is not None:
            _tail_last_n_lines(path, lines)
        else:
            _cat_log_file(path)
    except OSError as err:
        print(f"couldn't read log file: {err}", file=sys.stderr)
        raise typer.Exit(code=1) from err


def _logs_result(*, follow: bool, lines: int | None, path: Path) -> str:
    """Classify the run for the diagnostic log line: cat|tail|follow|missing."""
    if follow:
        return "follow"
    # A missing file short-circuits to the notice before any cat/tail happens, so
    # report "missing" even when -n was passed — the tail never ran.
    if not path.exists():
        return "missing"
    return "tail" if lines is not None else "cat"


def _cat_log_file(path: Path) -> None:
    """Stream the whole file to stdout byte-for-byte (no whole-file read)."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            sys.stdout.write(line)
    sys.stdout.flush()


def _tail_last_n_lines(path: Path, n: int) -> None:
    """Write the last ``n`` lines to stdout, keeping only ``n`` in memory."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        tail = deque(fh, maxlen=n)
    for line in tail:
        sys.stdout.write(line)
    sys.stdout.flush()


def _follow_log_file(path: Path, *, start_at_end: bool) -> None:
    """Follow the log file, re-opening across rotation (``tail -F`` semantics).

    Waits for the file to appear (AC-8), then reads new lines as they land. When
    the file's inode changes — rotation swapped ``chirpd.log`` for a fresh one —
    re-opens by name and reads the new file from the start so the first post-
    rotation lines are not lost. Exits cleanly on Ctrl-C (AC-7).
    """
    try:
        # Seek past existing content only when the file is already there. If we
        # had to wait for it to appear, the file is fresh and the first line is
        # exactly what the user wants to watch arrive (AC-8 rationale).
        seek_to_end = start_at_end and path.exists()
        _wait_for_log_file(path)
        fh = path.open("r", encoding="utf-8", errors="replace")
        # Read the inode off the open handle, not the name: if rotation slips in
        # between open() and the stat, a name-based stat would describe the new
        # file while fh still points at the old one, and the loop would never
        # detect the swap. os.fstat keeps inode and handle consistent.
        inode = os.fstat(fh.fileno()).st_ino
        if seek_to_end:
            fh.seek(0, os.SEEK_END)
        try:
            while True:
                line = fh.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                time.sleep(_FOLLOW_POLL_INTERVAL_SECONDS)
                try:
                    new_inode = path.stat().st_ino
                except FileNotFoundError:
                    # The file vanished mid-rotation; keep polling for its return.
                    continue
                if new_inode != inode:
                    try:
                        new_fh = path.open("r", encoding="utf-8", errors="replace")
                    except OSError:
                        # Lost the race between stat and open (the fresh file is
                        # not openable yet) — hold the current handle and retry.
                        continue
                    fh.close()
                    fh = new_fh
                    # Track the inode of the handle we now hold (see above).
                    inode = os.fstat(fh.fileno()).st_ino
        finally:
            fh.close()
    except KeyboardInterrupt:
        # An interactive ^C during follow is a clean exit, not an error (AC-7).
        pass


def _wait_for_log_file(path: Path) -> None:
    """Block until ``path`` exists, announcing the wait once (AC-8)."""
    announced = False
    while not path.exists():
        if not announced:
            print(f"waiting for log file at {path}...", file=sys.stderr)
            announced = True
        time.sleep(_WAIT_FILE_POLL_INTERVAL_SECONDS)


def _log_daemon_logs(result: str) -> None:
    # AC-13: emitted at command start (not exit), so duration_ms is 0 by design —
    # a ``-f`` run may never "finish" for a duration to measure.
    log_op_event(
        _logger,
        logging.INFO,
        f"daemon logs: {result}",
        req_id=new_request_id(),
        op="daemon_logs",
        duration_ms=0,
        result=result,
    )
