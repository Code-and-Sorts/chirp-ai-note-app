"""LaunchAgent install / uninstall for the ``chirpd`` daemon (macOS only).

This module owns the ``com.chirp.chirpd`` LaunchAgent: writing the plist into
``~/Library/LaunchAgents/``, loading it via ``launchctl``, and reversing both on
uninstall. The two public verbs — :func:`install_launch_agent` and
:func:`uninstall_launch_agent` — plus the introspection helper
:func:`is_launch_agent_installed` are **reusable**: ``chirp daemon enable`` /
``disable`` are thin CLI wrappers (in ``llm/cli/daemon.py``), and
EPIC-INIT-AND-MIGRATION's ``chirp init`` flow imports these functions directly
rather than shelling out to the CLI.

**The plist's ``KeepAlive`` policy is the single most counterintuitive part of
this design.** The plist uses ``KeepAlive = {SuccessfulExit: False}``, which tells
launchd to restart the daemon when it exits **non-zero** (a crash) but **not**
when it exits **zero** (an intentional exit). The daemon exits ``0`` on a
version mismatch (it detects a CLI running newer code and bows out cleanly), so
launchd does **not** respawn it. That is correct: the post-mismatch daemon is
running the wrong code, and the user's next ``chirp`` invocation lazy-spawns the
new version from ``PATH``. A bare ``KeepAlive = True`` would respawn the stale
daemon in a loop and defeat the lazy-upgrade design — hence the dict form.

**PATH propagation.** ``EnvironmentVariables.PATH`` is captured from
``os.environ["PATH"]`` at install time. launchd otherwise hands user agents a
minimal default ``PATH`` (``/usr/bin:/bin:/usr/sbin:/sbin``) that lacks Homebrew,
pyenv shims, and ``uv``'s tool bin — which can break ``mlx-lm``'s subprocess
invocations. ``HF_HOME`` is propagated only when the user has it set (FR38).

**Stale-path refresh.** The plist bakes in the absolute path to ``chirpd``
resolved via ``shutil.which`` at install time. If the user later relocates their
Python environment (deletes the venv, reinstalls via ``uv``), the plist points at
a dead path; re-running ``chirp daemon enable`` (or :func:`install_launch_agent`
with ``force=True``) refreshes it.

The module imports cleanly on non-Darwin — there are no top-level ``launchctl``
calls or platform-only imports — but the mutating functions raise
:class:`LaunchAgentError` when invoked off macOS.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

LAUNCH_AGENT_LABEL: Final[str] = "com.chirp.chirpd"
LAUNCH_AGENT_PLIST_PATH: Final[Path] = (
    Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
)
LAUNCH_AGENT_LOG_PATH: Final[Path] = (
    Path.home() / "Library" / "Logs" / "chirp" / "chirpd.log"
)

_LAUNCHCTL_TIMEOUT_S: Final[float] = 10.0
_MACOS_ONLY_MESSAGE: Final[str] = "LaunchAgent install is macOS-only"


class LaunchAgentError(Exception):
    """Base class for every LaunchAgent failure surfaced to the CLI / init flow."""


class LaunchAgentAlreadyInstalled(LaunchAgentError):
    """A plist is already present and ``install_launch_agent`` was called without ``force``."""


class LaunchAgentNotInstalled(LaunchAgentError):
    """``uninstall_launch_agent`` was called but no plist is present."""


class LaunchctlFailed(LaunchAgentError):
    """A ``launchctl`` invocation returned non-zero (or an unexpected result).

    Carries the exact command, return code, and ``stderr`` so the CLI can surface
    launchctl's own diagnostic verbatim instead of a lossy paraphrase.
    """

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        detail = stderr.strip()
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{' '.join(command)} failed (exit {returncode}){suffix}")


def _require_darwin() -> None:
    if sys.platform != "darwin":
        raise LaunchAgentError(_MACOS_ONLY_MESSAGE)


def _run_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``launchctl <args>`` capturing text output with a hard timeout.

    Normalizes the two ways the call itself can fail into the module's typed
    surface so they never escape as raw ``subprocess`` exceptions: a missing
    ``launchctl`` binary raises :class:`LaunchAgentError`, and a hung call that
    blows the timeout raises :class:`LaunchctlFailed` (returncode ``-1``).
    """
    command = ["launchctl", *args]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_LAUNCHCTL_TIMEOUT_S,
        )
    except FileNotFoundError as err:
        raise LaunchAgentError(
            "launchctl not found — LaunchAgent management requires macOS"
        ) from err
    except subprocess.TimeoutExpired as err:
        raise LaunchctlFailed(
            command, -1, f"launchctl timed out after {_LAUNCHCTL_TIMEOUT_S:g}s"
        ) from err


def _build_plist(chirpd_path: Path) -> dict[str, Any]:
    """Build the plist payload for :func:`plistlib.dump`.

    Private but tested directly. The two conditionals are the only branching:
    ``PATH`` is always captured from the current environment, ``HF_HOME`` only
    when set (omitted entirely otherwise).
    """
    environment: dict[str, str] = {"PATH": os.environ.get("PATH", "")}
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        environment["HF_HOME"] = hf_home

    return {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [str(chirpd_path)],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(LAUNCH_AGENT_LOG_PATH),
        "StandardErrorPath": str(LAUNCH_AGENT_LOG_PATH),
        "EnvironmentVariables": environment,
        "ProcessType": "Background",
    }


def _write_plist_atomic(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` as an XML plist to ``path`` via a ``.tmp`` + ``os.replace``."""
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            plistlib.dump(payload, handle, fmt=plistlib.FMT_XML)
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def install_launch_agent(*, force: bool = False) -> Path:
    """Resolve ``chirpd``, write the plist, and ``launchctl load -w`` it.

    Returns the plist path on success. Raises :class:`LaunchAgentAlreadyInstalled`
    if a plist is already present and ``force`` is False;
    :class:`LaunchctlFailed` if ``launchctl`` reports non-zero (or the post-load
    verification probe comes back empty); :class:`LaunchAgentError` off macOS or
    when ``chirpd`` is not on ``PATH``.
    """
    _require_darwin()

    plist_existed = LAUNCH_AGENT_PLIST_PATH.exists()
    if plist_existed and not force:
        raise LaunchAgentAlreadyInstalled(str(LAUNCH_AGENT_PLIST_PATH))

    chirpd_path = shutil.which("chirpd")
    if chirpd_path is None:
        raise LaunchAgentError(
            "chirpd binary not found on PATH — is chirp installed correctly?"
        )

    payload = _build_plist(Path(chirpd_path))
    LAUNCH_AGENT_PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # launchd opens StandardOut/ErrorPath itself and does NOT create missing
    # parent dirs — on a fresh machine the daemon would fail to start unless the
    # log dir already exists (the Python logging handler from story 5.1 only runs
    # after the process is up, too late for launchd's own redirect).
    LAUNCH_AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_plist_atomic(payload, LAUNCH_AGENT_PLIST_PATH)

    if force and plist_existed:
        # Best-effort: the previously-installed agent may not actually be loaded,
        # so a non-zero exit here is expected and ignored. `launchctl unload`
        # identifies the agent by the Label inside the plist, so the
        # freshly-written plist (same Label) unloads the running old agent.
        _run_launchctl(["unload", str(LAUNCH_AGENT_PLIST_PATH)])

    # ``-w`` marks the agent enabled so it persists across reboots; without it the
    # agent loads now but a reboot silently drops it.
    load = _run_launchctl(["load", "-w", str(LAUNCH_AGENT_PLIST_PATH)])
    if load.returncode != 0:
        raise LaunchctlFailed(load.args, load.returncode, load.stderr)

    listing = _run_launchctl(["list", LAUNCH_AGENT_LABEL])
    if listing.returncode != 0 or not listing.stdout.strip():
        raise LaunchctlFailed(listing.args, listing.returncode, listing.stderr)

    return LAUNCH_AGENT_PLIST_PATH


def uninstall_launch_agent() -> None:
    """``launchctl unload -w`` the agent, then delete the plist file.

    Raises :class:`LaunchAgentNotInstalled` if no plist is present;
    :class:`LaunchctlFailed` if ``unload`` reports non-zero — in which case the
    plist is **left on disk** so the user can inspect and retry rather than land
    in a half-removed state.
    """
    _require_darwin()

    if not LAUNCH_AGENT_PLIST_PATH.exists():
        raise LaunchAgentNotInstalled(str(LAUNCH_AGENT_PLIST_PATH))

    unload = _run_launchctl(["unload", "-w", str(LAUNCH_AGENT_PLIST_PATH)])
    if unload.returncode != 0:
        raise LaunchctlFailed(unload.args, unload.returncode, unload.stderr)

    LAUNCH_AGENT_PLIST_PATH.unlink()

    # The agent should now be absent; a zero exit means it is somehow still
    # loaded (macOS edge case) — surface it rather than silently report success.
    # The plist is already removed at this point (per the AC-4 flow), so say so:
    # `is_launch_agent_installed()` will now report False even though launchctl
    # still lists the agent, and the user should not expect a retry to find it.
    listing = _run_launchctl(["list", LAUNCH_AGENT_LABEL])
    if listing.returncode == 0:
        raise LaunchctlFailed(
            listing.args,
            listing.returncode,
            "launchctl still lists the agent after unload; the plist at "
            f"{LAUNCH_AGENT_PLIST_PATH} has already been removed — unload it "
            f"manually with: launchctl remove {LAUNCH_AGENT_LABEL}",
        )


def installed_chirpd_path(plist_path: Path | None = None) -> str | None:
    """Return the ``chirpd`` path recorded in the plist's ``ProgramArguments``.

    Reads ground truth straight from the plist :func:`install_launch_agent` wrote,
    so callers report the path **actually baked in** (AC-8 — the wrong-Python
    diagnostic) rather than a re-resolution that could drift from it. Returns
    ``None`` if the plist is absent, unreadable, or malformed.
    """
    target = plist_path if plist_path is not None else LAUNCH_AGENT_PLIST_PATH
    try:
        with target.open("rb") as handle:
            payload = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException):
        return None
    args = payload.get("ProgramArguments")
    if isinstance(args, list) and args:
        return str(args[0])
    return None


def is_launch_agent_installed() -> bool:
    """True iff the plist file exists **and** ``launchctl list`` reports the label.

    Returns False (never raises) off macOS or when the plist is absent — the
    plist check short-circuits before any ``launchctl`` call.
    """
    if sys.platform != "darwin":
        return False
    if not LAUNCH_AGENT_PLIST_PATH.exists():
        return False
    try:
        listing = _run_launchctl(["list", LAUNCH_AGENT_LABEL])
    except LaunchAgentError:
        # Honor the never-raises contract: a missing binary or a launchctl
        # timeout means we cannot confirm the agent is loaded → not installed.
        return False
    # exit 0 with empty stdout is the same unexpected state install rejects, so
    # require launchctl to actually echo the agent's plist back.
    return listing.returncode == 0 and bool(listing.stdout.strip())
