"""`chirp init` — smart 3-phase first-run setup.

Phase 0 (gate): require Apple Silicon. Everything downstream (the bundled
chirpd daemon, MLX inference) is arm64-only, so a non-arm64 machine fails
fast with exit code 7 before any other work.

Phase 1: verify homebrew, ffmpeg, daemon readiness (via `llm.client`'s
health handshake, which lazy-spawns chirpd), the registered default chat
model (via `llm.registry`), and the screen-recording permission. Print a
check table and ask whether to install what's missing.

Phase 2: install missing dependencies via Homebrew (macOS only) and rebuild
the capture_audio helper when needed. The daemon is part of the pip package
and is never "installed" here; model registration is the user's own
`chirp models add` step.

Phase 3: finalize — create the config/chroma/notes directories and print
the "your nest is ready" panel.

Re-running is idempotent: each phase is skipped cleanly when nothing is
missing. ``--recheck`` stops after phase 1; ``--switch-model`` flips the
registry's default chat alias.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomli_w
from rich.console import Console
from rich.text import Text

import audio_capture
from config.settings import ChirpSettings

EXIT_NOT_APPLE_SILICON = 7

# Single source of truth for the recommended first model — story 7.2's
# migration plan and the README sweep (7.5) reference these constants.
RECOMMENDED_CHAT_REPO = "mlx-community/gemma-4-4b-it-4bit"
SMALLER_CHAT_REPO = "mlx-community/gemma-4-e2b-it-8bit"

_MODELS_ADD_HINT = f"not configured — run 'chirp models add {RECOMMENDED_CHAT_REPO}'"


@dataclass
class DependencyStatus:
    name: str
    installed: bool
    detail: str
    required: bool = True


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _run(args: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def _require_apple_silicon(console: Console) -> int | None:
    """Phase 0 gate — returns EXIT_NOT_APPLE_SILICON on non-arm64, else None."""
    machine = platform.machine()
    if machine == "arm64":
        return None
    console.print(
        f" [red]chirp requires Apple Silicon (M1/M2/M3/M4 or newer). "
        f"Detected: {machine}.[/red]"
    )
    console.print(
        " [dim]if this Mac is Apple Silicon, you're likely running an x86_64 "
        "Python under Rosetta — install an arm64 Python and reinstall chirp.[/dim]"
    )
    return EXIT_NOT_APPLE_SILICON


def _brew_installed() -> DependencyStatus:
    path = _which("brew")
    if path:
        return DependencyStatus("homebrew", True, path)
    return DependencyStatus(
        "homebrew",
        False,
        "not found — install from https://brew.sh",
        required=platform.system() == "Darwin",
    )


def _ffmpeg_installed() -> DependencyStatus:
    path = _which("ffmpeg")
    if not path:
        return DependencyStatus("ffmpeg", False, "not found")
    code, out = _run([path, "-version"])
    detail = path
    if code == 0:
        first_line = out.splitlines()[0] if out else ""
        if first_line.startswith("ffmpeg version"):
            detail = first_line.split()[2]
    return DependencyStatus("ffmpeg", True, detail)


def _daemon_ready() -> DependencyStatus:
    """Probe daemon readiness through `llm.client`'s health handshake.

    The client owns lazy-spawn and the version handshake — no daemon-process
    logic lives here (architecture §Cross-Component Dependencies). Imports are
    lazy so a broken `llm` module can't make init un-importable.
    """
    from llm.client import LLMClient
    from llm.exceptions import LLMDaemonSpawnFailed, LLMTransportError

    try:
        payload = LLMClient().health_sync()
    except LLMDaemonSpawnFailed:
        return DependencyStatus(
            "chirpd",
            False,
            "daemon could not be started — run 'chirp daemon logs' for details",
        )
    except LLMTransportError as exc:
        return DependencyStatus("chirpd", False, f"daemon unreachable: {exc}")
    version = payload.get("version", "unknown")
    return DependencyStatus("chirpd", True, f"healthy · v{version}")


def _default_chat_registered() -> DependencyStatus:
    """Report whether models.toml has a default chat alias registered.

    A missing or unconfigured registry is the normal first-run state, not a
    failure — the row is non-blocking and points at `chirp models add`.
    """
    from llm.exceptions import LLMError
    from llm.registry import read_registry

    try:
        registry = read_registry()
    except (FileNotFoundError, LLMError):
        return DependencyStatus(
            "default chat model", False, _MODELS_ADD_HINT, required=False
        )
    alias = registry.default_chat
    if alias and alias in registry.models:
        return DependencyStatus("default chat model", True, f"default chat: {alias}")
    return DependencyStatus(
        "default chat model", False, _MODELS_ADD_HINT, required=False
    )


def _screen_recording_permission() -> DependencyStatus:
    if platform.system() != "Darwin":
        return DependencyStatus(
            name="screen recording permission",
            installed=True,
            required=False,
            detail=f"not applicable on {platform.system()}",
        )
    try:
        perms = audio_capture.check_permissions()
    except FileNotFoundError:
        return DependencyStatus(
            name="screen recording permission",
            installed=False,
            detail="capture_audio binary not built — run python -m audio_capture.build",
        )
    except RuntimeError as exc:
        return DependencyStatus(
            name="screen recording permission",
            installed=False,
            detail=f"permission probe failed: {exc} — try rebuilding with python -m audio_capture.build",
        )
    state = perms.get("screen_recording", "undetermined")
    if state == "granted":
        return DependencyStatus(
            name="screen recording permission",
            installed=True,
            detail="granted",
        )
    if state == "denied":
        return DependencyStatus(
            name="screen recording permission",
            installed=False,
            detail="denied — open System Settings → Privacy & Security → Screen Recording",
        )
    if state == "undetermined":
        return DependencyStatus(
            name="screen recording permission",
            installed=True,
            required=False,
            detail="will prompt on first record",
        )
    return DependencyStatus(
        name="screen recording permission",
        installed=False,
        detail=f"unexpected permission state {state!r} — rebuild the helper",
    )


def verify(settings: ChirpSettings, console: Console) -> list[DependencyStatus]:
    """Run phase 1 — returns the ordered status list and prints the table."""
    console.print()
    console.print(" [bold]Welcome to Chirp.[/bold] Let's set up your nest.")
    console.print()
    console.print(" [dim]checking what you've already got...[/dim]")
    console.print()

    statuses = [
        _brew_installed(),
        _ffmpeg_installed(),
        _daemon_ready(),
        _default_chat_registered(),
        _screen_recording_permission(),
    ]

    for status in statuses:
        _print_status(console, status)

    console.print()
    missing_deps = [s for s in statuses if s.required and not s.installed]
    chat_row = next(s for s in statuses if s.name == "default chat model")
    if missing_deps or not chat_row.installed:
        console.print(" [dim]──────────────────────────────────────────────[/dim]")
    if missing_deps:
        plural = "s" if len(missing_deps) > 1 else ""
        console.print(
            f"  need to install: [bold]{len(missing_deps)} piece{plural}[/bold]"
        )
    if not chat_row.installed:
        console.print(
            f"  next step: [bold]chirp models add {RECOMMENDED_CHAT_REPO}[/bold]"
            " [dim](~2 GB, balanced quality and speed)[/dim]"
        )
        console.print(
            f"  [dim]smaller alternative: chirp models add {SMALLER_CHAT_REPO}[/dim]"
        )
    if not missing_deps and chat_row.installed:
        console.print(" [green]everything's already in place.[/green]")

    return statuses


def _print_status(console: Console, status: DependencyStatus) -> None:
    if status.installed:
        icon = "[green]✓[/green]"
    elif status.required:
        icon = "[red]✗[/red]"
    else:
        icon = "[yellow]—[/yellow]"
    label = status.name.ljust(22)
    console.print(f" {icon} {label} [dim]· {status.detail}[/dim]")


def _confirm(console: Console, prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = (
            console.input(f" [bold]{prompt}[/bold] [dim]{suffix}[/dim] ")
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    if not answer:
        return default
    return answer in {"y", "yes"}


def install_missing(console: Console, statuses: list[DependencyStatus]) -> bool:
    """Phase 2 — brew install what's missing. Returns True if everything
    that was required is now installed, False if the user aborted."""
    if platform.system() != "Darwin":
        console.print(
            " [yellow]automatic install is macOS-only.[/yellow] "
            "install the listed pieces manually, then re-run `chirp init --recheck`."
        )
        return False

    brew = _which("brew")
    if not brew:
        console.print(
            " [red]homebrew not found[/red] — install from https://brew.sh, then re-run."
        )
        return False

    console.print()
    console.print(" [dim]installing dependencies via homebrew...[/dim]")
    console.print()

    user_action_required = False
    tasks: list[tuple[str, list[str]]] = []
    for status in statuses:
        if status.installed or not status.required:
            continue
        if status.name == "chirpd":
            # The daemon ships inside the pip package and is lazy-spawned by
            # llm.client — there is nothing to install when it fails to start.
            console.print(f" [red]✗[/red] the daemon failed to start — {status.detail}")
            console.print("   run [bold]chirp daemon logs[/bold] for details.")
            user_action_required = True
        elif status.name == "ffmpeg":
            tasks.append(("ffmpeg", [brew, "install", "ffmpeg"]))
        elif status.name == "screen recording permission" and status.detail.startswith(
            "denied"
        ):
            console.print(
                "[yellow]![/yellow] screen recording permission must be granted manually — "
                "open System Settings → Privacy & Security → Screen Recording, then re-run."
            )
            user_action_required = True
        elif status.name == "screen recording permission":
            tasks.append(
                ("capture_audio", [sys.executable, "-m", "audio_capture.build"])
            )

    for label, args in tasks:
        with console.status(f"[yellow]⠹[/yellow] {label} — installing..."):
            code, out = _run(args, timeout=600)
        if code == 0:
            console.print(f" [green]✓[/green] {' '.join(args)}")
        else:
            console.print(f" [red]✗[/red] {' '.join(args)}")
            console.print(
                f"   [dim]{out.strip().splitlines()[-1] if out else ''}[/dim]"
            )
            return False

    if user_action_required:
        console.print(
            "[red]init incomplete[/red] — resolve the items above and re-run."
        )
        return False

    return True


def _run_switch_model(settings: ChirpSettings, console: Console) -> int:
    """``--switch-model`` — flip the registry's default chat alias.

    Calls the same `llm.registry` functions `chirp models default` wraps —
    no subprocess hop.
    """
    from llm.exceptions import LLMError
    from llm.registry import read_registry, set_default_for_role, write_registry

    try:
        registry = read_registry()
        chat_aliases = sorted(
            alias for alias, entry in registry.models.items() if entry.role == "chat"
        )
    except (FileNotFoundError, LLMError):
        chat_aliases = []
    if not chat_aliases:
        console.print(
            f" [yellow]no chat model registered yet.[/yellow] run: "
            f"[bold]chirp models add {RECOMMENDED_CHAT_REPO}[/bold]"
        )
        console.print(
            f" [dim]smaller alternative: chirp models add {SMALLER_CHAT_REPO}[/dim]"
        )
        return 0

    console.print(" [bold]registered chat models:[/bold]")
    for idx, alias in enumerate(chat_aliases, start=1):
        marker = (
            " [#d97a3a]★ current default[/#d97a3a]"
            if alias == registry.default_chat
            else ""
        )
        console.print(f"   {idx}. {alias}{marker}")
    try:
        choice = console.input(
            " [green]›[/green] [dim]pick a new default (blank keeps current):[/dim] "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return 1
    if not choice:
        return 0
    if not (choice.isdigit() and 1 <= int(choice) <= len(chat_aliases)):
        console.print(" [red]invalid selection.[/red]")
        return 1
    chosen = chat_aliases[int(choice) - 1]
    write_registry(set_default_for_role(registry, chosen))
    console.print(f" [green]✓[/green] default chat set to [bold]{chosen}[/bold]")
    return 0


def _finalize_paths(settings: ChirpSettings, console: Console) -> None:
    """Phase 3 — ensure config/chroma/notes paths exist, print the summary."""
    config_path = ChirpSettings.get_config_path()
    chroma_dir = settings.notes_chat.index_dir / "chroma"
    notes_root = settings.directories.notes_root

    config_existed = config_path.exists()
    chroma_existed = chroma_dir.exists()
    notes_existed = notes_root.exists()

    if not config_existed:
        _merge_config(config_path, console=console)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    notes_root.mkdir(parents=True, exist_ok=True)

    _print_path_summary(
        console,
        config_path=config_path,
        config_changed=not config_existed,
        chroma_dir=chroma_dir,
        chroma_was_new=not chroma_existed,
        notes_root=notes_root,
        notes_was_new=not notes_existed,
    )

    console.print()
    console.print(" [bold]your nest is ready.[/bold] try:")
    console.print("   [dim]$[/dim] chirp record")
    console.print(
        '   [dim]$[/dim] chirp ask [yellow]"what did I decide last week?"[/yellow]'
    )


def _merge_config(config_path: Path, console: Console | None = None) -> None:
    """Write config.toml preserving any user keys already present.

    Model selection lives in models.toml (the registry); init only touches
    the user-editable settings file. On a corrupt file we copy the original
    aside as ``config.toml.bak-<ts>`` and warn loudly before writing a fresh
    config — never silently clobber hand-edited keys.
    """
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            with config_path.open("rb") as fh:
                existing = dict(tomllib.load(fh))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            backup = _backup_unparsable_config(config_path)
            if console is not None:
                console.print(
                    f"[yellow]⚠ {config_path.name} could not be parsed "
                    f"({exc}); backed up to {backup} and writing a fresh "
                    "config from defaults. Re-add any custom keys from the "
                    "backup.[/yellow]"
                )
            existing = {}

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("wb") as fh:
        tomli_w.dump(existing, fh)


def _backup_unparsable_config(config_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_name(f"{config_path.name}.bak-{timestamp}")
    config_path.rename(backup)
    return backup


def _print_path_summary(
    console: Console,
    *,
    config_path: Path,
    config_changed: bool,
    chroma_dir: Path,
    chroma_was_new: bool,
    notes_root: Path,
    notes_was_new: bool,
) -> None:
    console.print()
    console.print(
        _path_line(
            "saved config to",
            config_path,
            created_or_changed=config_changed,
        )
    )
    console.print(
        _path_line(
            "initialized chromadb at",
            chroma_dir,
            created_or_changed=chroma_was_new,
        )
    )
    console.print(
        _path_line(
            "notes dir:",
            notes_root,
            created_or_changed=notes_was_new,
        )
    )


def _path_line(
    label: str,
    path: Path,
    *,
    created_or_changed: bool,
) -> Text:
    line = Text(" ")
    if created_or_changed:
        line.append("✓", style="green")
    else:
        line.append("·", style="dim")
    line.append(f" {label} ")
    line.append(str(path), style="bold")
    return line


def _try_import_ollama_module() -> bool:
    try:
        import ollama  # noqa: F401
    except ImportError:
        return False
    return True


def _detect_ollama_install() -> dict[str, bool]:
    """Heuristics for a leftover Ollama install (OR'd; any one is enough)."""
    return {
        "cli_on_path": shutil.which("ollama") is not None,
        "data_dir_present": (Path.home() / ".ollama").is_dir(),
        "python_module_importable": _try_import_ollama_module(),
    }


_DETECTION_LABELS = {
    "cli_on_path": "[ollama on PATH]",
    "data_dir_present": "[~/.ollama directory]",
    "python_module_importable": "[ollama python module]",
}


def _print_migration_plan(console: Console, detection: dict[str, bool]) -> None:
    """Print the loud, informational-only Ollama migration plan.

    PRD §Project Scoping → Out of Scope forbids auto-uninstalling Ollama —
    cleanup commands are shown for the user to run, never executed. OQ6 is
    resolved to a loud multi-line plan (Devon's journey): offline-friendly
    and explicit beats a one-line pointer at an external URL.
    """
    rule = " [dim]──────────────────────────────────────────────────────────[/dim]"
    console.print()
    console.print(rule)
    console.print(" [bold]Ollama migration[/bold]")
    console.print(rule)
    console.print(
        " chirp 2.x no longer uses Ollama. The bundled chirpd daemon\n"
        " (visible in the verify table above) replaces it for chat\n"
        " and embeddings."
    )
    console.print()
    console.print(" Your existing chirp data is unchanged:")
    console.print(
        "   · ~/Documents/chirp/<slug>/        [dim](notes, transcripts, audio)[/dim]"
    )
    console.print("   · ~/.chirp/config.toml             [dim](settings)[/dim]")
    console.print("   · ~/.chirp/chroma/                 [dim](search index)[/dim]")
    console.print()
    console.print(" To finish the migration:")
    console.print("   1. Pick an MLX model [dim](GGUF models do not work)[/dim]:")
    console.print(f"        [bold]chirp models add {RECOMMENDED_CHAT_REPO}[/bold]")
    console.print("      [dim](~2 GB, balanced quality and speed)[/dim]")
    console.print()
    console.print("      Tight on RAM? Use the smaller-footprint variant:")
    console.print(f"        [bold]chirp models add {SMALLER_CHAT_REPO}[/bold]")
    console.print("   2. (Optional) Once you're satisfied with the new setup,")
    console.print(
        "      clean up Ollama manually. chirp will [bold]NOT[/bold] do this for you:"
    )
    console.print(
        "        [dim]brew uninstall ollama        # if installed via Homebrew[/dim]"
    )
    console.print(
        "        [dim]rm -rf ~/.ollama             # reclaims ~5 GB of GGUF files[/dim]"
    )
    console.print(
        "        [dim]unset OLLAMA_HOST            # if set in your shell rc[/dim]"
    )
    console.print()
    console.print(" Detected on this machine:")
    for key, label in _DETECTION_LABELS.items():
        answer = "yes" if detection.get(key) else "no"
        # markup=False: the bracketed labels would otherwise parse as Rich tags.
        console.print(f"   {label.ljust(28)}{answer}", markup=False)
    console.print(rule)


def run_init(
    settings: ChirpSettings,
    console: Console,
    recheck: bool = False,
    switch_model: bool = False,
) -> int:
    """Top-level entry — orchestrates the phases. Returns exit code."""
    code = _require_apple_silicon(console)
    if code is not None:
        return code

    if switch_model:
        return _run_switch_model(settings, console)

    statuses = verify(settings, console)
    if recheck:
        # --recheck is the migration touchpoint (Devon's journey); full init
        # is the fresh-setup flow and stays free of migration noise.
        detection = _detect_ollama_install()
        if any(detection.values()):
            _print_migration_plan(console, detection)
        return 0

    missing = [s for s in statuses if s.required and not s.installed]
    if missing:
        if not _confirm(console, "Install the missing pieces?", default=True):
            console.print(" [yellow]skipped install.[/yellow]")
            return 1
        if not install_missing(console, statuses):
            return 1
    else:
        console.print(
            " [dim]phase 2 · everything's already installed — skipping.[/dim]"
        )

    _finalize_paths(settings, console)
    return 0
