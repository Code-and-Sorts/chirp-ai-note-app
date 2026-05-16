"""`chirp init` — smart 4-phase first-run setup.

Phase 1: verify homebrew, ffmpeg, Ollama, and the configured models. Print a
check table and ask whether to install what's missing.

Phase 2: install missing dependencies via Homebrew (macOS only) and start
Ollama.

Phase 3: let the user pick chat + embedding models from a short list, with
sensible defaults (``llama3.1:8b``, ``nomic-embed-text``).

Phase 4: ``ollama pull`` the models with progress bars, persist the config,
and initialize ChromaDB.

Re-running is idempotent: each phase is skipped cleanly when nothing is
missing. ``--recheck`` stops after phase 1; ``--switch-model`` jumps to
phase 3.
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
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

import audio_capture
from config.settings import ChirpSettings


@dataclass
class DependencyStatus:
    name: str
    installed: bool
    detail: str
    required: bool = True


@dataclass
class ModelOption:
    tag: str
    size: str
    note: str


CHAT_MODELS = [
    ModelOption("llama3.1:8b", "4.7 GB", "recommended"),
    ModelOption("qwen2.5:7b", "4.4 GB", "faster"),
    ModelOption("phi3:mini", "2.3 GB", "low RAM"),
]

EMBEDDING_MODELS = [
    ModelOption("nomic-embed-text", "274 MB", "recommended"),
    ModelOption("mxbai-embed-large", "669 MB", "higher recall"),
    ModelOption("all-minilm", "46 MB", "lightweight"),
]


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _run(args: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


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


def _ollama_installed() -> DependencyStatus:
    path = _which("ollama")
    if not path:
        return DependencyStatus("Ollama", False, "not found — runs your local models")
    import requests

    try:
        resp = requests.get("http://localhost:11434/api/version", timeout=3)
        if resp.status_code == 200:
            return DependencyStatus(
                "Ollama",
                True,
                f"running · {resp.json().get('version', 'unknown')}",
            )
        return DependencyStatus(
            "Ollama",
            True,
            "installed but not responding on :11434",
        )
    except requests.exceptions.RequestException:
        return DependencyStatus(
            "Ollama",
            True,
            "installed · not running (brew services start ollama)",
        )


def _ollama_models() -> list[str]:
    import requests

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            return []
        return [m["name"] for m in resp.json().get("models", [])]
    except requests.exceptions.RequestException:
        return []


def _model_installed(tag: str, available: list[str]) -> bool:
    if tag in available:
        return True
    if f"{tag}:latest" in available:
        return True
    base = tag.split(":", 1)[0]
    return any(name.startswith(f"{base}:") for name in available)


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
        _ollama_installed(),
    ]

    ollama_up = statuses[-1].installed and "running" in statuses[-1].detail
    available = _ollama_models() if ollama_up else []

    chat_tag = settings.models.llm
    emb_tag = settings.notes_chat.emb_model

    if ollama_up:
        statuses.append(
            DependencyStatus(
                f"model: {chat_tag}",
                _model_installed(chat_tag, available),
                "installed" if _model_installed(chat_tag, available) else "missing",
            )
        )
        statuses.append(
            DependencyStatus(
                f"model: {emb_tag}",
                _model_installed(emb_tag, available),
                "installed" if _model_installed(emb_tag, available) else "missing",
            )
        )
    else:
        statuses.append(
            DependencyStatus(
                "models",
                False,
                "will check after ollama is installed",
                required=False,
            )
        )

    statuses.append(_screen_recording_permission())

    for status in statuses:
        _print_status(console, status)

    console.print()
    missing_deps = [
        s
        for s in statuses
        if s.required and not s.installed and not s.name.startswith("model:")
    ]
    missing_models = [
        s
        for s in statuses
        if s.required and not s.installed and s.name.startswith("model:")
    ]
    if missing_deps or missing_models:
        console.print(" [dim]──────────────────────────────────────────────[/dim]")
        if missing_deps:
            plural = "s" if len(missing_deps) > 1 else ""
            console.print(
                f"  need to install: [bold]{len(missing_deps)} piece{plural}[/bold]"
            )
        if missing_models:
            plural = "s" if len(missing_models) > 1 else ""
            console.print(
                f"  missing model{plural}: "
                f"[bold]{len(missing_models)}[/bold] [dim](pulled in phase 4)[/dim]"
            )
    else:
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

    denied_user_action_required = False
    tasks: list[tuple[str, list[str]]] = []
    for status in statuses:
        if status.installed or not status.required:
            continue
        if status.name == "Ollama":
            tasks.append(("ollama", [brew, "install", "ollama"]))
            tasks.append(("ollama service", [brew, "services", "start", "ollama"]))
        elif status.name == "ffmpeg":
            tasks.append(("ffmpeg", [brew, "install", "ffmpeg"]))
        elif status.name == "screen recording permission" and status.detail.startswith(
            "denied"
        ):
            console.print(
                "[yellow]![/yellow] screen recording permission must be granted manually — "
                "open System Settings → Privacy & Security → Screen Recording, then re-run."
            )
            denied_user_action_required = True
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

    if denied_user_action_required:
        console.print(
            "[red]init incomplete[/red] — grant screen recording permission and re-run."
        )
        return False

    return True


def pick_models(console: Console) -> tuple[str, str]:
    """Phase 3 — show the two picker boxes, return (chat_tag, embed_tag)."""
    console.print()
    console.print(" [dim]ollama is running. let's pick your models.[/dim]")
    console.print()
    chat = _pick(console, "Chat / notes generator", CHAT_MODELS)
    console.print()
    embed = _pick(
        console, "Embedding model (for ChromaDB · RAG search)", EMBEDDING_MODELS
    )
    return chat, embed


def _pick(console: Console, title: str, options: list[ModelOption]) -> str:
    console.print(f" [bold]{title}[/bold]")
    for idx, option in enumerate(options, start=1):
        marker = "[#d97a3a]●[/#d97a3a]" if idx == 1 else "[dim]○[/dim]"
        tag = f"[bold]{option.tag}[/bold]" if idx == 1 else f"[dim]{option.tag}[/dim]"
        console.print(
            f"   {marker} {idx}. {tag}  [dim]{option.size} · {option.note}[/dim]"
        )
    console.print(f"   [dim]{len(options) + 1}. custom… (type an ollama tag)[/dim]")
    try:
        choice: str = console.input(
            " [green]›[/green] [dim]pick one (default 1):[/dim] "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return options[0].tag
    if not choice:
        return options[0].tag
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1].tag
        if idx == len(options) + 1:
            custom: str = console.input(
                " [green]›[/green] [dim]ollama tag:[/dim] "
            ).strip()
            return custom or options[0].tag
    return choice


def keep_or_pick(console: Console, settings: ChirpSettings) -> tuple[str, str, bool]:
    """Phase 3 — show "keep these or pick new?" when current models are valid.

    Returns ``(chat_tag, embed_tag, models_changed)``. ``models_changed`` is
    False when the user kept the existing models and Phase 4 should leave
    ``models.*`` in ``config.toml`` untouched.
    """
    current_chat = settings.models.llm
    current_embed = settings.notes_chat.emb_model
    available = _ollama_models()
    current_present = (
        bool(current_chat)
        and bool(current_embed)
        and _model_installed(current_chat, available)
        and _model_installed(current_embed, available)
    )

    if current_present:
        console.print()
        console.print(f" [bold]current models:[/bold] {current_chat}, {current_embed}")
        try:
            answer = (
                console.input(
                    "   [bold]keep these, or pick new?[/bold] [dim][K/p][/dim] "
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            console.print()
            return current_chat, current_embed, False
        if answer in ("", "k", "keep"):
            return current_chat, current_embed, False

    chat, embed = pick_models(console)
    return chat, embed, True


def pull_and_finalize(
    console: Console,
    settings: ChirpSettings,
    chat_tag: str,
    embed_tag: str,
    models_changed: bool = True,
) -> None:
    """Phase 4 — pull the picked models, merge config.toml, ensure dirs."""
    installed = _ollama_models()
    to_pull = [
        tag for tag in (chat_tag, embed_tag) if not _model_installed(tag, installed)
    ]
    if to_pull:
        console.print()
        console.print(" [dim]pulling models...[/dim]")
        for tag in to_pull:
            _pull_model(console, tag)
    else:
        console.print(" [green]models already present.[/green]")

    config_path = ChirpSettings.get_config_path()
    chroma_dir = settings.notes_chat.index_dir / "chroma"
    notes_root = settings.directories.notes_root

    config_existed = config_path.exists()
    chroma_existed = chroma_dir.exists()
    notes_existed = notes_root.exists()

    settings.models.llm = chat_tag
    settings.notes_chat.emb_model = embed_tag

    config_written = False
    if models_changed or not config_existed:
        _merge_config(config_path, chat_tag, embed_tag, console=console)
        config_written = True

    chroma_dir.mkdir(parents=True, exist_ok=True)
    notes_root.mkdir(parents=True, exist_ok=True)

    _print_path_summary(
        console,
        config_path=config_path,
        config_changed=config_written,
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


def _merge_config(
    config_path: Path,
    chat_tag: str,
    embed_tag: str,
    console: Console | None = None,
) -> None:
    """Merge models.* fields into config.toml without dropping user keys.

    On a corrupt file we copy the original aside as ``config.toml.bak-<ts>``
    and warn loudly before writing a fresh config — never silently clobber
    hand-edited keys.
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

    existing.setdefault("models", {})["llm"] = chat_tag
    existing.setdefault("notes_chat", {})["emb_model"] = embed_tag

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


def _pull_model(console: Console, tag: str) -> None:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task = None
    with progress:
        task = progress.add_task(f"ollama pull {tag}", total=100)
        try:
            proc = subprocess.Popen(
                ["ollama", "pull", tag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            progress.update(task, description="[red]✗[/red] ollama not on PATH")
            return
        if proc.stdout is None:
            proc.wait()
            return
        for line in proc.stdout:
            pct = _parse_percent(line)
            if pct is not None:
                progress.update(task, completed=pct)
        proc.wait()
        if proc.returncode == 0:
            progress.update(task, completed=100)
        else:
            progress.update(
                task,
                description=f"[red]✗[/red] ollama pull {tag} failed",
            )


def _parse_percent(line: str) -> float | None:
    line = line.strip()
    if "%" not in line:
        return None
    for token in line.split():
        if token.endswith("%"):
            try:
                return float(token.rstrip("%"))
            except ValueError:
                return None
    return None


def run_init(
    settings: ChirpSettings,
    console: Console,
    recheck: bool = False,
    switch_model: bool = False,
) -> int:
    """Top-level entry — orchestrates the four phases. Returns exit code."""
    if switch_model:
        chat, embed = pick_models(console)
        pull_and_finalize(console, settings, chat, embed, models_changed=True)
        return 0

    statuses = verify(settings, console)
    if recheck:
        return 0

    missing = [
        s
        for s in statuses
        if s.required and not s.installed and not s.name.startswith("model:")
    ]
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

    ollama = _ollama_installed()
    if not ollama.installed or "running" not in ollama.detail:
        console.print(
            " [red]ollama isn't running.[/red] start it with `brew services start ollama`, then re-run."
        )
        return 1

    chat, embed, models_changed = keep_or_pick(console, settings)
    pull_and_finalize(console, settings, chat, embed, models_changed=models_changed)
    return 0
