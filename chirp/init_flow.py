"""`chirp init` — smart 4-phase first-run setup.

Phase 1: verify homebrew, ffmpeg, BlackHole 2ch, Ollama, and the configured
models. Print a check table and ask whether to install what's missing.

Phase 2: install missing dependencies via Homebrew (macOS only) and start
Ollama. Offer to open Audio MIDI Setup for the Multi-Output Device step.

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
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

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


def _blackhole_installed() -> DependencyStatus:
    if platform.system() != "Darwin":
        return DependencyStatus(
            "BlackHole 2ch",
            True,
            "skipped — not macOS",
            required=False,
        )
    code, out = _run(["system_profiler", "SPAudioDataType"], timeout=8.0)
    if code == 0 and "BlackHole" in out:
        return DependencyStatus("BlackHole 2ch", True, "installed")
    return DependencyStatus(
        "BlackHole 2ch",
        False,
        "not found — needed for system-audio capture",
    )


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
    except Exception:
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
    except Exception:
        return []


def _model_installed(tag: str, available: list[str]) -> bool:
    if tag in available:
        return True
    if f"{tag}:latest" in available:
        return True
    base = tag.split(":", 1)[0]
    return any(name.startswith(f"{base}:") for name in available)


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
        _blackhole_installed(),
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

    for status in statuses:
        _print_status(console, status)

    console.print()
    missing = [s for s in statuses if s.required and not s.installed]
    if missing:
        console.print(" [dim]──────────────────────────────────────────────[/dim]")
        plural = "s" if len(missing) > 1 else ""
        console.print(f"  need to install: [bold]{len(missing)} piece{plural}[/bold]")
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

    tasks: list[tuple[str, list[str]]] = []
    for status in statuses:
        if status.installed or not status.required:
            continue
        if status.name == "BlackHole 2ch":
            tasks.append(
                ("blackhole-2ch", [brew, "install", "--cask", "blackhole-2ch"])
            )
        elif status.name == "Ollama":
            tasks.append(("ollama", [brew, "install", "ollama"]))
            tasks.append(("ollama service", [brew, "services", "start", "ollama"]))
        elif status.name == "ffmpeg":
            tasks.append(("ffmpeg", [brew, "install", "ffmpeg"]))

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

    if any(s.name == "BlackHole 2ch" and not s.installed for s in statuses):
        console.print()
        console.print(
            " [yellow bold]![/yellow bold] BlackHole needs a one-time audio routing step:"
        )
        console.print("   open [bold]Audio MIDI Setup[/bold] and create a")
        console.print("   [bold]Multi-Output Device[/bold] = Speakers + BlackHole 2ch")
        if _confirm(console, "open Audio MIDI Setup now?", default=True):
            _run(["open", "-a", "Audio MIDI Setup"])
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
        choice = console.input(
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
            custom = console.input(" [green]›[/green] [dim]ollama tag:[/dim] ").strip()
            return custom or options[0].tag
    return choice


def pull_and_finalize(
    console: Console,
    settings: ChirpSettings,
    chat_tag: str,
    embed_tag: str,
) -> None:
    """Phase 4 — pull the picked models, save config, init chroma."""
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

    settings.models.llm = chat_tag
    settings.notes_chat.emb_model = embed_tag
    config_path = ChirpSettings.get_config_path()
    settings.save_to_file(config_path)
    settings.ensure_directories_exist()

    console.print()
    console.print(f" [green]✓[/green] saved config to [bold]{config_path}[/bold]")
    console.print(
        f" [green]✓[/green] initialized chromadb at [bold]{settings.notes_chat.index_dir / 'chroma'}[/bold]"
    )
    console.print(
        f" [green]✓[/green] notes dir: [bold]{settings.directories.notes}[/bold]"
    )
    console.print()
    console.print(" [bold]your nest is ready.[/bold] try:")
    console.print("   [dim]$[/dim] chirp record")
    console.print(
        '   [dim]$[/dim] chirp ask [yellow]"what did I decide last week?"[/yellow]'
    )


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
        assert proc.stdout is not None
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
        pull_and_finalize(console, settings, chat, embed)
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

    ollama = _ollama_installed()
    if not ollama.installed or "running" not in ollama.detail:
        console.print(
            " [red]ollama isn't running.[/red] start it with `brew services start ollama`, then re-run."
        )
        return 1

    chat, embed = pick_models(console)
    pull_and_finalize(console, settings, chat, embed)
    return 0
