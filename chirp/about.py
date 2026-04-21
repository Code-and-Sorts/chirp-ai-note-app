"""Animated `chirp about` command.

Three-beat reveal, matching the design handoff's live preview:

 1. spinner + status lines while chirp "wakes up"
 2. ASCII chick logo paints in line-by-line
 3. info lines (version, tagline, credits) type out while the beak keeps
    chirping (< ↔ v at ~2 Hz)
"""

from __future__ import annotations

import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from chirp.branding import (
    BEAK_CLOSED,
    BEAK_OPEN,
    LOGO_ACCENT,
    LOGO_ROWS,
    LOGO_YELLOW,
    REPO,
    TAGLINE,
)

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PHASE_ONE = [
    ("waking up the nest...", "loaded {notes_count} notes", 0.7),
    (None, "checked models", 0.18),
    ("tuning the chirp...", "ready", 0.55),
]

CHIRP_CYCLES = 6
CHIRP_INTERVAL = 0.45
LOGO_PAINT_INTERVAL = 0.14
INFO_INTERVAL = 0.1
SPIN_TICK = 1 / 12


def _count_notes(notes_root: Path) -> int:
    try:
        if not notes_root.exists():
            return 0
        return sum(1 for _ in notes_root.glob("*/notes.md"))
    except OSError:
        return 0


def _installed_version() -> str:
    try:
        return pkg_version("chirp-notes-ai")
    except PackageNotFoundError:
        return "dev"


def _prompt_line() -> Text:
    line = Text()
    line.append("$", style="cyan bold")
    line.append(" chirp ")
    line.append("about", style="yellow")
    return line


def _status(icon: str, icon_style: str, body: str) -> Text:
    line = Text()
    line.append(" ")
    line.append(icon, style=icon_style)
    line.append(" ")
    line.append(body, style="dim")
    return line


def _spinner(frame: int, body: str) -> Text:
    return _status(SPINNER_FRAMES[frame % len(SPINNER_FRAMES)], "yellow", body)


def _done(body: str) -> Text:
    return _status("✓", "green", body)


def _logo_line(row_idx: int, beak_open: bool) -> Text:
    """Build a single logo row as Rich Text with the beak correctly coloured."""
    row = LOGO_ROWS[row_idx]
    line = Text()
    line.append(row.prefix, style=LOGO_YELLOW)
    if row.has_beak:
        beak = BEAK_OPEN if beak_open else BEAK_CLOSED
        line.append(beak, style=f"bold {LOGO_ACCENT}")
    if row.suffix:
        line.append(row.suffix, style=LOGO_YELLOW)
    if row.has_note and beak_open:
        line.append("  ♪", style=LOGO_ACCENT)
    return line


def _info_lines(
    version: str,
    notes_count: int,
    chat_model: str,
    embed_model: str,
    notes_root: Path,
) -> list[Text]:
    lines: list[Text] = [Text("")]
    title = Text()
    title.append("   ")
    title.append("chirp", style="bold white")
    title.append(f" v{version}", style="dim")
    lines.append(title)
    lines.append(Text(f"   {TAGLINE}", style="dim"))
    lines.append(Text(""))
    lines.append(Text(f"   • {notes_count} notes at {notes_root}", style="dim"))
    lines.append(Text(f"   • chat:  {chat_model}", style="dim"))
    lines.append(Text(f"   • embed: {embed_model}", style="dim"))
    credit = Text()
    credit.append("   • made with ", style="dim")
    credit.append("♥", style="red")
    credit.append(" by Colby Timm", style="dim")
    lines.append(credit)
    lines.append(Text(f"   • {REPO}", style="dim"))
    lines.append(Text(""))
    return lines


def run_about(
    console: Console,
    settings,
    speed: float = 1.0,
    sleeper: Callable[[float], None] | None = None,
) -> None:
    """Run the full 3-phase about animation.

    Parameters mirror the design's live preview. ``sleeper`` defaults to
    ``time.sleep`` so tests can pass a no-op.
    """
    sleep = sleeper or time.sleep
    scale = 1.0 / max(speed, 0.01)

    notes_root = settings.directories.notes_root
    notes_count = _count_notes(notes_root)
    chat_model = settings.models.llm
    embed_model = settings.notes_chat.emb_model
    version = _installed_version()

    lines: list[Text] = [_prompt_line(), Text("")]

    with Live(Group(*lines), console=console, refresh_per_second=20) as live:
        # Phase 1 — paint the entire bird in one go
        logo_start = len(lines)
        for idx in range(len(LOGO_ROWS)):
            lines.append(_logo_line(idx, beak_open=False))
        live.update(Group(*lines))

        beak_row = next(i for i, row in enumerate(LOGO_ROWS) if row.has_beak)
        beak_idx = logo_start + beak_row

        sleep(0.3 * scale)

        # Phase 2 — spinner + status lines while chirp "wakes up"
        status_start = len(lines)
        for spinner_msg, done_template, dwell in PHASE_ONE:
            done_msg = done_template.format(
                notes_count=notes_count,
                chat_model=chat_model,
                embed_model=embed_model,
            )
            if spinner_msg is None:
                lines.append(_done(done_msg))
                live.update(Group(*lines))
                sleep(dwell * scale)
                continue
            lines.append(_spinner(0, spinner_msg))
            ticks = max(int(dwell / SPIN_TICK), 1)
            for frame in range(ticks):
                lines[-1] = _spinner(frame, spinner_msg)
                live.update(Group(*lines))
                sleep(SPIN_TICK * scale)
            lines[-1] = _done(done_msg)
            live.update(Group(*lines))

        sleep(0.3 * scale)

        # Phase 3 — clear the status lines and reveal info while the beak chirps
        del lines[status_start:]
        live.update(Group(*lines))

        info = _info_lines(version, notes_count, chat_model, embed_model, notes_root)
        open_state = False
        for info_line in info:
            lines.append(info_line)
            open_state = not open_state
            lines[beak_idx] = _logo_line(beak_row, beak_open=open_state)
            live.update(Group(*lines))
            sleep(INFO_INTERVAL * scale)

        for _ in range(CHIRP_CYCLES):
            open_state = not open_state
            lines[beak_idx] = _logo_line(beak_row, beak_open=open_state)
            live.update(Group(*lines))
            sleep(CHIRP_INTERVAL * scale)
