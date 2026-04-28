import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from typer.core import TyperGroup

from chirp.exceptions import AudioDeviceError, ConfigurationError, RecordingError
from config.settings import ChirpSettings, get_settings
from utils.file_utils import NoteRecord, list_notes

VISIBLE_COMMAND_ORDER = (
    "record",
    "transcribe",
    "notes",
    "ask",
    "search",
    "init",
    "about",
)


class OrderedCommandsGroup(TyperGroup):
    def list_commands(self, ctx):
        ordered = [name for name in VISIBLE_COMMAND_ORDER if name in self.commands]
        for name in self.commands:
            if name not in ordered:
                ordered.append(name)
        return ordered


app = typer.Typer(
    name="chirp",
    help="Chirp · AI notes for your terminal.",
    epilog="run `chirp COMMAND --help` for details",
    rich_markup_mode="rich",
    add_completion=False,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
    },
    cls=OrderedCommandsGroup,
)
console = Console()

MAIN_PANEL = "Commands"


def _prompt_title() -> str:
    console.print()
    console.print(" [yellow bold]title[/yellow bold] [dim](required)[/dim]")
    while True:
        try:
            value: str = console.input(" [green]›[/green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            raise typer.Exit(1)
        if value:
            return value
        console.print(" [dim]title can't be empty[/dim]")


def _prompt_timeframe() -> int | None:
    from utils.time_utils import parse_timeframe

    console.print()
    console.print(
        " [yellow bold]timeframe[/yellow bold] "
        "[dim](optional · e.g. 30s / 5m / 1h · press ⏎ to skip)[/dim]"
    )
    try:
        value = console.input(" [green]›[/green] ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None
    if not value:
        return None
    return parse_timeframe(value)


def _prompt_tags() -> list[str]:
    console.print()
    console.print(
        " [yellow bold]tags[/yellow bold] "
        "[dim](optional · comma-separated · e.g. meeting, pricing)[/dim]"
    )
    try:
        value = console.input(" [green]›[/green] ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return []
    return _parse_tag_input(value)


def _parse_tag_input(value: str) -> list[str]:
    if not value:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _resolve_mic_name(device_manager) -> str:
    try:
        default_idx = device_manager.get_default_input_device()
        devices_info = device_manager.list_devices()
        return next(
            (d["name"] for d in devices_info if d["index"] == default_idx),
            "default",
        )
    except Exception:
        return "default"


WAVEFORM_GLYPHS = "▁▂▄▅▇█"
WAVEFORM_WIDTH = 28


def _render_waveform_box(levels: "deque[float]") -> RenderableType:
    bar = Text()
    for slot in levels:
        if slot <= 0:
            bar.append("▁", style="dim")
            continue
        glyph_idx = min(int(slot * len(WAVEFORM_GLYPHS)), len(WAVEFORM_GLYPHS) - 1)
        bar.append(WAVEFORM_GLYPHS[glyph_idx], style="cyan")
    return Panel(bar, title="waveform", title_align="left", border_style="dim")


def _format_elapsed(seconds: float) -> str:
    total = int(max(seconds, 0))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _new_level_buffer() -> "deque[float]":
    return deque([0.0] * WAVEFORM_WIDTH, maxlen=WAVEFORM_WIDTH)


def _new_listening_spinner() -> Spinner:
    return Spinner("dots", text=Text(" listening...", style="cyan"))


@dataclass
class _RecordViewState:
    title: str | None
    cap_minutes: int | None
    mic_name: str
    elapsed_seconds: float = 0.0
    paused: bool = False
    stopped_by_cap: bool = False
    levels: "deque[float]" = field(default_factory=_new_level_buffer)
    listening_spinner: Spinner = field(default_factory=_new_listening_spinner)

    def push_level(self, level: float) -> None:
        self.levels.append(level)


def _render_record_view(state: _RecordViewState) -> RenderableType:
    lines: list[RenderableType] = []
    if state.title:
        lines.append(Text(f" [{state.title}]", style="cyan"))

    header = Text()
    header.append(" ● ", style="red bold")
    header.append("REC", style="bold white")
    header.append("  ·  ")
    header.append(_format_elapsed(state.elapsed_seconds))
    if state.cap_minutes:
        header.append(f" / {state.cap_minutes:02d}:00")
    header.append(f"  ·  mic: {state.mic_name}", style="dim")
    lines.append(header)

    lines.append(Text(""))
    lines.append(_render_waveform_box(state.levels))

    status: RenderableType
    if state.stopped_by_cap:
        status = Text(" ■ stopped", style="dim")
    elif state.paused:
        status = Text(" ⏸ paused", style="yellow")
    else:
        status = state.listening_spinner
    lines.append(status)

    lines.append(Text(""))
    lines.append(
        Text(
            " [space] pause   [q / ^C] stop & save   [x] discard",
            style="dim",
        )
    )
    return Group(*lines)


@app.command(rich_help_panel=MAIN_PANEL)
def record(
    duration: int | None = typer.Option(
        None,
        "--duration",
        "-d",
        help="Recording duration in minutes (press Ctrl+C to stop if not specified)",
    ),
    title: str | None = typer.Option(
        None, "--title", "-t", help="Meeting title for filename"
    ),
    timeframe: str | None = typer.Option(
        None,
        "--timeframe",
        help="Timeframe like 30s / 5m / 1h (auto-stops when reached)",
    ),
    tags: list[str] = typer.Option(
        None,
        "--tag",
        help="Tag to attach to the note (repeatable). Skips the tags prompt.",
    ),
    live_transcribe: bool = typer.Option(
        False,
        "--live-transcribe/--no-live-transcribe",
        help="Stream live transcription while recording",
    ),
    debug_live: bool = typer.Option(
        False,
        "--debug-live",
        help="Debug live transcription (captures intermediate audio chunks)",
        hidden=True,
    ),
):
    """Capture audio to a new note"""
    import shutil
    from datetime import datetime

    from recorder.audio_recorder import AudioRecorder
    from recorder.device_manager import DeviceManager
    from utils.time_utils import parse_timeframe

    settings = get_settings()
    device_manager = DeviceManager()

    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if title is None and is_tty:
        title = _prompt_title()
    if duration is None and timeframe is None and is_tty:
        try:
            duration = _prompt_timeframe()
        except ValueError as exc:
            console.print(f"[red]❌ {exc}[/red]")
            raise typer.Exit(1)
    elif timeframe is not None and duration is None:
        try:
            duration = parse_timeframe(timeframe)
        except ValueError as exc:
            console.print(f"[red]❌ {exc}[/red]")
            raise typer.Exit(1)

    resolved_tags: list[str] = list(tags) if tags else []
    if not resolved_tags and is_tty:
        resolved_tags = _prompt_tags()

    if live_transcribe:
        _run_live_transcription(
            settings,
            device_manager,
            title,
            duration,
            debug_live=debug_live,
            tags=resolved_tags,
        )
        return

    recorder = AudioRecorder(settings, device_manager)
    mic_name = _resolve_mic_name(device_manager)
    state = _RecordViewState(
        title=title,
        cap_minutes=duration,
        mic_name=mic_name,
    )
    control = {"discard": False}

    try:
        use_cbreak = sys.stdin.isatty() and hasattr(sys.stdin, "fileno")

        old_settings = None
        if use_cbreak:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        def _on_tick(level: float):
            if use_cbreak and select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch in ("\x1b", "\x03", "q"):
                    recorder.stop_recording()
                elif ch == " ":
                    if recorder.is_paused:
                        recorder.resume()
                        state.paused = False
                    else:
                        recorder.pause()
                        state.paused = True
                elif ch == "x":
                    control["discard"] = True
                    recorder.stop_recording()
            state.push_level(level)
            if recorder.start_time:
                state.elapsed_seconds = (
                    datetime.now() - recorder.start_time
                ).total_seconds()
            live.update(_render_record_view(state))

        try:
            with Live(
                _render_record_view(state),
                console=console,
                refresh_per_second=10,
            ) as live:
                recorder.start_recording(
                    duration_minutes=duration,
                    title=title,
                    level_callback=_on_tick,
                    tags=resolved_tags,
                )
        finally:
            if old_settings is not None:
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        note_dir = recorder.note_dir
        if control["discard"]:
            if note_dir and note_dir.exists():
                shutil.rmtree(note_dir, ignore_errors=True)
            console.print("[yellow]discarded.[/yellow]")
            return

        if note_dir is None:
            console.print("[yellow]nothing to save.[/yellow]")
            return

        console.print(f"[green]✅ saved to {note_dir}[/green]")
        console.print(" [dim]› chirp transcribe    · turn this into notes[/dim]")

    except KeyboardInterrupt:
        console.print("[yellow]Recording stopped by user[/yellow]")
    except AudioDeviceError as e:
        console.print(f"[red]❌ Audio device error: {str(e)}[/red]")
        console.print("[dim]Try 'chirp devices' to see available audio devices[/dim]")
        raise typer.Exit(1)
    except RecordingError as e:
        console.print(f"[red]❌ Recording error: {str(e)}[/red]")
        raise typer.Exit(1)
    except ConfigurationError as e:
        console.print(f"[red]❌ Configuration error: {str(e)}[/red]")
        console.print("[dim]Try 'chirp config --list' to check your settings[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        console.print("[dim]Please report this issue if it persists[/dim]")
        raise typer.Exit(1)


def _run_live_transcription(
    settings,
    device_manager,
    title: str | None,
    duration: int | None,
    debug_live: bool = False,
    tags: list[str] | None = None,
):
    from recorder.live_session import LiveSessionResult, LiveTranscriptionSession

    if title:
        console.print(f"[cyan]📝 Title: {title}[/cyan]")
    if duration:
        console.print(f"[cyan]⏱️ Planned duration: {duration} minutes[/cyan]")

    session = LiveTranscriptionSession(
        settings=settings,
        device_manager=device_manager,
        console=console,
        title=title,
        duration_minutes=duration,
        debug=debug_live,
        tags=list(tags or []),
    )

    try:
        result: LiveSessionResult = session.run()
    except ImportError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(1)
    except RecordingError as e:
        console.print(f"[red]❌ Live recording error: {str(e)}[/red]")
        raise typer.Exit(1)

    from utils.time_utils import format_duration

    console.print()
    console.print("[green]✅ Live recording complete[/green]")
    console.print(f"[dim]Audio saved to:[/dim] {result.audio_path}")
    console.print(
        f"[dim]Duration:[/dim] {format_duration(result.duration_seconds)}  •  [dim]Live words transcribed:[/dim] {result.total_words}"
    )
    console.print(
        "[dim]Run 'chirp transcribe' to generate the high-quality transcript.[/dim]"
    )


@app.command(rich_help_panel=MAIN_PANEL)
def transcribe(
    n: int | None = typer.Argument(
        None,
        help="Optional cap: process only the N oldest untranscribed notes.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-run all stages on already-transcribed notes."
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model override (e.g. tiny, base, small, medium, large-v3).",
    ),
    regen: bool = typer.Option(
        False,
        "--regen",
        help="Regenerate notes from existing transcripts; skip audio transcription. "
        "Useful after switching LLMs.",
    ),
):
    """Turn audio into text + summary"""
    from transcriber.batch_processor import BatchProcessor

    settings = get_settings()

    if regen:
        if n is not None:
            console.print(
                "[red]--regen processes all transcribed records; do not pass N.[/red]"
            )
            raise typer.Exit(2)
        if force:
            console.print(
                "[red]--regen and --force are mutually exclusive (--force re-transcribes; "
                "--regen reuses existing transcripts).[/red]"
            )
            raise typer.Exit(2)
        _run_regen_pipeline(settings)
        return

    if n is not None and n < 1:
        console.print("[red]N must be a positive integer.[/red]")
        raise typer.Exit(2)

    if model:
        console.print(f"[cyan]Using Whisper model: {model}[/cyan]")

    processor = BatchProcessor(settings, model_override=model)
    processor.run_queue(n=n, force=force, console=console)


def _run_regen_pipeline(settings) -> None:
    from notes.note_generator import NoteGenerator

    notes_root = settings.directories.notes_root
    records = [
        record for record in list_notes(notes_root) if record.transcript is not None
    ]

    if not records:
        console.print(
            f"[yellow]No transcripts found in {notes_root}. "
            "Run `chirp transcribe` first.[/yellow]"
        )
        return

    note_generator = NoteGenerator(settings)
    console.print(
        f"[bold blue]🧠 Regenerating notes for {len(records)} record(s)...[/bold blue]"
    )
    result = note_generator.generate_for_records(records, force=True)

    sub_results = result.get("results", [])
    success_count = sum(1 for r in sub_results if r.get("success"))
    if success_count:
        console.print(
            f"[green]✅ Regenerated notes for {success_count}/{len(sub_results)} record(s)[/green]"
        )
    if success_count < len(sub_results):
        failed = [r for r in sub_results if not r.get("success")]
        for failure in failed:
            slug = failure.get("slug", "<unknown>")
            error = failure.get("error", "unknown error")
            console.print(f"[red]  ✗ {slug}: {error}[/red]")


notes_app = typer.Typer(help="Browse, view, edit, or delete your notes")
app.add_typer(notes_app, name="notes", rich_help_panel=MAIN_PANEL)


class NoteNotFound(Exception):
    pass


class AmbiguousNoteId(Exception):
    def __init__(self, matches: list[str]) -> None:
        super().__init__(f"ambiguous: {len(matches)} matches")
        self.matches = matches


@notes_app.callback(invoke_without_command=True)
def notes_callback(
    ctx: typer.Context,
    tag: str | None = typer.Option(
        None,
        "--tag",
        help="Filter by tag (comma-separated values are AND-combined).",
    ),
):
    """Browse, view, edit, or delete your notes"""
    if ctx.invoked_subcommand is not None:
        if tag is not None:
            console.print("[red]--tag is only valid when listing notes.[/red]")
            raise typer.Exit(2)
        return
    _list_notes(tag)


def _parse_tag_filter(tag: str | None) -> list[str]:
    if not tag:
        return []
    return [piece.strip() for piece in tag.split(",") if piece.strip()]


def _list_notes(tag: str | None) -> None:
    settings = get_settings()
    all_records = [
        record
        for record in list_notes(settings.directories.notes_root)
        if record.notes is not None
    ]
    tag_filter = _parse_tag_filter(tag)
    if tag_filter:
        records = [
            record
            for record in all_records
            if all(t in record.tags for t in tag_filter)
        ]
    else:
        records = all_records

    if not all_records:
        console.print(
            f"[yellow]No notes found in {settings.directories.notes_root}[/yellow]"
        )
        console.print(
            "[dim]Run 'chirp transcribe' to create notes from recordings[/dim]"
        )
        return

    if tag_filter and not records:
        console.print()
        tag_label = ", ".join(tag_filter)
        console.print(
            f" [bold]Your notes[/bold] [dim]· 0 of {len(all_records)} · "
            f"tag: {tag_label}[/dim]"
        )
        console.print()
        console.print(f"[yellow]No notes matching tag '{tag_label}'.[/yellow]")
        return

    console.print()
    if tag_filter:
        tag_label = ", ".join(tag_filter)
        console.print(
            f" [bold]Your notes[/bold] [dim]· {len(records)} of "
            f"{len(all_records)} · tag: {tag_label}[/dim]"
        )
    else:
        console.print(
            f" [bold]Your notes[/bold] [dim]· {len(records)} total · "
            "sorted by date[/dim]"
        )
    console.print()

    table = Table(
        show_header=True,
        header_style="yellow bold",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("id", style="dim", no_wrap=True)
    table.add_column("title", style="white")
    table.add_column("date", style="cyan", no_wrap=True)
    table.add_column("length", style="dim", justify="right", no_wrap=True)
    table.add_column("tags", style="dim", no_wrap=True)

    for record in reversed(records):
        title = _resolve_display_title(record)

        try:
            stat = record.notes.stat() if record.notes else None
        except OSError:
            stat = None

        if stat is not None:
            date_str = record.created_at.strftime("%b %d").lower()
            length_str = f"{stat.st_size / 1024:.1f} KB"
        else:
            date_str = "?"
            length_str = "?"

        tag_cell = ", ".join(record.tags) if record.tags else "—"
        table.add_row(record.slug, title, date_str, length_str, tag_cell)

    console.print(table)
    console.print()
    console.print(" [dim]› chirp notes view <id>      · open a note read-only[/dim]")
    console.print(" [dim]› chirp notes edit <id>      · edit a note[/dim]")
    console.print(" [dim]› chirp notes delete <id>    · delete a note[/dim]")
    console.print(" [dim]› chirp notes --tag meeting  · filter by tag[/dim]")


def _resolve_note(records: list[NoteRecord], note_id: str) -> NoteRecord:
    if not note_id or not note_id.strip():
        raise NoteNotFound(note_id)
    exact = [record for record in records if record.slug == note_id]
    if len(exact) == 1:
        return exact[0]
    prefix_matches = [record for record in records if record.slug.startswith(note_id)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise AmbiguousNoteId([record.slug for record in prefix_matches])
    raise NoteNotFound(note_id)


def _load_notes_or_exit() -> list[NoteRecord]:
    settings = get_settings()
    records = [
        record
        for record in list_notes(settings.directories.notes_root)
        if record.notes is not None
    ]
    if not records:
        console.print(
            f"[yellow]No notes found in {settings.directories.notes_root}[/yellow]"
        )
        raise typer.Exit(1)
    return records


def _resolve_or_exit(note_id: str) -> NoteRecord:
    records = _load_notes_or_exit()
    try:
        return _resolve_note(records, note_id)
    except NoteNotFound:
        console.print(f"[red]✗ no note matching '{note_id}'[/red]")
        raise typer.Exit(1)
    except AmbiguousNoteId as exc:
        console.print(
            f"[red]✗ '{note_id}' matches {len(exc.matches)} notes — "
            "be more specific[/red]"
        )
        for slug in exc.matches:
            console.print(f"[dim]  • {slug}[/dim]")
        raise typer.Exit(1)


@notes_app.command("view")
def notes_view(note_id: str = typer.Argument(..., help="Note id (slug or prefix)")):
    """Open a note read-only in the terminal editor."""
    from notes.note_editor import ManualNoteEditor

    record = _resolve_or_exit(note_id)
    if record.notes is None:
        console.print(f"[red]✗ note '{record.slug}' has no notes.md[/red]")
        raise typer.Exit(1)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(
            "[yellow]Interactive editor requires a terminal. "
            "Please run from an interactive shell.[/yellow]"
        )
        raise typer.Exit(1)

    title = _resolve_display_title(record)
    content = record.notes.read_text(encoding="utf-8")
    editor = ManualNoteEditor(title, content, readonly=True)
    try:
        editor.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Editor cancelled[/dim]")


@notes_app.command("edit")
def notes_edit(note_id: str = typer.Argument(..., help="Note id (slug or prefix)")):
    """Edit a note in the terminal editor; saves rewrite notes.md and re-index."""
    from notes.note_editor import ManualNoteEditor

    settings = get_settings()
    record = _resolve_or_exit(note_id)
    if record.notes is None:
        console.print(f"[red]✗ note '{record.slug}' has no notes.md[/red]")
        raise typer.Exit(1)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(
            "[yellow]Interactive editor requires a terminal. "
            "Please run from an interactive shell.[/yellow]"
        )
        raise typer.Exit(1)

    title = _resolve_display_title(record)
    content = record.notes.read_text(encoding="utf-8")
    editor = ManualNoteEditor(title, content)
    try:
        result = editor.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Editor cancelled[/dim]")
        raise typer.Exit(1)

    if not result.saved:
        console.print("[yellow]Changes not saved.[/yellow]")
        return

    record.notes.write_text(result.content, encoding="utf-8")
    console.print(f"[green]✅ Updated note: {record.notes}[/green]")

    if settings.notes_chat.auto_index:
        _reindex_after_edit(settings, record)


def _reindex_after_edit(settings: ChirpSettings, record: NoteRecord) -> None:
    notes_path = record.notes
    if notes_path is None:
        return
    try:
        from notes_chat.index import IndexManager

        index_manager = IndexManager(settings)
        if index_manager._add_to_index(notes_path):
            manifest = index_manager._load_manifest()
            current_files = index_manager._scan_notes_files()
            file_path = str(notes_path)
            if file_path in current_files:
                manifest[file_path] = current_files[file_path]
                index_manager._save_manifest(manifest)
            index_manager._rebuild_bm25()
            console.print(f"[dim green]✓ Re-indexed {notes_path.name}[/dim green]")
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            f"[dim yellow]⚠️ Auto-indexing failed for "
            f"{notes_path.name}: {exc}[/dim yellow]"
        )


@notes_app.command("delete")
def notes_delete(
    note_id: str = typer.Argument(..., help="Note id (slug or prefix)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Delete a note's entire <slug>/ folder and remove it from the index."""
    import shutil

    settings = get_settings()
    record = _resolve_or_exit(note_id)
    title = _resolve_display_title(record)

    if not yes:
        confirmed = typer.confirm(
            f'Delete "{title}"? This removes the entire {record.dir.name}/ folder.',
            default=False,
        )
        if not confirmed:
            console.print("[yellow]Deletion cancelled.[/yellow]")
            return

    notes_path = record.notes
    try:
        shutil.rmtree(record.dir, ignore_errors=False)
    except OSError as exc:
        console.print(f"[red]✗ failed to delete {record.dir}: {exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✅ Deleted {record.dir}[/green]")

    if notes_path is not None:
        _drop_from_index(settings, notes_path)


def _drop_from_index(settings: ChirpSettings, notes_path: Path) -> None:
    try:
        from notes_chat.index import IndexManager

        index_manager = IndexManager(settings)
        index_manager._remove_from_index(str(notes_path))
        manifest = index_manager._load_manifest()
        manifest.pop(str(notes_path), None)
        index_manager._save_manifest(manifest)
        index_manager._rebuild_bm25()
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[dim yellow]⚠️ Failed to update index: {exc}[/dim yellow]")


def _resolve_display_title(record: NoteRecord) -> str:
    if record.title:
        return record.title
    if record.notes is None:
        return record.slug
    try:
        with record.notes.open(encoding="utf-8") as fh:
            for raw in fh:
                stripped = raw.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip()
    except OSError:
        pass
    return record.slug


@app.command(rich_help_panel=MAIN_PANEL)
def ask(
    question: str | None = typer.Option(
        None,
        "--question",
        "-q",
        help="Question to ask about your meetings (omit for interactive chat)",
    ),
    when: str | None = typer.Option(None, "--when", help="Time range filter"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show sources"),
    markdown: bool = typer.Option(
        True,
        "--markdown/--no-markdown",
        help="Render answers as markdown (code blocks, bullets, bold)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show prompt without calling LLM"
    ),
):
    """Chat with your notes"""
    from notes_chat.cli import ask

    ask(question, when, sources, dry_run, markdown=markdown)


@app.command(rich_help_panel=MAIN_PANEL)
def search():
    """Keyword search"""
    try:
        settings = get_settings()
        from notes_chat.search import LiveSearchSession

        session = LiveSearchSession(settings)
        session.start()

    except KeyboardInterrupt:
        console.print("\n[dim]Search cancelled[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Error during search: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command(rich_help_panel=MAIN_PANEL)
def init(
    recheck: bool = typer.Option(
        False, "--recheck", help="Only run the verify phase — don't install"
    ),
    switch_model: bool = typer.Option(
        False, "--switch-model", help="Skip to the model-picker phase"
    ),
):
    """First-run setup & model picker"""
    from chirp.init_flow import run_init

    settings = get_settings()
    code = run_init(settings, console, recheck=recheck, switch_model=switch_model)
    if code != 0:
        raise typer.Exit(code)


if __name__ == "__main__":
    app()


@app.command(rich_help_panel=MAIN_PANEL)
def about():
    """Show the bird 🐦"""
    from chirp.about import run_about

    settings = get_settings()
    try:
        run_about(console, settings)
    except KeyboardInterrupt:
        console.print()


@app.command(hidden=True)
def index(
    force: bool = typer.Option(False, "--force", help="Force full rebuild of index"),
):
    """Build or rebuild the notes search index"""
    from notes_chat.cli import index

    index(force)


@app.command(hidden=True)
def config(
    list_config: bool = typer.Option(
        False, "--list", "-l", help="List current configuration"
    ),
    notes_root: Path | None = typer.Option(
        None, "--notes-root", help="Set the notes root directory"
    ),
    whisper_model: str | None = typer.Option(
        None,
        "--whisper-model",
        help="Set Whisper model (e.g. tiny, base, small, medium, large-v3)",
    ),
    llm_model: str | None = typer.Option(
        None, "--llm-model", help="Set LLM model (e.g. llama3.1:8b)"
    ),
    ollama_url: str | None = typer.Option(
        None, "--ollama-url", help="Set Ollama server URL"
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="Set embedding model (e.g. nomic-embed-text)",
    ),
):
    """Manage Chirp configuration"""
    settings = get_settings()

    if list_config:
        panel = Panel.fit(
            f"""[cyan]Directories:[/cyan]
Notes Root: {settings.directories.notes_root}

[cyan]Models:[/cyan]
Whisper: {settings.models.whisper}
LLM: {settings.models.llm}
Ollama URL: {settings.models.ollama_url}
Embedding: {settings.notes_chat.emb_model}

[cyan]Audio:[/cyan]
Sample Rate: {settings.audio.sample_rate}
Channels: {settings.audio.channels}

[cyan]Monitoring:[/cyan]
Warning: {settings.monitoring.warning_minutes} minutes
Interval: {settings.monitoring.warning_interval} minutes""",
            title="🐣 Chirp Configuration",
        )
        console.print(panel)
        return

    changes_made = False

    if notes_root:
        settings.directories.notes_root = notes_root
        changes_made = True

    if whisper_model:
        settings.models.whisper = whisper_model
        changes_made = True

    if llm_model:
        settings.models.llm = llm_model
        changes_made = True

    if ollama_url:
        settings.models.ollama_url = ollama_url
        changes_made = True

    if embedding_model:
        settings.notes_chat.emb_model = embedding_model
        changes_made = True

    if changes_made:
        settings.save_to_file(ChirpSettings.get_config_path())
        settings.ensure_directories_exist()
        console.print("[green]✅ Configuration updated[/green]")


@app.command(hidden=True)
def devices():
    """List available audio devices"""
    from recorder.device_manager import DeviceManager

    device_manager = DeviceManager()
    devices_info = device_manager.list_devices()

    default_input_index = device_manager.get_default_input_device()
    default_output_index = device_manager.get_default_output_device()

    input_devices = [d for d in devices_info if d["max_input_channels"] > 0]
    output_devices = [d for d in devices_info if d["max_output_channels"] > 0]

    input_table = Table(title="Input Devices (microphones & capture)")
    input_table.add_column("", style="bold")
    input_table.add_column("ID", style="cyan")
    input_table.add_column("Name")
    input_table.add_column("Input Ch", style="yellow")
    input_table.add_column("Default Rate", style="blue")

    for device in input_devices:
        is_default = device["index"] == default_input_index
        marker = "▶" if is_default else ""
        name_style = "bold green" if is_default else ""
        suffix = " (system default)" if is_default else ""
        input_table.add_row(
            marker,
            str(device["index"]),
            Text(device["name"] + suffix, style=name_style),
            str(device["max_input_channels"]),
            f"{device['default_sample_rate']:.0f} Hz",
        )

    console.print(input_table)
    console.print()

    output_table = Table(title="Output Devices (speakers & routing)")
    output_table.add_column("", style="bold")
    output_table.add_column("ID", style="cyan")
    output_table.add_column("Name")
    output_table.add_column("Output Ch", style="yellow")
    output_table.add_column("Default Rate", style="blue")

    for device in output_devices:
        is_system_default = device["index"] == default_output_index
        marker = "◀" if is_system_default else ""
        name_style = "bold blue" if is_system_default else ""
        suffix = " (system default)" if is_system_default else ""
        output_table.add_row(
            marker,
            str(device["index"]),
            Text(device["name"] + suffix, style=name_style),
            str(device["max_output_channels"]),
            f"{device['default_sample_rate']:.0f} Hz",
        )

    console.print(output_table)

    console.print()
    if device_manager.check_blackhole_available():
        console.print("[green]✅ BlackHole detected and ready[/green]")
    elif device_manager.check_aggregate_available():
        console.print("[green]✅ Aggregate device detected and ready[/green]")
    elif device_manager.get_default_input_device() is not None:
        console.print(
            "[green]✅ Default input device detected (microphone recording ready)[/green]"
        )
        console.print(
            "[dim]For system audio capture, install BlackHole: https://existential.audio/blackhole/[/dim]"
        )
    else:
        console.print("[red]❌ No suitable input device found[/red]")
        console.print("Install BlackHole from: https://existential.audio/blackhole/")
        console.print("Or create an Aggregate Device in Audio MIDI Setup")
    console.print("[dim]Run 'chirp init' for first-run setup.[/dim]")
