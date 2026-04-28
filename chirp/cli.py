import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from chirp.exceptions import *
from config.settings import ChirpSettings, get_settings
from utils.file_utils import NoteRecord, list_notes

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
)
console = Console()

MAIN_PANEL = "Commands"

METER_WIDTH = 20

recording_spinner = Spinner("dots", style="bold green")


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
        " [yellow bold]timeframe[/yellow bold] [dim](optional · press ⏎ to skip)[/dim]"
    )
    try:
        value = console.input(" [green]›[/green] ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None
    if not value:
        return None
    return parse_timeframe(value)


def _print_record_header(
    title: str | None,
    duration_minutes: int | None,
    device_manager,
) -> None:
    duration_str = f"{duration_minutes:02d}:00" if duration_minutes else "∞"
    try:
        default_idx = device_manager.get_default_input_device()
        devices_info = device_manager.list_devices()
        mic_name = next(
            (d["name"] for d in devices_info if d["index"] == default_idx),
            "default",
        )
    except Exception:
        mic_name = "default"
    line = Text()
    line.append(" ● ", style="red bold")
    line.append("REC", style="bold white")
    line.append("  ·  00:00:00")
    if duration_minutes:
        line.append(f" / {duration_str}")
    line.append(f"  ·  mic: {mic_name}", style="dim")
    if title:
        console.print(Text(f" [{title}]", style="cyan"))
    console.print(line)


def _render_audio_meter(level: float) -> Table:
    filled = int(level * METER_WIDTH)
    bar = Text()
    for i in range(METER_WIDTH):
        if i < filled:
            if i < METER_WIDTH * 0.6:
                bar.append("━", style="green")
            elif i < METER_WIDTH * 0.85:
                bar.append("━", style="yellow")
            else:
                bar.append("━", style="red")
        else:
            bar.append("━", style="dim")

    label = Text()
    label.append(" Recording  ", style="bold green")

    suffix = Text("  (ESC or Ctrl+C to stop)", style="dim")

    row = Table.grid(padding=0)
    row.add_row(recording_spinner, label, bar, suffix)
    return row


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

    if live_transcribe:
        _run_live_transcription(
            settings, device_manager, title, duration, debug_live=debug_live
        )
        return

    recorder = AudioRecorder(settings, device_manager)

    console.print(" [dim]──────────────────────────────────────────[/dim]")
    console.print()
    _print_record_header(title, duration, device_manager)
    console.print()

    from utils.time_utils import format_duration, get_recording_duration

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

        def _check_key_and_update(level: float):
            if use_cbreak:
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b" or ch == "\x03":
                        recorder.stop_recording()
                        return
            live.update(_render_audio_meter(level))

        try:
            with Live(
                _render_audio_meter(0.0), console=console, refresh_per_second=10
            ) as live:
                filename = recorder.start_recording(
                    duration_minutes=duration,
                    title=title,
                    level_callback=_check_key_and_update,
                )
        finally:
            if old_settings is not None:
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if recorder.start_time:
            actual_duration = get_recording_duration(recorder.start_time)
            duration_str = format_duration(actual_duration)
            console.print(f"[green]✅ Recording saved: {filename}[/green]")
            console.print(f"[dim]Actual duration: {duration_str}[/dim]")
        else:
            console.print(f"[green]✅ Recording saved: {filename}[/green]")

        console.print(" [dim][space] pause  [q / ^C] stop & save  [x] discard[/dim]")
        console.print("[dim]Use 'chirp transcribe' to process this recording[/dim]")

    except KeyboardInterrupt:
        console.print("[yellow]Recording stopped by user[/yellow]")
        if recorder.start_time:
            actual_duration = get_recording_duration(recorder.start_time)
            duration_str = format_duration(actual_duration)
            console.print(f"[dim]Final recording duration: {duration_str}[/dim]")
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
    note_id: int | None = typer.Argument(
        None,
        help="Index of a recording to process (newest-first by creation time).",
    ),
    input_dir: Path | None = typer.Option(
        None, "--input", "-i", help="Input directory for audio files"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-transcribe already processed files"
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Stream transcription as it processes"
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model to use (e.g. tiny, base, small, medium, large-v3)",
    ),
    no_notes: bool = typer.Option(
        False,
        "--no-notes",
        help="Skip notes-generation (single-recording mode only).",
    ),
    no_index: bool = typer.Option(
        False,
        "--no-index",
        help="Skip chromadb indexing (single-recording mode only).",
    ),
):
    """Turn audio into text + summary"""
    from transcriber.batch_processor import BatchProcessor

    settings = get_settings()

    if note_id is not None:
        _run_transcribe_pipeline(
            settings, note_id, model=model, do_notes=not no_notes, do_index=not no_index
        )
        return

    if no_notes or no_index:
        console.print(
            "[red]--no-notes / --no-index are only valid when a recording index is provided "
            "(e.g. `chirp transcribe 1`).[/red]"
        )
        raise typer.Exit(2)

    notes_root = input_dir or settings.directories.notes_root
    records = [record for record in list_notes(notes_root) if record.audio is not None]

    if not records:
        console.print(f"[yellow]No audio files found in {notes_root}[/yellow]")
        return

    if model:
        console.print(f"[cyan]Using Whisper model: {model}[/cyan]")

    processor = BatchProcessor(settings, model_override=model)

    segment_callback = None
    if stream:
        from rich.console import Group
        from rich.live import Live
        from rich.text import Text

        streaming_text = Text()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        )
        task = progress.add_task("Transcribing audio files...", total=len(records))

        def on_segment(segment):
            text = segment.get("text", "").strip()
            if text:
                streaming_text.append(text + " ", style="cyan")

        segment_callback = on_segment

        with Live(
            Group(streaming_text, progress), console=console, refresh_per_second=4
        ):
            results = processor.process_records(
                records,
                force=force,
                progress_callback=lambda: progress.update(task, advance=1),
                on_segment=segment_callback,
            )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Transcribing audio files...", total=len(records))

            results = processor.process_records(
                records,
                force=force,
                progress_callback=lambda: progress.update(task, advance=1),
            )

    success_count = sum(1 for r in results if r["success"])
    processed_count = len(results)
    skipped_count = len(records) - processed_count

    if processed_count > 0:
        console.print(
            f"[green]✅ Successfully transcribed {success_count}/{processed_count} notes[/green]"
        )
        if not no_notes:
            console.print(
                "[dim]Run 'chirp generate' to turn transcripts into notes.[/dim]"
            )

    if skipped_count > 0 and not force:
        console.print(
            f"[yellow]⏭️  Skipped {skipped_count} note(s) - already transcribed (use --force to re-transcribe)[/yellow]"
        )


def _print_missing_recording(settings, records, note_id: int) -> None:
    notes_root = settings.directories.notes_root
    if not records:
        console.print(
            f"[red]No recording at index {note_id}. No recordings in {notes_root}.[/red]"
        )
        return
    preview_limit = 3
    console.print(
        f"[red]No recording at index {note_id}. Valid indices are 1-{len(records)} "
        f"(newest first, from {notes_root}):[/red]"
    )
    for idx, record in enumerate(records[:preview_limit], start=1):
        console.print(f"[dim]  {idx}. {record.slug}[/dim]")
    if len(records) > preview_limit:
        console.print(f"[dim]  ... and {len(records) - preview_limit} more[/dim]")


def _run_transcribe_pipeline(
    settings,
    note_id: int,
    model: str | None,
    do_notes: bool,
    do_index: bool,
) -> None:
    """Single-recording 4-step pipeline matching A6:
    transcribe (whisper) → generate notes (qwen · notes-mode) → index → save.
    """
    from rich.console import Group
    from rich.live import Live
    from rich.text import Text

    from transcriber.batch_processor import BatchProcessor

    records = [
        record
        for record in list_notes(settings.directories.notes_root)
        if record.audio is not None
    ]
    records_newest_first = list(reversed(records))
    if note_id < 1 or note_id > len(records_newest_first):
        _print_missing_recording(settings, records_newest_first, note_id)
        raise typer.Exit(1)

    target_record = records_newest_first[note_id - 1]
    assert target_record.audio is not None, "filtered list guarantees audio is present"
    audio_path: Path = target_record.audio

    steps = [
        ("loaded audio", "pending"),
        ("detected language", "pending"),
        ("transcribe (whisper)", "pending"),
        ("generate notes (qwen · notes-mode)", "pending" if do_notes else "skip"),
        ("index to chromadb", "pending" if do_index else "skip"),
        ("save", "pending"),
    ]

    streaming_text = Text()

    def render() -> Group:
        status_lines: list[Text] = []
        for label, state in steps:
            if state == "done":
                icon = Text("✓ ", style="green")
            elif state == "running":
                icon = Text("⠙ ", style="yellow")
            elif state == "skip":
                icon = Text("— ", style="dim")
            else:
                icon = Text("○ ", style="dim")
            line = Text()
            line.append(icon)
            line.append(label)
            status_lines.append(line)
        header = Text()
        header.append(audio_path.stem, style="bold white")
        header.append(f"  · {audio_path.name}", style="dim")
        preview = Text("\n  notes-mode output (streaming):\n", style="dim")
        return Group(header, Text(""), *status_lines, preview, streaming_text)

    def set_state(idx: int, state: str) -> None:
        label, _ = steps[idx]
        steps[idx] = (label, state)

    with Live(render(), console=console, refresh_per_second=10) as live:
        try:
            size_kb = audio_path.stat().st_size / 1024
            steps[0] = (f"loaded audio · {size_kb:.1f} KB", "done")
            live.update(render())

            set_state(1, "running")
            live.update(render())
            set_state(1, "done")

            set_state(2, "running")
            live.update(render())

            def on_segment(segment):
                text = segment.get("text", "").strip()
                if text:
                    streaming_text.append(text + " ", style="cyan")
                    live.update(render())

            processor = BatchProcessor(settings, model_override=model)
            results = processor.process_records(
                [target_record], force=False, on_segment=on_segment
            )
            set_state(2, "done")
            live.update(render())

            if do_notes:
                set_state(3, "running")
                live.update(render())
                from notes.note_generator import NoteGenerator

                refreshed_records = [
                    record
                    for record in list_notes(settings.directories.notes_root)
                    if record.slug == target_record.slug and record.transcript
                ]
                if refreshed_records:
                    NoteGenerator(settings).generate_for_records(
                        refreshed_records, force=True
                    )
                    set_state(3, "done")
                else:
                    set_state(3, "skip")
                live.update(render())

            if do_index and settings.notes_chat.auto_index:
                set_state(4, "running")
                live.update(render())
                from notes_chat.config import get_notes_config
                from notes_chat.index import build_index

                build_index(get_notes_config())
                set_state(4, "done")
                live.update(render())

            set_state(5, "done")
            live.update(render())
        except Exception as exc:
            console.print(f"[red]❌ Pipeline failed: {exc}[/red]")
            raise typer.Exit(1)

    if any(r.get("success") for r in results):
        console.print("[green]✅ Pipeline complete.[/green]")
    else:
        console.print("[yellow]Pipeline finished with warnings — see above.[/yellow]")


@app.command(name="notes", rich_help_panel=MAIN_PANEL)
def notes_list():
    """Browse, view, edit notes"""
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
        console.print(
            "[dim]Run 'chirp transcribe' to create notes from recordings[/dim]"
        )
        return

    console.print()
    console.print(
        f" [bold]Your notes[/bold] [dim]· {len(records)} total · sorted by date[/dim]"
    )
    console.print()

    table = Table(
        show_header=True,
        header_style="yellow bold",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", no_wrap=True, justify="right")
    table.add_column("title", style="white")
    table.add_column("date", style="cyan", no_wrap=True)
    table.add_column("length", style="dim", justify="right", no_wrap=True)

    for idx, record in enumerate(reversed(records), start=1):
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

        table.add_row(str(idx), title, date_str, length_str)

    console.print(table)
    console.print()
    console.print(" [dim]› chirp note [NAME]      · open a note by name[/dim]")
    console.print(" [dim]› chirp ask              · ask a question[/dim]")


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
