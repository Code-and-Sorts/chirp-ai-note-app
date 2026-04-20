import sys
from pathlib import Path
from typing import Optional

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
from utils.file_utils import (
    get_audio_files,
    get_notes_files,
    get_transcription_files,
)

app = typer.Typer(
    name="chirp",
    help="Chirp · AI notes for your terminal.",
    rich_markup_mode="rich",
    add_completion=False,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
    },
)
console = Console()

MAIN_PANEL = "Commands"
SETUP_PANEL = "Setup"
INFO_PANEL = "Info"

RECORDING_PANEL = MAIN_PANEL
CHAT_PANEL = MAIN_PANEL
PROCESSING_PANEL = MAIN_PANEL


def _test_ollama_connection(settings):
    """Test if Ollama is running and accessible"""
    try:
        import requests

        response = requests.get(f"{settings.models.ollama_url}/api/version", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def _test_llm_model_available(settings):
    """Test if the configured LLM model is available in Ollama"""
    try:
        import requests

        response = requests.get(f"{settings.models.ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            return settings.models.llm in available_models
        return False
    except Exception:
        return False


def _test_embedding_model_available(settings):
    """Test if the configured embedding model is available in Ollama"""
    try:
        import requests

        response = requests.get(f"{settings.models.ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]

            if settings.notes_chat.emb_model in available_models:
                return True

            if f"{settings.notes_chat.emb_model}:latest" in available_models:
                return True

            model_base = settings.notes_chat.emb_model.split(":")[0]
            for model in available_models:
                if model.startswith(f"{model_base}:"):
                    return True

            return False
        return False
    except Exception:
        return False


def _test_notes_index(settings):
    """Test if notes index exists and is accessible"""
    try:
        index_dir = settings.notes_chat.index_dir
        manifest_file = index_dir / "manifest.json"
        chroma_dir = index_dir / "chroma"
        return manifest_file.exists() and chroma_dir.exists()
    except Exception:
        return False


def _test_chroma_db(settings):
    """Test if ChromaDB is working for notes search"""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        index_dir = settings.notes_chat.index_dir
        client = chromadb.PersistentClient(
            path=str(index_dir / "chroma"),
            settings=ChromaSettings(allow_reset=True),
        )
        client.get_or_create_collection(name="notes")
        return True
    except Exception:
        return False


METER_WIDTH = 20

recording_spinner = Spinner("dots", style="bold green")


def _prompt_title() -> str:
    console.print()
    console.print(" [yellow bold]title[/yellow bold] [dim](required)[/dim]")
    while True:
        try:
            value = console.input(" [green]›[/green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            raise typer.Exit(1)
        if value:
            return value
        console.print(" [dim]title can't be empty[/dim]")


def _prompt_timeframe() -> Optional[int]:
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
    title: Optional[str],
    duration_minutes: Optional[int],
    device_manager,
) -> None:
    duration_str = (
        f"{duration_minutes:02d}:00" if duration_minutes else "∞"
    )
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
    duration: Optional[int] = typer.Option(
        None,
        "--duration",
        "-d",
        help="Recording duration in minutes (press Ctrl+C to stop if not specified)",
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Meeting title for filename"
    ),
    timeframe: Optional[str] = typer.Option(
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
        import sys

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

        console.print(
            " [dim][space] pause  [q / ^C] stop & save  [x] discard[/dim]"
        )
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
    title: Optional[str],
    duration: Optional[int],
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


@app.command(name="notes", rich_help_panel=MAIN_PANEL)
def notes_list():
    """Browse your notes"""
    settings = get_settings()
    notes_files = get_notes_files(settings.directories.notes)

    if not notes_files:
        console.print(
            f"[yellow]No notes found in {settings.directories.notes}[/yellow]"
        )
        console.print(
            "[dim]Run 'chirp transcribe' to create notes from recordings[/dim]"
        )
        return

    from datetime import datetime as dt

    console.print()
    console.print(
        f" [bold]Your notes[/bold] [dim]· {len(notes_files)} total · sorted by date[/dim]"
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

    sorted_notes = sorted(
        notes_files, key=lambda p: p.stat().st_mtime, reverse=True
    )
    for idx, note_file in enumerate(sorted_notes, start=1):
        title = note_file.stem
        try:
            with open(note_file, encoding="utf-8") as f:
                for raw in f:
                    stripped = raw.strip()
                    if stripped.startswith("# "):
                        title = stripped[2:].strip()
                        break
        except OSError:
            pass

        try:
            stat = note_file.stat()
            date_str = dt.fromtimestamp(stat.st_mtime).strftime("%b %d").lower()
            size_kb = stat.st_size / 1024
            length_str = f"{size_kb:.1f} KB"
        except OSError:
            date_str = "?"
            length_str = "?"

        table.add_row(str(idx), title, date_str, length_str)

    console.print(table)
    console.print()
    console.print(" [dim]› chirp note view <n>    · open note #n[/dim]")
    console.print(" [dim]› chirp ask              · ask a question[/dim]")


@app.command(rich_help_panel=MAIN_PANEL)
def search():
    """Keyword search through your notes"""
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


@app.command(name="note", rich_help_panel=MAIN_PANEL)
def note(
    name: Optional[str] = typer.Argument(
        None,
        metavar="[NAME]",
        help="Optional name for the note (defaults to note-YYYY-MM-DD)",
    ),
):
    """Create or edit a single note in the terminal editor."""
    from notes.manual_note_manager import ManualNoteManager
    from notes.note_editor import ManualNoteEditor

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(
            "[yellow]Interactive notes editor requires a terminal. Please run from an interactive shell.[/yellow]"
        )
        raise typer.Exit(1)

    settings = get_settings()
    manager = ManualNoteManager(settings)

    try:
        context = manager.prepare_note(name)
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[red]❌ Failed to prepare note: {exc}[/red]")
        raise typer.Exit(1)

    editor = ManualNoteEditor(context.title, context.content)

    try:
        result = editor.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Editor cancelled[/dim]")
        raise typer.Exit(1)
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[red]❌ Editor error: {exc}[/red]")
        raise typer.Exit(1)

    if not result.saved:
        if context.is_new:
            console.print("[yellow]Note discarded (not saved).[/yellow]")
        else:
            console.print("[yellow]Changes not saved.[/yellow]")
        raise typer.Exit(0)

    try:
        context.path.write_text(result.content, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[red]❌ Failed to write note: {exc}[/red]")
        raise typer.Exit(1)

    if context.is_new:
        console.print(f"[green]✅ Created note: {context.path}[/green]")
    else:
        console.print(f"[green]✅ Updated note: {context.path}[/green]")

    try:
        if settings.notes_chat.auto_index:
            from notes_chat.index import IndexManager

            index_manager = IndexManager(settings)
            success = index_manager._add_to_index(context.path)

            if success:
                manifest = index_manager._load_manifest()
                current_files = index_manager._scan_notes_files()

                file_path = str(context.path)
                if file_path in current_files:
                    manifest[file_path] = current_files[file_path]
                    index_manager._save_manifest(manifest)

                index_manager._rebuild_bm25()
                console.print(
                    f"[dim green]✓ Auto-indexed {context.path.name}[/dim green]"
                )
    except Exception as e:  # pragma: no cover - defensive
        console.print(
            f"[dim yellow]⚠️ Auto-indexing failed for {context.path.name}: {e}[/dim yellow]"
        )


@app.command(rich_help_panel=MAIN_PANEL)
def ask(
    question: Optional[str] = typer.Option(
        None,
        "--question",
        "-q",
        help="Question to ask about your meetings (omit for interactive chat)",
    ),
    when: Optional[str] = typer.Option(None, "--when", help="Time range filter"),
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
def transcribe(
    note_id: Optional[int] = typer.Argument(
        None,
        help="Optional index from `chirp notes` — pipeline for a single recording.",
    ),
    input_dir: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input directory for audio files"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-transcribe already processed files"
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Stream transcription as it processes"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model to use (e.g. tiny, base, small, medium, large-v3)",
    ),
    no_notes: bool = typer.Option(
        False, "--no-notes", help="Skip the notes-generation step"
    ),
    no_index: bool = typer.Option(
        False, "--no-index", help="Skip the chromadb indexing step"
    ),
):
    """Turn audio into text + AI notes (pipeline)"""
    from transcriber.batch_processor import BatchProcessor

    settings = get_settings()

    if note_id is not None:
        _run_transcribe_pipeline(
            settings, note_id, model=model, do_notes=not no_notes, do_index=not no_index
        )
        return

    if input_dir is None:
        input_dir = settings.directories.raw_audio

    audio_files = get_audio_files(input_dir)

    if not audio_files:
        console.print(f"[yellow]No audio files found in {input_dir}[/yellow]")
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
        task = progress.add_task(
            "Transcribing audio files...", total=len(audio_files)
        )

        def on_segment(segment):
            text = segment.get("text", "").strip()
            if text:
                streaming_text.append(text + " ", style="cyan")

        segment_callback = on_segment

        with Live(Group(streaming_text, progress), console=console, refresh_per_second=4):
            results = processor.process_files(
                audio_files,
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
            task = progress.add_task(
                "Transcribing audio files...", total=len(audio_files)
            )

            results = processor.process_files(
                audio_files,
                force=force,
                progress_callback=lambda: progress.update(task, advance=1),
            )

    success_count = sum(1 for r in results if r["success"])
    processed_count = len(results)
    skipped_count = len(audio_files) - processed_count

    if processed_count > 0:
        console.print(
            f"[green]✅ Successfully transcribed {success_count}/{processed_count} files[/green]"
        )
        if not no_notes:
            console.print(
                "[dim]Run 'chirp generate' to turn transcripts into notes.[/dim]"
            )

    if skipped_count > 0 and not force:
        console.print(
            f"[yellow]⏭️  Skipped {skipped_count} file(s) - already transcribed (use --force to re-transcribe)[/yellow]"
        )


def _run_transcribe_pipeline(
    settings,
    note_id: int,
    model: Optional[str],
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

    audio_files = sorted(
        get_audio_files(settings.directories.raw_audio),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if note_id < 1 or note_id > len(audio_files):
        console.print(
            f"[red]No recording at index {note_id} — run `chirp notes` to see available notes.[/red]"
        )
        raise typer.Exit(1)

    audio_path = audio_files[note_id - 1]

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
            results = processor.process_files(
                [audio_path], force=False, on_segment=on_segment
            )
            set_state(2, "done")
            live.update(render())

            if do_notes:
                set_state(3, "running")
                live.update(render())
                from notes.note_generator import NoteGenerator
                from utils.file_utils import get_transcription_files

                transcription_files = get_transcription_files(
                    settings.directories.transcriptions
                )
                note_generator = NoteGenerator(settings)
                note_generator.generate_daily_notes(transcription_files)
                set_state(3, "done")
                live.update(render())

            if do_index and settings.notes_chat.auto_index:
                set_state(4, "running")
                live.update(render())
                from notes_chat.index import build_index
                from notes_chat.config import get_notes_config

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


@app.command(rich_help_panel=SETUP_PANEL)
def generate(
    force: bool = typer.Option(
        False, "--force", "-f", help="Regenerate notes even if they exist"
    ),
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        "-n",
        help="Override the default filename for generated notes",
    ),
):
    """Generate meeting notes from transcriptions"""
    from notes.note_generator import NoteGenerator

    settings = get_settings()

    transcription_files = get_transcription_files(settings.directories.transcriptions)

    if not transcription_files:
        console.print(
            f"[yellow]No transcription files found in {settings.directories.transcriptions}[/yellow]"
        )
        console.print("[dim]Run 'chirp transcribe' first[/dim]")
        return

    note_generator = NoteGenerator(settings)

    console.print("[bold blue]🧠 Generating notes with AI...[/bold blue]")
    result = note_generator.generate_daily_notes(
        transcription_files, force=force, filename_override=filename
    )

    if result["success"]:
        console.print(f"[green]✅ Notes generated: {result['filename']}[/green]")
    else:
        console.print(f"[red]❌ Failed to generate notes: {result['error']}[/red]")


@app.command(name="transcribe-and-generate", rich_help_panel=SETUP_PANEL)
def transcribe_and_generate(
    input_dir: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input directory for audio files"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Process all files, including already processed ones",
    ),
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        "-n",
        help="Override the default filename for generated notes",
    ),
):
    """Process audio files (transcribe + generate notes)"""
    get_settings()

    try:
        transcribe(input_dir=input_dir, force=force)
        generate(force=force, filename=filename)
    except Exception as e:
        console.print(f"[red]Error in process command: {e}[/red]")


@app.command(rich_help_panel=SETUP_PANEL)
def index(
    force: bool = typer.Option(False, "--force", help="Force full rebuild of index"),
):
    """Build or rebuild the notes search index"""
    from notes_chat.cli import index

    index(force)


@app.command(rich_help_panel=INFO_PANEL)
def stats():
    """Show Chirp statistics"""
    settings = get_settings()

    table = Table(title="🐣 Chirp Statistics", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Value", style="green")

    audio_files = get_audio_files(settings.directories.raw_audio)
    transcription_files = get_transcription_files(settings.directories.transcriptions)
    notes_files = get_notes_files(settings.directories.notes)

    table.add_row("Raw Audio Files", str(len(audio_files)))
    table.add_row("Transcriptions", str(len(transcription_files)))
    table.add_row("Notes", str(len(notes_files)))
    table.add_row("Audio Directory", str(settings.directories.raw_audio))
    table.add_row("Transcription Directory", str(settings.directories.transcriptions))
    table.add_row("Notes Directory", str(settings.directories.notes))
    table.add_row("Whisper Model", settings.models.whisper)
    table.add_row("LLM Model", settings.models.llm)
    table.add_row("Embedding Model", settings.notes_chat.emb_model)
    table.add_row("Ollama URL", settings.models.ollama_url)

    console.print(table)


@app.command(rich_help_panel=SETUP_PANEL)
def config(
    list_config: bool = typer.Option(
        False, "--list", "-l", help="List current configuration"
    ),
    audio_dir: Optional[Path] = typer.Option(
        None, "--audio-dir", help="Set audio directory"
    ),
    transcription_dir: Optional[Path] = typer.Option(
        None, "--transcription-dir", help="Set transcription directory"
    ),
    notes_dir: Optional[Path] = typer.Option(
        None, "--notes-dir", help="Set notes directory"
    ),
    whisper_model: Optional[str] = typer.Option(
        None,
        "--whisper-model",
        help="Set Whisper model (e.g. tiny, base, small, medium, large-v3)",
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model", help="Set LLM model (e.g. llama3.1:8b)"
    ),
    ollama_url: Optional[str] = typer.Option(
        None, "--ollama-url", help="Set Ollama server URL"
    ),
    embedding_model: Optional[str] = typer.Option(
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
Audio: {settings.directories.raw_audio}
Transcriptions: {settings.directories.transcriptions}
Notes: {settings.directories.notes}
Templates: {settings.directories.templates}

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

    if audio_dir:
        settings.directories.raw_audio = audio_dir
        changes_made = True

    if transcription_dir:
        settings.directories.transcriptions = transcription_dir
        changes_made = True

    if notes_dir:
        settings.directories.notes = notes_dir
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


@app.command(rich_help_panel=INFO_PANEL)
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
    console.print("[dim]Run 'chirp setup' for audio configuration guide[/dim]")


@app.command(rich_help_panel=SETUP_PANEL)
def setup():
    """Step-by-step guide to configure audio for meeting recording"""
    from recorder.device_manager import DeviceManager

    device_manager = DeviceManager()

    console.print(
        Panel(
            "[bold]Audio Setup Guide[/bold]\n\n"
            "Chirp records system audio (e.g. Teams or Zoom calls) using\n"
            "BlackHole and a Multi-Output Device on macOS.",
            title="🐣 Chirp Setup",
        )
    )

    # Step 1: BlackHole
    console.print()
    has_blackhole = device_manager.check_blackhole_available()
    if has_blackhole:
        console.print("[green]Step 1: BlackHole ✅ Installed[/green]")
    else:
        console.print("[red]Step 1: Install BlackHole[/red]")
        console.print("  Download from: https://existential.audio/blackhole/")
        console.print(
            "  BlackHole is a virtual audio driver that captures system audio."
        )
        console.print()
        console.print(
            "[yellow]Install BlackHole and re-run 'chirp setup' to continue.[/yellow]"
        )
        return

    # Step 2: Multi-Output Device
    console.print()
    console.print('[bold]Step 2: Create a Multi-Output Device ("Chirp Output")[/bold]')
    console.print()
    console.print(
        "  This routes system audio to both your speakers AND BlackHole,\n"
        "  so you can still hear audio while Chirp records it."
    )
    console.print()
    console.print("  1. Open [bold]Audio MIDI Setup[/bold] (/Applications/Utilities/)")
    console.print(
        "  2. Click [bold]+[/bold] at bottom left → Create Multi-Output Device"
    )
    console.print(
        "  3. Check your [bold]speakers/headphones[/bold] (so you can still hear)"
    )
    console.print("  4. Check [bold]BlackHole 2ch[/bold]")
    console.print("  5. Rename it to [bold]Chirp Output[/bold] (double-click the name)")

    # Step 3: Aggregate Device
    console.print()
    console.print('[bold]Step 3: Create an Aggregate Device ("Chirp Input")[/bold]')
    console.print()
    console.print(
        "  This combines your microphone AND BlackHole into a single input,\n"
        "  so Chirp captures both your voice and system audio (e.g. remote\n"
        "  participants on a call)."
    )
    console.print()
    console.print(
        "  1. In [bold]Audio MIDI Setup[/bold], click [bold]+[/bold] → Create Aggregate Device"
    )
    console.print("  2. Check [bold]BlackHole 2ch[/bold]")
    console.print(
        "  3. Check your [bold]microphone[/bold] (e.g. Built-in, USB mic, etc.)"
    )
    console.print("  4. Rename it to [bold]Chirp Input[/bold] (double-click the name)")

    # Step 4: Set system audio
    console.print()
    console.print("[bold]Step 4: Set your system audio[/bold]")
    console.print()
    console.print(
        "  Go to [bold]System Settings → Sound[/bold]:\n"
        "  • [bold]Output[/bold] → select [bold]Chirp Output[/bold]\n"
        "  • [bold]Input[/bold]  → select [bold]Chirp Input[/bold]"
    )

    # Step 5: Done
    console.print()
    console.print("[bold]Step 5: Record![/bold]")
    console.print()
    console.print("  [bold]chirp record[/bold]")
    console.print()
    console.print(
        "  Chirp records from your system default input device.\n"
        "  Use 'chirp devices' to verify your setup (marked with ▶ and ◀)."
    )

    # Summary
    console.print()
    console.print(
        Panel(
            "[bold]How it works:[/bold]\n\n"
            "[cyan]Your voice[/cyan]    → Chirp Input (aggregate) → [bold]Chirp recording[/bold]\n"
            "[cyan]System audio[/cyan] → Chirp Output → Speakers (you hear it)\n"
            "                              → BlackHole → Chirp Input → [bold]Chirp recording[/bold]\n\n"
            "[bold]System settings:[/bold]\n"
            "  • [yellow]Output[/yellow] → [bold]Chirp Output[/bold] "
            "(Multi-Output Device)\n"
            "  • [yellow]Input[/yellow]  → [bold]Chirp Input[/bold] "
            "(Aggregate Device)",
            title="Summary",
        )
    )


@app.command(rich_help_panel=SETUP_PANEL)
def test():
    """Test Chirp dependencies and configuration"""
    from recorder.device_manager import DeviceManager

    settings = get_settings()

    external_tests = []
    device_manager = DeviceManager()
    external_tests.append(("PyAudio", device_manager.test_pyaudio()))
    external_tests.append(("BlackHole", device_manager.check_blackhole_available()))

    try:
        from faster_whisper import WhisperModel  # noqa: F401

        external_tests.append(("Faster Whisper", True))
    except ImportError:
        external_tests.append(("Faster Whisper", False))

    ollama_connected = _test_ollama_connection(settings)
    external_tests.append(("Ollama", ollama_connected))

    chroma_working = _test_chroma_db(settings)
    external_tests.append(("ChromaDB", chroma_working))

    config_tests = []
    settings.ensure_directories_exist()
    config_tests.append(("Directories", True))

    if ollama_connected:
        llm_available = _test_llm_model_available(settings)
        config_tests.append(("LLM Model Available", llm_available))

        embedding_available = _test_embedding_model_available(settings)
        config_tests.append(("Embedding Model Available", embedding_available))
    else:
        config_tests.append(("LLM Model Available", False))
        config_tests.append(("Embedding Model Available", False))

    notes_index_built = _test_notes_index(settings)
    config_tests.append(("Notes Index", notes_index_built))

    external_table = Table(title="🔧 External Dependencies")
    external_table.add_column("Component", style="cyan")
    external_table.add_column("Status", style="bold")

    external_passed = True
    for name, passed in external_tests:
        status = "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]"
        external_table.add_row(name, status)
        if not passed:
            external_passed = False

    console.print(external_table)

    config_table = Table(title="⚙️ Configuration & Setup")
    config_table.add_column("Component", style="cyan")
    config_table.add_column("Status", style="bold")

    config_passed = True
    for name, passed in config_tests:
        status = "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]"
        config_table.add_row(name, status)
        if not passed:
            config_passed = False

    console.print(config_table)

    if external_passed and config_passed:
        console.print("\n[green]🎉 All tests passed! Chirp is ready to use.[/green]")
    else:
        if not external_passed:
            console.print(
                "\n[red]⚠️  External dependencies missing. Install or run required software.[/red]"
            )
        if not config_passed:
            console.print(
                "\n[yellow]⚠️  Configuration issues. Run setup commands to fix.[/yellow]"
            )
        console.print("[dim]Run 'chirp --help' for setup commands[/dim]")


@app.command(rich_help_panel=INFO_PANEL)
def version():
    """Show the installed Chirp version"""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        ver = pkg_version("chirp-notes-ai")
    except PackageNotFoundError:
        ver = "dev (not installed as package)"

    console.print(f"[bold]chirp[/bold] {ver}")


@app.command(rich_help_panel=INFO_PANEL)
def about():
    """Show the animated Chirp logo + version info"""
    from chirp.about import run_about

    settings = get_settings()
    try:
        run_about(console, settings)
    except KeyboardInterrupt:
        console.print()


@app.command(rich_help_panel=SETUP_PANEL)
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
