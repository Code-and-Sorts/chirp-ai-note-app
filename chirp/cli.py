import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from chirp.exceptions import *
from config.settings import get_settings
from utils.file_utils import (
    get_audio_files,
    get_notes_files,
    get_transcription_files,
)

app = typer.Typer(
    name="chirp",
    help="🐣 Chirp - Meeting Recorder CLI that transcribes and generates AI notes",
    rich_markup_mode="rich",
)
console = Console()

RECORDING_PANEL = "🎙️  Recording"
CHAT_PANEL = "💬 Notes"
PROCESSING_PANEL = "⚡ Processing"
SETUP_PANEL = "⚙️  Setup & Diagnostics"
INFO_PANEL = "ℹ️  Information"


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


LEVEL_BLOCKS = "▁▂▃▄▅▆█"
EMPTY_BLOCK = "░"


def _render_audio_meter(level: float) -> Text:
    bar = Text()
    bar.append("🎙️ Recording  ")
    num_bars = 6
    for i in range(num_bars):
        threshold = i / num_bars
        if level > threshold:
            intensity = min(
                int((level - threshold) * num_bars * (len(LEVEL_BLOCKS) - 1)),
                len(LEVEL_BLOCKS) - 1,
            )
            bar.append(LEVEL_BLOCKS[intensity], style="green")
        else:
            bar.append(EMPTY_BLOCK, style="dim")
    bar.append("  (Ctrl+C to stop)", style="dim")
    return bar


@app.command(rich_help_panel=RECORDING_PANEL)
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
):
    """Start recording a meeting (press Ctrl+C to stop if no duration specified)"""
    from recorder.audio_recorder import AudioRecorder
    from recorder.device_manager import DeviceManager

    settings = get_settings()
    device_manager = DeviceManager()

    configured_device = settings.audio.input_device
    if not configured_device and not device_manager.check_blackhole_available():
        console.print(
            "[yellow]BlackHole not detected. Using default input device.[/yellow]"
        )
        console.print(
            "[dim]Tip: Use 'chirp config --input-device \"Aggregate Device\"' "
            "to set a specific device[/dim]"
        )

    recorder = AudioRecorder(settings, device_manager)

    if title:
        console.print(f"[cyan]📝 Title: {title}[/cyan]")
    if duration:
        console.print(f"[cyan]⏱️ Planned duration: {duration} minutes[/cyan]")

    from utils.time_utils import format_duration, get_recording_duration

    try:
        with Live(_render_audio_meter(0.0), console=console, refresh_per_second=10) as live:

            def _update_meter(level: float):
                live.update(_render_audio_meter(level))

            filename = recorder.start_recording(
                duration_minutes=duration, title=title, level_callback=_update_meter
            )

        if recorder.start_time:

            actual_duration = get_recording_duration(recorder.start_time)
            duration_str = format_duration(actual_duration)
            console.print(f"[green]✅ Recording saved: {filename}[/green]")
            console.print(f"[dim]Actual duration: {duration_str}[/dim]")
        else:
            console.print(f"[green]✅ Recording saved: {filename}[/green]")

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


@app.command(rich_help_panel=CHAT_PANEL)
def search():
    """Live search through your note titles"""
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


@app.command(rich_help_panel=CHAT_PANEL)
def notes(
    name: Optional[str] = typer.Argument(
        None,
        metavar="[NAME]",
        help="Optional name for the note (defaults to note-YYYY-MM-DD)",
    ),
):
    """Create or edit manual notes in the terminal editor."""
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


@app.command(rich_help_panel=CHAT_PANEL)
def ask(
    question: Optional[str] = typer.Option(
        None,
        "--question",
        "-q",
        help="Question to ask about your meetings (omit for interactive chat)",
    ),
    when: Optional[str] = typer.Option(None, "--when", help="Time range filter"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show sources"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show prompt without calling LLM"
    ),
):
    """Ask questions about your meeting notes"""
    from notes_chat.cli import ask

    ask(question, when, sources, dry_run)


@app.command(rich_help_panel=PROCESSING_PANEL)
def transcribe(
    input_dir: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input directory for audio files"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-transcribe already processed files"
    ),
):
    """Transcribe audio files to text"""
    from transcriber.batch_processor import BatchProcessor

    settings = get_settings()

    if input_dir is None:
        input_dir = settings.directories.raw_audio

    audio_files = get_audio_files(input_dir)

    if not audio_files:
        console.print(f"[yellow]No audio files found in {input_dir}[/yellow]")
        return

    processor = BatchProcessor(settings)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing audio files...", total=len(audio_files))

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

    if skipped_count > 0 and not force:
        console.print(
            f"[yellow]⏭️  Skipped {skipped_count} file(s) - already transcribed (use --force to re-transcribe)[/yellow]"
        )


@app.command(rich_help_panel=PROCESSING_PANEL)
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


@app.command(name="transcribe-and-generate", rich_help_panel=PROCESSING_PANEL)
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


@app.command(rich_help_panel=PROCESSING_PANEL)
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
    input_device: Optional[str] = typer.Option(
        None,
        "--input-device",
        help="Set input audio device name (e.g. 'Aggregate Device'). Use 'chirp devices' to see available devices.",
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

[cyan]Audio:[/cyan]
Sample Rate: {settings.audio.sample_rate}
Channels: {settings.audio.channels}
Input Device: {settings.audio.input_device or "auto (BlackHole > default)"}

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

    if input_device is not None:
        from recorder.device_manager import DeviceManager

        device_manager = DeviceManager()
        if device_manager.find_device_by_name(input_device):
            settings.audio.input_device = input_device
            changes_made = True
            console.print(f"[green]Input device set to: {input_device}[/green]")
        else:
            console.print(
                f"[red]Device '{input_device}' not found. Use 'chirp devices' to see available devices.[/red]"
            )
            raise typer.Exit(1)

    if changes_made:
        settings.save_to_file(Path("config/config.yaml"))
        settings.ensure_directories_exist()
        console.print("[green]✅ Configuration updated[/green]")


@app.command(rich_help_panel=INFO_PANEL)
def devices():
    """List available audio devices"""
    from recorder.device_manager import DeviceManager

    settings = get_settings()
    device_manager = DeviceManager()
    devices_info = device_manager.list_devices()

    selected_index = device_manager.get_recommended_device(
        configured_device=settings.audio.input_device
    )

    table = Table(title="Available Audio Devices")
    table.add_column("", style="bold")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Channels", style="yellow")
    table.add_column("Default Rate", style="blue")

    for device in devices_info:
        is_selected = device["index"] == selected_index
        marker = "▶" if is_selected else ""
        name_style = "bold green" if is_selected else "green"
        table.add_row(
            marker,
            str(device["index"]),
            Text(device["name"], style=name_style),
            f"In: {device['max_input_channels']}, Out: {device['max_output_channels']}",
            f"{device['default_sample_rate']:.0f} Hz",
        )

    console.print(table)

    if device_manager.check_blackhole_available():
        console.print("[green]✅ BlackHole detected and ready[/green]")
    else:
        console.print("[red]❌ BlackHole not found[/red]")
        console.print("Install from: https://existential.audio/blackhole/")


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


if __name__ == "__main__":
    app()
