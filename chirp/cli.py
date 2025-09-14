from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from chirp.exceptions import *
from config.settings import get_settings
from notes.note_generator import NoteGenerator
from recorder.audio_recorder import AudioRecorder
from recorder.device_manager import DeviceManager
from transcriber.batch_processor import BatchProcessor
from utils.file_utils import get_audio_files, get_transcription_files

app = typer.Typer(
    name="chirp",
    help="🐦 Chirp - Meeting Recorder CLI that transcribes and generates AI notes",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def record(
    duration: Optional[int] = typer.Option(
        None, "--duration", "-d", help="Recording duration in minutes"
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Meeting title for filename"
    ),
):
    """🎙️ Start recording a meeting"""
    settings = get_settings()
    device_manager = DeviceManager()

    if not device_manager.check_blackhole_available():
        console.print(
            "[red]❌ BlackHole not detected. Please install BlackHole audio driver.[/red]"
        )
        console.print("Download from: https://existential.audio/blackhole/")
        raise typer.Exit(1)

    recorder = AudioRecorder(settings, device_manager)

    try:
        with console.status("[bold green]🎙️ Recording in progress..."):
            filename = recorder.start_recording(duration_minutes=duration, title=title)

        console.print(f"[green]✅ Recording saved: {filename}[/green]")
        console.print("[dim]Use 'chirp transcribe' to process this recording[/dim]")

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


@app.command()
def process(
    input_dir: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input directory for audio files"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Process all files, including already processed ones",
    ),
):
    """⚡ Process audio files (transcribe + generate notes)"""
    get_settings()

    # Call commands directly since app.commands doesn't exist in Typer
    # This is a simplified process command that manually calls both operations
    try:
        # Call transcribe functionality
        transcribe(input_dir=input_dir, force=force)
        # Call notes functionality
        notes(force=force)
    except Exception as e:
        console.print(f"[red]Error in process command: {e}[/red]")


@app.command()
def transcribe(
    input_dir: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input directory for audio files"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-transcribe already processed files"
    ),
):
    """📝 Transcribe audio files to text"""
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


@app.command()
def notes(
    force: bool = typer.Option(
        False, "--force", "-f", help="Regenerate notes even if they exist"
    ),
):
    """📋 Generate meeting notes from transcriptions"""
    settings = get_settings()

    transcription_files = get_transcription_files(settings.directories.transcriptions)

    if not transcription_files:
        console.print(
            f"[yellow]No transcription files found in {settings.directories.transcriptions}[/yellow]"
        )
        console.print("[dim]Run 'chirp transcribe' first[/dim]")
        return

    note_generator = NoteGenerator(settings)

    with console.status("[bold blue]🧠 Generating notes with AI..."):
        result = note_generator.generate_daily_notes(transcription_files, force=force)

    if result["success"]:
        console.print(f"[green]✅ Notes generated: {result['filename']}[/green]")
    else:
        console.print(f"[red]❌ Failed to generate notes: {result['error']}[/red]")


@app.command()
def status():
    """📊 Show Chirp status and statistics"""
    settings = get_settings()

    table = Table(title="🐦 Chirp Status", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Value", style="green")

    audio_files = get_audio_files(settings.directories.raw_audio)
    transcription_files = get_transcription_files(settings.directories.transcriptions)

    table.add_row("Raw Audio Files", str(len(audio_files)))
    table.add_row("Transcriptions", str(len(transcription_files)))
    table.add_row("Audio Directory", str(settings.directories.raw_audio))
    table.add_row("Transcription Directory", str(settings.directories.transcriptions))
    table.add_row("Notes Directory", str(settings.directories.notes))
    table.add_row("Whisper Model", settings.models.whisper)
    table.add_row("LLM Model", settings.models.llm)

    console.print(table)


@app.command()
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
):
    """⚙️ Manage Chirp configuration"""
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

[cyan]Monitoring:[/cyan]
Warning: {settings.monitoring.warning_minutes} minutes
Interval: {settings.monitoring.warning_interval} minutes""",
            title="🐦 Chirp Configuration",
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

    if changes_made:
        settings.save_to_file(Path("config/config.yaml"))
        settings.ensure_directories_exist()
        console.print("[green]✅ Configuration updated[/green]")


@app.command()
def devices():
    """🎵 List available audio devices"""
    device_manager = DeviceManager()
    devices_info = device_manager.list_devices()

    table = Table(title="🎵 Available Audio Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Channels", style="yellow")
    table.add_column("Default Rate", style="blue")

    for device in devices_info:
        table.add_row(
            str(device["index"]),
            device["name"],
            f"In: {device['max_input_channels']}, Out: {device['max_output_channels']}",
            f"{device['default_sample_rate']:.0f} Hz",
        )

    console.print(table)

    if device_manager.check_blackhole_available():
        console.print("[green]✅ BlackHole detected and ready[/green]")
    else:
        console.print("[red]❌ BlackHole not found[/red]")
        console.print("Install from: https://existential.audio/blackhole/")


@app.command()
def test():
    """🧪 Test Chirp dependencies and configuration"""
    console.print("[bold]🧪 Testing Chirp Dependencies...[/bold]")

    tests = []

    device_manager = DeviceManager()
    tests.append(("PyAudio", device_manager.test_pyaudio()))
    tests.append(("BlackHole", device_manager.check_blackhole_available()))

    try:
        from faster_whisper import WhisperModel  # noqa: F401

        tests.append(("Faster Whisper", True))
    except ImportError:
        tests.append(("Faster Whisper", False))

    try:
        import requests

        response = requests.get("http://localhost:11434/api/version", timeout=5)
        tests.append(("Ollama", response.status_code == 200))
    except:
        tests.append(("Ollama", False))

    settings = get_settings()
    settings.ensure_directories_exist()
    tests.append(("Directories", True))

    table = Table(title="🧪 Dependency Test Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")

    all_passed = True
    for name, passed in tests:
        status = "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]"
        table.add_row(name, status)
        if not passed:
            all_passed = False

    console.print(table)

    if all_passed:
        console.print("\n[green]🎉 All tests passed! Chirp is ready to use.[/green]")
    else:
        console.print(
            "\n[red]⚠️  Some tests failed. Please check the installation.[/red]"
        )


notes_app = typer.Typer(help="Notes search and query commands")
app.add_typer(notes_app, name="notes")


@notes_app.command("index")
def notes_index(
    force: bool = typer.Option(False, "--force", help="Force full rebuild of index"),
):
    """Build or rebuild the notes search index."""
    from notes_chat.cli import index

    index(force)


@notes_app.command("ask")
def notes_ask(
    question: str = typer.Argument(..., help="Question to ask about your meetings"),
    when: Optional[str] = typer.Option(None, "--when", help="Time range filter"),
    sources: bool = typer.Option(True, "--sources/--no-sources", help="Show sources"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show prompt without calling LLM"
    ),
):
    """Ask questions about your meeting notes."""
    from notes_chat.cli import ask

    ask(question, when, sources, dry_run)


if __name__ == "__main__":
    app()
