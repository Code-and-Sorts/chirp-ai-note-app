import json
import logging
import logging.handlers
import math
import platform
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from typer.core import TyperGroup

import audio_capture
from chirp import exit_codes, glyphs
from chirp._console import apply_color_mode, stderr_console, stdout_console
from chirp.exceptions import AudioDeviceError, ConfigurationError, RecordingError
from config.settings import ChirpSettings, get_settings
from llm.cli.daemon import daemon_app
from llm.cli.models import app as models_app
from llm.registry import resolved_chat_model
from utils.file_utils import NoteRecord, list_notes

logger = logging.getLogger(__name__)

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

    def get_params(self, ctx):
        # Hide the generated completion meta-options to keep the documented
        # 7-command surface.
        params = super().get_params(ctx)
        for param in params:
            opts = set(getattr(param, "opts", [])) | set(
                getattr(param, "secondary_opts", [])
            )
            if opts & _COMPLETION_META_COMMANDS:
                param.hidden = True
        return params


app = typer.Typer(
    name="chirp",
    help="Chirp · AI notes for your terminal.",
    epilog="run `chirp COMMAND --help` for details",
    rich_markup_mode="rich",
    add_completion=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
    },
    cls=OrderedCommandsGroup,
)

# ``console`` aliases stderr: most ``console.print`` sites are diagnostics, and
# data outputs (note tables, search results) print to ``stdout_console``.
console = stderr_console

_COMPLETION_META_COMMANDS = frozenset(
    {
        "--install-completion",
        "--show-completion",
        "install-completion",
        "show-completion",
    }
)

MAIN_PANEL = "Commands"
MODELS_PANEL = "Models"


def _version_callback(value: bool) -> None:
    if value:
        from chirp.about import installed_version

        # Plain stdout line so `chirp --version` pipes cleanly.
        print(f"chirp {installed_version()}")
        raise typer.Exit()


def _cli_log_path() -> Path:
    subdir = (
        Path.home() / "Library" / "Logs" / "chirp"
        if sys.platform == "darwin"
        else Path.home() / ".cache" / "chirp"
    )
    return subdir / "chirp.log"


def _configure_cli_logging(verbose: bool) -> None:
    """Send chirp's logs to a rotating file, and to stderr only with --verbose.

    The ``record`` and live-transcription TUIs drive a Rich ``Live`` display on
    stderr. A logging ``StreamHandler`` on the same stream (the old
    ``basicConfig`` default) interleaves WARNING lines into the live frame,
    breaking its in-place redraw and flashing errors away before they can be
    read. Routing to a file keeps those diagnostics durable and off the TTY.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        if getattr(handler, "_chirp_cli_managed", False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = []

    log_path = _cli_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except OSError:
        verbose = True

    if verbose:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    for handler in handlers:
        handler.setLevel(level)
        handler._chirp_cli_managed = True  # type: ignore[attr-defined]
        root.addHandler(handler)


@contextmanager
def _quiet_tty_logging():
    """Detach TTY-bound log handlers for the duration of a Rich ``Live`` block.

    Even under --verbose, a stderr handler would corrupt the live display, so
    suspend any handler writing to stdout/stderr (the rotating file handler,
    whose stream is the logfile, is left attached) and restore them after.
    """
    root = logging.getLogger()
    suspended = [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler)
        and getattr(handler, "stream", None) in (sys.stderr, sys.stdout)
    ]
    for handler in suspended:
        root.removeHandler(handler)
    try:
        yield
    finally:
        for handler in suspended:
            root.addHandler(handler)


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        is_eager=True,
        callback=_version_callback,
        help="Show the chirp version and exit.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        "--plain",
        help="Disable colored output (also honors the NO_COLOR env var).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        "--debug",
        help="Also echo debug logs to stderr (always written to the chirp log file).",
    ),
) -> None:
    """Chirp · AI notes for your terminal."""
    _configure_cli_logging(verbose)
    apply_color_mode(no_color)


def _prompt_title() -> str:
    console.print()
    console.print(" [yellow bold]title[/yellow bold] [dim](required)[/dim]")
    while True:
        try:
            value: str = console.input(" [green]›[/green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            raise typer.Exit(exit_codes.RUNTIME_ERROR)
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
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
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
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    return _parse_tag_input(value)


def _parse_tag_input(value: str) -> list[str]:
    if not value:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _restore_terminal(fd: int | None, old_settings) -> None:
    """Restore cooked mode and show the cursor — safe to call more than once.

    Shared by the ``record`` ``finally`` block and its SIGTERM handler so a
    ``kill <pid>`` mid-recording leaves the TTY usable instead of stuck in
    cbreak with a hidden cursor.
    """
    if old_settings is not None and fd is not None:
        import termios

        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (termios.error, OSError) as exc:
            logger.debug("could not restore terminal attributes on teardown: %s", exc)
    try:
        console.show_cursor(True)
    except Exception as exc:  # noqa: BLE001 - cursor restore is best-effort on teardown
        logger.debug("could not restore cursor visibility on teardown: %s", exc)


def _resolve_mic_name(device_manager) -> str:
    try:
        default_idx = device_manager.get_default_input_device()
        devices_info = device_manager.list_devices()
        return next(
            (d["name"] for d in devices_info if d["index"] == default_idx),
            "default",
        )
    except (OSError, RuntimeError):
        return "default"


WAVEFORM_GLYPHS = "▁▂▄▅▇█"
WAVEFORM_WIDTH = 28
WAVEFORM_NOISE_FLOOR = 0.01


def _level_to_glyph_index(level: float) -> int:
    """Map an RMS level (0..1) to a glyph index using a sqrt curve.

    A linear mapping pins typical voice (RMS ≈ 0.05–0.3) at glyph 0–1, so
    the bar looks like a flat conveyor. Square-root spreads quiet-but-
    present audio across the full palette, giving the per-cell variation
    that reads as a wave.
    """
    if level <= WAVEFORM_NOISE_FLOOR:
        return -1
    scaled = math.sqrt(min(level, 1.0))
    return min(int(scaled * len(WAVEFORM_GLYPHS)), len(WAVEFORM_GLYPHS) - 1)


def _render_waveform_box(levels: "deque[float]") -> RenderableType:
    bar = Text()
    for slot in levels:
        glyph_idx = _level_to_glyph_index(slot)
        if glyph_idx < 0:
            bar.append("▁", style="dim")
        else:
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

    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if title is None and is_tty:
        title = _prompt_title()
    if duration is None and timeframe is None and is_tty:
        try:
            duration = _prompt_timeframe()
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(exit_codes.USAGE_ERROR)
    elif timeframe is not None and duration is None:
        try:
            duration = parse_timeframe(timeframe)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(exit_codes.USAGE_ERROR)

    resolved_tags: list[str] = list(tags) if tags else []
    if not resolved_tags and is_tty:
        resolved_tags = _prompt_tags()

    if live_transcribe:
        _run_live_transcription(
            settings,
            title,
            duration,
            debug_live=debug_live,
            tags=resolved_tags,
        )
        return

    recorder = AudioRecorder(settings)
    with DeviceManager() as device_manager:
        mic_name = _resolve_mic_name(device_manager)
    state = _RecordViewState(
        title=title,
        cap_minutes=duration,
        mic_name=mic_name,
    )
    control = {"discard": False}

    import signal

    fd: int | None = None
    old_settings = None
    previous_sigterm = None
    previous_sigwinch = None
    resize_pending = {"flag": False}

    try:
        use_cbreak = sys.stdin.isatty() and hasattr(sys.stdin, "fileno")

        if use_cbreak:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        def _on_sigterm(_signum, _frame):
            _restore_terminal(fd, old_settings)
            recorder.stop_recording()

        def _on_sigwinch(_signum, _frame):
            resize_pending["flag"] = True

        previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
        if hasattr(signal, "SIGWINCH"):
            previous_sigwinch = signal.signal(signal.SIGWINCH, _on_sigwinch)

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
            if resize_pending["flag"]:
                # Drop Rich's cached width on resize so it remeasures and the
                # waveform reflows cleanly.
                resize_pending["flag"] = False
                if hasattr(console, "_width"):
                    console._width = None
            live.update(_render_record_view(state), refresh=True)

        try:
            with (
                _quiet_tty_logging(),
                Live(
                    _render_record_view(state),
                    console=console,
                    refresh_per_second=10,
                ) as live,
            ):
                recorder.start_recording(
                    duration_minutes=duration,
                    title=title,
                    level_callback=_on_tick,
                    tags=resolved_tags,
                )
        finally:
            _restore_terminal(fd, old_settings)

        note_dir = recorder.note_dir
        if control["discard"]:
            if note_dir and note_dir.exists():
                shutil.rmtree(note_dir, ignore_errors=True)
            console.print("[yellow]discarded.[/yellow]")
            return

        if note_dir is None:
            console.print("[yellow]nothing to save.[/yellow]")
            return

        console.print(f"[green]saved to {note_dir}[/green]")
        console.print(
            f" [dim]{glyphs.INPUT_ARROW} chirp transcribe    "
            "· turn this into notes[/dim]"
        )

    except KeyboardInterrupt:
        console.print("[yellow]recording stopped by user[/yellow]")
    except AudioDeviceError as e:
        console.print(f"[red]audio device error: {e!s}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    except RecordingError as e:
        console.print(f"[red]recording error: {e!s}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    except ConfigurationError as e:
        console.print(f"[red]configuration error: {e!s}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    except Exception as e:  # noqa: BLE001 - top-level CLI handler; all specific errors caught above
        logger.debug("Unexpected error in record command: %s", e, exc_info=True)
        console.print(f"[red]unexpected error: {e!s}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if previous_sigwinch is not None and hasattr(signal, "SIGWINCH"):
            signal.signal(signal.SIGWINCH, previous_sigwinch)


def _run_live_transcription(
    settings,
    title: str | None,
    duration: int | None,
    debug_live: bool = False,
    tags: list[str] | None = None,
):
    from chirp.exceptions import WhisperModelLoadError
    from recorder.live_session import LiveSessionResult, LiveTranscriptionSession

    if title:
        console.print(f"[cyan]Title: {title}[/cyan]")
    if duration:
        console.print(f"[cyan]Planned duration: {duration} minutes[/cyan]")

    session = LiveTranscriptionSession(
        settings=settings,
        console=console,
        title=title,
        duration_minutes=duration,
        debug=debug_live,
        tags=list(tags or []),
    )

    try:
        with _quiet_tty_logging():
            result: LiveSessionResult = session.run()
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    except WhisperModelLoadError as e:
        console.print(f"[red]{e!s}[/red]")
        raise typer.Exit(exit_codes.MODEL_LOAD_FAILED)
    except RecordingError as e:
        console.print(f"[red]live recording error: {e!s}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)

    from utils.time_utils import format_duration

    console.print()
    console.print("[green]Live recording complete[/green]")
    console.print(f"[dim]Audio saved to:[/dim] {result.audio_path}")
    console.print(
        f"[dim]Duration:[/dim] {format_duration(result.duration_seconds)}  •  [dim]Live words transcribed:[/dim] {result.total_words}"
    )
    if result.dropped_chunks > 0 or result.dropped_frames > 0:
        console.print(
            f"[yellow]⚠ This machine couldn't keep up: "
            f"{result.dropped_chunks} speech chunk(s) and "
            f"{result.dropped_frames} audio frame(s) were dropped, so the live "
            f"transcript may be incomplete. The saved audio.wav is complete and "
            f"unaffected — run 'chirp transcribe' for the full transcript."
            f"[/yellow]"
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
            raise typer.Exit(exit_codes.USAGE_ERROR)
        if force:
            console.print(
                "[red]--regen and --force are mutually exclusive (--force re-transcribes; "
                "--regen reuses existing transcripts).[/red]"
            )
            raise typer.Exit(exit_codes.USAGE_ERROR)
        _run_regen_pipeline(settings)
        return

    if n is not None and n < 1:
        console.print("[red]N must be a positive integer.[/red]")
        raise typer.Exit(exit_codes.USAGE_ERROR)

    if model:
        console.print(f"[cyan]Using Whisper model: {model}[/cyan]")

    from chirp.exceptions import WhisperModelLoadError

    try:
        processor = BatchProcessor(settings, model_override=model)
    except WhisperModelLoadError as e:
        console.print(f"[red]{e!s}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
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
        f"[bold blue]Regenerating notes for {len(records)} record(s)…[/bold blue]"
    )
    result = note_generator.generate_for_records(records, force=True)

    sub_results = result.get("results", [])
    success_count = sum(1 for r in sub_results if r.get("success"))
    if success_count:
        console.print(
            f"[green]Regenerated notes for {success_count}/{len(sub_results)} record(s)[/green]"
        )
    if success_count < len(sub_results):
        failed = [r for r in sub_results if not r.get("success")]
        for failure in failed:
            slug = failure.get("slug", "<unknown>")
            error = failure.get("error", "unknown error")
            console.print(f"[red]  {glyphs.FAILURE} {slug}: {error}[/red]")


notes_app = typer.Typer(help="Browse, view, edit, or delete your notes")
app.add_typer(notes_app, name="notes", rich_help_panel=MAIN_PANEL)

app.add_typer(models_app, name="models", rich_help_panel=MODELS_PANEL)

# Hidden maintenance group: happy-path users never need the daemon, so it stays
# out of `chirp --help` and is surfaced only via `chirp daemon --help`. Mirrors
# the hidden flat commands `config`, `devices`, and `index` defined below.
app.add_typer(
    daemon_app,
    name="daemon",
    help="Daemon lifecycle and diagnostics (hidden).",
    hidden=True,
)


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
    json_output: bool = typer.Option(
        False, "--json", help="Emit the note list as JSON to stdout."
    ),
):
    """Browse, view, edit, or delete your notes"""
    if ctx.invoked_subcommand is not None:
        if tag is not None:
            console.print("[red]--tag is only valid when listing notes.[/red]")
            raise typer.Exit(exit_codes.USAGE_ERROR)
        return
    _list_notes(tag, json_output=json_output)


def _parse_tag_filter(tag: str | None) -> list[str]:
    if not tag:
        return []
    return [piece.strip() for piece in tag.split(",") if piece.strip()]


def _note_to_json(idx: int, record: NoteRecord) -> dict:
    return {
        "id": idx,
        "slug": record.slug,
        "title": _resolve_display_title(record),
        "date": record.created_at.date().isoformat(),
        "tags": list(record.tags),
        "notes_path": str(record.notes) if record.notes else None,
    }


def _list_notes(tag: str | None, json_output: bool = False) -> None:
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

    if json_output:
        payload = [
            _note_to_json(idx, record)
            for idx, record in enumerate(reversed(records), start=1)
        ]
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return

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

        tag_cell = ", ".join(record.tags) if record.tags else glyphs.PENDING
        table.add_row(str(idx), title, date_str, length_str, tag_cell)

    stdout_console.print(table)
    arrow = glyphs.INPUT_ARROW
    console.print()
    console.print(
        f" [dim]{arrow} chirp notes view <id>      · open a note read-only[/dim]"
    )
    console.print(f" [dim]{arrow} chirp notes edit <id>      · edit a note[/dim]")
    console.print(f" [dim]{arrow} chirp notes delete <id>    · delete a note[/dim]")
    console.print(f" [dim]{arrow} chirp notes --tag meeting  · filter by tag[/dim]")


def _resolve_note(records: list[NoteRecord], note_id: str) -> NoteRecord:
    if not note_id or not note_id.strip():
        raise NoteNotFound(note_id)
    cleaned = note_id.strip()

    if cleaned.isdigit():
        index = int(cleaned)
        newest_first = list(reversed(records))
        if 1 <= index <= len(newest_first):
            return newest_first[index - 1]
        raise NoteNotFound(note_id)

    exact = [record for record in records if record.slug == cleaned]
    if len(exact) == 1:
        return exact[0]
    prefix_matches = [record for record in records if record.slug.startswith(cleaned)]
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
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    return records


def _resolve_or_exit(note_id: str) -> NoteRecord:
    records = _load_notes_or_exit()
    try:
        return _resolve_note(records, note_id)
    except NoteNotFound:
        console.print(f"[red]{glyphs.FAILURE} no note matching '{note_id}'[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    except AmbiguousNoteId as exc:
        console.print(
            f"[red]{glyphs.FAILURE} '{note_id}' matches {len(exc.matches)} notes — "
            "be more specific[/red]"
        )
        for slug in exc.matches:
            console.print(f"[dim]  • {slug}[/dim]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)


@notes_app.command("view")
def notes_view(note_id: str = typer.Argument(..., help="Note id (slug or prefix)")):
    """Open a note read-only in the terminal editor."""
    from notes.note_editor import ManualNoteEditor

    record = _resolve_or_exit(note_id)
    if record.notes is None:
        console.print(
            f"[red]{glyphs.FAILURE} note '{record.slug}' has no notes.md[/red]"
        )
        raise typer.Exit(exit_codes.RUNTIME_ERROR)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(
            "[yellow]interactive editor requires a terminal. "
            "please run from an interactive shell.[/yellow]"
        )
        raise typer.Exit(exit_codes.RUNTIME_ERROR)

    title = _resolve_display_title(record)
    content = record.notes.read_text(encoding="utf-8")
    editor = ManualNoteEditor(title, content, readonly=True)
    try:
        editor.run()
    except KeyboardInterrupt:
        console.print("\n[dim]editor cancelled[/dim]")


def _external_editor_command() -> str | None:
    """The user's preferred external editor, if ``$VISUAL`` / ``$EDITOR`` is set."""
    import os

    return os.environ.get("VISUAL") or os.environ.get("EDITOR") or None


def _edit_in_external_editor(notes_path: Path, command: str) -> bool:
    """Open ``notes_path`` in the external editor; True if it edited the file.

    Returns False if the editor could not be launched so the caller can fall
    back to the built-in modal editor.
    """
    import shlex
    import subprocess

    argv = [*shlex.split(command), str(notes_path)]
    try:
        completed = subprocess.run(argv, check=False)
    except (OSError, ValueError) as exc:
        console.print(f"[red]could not launch external editor '{command}': {exc}[/red]")
        return False
    if completed.returncode != 0:
        console.print(
            f"[yellow]external editor exited with status "
            f"{completed.returncode}; notes.md left as the editor saved it.[/yellow]"
        )
    return True


@notes_app.command("edit")
def notes_edit(
    note_id: str = typer.Argument(..., help="Note id (slug or prefix)"),
    use_external_editor: bool = typer.Option(
        False,
        "--editor",
        help="Open notes.md in $VISUAL/$EDITOR instead of the built-in editor.",
    ),
):
    """Edit a note in the terminal editor; saves rewrite notes.md and re-index."""
    from notes.note_editor import ManualNoteEditor

    settings = get_settings()
    record = _resolve_or_exit(note_id)
    if record.notes is None:
        console.print(
            f"[red]{glyphs.FAILURE} note '{record.slug}' has no notes.md[/red]"
        )
        raise typer.Exit(exit_codes.RUNTIME_ERROR)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(
            "[yellow]interactive editor requires a terminal. "
            "please run from an interactive shell.[/yellow]"
        )
        raise typer.Exit(exit_codes.RUNTIME_ERROR)

    if use_external_editor:
        command = _external_editor_command()
        if command is None:
            console.print(
                "[red]--editor needs $VISUAL or $EDITOR set; "
                "neither is. drop --editor to use the built-in editor.[/red]"
            )
            raise typer.Exit(exit_codes.USAGE_ERROR)
        if _edit_in_external_editor(record.notes, command):
            console.print(f"[green]updated note: {record.notes}[/green]")
            if settings.notes_chat.auto_index:
                _reindex_after_edit(settings, record)
            return
        raise typer.Exit(exit_codes.RUNTIME_ERROR)

    title = _resolve_display_title(record)
    content = record.notes.read_text(encoding="utf-8")
    editor = ManualNoteEditor(title, content, start_in_insert=True)
    try:
        result = editor.run()
    except KeyboardInterrupt:
        console.print("\n[dim]editor cancelled[/dim]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)

    if not result.saved:
        console.print("[yellow]changes not saved.[/yellow]")
        return

    record.notes.write_text(result.content, encoding="utf-8")
    console.print(f"[green]updated note: {record.notes}[/green]")

    if settings.notes_chat.auto_index:
        _reindex_after_edit(settings, record)


def _reindex_after_edit(settings: ChirpSettings, record: NoteRecord) -> None:
    notes_path = record.notes
    if notes_path is None:
        return
    try:
        from notes_chat.index import IndexManager

        index_manager = IndexManager(settings)
        if index_manager.add_note(notes_path):
            console.print(
                f"[dim green]{glyphs.SUCCESS} re-indexed {notes_path.name}[/dim green]"
            )
    except Exception as exc:  # noqa: BLE001 - defensive auto-index; IndexManager can raise many types
        logger.debug("Auto-indexing failed for %s: %s", notes_path.name, exc)
        console.print(
            f"[dim yellow]auto-indexing failed for "
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
            console.print("[yellow]deletion cancelled.[/yellow]")
            return

    notes_path = record.notes
    try:
        shutil.rmtree(record.dir, ignore_errors=False)
    except OSError as exc:
        console.print(
            f"[red]{glyphs.FAILURE} failed to delete {record.dir}: {exc}[/red]"
        )
        raise typer.Exit(exit_codes.RUNTIME_ERROR)
    console.print(f"[green]deleted {record.dir}[/green]")

    if notes_path is not None:
        _drop_from_index(settings, notes_path)


def _drop_from_index(settings: ChirpSettings, notes_path: Path) -> None:
    try:
        from notes_chat.index import IndexManager

        index_manager = IndexManager(settings)
        index_manager.remove_note(notes_path)
    except Exception as exc:  # noqa: BLE001 - defensive auto-index; IndexManager can raise many types
        logger.debug("Failed to update index after delete: %s", exc)
        console.print(f"[dim yellow]failed to update index: {exc}[/dim yellow]")


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
    except OSError as exc:
        logger.debug("could not read note title from %s: %s", record.notes, exc)
    return record.slug


@app.command(rich_help_panel=MAIN_PANEL)
def ask(
    question: str | None = typer.Argument(
        None,
        help="Question to ask about your meetings (omit for interactive chat).",
    ),
    question_option: str | None = typer.Option(
        None,
        "--question",
        "-q",
        help="Same as the positional argument; kept for backwards compatibility.",
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
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the answer and sources as JSON to stdout (one-shot only).",
    ),
):
    """Chat with your notes"""
    from notes_chat.cli import ask

    if question is not None and question_option is not None:
        console.print(
            "[yellow]both a positional question and --question given; "
            "using the positional.[/yellow]"
        )
    resolved = question if question is not None else question_option
    ask(
        question=resolved,
        question_option=None,
        when=when,
        sources=sources,
        dry_run=dry_run,
        markdown=markdown,
        json_output=json_output,
    )


@app.command(rich_help_panel=MAIN_PANEL)
def search(
    query: str = typer.Argument(
        ..., help="Substring (or regex with --regex) to search for."
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Only notes from the last DURATION (e.g. 30d, 2w, 48h).",
    ),
    regex: bool = typer.Option(False, "--regex", help="Treat QUERY as a Python regex."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON and skip the rendered output."
    ),
):
    """Keyword search across transcripts and notes."""
    from notes_chat.search_keyword import (
        SearchOptions,
        render_no_matches,
        render_results,
        run_search,
        suggest_close_keywords,
        write_json,
    )
    from utils.time_utils import parse_since

    if not query or not query.strip():
        console.print("[red]search query is required.[/red]")
        raise typer.Exit(exit_codes.USAGE_ERROR)

    since_minutes: int | None = None
    if since is not None:
        try:
            since_minutes = parse_since(since)
        except ValueError as exc:
            console.print(f"[red]invalid --since: {exc}[/red]")
            raise typer.Exit(exit_codes.USAGE_ERROR)

    if regex:
        try:
            __import__("re").compile(query)
        except __import__("re").error as exc:
            console.print(f"[red]invalid regex: {exc.msg}[/red]")
            raise typer.Exit(exit_codes.USAGE_ERROR)

    settings = get_settings()
    options = SearchOptions(
        query=query,
        since_minutes=since_minutes,
        regex=regex,
        json=json_output,
    )

    result = run_search(settings, options)

    if json_output:
        write_json(result)
        return

    if result["matches"]:
        render_results(stdout_console, console, options, result)
        return

    bm25_path = settings.notes_chat.index_dir / "bm25.json"
    suggestions = suggest_close_keywords(bm25_path, query)
    render_no_matches(console, options, result["total_notes_scanned"], suggestions)


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
    from chirp.init_flow import require_apple_silicon, run_init

    # Gate before get_settings(): loading settings creates a default
    # config.toml when missing, and a non-arm64 machine must exit without
    # touching the filesystem (story 7.1 AC-1). run_init re-checks the gate
    # for callers that bypass the CLI.
    code = require_apple_silicon(console)
    if code is not None:
        raise typer.Exit(code)

    settings = get_settings()
    code = run_init(settings, console, recheck=recheck, switch_model=switch_model)
    if code != 0:
        raise typer.Exit(code)


if __name__ == "__main__":
    app()


@app.command(rich_help_panel=MAIN_PANEL)
def about(
    plain: bool = typer.Option(
        False,
        "--plain",
        "--no-animate",
        help="Print the static info panel without the animation (scripted/non-TTY).",
    ),
):
    """Show the bird 🐦"""
    from chirp.about import render_about_plain, run_about

    settings = get_settings()
    if plain:
        render_about_plain(stdout_console, settings)
        return
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
LLM: {resolved_chat_model(settings.models.llm)}
Embedding: {settings.notes_chat.emb_model}

[cyan]Audio:[/cyan]
Sample Rate: {settings.audio.sample_rate}
Channels: {settings.audio.channels}

[cyan]Monitoring:[/cyan]
Warning: {settings.monitoring.warning_minutes} minutes
Interval: {settings.monitoring.warning_interval} minutes""",
            title="Chirp Configuration",
        )
        stdout_console.print(panel)
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

    if embedding_model:
        settings.notes_chat.emb_model = embedding_model
        changes_made = True

    if changes_made:
        settings.save_to_file(ChirpSettings.get_config_path())
        settings.ensure_directories_exist()
        console.print("[green]configuration updated[/green]")


@app.command(hidden=True)
def devices():
    """List available audio devices"""
    from recorder.device_manager import DeviceManager

    with DeviceManager() as device_manager:
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

    stdout_console.print(input_table)
    stdout_console.print()

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

    stdout_console.print(output_table)

    console.print()
    console.print(
        "System audio is captured via the bundled Chirp.app helper; no virtual driver required."
    )
    if platform.system() == "Darwin":
        try:
            perms = audio_capture.check_permissions()
            state = perms.get("screen_recording")
            if state == "granted":
                console.print(f"[green]{glyphs.SUCCESS} capture ready[/green]")
            elif state == "undetermined":
                console.print(
                    f"[yellow]{glyphs.PENDING} capture will prompt on first "
                    "record[/yellow]"
                )
            elif state == "denied":
                console.print(
                    f"[red]{glyphs.FAILURE} screen recording permission denied[/red]"
                    " — open System Settings → Privacy & Security → Screen Recording"
                )
            else:
                console.print(
                    f"[red]{glyphs.FAILURE} capture_audio probe returned unexpected "
                    "state[/red] — rebuild with python -m audio_capture.build"
                )
        except FileNotFoundError:
            console.print(
                f"[red]{glyphs.FAILURE} capture_audio binary missing[/red] — "
                "run python -m audio_capture.build"
            )
        except RuntimeError as exc:
            console.print(
                f"[red]{glyphs.FAILURE} permission probe failed[/red]: {exc} — "
                "try rebuilding with python -m audio_capture.build"
            )
    else:
        console.print(
            f"[dim]{glyphs.PENDING} capture status not applicable on "
            f"{platform.system()}[/dim]"
        )
    console.print("[dim]Run 'chirp init' for first-run setup.[/dim]")
