from __future__ import annotations

import logging
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import tomli_w
from rich.console import Console, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from config.settings import ChirpSettings
from transcriber.whisper_transcriber import WhisperTranscriber
from utils.file_utils import (
    META_FILENAME,
    NOTES_FILENAME,
    TRANSCRIPT_FILENAME,
    NoteRecord,
    list_notes,
)
from utils.popup_manager import PopupManager

logger = logging.getLogger(__name__)


class Stage(Enum):
    LOAD_AUDIO = ("loaded audio", 0)
    TRANSCRIBE = ("transcribe", 1)
    GENERATE_NOTES = ("generate notes", 2)
    INDEX = ("index notes", 3)
    SAVE = ("save", 4)

    def __init__(self, label: str, idx: int) -> None:
        self.label = label
        self.idx = idx


class StageState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class StageStatus:
    state: StageState = StageState.PENDING
    detail: str | None = None


class ChecklistView:
    """Renders the 5-stage checklist plus an optional batch header line."""

    def __init__(self, header: str | None = None) -> None:
        self.header = header
        self.statuses: dict[Stage, StageStatus] = {
            stage: StageStatus() for stage in Stage
        }
        self._running_spinner = Spinner("dots", style="yellow")

    def start(self, stage: Stage) -> None:
        self.statuses[stage] = StageStatus(StageState.RUNNING)

    def done(self, stage: Stage, detail: str | None = None) -> None:
        self.statuses[stage] = StageStatus(StageState.DONE, detail)

    def fail(self, stage: Stage, error: str) -> None:
        self.statuses[stage] = StageStatus(StageState.FAILED, error)

    def render(self) -> RenderableType:
        body = Table.grid(padding=(0, 0))
        body.add_column(no_wrap=True, width=2)
        body.add_column(no_wrap=False)
        if self.header:
            body.add_row(Text(""), Text(self.header, style="bold white"))
            body.add_row(Text(""), Text(""))
        for stage in Stage:
            icon, label = self._render_stage_row(stage)
            body.add_row(icon, label)
        return body

    def _render_stage_row(self, stage: Stage) -> tuple[RenderableType, Text]:
        status = self.statuses[stage]
        if status.state is StageState.DONE:
            icon: RenderableType = Text("✓ ", style="green")
            label = Text(stage.label)
            if status.detail:
                label.append(f" · {status.detail}", style="dim")
        elif status.state is StageState.RUNNING:
            icon = self._running_spinner
            label = Text(stage.label, style="cyan")
        elif status.state is StageState.FAILED:
            icon = Text("✗ ", style="red bold")
            label = Text(stage.label, style="red")
            if status.detail:
                label.append(f" · {status.detail}", style="red")
        else:
            icon = Text("○ ", style="dim")
            label = Text(stage.label, style="dim")
        return icon, label


def _format_duration(seconds: float) -> str:
    total = round(max(seconds, 0.0))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _format_size_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def _audio_duration_seconds(transcriber: WhisperTranscriber, audio: Path) -> float:
    """Resolve clip duration in seconds.

    Prefers the `duration_s` field in `meta.toml` (written by the recorder
    after the file is finalized). Falls back to reading the WAV header so
    notes created before that field existed still report a real length.
    """
    metadata = transcriber._read_audio_metadata(audio)
    if metadata:
        for key in ("duration_s", "duration"):
            value = metadata.get(key)
            if value is None:
                continue
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                continue
            if seconds > 0:
                return seconds

    try:
        import wave

        with wave.open(str(audio), "rb") as fh:
            frames = fh.getnframes()
            rate = fh.getframerate()
            if rate > 0:
                return frames / float(rate)
    except (OSError, wave.Error):
        # Probe is best-effort; fall through to the 0.0 default.
        pass

    return 0.0


@dataclass
class _PipelineContext:
    record: NoteRecord
    view: ChecklistView
    duration_seconds: float | None = None
    transcript_words: int | None = None


class BatchProcessor:
    """Drives the 5-stage transcribe pipeline for a FIFO queue of notes.

    Public entry point is :meth:`run_queue`. Every note flows through the
    same sequence; failures in one record do not abort the batch.
    """

    def __init__(self, settings: ChirpSettings, model_override: str | None = None):
        if model_override:
            settings = settings.model_copy(
                update={
                    "models": settings.models.model_copy(
                        update={"whisper": model_override}
                    )
                }
            )
        self.settings = settings
        self.transcriber = WhisperTranscriber(settings)
        self.popup_manager = PopupManager()

    def run_queue(
        self,
        n: int | None = None,
        force: bool = False,
        console: Console | None = None,
    ) -> dict[str, int]:
        console = console or Console()
        records = self._select_queue(n=n, force=force)

        if not records:
            console.print(
                "[yellow]No notes to transcribe. Run `chirp record` first, or "
                "pass --force to re-run completed notes.[/yellow]"
            )
            return {"ok": 0, "failed": 0, "total": 0}

        ok = 0
        failed = 0
        total = len(records)
        for index, record in enumerate(records, start=1):
            header = self._format_header(index, total, record)
            view = ChecklistView(header)
            ctx = _PipelineContext(record=record, view=view)
            with Live(view.render(), console=console, refresh_per_second=12) as live:
                success = self._process_one(ctx, live)
            if success:
                ok += 1
            else:
                failed += 1

        if ok > 0:
            self.popup_manager.show_transcription_complete(ok)

        console.print()
        if failed:
            console.print(
                f"[bold]done[/bold] · [green]{ok} ok[/green] · "
                f"[red]{failed} failed[/red]"
            )
        else:
            console.print(f"[bold]done[/bold] · [green]{ok} ok[/green]")

        return {"ok": ok, "failed": failed, "total": total}

    def _select_queue(self, n: int | None, force: bool) -> list[NoteRecord]:
        records = list_notes(self.settings.directories.notes_root)
        candidates = [
            record
            for record in records
            if record.audio is not None and (force or not _is_complete(record))
        ]
        if n is not None:
            candidates = candidates[:n]
        return candidates

    def _format_header(self, index: int, total: int, record: NoteRecord) -> str:
        title = record.title or record.slug
        if total > 1:
            return f"{index} of {total} · {title}"
        return title

    def _process_one(self, ctx: _PipelineContext, live: Live) -> bool:
        steps: list[tuple[Stage, Callable[[_PipelineContext], None]]] = [
            (Stage.LOAD_AUDIO, self._stage_load_audio),
            (Stage.TRANSCRIBE, self._stage_transcribe),
            (Stage.GENERATE_NOTES, self._stage_generate_notes),
            (Stage.INDEX, self._stage_index),
            (Stage.SAVE, self._stage_save),
        ]
        for stage, runner in steps:
            ctx.view.start(stage)
            live.update(ctx.view.render())
            try:
                runner(ctx)
            except Exception as exc:  # noqa: BLE001 - pipeline stage; any stage can fail for many reasons
                logger.debug("Pipeline stage %s failed: %s", stage, exc)
                ctx.view.fail(stage, str(exc))
                live.update(ctx.view.render())
                return False
            live.update(ctx.view.render())
        return True

    def _stage_load_audio(self, ctx: _PipelineContext) -> None:
        audio = ctx.record.audio
        if audio is None or not audio.exists():
            raise FileNotFoundError(f"audio missing for {ctx.record.slug}")
        ctx.duration_seconds = _audio_duration_seconds(self.transcriber, audio)
        ctx.view.done(
            Stage.LOAD_AUDIO,
            f"{_format_duration(ctx.duration_seconds)} · {_format_size_mb(audio)}",
        )

    def _stage_transcribe(self, ctx: _PipelineContext) -> None:
        audio = ctx.record.audio
        assert audio is not None
        transcript_path = ctx.record.dir / TRANSCRIPT_FILENAME
        whisper_model = self.settings.models.whisper

        if _has_transcript(ctx.record):
            existing = transcript_path.read_text(encoding="utf-8")
            ctx.transcript_words = len(existing.split())
            ctx.view.done(
                Stage.TRANSCRIBE,
                f"{whisper_model} · {ctx.transcript_words} words · resumed",
            )
            return

        result = self.transcriber.transcribe_file(audio)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "whisper failed")
        full_text = result.get("full_text", "") or ""
        transcript_path.write_text(full_text, encoding="utf-8")
        ctx.transcript_words = len(full_text.split())
        ctx.view.done(
            Stage.TRANSCRIBE,
            f"{whisper_model} · {ctx.transcript_words} words",
        )

    def _stage_generate_notes(self, ctx: _PipelineContext) -> None:
        from notes.note_generator import NoteGenerator

        refreshed = _reload_record(self.settings, ctx.record.slug)
        if refreshed is None or refreshed.transcript is None:
            raise RuntimeError("transcript not found after stage 2")
        # Silence NoteGenerator's chunk-counter prints; they break Live's
        # in-place rendering and make the title appear multiple times in
        # scrollback.
        quiet_console = Console(quiet=True)
        result = NoteGenerator(
            self.settings, console=quiet_console
        ).generate_for_records([refreshed], force=True)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "note generation failed")
        ctx.view.done(Stage.GENERATE_NOTES, self.settings.models.llm)

    def _stage_index(self, ctx: _PipelineContext) -> None:
        if not self.settings.notes_chat.auto_index:
            ctx.view.done(Stage.INDEX, "auto-index off")
            return
        from notes_chat.index import IndexManager

        notes_path = ctx.record.dir / NOTES_FILENAME
        if not notes_path.exists():
            raise RuntimeError(f"{NOTES_FILENAME} not found after stage 3")
        index_manager = IndexManager(self.settings)
        if not index_manager._add_to_index(notes_path):
            raise RuntimeError("indexing failed")
        manifest = index_manager._load_manifest()
        current_files = index_manager._scan_notes_files()
        file_path = str(notes_path)
        if file_path in current_files:
            manifest[file_path] = current_files[file_path]
            index_manager._save_manifest(manifest)
        index_manager._rebuild_bm25()
        ctx.view.done(Stage.INDEX)

    def _stage_save(self, ctx: _PipelineContext) -> None:
        meta_path = ctx.record.dir / META_FILENAME
        meta = _read_meta(meta_path)
        meta["whisper_model"] = self.settings.models.whisper
        meta["llm_model"] = self.settings.models.llm
        meta["indexed_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
        if ctx.duration_seconds is not None:
            meta["duration_s"] = round(ctx.duration_seconds, 2)
        _write_meta(meta_path, meta)
        ctx.view.done(Stage.SAVE)


def _reload_record(settings: ChirpSettings, slug: str) -> NoteRecord | None:
    for record in list_notes(settings.directories.notes_root):
        if record.slug == slug:
            return record
    return None


def _has_transcript(record: NoteRecord) -> bool:
    if record.transcript is None:
        return False
    try:
        return record.transcript.stat().st_size > 0
    except OSError:
        return False


def _has_notes(record: NoteRecord) -> bool:
    if record.notes is None:
        return False
    try:
        return record.notes.stat().st_size > 0
    except OSError:
        return False


def _is_complete(record: NoteRecord) -> bool:
    """A record is 'complete' once it has both a transcript and a notes file.

    Records partway through the pipeline (transcript without notes, or notes
    without transcript) are still queued so the pipeline can resume from
    where it left off without re-running Whisper.
    """
    return _has_transcript(record) and _has_notes(record)


def _read_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("rb") as fh:
            return dict(tomllib.load(fh))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("wb") as fh:
        tomli_w.dump(meta, fh)
