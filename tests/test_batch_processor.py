from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import tomli_w
from rich.console import Console

from transcriber.batch_processor import (
    BatchProcessor,
    ChecklistView,
    Stage,
    StageState,
    _has_transcript,
    _read_meta,
    _StageSkipped,
    _write_meta,
)
from utils.file_utils import NoteRecord


@pytest.fixture(autouse=True)
def _isolate_resolved_chat_model(monkeypatch):
    """Pin ``resolved_chat_model`` to its fallback so meta/label assertions don't
    depend on the host's ``models.toml``."""
    monkeypatch.setattr(
        "transcriber.batch_processor.resolved_chat_model", lambda fallback: fallback
    )


def _seed_record_dir(tmp_path: Path, slug: str, with_transcript: bool = False) -> Path:
    note_dir = tmp_path / slug
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "audio.wav").write_bytes(b"audio data")
    if with_transcript:
        (note_dir / "transcript.txt").write_text("hi", encoding="utf-8")
    with (note_dir / "meta.toml").open("wb") as fh:
        tomli_w.dump(
            {
                "title": slug,
                "date": "2026-04-20T09:00:00",
                "tags": [],
            },
            fh,
        )
    return note_dir


def _settings(tmp_path: Path):
    settings = Mock()
    settings.directories = Mock()
    settings.directories.notes_root = tmp_path
    settings.models = Mock()
    settings.models.whisper = "tiny"
    settings.models.llm = "llama3.1:8b"
    settings.notes_chat = Mock()
    settings.notes_chat.auto_index = False
    return settings


def _processor(tmp_path: Path):
    with (
        patch("transcriber.batch_processor.WhisperTranscriber"),
        patch("transcriber.batch_processor.PopupManager"),
    ):
        return BatchProcessor(_settings(tmp_path))


class TestChecklistView:
    def test_renders_pending_then_running_then_done(self):
        view = ChecklistView(header="1 of 2 · Demo")
        view.start(Stage.LOAD_AUDIO)
        view.done(Stage.LOAD_AUDIO, "00:42 · 1.2 MB")
        assert view.statuses[Stage.LOAD_AUDIO].state is StageState.DONE
        assert view.statuses[Stage.TRANSCRIBE].state is StageState.PENDING

    def test_render_emits_checklist_lines(self):
        from io import StringIO

        view = ChecklistView(header="hello")
        view.done(Stage.LOAD_AUDIO, "00:42 · 1.2 MB")
        view.start(Stage.TRANSCRIBE)
        buffer = StringIO()
        Console(file=buffer, force_terminal=False, width=80).print(view.render())
        output = buffer.getvalue()
        assert "hello" in output
        assert "loaded audio" in output
        assert "00:42" in output
        assert "transcribe" in output

    def test_skip_renders_muted_detail_not_failure(self):
        from io import StringIO

        view = ChecklistView(header="clip")
        view.skip(Stage.GENERATE_NOTES, "recording too short to summarize")
        assert view.statuses[Stage.GENERATE_NOTES].state is StageState.SKIPPED
        buffer = StringIO()
        Console(file=buffer, force_terminal=False, width=80).print(view.render())
        output = buffer.getvalue()
        assert "recording too short to summarize" in output
        assert "✗" not in output


class TestSelectQueue:
    def test_select_queue_orders_by_list_notes(self, tmp_path):
        # list_notes sorts by created_at; meta.toml date drives order.
        first = _seed_record_dir(tmp_path, "first")
        second = _seed_record_dir(tmp_path, "second")
        with (first / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {"title": "first", "date": "2026-04-20T09:00:00", "tags": []}, fh
            )
        with (second / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {"title": "second", "date": "2026-04-20T10:00:00", "tags": []}, fh
            )
        proc = _processor(tmp_path)

        queue = proc._select_queue(n=None, force=False)
        assert [record.slug for record in queue] == ["first", "second"]

    def test_select_queue_skips_completed_notes(self, tmp_path):
        _seed_record_dir(tmp_path, "fresh")
        done_dir = _seed_record_dir(tmp_path, "done", with_transcript=True)
        (done_dir / "notes.md").write_text("# done", encoding="utf-8")
        proc = _processor(tmp_path)

        queue = proc._select_queue(n=None, force=False)
        assert [record.slug for record in queue] == ["fresh"]

    def test_force_includes_completed_notes(self, tmp_path):
        _seed_record_dir(tmp_path, "fresh")
        done_dir = _seed_record_dir(tmp_path, "done", with_transcript=True)
        (done_dir / "notes.md").write_text("# done", encoding="utf-8")
        proc = _processor(tmp_path)

        queue = proc._select_queue(n=None, force=True)
        assert {record.slug for record in queue} == {"fresh", "done"}

    def test_n_caps_queue_to_oldest(self, tmp_path):
        for slug, hour in [("a", 9), ("b", 10), ("c", 11)]:
            note_dir = _seed_record_dir(tmp_path, slug)
            with (note_dir / "meta.toml").open("wb") as fh:
                tomli_w.dump(
                    {
                        "title": slug,
                        "date": f"2026-04-20T{hour:02d}:00:00",
                        "tags": [],
                    },
                    fh,
                )
        proc = _processor(tmp_path)

        queue = proc._select_queue(n=2, force=False)
        assert [record.slug for record in queue] == ["a", "b"]


class TestRunQueueIntegration:
    def _stub_stages(self, monkeypatch, fail_slug: str | None = None):
        """Replace the heavy stages with deterministic stubs."""

        def fake_load_audio(self, ctx):
            ctx.duration_seconds = 1.5
            ctx.view.done(Stage.LOAD_AUDIO, "00:02 · 0.0 MB")

        monkeypatch.setattr(BatchProcessor, "_stage_load_audio", fake_load_audio)

        def fake_transcribe(self, ctx):
            transcript_path = ctx.record.dir / "transcript.txt"
            transcript_path.write_text("hello world", encoding="utf-8")
            ctx.transcript_words = 2
            ctx.view.done(Stage.TRANSCRIBE, "2 words")

        monkeypatch.setattr(BatchProcessor, "_stage_transcribe", fake_transcribe)

        def fake_generate(self, ctx):
            if fail_slug and ctx.record.slug == fail_slug:
                raise RuntimeError("note generation exploded")
            (ctx.record.dir / "notes.md").write_text("# notes", encoding="utf-8")
            ctx.view.done(Stage.GENERATE_NOTES)

        monkeypatch.setattr(BatchProcessor, "_stage_generate_notes", fake_generate)
        monkeypatch.setattr(
            BatchProcessor,
            "_stage_index",
            lambda self, ctx: ctx.view.done(Stage.INDEX, "auto-index off"),
        )

    def test_run_queue_writes_meta_for_each_record(self, tmp_path, monkeypatch):
        for slug, hour in [("alpha", 9), ("beta", 10)]:
            note_dir = _seed_record_dir(tmp_path, slug)
            with (note_dir / "meta.toml").open("wb") as fh:
                tomli_w.dump(
                    {
                        "title": slug,
                        "date": f"2026-04-20T{hour:02d}:00:00",
                        "tags": [],
                    },
                    fh,
                )

        self._stub_stages(monkeypatch)
        proc = _processor(tmp_path)

        result = proc.run_queue(console=Console(force_terminal=False))
        assert result == {"ok": 2, "skipped": 0, "failed": 0, "total": 2}

        for slug in ("alpha", "beta"):
            transcript = tmp_path / slug / "transcript.txt"
            notes = tmp_path / slug / "notes.md"
            meta = _read_meta(tmp_path / slug / "meta.toml")
            assert transcript.exists()
            assert notes.exists()
            assert meta["whisper_model"] == "tiny"
            assert meta["llm_model"] == "llama3.1:8b"
            assert "indexed_at" in meta
            assert meta["duration_s"] == 1.5

    def test_run_queue_failure_continues_to_next_record(self, tmp_path, monkeypatch):
        for slug, hour in [("first", 9), ("second", 10)]:
            note_dir = _seed_record_dir(tmp_path, slug)
            with (note_dir / "meta.toml").open("wb") as fh:
                tomli_w.dump(
                    {
                        "title": slug,
                        "date": f"2026-04-20T{hour:02d}:00:00",
                        "tags": [],
                    },
                    fh,
                )

        self._stub_stages(monkeypatch, fail_slug="first")
        proc = _processor(tmp_path)

        result = proc.run_queue(console=Console(force_terminal=False))
        assert result == {"ok": 1, "skipped": 0, "failed": 1, "total": 2}
        # second one still got its notes file
        assert (tmp_path / "second" / "notes.md").exists()
        # first one did NOT — generation failed
        assert not (tmp_path / "first" / "notes.md").exists()

    def test_run_queue_respects_n(self, tmp_path, monkeypatch):
        for slug, hour in [("a", 9), ("b", 10), ("c", 11)]:
            note_dir = _seed_record_dir(tmp_path, slug)
            with (note_dir / "meta.toml").open("wb") as fh:
                tomli_w.dump(
                    {
                        "title": slug,
                        "date": f"2026-04-20T{hour:02d}:00:00",
                        "tags": [],
                    },
                    fh,
                )

        self._stub_stages(monkeypatch)
        proc = _processor(tmp_path)

        result = proc.run_queue(n=2, console=Console(force_terminal=False))
        assert result == {"ok": 2, "skipped": 0, "failed": 0, "total": 2}
        assert (tmp_path / "a" / "notes.md").exists()
        assert (tmp_path / "b" / "notes.md").exists()
        assert not (tmp_path / "c" / "notes.md").exists()

    def test_run_queue_counts_skip_not_failure(self, tmp_path, monkeypatch):
        _seed_record_dir(tmp_path, "blip")
        self._stub_stages(monkeypatch)

        def skip_generate(self, ctx):
            raise _StageSkipped("recording too short to summarize")

        monkeypatch.setattr(BatchProcessor, "_stage_generate_notes", skip_generate)
        proc = _processor(tmp_path)

        result = proc.run_queue(console=Console(force_terminal=False))

        assert result == {"ok": 0, "skipped": 1, "failed": 0, "total": 1}
        # A skipped record runs no later stages, so no notes are written.
        assert not (tmp_path / "blip" / "notes.md").exists()


class TestMetaIO:
    def test_read_missing_returns_empty(self, tmp_path):
        assert _read_meta(tmp_path / "nope.toml") == {}

    def test_round_trip(self, tmp_path):
        path = tmp_path / "meta.toml"
        _write_meta(path, {"a": 1, "tags": ["x"]})
        assert _read_meta(path) == {"a": 1, "tags": ["x"]}


class TestHasTranscript:
    def test_missing_transcript(self):
        record = NoteRecord(
            slug="x",
            dir=Path("/tmp/x"),
            audio=None,
            transcript=None,
            notes=None,
            meta=None,
            created_at=datetime(2026, 4, 20),
        )
        assert _has_transcript(record) is False

    def test_empty_transcript(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("", encoding="utf-8")
        record = NoteRecord(
            slug="x",
            dir=tmp_path,
            audio=None,
            transcript=path,
            notes=None,
            meta=None,
            created_at=datetime(2026, 4, 20),
        )
        assert _has_transcript(record) is False

    def test_present_transcript(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("hello", encoding="utf-8")
        record = NoteRecord(
            slug="x",
            dir=tmp_path,
            audio=None,
            transcript=path,
            notes=None,
            meta=None,
            created_at=datetime(2026, 4, 20),
        )
        assert _has_transcript(record) is True


class TestResumeFromFailure:
    def test_select_queue_includes_transcript_without_notes(self, tmp_path):
        # Stage 3 failed last run: transcript exists, notes does not.
        partial = _seed_record_dir(tmp_path, "partial", with_transcript=True)
        complete = _seed_record_dir(tmp_path, "complete", with_transcript=True)
        (complete / "notes.md").write_text("# done", encoding="utf-8")

        with (
            patch("transcriber.batch_processor.WhisperTranscriber"),
            patch("transcriber.batch_processor.PopupManager"),
        ):
            proc = BatchProcessor(_settings(tmp_path))

        queue = proc._select_queue(n=None, force=False)
        slugs = {record.slug for record in queue}
        assert "partial" in slugs
        assert "complete" not in slugs
        assert partial.exists()  # untouched

    def test_resume_skips_whisper_when_transcript_exists(self, tmp_path, monkeypatch):
        # Seed a record that already has transcript.txt (no notes yet).
        note_dir = _seed_record_dir(tmp_path, "partial", with_transcript=True)
        (note_dir / "transcript.txt").write_text(
            "previous run transcript here", encoding="utf-8"
        )

        proc = _processor(tmp_path)

        whisper_called = {"count": 0}

        def boom(*args, **kwargs):
            whisper_called["count"] += 1
            raise AssertionError("whisper should not be called on resume")

        monkeypatch.setattr(proc.transcriber, "transcribe_file", boom)

        # Stub the other stages so the test stays focused on stage 2's resume.
        monkeypatch.setattr(
            BatchProcessor,
            "_stage_load_audio",
            lambda self, ctx: (
                setattr(ctx, "duration_seconds", 0.5) or ctx.view.done(Stage.LOAD_AUDIO)
            ),
        )
        monkeypatch.setattr(
            BatchProcessor,
            "_stage_generate_notes",
            lambda self, ctx: (
                (ctx.record.dir / "notes.md").write_text("# done", encoding="utf-8")
                or ctx.view.done(Stage.GENERATE_NOTES)
            ),
        )
        monkeypatch.setattr(
            BatchProcessor,
            "_stage_index",
            lambda self, ctx: ctx.view.done(Stage.INDEX, "auto-index off"),
        )

        result = proc.run_queue(console=Console(force_terminal=False))
        assert result == {"ok": 1, "skipped": 0, "failed": 0, "total": 1}
        assert whisper_called["count"] == 0
        # Transcript untouched
        assert (
            tmp_path / "partial" / "transcript.txt"
        ).read_text() == "previous run transcript here"
        # Notes now produced
        assert (tmp_path / "partial" / "notes.md").exists()
