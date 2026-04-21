from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from transcriber.batch_processor import BatchProcessor
from utils.file_utils import NoteRecord


def _make_record(tmp_path: Path) -> NoteRecord:
    note_dir = tmp_path / "team-sync-2026-04-20"
    note_dir.mkdir(parents=True, exist_ok=True)
    audio_path = note_dir / "audio.wav"
    audio_path.write_bytes(b"audio data")
    return NoteRecord(
        slug=note_dir.name,
        dir=note_dir,
        audio=audio_path,
        transcript=None,
        notes=None,
        meta=None,
        created_at=datetime(2026, 4, 20, 9, 0, 0),
        tags=[],
        title="Team Sync",
    )


def _settings():
    settings = Mock()
    settings.directories = Mock()
    settings.directories.notes_root = Path("/tmp/fake")
    return settings


def test_process_record_writes_transcript(tmp_path):
    record = _make_record(tmp_path)
    transcription_result = {
        "success": True,
        "full_text": "hello world",
        "metadata": {"recording_id": "abc"},
    }

    mock_transcriber = Mock()
    mock_transcriber.transcribe_file.return_value = transcription_result

    with (
        patch(
            "transcriber.batch_processor.WhisperTranscriber",
            return_value=mock_transcriber,
        ),
        patch("transcriber.batch_processor.PopupManager"),
    ):
        processor = BatchProcessor(_settings())

    result = processor._process_record(record, on_segment=None)

    transcript_path = record.dir / "transcript.txt"
    assert transcript_path.exists()
    assert transcript_path.read_text(encoding="utf-8") == "hello world"
    assert result["transcript_path"] == str(transcript_path)
    assert result["slug"] == record.slug


def test_filter_records_skips_already_transcribed(tmp_path):
    record_without_transcript = _make_record(tmp_path)

    second_dir = tmp_path / "other-note"
    second_dir.mkdir()
    (second_dir / "audio.wav").write_bytes(b"")
    (second_dir / "transcript.txt").write_text("done", encoding="utf-8")
    record_with_transcript = NoteRecord(
        slug=second_dir.name,
        dir=second_dir,
        audio=second_dir / "audio.wav",
        transcript=second_dir / "transcript.txt",
        notes=None,
        meta=None,
        created_at=datetime.now(),
    )

    with (
        patch("transcriber.batch_processor.WhisperTranscriber"),
        patch("transcriber.batch_processor.PopupManager"),
    ):
        processor = BatchProcessor(_settings())

    pending = processor._filter_records(
        [record_without_transcript, record_with_transcript], force=False
    )
    assert pending == [record_without_transcript]

    forced = processor._filter_records(
        [record_without_transcript, record_with_transcript], force=True
    )
    assert forced == [record_without_transcript, record_with_transcript]


def test_process_records_returns_empty_when_no_candidates(tmp_path):
    with (
        patch("transcriber.batch_processor.WhisperTranscriber"),
        patch("transcriber.batch_processor.PopupManager"),
    ):
        processor = BatchProcessor(_settings())

    assert processor.process_records([], force=False) == []
