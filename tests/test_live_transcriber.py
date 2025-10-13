import queue
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from recorder.live_transcriber import LiveTranscriber
from recorder.live_types import DashboardEvent, SpeechChunk, TranscriptSegment


def test_live_transcriber_emits_events():
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    responses = [
        {
            "segments": [{"start": 0.0, "end": 0.5, "text": " hello world "}],
            "metadata": {"language": "en"},
        },
        {
            "segments": [
                {"start": 0.0, "end": 0.5, "text": " hello world "},
                {"start": 0.5, "end": 1.0, "text": " another line "},
            ],
            "metadata": {"language": "en"},
        },
    ]

    def fake_transcribe(path, fast_mode=False, language=None):
        return responses.pop(0) if responses else {"segments": [], "metadata": {}}

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.transcribe_file.side_effect = fake_transcribe

        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            recording_start=0.5,
            sample_rate=16000,
            transcription_interval=0.0,
        )

        transcriber.start()
        chunk_queue.put(
            SpeechChunk(
                data=(np.ones(32000, dtype=np.int16).tobytes()),
                start=0.0,
                end=1.0,
            )
        )
        chunk_queue.put(
            SpeechChunk(
                data=(np.ones(32000, dtype=np.int16).tobytes()),
                start=1.0,
                end=2.0,
            )
        )
        time.sleep(0.1)
        stop_event.set()
        transcriber.join(timeout=1)

    transcript_events = [
        event for event in list(event_queue.queue) if event.type == "transcript"
    ]
    assert transcript_events, "expected transcript events"
    combined_text = " ".join(
        segment.text
        for event in transcript_events
        for segment in event.payload["segments"]
    )
    assert "hello world" in combined_text
    assert "another line" in combined_text


def test_live_transcriber_export(tmp_path: Path):
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.transcribe_file.return_value = {"segments": [], "metadata": {}}

        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            recording_start=0.0,
            sample_rate=16000,
            debug_dir=tmp_path,
            transcription_interval=0.0,
        )

        transcriber._segments.extend(  # type: ignore[attr-defined]
            [
                TranscriptSegment(text="foo", start=0.0, end=1.0, words=1),
                TranscriptSegment(text="bar", start=1.0, end=2.0, words=1),
            ]
        )

        output_path = tmp_path / "transcript.txt"
        transcriber.export_transcript(output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "foo" in content
    assert "bar" in content


def test_write_debug_chunk_writes_files(tmp_path: Path):
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.transcribe_file.return_value = {"segments": [], "metadata": {}}

        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            recording_start=0.0,
            sample_rate=16000,
            debug_dir=tmp_path,
        )

        pcm = (np.ones(16000, dtype=np.int16) * 1000).astype(np.int16).tobytes()
        transcriber._write_debug_chunk(pcm, 16000, [])

    chunk_files = sorted(tmp_path.glob("chunk_0000*.wav"))
    assert chunk_files, "expected chunk wav"
    text_files = sorted(tmp_path.glob("chunk_0000*.txt"))
    summary_files = sorted(tmp_path.glob("chunk_0000*_summary.txt"))
    assert text_files, "expected chunk txt"
    assert summary_files, "expected summary"
