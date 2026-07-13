import queue
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from recorder.live_transcriber import LiveTranscriber
from recorder.live_types import DashboardEvent, SpeechChunk, TranscriptSegment


def _make_chunk(start: float, end: float, sample_rate: int = 16000) -> SpeechChunk:
    num_samples = int((end - start) * sample_rate)
    data = (np.ones(num_samples, dtype=np.int16) * 1000).tobytes()
    return SpeechChunk(data=data, start=start, end=end)


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


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
        assert _wait_until(lambda: not responses and chunk_queue.empty())
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


def test_overlap_detection_filters_segments():
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    def fake_transcribe(path, fast_mode=False, language=None):
        return {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello world"},
                {"start": 0.2, "end": 1.0, "text": "hello world again"},
            ],
            "metadata": {"language": "en"},
        }

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_cls.return_value.transcribe_file.side_effect = fake_transcribe

        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            sample_rate=16000,
            transcription_interval=0.0,
            overlap_threshold=0.3,
            poll_timeout=0.02,
        )

        transcriber.start()
        chunk_queue.put(_make_chunk(0.0, 2.0))
        assert _wait_until(lambda: transcriber.segments)
        stop_event.set()
        transcriber.join(timeout=1)

    texts = [s.text for s in transcriber.segments]
    assert "hello world" in texts
    assert "hello world again" not in texts


def test_buffer_pruning_after_segments():
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    def fake_transcribe(path, fast_mode=False, language=None):
        return {
            "segments": [{"start": 0.0, "end": 0.5, "text": "pruned"}],
            "metadata": {},
        }

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_cls.return_value.transcribe_file.side_effect = fake_transcribe

        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            sample_rate=16000,
            transcription_interval=0.0,
            poll_timeout=0.02,
        )

        chunk = _make_chunk(0.0, 1.0)
        original_buffer_size = len(chunk.data)

        transcriber.start()
        chunk_queue.put(chunk)
        assert _wait_until(lambda: transcriber._buffer_offset_seconds > 0.0)
        stop_event.set()
        transcriber.join(timeout=1)

    assert len(transcriber._pcm_buffer) < original_buffer_size
    assert transcriber._buffer_offset_seconds > 0.0


def test_empty_chunk_handling():
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_cls.return_value.transcribe_file.return_value = {
            "segments": [],
            "metadata": {},
        }

        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            sample_rate=16000,
            transcription_interval=0.0,
            poll_timeout=0.02,
        )

        transcriber.start()
        chunk_queue.put(SpeechChunk(data=b"", start=0.0, end=0.0))
        assert _wait_until(chunk_queue.empty)
        stop_event.set()
        transcriber.join(timeout=1)

    transcript_events = [e for e in list(event_queue.queue) if e.type == "transcript"]
    assert len(transcript_events) == 0


def test_force_transcription_on_stop():
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    def fake_transcribe(path, fast_mode=False, language=None):
        return {
            "segments": [{"start": 0.0, "end": 0.5, "text": "forced"}],
            "metadata": {},
        }

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_cls.return_value.transcribe_file.side_effect = fake_transcribe

        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            sample_rate=16000,
            transcription_interval=999.0,
            poll_timeout=0.02,
        )

        transcriber.start()
        chunk_queue.put(_make_chunk(0.0, 1.0))
        assert _wait_until(chunk_queue.empty)
        stop_event.set()
        transcriber.join(timeout=2)

    assert any(s.text == "forced" for s in transcriber.segments)


def test_export_transcript_is_race_free_under_concurrent_mutation(tmp_path: Path):
    # AC-3: export_transcript reads through the lock-guarded `segments`
    # property, which must return a copy taken under the lock — never the live
    # list — so a concurrently mutating worker cannot tear an export.
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_cls.return_value.transcribe_file.return_value = {
            "segments": [],
            "metadata": {},
        }
        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            sample_rate=16000,
        )

    errors: list[Exception] = []
    mutate_stop = threading.Event()

    def mutate() -> None:
        # Append AND periodically truncate so exports see both growth and
        # shrinkage, while the list stays bounded — an unthrottled append-only
        # loop makes each export O(total appends) and the test quadratic.
        index = 0
        while not mutate_stop.is_set():
            with transcriber._lock:  # type: ignore[attr-defined]
                segments = transcriber._segments  # type: ignore[attr-defined]
                if len(segments) >= 512:
                    del segments[: len(segments) // 2]
                segments.append(
                    TranscriptSegment(
                        text=f"seg-{index}", start=index, end=index + 1, words=1
                    )
                )
            index += 1
            # Yield so the exporter can win the lock; without this the spin
            # loop starves it and the test crawls under GIL contention.
            time.sleep(0)

    def export() -> None:
        try:
            for _ in range(200):
                snapshot = transcriber.segments
                frozen = list(snapshot)
                transcriber.export_transcript(tmp_path / "transcript.txt")
                # The property must hand back a copy taken under the lock; a
                # live reference mutates between these two reads.
                assert snapshot is not transcriber._segments  # type: ignore[attr-defined]
                assert snapshot == frozen, "snapshot mutated by worker"
        except Exception as exc:  # noqa: BLE001 - capturing any race failure
            errors.append(exc)

    mutator = threading.Thread(target=mutate, daemon=True)
    exporter = threading.Thread(target=export, daemon=True)
    mutator.start()
    exporter.start()
    exporter.join(timeout=30)
    mutate_stop.set()
    mutator.join(timeout=5)

    # An abandoned worker outlives the test and steals GIL time from the rest
    # of the suite, so a leak must fail here rather than slow everything else.
    assert not exporter.is_alive(), "exporter thread did not finish"
    assert not mutator.is_alive(), "mutator thread did not finish"
    assert not errors, f"export raced with mutation: {errors!r}"


def test_close_tears_down_underlying_whisper_transcriber():
    # AC-5: LiveTranscriber.close() releases the wrapped WhisperTranscriber.
    settings = Mock()
    chunk_queue: queue.Queue[SpeechChunk] = queue.Queue()
    event_queue: queue.Queue[DashboardEvent] = queue.Queue()
    stop_event = threading.Event()

    with patch("recorder.live_transcriber.WhisperTranscriber") as mock_cls:
        mock_instance = mock_cls.return_value
        transcriber = LiveTranscriber(
            settings=settings,
            chunk_queue=chunk_queue,
            event_queue=event_queue,
            stop_event=stop_event,
            sample_rate=16000,
        )

        transcriber.close()
        transcriber.close()

    assert mock_instance.close.call_count == 2
