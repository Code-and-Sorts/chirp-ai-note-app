import queue
import threading
import time

import pytest

from recorder.live_types import AudioFrame
from recorder.vad_chunker import VADChunker

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_BYTES = int(SAMPLE_RATE * (FRAME_MS / 1000.0)) * 2


class DummyVAD:
    def __init__(self, decisions):
        self._decisions = decisions
        self._index = 0

    def __call__(self, _data, _sample_rate):
        decision = self._decisions[min(self._index, len(self._decisions) - 1)]
        self._index += 1
        return 0.9 if decision else 0.1


def _make_frame(timestamp: float, level: float = 0.1) -> AudioFrame:
    return AudioFrame(
        data=b"\x00" * FRAME_BYTES,
        timestamp=timestamp,
        duration=FRAME_MS / 1000.0,
        level=level,
    )


def _feed_frames(frame_queue, frames, delay=0.002):
    for frame in frames:
        frame_queue.put(frame)
        time.sleep(delay)


def test_vad_chunker_emits_chunk():
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    chunk_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    decisions = [False] * 3 + [True] * 6 + [False] * 4

    chunker = VADChunker(
        frame_queue=frame_queue,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        padding_ms=30,
        vad_factory=lambda: DummyVAD(decisions),
        energy_threshold=0.0,
        poll_timeout=0.02,
    )
    chunker.start()

    timestamp = 0.0
    for decision in decisions:
        frame = _make_frame(timestamp, level=0.1 if decision else 0.0)
        frame_queue.put(frame)
        timestamp += FRAME_MS / 1000.0
        time.sleep(0.005)

    time.sleep(0.2)
    stop_event.set()
    chunker.join(timeout=1)

    assert not chunk_queue.empty()
    chunk = chunk_queue.get()
    assert chunk.start == pytest.approx(0.03, abs=0.05)
    assert chunk.end > chunk.start
    assert len(chunk.data) > 0


def test_max_chunk_seconds_enforcement():
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    chunk_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    num_frames = 30
    decisions = [True] * num_frames

    chunker = VADChunker(
        frame_queue=frame_queue,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        padding_ms=30,
        vad_factory=lambda: DummyVAD(decisions),
        energy_threshold=0.0,
        max_chunk_seconds=0.1,
        poll_timeout=0.02,
    )
    chunker.start()

    frames = [_make_frame(i * FRAME_MS / 1000.0) for i in range(num_frames)]
    _feed_frames(frame_queue, frames)

    time.sleep(0.3)
    stop_event.set()
    chunker.join(timeout=1)

    assert not chunk_queue.empty()
    chunk = chunk_queue.get()
    chunk_duration = chunk.end - chunk.start
    assert chunk_duration <= 0.15


def test_energy_threshold_filtering():
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    chunk_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    decisions = [False] * 20

    chunker = VADChunker(
        frame_queue=frame_queue,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        padding_ms=30,
        vad_factory=lambda: DummyVAD(decisions),
        energy_threshold=0.5,
        poll_timeout=0.02,
    )
    chunker.start()

    frames = [_make_frame(i * FRAME_MS / 1000.0, level=0.4) for i in range(20)]
    _feed_frames(frame_queue, frames)

    time.sleep(0.2)
    stop_event.set()
    chunker.join(timeout=1)

    assert chunk_queue.empty()


def test_padding_buffer_tail_appended():
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    chunk_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    decisions = [False] * 3 + [True] * 6 + [False] * 4

    chunker = VADChunker(
        frame_queue=frame_queue,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        padding_ms=30,
        vad_factory=lambda: DummyVAD(decisions),
        energy_threshold=0.0,
        poll_timeout=0.02,
    )
    chunker.start()

    timestamp = 0.0
    for decision in decisions:
        frame = _make_frame(timestamp, level=0.1 if decision else 0.0)
        frame_queue.put(frame)
        timestamp += FRAME_MS / 1000.0
        time.sleep(0.005)

    time.sleep(0.2)
    stop_event.set()
    chunker.join(timeout=1)

    assert not chunk_queue.empty()
    chunk = chunk_queue.get()
    speech_only_bytes = 6 * FRAME_BYTES
    assert len(chunk.data) > speech_only_bytes


def test_stop_event_flushes_active_speech():
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    chunk_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    decisions = [True] * 10

    chunker = VADChunker(
        frame_queue=frame_queue,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        padding_ms=30,
        vad_factory=lambda: DummyVAD(decisions),
        energy_threshold=0.0,
        max_chunk_seconds=999,
        poll_timeout=0.02,
    )
    chunker.start()

    frames = [_make_frame(i * FRAME_MS / 1000.0) for i in range(10)]
    _feed_frames(frame_queue, frames)

    time.sleep(0.2)
    stop_event.set()
    chunker.join(timeout=1)

    assert not chunk_queue.empty()
    chunk = chunk_queue.get()
    assert len(chunk.data) > 0


def test_full_chunk_queue_increments_drop_counter_and_emits_event():
    # AC-4: a full chunk_queue must count the dropped speech chunk and surface
    # a `dropped` dashboard event rather than discarding it silently.
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    chunk_queue: queue.Queue = queue.Queue(maxsize=1)
    event_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    chunker = VADChunker(
        frame_queue=frame_queue,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        vad_factory=lambda: DummyVAD([True]),
        event_queue=event_queue,
        poll_timeout=0.02,
    )

    chunker._voiced_frames = [_make_frame(0.0), _make_frame(0.01)]
    chunker._emit_chunk()
    assert chunker.dropped_chunks == 0

    chunker._voiced_frames = [_make_frame(0.02), _make_frame(0.03)]
    chunker._emit_chunk()

    assert chunker.dropped_chunks == 1
    dropped_events = [
        event for event in list(event_queue.queue) if event.type == "dropped"
    ]
    assert dropped_events
    assert dropped_events[-1].payload["dropped_chunks"] == 1
