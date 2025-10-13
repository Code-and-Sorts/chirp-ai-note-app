import queue
import threading
import time

import pytest

from recorder.live_types import AudioFrame
from recorder.vad_chunker import VADChunker


class DummyVAD:
    def __init__(self, decisions):
        self._decisions = decisions
        self._index = 0

    def is_speech(self, _data, _sample_rate):
        decision = self._decisions[min(self._index, len(self._decisions) - 1)]
        self._index += 1
        return decision


def test_vad_chunker_emits_chunk():
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    chunk_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # initial silence, speech, trailing silence
    decisions = [False] * 3 + [True] * 6 + [False] * 4

    def factory():
        return DummyVAD(decisions)

    chunker = VADChunker(
        frame_queue=frame_queue,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        sample_rate=16000,
        frame_ms=10,
        padding_ms=30,
        vad_factory=factory,
        energy_threshold=0.0,
    )
    chunker.start()

    timestamp = 0.0
    for decision in decisions:
        frame = AudioFrame(
            data=b"\x00" * 320,
            timestamp=timestamp,
            duration=0.01,
            level=0.1 if decision else 0.0,
        )
        frame_queue.put(frame)
        timestamp += 0.01
        time.sleep(0.005)

    time.sleep(0.2)
    stop_event.set()
    chunker.join(timeout=1)

    assert not chunk_queue.empty()
    chunk = chunk_queue.get()
    assert chunk.start == pytest.approx(0.03, abs=0.05)
    assert chunk.end > chunk.start
    assert len(chunk.data) > 0
