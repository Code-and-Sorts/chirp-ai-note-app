"""Tests for `recorder.live_audio.LiveAudioStream`.

Mocks `AudioCapture` so the mixer thread runs against deterministic fake
frames; no Swift helper or real audio hardware is required.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from recorder.audio_mixer import SOURCE_MICROPHONE, SOURCE_SYSTEM
from recorder.live_audio import LiveAudioStream
from recorder.live_types import AudioFrame


@pytest.fixture(autouse=True)
def _force_darwin_platform(request: pytest.FixtureRequest) -> Iterator[None]:
    if "real_platform" in request.keywords:
        yield
        return
    with mock.patch.object(sys, "platform", "darwin"):
        yield


def _make_stream(
    debug_dir: Path | None = None,
) -> tuple[
    LiveAudioStream,
    queue.Queue[AudioFrame],
    queue.Queue[float],
    threading.Event,
]:
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    level_queue: queue.Queue[float] = queue.Queue()
    stop_event = threading.Event()
    settings = mock.MagicMock()
    settings.audio.sample_rate = 16000
    device_manager = mock.MagicMock()
    stream = LiveAudioStream(
        settings=settings,
        device_manager=device_manager,
        frame_queue=frame_queue,
        stop_event=stop_event,
        level_queue=level_queue,
        debug_dir=debug_dir,
    )
    return stream, frame_queue, level_queue, stop_event


class _FakeAudioCapture:
    """Test double whose context-manager protocol mirrors `AudioCapture`."""

    def __init__(
        self,
        frames: list[tuple[int, int, np.ndarray]],
        block_after_drain: bool = True,
    ) -> None:
        self._frames = frames
        self._block_after_drain = block_after_drain
        self._exit_event = threading.Event()
        self.mic_device_name = "MockMic"

    def __enter__(self) -> _FakeAudioCapture:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._exit_event.set()

    def frames(self) -> Iterator[tuple[int, int, np.ndarray]]:
        for frame in self._frames:
            if self._exit_event.is_set():
                return
            yield frame
        if self._block_after_drain:
            self._exit_event.wait(timeout=2.0)


def _paired_frames(
    count: int, samples_per_chunk: int = 512
) -> list[tuple[int, int, np.ndarray]]:
    sample_rate = 16000
    chunk_us = (samples_per_chunk * 1_000_000) // sample_rate
    sequence: list[tuple[int, int, np.ndarray]] = []
    for index in range(count):
        timestamp_us = index * chunk_us
        sys_samples = np.full(samples_per_chunk, 0.2, dtype=np.float32)
        mic_samples = np.full(samples_per_chunk, 0.1, dtype=np.float32)
        sequence.append((SOURCE_SYSTEM, timestamp_us, sys_samples))
        sequence.append((SOURCE_MICROPHONE, timestamp_us, mic_samples))
    return sequence


def _wait_for_queue(target: queue.Queue, expected: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while target.qsize() < expected and time.monotonic() < deadline:
        time.sleep(0.01)


def test_start_pushes_audio_frames_to_queue() -> None:
    stream, frame_queue, level_queue, stop_event = _make_stream()
    fake = _FakeAudioCapture(_paired_frames(5))

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        _wait_for_queue(frame_queue, expected=4, timeout=2.0)
        stop_event.set()
        stream.stop()

    collected: list[AudioFrame] = []
    while not frame_queue.empty():
        collected.append(frame_queue.get_nowait())

    assert len(collected) >= 4
    for frame in collected:
        assert isinstance(frame.data, bytes)
        assert len(frame.data) == 512 * 2
        assert frame.duration == pytest.approx(0.032)
        assert frame.level >= 0.0
        assert frame.level <= 1.0

    assert not level_queue.empty()


def test_start_silence_pads_when_one_source_stalls() -> None:
    stream, frame_queue, _level_queue, stop_event = _make_stream()
    sample_rate = 16000
    chunk_samples = 512
    chunk_us = (chunk_samples * 1_000_000) // sample_rate
    mic_only = [
        (
            SOURCE_MICROPHONE,
            index * chunk_us,
            np.full(chunk_samples, 0.15, dtype=np.float32),
        )
        for index in range(8)
    ]
    fake = _FakeAudioCapture(mic_only)

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        _wait_for_queue(frame_queue, expected=4, timeout=2.0)
        stop_event.set()
        stream.stop()

    assert frame_queue.qsize() >= 4
    first = frame_queue.get_nowait()
    assert isinstance(first.data, bytes)
    assert len(first.data) == 512 * 2


def test_save_recording_writes_16khz_mono_int16_wav(tmp_path: Path) -> None:
    stream, frame_queue, _level_queue, stop_event = _make_stream()
    fake = _FakeAudioCapture(_paired_frames(6))

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        _wait_for_queue(frame_queue, expected=4, timeout=2.0)
        stop_event.set()
        stream.stop()

    out_path = tmp_path / "out.wav"
    stream.save_recording(out_path)

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0


def test_start_raises_on_non_macos() -> None:
    stream, _frame_queue, _level_queue, _stop_event = _make_stream()
    with mock.patch.object(sys, "platform", "linux"):
        with pytest.raises(RuntimeError, match="macOS"):
            stream.start()


def test_close_is_idempotent() -> None:
    stream, _frame_queue, _level_queue, _stop_event = _make_stream()
    fake = _FakeAudioCapture(_paired_frames(2))

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        time.sleep(0.05)
        stream.close()
        stream.close()


def test_capture_error_is_set_when_mixer_thread_crashes() -> None:
    stream, _frame_queue, _level_queue, stop_event = _make_stream()

    class CrashingCapture:
        mic_device_name = "MockMic"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def frames(self):
            for index in range(2):
                yield (
                    SOURCE_SYSTEM,
                    index * 32_000,
                    np.full(512, 0.1, dtype=np.float32),
                )
            raise RuntimeError("mixer-boom")

    with mock.patch("recorder.live_audio.AudioCapture", return_value=CrashingCapture()):
        stream.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and stream.capture_error is None:
            time.sleep(0.01)
        stop_event.set()
        stream.stop()

    assert isinstance(stream.capture_error, RuntimeError)
    assert "mixer-boom" in str(stream.capture_error)
    assert stop_event.is_set()


def test_audio_frame_timestamps_are_synthesized_from_frame_index() -> None:
    # Mixer-thread scheduling latency must NOT leak into AudioFrame.timestamp:
    # downstream VAD / chunking expects timestamps that advance by exactly
    # frame_duration (32 ms) regardless of when the mixer happened to publish.
    stream, frame_queue, _level_queue, stop_event = _make_stream()
    fake = _FakeAudioCapture(_paired_frames(6))

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        _wait_for_queue(frame_queue, expected=5, timeout=2.0)
        stop_event.set()
        stream.stop()

    timestamps: list[float] = []
    while not frame_queue.empty():
        timestamps.append(frame_queue.get_nowait().timestamp)

    assert len(timestamps) >= 5
    for index, ts in enumerate(timestamps):
        assert ts == pytest.approx(index * 0.032, abs=1e-9)
