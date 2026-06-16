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
    stream = LiveAudioStream(
        settings=settings,
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


def test_start_cleans_up_audio_capture_when_thread_start_fails() -> None:
    # If anything between AudioCapture entry and mixer-thread start fails
    # (e.g., thread exhaustion), `start()` must call __exit__ on the
    # context so the helper subprocess doesn't leak.
    stream, _frame_queue, _level_queue, _stop_event = _make_stream()
    fake = _ExitTrackingFakeCapture(_paired_frames(2))

    with (
        mock.patch("recorder.live_audio.AudioCapture", return_value=fake),
        mock.patch(
            "recorder.live_audio.threading.Thread",
            side_effect=RuntimeError("can't start new thread"),
        ),
        pytest.raises(RuntimeError, match="can't start new thread"),
    ):
        stream.start()

    assert fake.exit_calls == 1
    assert stream._cap_ctx is None
    assert stream._cap is None
    assert stream._mixer_thread is None


class _ExitTrackingFakeCapture(_FakeAudioCapture):
    def __init__(
        self,
        frames: list[tuple[int, int, np.ndarray]],
    ) -> None:
        super().__init__(frames, block_after_drain=False)
        self.exit_calls = 0

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exit_calls += 1
        return super().__exit__(exc_type, exc, tb)


def test_close_is_idempotent() -> None:
    stream, _frame_queue, _level_queue, _stop_event = _make_stream()
    fake = _FakeAudioCapture(_paired_frames(2))

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        assert stream._mixer_thread is not None, (
            "mixer thread must exist before testing idempotent close"
        )
        assert stream._mixer_thread.is_alive(), (
            "mixer thread must be alive before testing idempotent close"
        )
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


def test_mixer_thread_sets_stop_event_on_clean_helper_eof() -> None:
    # H6: when `cap.frames()` exhausts cleanly (helper closes stdout,
    # exits 0) while stop_event is not yet set, the mixer thread records
    # a sentinel capture_error and sets stop_event so the session doesn't
    # wait forever.
    stream, _frame_queue, _level_queue, stop_event = _make_stream()
    fake = _FakeAudioCapture(_paired_frames(2), block_after_drain=False)

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.01)
        stream.stop()

    assert stop_event.is_set()
    assert isinstance(stream.capture_error, RuntimeError)
    assert "clean EOF" in str(stream.capture_error)


def test_mic_device_name_is_exposed_from_audio_capture() -> None:
    stream, _frame_queue, _level_queue, stop_event = _make_stream()
    fake = _FakeAudioCapture(_paired_frames(2), block_after_drain=False)
    fake.mic_device_name = "Studio Mic Pro"

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.01)
        stream.stop()

    assert stream.mic_device_name == "Studio Mic Pro"


def test_frame_samples_derived_from_frame_ms() -> None:
    # Non-default frame_ms must drive frame size and AudioFrame duration
    # consistently — earlier hard-coded 512 samples was a latent bug.
    frame_queue: queue.Queue[AudioFrame] = queue.Queue()
    level_queue: queue.Queue[float] = queue.Queue()
    stop_event = threading.Event()
    settings = mock.MagicMock()
    settings.audio.sample_rate = 16000
    stream = LiveAudioStream(
        settings=settings,
        frame_queue=frame_queue,
        stop_event=stop_event,
        level_queue=level_queue,
        frame_ms=64,  # non-default
    )
    sample_rate = 16000
    chunk_us = (1024 * 1_000_000) // sample_rate  # 64 ms per chunk
    sequence: list[tuple[int, int, np.ndarray]] = []
    for index in range(4):
        ts = index * chunk_us
        sequence.append((SOURCE_SYSTEM, ts, np.full(1024, 0.2, dtype=np.float32)))
        sequence.append((SOURCE_MICROPHONE, ts, np.full(1024, 0.1, dtype=np.float32)))
    fake = _FakeAudioCapture(sequence, block_after_drain=False)

    with mock.patch("recorder.live_audio.AudioCapture", return_value=fake):
        stream.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.01)
        stream.stop()

    collected: list[AudioFrame] = []
    while not frame_queue.empty():
        collected.append(frame_queue.get_nowait())

    assert len(collected) >= 3
    for frame in collected:
        # 64 ms × 16 kHz × 2 bytes per int16 = 2048 bytes
        assert len(frame.data) == 1024 * 2
        assert frame.duration == pytest.approx(0.064)


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


def test_full_frame_queue_increments_drop_counter() -> None:
    # AC-4: a full frame_queue is counted, not silently dropped. The WAV write
    # happens inside _publish_mixed_frame before the enqueue, so the saved
    # recording is independent of these drops.
    frame_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=1)
    level_queue: queue.Queue[float] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    settings = mock.MagicMock()
    settings.audio.sample_rate = 16000
    stream = LiveAudioStream(
        settings=settings,
        frame_queue=frame_queue,
        stop_event=stop_event,
        level_queue=level_queue,
    )

    mixed = np.full(512, 0.1, dtype=np.float32)
    stream._publish_mixed_frame(mixed)
    assert stream.dropped_frames == 0

    stream._publish_mixed_frame(mixed)
    stream._publish_mixed_frame(mixed)

    assert stream.dropped_frames == 2
    assert frame_queue.qsize() == 1
