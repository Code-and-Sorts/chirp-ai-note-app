import sys
import threading
import time
import tomllib
import wave
from collections.abc import Callable, Iterator
from unittest.mock import Mock, patch

import numpy as np
import pytest

from audio_capture import SOURCE_MICROPHONE, SOURCE_SYSTEM
from recorder.audio_recorder import AudioRecorder


def _build_settings(notes_root):
    settings = Mock()
    directories = Mock()
    directories.notes_root = notes_root
    settings.directories = directories
    audio = Mock()
    audio.sample_rate = 16000
    audio.channels = 2
    audio.chunk_size = 1024
    audio.format = "wav"
    settings.audio = audio
    monitoring = Mock()
    monitoring.max_recording_hours = 8
    settings.monitoring = monitoring
    return settings


def _build_device_manager():
    device_manager = Mock()
    device_manager.list_devices.return_value = [
        {"index": 0, "name": "Built-in Microphone"}
    ]
    return device_manager


def _paired_float_frames(
    pair_count: int, samples_per_frame: int = 512, frame_us: int = 32_000
) -> list[tuple[int, int, np.ndarray]]:
    frames: list[tuple[int, int, np.ndarray]] = []
    for i in range(pair_count):
        ts = i * frame_us
        sys_chunk = np.full(samples_per_frame, 0.1, dtype=np.float32)
        mic_chunk = np.full(samples_per_frame, 0.2, dtype=np.float32)
        frames.append((SOURCE_SYSTEM, ts, sys_chunk))
        frames.append((SOURCE_MICROPHONE, ts, mic_chunk))
    return frames


class FakeAudioCapture:
    def __init__(
        self,
        frames: list[tuple[int, int, np.ndarray]],
        mic_device_name: str | None = "MockMic",
        per_frame_delay_s: float = 0.0,
        on_frame: Callable[[int], None] | None = None,
    ) -> None:
        self._frames = frames
        self.mic_device_name = mic_device_name
        self._per_frame_delay_s = per_frame_delay_s
        self._on_frame = on_frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def frames(self) -> Iterator[tuple[int, int, np.ndarray]]:
        for index, frame in enumerate(self._frames):
            if self._per_frame_delay_s:
                time.sleep(self._per_frame_delay_s)
            if self._on_frame is not None:
                self._on_frame(index)
            yield frame


@pytest.fixture
def mock_settings(tmp_path):
    return _build_settings(tmp_path)


@pytest.fixture
def mock_device_manager():
    return _build_device_manager()


class TestAudioRecorder:
    def test_initialization(self, mock_settings, mock_device_manager):
        recorder = AudioRecorder(mock_settings, mock_device_manager)

        assert recorder.settings == mock_settings
        assert recorder.device_manager == mock_device_manager
        assert recorder.is_recording is False
        assert recorder.title is None
        assert recorder.current_level == 0.0
        assert recorder.slug is None

    def test_write_initial_meta_writes_toml_fields(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        from datetime import datetime

        recorder = AudioRecorder(mock_settings, mock_device_manager)

        note_dir = tmp_path / "standup-2026-04-20"
        note_dir.mkdir()
        recorder._write_initial_meta(
            note_dir=note_dir,
            title="Standup",
            recorded_at=datetime(2026, 4, 20, 9, 0, 0),
            mic="Built-in Microphone",
            tags=["ops"],
        )

        with (note_dir / "meta.toml").open("rb") as fh:
            meta = tomllib.load(fh)

        assert meta["title"] == "Standup"
        assert meta["mic"] == "Built-in Microphone"
        assert meta["tags"] == ["ops"]
        assert meta["date"].startswith("2026-04-20")

    def test_update_meta_duration_merges_with_existing(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        import tomli_w

        recorder = AudioRecorder(mock_settings, mock_device_manager)

        note_dir = tmp_path / "standup-2026-04-20"
        note_dir.mkdir()
        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump({"title": "Existing", "tags": []}, fh)

        recorder._update_meta_duration(note_dir, 123.4)

        with (note_dir / "meta.toml").open("rb") as fh:
            meta = tomllib.load(fh)

        assert meta["title"] == "Existing"
        assert meta["duration_s"] == pytest.approx(123.4)

    def test_save_recording_no_frames_raises_error(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        recorder = AudioRecorder(mock_settings, mock_device_manager)
        recorder.frames = []

        with pytest.raises(RuntimeError, match="No audio data recorded"):
            recorder._save_recording(tmp_path / "audio.wav")

    def test_start_recording_cleans_up_note_dir_when_no_audio_captured(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        recorder = AudioRecorder(mock_settings, mock_device_manager)

        fake_cap = FakeAudioCapture(frames=[], mic_device_name="MockMic")

        def stop_recorder_after_start():
            time.sleep(0.05)
            recorder.is_recording = False

        stopper = threading.Thread(target=stop_recorder_after_start, daemon=True)

        with patch("recorder.audio_recorder.AudioCapture", return_value=fake_cap):
            stopper.start()
            with pytest.raises(RuntimeError, match="No audio data recorded"):
                recorder.start_recording(title="empty")
        stopper.join(timeout=1.0)

        created_dirs = list(tmp_path.iterdir())
        assert created_dirs == [], "empty note dir should have been cleaned up"

    def test_start_recording_writes_int16_wav_at_16khz(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        recorder = AudioRecorder(mock_settings, mock_device_manager)
        frames = _paired_float_frames(pair_count=5)
        fake_cap = FakeAudioCapture(frames=frames, mic_device_name="MockMic")

        def stop_when_frames_recorded():
            for _ in range(50):
                if len(recorder.frames) >= 5:
                    break
                time.sleep(0.02)
            recorder.is_recording = False

        stopper = threading.Thread(target=stop_when_frames_recorded, daemon=True)

        with patch("recorder.audio_recorder.AudioCapture", return_value=fake_cap):
            stopper.start()
            slug = recorder.start_recording(title="wav-shape")
        stopper.join(timeout=2.0)

        wav_path = tmp_path / slug / "audio.wav"
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wave_file:
            assert wave_file.getsampwidth() == 2
            assert wave_file.getframerate() == 16000
            assert wave_file.getnchannels() == 1
            assert wave_file.getnframes() > 0

    def test_start_recording_uses_mic_device_name_from_audio_capture(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        recorder = AudioRecorder(mock_settings, mock_device_manager)
        frames = _paired_float_frames(pair_count=2)
        fake_cap = FakeAudioCapture(frames=frames, mic_device_name="Studio Mic Pro")

        def stop_when_frames_recorded():
            for _ in range(50):
                if len(recorder.frames) >= 2:
                    break
                time.sleep(0.02)
            recorder.is_recording = False

        stopper = threading.Thread(target=stop_when_frames_recorded, daemon=True)

        with patch("recorder.audio_recorder.AudioCapture", return_value=fake_cap):
            stopper.start()
            slug = recorder.start_recording(title="mic-name")
        stopper.join(timeout=2.0)

        with (tmp_path / slug / "meta.toml").open("rb") as fh:
            meta = tomllib.load(fh)
        assert meta["mic"] == "Studio Mic Pro"

    def test_start_recording_falls_back_to_default_when_mic_device_name_none(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        recorder = AudioRecorder(mock_settings, mock_device_manager)
        frames = _paired_float_frames(pair_count=2)
        fake_cap = FakeAudioCapture(frames=frames, mic_device_name=None)

        def stop_when_frames_recorded():
            for _ in range(50):
                if len(recorder.frames) >= 2:
                    break
                time.sleep(0.02)
            recorder.is_recording = False

        stopper = threading.Thread(target=stop_when_frames_recorded, daemon=True)

        with patch("recorder.audio_recorder.AudioCapture", return_value=fake_cap):
            stopper.start()
            slug = recorder.start_recording(title="mic-fallback")
        stopper.join(timeout=2.0)

        with (tmp_path / slug / "meta.toml").open("rb") as fh:
            meta = tomllib.load(fh)
        assert meta["mic"] == "default"

    def test_start_recording_raises_on_non_macos(
        self, mock_settings, mock_device_manager
    ):
        recorder = AudioRecorder(mock_settings, mock_device_manager)
        with patch.object(sys, "platform", "linux"):
            with pytest.raises(
                RuntimeError, match="chirp record requires macOS 13 or later"
            ):
                recorder.start_recording(title="non-mac")

    def test_start_recording_cleans_up_when_audio_capture_fails_to_start(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        recorder = AudioRecorder(mock_settings, mock_device_manager)

        class FailingCapture:
            def __enter__(self):
                raise RuntimeError("helper-startup-boom")

            def __exit__(self, exc_type, exc, tb):
                return None

        with patch(
            "recorder.audio_recorder.AudioCapture", return_value=FailingCapture()
        ):
            with pytest.raises(RuntimeError, match="helper-startup-boom"):
                recorder.start_recording(title="startup-fail")

        assert recorder.is_recording is False
        assert list(tmp_path.iterdir()) == []

    def test_start_recording_unblocks_when_helper_eofs_cleanly(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        # Helper closes stdout and exits 0; cap.frames() exhausts cleanly
        # without raising. The worker must flip is_recording=False so
        # start_recording's main wait loop exits instead of polling
        # forever.
        recorder = AudioRecorder(mock_settings, mock_device_manager)
        frames = _paired_float_frames(pair_count=3)
        fake_cap = FakeAudioCapture(frames=frames, mic_device_name="MockMic")

        with patch("recorder.audio_recorder.AudioCapture", return_value=fake_cap):
            slug = recorder.start_recording(title="clean-eof")

        assert slug is not None
        assert recorder.is_recording is False
        assert (tmp_path / slug / "audio.wav").exists()

    def test_start_recording_surfaces_worker_crash_after_partial_capture(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        # Feeds 4 paired sys+mic chunks (drives the mixer to produce
        # output frames into self.frames) and *then* raises mid-iteration.
        # Without crash propagation the recorder would silently truncate
        # and return success — this test pins the new behavior of raising
        # and discarding the partial recording.
        recorder = AudioRecorder(mock_settings, mock_device_manager)

        class CrashAfterPartial:
            def __init__(self):
                self.mic_device_name = "MockMic"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def frames(self):
                for i in range(4):
                    ts = i * 32_000
                    yield (
                        SOURCE_SYSTEM,
                        ts,
                        np.full(512, 0.1, dtype=np.float32),
                    )
                    yield (
                        SOURCE_MICROPHONE,
                        ts,
                        np.full(512, 0.2, dtype=np.float32),
                    )
                # Give the worker a moment to drain the mixer before crashing
                # so self.frames is non-empty when the exception fires.
                time.sleep(0.05)
                raise RuntimeError("worker-boom")

        with patch(
            "recorder.audio_recorder.AudioCapture",
            return_value=CrashAfterPartial(),
        ):
            with pytest.raises(
                RuntimeError, match="audio capture worker crashed mid-recording"
            ) as excinfo:
                recorder.start_recording(title="worker-crash")

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "worker-boom" in str(excinfo.value.__cause__)
        assert recorder.is_recording is False
        assert list(tmp_path.iterdir()) == [], (
            "partial recording must be discarded, not silently truncated"
        )


class TestAudioRecorderPause:
    def test_pause_drops_frames_in_capture_worker(self, tmp_path):
        settings = _build_settings(tmp_path)
        recorder = AudioRecorder(settings, _build_device_manager())

        pair_count = 8
        frames = _paired_float_frames(pair_count=pair_count)

        pause_index = 4

        def on_frame(index: int) -> None:
            if index == pause_index:
                recorder.pause()
            elif index == pause_index + 4:
                recorder.resume()

        fake_cap = FakeAudioCapture(
            frames=frames,
            mic_device_name="MockMic",
            per_frame_delay_s=0.01,
            on_frame=on_frame,
        )

        def stop_after_capture_completes():
            for _ in range(100):
                time.sleep(0.02)
                if not (
                    recorder._capture_thread and recorder._capture_thread.is_alive()
                ):
                    break
            recorder.is_recording = False

        stopper = threading.Thread(target=stop_after_capture_completes, daemon=True)

        with patch("recorder.audio_recorder.AudioCapture", return_value=fake_cap):
            stopper.start()
            slug = recorder.start_recording(title="pause-test")
        stopper.join(timeout=3.0)

        assert slug is not None
        assert 0 < len(recorder.frames) < pair_count
