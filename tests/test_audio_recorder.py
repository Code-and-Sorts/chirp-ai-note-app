import array
import tomllib
from unittest.mock import Mock, patch

import pyaudio
import pytest

from recorder.audio_recorder import AudioRecorder


class TestAudioRecorder:
    @pytest.fixture
    def mock_settings(self, tmp_path):
        settings = Mock()
        directories = Mock()
        directories.notes_root = tmp_path
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

    @pytest.fixture
    def mock_device_manager(self):
        device_manager = Mock()
        device_manager.get_recommended_device.return_value = 0
        device_manager.get_device_info.return_value = {
            "maxInputChannels": 2,
            "defaultSampleRate": 16000,
        }
        device_manager.list_devices.return_value = [
            {"index": 0, "name": "Built-in Microphone"}
        ]
        return device_manager

    def test_initialization(self, mock_settings, mock_device_manager):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)

            assert recorder.settings == mock_settings
            assert recorder.device_manager == mock_device_manager
            assert recorder.is_recording is False
            assert recorder.title is None
            assert recorder.current_level == 0.0
            assert recorder.slug is None

    def test_audio_callback_computes_level_for_silence(
        self, mock_settings, mock_device_manager
    ):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)
            recorder.is_recording = True

            silent_data = array.array("h", [0] * 1024).tobytes()
            recorder._audio_callback(silent_data, 1024, {}, 0)
            assert recorder.current_level == 0.0

    def test_audio_callback_computes_level_for_loud_audio(
        self, mock_settings, mock_device_manager
    ):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)
            recorder.is_recording = True

            loud_data = array.array("h", [32767] * 1024).tobytes()
            recorder._audio_callback(loud_data, 1024, {}, 0)
            assert recorder.current_level > 0.99

    def test_audio_callback_handles_empty_data(
        self, mock_settings, mock_device_manager
    ):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)
            recorder.is_recording = True

            result = recorder._audio_callback(b"", 0, {}, 0)
            assert recorder.current_level == 0.0
            assert result == (None, pyaudio.paContinue)

    def test_audio_callback_handles_odd_byte_length(
        self, mock_settings, mock_device_manager
    ):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)
            recorder.is_recording = True

            odd_data = array.array("h", [16000] * 100).tobytes() + b"\x00"
            result = recorder._audio_callback(odd_data, 100, {}, 0)
            assert recorder.current_level > 0.0
            assert result == (None, pyaudio.paContinue)

    def test_write_initial_meta_writes_toml_fields(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        from datetime import datetime

        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
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

        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
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

    def test_mixdown_channels_4_to_2(self, mock_settings, mock_device_manager):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)

            samples = array.array("h", [100, 200, 300, 400, 500, 600, 700, 800])
            result = recorder._mixdown_channels(samples.tobytes(), 4, 2)
            output = array.array("h")
            output.frombytes(result)

            assert output[0] == (100 + 300) // 2
            assert output[1] == (200 + 400) // 2
            assert output[2] == (500 + 700) // 2
            assert output[3] == (600 + 800) // 2

    def test_save_recording_no_frames_raises_error(
        self, tmp_path, mock_settings, mock_device_manager
    ):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)
            recorder.frames = []

            with pytest.raises(RuntimeError, match="No audio data recorded"):
                recorder._save_recording(tmp_path / "audio.wav", 2, 2, 16000)

    def test_start_recording_cleans_up_note_dir_when_no_audio_captured(
        self, tmp_path, mock_settings, mock_device_manager, monkeypatch
    ):
        import threading

        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)

        def instant_exit(*a, **kw):
            recorder.is_recording = False

        monkeypatch.setattr(
            threading.Event, "wait", lambda self, timeout=None: instant_exit()
        )

        with patch.object(recorder, "_cleanup_recording"):
            recorder.audio.open = Mock(return_value=Mock(start_stream=Mock()))
            recorder.audio.get_sample_size = Mock(return_value=2)
            with pytest.raises(RuntimeError, match="No audio data recorded"):
                recorder.start_recording(title="empty")

        created_dirs = list(tmp_path.iterdir())
        assert created_dirs == [], "empty note dir should have been cleaned up"
