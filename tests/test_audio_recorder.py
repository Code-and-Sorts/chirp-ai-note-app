import array
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from recorder.audio_recorder import AudioRecorder


class TestAudioRecorder:
    @pytest.fixture
    def mock_settings(self):
        settings = Mock()
        directories = Mock()
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_path.__truediv__ = Mock(return_value=Path("/mock/audio/test.wav"))
        directories.raw_audio = mock_path
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
        return device_manager

    def test_initialization(self, mock_settings, mock_device_manager):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)

            assert recorder.settings == mock_settings
            assert recorder.device_manager == mock_device_manager
            assert recorder.is_recording is False
            assert recorder.title is None
            assert recorder.current_level == 0.0

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

    def test_save_recording_with_title_creates_metadata_file(
        self, mock_settings, mock_device_manager
    ):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)
            recorder.frames = [b"mock_audio_data"]
            recorder.start_time = Mock()
            recorder.start_time.isoformat.return_value = "2025-09-17T14:30:00"

            test_file_path = Path("/mock/audio/test_recording.wav")
            test_title = "Important Meeting"

            with patch("recorder.audio_recorder.wave.open") as mock_wave:
                with patch("builtins.open", create=True) as mock_file_open:
                    with patch("json.dump") as mock_json_dump:
                        mock_wave_file = Mock()
                        mock_wave.return_value.__enter__.return_value = mock_wave_file

                        recorder._save_recording(test_file_path, 2, 16000, test_title)

                        expected_metadata_path = Path(
                            "/mock/audio/test_recording.wav.meta"
                        )
                        mock_file_open.assert_called_with(
                            expected_metadata_path, "w", encoding="utf-8"
                        )

                        expected_metadata = {
                            "title": "Important Meeting",
                            "recorded_at": "2025-09-17T14:30:00",
                            "channels": 2,
                            "sample_rate": 16000,
                        }
                        mock_json_dump.assert_called_once()
                        call_args = mock_json_dump.call_args
                        assert call_args[0][0] == expected_metadata

    def test_save_recording_without_title_no_metadata_file(
        self, mock_settings, mock_device_manager
    ):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)
            recorder.frames = [b"mock_audio_data"]

            test_file_path = Path("/mock/audio/test_recording.wav")

            with patch("recorder.audio_recorder.wave.open") as mock_wave:
                with patch("builtins.open", create=True) as mock_file_open:
                    with patch("json.dump") as mock_json_dump:
                        mock_wave_file = Mock()
                        mock_wave.return_value.__enter__.return_value = mock_wave_file

                        recorder._save_recording(test_file_path, 2, 16000, None)

                        mock_file_open.assert_not_called()
                        mock_json_dump.assert_not_called()

    def test_title_storage(self, mock_settings, mock_device_manager):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)

            assert recorder.title is None

            test_title = "Weekly Standup"
            recorder.title = test_title
            assert recorder.title == test_title

    def test_save_recording_no_frames_raises_error(
        self, mock_settings, mock_device_manager
    ):
        with patch("recorder.audio_recorder.pyaudio.PyAudio"):
            recorder = AudioRecorder(mock_settings, mock_device_manager)
            recorder.frames = []

            test_file_path = Path("/mock/audio/test_recording.wav")

            with pytest.raises(RuntimeError, match="No audio data recorded"):
                recorder._save_recording(test_file_path, 2, 16000, "Test Title")
