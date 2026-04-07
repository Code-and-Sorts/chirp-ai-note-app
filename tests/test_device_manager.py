from unittest.mock import Mock, patch

import pytest

from recorder.device_manager import DeviceManager


class TestDeviceManager:
    def test_initialization_success(self):
        with patch("recorder.device_manager.pyaudio.PyAudio") as mock_pyaudio:
            mock_audio = Mock()
            mock_pyaudio.return_value = mock_audio

            device_manager = DeviceManager()

            assert device_manager.audio == mock_audio
            mock_pyaudio.assert_called_once()

    def test_initialization_failure_raises_runtime_error(self):
        with patch("recorder.device_manager.pyaudio.PyAudio") as mock_pyaudio:
            mock_pyaudio.side_effect = Exception("PyAudio init failed")

            with pytest.raises(RuntimeError, match="Failed to initialize PyAudio"):
                DeviceManager()

    def test_list_devices_returns_empty_when_audio_none(self):
        device_manager = DeviceManager.__new__(DeviceManager)
        device_manager.audio = None

        devices = device_manager.list_devices()

        assert devices == []

    def test_list_devices_returns_formatted_device_info(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()
            device_manager.audio.get_device_count.return_value = 2

            mock_device_info = {
                "name": "Test Device",
                "maxInputChannels": 2,
                "maxOutputChannels": 2,
                "defaultSampleRate": 44100.0,
                "hostApi": 0,
            }
            device_manager.audio.get_device_info_by_index.return_value = (
                mock_device_info
            )

            devices = device_manager.list_devices()

            assert len(devices) == 2
            assert devices[0]["index"] == 0
            assert devices[0]["name"] == "Test Device"
            assert devices[0]["max_input_channels"] == 2
            assert devices[0]["default_sample_rate"] == 44100.0

    def test_list_devices_skips_failed_device_queries(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()
            device_manager.audio.get_device_count.return_value = 2

            device_manager.audio.get_device_info_by_index.side_effect = [
                Exception("Device 0 failed"),
                {
                    "name": "Working Device",
                    "maxInputChannels": 1,
                    "maxOutputChannels": 0,
                    "defaultSampleRate": 48000.0,
                    "hostApi": 0,
                },
            ]

            devices = device_manager.list_devices()

            assert len(devices) == 1
            assert devices[0]["name"] == "Working Device"

    def test_find_blackhole_device_returns_index_when_found(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()

            with patch.object(device_manager, "list_devices") as mock_list:
                mock_list.return_value = [
                    {
                        "index": 0,
                        "name": "Built-in Microphone",
                        "max_input_channels": 1,
                    },
                    {"index": 1, "name": "BlackHole 2ch", "max_input_channels": 2},
                    {"index": 2, "name": "Built-in Output", "max_input_channels": 0},
                ]

                result = device_manager.find_blackhole_device()

                assert result == 1

    def test_find_blackhole_device_returns_none_when_not_found(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()

            with patch.object(device_manager, "list_devices") as mock_list:
                mock_list.return_value = [
                    {
                        "index": 0,
                        "name": "Built-in Microphone",
                        "max_input_channels": 1,
                    },
                    {"index": 1, "name": "Built-in Output", "max_input_channels": 0},
                ]

                result = device_manager.find_blackhole_device()

                assert result is None

    def test_find_blackhole_device_ignores_output_only_devices(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()

            with patch.object(device_manager, "list_devices") as mock_list:
                mock_list.return_value = [
                    {
                        "index": 0,
                        "name": "BlackHole 2ch",
                        "max_input_channels": 0,
                    },
                    {"index": 1, "name": "BlackHole 16ch", "max_input_channels": 16},
                ]

                result = device_manager.find_blackhole_device()

                assert result == 1

    def test_get_default_input_device_returns_index(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()
            device_manager.audio.get_default_input_device_info.return_value = {
                "index": 5
            }

            result = device_manager.get_default_input_device()

            assert result == 5

    def test_get_default_input_device_returns_none_on_failure(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()
            device_manager.audio.get_default_input_device_info.side_effect = Exception(
                "No default device"
            )

            result = device_manager.get_default_input_device()

            assert result is None

    def test_get_default_output_device_returns_index(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()
            device_manager.audio.get_default_output_device_info.return_value = {
                "index": 7
            }

            result = device_manager.get_default_output_device()

            assert result == 7

    def test_get_default_output_device_returns_none_on_failure(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()
            device_manager.audio.get_default_output_device_info.side_effect = Exception(
                "No default device"
            )

            result = device_manager.get_default_output_device()

            assert result is None

    def test_get_device_info_returns_device_info(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()

            mock_info = {"name": "Test Device", "maxInputChannels": 2}
            device_manager.audio.get_device_info_by_index.return_value = mock_info

            result = device_manager.get_device_info(1)

            assert result == mock_info

    def test_get_device_info_returns_none_on_failure(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()
            device_manager.audio.get_device_info_by_index.side_effect = Exception(
                "Invalid device"
            )

            result = device_manager.get_device_info(999)

            assert result is None

    def test_test_device_returns_false_when_audio_none(self):
        device_manager = DeviceManager.__new__(DeviceManager)
        device_manager.audio = None

        result = device_manager.test_device(0)

        assert result is False

    def test_test_device_returns_false_for_no_input_channels(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()

            with patch.object(device_manager, "get_device_info") as mock_get_info:
                mock_get_info.return_value = {"maxInputChannels": 0}

                result = device_manager.test_device(0)

                assert result is False

    def test_test_device_opens_and_closes_stream_successfully(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()

            mock_stream = Mock()
            device_manager.audio.open.return_value = mock_stream

            with patch.object(device_manager, "get_device_info") as mock_get_info:
                mock_get_info.return_value = {
                    "maxInputChannels": 2,
                    "defaultSampleRate": 44100.0,
                }

                result = device_manager.test_device(0, sample_rate=16000, channels=1)

                assert result is True
                device_manager.audio.open.assert_called_once()
                mock_stream.close.assert_called_once()

    def test_test_device_returns_false_on_stream_failure(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()
            device_manager.audio = Mock()
            device_manager.audio.open.side_effect = Exception("Stream failed")

            with patch.object(device_manager, "get_device_info") as mock_get_info:
                mock_get_info.return_value = {
                    "maxInputChannels": 2,
                    "defaultSampleRate": 44100.0,
                }

                result = device_manager.test_device(0)

                assert result is False

    def test_find_device_by_name_exact_match(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()

            with patch.object(device_manager, "list_devices") as mock_list:
                mock_list.return_value = [
                    {"index": 0, "name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0},
                    {"index": 1, "name": "Aggregate Device", "max_input_channels": 2, "max_output_channels": 0},
                ]

                result = device_manager.find_device_by_name("Aggregate Device")
                assert result == 1

    def test_find_device_by_name_partial_match(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()

            with patch.object(device_manager, "list_devices") as mock_list:
                mock_list.return_value = [
                    {"index": 0, "name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0},
                    {"index": 1, "name": "My Aggregate Device", "max_input_channels": 2, "max_output_channels": 0},
                ]

                result = device_manager.find_device_by_name("Aggregate")
                assert result == 1

    def test_find_device_by_name_not_found(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()

            with patch.object(device_manager, "list_devices") as mock_list:
                mock_list.return_value = [
                    {"index": 0, "name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0},
                ]

                result = device_manager.find_device_by_name("Aggregate Device")
                assert result is None

    def test_get_recommended_device_returns_system_default(self):
        with patch("recorder.device_manager.pyaudio.PyAudio"):
            device_manager = DeviceManager()

            with patch.object(
                device_manager, "get_default_input_device"
            ) as mock_default:
                mock_default.return_value = 3

                result = device_manager.get_recommended_device()

                assert result == 3
