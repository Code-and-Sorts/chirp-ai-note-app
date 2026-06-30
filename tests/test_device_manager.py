from unittest.mock import patch

from recorder.device_manager import DeviceManager

_QUERY = "recorder.device_manager.sd.query_devices"


def _device(index, name, *, in_ch=1, out_ch=0, rate=48000.0):
    return {
        "index": index,
        "name": name,
        "max_input_channels": in_ch,
        "max_output_channels": out_ch,
        "default_samplerate": rate,
        "hostapi": 0,
    }


class TestDeviceManager:
    def test_list_devices_returns_formatted_device_info(self):
        devices = [
            _device(0, "Built-in Microphone"),
            _device(1, "Speakers", in_ch=0, out_ch=2),
        ]
        with patch(_QUERY, return_value=devices):
            result = DeviceManager().list_devices()

        assert len(result) == 2
        assert result[0]["index"] == 0
        assert result[0]["name"] == "Built-in Microphone"
        assert result[0]["max_input_channels"] == 1
        assert result[0]["default_sample_rate"] == 48000.0
        assert result[0]["host_api"] == 0

    def test_list_devices_returns_empty_on_error(self):
        with patch(_QUERY, side_effect=Exception("PortAudio unavailable")):
            assert DeviceManager().list_devices() == []

    def test_get_default_input_device_returns_index(self):
        with patch(_QUERY, return_value=_device(5, "Mic")):
            assert DeviceManager().get_default_input_device() == 5

    def test_get_default_input_device_returns_none_when_no_default(self):
        with patch(_QUERY, side_effect=Exception("no default input")):
            assert DeviceManager().get_default_input_device() is None

    def test_get_default_output_device_returns_index(self):
        with patch(_QUERY, return_value=_device(7, "Speakers", in_ch=0, out_ch=2)):
            assert DeviceManager().get_default_output_device() == 7

    def test_get_default_output_device_returns_none_when_no_default(self):
        with patch(_QUERY, side_effect=Exception("no default output")):
            assert DeviceManager().get_default_output_device() is None

    def test_find_device_by_name_exact_match_is_case_insensitive(self):
        devices = [
            _device(0, "Built-in Microphone"),
            _device(1, "Aggregate Device", in_ch=2),
        ]
        with patch(_QUERY, return_value=devices):
            assert DeviceManager().find_device_by_name("aggregate device") == 1

    def test_find_device_by_name_partial_match(self):
        devices = [
            _device(0, "Built-in Microphone"),
            _device(1, "My Aggregate Device", in_ch=2),
        ]
        with patch(_QUERY, return_value=devices):
            assert DeviceManager().find_device_by_name("Aggregate") == 1

    def test_find_device_by_name_not_found(self):
        with patch(_QUERY, return_value=[_device(0, "Built-in Microphone")]):
            assert DeviceManager().find_device_by_name("Aggregate Device") is None

    def test_get_recommended_device_uses_default_input(self):
        with patch(_QUERY, return_value=_device(3, "Mic")):
            assert DeviceManager().get_recommended_device() == 3

    def test_usable_as_context_manager(self):
        with patch(_QUERY, return_value=[]):
            with DeviceManager() as device_manager:
                assert device_manager.list_devices() == []
