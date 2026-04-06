from typing import Optional

import pyaudio


class DeviceManager:
    def __init__(self):
        self.audio = None
        self._initialize_audio()

    def _initialize_audio(self):
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyAudio: {str(e)}")

    def __del__(self):
        if self.audio:
            self.audio.terminate()

    def list_devices(self) -> list[dict]:
        if not self.audio:
            return []

        devices = []
        device_count = self.audio.get_device_count()

        for i in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                devices.append(
                    {
                        "index": i,
                        "name": device_info.get("name", "Unknown"),
                        "max_input_channels": device_info.get("maxInputChannels", 0),
                        "max_output_channels": device_info.get("maxOutputChannels", 0),
                        "default_sample_rate": device_info.get("defaultSampleRate", 0),
                        "host_api": device_info.get("hostApi", 0),
                    }
                )
            except Exception:
                continue

        return devices

    def find_blackhole_device(self) -> Optional[int]:
        devices = self.list_devices()

        blackhole_names = ["BlackHole 2ch", "BlackHole 16ch", "BlackHole"]

        for device in devices:
            device_name = device["name"].lower()
            for blackhole_name in blackhole_names:
                if (
                    blackhole_name.lower() in device_name
                    and device["max_input_channels"] > 0
                ):
                    return int(device["index"])

        return None

    def check_blackhole_available(self) -> bool:
        return self.find_blackhole_device() is not None

    def get_default_input_device(self) -> Optional[int]:
        if not self.audio:
            return None

        try:
            default_device = self.audio.get_default_input_device_info()
            device_index = default_device.get("index")
            return int(device_index) if device_index is not None else None
        except Exception:
            return None

    def get_device_info(self, device_index: int) -> Optional[dict]:
        if not self.audio:
            return None

        try:
            device_info = self.audio.get_device_info_by_index(device_index)
            return dict(device_info) if device_info else None
        except Exception:
            return None

    def test_device(
        self, device_index: int, sample_rate: int = 16000, channels: int = 2
    ) -> bool:
        if not self.audio:
            return False

        device_info = self.get_device_info(device_index)
        if not device_info or device_info["maxInputChannels"] == 0:
            return False

        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=min(channels, device_info["maxInputChannels"]),
                rate=int(min(sample_rate, device_info["defaultSampleRate"])),
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
            )
            stream.close()
            return True
        except Exception:
            return False

    def find_device_by_name(self, name: str) -> Optional[int]:
        devices = self.list_devices()
        search_name = name.lower()

        for device in devices:
            if device["name"].lower() == search_name and device["max_input_channels"] > 0:
                return int(device["index"])

        for device in devices:
            if (
                search_name in device["name"].lower()
                and device["max_input_channels"] > 0
            ):
                return int(device["index"])

        return None

    def get_recommended_device(
        self, configured_device: Optional[str] = None
    ) -> Optional[int]:
        if configured_device:
            device = self.find_device_by_name(configured_device)
            if device is not None:
                return device

        blackhole_device = self.find_blackhole_device()
        if blackhole_device is not None:
            return blackhole_device

        return self.get_default_input_device()

    def get_device_sample_rates(self, device_index: int) -> list[int]:
        standard_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        supported_rates: list[int] = []

        device_info = self.get_device_info(device_index)
        if not device_info:
            return supported_rates

        for rate in standard_rates:
            if self._test_sample_rate(device_index, rate):
                supported_rates.append(rate)

        return supported_rates

    def _test_sample_rate(self, device_index: int, sample_rate: int) -> bool:
        if not self.audio:
            return False

        try:
            self.audio.get_device_info_by_index(device_index)
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
            )
            stream.close()
            return True
        except Exception:
            return False

    def test_pyaudio(self) -> bool:
        try:
            if not self.audio:
                return False
            device_count = self.audio.get_device_count()
            return bool(device_count > 0)
        except Exception:
            return False
