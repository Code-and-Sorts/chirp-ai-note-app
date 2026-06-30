import logging

import sounddevice as sd

logger = logging.getLogger(__name__)


class DeviceManager:
    """Enumerate audio devices via sounddevice (PortAudio).

    Used for the `chirp devices` listing and the mic-name label shown while
    recording; the actual capture runs through ScreenCaptureKit, not PortAudio.
    sounddevice owns PortAudio's lifecycle and ships it inside its wheel, so
    queries degrade to empty/None instead of raising when audio is unavailable.
    """

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def list_devices(self) -> list[dict]:
        try:
            devices = sd.query_devices()
        except Exception as exc:  # noqa: BLE001 - PortAudio can raise many types
            logger.debug("query_devices failed: %s", exc)
            return []

        return [
            {
                "index": device["index"],
                "name": device.get("name", "Unknown"),
                "max_input_channels": device.get("max_input_channels", 0),
                "max_output_channels": device.get("max_output_channels", 0),
                "default_sample_rate": device.get("default_samplerate", 0),
                "host_api": device.get("hostapi", 0),
            }
            for device in devices
        ]

    def get_default_input_device(self) -> int | None:
        return self._default_device_index("input")

    def get_default_output_device(self) -> int | None:
        return self._default_device_index("output")

    def _default_device_index(self, kind: str) -> int | None:
        try:
            info = sd.query_devices(kind=kind)
        except Exception as exc:  # noqa: BLE001 - raises when no default of this kind
            logger.debug("no default %s device: %s", kind, exc)
            return None
        index = info.get("index")
        return int(index) if index is not None else None

    def find_device_by_name(self, name: str) -> int | None:
        devices = self.list_devices()
        search_name = name.lower()

        for device in devices:
            if (
                device["name"].lower() == search_name
                and device["max_input_channels"] > 0
            ):
                return int(device["index"])

        for device in devices:
            if (
                search_name in device["name"].lower()
                and device["max_input_channels"] > 0
            ):
                return int(device["index"])

        return None

    def get_recommended_device(self) -> int | None:
        return self.get_default_input_device()
