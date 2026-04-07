import array
import logging
import math
import threading
import wave
from datetime import datetime
from pathlib import Path
from threading import Timer
from typing import Callable, Optional

import pyaudio

from config.settings import ChirpSettings
from recorder.device_manager import DeviceManager
from recorder.meeting_monitor import MeetingMonitor
from utils.file_utils import generate_audio_filename
from utils.time_utils import get_recording_duration

logger = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(self, settings: ChirpSettings, device_manager: DeviceManager):
        self.settings = settings
        self.device_manager = device_manager
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.frames: list[bytes] = []
        self.stream = None
        self.recording_thread: Optional[Timer] = None
        self.monitor: Optional[MeetingMonitor] = None
        self.start_time: Optional[datetime] = None
        self.title: Optional[str] = None
        self.current_level: float = 0.0

    def __del__(self):
        if self.audio:
            self.audio.terminate()

    def start_recording(
        self,
        duration_minutes: Optional[int] = None,
        title: Optional[str] = None,
        level_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        if self.is_recording:
            raise RuntimeError("Recording already in progress")

        device_index = self.device_manager.get_recommended_device()
        if device_index is None:
            raise RuntimeError("No suitable audio device found")

        device_info = self.device_manager.get_device_info(device_index)
        if not device_info or device_info["maxInputChannels"] == 0:
            raise RuntimeError("Selected device has no input channels")

        self.settings.directories.raw_audio.mkdir(parents=True, exist_ok=True)

        filename = generate_audio_filename(title, self.settings.audio.format)
        file_path = self.settings.directories.raw_audio / filename

        record_channels = int(device_info["maxInputChannels"])
        output_channels = min(self.settings.audio.channels, record_channels)
        sample_rate = min(
            self.settings.audio.sample_rate, int(device_info["defaultSampleRate"])
        )

        self._record_channels = record_channels
        self._output_channels = output_channels
        self.frames = []
        self.is_recording = True
        self.start_time = datetime.now()
        self.title = title

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=record_channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.settings.audio.chunk_size,
            stream_callback=self._audio_callback,
        )

        self.monitor = MeetingMonitor(
            self.settings.monitoring,
            self.start_time,
            self._on_warning,
            self._should_stop_recording,
        )

        if self.stream:
            self.stream.start_stream()
        if self.monitor:
            self.monitor.start()

        if duration_minutes:
            self.recording_thread = Timer(
                duration_minutes * 60, self._stop_recording_timer
            )
            if self.recording_thread:
                self.recording_thread.start()

        try:
            while self.is_recording:
                threading.Event().wait(0.1)
                if level_callback is not None:
                    try:
                        level_callback(self.current_level)
                    except Exception:
                        logger.debug("Level callback failed, disabling", exc_info=True)
                        level_callback = None
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup_recording()

        self._save_recording(
            file_path,
            self._record_channels,
            self._output_channels,
            sample_rate,
            self.title,
        )

        return filename

    def stop_recording(self):
        self.is_recording = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            if in_data:
                self.frames.append(in_data)
            else:
                self.current_level = 0.0
                return (None, pyaudio.paContinue)

            try:
                if len(in_data) < 2:
                    self.current_level = 0.0
                    return (None, pyaudio.paContinue)

                level_data = in_data
                if len(level_data) % 2 != 0:
                    level_data = level_data[:-1]

                if len(level_data) < 2:
                    self.current_level = 0.0
                    return (None, pyaudio.paContinue)

                samples = array.array("h")
                samples.frombytes(level_data)
                rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                self.current_level = min(rms / 32768.0, 1.0)
            except Exception:
                self.current_level = 0.0
        return (None, pyaudio.paContinue)

    def _stop_recording_timer(self):
        self.is_recording = False

    def _cleanup_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.monitor:
            self.monitor.stop()
            self.monitor = None

        if self.recording_thread:
            self.recording_thread.cancel()
            self.recording_thread = None

    def _save_recording(
        self,
        file_path: Path,
        record_channels: int,
        output_channels: int,
        sample_rate: int,
        title: Optional[str] = None,
    ):
        if not self.frames:
            raise RuntimeError("No audio data recorded")

        raw_data = b"".join(self.frames)

        if record_channels > output_channels:
            raw_data = self._mixdown_channels(
                raw_data, record_channels, output_channels
            )

        with wave.open(str(file_path), "wb") as wave_file:
            wave_file.setnchannels(output_channels)
            wave_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(raw_data)

        if title:
            import json

            metadata_file = file_path.with_suffix(f"{file_path.suffix}.meta")
            metadata = {
                "title": title,
                "recorded_at": self.start_time.isoformat() if self.start_time else None,
                "channels": output_channels,
                "sample_rate": sample_rate,
            }
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

    @staticmethod
    def _mixdown_channels(
        raw_data: bytes, input_channels: int, output_channels: int
    ) -> bytes:
        samples = array.array("h")
        samples.frombytes(raw_data)

        total_frames = len(samples) // input_channels
        output = array.array("h")

        for frame in range(total_frames):
            offset = frame * input_channels
            for out_ch in range(output_channels):
                mixed = 0
                sources = 0
                for in_ch in range(out_ch, input_channels, output_channels):
                    mixed += samples[offset + in_ch]
                    sources += 1
                mixed = max(-32768, min(32767, mixed // sources))
                output.append(mixed)

        return output.tobytes()

    def _on_warning(self, elapsed_minutes: int):
        from utils.popup_manager import PopupManager

        popup_manager = PopupManager()
        popup_manager.show_recording_warning(elapsed_minutes)

    def _should_stop_recording(self) -> bool:
        if not self.start_time:
            return False

        max_hours = self.settings.monitoring.max_recording_hours
        elapsed_hours = get_recording_duration(self.start_time) / 3600

        return elapsed_hours >= max_hours

    def get_recording_status(self) -> dict:
        if not self.is_recording or not self.start_time:
            return {"is_recording": False, "duration": 0, "start_time": None}

        return {
            "is_recording": True,
            "duration": get_recording_duration(self.start_time),
            "start_time": self.start_time,
        }
