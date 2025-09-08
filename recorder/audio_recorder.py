import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyaudio

from config.settings import ChirpSettings
from recorder.device_manager import DeviceManager
from recorder.meeting_monitor import MeetingMonitor
from utils.file_utils import generate_audio_filename
from utils.time_utils import get_recording_duration


class AudioRecorder:
    def __init__(self, settings: ChirpSettings, device_manager: DeviceManager):
        self.settings = settings
        self.device_manager = device_manager
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.frames = []
        self.stream = None
        self.recording_thread = None
        self.monitor = None
        self.start_time = None

    def __del__(self):
        if self.audio:
            self.audio.terminate()

    def start_recording(
        self, duration_minutes: Optional[int] = None, title: Optional[str] = None
    ) -> str:
        if self.is_recording:
            raise RuntimeError("Recording already in progress")

        device_index = self.device_manager.get_recommended_device()
        if device_index is None:
            raise RuntimeError("No suitable audio device found")

        device_info = self.device_manager.get_device_info(device_index)
        if device_info["maxInputChannels"] == 0:
            raise RuntimeError("Selected device has no input channels")

        self.settings.directories.raw_audio.mkdir(parents=True, exist_ok=True)

        filename = generate_audio_filename(title, self.settings.audio.format)
        file_path = self.settings.directories.raw_audio / filename

        max_channels = min(
            self.settings.audio.channels, device_info["maxInputChannels"]
        )
        sample_rate = min(
            self.settings.audio.sample_rate, int(device_info["defaultSampleRate"])
        )

        self.frames = []
        self.is_recording = True
        self.start_time = datetime.now()

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=max_channels,
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

        self.stream.start_stream()
        self.monitor.start()

        if duration_minutes:
            self.recording_thread = threading.Timer(
                duration_minutes * 60, self._stop_recording_timer
            )
            self.recording_thread.start()

        try:
            while self.is_recording:
                threading.Event().wait(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup_recording()

        self._save_recording(file_path, max_channels, sample_rate)

        return filename

    def stop_recording(self):
        self.is_recording = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(in_data)
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

    def _save_recording(self, file_path: Path, channels: int, sample_rate: int):
        if not self.frames:
            raise RuntimeError("No audio data recorded")

        with wave.open(str(file_path), "wb") as wave_file:
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(b"".join(self.frames))

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
