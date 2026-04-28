import array
import logging
import math
import threading
import tomllib
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from threading import Timer

import pyaudio
import tomli_w

from config.settings import ChirpSettings
from recorder.device_manager import DeviceManager
from recorder.meeting_monitor import MeetingMonitor
from utils.file_utils import (
    AUDIO_FILENAME,
    META_FILENAME,
    slugify,
)
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
        self.recording_thread: Timer | None = None
        self.monitor: MeetingMonitor | None = None
        self.start_time: datetime | None = None
        self.title: str | None = None
        self.current_level: float = 0.0
        self._record_channels: int = 1
        self._output_channels: int = 1
        self.note_dir: Path | None = None
        self.slug: str | None = None
        self._paused: bool = False

    def __del__(self):
        if self.audio:
            self.audio.terminate()

    def start_recording(
        self,
        duration_minutes: int | None = None,
        title: str | None = None,
        level_callback: Callable[[float], None] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        if self.is_recording:
            raise RuntimeError("Recording already in progress")

        device_index = self.device_manager.get_recommended_device()
        if device_index is None:
            raise RuntimeError("No suitable audio device found")

        device_info = self.device_manager.get_device_info(device_index)
        if not device_info or device_info["maxInputChannels"] == 0:
            raise RuntimeError("Selected device has no input channels")

        notes_root = self.settings.directories.notes_root
        notes_root.mkdir(parents=True, exist_ok=True)

        effective_title = title or "untitled"
        recorded_date = datetime.now()
        slug = slugify(effective_title, recorded_date.date(), notes_root)
        note_dir = notes_root / slug
        note_dir.mkdir(parents=True, exist_ok=True)
        audio_path = note_dir / AUDIO_FILENAME

        mic_name = self._resolve_mic_name(device_index)

        record_channels = int(device_info["maxInputChannels"])
        output_channels = min(self.settings.audio.channels, record_channels)
        sample_rate = min(
            self.settings.audio.sample_rate, int(device_info["defaultSampleRate"])
        )

        self._record_channels = record_channels
        self._output_channels = output_channels
        self.frames = []
        self.is_recording = True
        self._paused = False
        self.start_time = recorded_date
        self.title = effective_title
        self.note_dir = note_dir
        self.slug = slug

        self._write_initial_meta(
            note_dir=note_dir,
            title=effective_title,
            recorded_at=recorded_date,
            mic=mic_name,
            tags=list(tags or []),
        )

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

        if not self.frames:
            import shutil

            shutil.rmtree(note_dir, ignore_errors=True)
            raise RuntimeError("No audio data recorded")

        self._save_recording(
            audio_path,
            self._record_channels,
            self._output_channels,
            sample_rate,
        )

        duration_s = get_recording_duration(self.start_time) if self.start_time else 0.0
        self._update_meta_duration(note_dir, duration_s)

        return slug

    def stop_recording(self):
        self.is_recording = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return getattr(self, "_paused", False)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            if not in_data:
                self.current_level = 0.0
                return (None, pyaudio.paContinue)

            try:
                sanitized = in_data
                if len(sanitized) % 2 != 0:
                    sanitized = sanitized[:-1]

                if len(sanitized) < 2:
                    self.current_level = 0.0
                    return (None, pyaudio.paContinue)

                if not self.is_paused:
                    self.frames.append(sanitized)

                samples = array.array("h")
                samples.frombytes(sanitized)
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

    def _write_initial_meta(
        self,
        note_dir: Path,
        title: str,
        recorded_at: datetime,
        mic: str,
        tags: list[str],
    ) -> None:
        meta = {
            "title": title,
            "date": recorded_at.isoformat(),
            "mic": mic,
            "tags": tags,
        }
        _write_meta(note_dir / META_FILENAME, meta)

    def _update_meta_duration(self, note_dir: Path, duration_s: float) -> None:
        meta_path = note_dir / META_FILENAME
        meta = _read_meta(meta_path)
        meta["duration_s"] = float(duration_s)
        _write_meta(meta_path, meta)

    def _resolve_mic_name(self, device_index: int) -> str:
        try:
            devices = self.device_manager.list_devices()
            for device in devices:
                if device["index"] == device_index:
                    name = device.get("name")
                    if isinstance(name, str):
                        return name
        except Exception:
            pass
        return "default"

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
                mixed = max(-32768, min(32767, int(mixed / sources)))
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


def _read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("rb") as fh:
            return tomllib.load(fh)
    except Exception:
        return {}


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("wb") as fh:
        tomli_w.dump(meta, fh)
