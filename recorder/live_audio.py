from __future__ import annotations

import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio

from config.settings import ChirpSettings
from recorder.device_manager import DeviceManager
from recorder.live_types import AudioFrame


class LiveAudioStream:
    def __init__(
        self,
        settings: ChirpSettings,
        device_manager: DeviceManager,
        frame_queue: queue.Queue[AudioFrame],
        stop_event: threading.Event,
        level_queue: queue.Queue[float],
        frame_ms: int = 20,
        channels: int = 2,
        debug_dir: Path | None = None,
    ):
        self.settings = settings
        self.device_manager = device_manager
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.level_queue = level_queue
        self.frame_ms = frame_ms
        self.channels = channels
        self.debug_dir = debug_dir
        self._debug_frames: list[bytes] = []

        self._stream: pyaudio.Stream | None = None
        self._audio = pyaudio.PyAudio()
        self._frames: list[bytes] = []
        self._start_time: float | None = None
        self._frame_duration = frame_ms / 1000.0
        self._sample_rate = settings.audio.sample_rate
        self._frame_samples = max(1, int(self._sample_rate * (self.frame_ms / 1000.0)))
        self._frame_index = 0
        self._lock = threading.Lock()
        self._recorded_at: datetime | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frames(self) -> list[bytes]:
        return self._frames

    @property
    def frame_duration(self) -> float:
        return self._frame_duration

    def start(self):
        device_index = self.device_manager.get_recommended_device()
        if device_index is None:
            raise RuntimeError("No suitable audio device found for live transcription")

        device_info = self.device_manager.get_device_info(device_index)
        if not device_info or device_info["maxInputChannels"] == 0:
            raise RuntimeError("Selected device has no input channels")

        max_inputs = int(device_info.get("maxInputChannels", 0))
        channels = min(self.channels, max_inputs if max_inputs > 0 else 1)
        channels = max(1, channels)

        default_rate = device_info.get("defaultSampleRate", self._sample_rate)
        try:
            default_rate = int(float(default_rate))
        except (TypeError, ValueError):
            default_rate = self._sample_rate

        candidate_rates: list[int] = []
        for rate in [self._sample_rate, 16000, 48000, 32000, default_rate]:
            try:
                rate_int = int(float(rate))
            except (TypeError, ValueError):
                continue
            if rate_int > 0 and rate_int not in candidate_rates:
                candidate_rates.append(rate_int)

        stream: pyaudio.Stream | None = None
        last_error: Exception | None = None

        for rate in candidate_rates:
            frame_samples = max(1, int(rate * (self.frame_ms / 1000.0)))
            try:
                stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=frame_samples,
                    stream_callback=self._callback,
                )
                self._sample_rate = rate
                self._frame_samples = frame_samples
                break
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_error = exc
                continue

        if stream is None:
            message = "Failed to open audio stream"
            if last_error:
                message += f": {last_error}"
            raise RuntimeError(message)

        self._frame_duration = self._frame_samples / self._sample_rate
        self._frames.clear()
        self._frame_index = 0
        self._start_time = time.monotonic()
        self._recorded_at = datetime.now()

        self._stream = stream
        self.channels = channels
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status_flags):  # noqa: D401
        if self.stop_event.is_set():
            return (None, pyaudio.paComplete)

        with self._lock:
            if self._start_time is None:
                self._start_time = time.monotonic()

            timestamp = self._frame_index * self._frame_duration
            self._frame_index += 1
            self._frames.append(in_data)

        pcm = np.frombuffer(in_data, dtype=np.int16)
        if self.channels > 1 and pcm.size:
            pcm = pcm.reshape(-1, self.channels)
            mono = np.mean(pcm, axis=1).astype(np.int16)
            peak = np.max(np.abs(pcm), axis=0).mean() / 32768.0 if pcm.size else 0.0
        else:
            mono = pcm.astype(np.int16, copy=False)
            peak = np.max(np.abs(mono)) / 32768.0 if mono.size else 0.0

        mono_bytes = np.ascontiguousarray(mono).tobytes()
        if mono.size:
            peak = min(1.0, peak)
        else:
            peak = 0.0
        frame_level = float(peak)
        frame = AudioFrame(
            data=mono_bytes,
            timestamp=timestamp,
            duration=self._frame_duration,
            level=frame_level,
        )

        if self.debug_dir is not None:
            self._debug_frames.append(mono_bytes)

        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop frame to avoid blocking the audio callback thread
            pass

        try:
            self.level_queue.put_nowait(frame_level)
        except queue.Full:
            pass

        return (None, pyaudio.paContinue)

    def stop(self):
        self.stop_event.set()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def save_recording(self, file_path: Path, title: str | None = None):
        import json

        if not self._frames:
            raise RuntimeError("No audio data captured")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(file_path), "wb") as wave_file:
            wave_file.setnchannels(self.channels)
            wave_file.setsampwidth(self._audio.get_sample_size(pyaudio.paInt16))
            wave_file.setframerate(self._sample_rate)
            wave_file.writeframes(b"".join(self._frames))

        if self.debug_dir is not None and self._debug_frames:
            debug_chunk_path = self.debug_dir / f"{file_path.stem}_mono.wav"
            with wave.open(str(debug_chunk_path), "wb") as debug_wave:
                debug_wave.setnchannels(1)
                debug_wave.setsampwidth(self._audio.get_sample_size(pyaudio.paInt16))
                debug_wave.setframerate(self._sample_rate)
                debug_wave.writeframes(b"".join(self._debug_frames))

        if title:
            metadata_file = file_path.with_suffix(f"{file_path.suffix}.meta")
            metadata = {
                "title": title,
                "recorded_at": self._recorded_at.isoformat()
                if self._recorded_at
                else None,
                "channels": self.channels,
                "sample_rate": self._sample_rate,
            }
            with metadata_file.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)

    def close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio:
            self._audio.terminate()
            self._audio = None
