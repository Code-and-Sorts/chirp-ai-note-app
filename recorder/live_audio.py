from __future__ import annotations

import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from audio_capture import AudioCapture, check_macos_version
from config.settings import ChirpSettings
from recorder._audio_utils import (
    float32_to_int16_bytes,
    warn_if_audio_settings_overridden,
)
from recorder.audio_mixer import (
    SOURCE_MICROPHONE,
    SOURCE_SYSTEM,
    StereoToMonoMixer,
)
from recorder.live_types import AudioFrame

if TYPE_CHECKING:
    # Imported only for type hints. `recorder.device_manager` pulls in
    # PyAudio at import time, which would defeat this module's
    # platform-neutral import goal on hosts where PyAudio is unavailable.
    from recorder.device_manager import DeviceManager

logger = logging.getLogger(__name__)

_LIVE_SAMPLE_RATE = 16000
_LIVE_CHANNELS = 1
_MIXER_THREAD_JOIN_TIMEOUT = 2.0


class LiveAudioStream:
    def __init__(
        self,
        settings: ChirpSettings,
        device_manager: DeviceManager | None = None,
        frame_queue: queue.Queue[AudioFrame] | None = None,
        stop_event: threading.Event | None = None,
        level_queue: queue.Queue[float] | None = None,
        frame_ms: int = 32,
        channels: int = 2,
        debug_dir: Path | None = None,
    ):
        self.settings = settings
        self.frame_queue: queue.Queue[AudioFrame] = frame_queue or queue.Queue()
        self.stop_event = stop_event or threading.Event()
        self.level_queue: queue.Queue[float] = level_queue or queue.Queue()
        self.frame_ms = frame_ms
        # `channels` constructor arg is retained for backward compatibility;
        # AudioCapture always produces mono mixed output.
        self.channels = _LIVE_CHANNELS
        self.debug_dir = debug_dir
        self._debug_frames: list[bytes] = []

        self._temp_wav_path: Path | None = None
        self._wave: wave.Wave_write | None = None
        self._start_time: float | None = None
        self._frame_duration = frame_ms / 1000.0
        self._sample_rate = _LIVE_SAMPLE_RATE
        self._frame_samples = max(1, int(self._sample_rate * (self.frame_ms / 1000.0)))
        self._recorded_at: datetime | None = None

        self._cap_ctx: AudioCapture | None = None
        self._cap: AudioCapture | None = None
        self._mixer_thread: threading.Thread | None = None
        self._frame_index = 0
        self._capture_error: BaseException | None = None
        self._mic_device_name: str | None = None

    @property
    def capture_error(self) -> BaseException | None:
        return self._capture_error

    @property
    def mic_device_name(self) -> str | None:
        return self._mic_device_name

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_duration(self) -> float:
        return self._frame_duration

    def start(self) -> None:
        check_macos_version()
        warn_if_audio_settings_overridden(
            self.settings,
            logger=logger,
            component="live_audio",
            output_sample_rate=_LIVE_SAMPLE_RATE,
            output_channels=_LIVE_CHANNELS,
        )

        self._sample_rate = _LIVE_SAMPLE_RATE
        self.channels = _LIVE_CHANNELS
        self._frame_samples = max(1, int(self._sample_rate * self.frame_ms / 1000.0))
        self._frame_duration = self._frame_samples / self._sample_rate
        self._debug_frames.clear()
        self._frame_index = 0
        self._capture_error = None
        self._start_time = time.monotonic()
        self._recorded_at = datetime.now()

        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        self._temp_wav_path = Path(tmp_name)
        wav = wave.open(tmp_name, "wb")
        wav.setnchannels(_LIVE_CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(self._sample_rate)
        self._wave = wav

        cap_ctx = AudioCapture()
        cap = cap_ctx.__enter__()
        try:
            self._cap_ctx = cap_ctx
            self._cap = cap
            self._mic_device_name = cap.mic_device_name

            thread = threading.Thread(
                target=self._mixer_loop,
                name="audio-capture-mixer",
                daemon=True,
            )
            self._mixer_thread = thread
            thread.start()
        except BaseException:
            # If anything between AudioCapture entry and thread start fails
            # (e.g., thread exhaustion), the helper subprocess would
            # otherwise leak because nothing else calls __exit__. Tear
            # the context down before re-raising.
            try:
                cap_ctx.__exit__(None, None, None)
            finally:
                self._cap_ctx = None
                self._cap = None
                self._mixer_thread = None
            raise

    def _mixer_loop(self) -> None:
        cap = self._cap
        if cap is None:
            return
        mixer = StereoToMonoMixer(
            frame_ms=self.frame_ms,
            sample_rate=_LIVE_SAMPLE_RATE,
            gap_ms=100,
        )
        try:
            for source, timestamp_us, samples in cap.frames():
                if self.stop_event.is_set():
                    break
                if source not in (SOURCE_SYSTEM, SOURCE_MICROPHONE):
                    continue
                mixer.feed(source, timestamp_us, samples)
                for _ts_us, mixed in mixer.drain():
                    self._publish_mixed_frame(mixed)
            for _ts_us, mixed in mixer.flush():
                self._publish_mixed_frame(mixed)
        except Exception as exc:
            logger.exception("audio-capture-mixer thread crashed")
            self._capture_error = exc
        finally:
            # Cover both crash and clean-EOF paths. If the loop exited cleanly
            # before stop_event was set, the helper closed stdout unexpectedly —
            # mark it as an abnormal end so live_session doesn't save a silently
            # truncated recording.
            if not self.stop_event.is_set() and self._capture_error is None:
                self._capture_error = RuntimeError(
                    "live capture ended unexpectedly (clean EOF)"
                )
            self.stop_event.set()

    def _publish_mixed_frame(self, mixed: np.ndarray) -> None:
        if mixed.size:
            peak = min(1.0, float(np.max(np.abs(mixed))))
        else:
            peak = 0.0
        int16_bytes = float32_to_int16_bytes(mixed)
        if self._wave is not None:
            self._wave.writeframes(int16_bytes)
        if self.debug_dir is not None:
            self._debug_frames.append(int16_bytes)
        timestamp_seconds = self._frame_index * self._frame_duration
        self._frame_index += 1
        frame = AudioFrame(
            data=int16_bytes,
            timestamp=timestamp_seconds,
            duration=self._frame_duration,
            level=peak,
        )
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        try:
            self.level_queue.put_nowait(peak)
        except queue.Full:
            pass

    def stop(self) -> None:
        self.stop_event.set()
        # Signal the helper to stop first so the mixer thread can drain cleanly
        # before we join it. Closing stdout (via __exit__) while _read_frame is
        # blocking would raise ValueError inside the mixer thread.
        cap = self._cap
        proc = getattr(cap, "_proc", None) if cap is not None else None
        if proc is not None:
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass
        thread = self._mixer_thread
        if thread is not None:
            thread.join(timeout=_MIXER_THREAD_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.warning(
                    "audio-capture-mixer thread did not join within %.1fs; "
                    "capture_error may be stale",
                    _MIXER_THREAD_JOIN_TIMEOUT,
                )
            self._mixer_thread = None
        cap_ctx = self._cap_ctx
        if cap_ctx is not None:
            try:
                cap_ctx.__exit__(None, None, None)
            finally:
                self._cap_ctx = None
                self._cap = None
        if self._wave is not None:
            try:
                self._wave.close()
            except Exception:
                pass
            self._wave = None

    def save_recording(self, file_path: Path, title: str | None = None) -> None:
        import json

        if self._temp_wav_path is None or not self._temp_wav_path.exists():
            raise RuntimeError("No audio data captured")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self._temp_wav_path), str(file_path))
        self._temp_wav_path = None

        if self.debug_dir is not None and self._debug_frames:
            debug_chunk_path = self.debug_dir / f"{file_path.stem}_mono.wav"
            with wave.open(str(debug_chunk_path), "wb") as debug_wave:
                debug_wave.setnchannels(1)
                debug_wave.setsampwidth(2)
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

    def close(self) -> None:
        if self._cap_ctx is not None or self._mixer_thread is not None:
            self.stop()
        if self._wave is not None:
            try:
                self._wave.close()
            except Exception:
                pass
            self._wave = None
        if self._temp_wav_path is not None:
            try:
                self._temp_wav_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._temp_wav_path = None
