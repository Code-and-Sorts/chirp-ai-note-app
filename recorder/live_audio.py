from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from audio_capture import AudioCapture, check_macos_version
from config.settings import ChirpSettings
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


def _warn_if_audio_settings_overridden(settings: ChirpSettings) -> None:
    """Log a one-line warning when configured audio format != live output.

    Mirrors the offline recorder's check: live capture is hardcoded to
    16 kHz mono and any user override is silently ignored, so we surface
    the mismatch in logs rather than producing audio that disagrees with
    the displayed config.
    """
    sample_rate = settings.audio.sample_rate
    channels = settings.audio.channels
    if sample_rate != _LIVE_SAMPLE_RATE or channels != _LIVE_CHANNELS:
        logger.warning(
            "live_audio: settings.audio.sample_rate=%s channels=%s differ "
            "from live capture output (%s Hz, %s channel%s); live frames "
            "and saved WAV will be 16 kHz mono regardless",
            sample_rate,
            channels,
            _LIVE_SAMPLE_RATE,
            _LIVE_CHANNELS,
            "" if _LIVE_CHANNELS == 1 else "s",
        )


class LiveAudioStream:
    def __init__(
        self,
        settings: ChirpSettings,
        device_manager: DeviceManager,
        frame_queue: queue.Queue[AudioFrame],
        stop_event: threading.Event,
        level_queue: queue.Queue[float],
        frame_ms: int = 32,
        channels: int = 2,
        debug_dir: Path | None = None,
    ):
        self.settings = settings
        self.device_manager = device_manager
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.level_queue = level_queue
        self.frame_ms = frame_ms
        # `channels` constructor arg is retained for backward compatibility;
        # AudioCapture always produces mono mixed output.
        self.channels = _LIVE_CHANNELS
        self.debug_dir = debug_dir
        self._debug_frames: list[bytes] = []

        self._frames: list[bytes] = []
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
    def frames(self) -> list[bytes]:
        return self._frames

    @property
    def frame_duration(self) -> float:
        return self._frame_duration

    def start(self):
        check_macos_version()
        _warn_if_audio_settings_overridden(self.settings)

        self._sample_rate = _LIVE_SAMPLE_RATE
        self.channels = _LIVE_CHANNELS
        self._frame_samples = max(1, int(self._sample_rate * self.frame_ms / 1000.0))
        self._frame_duration = self._frame_samples / self._sample_rate
        self._frames.clear()
        self._debug_frames.clear()
        self._frame_index = 0
        self._capture_error = None
        self._start_time = time.monotonic()
        self._recorded_at = datetime.now()

        cap_ctx = AudioCapture()
        cap = cap_ctx.__enter__()
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
            # Flush any sub-frame tail before the thread exits — see
            # StereoToMonoMixer.flush docstring for the rationale.
            tail = mixer.flush()
            if tail is not None:
                _, mixed = tail
                self._publish_mixed_frame(mixed)
        except Exception as exc:
            # Stash the error so the live session can surface it instead
            # of completing with a silently truncated recording.
            logger.exception("audio-capture-mixer thread crashed")
            self._capture_error = exc
        finally:
            # `cap.frames()` can also exhaust cleanly when the helper
            # closes stdout and exits 0; in that case no exception fires
            # but no further frames will arrive. Setting stop_event in
            # `finally` covers both crash and clean-EOF paths so a live
            # session without a duration cap doesn't wait forever.
            self.stop_event.set()

    def _publish_mixed_frame(self, mixed: np.ndarray) -> None:
        if mixed.size:
            peak = min(1.0, float(np.max(np.abs(mixed))))
        else:
            peak = 0.0
        clipped = np.clip(mixed, -1.0, 1.0)
        int16_bytes = (clipped * 32767).astype(np.int16).tobytes()
        # Synthesize a monotonic per-frame timestamp so VAD / chunker
        # cadence reflects the audio timeline rather than mixer-thread
        # scheduling latency.
        timestamp_seconds = self._frame_index * self._frame_duration
        self._frame_index += 1
        frame = AudioFrame(
            data=int16_bytes,
            timestamp=timestamp_seconds,
            duration=self._frame_duration,
            level=peak,
        )
        self._frames.append(int16_bytes)
        if self.debug_dir is not None:
            self._debug_frames.append(int16_bytes)
        with contextlib.suppress(queue.Full):
            self.frame_queue.put_nowait(frame)
        with contextlib.suppress(queue.Full):
            self.level_queue.put_nowait(peak)

    def stop(self):
        self.stop_event.set()
        thread = self._mixer_thread
        cap_ctx = self._cap_ctx
        if cap_ctx is not None:
            try:
                cap_ctx.__exit__(None, None, None)
            finally:
                self._cap_ctx = None
                self._cap = None
        if thread is not None:
            thread.join(timeout=_MIXER_THREAD_JOIN_TIMEOUT)
            if thread.is_alive():
                # If the mixer thread didn't exit within the timeout,
                # `capture_error` may not yet be fully published. Surface
                # this so the caller can correlate a missing-error report
                # against a real timeout.
                logger.warning(
                    "audio-capture-mixer thread did not join within %.1fs; "
                    "capture_error may be stale",
                    _MIXER_THREAD_JOIN_TIMEOUT,
                )
            self._mixer_thread = None

    def save_recording(self, file_path: Path, title: str | None = None):
        import json

        if not self._frames:
            raise RuntimeError("No audio data captured")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(file_path), "wb") as wave_file:
            wave_file.setnchannels(self.channels)
            wave_file.setsampwidth(2)
            wave_file.setframerate(self._sample_rate)
            wave_file.writeframes(b"".join(self._frames))

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

    def close(self):
        if self._cap_ctx is not None or self._mixer_thread is not None:
            self.stop()
