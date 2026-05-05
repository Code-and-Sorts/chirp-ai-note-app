from __future__ import annotations

import logging
import threading
import tomllib
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from threading import Timer
from typing import TYPE_CHECKING

import numpy as np
import tomli_w

from audio_capture import AudioCapture, check_macos_version
from config.settings import ChirpSettings
from recorder.audio_mixer import StereoToMonoMixer
from recorder.meeting_monitor import MeetingMonitor
from utils.file_utils import (
    AUDIO_FILENAME,
    META_FILENAME,
    slugify,
)
from utils.time_utils import get_recording_duration

if TYPE_CHECKING:
    # Imported only for type hints. `recorder.device_manager` pulls in
    # PyAudio at import time, which would defeat this module's
    # platform-neutral import goal on hosts where PyAudio is unavailable.
    from recorder.device_manager import DeviceManager

logger = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 16000
OUTPUT_CHANNELS = 1
OUTPUT_SAMPLE_WIDTH_BYTES = 2


class AudioRecorder:
    def __init__(self, settings: ChirpSettings, device_manager: DeviceManager):
        self.settings = settings
        self.device_manager = device_manager
        self.is_recording = False
        # int16 PCM bytes per drained frame. Storing the int16 view rather
        # than the float32 ndarray halves the in-memory footprint of long
        # recordings (~115 MB/hour at 16 kHz mono vs. ~230 MB/hour for
        # float32) — important because settings.monitoring caps sessions
        # at 8 hours by default.
        self.frames: list[bytes] = []
        self.recording_thread: Timer | None = None
        self.monitor: MeetingMonitor | None = None
        self.start_time: datetime | None = None
        self.title: str | None = None
        self.current_level: float = 0.0
        self.note_dir: Path | None = None
        self.slug: str | None = None
        self._paused: bool = False
        self._capture_thread: threading.Thread | None = None
        self._capture_error: BaseException | None = None

    def start_recording(
        self,
        duration_minutes: int | None = None,
        title: str | None = None,
        level_callback: Callable[[float], None] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        if self.is_recording:
            raise RuntimeError("Recording already in progress")

        check_macos_version()
        _warn_if_audio_settings_overridden(self.settings)
        self._capture_error = None

        notes_root = self.settings.directories.notes_root
        notes_root.mkdir(parents=True, exist_ok=True)

        effective_title = title or "untitled"
        recorded_date = datetime.now()
        slug = slugify(effective_title, recorded_date.date(), notes_root)
        note_dir = notes_root / slug
        note_dir.mkdir(parents=True, exist_ok=True)
        audio_path = note_dir / AUDIO_FILENAME

        self.frames = []
        self.is_recording = True
        self._paused = False
        self.current_level = 0.0
        self.start_time = recorded_date
        self.title = effective_title
        self.note_dir = note_dir
        self.slug = slug

        try:
            with AudioCapture() as cap:
                mic_name = cap.mic_device_name or "default"

                self._write_initial_meta(
                    note_dir=note_dir,
                    title=effective_title,
                    recorded_at=recorded_date,
                    mic=mic_name,
                    tags=list(tags or []),
                )

                self.monitor = MeetingMonitor(
                    self.settings.monitoring,
                    self.start_time,
                    self._on_warning,
                    self._should_stop_recording,
                )
                self.monitor.start()

                if duration_minutes:
                    self.recording_thread = Timer(
                        duration_minutes * 60, self._stop_recording_timer
                    )
                    self.recording_thread.start()

                self._capture_thread = threading.Thread(
                    target=self._capture_worker,
                    args=(cap,),
                    name="audio-recorder-capture",
                    daemon=True,
                )
                self._capture_thread.start()

                try:
                    while self.is_recording:
                        threading.Event().wait(0.1)
                        if level_callback is not None:
                            try:
                                level_callback(self.current_level)
                            except Exception:
                                logger.debug(
                                    "Level callback failed, disabling", exc_info=True
                                )
                                level_callback = None
                except KeyboardInterrupt:
                    self.is_recording = False
        except KeyboardInterrupt:
            pass
        except Exception:
            import shutil

            shutil.rmtree(note_dir, ignore_errors=True)
            raise
        finally:
            self._cleanup_recording()
            self.is_recording = False

        if self._capture_error is not None:
            import shutil

            shutil.rmtree(note_dir, ignore_errors=True)
            raise RuntimeError(
                "audio capture worker crashed mid-recording; recording discarded"
            ) from self._capture_error

        if not self.frames:
            import shutil

            shutil.rmtree(note_dir, ignore_errors=True)
            raise RuntimeError("No audio data recorded")

        self._save_recording(audio_path)

        duration_s = get_recording_duration(self.start_time) if self.start_time else 0.0
        self._update_meta_duration(note_dir, duration_s)

        return slug

    def stop_recording(self):
        self.is_recording = False

    def pause(self) -> None:
        self._paused = True
        self.current_level = 0.0

    def resume(self) -> None:
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return getattr(self, "_paused", False)

    def _capture_worker(self, cap: AudioCapture) -> None:
        mixer = StereoToMonoMixer(
            frame_ms=32, sample_rate=OUTPUT_SAMPLE_RATE, gap_ms=100
        )
        try:
            for source, timestamp_us, samples in cap.frames():
                if not self.is_recording:
                    break
                mixer.feed(source, timestamp_us, samples)
                for _, mixed in mixer.drain():
                    if self._paused:
                        continue
                    self._append_mixed_frame(mixed)
            # Flush any sub-frame tail — helper chunks are arbitrary-sized,
            # so a clean EOF often leaves <512 samples buffered per source.
            tail = mixer.flush()
            if tail is not None and not self._paused:
                _, mixed = tail
                self._append_mixed_frame(mixed)
        except Exception as exc:
            # Stash the error so `start_recording` re-raises it instead of
            # silently truncating the partial recording.
            logger.exception("audio-recorder-capture worker crashed")
            self._capture_error = exc
        finally:
            # `cap.frames()` can also exhaust cleanly on helper EOF; flip
            # the flag in `finally` so the main wait loop exits in both
            # crash and clean-EOF paths instead of polling forever.
            self.is_recording = False

    def _stop_recording_timer(self):
        self.is_recording = False

    def _cleanup_recording(self):
        if self._capture_thread is not None:
            if self._capture_thread.ident is not None:
                self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self.monitor:
            self.monitor.stop()
            self.monitor = None

        if self.recording_thread:
            self.recording_thread.cancel()
            self.recording_thread = None

    def _append_mixed_frame(self, mixed: np.ndarray) -> None:
        clipped = np.clip(mixed, -1.0, 1.0)
        int16 = (clipped * 32767).astype(np.int16, copy=False)
        self.frames.append(int16.tobytes())
        self.current_level = min(1.0, float(np.max(np.abs(mixed))))

    def _save_recording(self, file_path: Path) -> None:
        if not self.frames:
            raise RuntimeError("No audio data recorded")

        with wave.open(str(file_path), "wb") as wave_file:
            wave_file.setnchannels(OUTPUT_CHANNELS)
            wave_file.setsampwidth(OUTPUT_SAMPLE_WIDTH_BYTES)
            wave_file.setframerate(OUTPUT_SAMPLE_RATE)
            wave_file.writeframes(b"".join(self.frames))

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


def _warn_if_audio_settings_overridden(settings: ChirpSettings) -> None:
    """Log a one-line warning when configured audio format != recorder output.

    The bundled CaptureAudio.app helper is hardcoded to 16 kHz mono. If a
    user has overridden `settings.audio.sample_rate` or `.channels` to a
    different value, the recorder will silently produce 16 kHz mono
    anyway. Surface that mismatch so the discrepancy is visible to
    operators reading logs rather than baked into the WAV.
    """
    sample_rate = settings.audio.sample_rate
    channels = settings.audio.channels
    if sample_rate != OUTPUT_SAMPLE_RATE or channels != OUTPUT_CHANNELS:
        logger.warning(
            "audio_recorder: settings.audio.sample_rate=%s channels=%s "
            "differ from recorder output (%s Hz, %s channel%s); produced "
            "WAV will be 16 kHz mono regardless",
            sample_rate,
            channels,
            OUTPUT_SAMPLE_RATE,
            OUTPUT_CHANNELS,
            "" if OUTPUT_CHANNELS == 1 else "s",
        )


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
