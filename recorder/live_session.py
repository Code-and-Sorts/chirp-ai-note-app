from __future__ import annotations

import queue
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recorder.vad_chunker import VADChunker

import tomli_w
from rich.console import Console

from chirp.exceptions import RecordingError
from config.settings import ChirpSettings
from recorder.device_manager import DeviceManager
from recorder.live_audio import LiveAudioStream
from recorder.live_dashboard import LiveDashboard
from recorder.live_transcriber import LiveTranscriber
from recorder.live_types import DashboardEvent, SpeechChunk
from utils.file_utils import AUDIO_FILENAME, META_FILENAME, slugify


@dataclass
class LiveSessionResult:
    audio_path: Path
    transcript_path: Path | None
    duration_seconds: float
    total_words: int


class LiveTranscriptionSession:
    def __init__(
        self,
        settings: ChirpSettings,
        device_manager: DeviceManager,
        console: Console,
        title: str | None = None,
        duration_minutes: int | None = None,
        debug: bool = False,
        tags: list[str] | None = None,
    ):
        self.settings = settings
        self.device_manager = device_manager
        self.console = console
        self.title = title
        self.duration_minutes = duration_minutes
        self.debug = debug
        self.tags: list[str] = list(tags or [])

        self.stop_event = threading.Event()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=500)
        self.chunk_queue: queue.Queue = queue.Queue(maxsize=50)
        self.event_queue: queue.Queue[DashboardEvent] = queue.Queue(maxsize=200)
        self.level_queue: queue.Queue[float] = queue.Queue(maxsize=50)

        self.audio_stream: LiveAudioStream | None = None
        self.dashboard: LiveDashboard | None = None
        self.vad_chunker: VADChunker | None = None
        self.transcriber: LiveTranscriber | None = None
        self.start_time = time.monotonic()

    def run(self) -> LiveSessionResult:
        self.start_time = time.monotonic()
        self.audio_stream = LiveAudioStream(
            settings=self.settings,
            device_manager=self.device_manager,
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
            level_queue=self.level_queue,
            debug_dir=self._debug_dir if self.debug else None,
        )

        try:
            self.audio_stream.start()
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise RecordingError(str(exc)) from exc

        self._publish_event(
            DashboardEvent(
                type="info",
                payload={
                    "sample_rate": self.audio_stream.sample_rate,
                    "channels": self.audio_stream.channels,
                },
            )
        )

        self._start_level_forwarder()

        if self.debug:
            self._start_direct_forwarder()
        else:
            from recorder.vad_chunker import VADChunker

            self.vad_chunker = VADChunker(
                frame_queue=self.frame_queue,
                chunk_queue=self.chunk_queue,
                stop_event=self.stop_event,
                sample_rate=self.audio_stream.sample_rate,
                energy_threshold=0.005,
                event_queue=self.event_queue,
                max_chunk_seconds=15.0,
            )
            self.vad_chunker.start()

        self.transcriber = LiveTranscriber(
            settings=self.settings,
            chunk_queue=self.chunk_queue,
            event_queue=self.event_queue,
            stop_event=self.stop_event,
            meeting_name=self.title,
            sample_rate=self.audio_stream.sample_rate,
            debug_dir=self._debug_dir if self.debug else None,
            transcription_interval=1.0,
        )
        self.transcriber.start()

        self.dashboard = LiveDashboard(
            console=self.console,
            event_queue=self.event_queue,
            stop_event=self.stop_event,
            start_time=self.start_time,
        )

        dashboard_thread = threading.Thread(target=self.dashboard.run, daemon=True)
        dashboard_thread.start()

        try:
            self._wait_for_completion()
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            self._stop_pipeline()
            dashboard_thread.join(timeout=1)

        if not self.audio_stream:
            raise RecordingError("Audio stream not initialized")

        capture_error = self.audio_stream.capture_error
        if capture_error is not None:
            raise RecordingError(
                f"audio capture worker crashed mid-recording: {capture_error}"
            ) from capture_error

        notes_root = self.settings.directories.notes_root
        notes_root.mkdir(parents=True, exist_ok=True)

        from datetime import date

        effective_title = self.title or "untitled"
        slug = slugify(effective_title, date.today(), notes_root)
        note_dir = notes_root / slug
        note_dir.mkdir(parents=True, exist_ok=True)
        audio_path = note_dir / AUDIO_FILENAME

        self.audio_stream.save_recording(audio_path, title=self.title)
        self.audio_stream.close()

        self._write_live_meta(note_dir, effective_title)

        total_words = 0
        transcript_path = None
        if self.transcriber:
            total_words = self.transcriber.total_words
            if self.transcriber.segments:
                transcript_path = note_dir / "transcript.live.txt"
                self.transcriber.export_transcript(transcript_path)

        duration_seconds = time.monotonic() - self.start_time

        return LiveSessionResult(
            audio_path=audio_path,
            transcript_path=transcript_path,
            duration_seconds=duration_seconds,
            total_words=total_words,
        )

    def _stop_pipeline(self):
        self.stop_event.set()
        if self.audio_stream:
            self.audio_stream.stop()
        if self.vad_chunker:
            self.vad_chunker.join(timeout=1)
        if self.transcriber:
            self.transcriber.join(timeout=1)

    def stop(self):
        self._stop_pipeline()
        if self.audio_stream:
            self.audio_stream.close()

    def _wait_for_completion(self):
        if self.duration_minutes is None:
            while not self.stop_event.is_set():
                time.sleep(0.1)
            return

        target_seconds = self.duration_minutes * 60
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - self.start_time
            if elapsed >= target_seconds:
                self.stop_event.set()
                break
            time.sleep(0.1)

    def _start_level_forwarder(self):
        def forward_levels():
            while not self.stop_event.is_set():
                try:
                    level = self.level_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                event = DashboardEvent(type="level", payload={"value": level})
                try:
                    self.event_queue.put_nowait(event)
                except queue.Full:
                    pass

        threading.Thread(target=forward_levels, daemon=True).start()

    def _start_direct_forwarder(self):
        chunk_counter = [0]

        def _emit_chunk(
            buffer: list[bytes], start: float | None, frame_duration: float
        ):
            if not buffer or start is None:
                return
            data = b"".join(buffer)
            end = start + len(buffer) * frame_duration
            chunk = SpeechChunk(data=data, start=start, end=end)

            if self.debug:
                chunk_path = (
                    self._debug_dir / f"direct_chunk_{chunk_counter[0]:04d}.wav"
                )
                self._write_debug_chunk(data, chunk_path)
                chunk_counter[0] += 1

            try:
                self.chunk_queue.put_nowait(chunk)
            except queue.Full:
                pass

        def forward_frames():
            frame_duration = self.audio_stream.frame_duration
            frames_per_chunk = max(1, int(round(1.0 / frame_duration)))
            audio_buffer: list[bytes] = []
            chunk_start: float | None = None

            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    if audio_buffer and self.stop_event.is_set():
                        _emit_chunk(audio_buffer, chunk_start, frame_duration)
                        audio_buffer.clear()
                        chunk_start = None
                    continue

                if chunk_start is None:
                    chunk_start = frame.timestamp
                audio_buffer.append(frame.data)

                if len(audio_buffer) >= frames_per_chunk:
                    _emit_chunk(audio_buffer, chunk_start, frame_duration)
                    audio_buffer.clear()
                    chunk_start = None

            if audio_buffer:
                _emit_chunk(audio_buffer, chunk_start, frame_duration)

        threading.Thread(target=forward_frames, daemon=True).start()

    def _publish_event(self, event: DashboardEvent):
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass

    @property
    def _debug_dir(self) -> Path:
        debug_dir = self.settings.directories.notes_root / ".debug-live"
        if self.debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    def _write_live_meta(self, note_dir: Path, title: str) -> None:
        from datetime import datetime

        mic = "default"
        if self.audio_stream is not None:
            mic = self.audio_stream.mic_device_name or "default"

        meta = {
            "title": title,
            "date": datetime.now().isoformat(),
            "mic": mic,
            "tags": list(self.tags),
        }
        meta_path = note_dir / META_FILENAME
        with meta_path.open("wb") as fh:
            tomli_w.dump(meta, fh)

    def _write_debug_chunk(self, data: bytes, path: Path):
        if not data or not self.audio_stream:
            return
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(self.audio_stream.sample_rate)
            fh.writeframes(data)
