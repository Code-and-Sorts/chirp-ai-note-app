from __future__ import annotations

import logging
import queue
import tempfile
import threading
import wave
from pathlib import Path

from config.settings import ChirpSettings
from notes.constants import DEFAULT_MEETING_NAME
from recorder.live_types import DashboardEvent, SpeechChunk, TranscriptSegment
from transcriber.whisper_transcriber import WhisperTranscriber
from utils.file_utils import atomic_write_text

logger = logging.getLogger(__name__)


class LiveTranscriber(threading.Thread):
    def __init__(
        self,
        settings: ChirpSettings,
        chunk_queue: queue.Queue[SpeechChunk],
        event_queue: queue.Queue[DashboardEvent],
        stop_event: threading.Event,
        meeting_name: str | None = None,
        sample_rate: int = 16000,
        debug_dir: Path | None = None,
        transcription_interval: float = 3.0,
        overlap_threshold: float = 0.3,
        poll_timeout: float = 0.1,
    ):
        super().__init__(daemon=True)
        self.settings = settings
        self.chunk_queue = chunk_queue
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.meeting_name = meeting_name or DEFAULT_MEETING_NAME
        self.sample_rate = sample_rate
        self.debug_dir = debug_dir
        self._debug_index = 0
        self._processed_chunks = 0
        self.transcription_interval = transcription_interval
        self.overlap_threshold = overlap_threshold
        self.poll_timeout = poll_timeout

        self._pcm_buffer = bytearray()
        self._buffer_offset_seconds = 0.0
        self._last_chunk_end = 0.0
        self._last_transcribe_at = 0.0
        self._last_emitted_end = 0.0

        self.transcriber = WhisperTranscriber(settings)
        self._last_emit_time = 0.0
        self._segments: list[TranscriptSegment] = []
        self._language: str | None = None
        self._total_words = 0
        self._lock = threading.Lock()

    @property
    def segments(self) -> list[TranscriptSegment]:
        with self._lock:
            return list(self._segments)

    @property
    def language(self) -> str | None:
        with self._lock:
            return self._language

    @property
    def total_words(self) -> int:
        with self._lock:
            return self._total_words

    def run(self):
        while True:
            try:
                chunk = self.chunk_queue.get(timeout=self.poll_timeout)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                self._maybe_transcribe(force=False)
                continue

            try:
                self._process_chunk(chunk)
            except Exception as exc:  # noqa: BLE001 - process_chunk can raise any type; publish and continue
                self._publish_event(
                    "error",
                    {
                        "message": f"Live transcriber error: {exc}",
                    },
                )
                continue

        self._maybe_transcribe(force=True)

    def _process_chunk(self, chunk: SpeechChunk):
        self._publish_event("chunk", {"duration": chunk.end - chunk.start})
        self._pcm_buffer.extend(chunk.data)
        self._last_chunk_end = max(self._last_chunk_end, chunk.end)

        self._maybe_transcribe(force=False)

    def _publish_event(self, event_type: str, payload: dict):
        event = DashboardEvent(type=event_type, payload=payload)
        try:
            self.event_queue.put_nowait(event)
        except queue.Full as exc:
            logger.debug(
                "dropped dashboard %s event; event queue full: %s", event_type, exc
            )

    def _maybe_transcribe(self, force: bool):
        if not self._pcm_buffer:
            return

        if (
            not force
            and self.transcription_interval > 0
            and (
                self._last_chunk_end - self._last_transcribe_at
                < self.transcription_interval
            )
        ):
            return

        pcm_bytes = bytes(self._pcm_buffer)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)
            with wave.open(tmp, "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(self.sample_rate)
                fh.writeframes(pcm_bytes)

        try:
            result = self.transcriber.transcribe_file(
                temp_path,
                fast_mode=True,
                language=self._language,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

        metadata = result.get("metadata", {})

        segments = result.get("segments", [])
        new_segments: list[TranscriptSegment] = []

        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))

            absolute_start = self._buffer_offset_seconds + start
            absolute_end = self._buffer_offset_seconds + end

            segment_duration = absolute_end - absolute_start
            overlap_amount = max(0, self._last_emitted_end - absolute_start)

            if overlap_amount > self.overlap_threshold * segment_duration:
                continue

            if absolute_start < self._last_emitted_end:
                absolute_start = self._last_emitted_end
            if absolute_end <= absolute_start:
                continue

            transcript_segment = TranscriptSegment(
                text=text,
                start=absolute_start,
                end=absolute_end,
                words=len(text.split()),
            )
            new_segments.append(transcript_segment)
            self._last_emitted_end = max(self._last_emitted_end, absolute_end)

        with self._lock:
            if metadata and metadata.get("language") and not self._language:
                self._language = metadata.get("language")
            self._segments.extend(new_segments)
            self._total_words += sum(s.words for s in new_segments)
            self._last_emit_time = max(
                self._last_emit_time,
                max((s.end for s in new_segments), default=self._last_emit_time),
            )

        if new_segments:
            payload = {
                "segments": new_segments,
                "language": self._language,
                "total_words": self._total_words,
            }
            self._publish_event("transcript", payload)

        self._processed_chunks += 1
        self._publish_event(
            "transcriber",
            {
                "processed": self._processed_chunks,
                "new_segments": len(new_segments),
            },
        )

        if segments:
            max_segment_end = max(float(seg.get("end", 0.0)) for seg in segments)
            prune_seconds = max_segment_end
        else:
            prune_seconds = self._last_chunk_end - self._buffer_offset_seconds

        if prune_seconds > 0:
            prune_samples = int(prune_seconds * self.sample_rate)
            prune_bytes = prune_samples * 2
            if prune_bytes > 0 and prune_bytes <= len(self._pcm_buffer):
                self._pcm_buffer = self._pcm_buffer[prune_bytes:]
                self._buffer_offset_seconds += prune_seconds

        self._last_transcribe_at = self._last_chunk_end

        if self.debug_dir is not None:
            self._write_debug_chunk(pcm_bytes, self.sample_rate, segments)

    def _write_debug_chunk(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        segments: list,
    ) -> None:
        if not pcm_bytes or self.debug_dir is None:
            return

        debug_path = self.debug_dir / f"chunk_{self._debug_index:04d}.wav"
        with wave.open(str(debug_path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(sample_rate)
            fh.writeframes(pcm_bytes)

        transcript_path = self.debug_dir / f"chunk_{self._debug_index:04d}.txt"
        summary_path = self.debug_dir / f"chunk_{self._debug_index:04d}_summary.txt"
        with transcript_path.open("w", encoding="utf-8") as txt:
            if segments:
                for segment in segments:
                    txt.write(
                        f"{float(segment.get('start', 0.0)):.2f}-{float(segment.get('end', 0.0)):.2f}: {segment.get('text', '').strip()}\n"
                    )
            else:
                txt.write("(no segments)\n")

        with summary_path.open("w", encoding="utf-8") as summary:
            if segments:
                summary.write(
                    "\n".join(
                        segment.get("text", "").strip()
                        for segment in segments
                        if segment.get("text", "").strip()
                    )
                )
            else:
                summary.write("(no text)\n")

        self._debug_index += 1

    def export_transcript(self, output_path: Path):
        segments = self.segments
        if not segments:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for segment in segments:
            timestamp = self._format_timestamp(segment.start)
            lines.append(f"[{timestamp}] {segment.text}")
        atomic_write_text(output_path, "\n".join(lines))

    def close(self) -> None:
        """Release the underlying Whisper model. Idempotent."""
        transcriber = getattr(self, "transcriber", None)
        if transcriber is not None:
            transcriber.close()

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
