from __future__ import annotations

import collections
import queue
import threading
from collections.abc import Callable

from recorder.live_types import AudioFrame, SpeechChunk

SILERO_FRAME_SAMPLES_16K = 512
SILERO_FRAME_MS = 32


class VADChunker(threading.Thread):
    def __init__(
        self,
        frame_queue: queue.Queue[AudioFrame],
        chunk_queue: queue.Queue[SpeechChunk],
        stop_event: threading.Event,
        sample_rate: int,
        frame_ms: int = SILERO_FRAME_MS,
        padding_ms: int = 300,
        speech_threshold: float = 0.5,
        vad_factory: Callable | None = None,
        energy_threshold: float = 0.01,
        event_queue: queue.Queue | None = None,
        max_chunk_seconds: float = 10.0,
        status_interval: int = 50,
        poll_timeout: float = 0.1,
    ):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.chunk_queue = chunk_queue
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.padding_ms = padding_ms
        self.speech_threshold = speech_threshold
        if vad_factory:
            self.vad = vad_factory()
        else:
            self.vad = _load_silero_model()
        self.energy_threshold = energy_threshold
        self.event_queue = event_queue
        self.max_chunk_seconds = max_chunk_seconds
        self.status_interval = status_interval
        self.poll_timeout = poll_timeout

        self._padding_frames = max(1, int(self.padding_ms / self.frame_ms))
        self._ring_buffer: collections.deque[tuple[AudioFrame, bool]] = (
            collections.deque(maxlen=self._padding_frames)
        )
        self._triggered = False
        self._voiced_frames: list[AudioFrame] = []
        self._frame_count = 0
        self._speech_frame_count = 0
        self._chunk_count = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=self.poll_timeout)
            except queue.Empty:
                continue

            self._frame_count += 1
            is_speech = self._is_speech(frame)
            if is_speech:
                self._speech_frame_count += 1

            self._update_state(frame, is_speech)

            if self._frame_count % self.status_interval == 0:
                self._publish_event(
                    "vad_status",
                    {
                        "frames": self._frame_count,
                        "speech_frames": self._speech_frame_count,
                        "triggered": self._triggered,
                        "chunks_emitted": self._chunk_count,
                    },
                )

        if self._triggered and self._voiced_frames:
            self._emit_chunk()

    def _is_speech(self, frame: AudioFrame) -> bool:
        if frame.level < self.energy_threshold:
            return False

        try:
            result = self.vad(frame.data, self.sample_rate)
            return float(result) >= self.speech_threshold
        except Exception:
            return False

    def _update_state(self, frame: AudioFrame, is_speech: bool):
        if not self._triggered:
            self._ring_buffer.append((frame, is_speech))
            num_voiced = len([1 for _, speech in self._ring_buffer if speech])
            if num_voiced > 0.5 * self._padding_frames:
                self._triggered = True
                self._voiced_frames.extend(f for f, _ in self._ring_buffer)
                self._ring_buffer.clear()
        else:
            self._voiced_frames.append(frame)
            if is_speech:
                self._ring_buffer.clear()
            else:
                self._ring_buffer.append((frame, is_speech))

            if self._voiced_frames:
                chunk_duration = (
                    self._voiced_frames[-1].timestamp
                    + self._voiced_frames[-1].duration
                    - self._voiced_frames[0].timestamp
                )
                if chunk_duration >= self.max_chunk_seconds:
                    self._emit_chunk()
                    self._triggered = False
                    self._ring_buffer.clear()
                    self._voiced_frames.clear()
                    return

            num_unvoiced = len([1 for _, speech in self._ring_buffer if not speech])
            if num_unvoiced > 0.9 * self._padding_frames:
                self._emit_chunk()
                self._triggered = False
                self._ring_buffer.clear()
                self._voiced_frames.clear()

    def _emit_chunk(self, tail_frames: list[AudioFrame] | None = None):
        if not self._voiced_frames:
            return
        frames = list(self._voiced_frames)
        if tail_frames:
            frames.extend(tail_frames)
        data = b"".join(frame.data for frame in frames)
        start = frames[0].timestamp
        end = frames[-1].timestamp + frames[-1].duration
        chunk = SpeechChunk(data=data, start=start, end=end)
        try:
            self.chunk_queue.put_nowait(chunk)
            self._chunk_count += 1
            self._publish_event(
                "chunk_emitted",
                {
                    "chunk_id": self._chunk_count,
                    "duration": end - start,
                    "frames": len(frames),
                },
            )
        except queue.Full:
            pass

    def _publish_event(self, event_type: str, payload: dict):
        if self.event_queue is None:
            return
        from recorder.live_types import DashboardEvent

        event = DashboardEvent(type=event_type, payload=payload)
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass


class _SileroVADWrapper:
    def __init__(self):
        import torch
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._torch = torch

    def __call__(self, pcm_bytes: bytes, sample_rate: int) -> float:
        import numpy as np

        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(pcm)
        return float(self._model(tensor, sample_rate))


def _load_silero_model():
    try:
        return _SileroVADWrapper()
    except ImportError:
        raise ImportError(
            "Live transcription requires silero-vad. "
            "Install it with: pip install 'chirp-notes-ai[live]'"
        )
