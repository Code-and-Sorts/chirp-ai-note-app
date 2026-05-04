"""Pair-aligning mixer for the dual-source `audio_capture` stream.

`StereoToMonoMixer` re-windows arbitrarily-sized incoming chunks (system
and microphone, each tagged float32 PCM at 16 kHz) into fixed-size mono
output frames. A frame is emitted when both sources have at least
`frame_samples` of buffered data, or when one source has stalled — its
`last_seen_end` timestamp lags the other source's currently-buffered
extent by more than `gap_ms` — in which case the stalled source is
silence-padded.

The mixer is pure: no threads, no I/O, no wall-clock. Stall detection
is driven entirely by input timestamps from the helper, so unit tests
can deterministically reproduce paired, re-windowed, and gap-padded
output by feeding numpy arrays.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

SOURCE_SYSTEM = 1
SOURCE_MICROPHONE = 2

_SOURCES: tuple[int, int] = (SOURCE_SYSTEM, SOURCE_MICROPHONE)


class StereoToMonoMixer:
    def __init__(
        self,
        frame_ms: int = 32,
        sample_rate: int = 16000,
        gap_ms: int = 100,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if gap_ms < 0:
            raise ValueError("gap_ms must be non-negative")

        self._frame_samples = sample_rate * frame_ms // 1000
        if self._frame_samples <= 0:
            raise ValueError("frame_ms × sample_rate yields zero samples per frame")

        self._frame_us = frame_ms * 1000
        self._gap_us = gap_ms * 1000
        self._sample_rate = sample_rate
        self._max_buffer_samples = self._frame_samples * 8

        self._buffers: dict[int, np.ndarray] = {
            s: np.zeros(0, dtype=np.float32) for s in _SOURCES
        }
        self._head_ts: dict[int, int | None] = dict.fromkeys(_SOURCES)
        self._last_seen_end: dict[int, int | None] = dict.fromkeys(_SOURCES)

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def feed(self, source: int, timestamp_us: int, samples: np.ndarray) -> None:
        if source not in _SOURCES:
            return
        if samples.size == 0:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)

        if self._head_ts[source] is None:
            self._head_ts[source] = timestamp_us

        self._buffers[source] = np.concatenate([self._buffers[source], samples])

        excess = self._buffers[source].size - self._max_buffer_samples
        if excess > 0:
            self._buffers[source] = self._buffers[source][excess:]
            head = self._head_ts[source]
            if head is not None:
                self._head_ts[source] = head + (excess * 1_000_000) // self._sample_rate

        chunk_duration_us = (samples.size * 1_000_000) // self._sample_rate
        self._last_seen_end[source] = timestamp_us + chunk_duration_us

    def drain(self) -> Iterator[tuple[int, np.ndarray]]:
        while True:
            sys_ready = self._buffers[SOURCE_SYSTEM].size >= self._frame_samples
            mic_ready = self._buffers[SOURCE_MICROPHONE].size >= self._frame_samples

            if sys_ready and mic_ready:
                head = max(
                    self._head_ts[SOURCE_SYSTEM] or 0,
                    self._head_ts[SOURCE_MICROPHONE] or 0,
                )
                sys_frame = self._take_frame(SOURCE_SYSTEM)
                mic_frame = self._take_frame(SOURCE_MICROPHONE)
                yield head, _mix(sys_frame, mic_frame)
                continue

            if sys_ready and self._is_stalled(SOURCE_MICROPHONE, SOURCE_SYSTEM):
                stalled_head = self._head_ts[SOURCE_SYSTEM]
                assert stalled_head is not None
                sys_frame = self._take_frame(SOURCE_SYSTEM)
                yield stalled_head, _mix(sys_frame, _silence(self._frame_samples))
                continue

            if mic_ready and self._is_stalled(SOURCE_SYSTEM, SOURCE_MICROPHONE):
                stalled_head = self._head_ts[SOURCE_MICROPHONE]
                assert stalled_head is not None
                mic_frame = self._take_frame(SOURCE_MICROPHONE)
                yield stalled_head, _mix(_silence(self._frame_samples), mic_frame)
                continue

            return

    def _take_frame(self, source: int) -> np.ndarray:
        frame = self._buffers[source][: self._frame_samples].copy()
        self._buffers[source] = self._buffers[source][self._frame_samples :]
        if self._buffers[source].size == 0:
            self._head_ts[source] = None
        else:
            head = self._head_ts[source]
            if head is not None:
                self._head_ts[source] = head + self._frame_us
        return frame

    def _is_stalled(self, lagging: int, leading: int) -> bool:
        # A source with at least one full frame already buffered is not
        # stalled — drain() will pair it normally on the next iteration.
        # A *partial* buffer (0 < size < frame_samples) still counts as
        # stalled when the timestamp gap criterion fires; otherwise the
        # leading source's audio would be discarded forever waiting for
        # the lagging side's partial to fill (which it may never do).
        if self._buffers[lagging].size >= self._frame_samples:
            return False
        leading_head = self._head_ts[leading]
        if leading_head is None:
            return False
        leading_buffer_us = (
            self._buffers[leading].size * 1_000_000
        ) // self._sample_rate
        leading_end = leading_head + leading_buffer_us
        last_end_lag = self._last_seen_end[lagging] or 0
        return bool(leading_end - last_end_lag > self._gap_us)


def _silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mixed: np.ndarray = np.clip(a + b, -1.0, 1.0).astype(np.float32, copy=False)
    return mixed


__all__ = ["SOURCE_MICROPHONE", "SOURCE_SYSTEM", "StereoToMonoMixer"]
