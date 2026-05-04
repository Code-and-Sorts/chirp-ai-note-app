from __future__ import annotations

import numpy as np
import pytest

from recorder.audio_mixer import (
    SOURCE_MICROPHONE,
    SOURCE_SYSTEM,
    StereoToMonoMixer,
)


def _const(n: int, value: float) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


class TestPairedAlignment:
    def test_emits_single_frame_when_both_sources_provide_one_frame(self):
        mixer = StereoToMonoMixer()
        mixer.feed(SOURCE_SYSTEM, 1_000, _const(512, 0.10))
        mixer.feed(SOURCE_MICROPHONE, 1_000, _const(512, 0.20))

        output = list(mixer.drain())

        assert len(output) == 1
        ts, mixed = output[0]
        assert ts == 1_000
        assert mixed.dtype == np.float32
        assert mixed.size == 512
        np.testing.assert_allclose(mixed, 0.30, atol=1e-6)

    def test_rewindows_oversized_chunks_into_multiple_frames(self):
        mixer = StereoToMonoMixer()
        mixer.feed(SOURCE_SYSTEM, 0, _const(1024, 0.10))
        mixer.feed(SOURCE_MICROPHONE, 0, _const(1024, 0.10))

        output = list(mixer.drain())

        assert len(output) == 2
        timestamps = [ts for ts, _ in output]
        assert timestamps == [0, 32_000]
        for _, mixed in output:
            assert mixed.size == 512
            np.testing.assert_allclose(mixed, 0.20, atol=1e-6)

    def test_clipping_caps_overload(self):
        mixer = StereoToMonoMixer()
        mixer.feed(SOURCE_SYSTEM, 0, _const(512, 0.9))
        mixer.feed(SOURCE_MICROPHONE, 0, _const(512, 0.9))

        output = list(mixer.drain())

        assert len(output) == 1
        _, mixed = output[0]
        assert float(mixed.max()) == pytest.approx(1.0)
        assert float(mixed.min()) == pytest.approx(1.0)

    def test_drain_blocks_until_both_sources_have_one_frame(self):
        mixer = StereoToMonoMixer()
        mixer.feed(SOURCE_SYSTEM, 0, _const(512, 0.5))

        assert list(mixer.drain()) == []

        mixer.feed(SOURCE_MICROPHONE, 0, _const(512, 0.5))
        assert len(list(mixer.drain())) == 1


class TestSilencePadding:
    def test_pads_when_lagging_source_falls_behind_more_than_gap(self):
        mixer = StereoToMonoMixer(gap_ms=100)
        # Both seed at ts=0 — establishes last_seen_end for both sources.
        mixer.feed(SOURCE_SYSTEM, 0, _const(512, 0.4))
        mixer.feed(SOURCE_MICROPHONE, 0, _const(512, 0.4))
        assert len(list(mixer.drain())) == 1

        # Now mic forges ahead by 4 frames (>100ms) without sys.
        for i in range(4):
            ts = (i + 1) * 32_000
            mixer.feed(SOURCE_MICROPHONE, ts, _const(512, 0.4))

        output = list(mixer.drain())

        assert len(output) >= 1
        # First emit happens once leading_end - last_end_sys > 100_000us.
        first_ts, first_mixed = output[0]
        np.testing.assert_allclose(first_mixed, 0.4, atol=1e-6)
        assert first_ts >= 32_000

    def test_does_not_pad_when_within_gap(self):
        mixer = StereoToMonoMixer(gap_ms=100)
        mixer.feed(SOURCE_MICROPHONE, 0, _const(512, 0.5))

        assert list(mixer.drain()) == []

    def test_silence_pads_on_stalled_system_when_only_mic_running(self):
        mixer = StereoToMonoMixer(gap_ms=100, frame_ms=32, sample_rate=16000)
        # Mic feeds 6 frames consecutively; sys never feeds.
        for i in range(6):
            ts = i * 32_000
            mixer.feed(SOURCE_MICROPHONE, ts, _const(512, 0.5))

        output = list(mixer.drain())

        assert len(output) >= 1
        for _, mixed in output:
            np.testing.assert_allclose(mixed, 0.5, atol=1e-6)


class TestBufferCap:
    def test_caps_buffer_to_eight_frames(self):
        mixer = StereoToMonoMixer()
        # Feed 16 frames worth into one source without draining.
        for i in range(16):
            mixer.feed(SOURCE_SYSTEM, i * 32_000, _const(512, 0.1))

        # Buffer cap is 8 frames × 512 = 4096 samples.
        assert mixer._buffers[SOURCE_SYSTEM].size == 8 * 512


class TestValidation:
    def test_zero_frame_ms_raises(self):
        with pytest.raises(ValueError):
            StereoToMonoMixer(frame_ms=0)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError):
            StereoToMonoMixer(gap_ms=-1)

    def test_zero_sample_rate_raises(self):
        with pytest.raises(ValueError):
            StereoToMonoMixer(sample_rate=0)


class TestFeedRobustness:
    def test_unknown_source_is_ignored(self):
        mixer = StereoToMonoMixer()
        mixer.feed(99, 0, _const(512, 0.5))
        mixer.feed(SOURCE_SYSTEM, 0, _const(512, 0.5))
        mixer.feed(SOURCE_MICROPHONE, 0, _const(512, 0.5))
        assert len(list(mixer.drain())) == 1

    def test_empty_samples_are_ignored(self):
        mixer = StereoToMonoMixer()
        mixer.feed(SOURCE_SYSTEM, 0, np.zeros(0, dtype=np.float32))
        assert mixer._buffers[SOURCE_SYSTEM].size == 0

    def test_non_float32_samples_are_converted(self):
        mixer = StereoToMonoMixer()
        mixer.feed(SOURCE_SYSTEM, 0, np.full(512, 0.25, dtype=np.float64))
        mixer.feed(SOURCE_MICROPHONE, 0, np.full(512, 0.25, dtype=np.float64))
        output = list(mixer.drain())
        assert len(output) == 1
        _, mixed = output[0]
        assert mixed.dtype == np.float32
        np.testing.assert_allclose(mixed, 0.5, atol=1e-6)
