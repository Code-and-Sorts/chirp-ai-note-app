import wave

import numpy as np
import pytest

from transcriber.audio_loader import (
    TARGET_SAMPLE_RATE,
    UnsupportedAudioError,
    load_audio,
)


def _write_wav(path, samples_int16, *, channels=1, rate=TARGET_SAMPLE_RATE, width=2):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(samples_int16.astype("<i2").tobytes())


def test_loads_16k_mono_pcm_as_normalized_float32(tmp_path):
    path = tmp_path / "mono.wav"
    samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    _write_wav(path, samples)

    audio = load_audio(path)

    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, samples.astype(np.float32) / 32768.0, atol=1e-7)


def test_downmixes_stereo_to_mono(tmp_path):
    path = tmp_path / "stereo.wav"
    interleaved = np.array([1000, 3000, -2000, 2000], dtype=np.int16)
    _write_wav(path, interleaved, channels=2)

    audio = load_audio(path)

    expected = np.array([2000.0, 0.0], dtype=np.float32) / 32768.0
    assert len(audio) == 2
    np.testing.assert_allclose(audio, expected, atol=1e-7)


def test_resamples_to_target_rate(tmp_path):
    path = tmp_path / "8k.wav"
    samples = np.zeros(800, dtype=np.int16)
    _write_wav(path, samples, rate=8000)

    audio = load_audio(path)

    assert len(audio) == 1600


def test_eight_bit_pcm_is_centered(tmp_path):
    path = tmp_path / "8bit.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(bytes([128, 255, 0]))

    audio = load_audio(path)

    np.testing.assert_allclose(audio, [0.0, 127 / 128, -1.0], atol=1e-7)


def test_rejects_non_wav(tmp_path):
    path = tmp_path / "fake.mp3"
    path.write_bytes(b"ID3\x04\x00not really audio")

    with pytest.raises(UnsupportedAudioError):
        load_audio(path)


def test_rejects_unsupported_sample_width(tmp_path):
    path = tmp_path / "24bit.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(3)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(b"\x00\x00\x00\x01\x00\x00")

    with pytest.raises(UnsupportedAudioError):
        load_audio(path)
