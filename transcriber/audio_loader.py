"""Decode WAV audio to whisper's input format without the ffmpeg CLI.

mlx-whisper's own ``load_audio`` shells out to the ``ffmpeg`` binary even for a
plain PCM WAV. Chirp records 16 kHz mono 16-bit PCM — already whisper's target
format — so the common path is a direct ``wave`` read with no resampling. Stereo
or off-rate WAVs (e.g. imported via ``transcribe --regen``) are downmixed and
resampled with scipy. Non-WAV containers are rejected with a clear message
rather than dragging the whole codec surface back in.
"""

from __future__ import annotations

import wave
from math import gcd
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16000


class UnsupportedAudioError(ValueError):
    pass


def load_audio(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise UnsupportedAudioError(
            f"{path} is not a readable PCM WAV file ({exc}). Convert it to WAV "
            "(16 kHz mono is ideal) before transcribing."
        ) from exc

    samples = _pcm_to_float32(frames, sample_width)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    if frame_rate != TARGET_SAMPLE_RATE:
        samples = _resample(samples, frame_rate, TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _pcm_to_float32(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        return data / 32768.0
    if sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
        return data / 2147483648.0
    raise UnsupportedAudioError(
        f"unsupported WAV sample width: {sample_width * 8}-bit. "
        "Convert to 16-bit PCM WAV before transcribing."
    )


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    divisor = gcd(src_rate, dst_rate)
    resampled = resample_poly(samples, dst_rate // divisor, src_rate // divisor)
    return np.asarray(resampled, dtype=np.float32)
