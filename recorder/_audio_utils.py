"""Shared helpers for the recorder paths."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config.settings import ChirpSettings


def warn_if_audio_settings_overridden(
    settings: ChirpSettings,
    *,
    logger: logging.Logger,
    component: str,
    output_sample_rate: int,
    output_channels: int,
) -> None:
    sample_rate = settings.audio.sample_rate
    channels = settings.audio.channels
    if sample_rate == output_sample_rate and channels == output_channels:
        return
    logger.warning(
        "%s: settings.audio.sample_rate=%s channels=%s differ from output "
        "(%s Hz, %s channel%s); produced audio will be %s Hz mono regardless",
        component,
        sample_rate,
        channels,
        output_sample_rate,
        output_channels,
        "" if output_channels == 1 else "s",
        output_sample_rate,
    )


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)
    int16: np.ndarray = (audio * 32767.0).astype(np.int16, copy=False)
    result: bytes = int16.tobytes()
    return result


__all__ = ["float32_to_int16_bytes", "warn_if_audio_settings_overridden"]
