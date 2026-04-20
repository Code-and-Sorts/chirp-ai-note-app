from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AudioFrame:
    data: bytes
    timestamp: float
    duration: float
    level: float


@dataclass(slots=True)
class SpeechChunk:
    data: bytes
    start: float
    end: float


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: int


@dataclass(slots=True)
class DashboardEvent:
    type: str
    payload: dict
