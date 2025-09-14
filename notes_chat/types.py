from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class NoteMeta(BaseModel):
    path: Path
    title: str
    date: datetime
    participants: list[str]
    duration: int
    mtime: float
    size: int


class Chunk(BaseModel):
    id: str
    path: Path
    content: str
    meta: NoteMeta
    first_96_chars: str


class Diagnostics(BaseModel):
    retrieved_chunks: int
    context_chars: int
    cache_hit: bool
    processing_time_ms: float


class TimeRange(BaseModel):
    start: datetime
    end_exclusive: datetime
