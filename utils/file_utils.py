from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import tomllib

AUDIO_FILENAME = "audio.wav"
TRANSCRIPT_FILENAME = "transcript.txt"
NOTES_FILENAME = "notes.md"
META_FILENAME = "meta.toml"


@dataclass
class NoteRecord:
    slug: str
    dir: Path
    audio: Path | None
    transcript: Path | None
    notes: Path | None
    meta: Path | None
    created_at: datetime
    tags: list[str] = field(default_factory=list)
    title: str | None = None


def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join(c for c in filename if c not in invalid_chars)
    return sanitized.strip()[:50]


def slugify(title: str, note_date: date, notes_root: Path | None = None) -> str:
    base = _kebab_case(title)
    if not base:
        base = "note"
    candidate = f"{base}-{note_date.strftime('%Y-%m-%d')}"

    if notes_root is None or not notes_root.exists():
        return candidate

    if not (notes_root / candidate).exists():
        return candidate

    suffix = 2
    while (notes_root / f"{candidate}-{suffix}").exists():
        suffix += 1
    return f"{candidate}-{suffix}"


def _kebab_case(value: str) -> str:
    lowered = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
    return cleaned.strip("-")


def list_notes(notes_root: Path) -> list[NoteRecord]:
    if not notes_root.exists():
        return []

    records: list[NoteRecord] = []
    for entry in notes_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        record = _build_record(entry)
        if record is not None:
            records.append(record)

    records.sort(key=lambda rec: rec.created_at)
    return records


def _build_record(entry: Path) -> NoteRecord | None:
    audio_path = entry / AUDIO_FILENAME
    transcript_path = entry / TRANSCRIPT_FILENAME
    notes_path = entry / NOTES_FILENAME
    meta_path = entry / META_FILENAME

    meta_data: dict = {}
    if meta_path.exists():
        try:
            with meta_path.open("rb") as fh:
                meta_data = tomllib.load(fh)
        except Exception:
            meta_data = {}

    created_at = _resolve_created_at(meta_data, entry)
    tags = meta_data.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    title = meta_data.get("title")
    if not isinstance(title, str):
        title = None

    return NoteRecord(
        slug=entry.name,
        dir=entry,
        audio=audio_path if audio_path.exists() else None,
        transcript=transcript_path if transcript_path.exists() else None,
        notes=notes_path if notes_path.exists() else None,
        meta=meta_path if meta_path.exists() else None,
        created_at=created_at,
        tags=list(tags),
        title=title,
    )


def _resolve_created_at(meta: dict, entry: Path) -> datetime:
    date_value = meta.get("date")
    if isinstance(date_value, datetime):
        return date_value
    if isinstance(date_value, date):
        return datetime(date_value.year, date_value.month, date_value.day)
    if isinstance(date_value, str):
        try:
            return datetime.fromisoformat(date_value)
        except ValueError:
            pass

    try:
        return datetime.fromtimestamp(entry.stat().st_mtime)
    except OSError:
        return datetime.now()


def move_file(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    except Exception:
        return False


def copy_file(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return True
    except Exception:
        return False


def get_file_size_mb(file_path: Path) -> float:
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def ensure_directory(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def clean_old_files(directory: Path, days_old: int = 30) -> int:
    if not directory.exists():
        return 0

    cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    deleted_count = 0

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception:
                pass

    return deleted_count
