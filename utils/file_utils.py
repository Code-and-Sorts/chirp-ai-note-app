from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tomllib
import unicodedata
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import tomli_w

AUDIO_FILENAME = "audio.wav"
TRANSCRIPT_FILENAME = "transcript.txt"
NOTES_FILENAME = "notes.md"
META_FILENAME = "meta.toml"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    try:
        existing_mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        existing_mode = None
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            tmp_path.chmod(existing_mode)
        tmp_path.replace(path)
    except OSError:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, content.encode(encoding))


def atomic_write_toml(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(path, tomli_w.dumps(payload).encode("utf-8"))


def merge_note_meta(note_dir: Path, updates: dict[str, Any]) -> None:
    """Merge ``updates`` into a note's meta.toml, starting fresh if corrupt."""
    meta_path = note_dir / META_FILENAME
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with meta_path.open("rb") as fh:
                meta = dict(tomllib.load(fh))
        except (OSError, tomllib.TOMLDecodeError):
            meta = {}
    meta.update(updates)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_toml(meta_path, meta)


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=indent).encode("utf-8"))


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
    template: str | None = None


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
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_folded = decomposed.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_folded.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
    return cleaned.strip("-")


def list_notes(notes_root: Path) -> list[NoteRecord]:
    if not notes_root.exists():
        return []

    try:
        entries = list(notes_root.iterdir())
    except (PermissionError, NotADirectoryError, OSError):
        return []

    records: list[NoteRecord] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        record = _build_record(entry)
        if record is not None:
            records.append(record)

    records.sort(key=lambda rec: (rec.created_at, rec.slug))
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
        except (tomllib.TOMLDecodeError, OSError):
            meta_data = {}

    created_at = _resolve_created_at(meta_data, entry)
    tags = meta_data.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    title = meta_data.get("title")
    if not isinstance(title, str):
        title = None

    template = meta_data.get("template")
    if not isinstance(template, str):
        template = None

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
        template=template,
    )


def _resolve_created_at(meta: dict, entry: Path) -> datetime:
    date_value = meta.get("date")
    if isinstance(date_value, datetime):
        return _as_naive(date_value)
    if isinstance(date_value, date):
        return datetime(date_value.year, date_value.month, date_value.day)
    if isinstance(date_value, str):
        try:
            return _as_naive(datetime.fromisoformat(date_value))
        except ValueError:
            pass

    try:
        return datetime.fromtimestamp(entry.stat().st_mtime)
    except OSError:
        return datetime.now()


def _as_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


def move_file(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    except OSError:
        return False


def copy_file(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return True
    except OSError:
        return False


def get_file_size_mb(file_path: Path) -> float:
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def ensure_directory(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def ensure_private_directory(path: Path, *, tighten_existing: bool = False) -> None:
    existed = path.is_dir()
    path.mkdir(parents=True, exist_ok=True)
    if not existed or tighten_existing:
        with suppress(OSError):
            path.chmod(0o700)


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
            except OSError:
                # One file failing should not stop the bulk delete; counter just doesn't increment.
                pass

    return deleted_count
