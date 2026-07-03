"""Utilities for preparing manual notes for the CLI editor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config.settings import ChirpSettings
from utils.file_utils import (
    META_FILENAME,
    NOTES_FILENAME,
    atomic_write_toml,
    ensure_directory,
    slugify,
)

DEFAULT_NOTE_PREFIX = "note"


@dataclass
class NoteContext:
    path: Path
    title: str
    content: str
    is_new: bool


class ManualNoteManager:
    def __init__(self, settings: ChirpSettings):
        self._settings = settings

    def prepare_note(
        self, name: str | None = None, now: datetime | None = None
    ) -> NoteContext:
        current_time = now or datetime.now()

        title = self._resolve_title(name, current_time)
        notes_root = self._settings.directories.notes_root
        ensure_directory(notes_root)

        note_dir = self._resolve_note_dir(notes_root, title, current_time)
        ensure_directory(note_dir)

        note_path = note_dir / NOTES_FILENAME
        meta_path = note_dir / META_FILENAME

        if note_path.exists():
            content = self._prepare_existing_content(note_path, title, current_time)
            is_new = False
        else:
            content = self._prepare_new_content(title, current_time)
            is_new = True
            self._write_initial_meta(meta_path, title, current_time)

        return NoteContext(path=note_path, title=title, content=content, is_new=is_new)

    def _resolve_title(self, provided: str | None, now: datetime) -> str:
        if provided and provided.strip():
            return provided.strip()
        return DEFAULT_NOTE_PREFIX

    def _resolve_note_dir(self, notes_root: Path, title: str, now: datetime) -> Path:
        existing_slug = self._find_existing_slug(notes_root, title)
        if existing_slug is not None:
            return notes_root / existing_slug
        slug = slugify(title, now.date(), notes_root)
        return notes_root / slug

    def _find_existing_slug(self, notes_root: Path, title: str) -> str | None:
        if not notes_root.exists():
            return None
        for candidate in notes_root.iterdir():
            if not candidate.is_dir():
                continue
            meta_path = candidate / META_FILENAME
            if not meta_path.exists():
                continue
            try:
                import tomllib

                with meta_path.open("rb") as fh:
                    meta = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                continue
            if meta.get("title") == title:
                return candidate.name
        return None

    def _prepare_new_content(self, title: str, now: datetime) -> str:
        timestamp = now.strftime("%Y-%m-%d %H:%M")
        return f"# {title}\n\n{timestamp}\n\n"

    def _prepare_existing_content(self, path: Path, title: str, now: datetime) -> str:
        existing = path.read_text(encoding="utf-8")
        header, remainder = self._split_header(existing, title)

        timestamp = now.strftime("%Y-%m-%d %H:%M")
        remainder = remainder.lstrip("\n")

        if remainder:
            return f"{header}\n\n{timestamp}\n\n{remainder}"
        return f"{header}\n\n{timestamp}\n\n"

    def _split_header(self, content: str, fallback_title: str) -> tuple[str, str]:
        if not content:
            return f"# {fallback_title}", ""

        first_newline = content.find("\n")
        if first_newline == -1:
            header_line = content.strip()
            remainder = ""
        else:
            header_line = content[:first_newline].strip()
            remainder = content[first_newline + 1 :]

        if not header_line.startswith("# "):
            header_line = f"# {fallback_title}"

        return header_line, remainder

    def _write_initial_meta(self, meta_path: Path, title: str, now: datetime) -> None:
        meta = {
            "title": title,
            "date": now.isoformat(),
            "tags": [],
            "source": "manual",
        }
        atomic_write_toml(meta_path, meta)
