"""Utilities for preparing manual notes for the CLI editor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config.settings import ChirpSettings
from utils.file_utils import ensure_directory, sanitize_filename

DEFAULT_NOTE_PREFIX = "note"


@dataclass
class NoteContext:
    """Information required to edit a manual note."""

    path: Path
    title: str
    content: str
    is_new: bool


class ManualNoteManager:
    """Prepare manual notes for editing and handle boilerplate content."""

    def __init__(self, settings: ChirpSettings):
        self._settings = settings

    def prepare_note(
        self, name: str | None = None, now: datetime | None = None
    ) -> NoteContext:
        """Create or update a manual note and return the editing context."""

        current_time = now or datetime.now()

        title = self._resolve_title(name, current_time)
        note_path = self._resolve_path(title)

        ensure_directory(note_path.parent)

        if note_path.exists():
            content = self._prepare_existing_content(note_path, title, current_time)
            is_new = False
        else:
            content = self._prepare_new_content(title, current_time)
            is_new = True

        return NoteContext(path=note_path, title=title, content=content, is_new=is_new)

    def _resolve_title(self, provided: str | None, now: datetime) -> str:
        if provided and provided.strip():
            return provided.strip()

        return f"{DEFAULT_NOTE_PREFIX}-{now.strftime('%Y-%m-%d')}"

    def _resolve_path(self, title: str) -> Path:
        notes_dir = Path(self._settings.directories.notes)

        sanitized = sanitize_filename(title)
        sanitized = sanitized.replace(" ", "-")
        if not sanitized:
            sanitized = (
                f"{DEFAULT_NOTE_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M')}"
            )

        filename = f"{sanitized}.md"
        return notes_dir / filename

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
