from datetime import datetime

import tomllib

from config.settings import ChirpSettings
from notes.manual_note_manager import ManualNoteManager


def _make_settings(tmp_path):
    settings = ChirpSettings()
    settings.directories.notes_root = tmp_path
    return settings


def test_prepare_note_new_creates_folder_and_meta(tmp_path):
    manager = ManualNoteManager(_make_settings(tmp_path))

    now = datetime(2024, 5, 1, 10, 30)
    context = manager.prepare_note("Project Kickoff", now=now)

    note_dir = tmp_path / "project-kickoff-2024-05-01"
    assert context.is_new is True
    assert context.path == note_dir / "notes.md"
    assert context.title == "Project Kickoff"
    assert context.content == "# Project Kickoff\n\n2024-05-01 10:30\n\n"

    with (note_dir / "meta.toml").open("rb") as fh:
        meta = tomllib.load(fh)
    assert meta["title"] == "Project Kickoff"
    assert meta["source"] == "manual"


def test_prepare_note_existing_inserts_timestamp_above_previous_content(tmp_path):
    manager = ManualNoteManager(_make_settings(tmp_path))

    note_dir = tmp_path / "updates-2024-04-15"
    note_dir.mkdir()
    note_path = note_dir / "notes.md"
    note_path.write_text(
        "# Updates\n\n2024-04-15 09:15\n\nInitial entry\n",
        encoding="utf-8",
    )

    import tomli_w

    with (note_dir / "meta.toml").open("wb") as fh:
        tomli_w.dump(
            {
                "title": "Updates",
                "date": "2024-04-15T09:15:00",
                "tags": [],
                "source": "manual",
            },
            fh,
        )

    now = datetime(2024, 4, 20, 14, 5)
    context = manager.prepare_note("Updates", now=now)

    assert context.is_new is False
    assert context.path == note_path
    assert context.content == (
        "# Updates\n\n2024-04-20 14:05\n\n2024-04-15 09:15\n\nInitial entry\n"
    )


def test_prepare_note_uses_default_name_when_not_provided(tmp_path):
    manager = ManualNoteManager(_make_settings(tmp_path))

    now = datetime(2024, 7, 4, 8, 0)
    context = manager.prepare_note(None, now=now)

    note_dir = tmp_path / "note-2024-07-04"
    assert context.is_new is True
    assert context.title == "note"
    assert context.path == note_dir / "notes.md"
    assert context.content == "# note\n\n2024-07-04 08:00\n\n"
