from datetime import datetime

from config.settings import ChirpSettings
from notes.manual_note_manager import ManualNoteManager


def _make_settings(tmp_path):
    settings = ChirpSettings()
    settings.directories.notes = tmp_path
    return settings


def test_prepare_note_new_creates_header_and_timestamp(tmp_path):
    settings = _make_settings(tmp_path)
    manager = ManualNoteManager(settings)

    now = datetime(2024, 5, 1, 10, 30)
    context = manager.prepare_note("Project Kickoff", now=now)

    assert context.is_new is True
    assert context.path == tmp_path / "Project-Kickoff.md"
    assert context.title == "Project Kickoff"
    assert context.content == "# Project Kickoff\n\n2024-05-01 10:30\n\n"


def test_prepare_note_existing_inserts_timestamp_above_previous_content(tmp_path):
    settings = _make_settings(tmp_path)
    manager = ManualNoteManager(settings)

    note_path = tmp_path / "Updates.md"
    note_path.write_text(
        "# Updates\n\n2024-04-15 09:15\n\nInitial entry\n",
        encoding="utf-8",
    )

    now = datetime(2024, 4, 20, 14, 5)
    context = manager.prepare_note("Updates", now=now)

    assert context.is_new is False
    assert context.path == note_path
    assert context.content == (
        "# Updates\n\n2024-04-20 14:05\n\n2024-04-15 09:15\n\nInitial entry\n"
    )


def test_prepare_note_uses_default_name_when_not_provided(tmp_path):
    settings = _make_settings(tmp_path)
    manager = ManualNoteManager(settings)

    now = datetime(2024, 7, 4, 8, 0)
    context = manager.prepare_note(None, now=now)

    assert context.is_new is True
    assert context.title == "note-2024-07-04"
    assert context.path == tmp_path / "note-2024-07-04.md"
    assert context.content == "# note-2024-07-04\n\n2024-07-04 08:00\n\n"
