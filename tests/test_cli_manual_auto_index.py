from pathlib import Path
from unittest.mock import Mock, patch

from config.settings import ChirpSettings
from notes.manual_note_manager import NoteContext
from notes.note_editor import EditorResult


def _make_settings(tmp_path: Path, enabled: bool) -> ChirpSettings:
    settings = ChirpSettings()
    settings.directories.notes = tmp_path
    settings.notes_chat.auto_index = enabled
    return settings


def _patch_tty(monkeypatch):
    monkeypatch.setattr("chirp.cli.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("chirp.cli.sys.stdout.isatty", lambda: True)


def test_notes_auto_index_enabled_success(tmp_path, monkeypatch):
    from chirp.cli import notes as notes_cmd

    settings = _make_settings(tmp_path, enabled=True)
    monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

    note_path = tmp_path / "Test-Note.md"
    context = NoteContext(
        path=note_path,
        title="Test Note",
        content="# Test Note\n",
        is_new=True,
    )

    class DummyManager:
        def __init__(self, _):
            pass

        def prepare_note(self, name):
            return context

    class DummyEditor:
        def __init__(self, title, initial_content, readonly: bool = False):
            pass

        def run(self):
            return EditorResult(content="# Test Note\nBody\n", saved=True)

    monkeypatch.setattr("notes.manual_note_manager.ManualNoteManager", DummyManager)
    monkeypatch.setattr("notes.note_editor.ManualNoteEditor", DummyEditor)
    _patch_tty(monkeypatch)

    with patch("notes_chat.index.IndexManager") as mock_manager_class:
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager._add_to_index.return_value = True
        mock_manager._load_manifest.return_value = {}
        mock_manager._scan_notes_files.return_value = {
            str(note_path): {"mtime": 1, "size": 10, "path": str(note_path)}
        }

        notes_cmd(name="Test Note")

        mock_manager._add_to_index.assert_called_once_with(note_path)
        mock_manager._save_manifest.assert_called_once()
        mock_manager._rebuild_bm25.assert_called_once()


def test_notes_auto_index_disabled(tmp_path, monkeypatch):
    from chirp.cli import notes as notes_cmd

    settings = _make_settings(tmp_path, enabled=False)
    monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

    note_path = tmp_path / "Test-Note.md"
    context = NoteContext(
        path=note_path,
        title="Test Note",
        content="# Test Note\n",
        is_new=True,
    )

    class DummyManager:
        def __init__(self, _):
            pass

        def prepare_note(self, name):
            return context

    class DummyEditor:
        def __init__(self, title, initial_content, readonly: bool = False):
            pass

        def run(self):
            return EditorResult(content="# Test Note\nBody\n", saved=True)

    monkeypatch.setattr("notes.manual_note_manager.ManualNoteManager", DummyManager)
    monkeypatch.setattr("notes.note_editor.ManualNoteEditor", DummyEditor)
    _patch_tty(monkeypatch)

    with patch("notes_chat.index.IndexManager") as mock_manager_class:
        notes_cmd(name="Test Note")

        mock_manager_class.assert_not_called()


def test_notes_auto_index_add_failure(tmp_path, monkeypatch):
    from chirp.cli import notes as notes_cmd

    settings = _make_settings(tmp_path, enabled=True)
    monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

    note_path = tmp_path / "Test-Note.md"
    context = NoteContext(
        path=note_path,
        title="Test Note",
        content="# Test Note\n",
        is_new=True,
    )

    class DummyManager:
        def __init__(self, _):
            pass

        def prepare_note(self, name):
            return context

    class DummyEditor:
        def __init__(self, title, initial_content, readonly: bool = False):
            pass

        def run(self):
            return EditorResult(content="# Test Note\nBody\n", saved=True)

    monkeypatch.setattr("notes.manual_note_manager.ManualNoteManager", DummyManager)
    monkeypatch.setattr("notes.note_editor.ManualNoteEditor", DummyEditor)
    _patch_tty(monkeypatch)

    with patch("notes_chat.index.IndexManager") as mock_manager_class:
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager._add_to_index.return_value = False

        notes_cmd(name="Test Note")

        mock_manager._add_to_index.assert_called_once_with(note_path)
        mock_manager._save_manifest.assert_not_called()
        mock_manager._rebuild_bm25.assert_not_called()
