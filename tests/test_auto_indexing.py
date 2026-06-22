from unittest.mock import Mock, patch

from config.settings import ChirpSettings
from notes.note_generator import NoteGenerator


class TestAutoIndexing:
    def test_auto_index_enabled_success(self, tmp_path):
        settings = ChirpSettings()
        settings.notes_chat.auto_index = True
        settings.directories.notes_root = tmp_path

        generator = NoteGenerator(settings)
        note_dir = tmp_path / "sample-2026-04-20"
        note_dir.mkdir()
        notes_path = note_dir / "notes.md"
        notes_path.write_text("# Test Note\nContent")

        with patch("notes_chat.index.IndexManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.add_note.return_value = True

            generator._auto_index_note(notes_path)

            mock_manager.add_note.assert_called_once_with(
                notes_path, guard_embed_fingerprint=True, incremental_bm25=True
            )

    def test_auto_index_skipped_on_embed_model_change(self, tmp_path):
        """L1: an embed-model change surfaced by add_note aborts cleanly.

        ``add_note`` raising ``EmbedModelChanged`` must be caught and surfaced as
        a skip rather than crashing note generation.
        """
        from chirp.exceptions import EmbedModelChanged

        settings = ChirpSettings()
        settings.notes_chat.auto_index = True
        settings.directories.notes_root = tmp_path

        generator = NoteGenerator(settings)
        note_dir = tmp_path / "sample-2026-04-20"
        note_dir.mkdir()
        notes_path = note_dir / "notes.md"
        notes_path.write_text("# Test Note\nContent")

        with patch("notes_chat.index.IndexManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.add_note.side_effect = EmbedModelChanged(
                "embed model changed (a -> b); run `chirp index --force`"
            )

            generator._auto_index_note(notes_path)

            mock_manager.add_note.assert_called_once_with(
                notes_path, guard_embed_fingerprint=True, incremental_bm25=True
            )

    def test_auto_index_disabled(self, tmp_path):
        settings = ChirpSettings()
        settings.notes_chat.auto_index = False
        settings.directories.notes_root = tmp_path

        generator = NoteGenerator(settings)
        notes_path = tmp_path / "sample" / "notes.md"

        with patch("notes_chat.index.IndexManager") as mock_manager_class:
            generator._auto_index_note(notes_path)

            mock_manager_class.assert_not_called()

    def test_auto_index_graceful_failure(self, tmp_path):
        settings = ChirpSettings()
        settings.notes_chat.auto_index = True
        settings.directories.notes_root = tmp_path

        generator = NoteGenerator(settings)
        notes_path = tmp_path / "sample" / "notes.md"

        with patch("notes_chat.index.IndexManager") as mock_manager_class:
            mock_manager_class.side_effect = Exception("index manager not available")

            generator._auto_index_note(notes_path)

    def test_auto_index_add_failure(self, tmp_path):
        settings = ChirpSettings()
        settings.notes_chat.auto_index = True
        settings.directories.notes_root = tmp_path

        generator = NoteGenerator(settings)
        notes_path = tmp_path / "sample" / "notes.md"

        with patch("notes_chat.index.IndexManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.add_note.return_value = False

            generator._auto_index_note(notes_path)

            mock_manager.add_note.assert_called_once_with(
                notes_path, guard_embed_fingerprint=True, incremental_bm25=True
            )
