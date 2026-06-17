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
            mock_manager._add_to_index.return_value = True
            mock_manager._load_manifest.return_value = {}
            mock_manager._scan_notes_files.return_value = {
                str(notes_path): {"mtime": 123, "size": 100, "path": str(notes_path)}
            }

            generator._auto_index_note(notes_path)

            mock_manager._add_to_index.assert_called_once_with(notes_path)
            mock_manager._save_manifest.assert_called_once()
            # AC-6: a single save appends, never triggers a full-corpus rebuild.
            mock_manager.append_bm25_for_file.assert_called_once_with(str(notes_path))
            mock_manager._rebuild_bm25.assert_not_called()
            # L1: auto-index uses the same embed-fingerprint guard + stamp as build.
            mock_manager._ensure_embed_fingerprint.assert_called_once()
            mock_manager._stamp_fingerprint_if_missing.assert_called_once()

    def test_auto_index_skipped_on_embed_model_change(self, tmp_path):
        """L1: an embed-model change is detected through the auto-index path.

        ``_ensure_embed_fingerprint`` raising ``EmbedModelChanged`` must abort
        the save (no add, no manifest write) and surface the skip, instead of
        appending mismatched vectors via the auto-index back door.
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
            mock_manager._ensure_embed_fingerprint.side_effect = EmbedModelChanged(
                "embed model changed (a -> b); run `chirp index --force`"
            )

            generator._auto_index_note(notes_path)

            mock_manager._ensure_embed_fingerprint.assert_called_once()
            mock_manager._add_to_index.assert_not_called()
            mock_manager._save_manifest.assert_not_called()
            mock_manager.append_bm25_for_file.assert_not_called()

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
            mock_manager._add_to_index.return_value = False

            generator._auto_index_note(notes_path)

            mock_manager._add_to_index.assert_called_once_with(notes_path)
            mock_manager._save_manifest.assert_not_called()
            mock_manager._rebuild_bm25.assert_not_called()
