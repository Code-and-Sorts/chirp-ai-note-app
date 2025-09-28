from unittest.mock import Mock, patch

from config.settings import ChirpSettings
from notes_chat.index import IndexManager


class TestIndexManifest:
    def test_signature_calculation(self, tmp_path):
        """Test file signature calculation."""
        config = ChirpSettings()
        config.directories.notes = tmp_path
        config.notes_chat.index_dir = tmp_path / ".notes_index"

        note_file = tmp_path / "test.md"
        note_file.write_text(
            "# Test Meeting\n\nThis is some longer content that should be sufficient for chunking and indexing purposes. It contains enough text to pass the minimum length requirements."
        )

        manager = IndexManager(config)
        files = manager._scan_notes_files()

        assert str(note_file) in files
        assert "mtime" in files[str(note_file)]
        assert "size" in files[str(note_file)]
        assert files[str(note_file)]["size"] > 0

    @patch("requests.post")
    def test_idempotent_skip(self, mock_post, tmp_path):
        """Test that unchanged files are skipped."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 384}
        mock_post.return_value = mock_response

        config = ChirpSettings()
        config.directories.notes = tmp_path
        config.notes_chat.index_dir = tmp_path / ".notes_index"

        note_file = tmp_path / "test.md"
        note_file.write_text(
            "# Test Meeting\n\nThis is some longer content that should be sufficient for chunking and indexing purposes. It contains enough text to pass the minimum length requirements."
        )

        manager = IndexManager(config)

        result1 = manager.build_index()
        assert result1["success"]
        assert result1["files_processed"] == 1

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["files_processed"] == 0
        assert "up to date" in result2["message"]

    @patch("requests.post")
    def test_force_rebuild(self, mock_post, tmp_path):
        """Test --force rebuild behavior."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 384}
        mock_post.return_value = mock_response

        config = ChirpSettings()
        config.directories.notes = tmp_path
        config.notes_chat.index_dir = tmp_path / ".notes_index"

        note_file = tmp_path / "test.md"
        note_file.write_text(
            "# Test Meeting\n\nThis is some longer content that should be sufficient for chunking and indexing purposes. It contains enough text to pass the minimum length requirements."
        )

        manager = IndexManager(config)

        result1 = manager.build_index()
        assert result1["success"]

        result2 = manager.build_index(force=True)
        assert result2["success"]
        assert result2["files_processed"] == 1

    @patch("requests.post")
    def test_file_removal_detection(self, mock_post, tmp_path):
        """Test detection of removed files."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 384}
        mock_post.return_value = mock_response

        config = ChirpSettings()
        config.directories.notes = tmp_path
        config.notes_chat.index_dir = tmp_path / ".notes_index"

        note_file = tmp_path / "test.md"
        note_file.write_text(
            "# Test Meeting\n\nThis is some longer content that should be sufficient for chunking and indexing purposes. It contains enough text to pass the minimum length requirements."
        )

        manager = IndexManager(config)

        result1 = manager.build_index()
        assert result1["success"]

        note_file.unlink()

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["removed"] == 1

    @patch("requests.post")
    def test_file_modification_detection(self, mock_post, tmp_path):
        """Test detection of modified files."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 384}
        mock_post.return_value = mock_response

        config = ChirpSettings()
        config.directories.notes = tmp_path
        config.notes_chat.index_dir = tmp_path / ".notes_index"

        note_file = tmp_path / "test.md"
        note_file.write_text(
            "# Test Meeting\n\nThis is some longer content that should be sufficient for chunking and indexing purposes. It contains enough text to pass the minimum length requirements."
        )

        manager = IndexManager(config)

        result1 = manager.build_index()
        assert result1["success"]

        import time

        time.sleep(0.1)
        note_file.write_text("# Test Meeting\n\nModified content")

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["modified"] == 1

    @patch("requests.post")
    def test_embedding_failure_handling(self, mock_post, tmp_path):
        """Test handling of embedding API failures."""
        mock_post.return_value.status_code = 500

        config = ChirpSettings()
        config.directories.notes = tmp_path
        config.notes_chat.index_dir = tmp_path / ".notes_index"

        note_file = tmp_path / "test.md"
        note_file.write_text(
            "# Test Meeting\n\nThis is some longer content that should be sufficient for chunking and indexing purposes. It contains enough text to pass the minimum length requirements."
        )

        manager = IndexManager(config)
        result = manager.build_index()

        assert result["success"]
        assert result["files_processed"] == 0

    def test_metadata_extraction(self, tmp_path):
        """Test extraction of metadata from notes files."""
        config = ChirpSettings()
        config.directories.notes = tmp_path
        config.notes_chat.index_dir = tmp_path / ".notes_index"

        content = """# Weekly Standup Meeting

**Duration:** 45m
**Participants:** Alice, Bob, Charlie

## Summary
Test meeting content
"""

        note_file = tmp_path / "meetings_2025_01_15.md"
        note_file.write_text(content)

        manager = IndexManager(config)
        meta = manager._extract_metadata(note_file, content)

        assert meta is not None
        assert meta.title == "Weekly Standup Meeting"
        assert meta.duration == 45
        assert "Alice" in meta.participants
        assert "Bob" in meta.participants
        assert "Charlie" in meta.participants
