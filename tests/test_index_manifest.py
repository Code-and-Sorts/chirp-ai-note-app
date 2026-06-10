from config.settings import ChirpSettings
from llm.exceptions import LLMTransportError
from notes_chat.index import IndexManager

_NOTE_BODY = (
    "# Test Meeting\n\nThis is some longer content that should be sufficient "
    "for chunking and indexing purposes. It contains enough text to pass the "
    "minimum length requirements."
)


class _FakeEmbedClient:
    """Fake LLMClient.embed_sync: one vector per input, in order."""

    def __init__(self, dim: int = 384, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail

    def embed_sync(self, inputs, model="default"):
        if self.fail:
            raise LLMTransportError("embed daemon unavailable")
        return [[0.1] * self.dim for _ in inputs]


def _make_config(tmp_path):
    config = ChirpSettings()
    config.directories.notes_root = tmp_path
    config.notes_chat.index_dir = tmp_path / ".notes_index"
    return config


def _seed_note(tmp_path) -> "object":
    note_dir = tmp_path / "test-2026-04-20"
    note_dir.mkdir()
    note_file = note_dir / "notes.md"
    note_file.write_text(_NOTE_BODY)
    return note_file


class TestIndexManifest:
    def test_signature_calculation(self, tmp_path):
        """Test file signature calculation."""
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)

        manager = IndexManager(config)
        files = manager._scan_notes_files()

        assert str(note_file) in files
        assert "mtime" in files[str(note_file)]
        assert "size" in files[str(note_file)]
        assert files[str(note_file)]["size"] > 0

    def test_idempotent_skip(self, tmp_path):
        """Test that unchanged files are skipped."""
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        result1 = manager.build_index()
        assert result1["success"]
        assert result1["files_processed"] == 1

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["files_processed"] == 0
        assert "up to date" in result2["message"]

    def test_force_rebuild(self, tmp_path):
        """Test --force rebuild behavior."""
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        result1 = manager.build_index()
        assert result1["success"]

        result2 = manager.build_index(force=True)
        assert result2["success"]
        assert result2["files_processed"] == 1

    def test_file_removal_detection(self, tmp_path):
        """Test detection of removed files."""
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        result1 = manager.build_index()
        assert result1["success"]

        note_file.unlink()

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["removed"] == 1

    def test_file_modification_detection(self, tmp_path):
        """Test detection of modified files."""
        config = _make_config(tmp_path)
        note_file = _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient())

        result1 = manager.build_index()
        assert result1["success"]

        import time

        time.sleep(0.1)
        note_file.write_text("# Test Meeting\n\nModified content")

        result2 = manager.build_index()
        assert result2["success"]
        assert result2["modified"] == 1

    def test_embedding_failure_handling(self, tmp_path):
        """A failing embed leaves build_index successful but indexes nothing."""
        config = _make_config(tmp_path)
        _seed_note(tmp_path)

        manager = IndexManager(config, llm_client=_FakeEmbedClient(fail=True))
        result = manager.build_index()

        assert result["success"]
