from pathlib import Path
from unittest.mock import patch

from config.settings import ChirpSettings


class TestChirpSettings:
    def test_default_settings_creation(self):
        settings = ChirpSettings()

        assert settings.models.whisper == "base"
        assert settings.models.llm == "llama3.1:8b"
        assert settings.audio.sample_rate == 16000
        assert settings.audio.channels == 2

    def test_directories_config_paths(self):
        settings = ChirpSettings()

        assert isinstance(settings.directories.raw_audio, Path)
        assert isinstance(settings.directories.transcriptions, Path)
        assert isinstance(settings.directories.notes, Path)
        assert isinstance(settings.directories.templates, Path)

    def test_ensure_directories_exist(self):
        settings = ChirpSettings()

        with patch.object(Path, "mkdir") as mock_mkdir:
            settings.ensure_directories_exist()

            assert mock_mkdir.call_count == 8
