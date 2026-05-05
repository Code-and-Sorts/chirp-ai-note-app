import tomllib
from pathlib import Path
from unittest.mock import patch

from config.settings import ChirpSettings


class TestChirpSettings:
    def test_default_settings_creation(self):
        settings = ChirpSettings()

        assert settings.models.whisper == "large-v3-turbo"
        assert settings.models.llm == "llama3.1:8b"
        assert settings.audio.sample_rate == 16000
        assert settings.audio.channels == 1

    def test_directories_config_paths(self):
        settings = ChirpSettings()

        assert isinstance(settings.directories.notes_root, Path)

    def test_notes_root_default_is_lowercase_chirp(self):
        settings = ChirpSettings()

        assert settings.directories.notes_root.name == "chirp"

    def test_ensure_directories_exist_creates_expected_paths(self, tmp_path):
        settings = ChirpSettings()
        settings.directories.notes_root = tmp_path / "notes"
        settings.notes_chat.index_dir = tmp_path / "home"

        with patch(
            "config.settings.default_chirp_home", return_value=tmp_path / "home"
        ):
            settings.ensure_directories_exist()

        assert (tmp_path / "notes").is_dir()
        assert (tmp_path / "home").is_dir()
        assert (tmp_path / "home" / "chroma").is_dir()

    def test_config_path_lives_under_chirp_home(self):
        assert ChirpSettings.get_config_path() == Path.home() / ".chirp" / "config.toml"

    def test_save_and_load_round_trip(self, tmp_path):
        settings = ChirpSettings()
        settings.directories.notes_root = tmp_path / "custom-root"
        config_path = tmp_path / "config.toml"

        settings.save_to_file(config_path)

        assert config_path.exists()
        with config_path.open("rb") as fh:
            parsed = tomllib.load(fh)
        assert parsed["directories"]["notes_root"] == str(tmp_path / "custom-root")

        reloaded = ChirpSettings.load_from_file(config_path)
        assert reloaded.directories.notes_root == tmp_path / "custom-root"
