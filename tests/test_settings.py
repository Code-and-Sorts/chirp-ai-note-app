import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import (
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_MAX_RESIDENT_CHAT,
    DEFAULT_MAX_RESIDENT_EMBED,
    SUPPORTED_CONFIG_SCHEMA_VERSION,
    ChirpSettings,
    resolve_inference_timeout_seconds,
    resolve_max_resident_chat,
    resolve_max_resident_embed,
)


class TestChirpSettings:
    def test_default_settings_creation(self):
        settings = ChirpSettings()

        assert settings.models.whisper == "large-v3-turbo"
        assert settings.models.llm == "llama3.1:8b"
        assert settings.audio.sample_rate == 16000
        assert settings.audio.channels == 1

    def test_semantic_retrieval_defaults_off(self):
        assert ChirpSettings().notes_chat.semantic_enabled is False

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
        settings.notes_chat.semantic_enabled = True

        with patch(
            "config.settings.default_chirp_home", return_value=tmp_path / "home"
        ):
            settings.ensure_directories_exist()

        assert (tmp_path / "notes").is_dir()
        assert (tmp_path / "home").is_dir()
        assert (tmp_path / "home" / "chroma").is_dir()

    def test_ensure_directories_exist_skips_chroma_when_lexical_only(self, tmp_path):
        """A lexical-only install must not create an empty chroma/ directory."""
        settings = ChirpSettings()
        settings.directories.notes_root = tmp_path / "notes"
        settings.notes_chat.index_dir = tmp_path / "home"
        settings.notes_chat.semantic_enabled = False

        with patch(
            "config.settings.default_chirp_home", return_value=tmp_path / "home"
        ):
            settings.ensure_directories_exist()

        assert (tmp_path / "home").is_dir()
        assert not (tmp_path / "home" / "chroma").exists()

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

    def test_recommended_embed_model_round_trips(self, tmp_path):
        settings = ChirpSettings()
        settings.notes_chat.recommended_embed_model = "custom-embed"
        config_path = tmp_path / "config.toml"

        settings.save_to_file(config_path)
        reloaded = ChirpSettings.load_from_file(config_path)

        assert reloaded.notes_chat.recommended_embed_model == "custom-embed"

    def test_legacy_emb_model_key_loads_without_error(self, tmp_path):
        """An old config carrying the renamed ``emb_model`` key must still load.

        The key was renamed to ``recommended_embed_model``; pydantic ignores the
        stale extra key rather than raising, so existing configs survive upgrade.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[notes_chat]\nemb_model = "nomic-embed-text"\nk = 7\n',
            encoding="utf-8",
        )

        reloaded = ChirpSettings.load_from_file(config_path)

        assert not hasattr(reloaded.notes_chat, "emb_model")
        assert reloaded.notes_chat.k == 7
        assert reloaded.notes_chat.recommended_embed_model == "bge-small-en-v1.5-bf16"

    def test_non_table_init_value_does_not_block_load(self, tmp_path):
        """A hand-written non-table `init` value must not crash settings load.

        config.toml is user-editable; a stray ``init = "..."`` previously made
        every CLI command fail at ChirpSettings validation time.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('init = "oops"\n', encoding="utf-8")

        reloaded = ChirpSettings.load_from_file(config_path)

        assert reloaded.init.launch_agent_prompted_at is None


class TestTolerantConfigLoad:
    """AC-1/AC-2/AC-3: a hand-edited config never bricks the CLI at load."""

    def test_malformed_value_does_not_block_load(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[monitoring]\nwarning_minutes = "soon"\n', encoding="utf-8"
        )

        reloaded = ChirpSettings.load_from_file(config_path)

        assert reloaded.monitoring.warning_minutes == 60
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "warning_minutes" in captured.err
        assert str(config_path) in captured.err

    def test_malformed_field_preserves_other_valid_fields(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        custom_root = tmp_path / "my-notes"
        config_path.write_text(
            f'[directories]\nnotes_root = "{custom_root}"\n\n[notes_chat]\nk = "ten"\n',
            encoding="utf-8",
        )

        reloaded = ChirpSettings.load_from_file(config_path)

        assert reloaded.notes_chat.k == 10
        assert reloaded.directories.notes_root == custom_root
        assert "k" in capsys.readouterr().err

    def test_unknown_schema_version_warns_but_loads(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text("schema_version = 999\n", encoding="utf-8")

        reloaded = ChirpSettings.load_from_file(config_path)

        assert isinstance(reloaded, ChirpSettings)
        captured = capsys.readouterr()
        assert "schema_version" in captured.err
        assert "999" in captured.err

    def test_missing_schema_version_defaults_to_current(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("[monitoring]\nwarning_minutes = 30\n", encoding="utf-8")

        reloaded = ChirpSettings.load_from_file(config_path)

        assert reloaded.schema_version == SUPPORTED_CONFIG_SCHEMA_VERSION
        assert reloaded.monitoring.warning_minutes == 30

    def test_schema_version_round_trips_through_save(self, tmp_path):
        config_path = tmp_path / "config.toml"
        ChirpSettings().save_to_file(config_path)

        with config_path.open("rb") as fh:
            parsed = tomllib.load(fh)

        assert parsed["schema_version"] == SUPPORTED_CONFIG_SCHEMA_VERSION

    def test_unknown_top_level_key_warns(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text("[notez]\nfoo = 1\n", encoding="utf-8")

        reloaded = ChirpSettings.load_from_file(config_path)

        assert isinstance(reloaded, ChirpSettings)
        assert "notez" in capsys.readouterr().err

    def test_invalid_toml_does_not_block_load(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text('[monitoring\nwarning_minutes = "', encoding="utf-8")

        reloaded = ChirpSettings.load_from_file(config_path)

        assert isinstance(reloaded, ChirpSettings)
        assert reloaded.monitoring.warning_minutes == 60
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "not valid TOML" in captured.err
        assert str(config_path) in captured.err

    def test_bad_type_schema_version_warns_once(self, tmp_path, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text('schema_version = "v1"\n', encoding="utf-8")

        reloaded = ChirpSettings.load_from_file(config_path)

        assert reloaded.schema_version == SUPPORTED_CONFIG_SCHEMA_VERSION
        warning_lines = [
            line for line in capsys.readouterr().err.splitlines() if "Warning:" in line
        ]
        assert len(warning_lines) == 1
        assert "does not recognise" not in "\n".join(warning_lines)


class TestResolveInferenceTimeout:
    """AC-1: env → config → default precedence, mirroring the idle-timeout shape."""

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHIRP_INFERENCE_TIMEOUT", "12.5")
        assert resolve_inference_timeout_seconds() == 12.5

    def test_config_value_used_when_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CHIRP_INFERENCE_TIMEOUT", raising=False)

        class _LLM:
            inference_timeout_seconds = 7.0

        class _Settings:
            llm = _LLM()

        monkeypatch.setattr("config.settings.get_settings", lambda: _Settings())
        assert resolve_inference_timeout_seconds() == 7.0

    def test_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CHIRP_INFERENCE_TIMEOUT", raising=False)

        def _boom() -> object:
            raise RuntimeError("config blew up")

        monkeypatch.setattr("config.settings.get_settings", _boom)
        assert resolve_inference_timeout_seconds() == DEFAULT_INFERENCE_TIMEOUT_SECONDS

    def test_non_numeric_env_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHIRP_INFERENCE_TIMEOUT", "not-a-number")

        class _LLM:
            inference_timeout_seconds = 9.0

        class _Settings:
            llm = _LLM()

        monkeypatch.setattr("config.settings.get_settings", lambda: _Settings())
        assert resolve_inference_timeout_seconds() == 9.0

    def test_llm_settings_default_field(self) -> None:
        assert (
            ChirpSettings().llm.inference_timeout_seconds
            == DEFAULT_INFERENCE_TIMEOUT_SECONDS
        )


class TestResolveResidentCaps:
    """M1/AC-5: resident caps are operator-configurable via env → config."""

    def test_chat_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHIRP_MAX_RESIDENT_CHAT", "3")
        assert resolve_max_resident_chat() == 3

    def test_embed_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHIRP_MAX_RESIDENT_EMBED", "2")
        assert resolve_max_resident_embed() == 2

    def test_chat_config_value_used_when_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CHIRP_MAX_RESIDENT_CHAT", raising=False)

        class _LLM:
            max_resident_chat = 4

        class _Settings:
            llm = _LLM()

        monkeypatch.setattr("config.settings.get_settings", lambda: _Settings())
        assert resolve_max_resident_chat() == 4

    def test_falls_back_to_default_on_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CHIRP_MAX_RESIDENT_CHAT", raising=False)
        monkeypatch.delenv("CHIRP_MAX_RESIDENT_EMBED", raising=False)

        def _boom() -> object:
            raise RuntimeError("config blew up")

        monkeypatch.setattr("config.settings.get_settings", _boom)
        assert resolve_max_resident_chat() == DEFAULT_MAX_RESIDENT_CHAT
        assert resolve_max_resident_embed() == DEFAULT_MAX_RESIDENT_EMBED

    def test_non_numeric_env_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHIRP_MAX_RESIDENT_CHAT", "lots")

        class _LLM:
            max_resident_chat = 5

        class _Settings:
            llm = _LLM()

        monkeypatch.setattr("config.settings.get_settings", lambda: _Settings())
        assert resolve_max_resident_chat() == 5

    def test_llm_settings_default_fields(self) -> None:
        settings = ChirpSettings()
        assert settings.llm.max_resident_chat == DEFAULT_MAX_RESIDENT_CHAT
        assert settings.llm.max_resident_embed == DEFAULT_MAX_RESIDENT_EMBED
