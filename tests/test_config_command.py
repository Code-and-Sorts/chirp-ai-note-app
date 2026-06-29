"""Tests for ``chirp config`` — the semantic enable/disable flow and listing.

Every external boundary is mocked: ``register_model`` (HF download + registry),
``LLMClient`` (chirpd verify-load), and ``build_index`` (Chroma backfill). The
config file and chirp home are redirected under ``tmp_path`` so no test touches
the host's real ``~/.chirp``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from chirp import cli
from chirp.init_flow import RECOMMENDED_EMBED_REPO
from config.settings import ChirpSettings
from llm.exceptions import LLMModelLoadFailed

runner = CliRunner()


@pytest.fixture
def config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config_path = tmp_path / "config.toml"
    notes_root = tmp_path / "notes"
    index_dir = tmp_path / "home"
    notes_root.mkdir()
    index_dir.mkdir()

    seed = ChirpSettings()
    seed.directories.notes_root = notes_root
    seed.notes_chat.index_dir = index_dir
    seed.notes_chat.semantic_enabled = False
    seed.save_to_file(config_path)

    monkeypatch.setattr(
        ChirpSettings, "get_config_path", classmethod(lambda cls: config_path)
    )
    monkeypatch.setattr(
        cli, "get_settings", lambda: ChirpSettings.load_from_file(config_path)
    )
    monkeypatch.setattr("config.settings.default_chirp_home", lambda: index_dir)
    return SimpleNamespace(
        config_path=config_path, notes_root=notes_root, index_dir=index_dir
    )


def _reload(config_path: Path) -> ChirpSettings:
    return ChirpSettings.load_from_file(config_path)


def test_enable_registers_verifies_flips_and_rebuilds(
    config_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []

    def _register(repo, *, role=None, alias=None, warm=True, **_):
        order.append("register")
        assert repo == RECOMMENDED_EMBED_REPO
        assert role == "embed"
        assert warm is False
        return "bge-small-en-v1.5"

    client = MagicMock()
    client.model_load_sync.side_effect = lambda *a, **k: order.append("verify")

    def _build(config, force=False, progress_callback=None):
        order.append("build")
        assert force is True
        assert _reload(config_env.config_path).notes_chat.semantic_enabled is True
        return {"success": True}

    monkeypatch.setattr("llm.cli.models.register_model", _register)
    monkeypatch.setattr("llm.client.LLMClient", MagicMock(return_value=client))
    monkeypatch.setattr("notes_chat.index.build_index", _build)

    result = runner.invoke(cli.app, ["config", "--semantic"])

    assert result.exit_code == 0
    assert order == ["register", "verify", "build"]
    client.model_load_sync.assert_called_once_with("bge-small-en-v1.5", "embed")
    assert _reload(config_env.config_path).notes_chat.semantic_enabled is True


def test_enable_aborts_and_keeps_flag_off_when_load_fails(
    config_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MagicMock()
    client.model_load_sync.side_effect = LLMModelLoadFailed("bert not supported")
    build = MagicMock()

    monkeypatch.setattr(
        "llm.cli.models.register_model", MagicMock(return_value="bge-small-en-v1.5")
    )
    monkeypatch.setattr("llm.client.LLMClient", MagicMock(return_value=client))
    monkeypatch.setattr("notes_chat.index.build_index", build)

    result = runner.invoke(cli.app, ["config", "--semantic"])

    assert result.exit_code == 4
    assert "stays off" in result.stderr
    build.assert_not_called()
    assert _reload(config_env.config_path).notes_chat.semantic_enabled is False


def test_disable_with_purge_removes_chroma(config_env: SimpleNamespace) -> None:
    chroma = config_env.index_dir / "chroma"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_text("x")
    _seed_semantic_on(config_env)

    result = runner.invoke(cli.app, ["config", "--no-semantic", "--purge"])

    assert result.exit_code == 0
    assert not chroma.exists()
    assert _reload(config_env.config_path).notes_chat.semantic_enabled is False


def test_enable_when_already_on_is_a_noop(
    config_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_semantic_on(config_env)
    register = MagicMock()
    build = MagicMock()
    monkeypatch.setattr("llm.cli.models.register_model", register)
    monkeypatch.setattr("llm.client.LLMClient", MagicMock())
    monkeypatch.setattr("notes_chat.index.build_index", build)

    result = runner.invoke(cli.app, ["config", "--semantic"])

    assert result.exit_code == 0
    assert "already on" in result.stderr
    register.assert_not_called()
    build.assert_not_called()
    assert _reload(config_env.config_path).notes_chat.semantic_enabled is True


def test_enable_with_purge_warns_but_proceeds(
    config_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MagicMock()
    monkeypatch.setattr(
        "llm.cli.models.register_model", MagicMock(return_value="bge-small-en-v1.5")
    )
    monkeypatch.setattr("llm.client.LLMClient", MagicMock(return_value=client))
    monkeypatch.setattr(
        "notes_chat.index.build_index", MagicMock(return_value={"success": True})
    )

    result = runner.invoke(cli.app, ["config", "--semantic", "--purge"])

    assert result.exit_code == 0
    assert "no effect with --semantic" in result.stderr
    assert _reload(config_env.config_path).notes_chat.semantic_enabled is True


def test_disable_purge_when_chroma_missing(config_env: SimpleNamespace) -> None:
    _seed_semantic_on(config_env)
    assert not (config_env.index_dir / "chroma").exists()

    result = runner.invoke(cli.app, ["config", "--no-semantic", "--purge"])

    assert result.exit_code == 0
    assert "nothing to purge" in result.stderr
    assert _reload(config_env.config_path).notes_chat.semantic_enabled is False


def test_scalar_setter_applies_alongside_semantic_toggle(
    config_env: SimpleNamespace, tmp_path: Path
) -> None:
    new_root = tmp_path / "relocated"

    result = runner.invoke(
        cli.app, ["config", "--no-semantic", "--notes-root", str(new_root)]
    )

    assert result.exit_code == 0
    reloaded = _reload(config_env.config_path)
    assert reloaded.directories.notes_root == new_root
    assert reloaded.notes_chat.semantic_enabled is False


def test_scalar_setter_persists_through_enable(
    config_env: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    new_root = tmp_path / "relocated"
    monkeypatch.setattr(
        "llm.cli.models.register_model", MagicMock(return_value="bge-small-en-v1.5")
    )
    monkeypatch.setattr("llm.client.LLMClient", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "notes_chat.index.build_index", MagicMock(return_value={"success": True})
    )

    result = runner.invoke(
        cli.app, ["config", "--semantic", "--notes-root", str(new_root)]
    )

    assert result.exit_code == 0
    reloaded = _reload(config_env.config_path)
    assert reloaded.directories.notes_root == new_root
    assert reloaded.notes_chat.semantic_enabled is True


def test_disable_without_purge_keeps_chroma(config_env: SimpleNamespace) -> None:
    chroma = config_env.index_dir / "chroma"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_text("x")
    _seed_semantic_on(config_env)

    result = runner.invoke(cli.app, ["config", "--no-semantic"])

    assert result.exit_code == 0
    assert chroma.exists()
    assert _reload(config_env.config_path).notes_chat.semantic_enabled is False


def _seed_semantic_on(config_env: SimpleNamespace) -> None:
    settings = _reload(config_env.config_path)
    settings.notes_chat.semantic_enabled = True
    settings.save_to_file(config_env.config_path)


def test_list_shows_resolved_embed_alias(
    config_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str | None] = []

    def _spy(fallback=None, path=None):
        calls.append(fallback)
        return "spy-embed"

    monkeypatch.setattr(cli, "resolved_embed_model", _spy)

    result = runner.invoke(cli.app, ["config", "--list"])

    assert result.exit_code == 0
    assert "spy-embed" in result.stdout
    assert "nomic-embed-text" not in result.stdout
    assert calls == ["bge-small-en-v1.5"]


def test_embedding_model_option_is_gone(config_env: SimpleNamespace) -> None:
    result = runner.invoke(cli.app, ["config", "--embedding-model", "nomic-embed-text"])

    assert result.exit_code == 2


def test_purge_without_disable_is_a_noop_warning(
    config_env: SimpleNamespace,
) -> None:
    result = runner.invoke(cli.app, ["config", "--purge"])

    assert result.exit_code == 0
    assert "no effect" in result.stderr
    assert _reload(config_env.config_path).notes_chat.semantic_enabled is False
