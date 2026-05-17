"""Tests for :mod:`llm.registry`."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm.exceptions import LLMMalformedResponse, LLMModelNotFound
from llm.registry import Registry, RegistryEntry, read_registry, resolve_alias

_WELL_FORMED_TOML = """\
schema_version = 1
default_chat = "gemma-4-4b"
default_embed = "nomic-text-v1"

[models.gemma-4-4b]
hf_repo = "mlx-community/gemma-4-4b-it-4bit"
role = "chat"

[models.gemma-4-4b.options]
temperature = 0.6

[models.nomic-text-v1]
hf_repo = "mlx-community/nomic-embed-text-v1.5"
role = "embed"
"""


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_read_registry_returns_empty_when_file_missing(tmp_path: Path) -> None:
    registry = read_registry(tmp_path / "absent.toml")
    assert registry.models == {}
    assert registry.default_chat is None
    assert registry.default_embed is None
    assert registry.schema_version == 1


def test_read_registry_parses_well_formed_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "models.toml", _WELL_FORMED_TOML)
    registry = read_registry(path)
    assert registry.default_chat == "gemma-4-4b"
    assert registry.default_embed == "nomic-text-v1"
    assert registry.models["gemma-4-4b"].hf_repo == "mlx-community/gemma-4-4b-it-4bit"
    assert registry.models["gemma-4-4b"].role == "chat"
    assert registry.models["gemma-4-4b"].options == {"temperature": 0.6}
    assert registry.models["nomic-text-v1"].role == "embed"


def test_read_registry_rejects_unknown_schema_version(tmp_path: Path) -> None:
    body = 'schema_version = 99\n[models.foo]\nhf_repo = "a/b"\nrole = "chat"\n'
    path = _write(tmp_path / "models.toml", body)
    with pytest.raises(LLMModelNotFound) as exc_info:
        read_registry(path)
    assert "re-init" in exc_info.value.message
    assert exc_info.value.details["schema_version"] == 99


def test_read_registry_raises_malformed_on_bad_toml(tmp_path: Path) -> None:
    path = _write(tmp_path / "models.toml", "schema_version = 1\nthis is not toml")
    with pytest.raises(LLMMalformedResponse):
        read_registry(path)


def test_read_registry_defaults_path_to_models_toml_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "models.toml"
    monkeypatch.setattr("llm.registry.MODELS_TOML_PATH", target)
    registry = read_registry()
    assert registry.models == {}


def test_resolve_alias_by_name() -> None:
    entry = RegistryEntry(hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat")
    registry = Registry(schema_version=1, models={"gemma-4-4b": entry})
    assert resolve_alias(registry, "gemma-4-4b", "chat") is entry


def test_resolve_alias_default_chat() -> None:
    entry = RegistryEntry(hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat")
    registry = Registry(
        schema_version=1,
        default_chat="gemma-4-4b",
        models={"gemma-4-4b": entry},
    )
    assert resolve_alias(registry, "default", "chat") is entry


def test_resolve_alias_default_when_unset_raises() -> None:
    registry = Registry(schema_version=1, models={})
    with pytest.raises(LLMModelNotFound):
        resolve_alias(registry, "default", "chat")


def test_resolve_alias_default_points_to_missing_entry_raises() -> None:
    registry = Registry(schema_version=1, default_chat="ghost", models={})
    with pytest.raises(LLMModelNotFound):
        resolve_alias(registry, "default", "chat")


def test_resolve_alias_raw_org_repo() -> None:
    registry = Registry(schema_version=1, models={})
    entry = resolve_alias(registry, "mlx-community/foo", "chat")
    assert entry.hf_repo == "mlx-community/foo"
    assert entry.role == "chat"
    assert entry.options == {}


def test_resolve_alias_unknown_raises() -> None:
    registry = Registry(schema_version=1, models={})
    with pytest.raises(LLMModelNotFound):
        resolve_alias(registry, "bogus", "chat")


def test_resolve_alias_by_name_rejects_role_mismatch() -> None:
    registry = Registry(
        schema_version=1,
        models={
            "nomic": RegistryEntry(hf_repo="mlx-community/nomic-embed", role="embed"),
        },
    )
    with pytest.raises(LLMModelNotFound) as exc:
        resolve_alias(registry, "nomic", "chat")
    assert exc.value.details["registered_role"] == "embed"
    assert exc.value.details["requested_role"] == "chat"


def test_resolve_alias_default_rejects_role_mismatch() -> None:
    registry = Registry(
        schema_version=1,
        default_chat="mislabeled",
        models={
            "mislabeled": RegistryEntry(
                hf_repo="mlx-community/embed-as-chat", role="embed"
            ),
        },
    )
    with pytest.raises(LLMModelNotFound) as exc:
        resolve_alias(registry, "default", "chat")
    assert exc.value.details["registered_role"] == "embed"
    assert exc.value.details["requested_role"] == "chat"
