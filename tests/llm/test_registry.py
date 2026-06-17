"""Tests for :mod:`llm.registry`."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

from llm.exceptions import LLMMalformedResponse, LLMModelNotFound
from llm.registry import (
    HEADER_COMMENT,
    Registry,
    RegistryEntry,
    RegistryWriteError,
    alias_for_repo,
    read_registry,
    remove_model,
    resolve_alias,
    set_default_for_role,
    upsert_model,
    write_registry,
)

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
    # AC-10: point at the real entry point, not the nonexistent `models init`.
    assert "chirp models add" in exc_info.value.message
    assert "models init" not in exc_info.value.message
    assert "EPIC-MODEL-REGISTRY" not in exc_info.value.message
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


# ---------------------------------------------------------------------------
# Writer + mutation helpers (Story 4.1)
# ---------------------------------------------------------------------------


def _chat_entry() -> RegistryEntry:
    return RegistryEntry(
        hf_repo="mlx-community/gemma-4-4b-it-4bit",
        role="chat",
        options={"temperature": 0.7, "top_p": 0.9, "max_tokens": 2048},
    )


def _embed_entry() -> RegistryEntry:
    return RegistryEntry(
        hf_repo="mlx-community/bge-small-en-v1.5",
        role="embed",
    )


def test_registry_round_trip_empty(tmp_path: Path) -> None:
    target = tmp_path / "models.toml"
    original = Registry(schema_version=1)
    write_registry(original, path=target)
    loaded = read_registry(target)
    assert loaded == original


def test_registry_round_trip_single_chat(tmp_path: Path) -> None:
    target = tmp_path / "models.toml"
    original = Registry(
        schema_version=1,
        default_chat="gemma-4-4b-it-4bit",
        models={"gemma-4-4b-it-4bit": _chat_entry()},
    )
    write_registry(original, path=target)
    loaded = read_registry(target)
    assert loaded == original


def test_registry_round_trip_chat_and_embed(tmp_path: Path) -> None:
    target = tmp_path / "models.toml"
    original = Registry(
        schema_version=1,
        default_chat="gemma-4-4b-it-4bit",
        default_embed="bge-small-en-v1.5",
        models={
            "gemma-4-4b-it-4bit": _chat_entry(),
            "bge-small-en-v1.5": _embed_entry(),
        },
    )
    write_registry(original, path=target)
    loaded = read_registry(target)
    assert loaded == original


def test_write_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "models.toml"
    write_registry(Registry(schema_version=1), path=target)
    assert target.exists()
    assert target.parent.is_dir()


def test_write_uses_default_path_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "default-models.toml"
    monkeypatch.setattr("llm.registry.MODELS_TOML_PATH", target)
    write_registry(Registry(schema_version=1))
    assert target.exists()


def test_write_is_atomic_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "models.toml"
    original = Registry(
        schema_version=1,
        default_chat="gemma-4-4b-it-4bit",
        models={"gemma-4-4b-it-4bit": _chat_entry()},
    )
    write_registry(original, path=target)
    original_bytes = target.read_bytes()

    def boom(_src: str, _dst: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("llm.registry.os.replace", boom)

    replacement = Registry(
        schema_version=1,
        default_chat="other",
        models={"other": RegistryEntry(hf_repo="mlx-community/other", role="chat")},
    )
    with pytest.raises(RegistryWriteError):
        write_registry(replacement, path=target)

    assert target.read_bytes() == original_bytes
    assert read_registry(target) == original
    assert not list(target.parent.glob(f"{target.name}.*.tmp"))


def test_write_wraps_mkdir_failure_as_registry_write_error(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "blocked"
    occupied.write_text("not a directory")
    target = occupied / "models.toml"

    with pytest.raises(RegistryWriteError) as exc:
        write_registry(Registry(schema_version=1), path=target)
    assert "failed to write models.toml" in str(exc.value)
    assert occupied.read_text() == "not a directory"


def test_write_succeeds_even_when_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "models.toml"
    real_fsync = os.fsync
    real_open = os.open
    directory_fds: set[int] = set()

    def tracking_open(*args: Any, **kwargs: Any) -> int:
        fd = real_open(*args, **kwargs)
        directory_fds.add(fd)
        return fd

    def fsync_fails_for_directory(fd: int) -> None:
        if fd in directory_fds:
            raise OSError("simulated directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr("llm.registry.os.open", tracking_open)
    monkeypatch.setattr("llm.registry.os.fsync", fsync_fails_for_directory)

    registry = Registry(
        schema_version=1,
        default_chat="gemma-4-4b-it-4bit",
        models={"gemma-4-4b-it-4bit": _chat_entry()},
    )
    write_registry(registry, path=target)

    assert directory_fds, "_fsync_directory_best_effort never opened a directory fd"
    assert read_registry(target) == registry


def test_upsert_model_inserts(tmp_path: Path) -> None:
    target = tmp_path / "models.toml"
    registry = Registry(schema_version=1)
    updated = upsert_model(registry, "gemma-4-4b-it-4bit", _chat_entry())
    assert registry.models == {}
    assert updated.models["gemma-4-4b-it-4bit"] == _chat_entry()
    write_registry(updated, path=target)
    assert read_registry(target) == updated


def test_upsert_model_replaces(tmp_path: Path) -> None:
    target = tmp_path / "models.toml"
    registry = Registry(
        schema_version=1,
        models={"gemma-4-4b-it-4bit": _chat_entry()},
    )
    replacement = RegistryEntry(
        hf_repo="mlx-community/gemma-4-4b-it-4bit",
        role="chat",
        options={"temperature": 0.1},
    )
    updated = upsert_model(registry, "gemma-4-4b-it-4bit", replacement)
    assert updated.models["gemma-4-4b-it-4bit"].options == {"temperature": 0.1}
    write_registry(updated, path=target)
    assert read_registry(target) == updated


def test_upsert_rejects_empty_or_slashed_alias() -> None:
    registry = Registry(schema_version=1)
    entry = _chat_entry()
    with pytest.raises(ValueError, match="alias must be non-empty"):
        upsert_model(registry, "", entry)
    with pytest.raises(ValueError, match="must match"):
        upsert_model(registry, "mlx-community/gemma", entry)


def test_upsert_rejects_invalid_alias_characters() -> None:
    registry = Registry(schema_version=1)
    entry = _chat_entry()
    for bad in ("Gemma-4-4b", "has spaces", "exclaim!", "über"):
        with pytest.raises(ValueError, match="must match"):
            upsert_model(registry, bad, entry)


def test_remove_model_clears_default() -> None:
    registry = Registry(
        schema_version=1,
        default_chat="gemma-4-4b-it-4bit",
        models={"gemma-4-4b-it-4bit": _chat_entry()},
    )
    updated = remove_model(registry, "gemma-4-4b-it-4bit")
    assert updated.models == {}
    assert updated.default_chat is None
    assert registry.default_chat == "gemma-4-4b-it-4bit"


def test_remove_model_clears_embed_default() -> None:
    registry = Registry(
        schema_version=1,
        default_embed="bge-small-en-v1.5",
        models={"bge-small-en-v1.5": _embed_entry()},
    )
    updated = remove_model(registry, "bge-small-en-v1.5")
    assert updated.models == {}
    assert updated.default_embed is None


def test_remove_unknown_alias_raises_keyerror() -> None:
    registry = Registry(schema_version=1)
    with pytest.raises(KeyError):
        remove_model(registry, "ghost")


def test_set_default_for_role_chat() -> None:
    registry = Registry(
        schema_version=1,
        models={"gemma-4-4b-it-4bit": _chat_entry()},
    )
    updated = set_default_for_role(registry, "gemma-4-4b-it-4bit")
    assert updated.default_chat == "gemma-4-4b-it-4bit"
    assert updated.default_embed is None


def test_set_default_for_role_embed() -> None:
    registry = Registry(
        schema_version=1,
        models={"bge-small-en-v1.5": _embed_entry()},
    )
    updated = set_default_for_role(registry, "bge-small-en-v1.5")
    assert updated.default_embed == "bge-small-en-v1.5"
    assert updated.default_chat is None


def test_set_default_for_role_unknown_alias_raises_keyerror() -> None:
    registry = Registry(schema_version=1)
    with pytest.raises(KeyError):
        set_default_for_role(registry, "ghost")


def test_alias_for_repo_strips_org_and_lowercases() -> None:
    assert alias_for_repo("mlx-community/gemma-4-4b-it-4bit") == "gemma-4-4b-it-4bit"
    assert alias_for_repo("BAAI/bge-small-en-v1.5") == "bge-small-en-v1.5"
    assert alias_for_repo("Gemma-4-4b-it-4bit") == "gemma-4-4b-it-4bit"


def test_alias_for_repo_rejects_empty_after_strip() -> None:
    with pytest.raises(ValueError, match="does not yield a valid alias"):
        alias_for_repo("mlx-community/")


def test_alias_for_repo_rejects_nested_slash() -> None:
    with pytest.raises(ValueError, match="nested slashes"):
        alias_for_repo("org/sub/repo")


def test_alias_for_repo_rejects_invalid_chars() -> None:
    with pytest.raises(ValueError, match="does not yield a valid alias"):
        alias_for_repo("mlx-community/has spaces")


def test_header_comment_present_and_parseable(tmp_path: Path) -> None:
    target = tmp_path / "models.toml"
    registry = Registry(
        schema_version=1,
        default_chat="gemma-4-4b-it-4bit",
        models={"gemma-4-4b-it-4bit": _chat_entry()},
    )
    write_registry(registry, path=target)
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# chirp models registry — schema_version 1")
    assert HEADER_COMMENT in text
    parsed = tomllib.loads(text)
    assert parsed["schema_version"] == 1
    assert parsed["default_chat"] == "gemma-4-4b-it-4bit"
    assert parsed["models"]["gemma-4-4b-it-4bit"]["hf_repo"] == (
        "mlx-community/gemma-4-4b-it-4bit"
    )


def test_unicode_options_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "models.toml"
    entry = RegistryEntry(
        hf_repo="mlx-community/gemma-4-4b-it-4bit",
        role="chat",
        options={
            "system_prompt": 'you are a助手 with "quotes" and \\backslashes\\',
        },
    )
    original = Registry(schema_version=1, models={"gemma-4-4b-it-4bit": entry})
    write_registry(original, path=target)
    loaded = read_registry(target)
    assert loaded == original
    assert loaded.models["gemma-4-4b-it-4bit"].options["system_prompt"] == (
        'you are a助手 with "quotes" and \\backslashes\\'
    )


def test_dotted_alias_round_trips_as_single_key(tmp_path: Path) -> None:
    """A dotted alias must stay one ``models`` key, not nest as a TOML table.

    ``_ALIAS_RE`` permits dots, and TOML treats an unquoted dotted key as a
    nesting operator — so this guards that ``tomli_w`` quotes the key on write.
    """
    target = tmp_path / "models.toml"
    alias = "qwen2.5-0.5b-instruct-4bit"
    original = Registry(schema_version=1, models={alias: _chat_entry()})
    write_registry(original, path=target)

    parsed = tomllib.loads(target.read_text(encoding="utf-8"))
    assert list(parsed["models"]) == [alias]
    assert read_registry(target) == original
