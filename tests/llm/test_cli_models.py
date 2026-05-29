"""Tests for the ``chirp models add`` subcommand (story 4.3).

Every I/O boundary is mocked: :mod:`llm.hf` (HuggingFace), :class:`LLMClient`
(daemon warm), and the registry path is redirected to ``tmp_path`` via the
``CHIRP_REGISTRY_PATH`` env var. The suite runs on Linux CI with no MLX,
network, or daemon subprocess.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from llm import hf
from llm.cli import models as models_module
from llm.cli._progress import RichProgressCallback
from llm.exceptions import LLMDaemonUnreachable, LLMModelLoadFailed
from llm.registry import (
    Registry,
    RegistryEntry,
    alias_for_repo,
    read_registry,
    write_registry,
)

CHAT_REPO = "mlx-community/gemma-4-4b-it-4bit"
EMBED_REPO = "mlx-community/bge-small-en-v1.5"
AMBIGUOUS_REPO = "mlx-community/mystery-model"

runner = CliRunner()


def _chat_metadata(repo: str = CHAT_REPO) -> hf.HfRepoMetadata:
    return hf.HfRepoMetadata(
        repo_id=repo,
        tags=["text-generation"],
        siblings_count=3,
        architectures=["GemmaForCausalLM"],
        sha="deadbeef",
    )


def _ambiguous_metadata(repo: str = AMBIGUOUS_REPO) -> hf.HfRepoMetadata:
    return hf.HfRepoMetadata(
        repo_id=repo,
        tags=[],
        siblings_count=1,
        architectures=["BertModel"],
        sha="deadbeef",
    )


@pytest.fixture
def registry_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models.toml"
    monkeypatch.setenv("CHIRP_REGISTRY_PATH", str(path))
    return path


def _seed_registry(path: Path, registry: Registry) -> None:
    write_registry(registry, path=path)


def _mock_hf(
    monkeypatch: pytest.MonkeyPatch,
    *,
    validate: object,
    download: object | None = None,
) -> MagicMock:
    if isinstance(validate, Exception):
        validate_mock = MagicMock(side_effect=validate)
    else:
        validate_mock = MagicMock(return_value=validate)
    monkeypatch.setattr(hf, "validate_repo", validate_mock)

    if isinstance(download, Exception):
        download_mock = MagicMock(side_effect=download)
    else:
        download_mock = MagicMock(return_value=download)
    monkeypatch.setattr(hf, "download_model", download_mock)
    return download_mock


def _mock_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    side_effect: Exception | None = None,
) -> MagicMock:
    client = MagicMock()
    if side_effect is not None:
        client.model_load_sync.side_effect = side_effect
    else:
        client.model_load_sync.return_value = {"event": "ready"}
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(models_module, "LLMClient", factory)
    return factory


def test_add_first_chat_model_sets_default_and_warms(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    alias = alias_for_repo(CHAT_REPO)
    assert alias in registry.models
    assert registry.models[alias].role == "chat"
    assert registry.default_chat == alias
    assert "Ready." in result.stderr


def test_add_second_chat_model_does_not_promote_default(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_registry(
        registry_path,
        Registry(
            schema_version=1,
            default_chat="first",
            models={"first": RegistryEntry(hf_repo="org/first", role="chat")},
        ),
    )
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    alias = alias_for_repo(CHAT_REPO)
    assert alias in registry.models
    assert registry.default_chat == "first"


def test_add_with_alias_flag_overrides_inferred(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO, "--alias", "custom"])

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    assert "custom" in registry.models
    assert alias_for_repo(CHAT_REPO) not in registry.models


def test_add_with_role_flag_overrides_inferred(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_ambiguous_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(
        models_module.app, ["add", AMBIGUOUS_REPO, "--role", "embed"]
    )

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    alias = alias_for_repo(AMBIGUOUS_REPO)
    assert registry.models[alias].role == "embed"
    assert registry.default_embed == alias


def test_add_no_warm_skips_model_load(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    factory = _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO, "--no-warm"])

    assert result.exit_code == 0
    factory.assert_not_called()
    registry = read_registry(path=registry_path)
    assert alias_for_repo(CHAT_REPO) in registry.models


def test_add_repo_not_found_exits_5_no_registry_write(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(
        monkeypatch,
        validate=hf.HfRepoNotFound(CHAT_REPO),
        download=MagicMock(),
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 5
    assert not registry_path.exists()


def test_add_network_error_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(
        monkeypatch,
        validate=hf.HfNetworkError(CHAT_REPO, original=RuntimeError("boom")),
        download=MagicMock(),
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1
    assert not registry_path.exists()


def test_add_role_ambiguous_exits_2_no_registry_write(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_ambiguous_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", AMBIGUOUS_REPO])

    assert result.exit_code == 2
    assert not registry_path.exists()
    assert "Pass --role" in result.stderr


def test_add_download_failed_exits_1_no_registry_write(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(
        monkeypatch,
        validate=_chat_metadata(),
        download=hf.HfDownloadFailed(CHAT_REPO, original=OSError("disk full")),
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1
    assert not registry_path.exists()


def test_add_warm_failed_preserves_registry_exit_4(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch, side_effect=LLMModelLoadFailed("unsupported arch"))

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 4
    registry = read_registry(path=registry_path)
    assert alias_for_repo(CHAT_REPO) in registry.models
    assert "chirp models pull" in result.stderr
    assert "chirp daemon logs" in result.stderr


def test_add_warm_failed_via_transport_preserves_registry(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch, side_effect=LLMDaemonUnreachable("no socket"))

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 4
    registry = read_registry(path=registry_path)
    assert alias_for_repo(CHAT_REPO) in registry.models
    assert "daemon" in result.stderr


def test_add_unsupported_schema_version_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path.write_text("schema_version = 99\n", encoding="utf-8")
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1
    assert registry_path.read_text(encoding="utf-8") == "schema_version = 99\n"


def test_add_idempotent_second_run(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    first = runner.invoke(models_module.app, ["add", CHAT_REPO])
    second = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert first.exit_code == 0
    assert second.exit_code == 0
    registry = read_registry(path=registry_path)
    alias = alias_for_repo(CHAT_REPO)
    assert list(registry.models) == [alias]
    assert registry.default_chat == alias


def test_add_role_change_warning(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias = alias_for_repo(CHAT_REPO)
    _seed_registry(
        registry_path,
        Registry(
            schema_version=1,
            default_chat=alias,
            models={alias: RegistryEntry(hf_repo=CHAT_REPO, role="chat")},
        ),
    )
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO, "--role", "embed"])

    assert result.exit_code == 0
    assert "role changed from chat to embed" in result.stderr
    registry = read_registry(path=registry_path)
    assert registry.models[alias].role == "embed"
    assert registry.default_chat == alias


def test_add_alias_override_coexists_with_inferred(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    first = runner.invoke(models_module.app, ["add", CHAT_REPO])
    second = runner.invoke(models_module.app, ["add", CHAT_REPO, "--alias", "other"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    registry = read_registry(path=registry_path)
    inferred = alias_for_repo(CHAT_REPO)
    assert {inferred, "other"} <= set(registry.models)
    assert registry.models[inferred].hf_repo == CHAT_REPO
    assert registry.models["other"].hf_repo == CHAT_REPO


def test_add_writes_to_stderr_only(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 0
    assert result.stdout == ""


def test_add_validate_unexpected_error_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(
        monkeypatch,
        validate=hf.HfUnexpectedError(CHAT_REPO, original=RuntimeError("weird")),
        download=MagicMock(),
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1
    assert not registry_path.exists()


def test_add_uninferable_alias_exits_2(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested_repo = "org/team/model"
    _mock_hf(
        monkeypatch,
        validate=_chat_metadata(nested_repo),
        download=MagicMock(),
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", nested_repo])

    assert result.exit_code == 2
    assert "Pass --alias" in result.stderr
    assert not registry_path.exists()


def test_add_download_repo_not_found_exits_5(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(
        monkeypatch,
        validate=_chat_metadata(),
        download=hf.HfRepoNotFound(CHAT_REPO),
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 5
    assert not registry_path.exists()


def test_add_download_network_error_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(
        monkeypatch,
        validate=_chat_metadata(),
        download=hf.HfNetworkError(CHAT_REPO, original=RuntimeError("503")),
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1
    assert not registry_path.exists()


def test_add_download_unexpected_error_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(
        monkeypatch,
        validate=_chat_metadata(),
        download=hf.HfUnexpectedError(CHAT_REPO, original=RuntimeError("weird")),
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1
    assert not registry_path.exists()


def test_add_malformed_registry_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path.write_text("this is not = valid = toml", encoding="utf-8")
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1


def test_add_invalid_explicit_alias_exits_2(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)

    result = runner.invoke(
        models_module.app, ["add", CHAT_REPO, "--alias", "Bad Alias"]
    )

    assert result.exit_code == 2
    assert not registry_path.exists()


def test_add_registry_write_error_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm.registry import RegistryWriteError

    _mock_hf(monkeypatch, validate=_chat_metadata(), download=MagicMock())
    _mock_client(monkeypatch)
    monkeypatch.setattr(
        models_module,
        "write_registry",
        MagicMock(side_effect=RegistryWriteError("permission denied")),
    )

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1
    assert "Could not write registry" in result.stderr


def test_rich_progress_callback_non_tty_emits_status_lines() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False)
    callback = RichProgressCallback(CHAT_REPO, console=console)

    callback.on_start(100)
    callback.on_progress(50, 100)
    callback.on_done()

    output = buffer.getvalue()
    assert f"Downloading {CHAT_REPO}..." in output
    assert f"Downloaded {CHAT_REPO} (50 bytes)" in output


def test_rich_progress_callback_tty_uses_progress_bar() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    callback = RichProgressCallback(CHAT_REPO, console=console)

    callback.on_start(100)
    callback.on_progress(50, 100)
    callback.on_done()

    assert f"Downloading {CHAT_REPO}" in buffer.getvalue()
