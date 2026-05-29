"""Tests for the ``chirp models add`` subcommand (story 4.3).

Every I/O boundary is mocked: :mod:`llm.hf` (HuggingFace), :class:`LLMClient`
(daemon warm), and the registry path is redirected to ``tmp_path`` via the
``CHIRP_REGISTRY_PATH`` env var. The suite runs on Linux CI with no MLX,
network, or daemon subprocess.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from llm import hf
from llm.cli import models as models_module
from llm.cli._progress import RichProgressCallback
from llm.exceptions import (
    LLMConnectionLost,
    LLMDaemonUnreachable,
    LLMMalformedResponse,
    LLMModelLoadFailed,
    LLMVersionMismatch,
)
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
    assert registry.default_embed is None


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


def _mock_list_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: list[dict[str, object]] | None = None,
    unreachable: bool = False,
    side_effect: Exception | None = None,
) -> MagicMock:
    client = MagicMock()
    if side_effect is not None:
        client.model_list_sync.side_effect = side_effect
    elif unreachable:
        client.model_list_sync.side_effect = LLMDaemonUnreachable("no socket")
    else:
        client.model_list_sync.return_value = models or []
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(models_module, "LLMClient", factory)
    return client


@pytest.fixture
def wide_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Widen the stdout table console so cells are not ellipsized in capture."""
    monkeypatch.setattr(models_module, "stdout_console", Console(width=200))


def _seed_chat_model(path: Path, alias: str = "gemma-4-4b-it-4bit") -> None:
    _seed_registry(
        path,
        Registry(
            schema_version=1,
            default_chat=alias,
            models={alias: RegistryEntry(hf_repo=CHAT_REPO, role="chat")},
        ),
    )


def _seed_chat_and_embed(path: Path) -> None:
    _seed_registry(
        path,
        Registry(
            schema_version=1,
            default_chat="gemma-4-4b-it-4bit",
            default_embed="bge-small-en-v1.5",
            models={
                "gemma-4-4b-it-4bit": RegistryEntry(hf_repo=CHAT_REPO, role="chat"),
                "bge-small-en-v1.5": RegistryEntry(hf_repo=EMBED_REPO, role="embed"),
            },
        ),
    )


def test_list_empty_registry_tty(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_list_client(monkeypatch, models=[])

    result = runner.invoke(models_module.app, ["list"])

    assert result.exit_code == 0
    assert "chirp models add" in result.stderr
    assert result.stdout == ""


def test_list_empty_registry_json(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_list_client(monkeypatch, models=[])

    result = runner.invoke(models_module.app, ["list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["models"] == []
    assert payload["default_chat"] is None
    assert payload["daemon_reachable"] is True


def test_list_one_chat_model_tty(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch, wide_table: None
) -> None:
    _seed_chat_model(registry_path)
    _mock_list_client(
        monkeypatch, models=[{"alias": "gemma-4-4b-it-4bit", "loaded": True}]
    )

    result = runner.invoke(models_module.app, ["list"])

    assert result.exit_code == 0
    assert "gemma-4-4b-it-4bit" in result.stdout
    assert "★" in result.stdout
    assert "●" in result.stdout


def test_list_one_chat_model_json(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    _mock_list_client(
        monkeypatch, models=[{"alias": "gemma-4-4b-it-4bit", "loaded": True}]
    )

    result = runner.invoke(models_module.app, ["list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["default_chat"] == "gemma-4-4b-it-4bit"
    assert len(payload["models"]) == 1
    model = payload["models"][0]
    assert model["alias"] == "gemma-4-4b-it-4bit"
    assert model["role"] == "chat"
    assert model["default"] is True
    assert model["loaded"] is True


def test_list_chat_and_embed_tty(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch, wide_table: None
) -> None:
    _seed_chat_and_embed(registry_path)
    _mock_list_client(
        monkeypatch, models=[{"alias": "bge-small-en-v1.5", "loaded": True}]
    )

    result = runner.invoke(models_module.app, ["list"])

    assert result.exit_code == 0
    assert "gemma-4-4b-it-4bit" in result.stdout
    assert "bge-small-en-v1.5" in result.stdout
    assert result.stdout.count("★") == 2


def test_list_chat_and_embed_json(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_and_embed(registry_path)
    _mock_list_client(
        monkeypatch, models=[{"alias": "bge-small-en-v1.5", "loaded": True}]
    )

    result = runner.invoke(models_module.app, ["list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    by_alias = {model["alias"]: model for model in payload["models"]}
    assert by_alias["gemma-4-4b-it-4bit"]["loaded"] is False
    assert by_alias["bge-small-en-v1.5"]["loaded"] is True
    assert by_alias["bge-small-en-v1.5"]["role"] == "embed"
    assert payload["default_embed"] == "bge-small-en-v1.5"


def test_list_daemon_unreachable_tty(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch, wide_table: None
) -> None:
    _seed_chat_model(registry_path)
    _mock_list_client(monkeypatch, unreachable=True)

    result = runner.invoke(models_module.app, ["list"])

    assert result.exit_code == 0
    assert "daemon not running" in result.stderr
    assert "●" not in result.stdout
    assert "—" in result.stdout


def test_list_daemon_unreachable_json(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    _mock_list_client(monkeypatch, unreachable=True)

    result = runner.invoke(models_module.app, ["list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["daemon_reachable"] is False
    assert all(model["loaded"] is None for model in payload["models"])


@pytest.mark.parametrize(
    "error",
    [
        LLMVersionMismatch("client 2 != daemon 1"),
        LLMConnectionLost("broken pipe mid-request"),
        LLMMalformedResponse("daemon sent junk"),
    ],
)
def test_list_soft_fails_on_transport_or_protocol_error(
    registry_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """A reachable-but-broken daemon must not crash the diagnostic command.

    ``list`` only catches ``LLMDaemonUnreachable`` historically; a version
    mismatch / dropped connection / malformed reply should be treated the
    same way — loaded state unknown, exit 0 — not surface a raw traceback.
    """
    _seed_chat_model(registry_path)
    _mock_list_client(monkeypatch, side_effect=error)

    result = runner.invoke(models_module.app, ["list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["daemon_reachable"] is False
    assert all(model["loaded"] is None for model in payload["models"])


def test_list_does_not_spawn_daemon(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    client = _mock_list_client(monkeypatch, models=[])

    result = runner.invoke(models_module.app, ["list"])

    assert result.exit_code == 0
    client.model_list_sync.assert_called_once_with(spawn_if_absent=False)


def test_list_unsupported_schema_version_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path.write_text("schema_version = 99\n", encoding="utf-8")
    _mock_list_client(monkeypatch, models=[])

    result = runner.invoke(models_module.app, ["list"])

    assert result.exit_code == 1
    assert "schema version" in result.stderr


def test_list_json_is_jq_compatible(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    _mock_list_client(
        monkeypatch, models=[{"alias": "gemma-4-4b-it-4bit", "loaded": True}]
    )

    result = runner.invoke(models_module.app, ["list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "schema_version",
        "default_chat",
        "default_embed",
        "models",
        "daemon_reachable",
    }


def test_list_stdout_clean_in_json_mode(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    _mock_list_client(
        monkeypatch, models=[{"alias": "gemma-4-4b-it-4bit", "loaded": True}]
    )

    result = runner.invoke(models_module.app, ["list", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) is not None
    assert result.stdout.endswith("}\n")
    assert result.stderr == ""


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


def test_rich_progress_callback_close_stops_live_display_silently() -> None:
    """``close`` tears down a started bar idempotently and prints no success."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    callback = RichProgressCallback(CHAT_REPO, console=console)

    callback.on_start(100)
    assert callback._progress is not None

    callback.close()
    assert callback._progress is None
    callback.close()  # idempotent — must not raise on the second call

    assert f"Downloaded {CHAT_REPO}" not in buffer.getvalue()


def test_add_download_failure_tears_down_live_progress(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TTY download that fails after ``on_start`` must not leave the bar live.

    Otherwise the Rich live region keeps running with a hidden cursor and the
    error line can be overdrawn. ``_download``'s ``finally`` must stop it.
    """
    captured: dict[str, RichProgressCallback] = {}

    def make_callback(repo_id: str) -> RichProgressCallback:
        callback = RichProgressCallback(
            repo_id, console=Console(file=io.StringIO(), force_terminal=True)
        )
        captured["callback"] = callback
        return callback

    def failing_download(repo_id: str, *, progress: RichProgressCallback) -> None:
        progress.on_start(100)
        assert progress._progress is not None
        raise hf.HfNetworkError(repo_id, original=OSError("connection reset"))

    monkeypatch.setattr(models_module, "RichProgressCallback", make_callback)
    monkeypatch.setattr(hf, "validate_repo", MagicMock(return_value=_chat_metadata()))
    monkeypatch.setattr(hf, "download_model", failing_download)
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["add", CHAT_REPO])

    assert result.exit_code == 1
    assert captured["callback"]._progress is None


# --- story 4.5: show / default / remove / pull -----------------------------

CHAT_ALIAS = "gemma-4-4b-it-4bit"
EMBED_ALIAS = "bge-small-en-v1.5"


def _seed_two_chat(path: Path) -> None:
    _seed_registry(
        path,
        Registry(
            schema_version=1,
            default_chat="first",
            default_embed=EMBED_ALIAS,
            models={
                "first": RegistryEntry(hf_repo="org/first", role="chat"),
                "second": RegistryEntry(hf_repo="org/second", role="chat"),
                EMBED_ALIAS: RegistryEntry(hf_repo=EMBED_REPO, role="embed"),
            },
        ),
    )


def _mock_cache_path(monkeypatch: pytest.MonkeyPatch, value: Path | None) -> None:
    monkeypatch.setattr(hf, "resolved_cache_path", MagicMock(return_value=value))


def _make_download_result(
    *, cache_hit: bool = False, bytes_downloaded: int = 0
) -> hf.DownloadResult:
    return hf.DownloadResult(
        repo_id=CHAT_REPO,
        local_path=Path("/cache/snap"),
        bytes_downloaded=bytes_downloaded,
        cache_hit=cache_hit,
    )


# show -----------------------------------------------------------------------


def test_show_tty_renders_panel(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch, wide_table: None
) -> None:
    _seed_chat_model(registry_path)
    _mock_cache_path(monkeypatch, None)
    _mock_list_client(monkeypatch, models=[{"alias": CHAT_ALIAS, "loaded": True}])

    result = runner.invoke(models_module.app, ["show", CHAT_ALIAS])

    assert result.exit_code == 0
    assert CHAT_ALIAS in result.stdout
    assert CHAT_REPO in result.stdout
    assert "chat" in result.stdout


def test_show_json_emits_schema(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    _mock_cache_path(monkeypatch, None)
    _mock_list_client(monkeypatch, models=[{"alias": CHAT_ALIAS, "loaded": True}])

    result = runner.invoke(models_module.app, ["show", CHAT_ALIAS, "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "alias",
        "hf_repo",
        "role",
        "default",
        "loaded",
        "cache_path",
        "options",
        "daemon_reachable",
    }
    assert payload["alias"] == CHAT_ALIAS
    assert payload["hf_repo"] == CHAT_REPO
    assert payload["role"] == "chat"
    assert payload["default"] is True
    assert payload["loaded"] is True
    assert payload["daemon_reachable"] is True


def test_show_renders_options_rows(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch, wide_table: None
) -> None:
    _seed_registry(
        registry_path,
        Registry(
            schema_version=1,
            default_chat=CHAT_ALIAS,
            models={
                CHAT_ALIAS: RegistryEntry(
                    hf_repo=CHAT_REPO, role="chat", options={"temperature": 0.7}
                )
            },
        ),
    )
    _mock_cache_path(monkeypatch, None)
    _mock_list_client(monkeypatch, models=[])

    tty = runner.invoke(models_module.app, ["show", CHAT_ALIAS])
    js = runner.invoke(models_module.app, ["show", CHAT_ALIAS, "--json"])

    assert tty.exit_code == 0
    assert "options.temperature" in tty.stdout
    assert "0.7" in tty.stdout
    assert js.exit_code == 0
    assert json.loads(js.stdout)["options"] == {"temperature": 0.7}


def test_show_unknown_alias_exits_5(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    _mock_cache_path(monkeypatch, None)
    _mock_list_client(monkeypatch, models=[])

    result = runner.invoke(models_module.app, ["show", "nope"])

    assert result.exit_code == 5
    assert "not registered" in result.stderr
    assert "chirp models list" in result.stderr


def test_show_daemon_unreachable_loaded_null_in_json(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    _mock_cache_path(monkeypatch, None)
    _mock_list_client(monkeypatch, unreachable=True)

    result = runner.invoke(models_module.app, ["show", CHAT_ALIAS, "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["daemon_reachable"] is False
    assert payload["loaded"] is None


def test_show_cache_path_null_when_not_cached(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    _mock_cache_path(monkeypatch, None)
    _mock_list_client(monkeypatch, models=[])

    result = runner.invoke(models_module.app, ["show", CHAT_ALIAS, "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["cache_path"] is None


def test_show_cache_path_resolved_when_cached(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    snapshot = Path("/cache/hub/models--mlx-community--gemma/snapshots/abc")
    _mock_cache_path(monkeypatch, snapshot)
    _mock_list_client(monkeypatch, models=[])

    result = runner.invoke(models_module.app, ["show", CHAT_ALIAS, "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["cache_path"] == str(snapshot)


# default --------------------------------------------------------------------


def test_default_flips_chat(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_two_chat(registry_path)

    result = runner.invoke(models_module.app, ["default", "second"])

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    assert registry.default_chat == "second"
    assert "Set second as default chat." in result.stderr


def test_default_flips_embed(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_registry(
        registry_path,
        Registry(
            schema_version=1,
            default_embed=None,
            models={EMBED_ALIAS: RegistryEntry(hf_repo=EMBED_REPO, role="embed")},
        ),
    )

    result = runner.invoke(models_module.app, ["default", EMBED_ALIAS])

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    assert registry.default_embed == EMBED_ALIAS
    assert "default embed" in result.stderr


def test_default_unknown_alias_exits_5(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)

    result = runner.invoke(models_module.app, ["default", "nope"])

    assert result.exit_code == 5
    assert "not registered" in result.stderr


def test_default_invalid_role_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    monkeypatch.setattr(
        models_module,
        "set_default_for_role",
        MagicMock(side_effect=ValueError("unsupported role")),
    )

    result = runner.invoke(models_module.app, ["default", CHAT_ALIAS])

    assert result.exit_code == 1
    assert "invalid role" in result.stderr


def test_default_does_not_warm(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_two_chat(registry_path)
    factory = _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["default", "second"])

    assert result.exit_code == 0
    factory.assert_not_called()


def test_default_does_not_modify_other_default(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_two_chat(registry_path)

    result = runner.invoke(models_module.app, ["default", "second"])

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    assert registry.default_chat == "second"
    assert registry.default_embed == EMBED_ALIAS


# remove ---------------------------------------------------------------------


def test_remove_drops_entry(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_and_embed(registry_path)

    result = runner.invoke(models_module.app, ["remove", EMBED_ALIAS])

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    assert EMBED_ALIAS not in registry.models
    assert CHAT_ALIAS in registry.models
    assert "Removed bge-small-en-v1.5." in result.stderr


def test_remove_clears_default_when_removed_alias_was_default(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)

    result = runner.invoke(models_module.app, ["remove", CHAT_ALIAS])

    assert result.exit_code == 0
    registry = read_registry(path=registry_path)
    assert registry.default_chat is None
    assert CHAT_ALIAS not in registry.models


def test_remove_unknown_alias_exits_5(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)

    result = runner.invoke(models_module.app, ["remove", "nope"])

    assert result.exit_code == 5
    assert "not registered" in result.stderr


def test_remove_purge_deletes_cache_dir(
    registry_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    hub_root = tmp_path / "hub"
    cache_dir = hub_root / "models--mlx-community--gemma-4-4b-it-4bit"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(hf, "hf_hub_cache_root", lambda: hub_root)
    monkeypatch.setattr(hf, "cache_dir_for_repo", lambda repo: cache_dir)
    rmtree = MagicMock()
    monkeypatch.setattr(models_module.shutil, "rmtree", rmtree)

    result = runner.invoke(models_module.app, ["remove", CHAT_ALIAS, "--purge"])

    assert result.exit_code == 0
    rmtree.assert_called_once_with(cache_dir.resolve(), ignore_errors=False)
    assert "purged cache" in result.stderr


def test_remove_purge_warns_if_cache_missing(
    registry_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    hub_root = tmp_path / "hub"
    cache_dir = hub_root / "models--mlx-community--gemma-4-4b-it-4bit"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(hf, "hf_hub_cache_root", lambda: hub_root)
    monkeypatch.setattr(hf, "cache_dir_for_repo", lambda repo: cache_dir)
    monkeypatch.setattr(
        models_module.shutil,
        "rmtree",
        MagicMock(side_effect=FileNotFoundError("gone")),
    )

    result = runner.invoke(models_module.app, ["remove", CHAT_ALIAS, "--purge"])

    assert result.exit_code == 0
    assert "Warning" in result.stderr
    assert "not found" in result.stderr
    registry = read_registry(path=registry_path)
    assert CHAT_ALIAS not in registry.models


def test_remove_purge_warns_on_permission_error(
    registry_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    hub_root = tmp_path / "hub"
    cache_dir = hub_root / "models--mlx-community--gemma-4-4b-it-4bit"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(hf, "hf_hub_cache_root", lambda: hub_root)
    monkeypatch.setattr(hf, "cache_dir_for_repo", lambda repo: cache_dir)
    monkeypatch.setattr(
        models_module.shutil,
        "rmtree",
        MagicMock(side_effect=PermissionError("denied")),
    )

    result = runner.invoke(models_module.app, ["remove", CHAT_ALIAS, "--purge"])

    assert result.exit_code == 0
    assert "Warning" in result.stderr
    assert str(cache_dir.resolve()) in result.stderr
    assert CHAT_ALIAS not in read_registry(path=registry_path).models


def test_remove_purge_refuses_path_outside_hf_cache(
    registry_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    hub_root = tmp_path / "hub"
    hub_root.mkdir()
    outside = tmp_path / "outside"
    monkeypatch.setattr(hf, "hf_hub_cache_root", lambda: hub_root)
    monkeypatch.setattr(hf, "cache_dir_for_repo", lambda repo: outside)
    rmtree = MagicMock()
    monkeypatch.setattr(models_module.shutil, "rmtree", rmtree)

    result = runner.invoke(models_module.app, ["remove", CHAT_ALIAS, "--purge"])

    assert result.exit_code == 1
    rmtree.assert_not_called()
    assert "outside" in result.stderr


def test_remove_purge_refuses_when_cache_dir_equals_hub_root(
    registry_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the degenerate case where the repo folder resolves to the root.

    ``Path.is_relative_to`` returns True for equal paths, so a guard that only
    checks ``is_relative_to`` would let an empty/degenerate repo folder rmtree
    the entire HF cache root. The equal-path case must be refused.
    """
    _seed_chat_model(registry_path)
    hub_root = tmp_path / "hub"
    hub_root.mkdir()
    monkeypatch.setattr(hf, "hf_hub_cache_root", lambda: hub_root)
    monkeypatch.setattr(hf, "cache_dir_for_repo", lambda repo: hub_root)
    rmtree = MagicMock()
    monkeypatch.setattr(models_module.shutil, "rmtree", rmtree)

    result = runner.invoke(models_module.app, ["remove", CHAT_ALIAS, "--purge"])

    assert result.exit_code == 1
    rmtree.assert_not_called()


def test_remove_does_not_call_daemon(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_and_embed(registry_path)
    factory = _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["remove", EMBED_ALIAS])

    assert result.exit_code == 0
    factory.assert_not_called()


# pull -----------------------------------------------------------------------


def test_pull_redownloads_and_warms(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    download = MagicMock(return_value=_make_download_result(bytes_downloaded=123))
    monkeypatch.setattr(hf, "download_model", download)
    factory = _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["pull", CHAT_ALIAS])

    assert result.exit_code == 0
    download.assert_called_once()
    factory.return_value.model_load_sync.assert_called_once_with(CHAT_ALIAS, "chat")
    assert "Pulled gemma-4-4b-it-4bit (123 bytes)." in result.stderr
    assert "Warmed gemma-4-4b-it-4bit." in result.stderr


def test_pull_no_warm_skips_model_load(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    monkeypatch.setattr(
        hf,
        "download_model",
        MagicMock(return_value=_make_download_result(cache_hit=True)),
    )
    factory = _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["pull", CHAT_ALIAS, "--no-warm"])

    assert result.exit_code == 0
    factory.assert_not_called()
    assert "cache hit" in result.stderr
    assert "Skipped warm" in result.stderr


def test_pull_unknown_alias_exits_5(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    download = MagicMock()
    monkeypatch.setattr(hf, "download_model", download)

    result = runner.invoke(models_module.app, ["pull", "nope"])

    assert result.exit_code == 5
    download.assert_not_called()


def test_pull_download_failed_exits_1(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    monkeypatch.setattr(
        hf,
        "download_model",
        MagicMock(side_effect=hf.HfDownloadFailed(CHAT_REPO, original=OSError("io"))),
    )
    factory = _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["pull", CHAT_ALIAS])

    assert result.exit_code == 1
    factory.assert_not_called()


def test_pull_does_not_modify_registry(
    registry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_chat_model(registry_path)
    before = registry_path.read_bytes()
    monkeypatch.setattr(
        hf, "download_model", MagicMock(return_value=_make_download_result())
    )
    _mock_client(monkeypatch)

    result = runner.invoke(models_module.app, ["pull", CHAT_ALIAS])

    assert result.exit_code == 0
    assert registry_path.read_bytes() == before
