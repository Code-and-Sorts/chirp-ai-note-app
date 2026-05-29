"""Tests for :mod:`llm.hf` — the HuggingFace boundary module.

All ``huggingface_hub`` calls are patched at the import boundary; no test
makes a real network call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from llm.hf import (
    DownloadResult,
    HfDownloadFailed,
    HfError,
    HfNetworkError,
    HfRepoMetadata,
    HfRepoNotFound,
    HfUnexpectedError,
    RoleInferenceAmbiguous,
    _status_code_from,
    cache_dir_for_repo,
    download_model,
    hf_hub_cache_root,
    infer_role,
    resolved_cache_path,
    validate_repo,
)


def _http_error(message: str, *, status: int = 503) -> HfHubHTTPError:
    response = MagicMock(status_code=status, text=message)
    return HfHubHTTPError(message, response=response)


def _repo_not_found(message: str) -> RepositoryNotFoundError:
    response = MagicMock(status_code=404, text=message)
    return RepositoryNotFoundError(message, response=response)


class _FakeSibling:
    pass


@dataclass
class _FakeRepoInfo:
    tags: list[str] = field(default_factory=list)
    siblings: list[_FakeSibling] = field(default_factory=list)
    sha: str | None = "deadbeef"


def _make_config_file(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# validate_repo
# ---------------------------------------------------------------------------


def test_validate_repo_returns_metadata(tmp_path: Path) -> None:
    info = _FakeRepoInfo(
        tags=["text-generation", "mlx"],
        siblings=[_FakeSibling(), _FakeSibling(), _FakeSibling()],
        sha="abc123",
    )
    config_path = _make_config_file(tmp_path, {"architectures": ["LlamaForCausalLM"]})

    with (
        patch("llm.hf.HfApi.repo_info", return_value=info),
        patch("llm.hf.hf_hub_download", return_value=str(config_path)),
    ):
        metadata = validate_repo("mlx-community/gemma-4-4b-it-4bit")

    assert metadata == HfRepoMetadata(
        repo_id="mlx-community/gemma-4-4b-it-4bit",
        tags=["text-generation", "mlx"],
        siblings_count=3,
        architectures=["LlamaForCausalLM"],
        sha="abc123",
    )


def test_validate_repo_not_found() -> None:
    with patch(
        "llm.hf.HfApi.repo_info",
        side_effect=_repo_not_found("missing"),
    ):
        with pytest.raises(HfRepoNotFound) as exc:
            validate_repo("org/missing")
    assert exc.value.repo_id == "org/missing"


def test_validate_repo_network_error() -> None:
    original = _http_error("503 Service Unavailable", status=503)
    with patch("llm.hf.HfApi.repo_info", side_effect=original):
        with pytest.raises(HfNetworkError) as exc:
            validate_repo("org/repo")
    assert exc.value.repo_id == "org/repo"
    assert exc.value.original is original
    assert "status 503" in str(exc.value)


def test_status_code_from_returns_none_when_response_missing() -> None:
    class _NoResponse(Exception):
        pass

    assert _status_code_from(_NoResponse("connection refused")) is None


def test_hf_network_error_message_omits_status_when_unavailable() -> None:
    err = HfNetworkError("org/repo", original=Exception("connection refused"))
    assert "status" not in str(err)


def test_validate_repo_wraps_unexpected_error() -> None:
    original = RuntimeError("unexpected hf state")
    with patch("llm.hf.HfApi.repo_info", side_effect=original):
        with pytest.raises(HfUnexpectedError) as exc:
            validate_repo("org/repo")
    assert exc.value.repo_id == "org/repo"
    assert exc.value.original is original
    assert isinstance(exc.value, HfError)


def test_validate_repo_wraps_unexpected_error_from_config_download() -> None:
    info = _FakeRepoInfo(tags=["text-generation"])
    original = ValueError("hf decided to validate weird")
    with (
        patch("llm.hf.HfApi.repo_info", return_value=info),
        patch("llm.hf.hf_hub_download", side_effect=original),
    ):
        with pytest.raises(HfUnexpectedError) as exc:
            validate_repo("org/repo")
    assert exc.value.original is original


def test_validate_repo_config_json_absent() -> None:
    info = _FakeRepoInfo(tags=["text-generation"], sha="z")
    with (
        patch("llm.hf.HfApi.repo_info", return_value=info),
        patch(
            "llm.hf.hf_hub_download",
            side_effect=EntryNotFoundError("no config.json"),
        ),
    ):
        metadata = validate_repo("org/no-config")

    assert metadata.architectures == []


def test_validate_repo_propagates_repo_not_found_from_config_download() -> None:
    info = _FakeRepoInfo(tags=["text-generation"])
    with (
        patch("llm.hf.HfApi.repo_info", return_value=info),
        patch(
            "llm.hf.hf_hub_download",
            side_effect=_repo_not_found("vanished"),
        ),
    ):
        with pytest.raises(HfRepoNotFound):
            validate_repo("org/vanished")


def test_validate_repo_propagates_network_error_from_config_download() -> None:
    info = _FakeRepoInfo(tags=["text-generation"])
    original = _http_error("502")
    with (
        patch("llm.hf.HfApi.repo_info", return_value=info),
        patch("llm.hf.hf_hub_download", side_effect=original),
    ):
        with pytest.raises(HfNetworkError) as exc:
            validate_repo("org/repo")
    assert exc.value.original is original


def test_validate_repo_ignores_corrupt_config_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("not json {", encoding="utf-8")
    info = _FakeRepoInfo()

    with (
        patch("llm.hf.HfApi.repo_info", return_value=info),
        patch("llm.hf.hf_hub_download", return_value=str(config_path)),
    ):
        metadata = validate_repo("org/corrupt")

    assert metadata.architectures == []


def test_validate_repo_ignores_non_list_architectures(tmp_path: Path) -> None:
    config_path = _make_config_file(tmp_path, {"architectures": "LlamaForCausalLM"})
    info = _FakeRepoInfo()
    with (
        patch("llm.hf.HfApi.repo_info", return_value=info),
        patch("llm.hf.hf_hub_download", return_value=str(config_path)),
    ):
        metadata = validate_repo("org/weird-config")
    assert metadata.architectures == []


# ---------------------------------------------------------------------------
# infer_role
# ---------------------------------------------------------------------------


def _metadata(
    repo_id: str = "org/repo",
    tags: list[str] | None = None,
    architectures: list[str] | None = None,
) -> HfRepoMetadata:
    return HfRepoMetadata(
        repo_id=repo_id,
        tags=tags or [],
        siblings_count=0,
        architectures=architectures or [],
        sha=None,
    )


def test_infer_role_embed_by_tag_sentence_transformers() -> None:
    assert infer_role(_metadata(tags=["sentence-transformers"])) == "embed"


def test_infer_role_embed_by_tag_feature_extraction() -> None:
    assert infer_role(_metadata(tags=["feature-extraction"])) == "embed"


def test_infer_role_embed_by_name_bge() -> None:
    assert infer_role(_metadata(repo_id="mlx-community/bge-small-en-v1.5")) == "embed"


@pytest.mark.parametrize(
    "repo_id",
    [
        "intfloat/e5-large-v2",
        "thenlper/gte-small",
        "sentence-transformers/all-mpnet-base-v2",
        "microsoft/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
    ],
)
def test_infer_role_embed_by_name_e5_gte_mpnet_minilm(repo_id: str) -> None:
    assert infer_role(_metadata(repo_id=repo_id)) == "embed"


def test_infer_role_chat_by_tag_text_generation() -> None:
    assert infer_role(_metadata(tags=["text-generation"])) == "chat"


def test_infer_role_chat_by_tag_conversational() -> None:
    assert infer_role(_metadata(tags=["conversational"])) == "chat"


def test_infer_role_chat_by_architecture_for_causal_lm() -> None:
    assert infer_role(_metadata(architectures=["LlamaForCausalLM"])) == "chat"


def test_infer_role_chat_by_architecture_for_conditional_generation() -> None:
    assert infer_role(_metadata(architectures=["T5ForConditionalGeneration"])) == "chat"


def test_infer_role_bare_bert_model_falls_through_to_ambiguous() -> None:
    with pytest.raises(RoleInferenceAmbiguous):
        infer_role(_metadata(repo_id="org/some-bert", architectures=["BertModel"]))


def test_infer_role_ambiguous_raises() -> None:
    with pytest.raises(RoleInferenceAmbiguous) as exc:
        infer_role(_metadata(repo_id="org/generic-name", architectures=["Whatever"]))
    assert exc.value.repo_id == "org/generic-name"
    assert "--role" in exc.value.hint


def test_infer_role_priority_embed_tag_beats_chat_arch() -> None:
    metadata = _metadata(
        tags=["sentence-transformers"],
        architectures=["LlamaForCausalLM"],
    )
    assert infer_role(metadata) == "embed"


# ---------------------------------------------------------------------------
# download_model
# ---------------------------------------------------------------------------


@dataclass
class _RecordingCallback:
    starts: list[int | None] = field(default_factory=list)
    progress: list[tuple[int, int | None]] = field(default_factory=list)
    dones: int = 0

    def on_start(self, total_bytes: int | None) -> None:
        self.starts.append(total_bytes)

    def on_progress(self, bytes_downloaded: int, total_bytes: int | None) -> None:
        self.progress.append((bytes_downloaded, total_bytes))

    def on_done(self) -> None:
        self.dones += 1


def test_download_model_returns_result(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    with patch("llm.hf.snapshot_download", return_value=str(snapshot_dir)):
        result = download_model("mlx-community/gemma-4-4b-it-4bit")

    assert result == DownloadResult(
        repo_id="mlx-community/gemma-4-4b-it-4bit",
        local_path=snapshot_dir,
        bytes_downloaded=0,
        cache_hit=True,
    )


def test_download_model_repo_not_found() -> None:
    with patch(
        "llm.hf.snapshot_download",
        side_effect=_repo_not_found("gone"),
    ):
        with pytest.raises(HfRepoNotFound):
            download_model("org/missing")


def test_download_model_os_error_disk_full() -> None:
    original = OSError("No space left on device")
    with patch("llm.hf.snapshot_download", side_effect=original):
        with pytest.raises(HfDownloadFailed) as exc:
            download_model("org/repo")
    assert exc.value.original is original


def test_download_model_network_error() -> None:
    original = _http_error("503")
    with patch("llm.hf.snapshot_download", side_effect=original):
        with pytest.raises(HfNetworkError) as exc:
            download_model("org/repo")
    assert exc.value.original is original


def test_download_model_wraps_unexpected_error() -> None:
    original = RuntimeError("hf surprise")
    with patch("llm.hf.snapshot_download", side_effect=original):
        with pytest.raises(HfUnexpectedError) as exc:
            download_model("org/repo")
    assert exc.value.original is original


def test_download_model_progress_callback_invoked(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    recorder = _RecordingCallback()

    def fake_snapshot_download(
        *, repo_id: str, tqdm_class: type, **_kwargs: Any
    ) -> str:
        bar = tqdm_class(total=2048, unit="B")
        bar.update(1024)
        bar.update(1024)
        bar.close()
        return str(snapshot_dir)

    with patch("llm.hf.snapshot_download", side_effect=fake_snapshot_download):
        result = download_model("mlx-community/gemma-4-4b-it-4bit", progress=recorder)

    assert recorder.starts == [2048]
    assert recorder.progress == [(1024, 2048), (2048, 2048)]
    assert recorder.dones == 1
    assert result.bytes_downloaded == 2048
    assert result.cache_hit is False
    assert result.local_path == snapshot_dir


def test_download_model_mirrors_huggingface_two_bar_usage(tmp_path: Path) -> None:
    """Replay how huggingface_hub (as of 1.8.x) actually drives ``tqdm_class``.

    ``snapshot_download`` constructs the class twice: a shared byte bar
    (``unit="B"``) whose ``total`` is grown after construction as files are
    discovered, and a ``thread_map`` file-count bar that wraps an iterable.
    Only the byte bar should reach the callback; iterating the file-count bar
    must still yield every item so the real downloads run.
    """
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    recorder = _RecordingCallback()
    repo_files = ["model.safetensors", "config.json"]

    def fake_snapshot_download(
        *, repo_id: str, tqdm_class: type, **_kwargs: Any
    ) -> str:
        byte_bar = tqdm_class(total=0, initial=0, unit="B", unit_scale=True)
        byte_bar.total += 512  # hf grows the total as files are discovered
        downloaded = list(tqdm_class(iter(repo_files), total=len(repo_files)))
        byte_bar.update(256)
        byte_bar.update(256)
        byte_bar.close()
        assert downloaded == repo_files  # file-count bar must pass items through
        return str(snapshot_dir)

    with patch("llm.hf.snapshot_download", side_effect=fake_snapshot_download):
        result = download_model("mlx-community/two-bar", progress=recorder)

    assert recorder.starts == [None]  # byte total is 0 at construction time
    assert recorder.progress == [(256, 512), (512, 512)]
    assert recorder.dones == 1
    assert result.bytes_downloaded == 512
    assert result.cache_hit is False


def test_download_model_sums_bytes_across_multiple_byte_bars(tmp_path: Path) -> None:
    """Hardening: if huggingface_hub ever splits byte progress across more than
    one ``unit="B"`` bar, every bar's updates are summed — not just the first's
    — so bytes can't be undercounted into a false ``cache_hit``."""
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    recorder = _RecordingCallback()

    def fake_snapshot_download(
        *, repo_id: str, tqdm_class: type, **_kwargs: Any
    ) -> str:
        first = tqdm_class(total=100, unit="B")
        first.update(100)
        second = tqdm_class(total=200, unit="B")
        second.update(200)
        return str(snapshot_dir)

    with patch("llm.hf.snapshot_download", side_effect=fake_snapshot_download):
        result = download_model("mlx-community/multi-bar", progress=recorder)

    assert recorder.starts == [100]  # on_start fires once, for the first bar
    assert recorder.dones == 1
    assert result.bytes_downloaded == 300
    assert result.cache_hit is False


def test_download_model_cache_hit_reports_zero_bytes(tmp_path: Path) -> None:
    """Warm cache: the byte bar is still constructed (on_start/on_done fire),
    but no bytes transfer, so the result is flagged ``cache_hit``."""
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    recorder = _RecordingCallback()

    def fake_snapshot_download(
        *, repo_id: str, tqdm_class: type, **_kwargs: Any
    ) -> str:
        byte_bar = tqdm_class(total=0, initial=0, unit="B", unit_scale=True)
        byte_bar.close()
        return str(snapshot_dir)

    with patch("llm.hf.snapshot_download", side_effect=fake_snapshot_download):
        result = download_model("mlx-community/cached", progress=recorder)

    assert recorder.starts == [None]
    assert recorder.progress == []
    assert recorder.dones == 1
    assert result.bytes_downloaded == 0
    assert result.cache_hit is True


# --- story 4.5: cache-path helpers -----------------------------------------


def test_resolved_cache_path_returns_snapshot_dir_when_cached(tmp_path: Path) -> None:
    config_file = tmp_path / "snapshots" / "abc" / "config.json"
    with patch("llm.hf.try_to_load_from_cache", return_value=str(config_file)) as probe:
        result = resolved_cache_path("mlx-community/gemma-4-4b-it-4bit")

    probe.assert_called_once_with(
        repo_id="mlx-community/gemma-4-4b-it-4bit", filename="config.json"
    )
    assert result == config_file.parent


def test_resolved_cache_path_none_when_not_cached() -> None:
    with patch("llm.hf.try_to_load_from_cache", return_value=None):
        assert resolved_cache_path("mlx-community/gemma-4-4b-it-4bit") is None


def test_resolved_cache_path_none_when_known_absent_sentinel() -> None:
    with patch("llm.hf.try_to_load_from_cache", return_value=object()):
        assert resolved_cache_path("mlx-community/gemma-4-4b-it-4bit") is None


def test_cache_dir_for_repo_lives_under_hub_root() -> None:
    cache_dir = cache_dir_for_repo("mlx-community/gemma-4-4b-it-4bit")
    assert cache_dir.is_relative_to(hf_hub_cache_root())
    assert cache_dir.name == "models--mlx-community--gemma-4-4b-it-4bit"
