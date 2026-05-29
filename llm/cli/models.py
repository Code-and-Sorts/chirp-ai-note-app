"""``chirp models`` Typer subcommands.

This module is the eventual home of all six ``chirp models`` subcommands.
Story 4.3 adds the first — ``add`` — which pulls together the three pieces of
plumbing the model-registry epic introduces: HuggingFace validation/download
(:mod:`llm.hf`), atomic registry writes (:mod:`llm.registry`), and daemon warm
(:class:`llm.client.LLMClient`).

The execution order is fixed (epic §3 decision 8): validate → resolve role →
resolve alias → download → read registry → mutate → write → warm. Any failure
before the registry write aborts without touching ``models.toml``. A failed
warm leaves the registry write intact — the model is registered the moment its
weights land, and the user retries the warm with ``chirp models pull``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, NoReturn

import typer

from chirpd.paths import MODELS_TOML_PATH
from llm import hf
from llm.cli._console import console
from llm.cli._progress import RichProgressCallback
from llm.client import LLMClient
from llm.exceptions import (
    LLMMalformedResponse,
    LLMModelError,
    LLMModelNotFound,
    LLMTransportError,
)
from llm.registry import (
    ModelRole,
    Registry,
    RegistryEntry,
    RegistryWriteError,
    alias_for_repo,
    read_registry,
    set_default_for_role,
    upsert_model,
    write_registry,
)

app = typer.Typer(name="models", help="Manage chirp's MLX model registry")


@app.callback()
def models_main() -> None:
    """Manage chirp's MLX model registry.

    This callback exists to keep ``models`` a multi-command group while ``add``
    is its only subcommand; without it Typer auto-promotes the lone command to
    the group root and ``chirp models add`` would stop parsing. Future
    subcommands (4.4/4.5) make it load-bearing.
    """


@app.command("add")
def add_command(
    hf_repo: str = typer.Argument(
        ..., help="HuggingFace repo id, e.g. mlx-community/gemma-4-4b-it-4bit."
    ),
    alias: str | None = typer.Option(
        None, "--alias", help="Override the alias inferred from the repo name."
    ),
    role: Literal["chat", "embed"] | None = typer.Option(
        None, "--role", help="Force the model role instead of inferring it."
    ),
    no_warm: bool = typer.Option(
        False,
        "--no-warm",
        help="Skip warming the model on the daemon after registering it.",
    ),
) -> None:
    """Download a model from HuggingFace, register it, and warm it on the daemon."""
    metadata = _validate_repo(hf_repo)
    resolved_role = _resolve_role(hf_repo, metadata, role)
    resolved_alias = _resolve_alias(hf_repo, alias)
    _download(hf_repo, resolved_alias)

    registry = _read_registry()
    previous_entry = registry.models.get(resolved_alias)
    default_was_unset = _default_unset_for_role(registry, resolved_role)
    registry = _mutate(registry, resolved_alias, hf_repo, resolved_role)
    if default_was_unset:
        registry = set_default_for_role(registry, resolved_alias)
    _write(registry)

    _warn_on_role_change(resolved_alias, previous_entry, resolved_role)

    if not no_warm:
        _warm(resolved_alias, resolved_role)


def _validate_repo(hf_repo: str) -> hf.HfRepoMetadata:
    try:
        return hf.validate_repo(hf_repo)
    except hf.HfRepoNotFound:
        _exit(
            f"Error: HuggingFace repo not found: {hf_repo}. "
            "Check the repo name and try again.",
            5,
        )
    except hf.HfNetworkError as err:
        _exit(
            f"Error: Could not reach HuggingFace ({err.original}). "
            "Check your connection and try again.",
            1,
        )
    except hf.HfError as err:
        _exit(f"Error: HuggingFace error validating {hf_repo}: {err}.", 1)


def _resolve_role(
    hf_repo: str,
    metadata: hf.HfRepoMetadata,
    role: Literal["chat", "embed"] | None,
) -> ModelRole:
    if role is not None:
        return role
    try:
        return hf.infer_role(metadata)
    except hf.RoleInferenceAmbiguous:
        _exit(
            f"Error: Could not infer role for {hf_repo}. "
            "Pass --role chat or --role embed explicitly.",
            2,
        )


def _resolve_alias(hf_repo: str, alias: str | None) -> str:
    if alias is not None:
        return alias
    try:
        return alias_for_repo(hf_repo)
    except ValueError:
        _exit(
            f"Error: Could not infer alias from {hf_repo}. Pass --alias <name> "
            "explicitly (lowercase letters, digits, dot, dash, underscore).",
            2,
        )


def _download(hf_repo: str, alias: str) -> None:
    callback = RichProgressCallback(hf_repo)
    try:
        hf.download_model(hf_repo, progress=callback)
    except hf.HfRepoNotFound:
        _exit(
            f"Error: HuggingFace repo not found: {hf_repo}. "
            "Check the repo name and try again.",
            5,
        )
    except hf.HfNetworkError as err:
        _exit(
            f"Error: Could not reach HuggingFace ({err.original}). "
            "Check your connection and try again.",
            1,
        )
    except hf.HfDownloadFailed as err:
        _exit(
            f"Error: Download failed: {err.original}. Try again, or run "
            f"`chirp models pull {alias}` after fixing the cause.",
            1,
        )
    except hf.HfError as err:
        _exit(f"Error: Download failed for {hf_repo}: {err}.", 1)


def _read_registry() -> Registry:
    try:
        return read_registry(path=_registry_path())
    except LLMModelNotFound as err:
        version = err.details.get("schema_version")
        _exit(
            f"Error: Existing models.toml has unsupported schema version "
            f"{version}. Delete {_registry_path_display()} and re-run.",
            1,
        )
    except LLMMalformedResponse as err:
        _exit(
            f"Error: Existing models.toml is not valid: {err}. "
            f"Delete {_registry_path_display()} and re-run.",
            1,
        )


def _mutate(registry: Registry, alias: str, hf_repo: str, role: ModelRole) -> Registry:
    entry = RegistryEntry(hf_repo=hf_repo, role=role, options={})
    try:
        return upsert_model(registry, alias, entry)
    except ValueError:
        _exit(
            f"Error: Invalid alias {alias!r}. Pass --alias <name> explicitly "
            "(lowercase letters, digits, dot, dash, underscore).",
            2,
        )


def _write(registry: Registry) -> None:
    try:
        write_registry(registry, path=_registry_path())
    except RegistryWriteError as err:
        _exit(
            f"Error: Could not write registry: {err}. Check directory permissions.",
            1,
        )


def _warm(alias: str, role: ModelRole) -> None:
    console.print(f"Warming {alias}...", markup=False, soft_wrap=True)
    try:
        LLMClient().model_load_sync(alias, role)
    except LLMModelError as err:
        _exit(
            f"Error: Model registered but warm failed: {err}. Run "
            f"`chirp models pull {alias}` to retry or `chirp daemon logs` "
            "to diagnose.",
            4,
        )
    except LLMTransportError as err:
        _exit(
            f"Error: Model registered but warm failed: the chirp daemon is "
            f"unreachable ({err}). Run `chirp models pull {alias}` to retry or "
            "`chirp daemon logs` to diagnose.",
            4,
        )
    console.print("Ready.", markup=False, soft_wrap=True)


def _warn_on_role_change(
    alias: str, previous_entry: RegistryEntry | None, role: ModelRole
) -> None:
    if previous_entry is not None and previous_entry.role != role:
        console.print(
            f"Note: alias {alias} role changed from {previous_entry.role} to "
            f"{role}; defaults unchanged — verify with `chirp models list`.",
            markup=False,
            soft_wrap=True,
        )


def _default_unset_for_role(registry: Registry, role: ModelRole) -> bool:
    if role == "chat":
        return registry.default_chat is None
    return registry.default_embed is None


def _registry_path() -> Path | None:
    override = os.environ.get("CHIRP_REGISTRY_PATH")
    return Path(override) if override else None


def _registry_path_display() -> Path:
    return _registry_path() or MODELS_TOML_PATH


def _exit(message: str, code: int) -> NoReturn:
    console.print(message, style="red", markup=False, soft_wrap=True)
    raise typer.Exit(code=code)
