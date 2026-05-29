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

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

import typer
from rich.table import Table

from chirpd.paths import MODELS_TOML_PATH
from llm import hf
from llm.cli._console import console, stdout_console
from llm.cli._progress import RichProgressCallback
from llm.client import LLMClient
from llm.exceptions import (
    LLMDaemonUnreachable,
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

    This callback keeps ``models`` a multi-command group so Typer does not
    auto-promote a lone subcommand to the group root. ``add`` (4.3) and ``list``
    (4.4) live here; ``show``/``default``/``remove``/``pull`` follow in 4.5.
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


# TODO(4.5): include cache_path (huggingface_hub.try_to_load_from_cache) in the
# `show` command. Story 4.4 deferred it from the `list` view to keep the table
# narrow and the JSON schema minimal.


@dataclass(frozen=True)
class ListRow:
    """One rendered row of ``chirp models list`` — one registered alias."""

    alias: str
    role: ModelRole
    is_default: bool
    loaded: bool
    hf_repo: str


@app.command("list")
def list_command(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a JSON document on stdout instead of a table.",
    ),
) -> None:
    """List registered models with role, default, and daemon-loaded state."""
    registry = _read_registry()
    daemon_reachable, loaded_aliases = _query_loaded_state()
    rows = _compose_rows(registry, loaded_aliases)
    if json_output:
        _render_list_json(registry, rows, daemon_reachable)
    else:
        _render_list_table(rows, daemon_reachable)


def _query_loaded_state() -> tuple[bool, set[str]]:
    """Return ``(daemon_reachable, loaded_aliases)`` without spawning a daemon.

    ``list`` is a diagnostic command, so a missing daemon is a soft fail: we
    report loaded state as unknown rather than lazy-spawning one (which would
    mask the very "is the daemon running?" question ``list`` answers).
    """
    try:
        models = LLMClient().model_list_sync(spawn_if_absent=False)
    except LLMDaemonUnreachable:
        return False, set()
    loaded_aliases = {
        model["alias"]
        for model in models
        if isinstance(model.get("alias"), str) and model.get("loaded")
    }
    return True, loaded_aliases


def _compose_rows(registry: Registry, loaded_aliases: set[str]) -> list[ListRow]:
    """Build the sorted render rows from the registry and loaded-alias set.

    Sorted by ``(role, alias)`` ascending — ``chat`` sorts before ``embed`` —
    so the table order is stable across runs.
    """
    defaults = {
        alias
        for alias in (registry.default_chat, registry.default_embed)
        if alias is not None
    }
    rows = [
        ListRow(
            alias=alias,
            role=entry.role,
            is_default=alias in defaults,
            loaded=alias in loaded_aliases,
            hf_repo=entry.hf_repo,
        )
        for alias, entry in registry.models.items()
    ]
    rows.sort(key=lambda row: (row.role, row.alias))
    return rows


def _render_list_table(rows: list[ListRow], daemon_reachable: bool) -> None:
    if not rows:
        console.print(
            "No models registered. Run `chirp models add <hf-repo>` to get started.",
            markup=False,
            soft_wrap=True,
        )
        return
    if not daemon_reachable:
        console.print(
            "(daemon not running — loaded state unknown)",
            markup=False,
            soft_wrap=True,
        )
    table = Table(title="Models")
    for column in ("alias", "role", "default", "loaded", "hf_repo"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row.alias,
            row.role,
            "★" if row.is_default else "",
            "●" if daemon_reachable and row.loaded else "—",
            row.hf_repo,
        )
    stdout_console.print(table)


def _render_list_json(
    registry: Registry, rows: list[ListRow], daemon_reachable: bool
) -> None:
    payload: dict[str, Any] = {
        "schema_version": registry.schema_version,
        "default_chat": registry.default_chat,
        "default_embed": registry.default_embed,
        "models": [
            {
                "alias": row.alias,
                "role": row.role,
                "default": row.is_default,
                "loaded": row.loaded if daemon_reachable else None,
                "hf_repo": row.hf_repo,
            }
            for row in rows
        ],
        "daemon_reachable": daemon_reachable,
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")


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
