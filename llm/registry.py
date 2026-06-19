"""Typed model-registry I/O for ``models.toml``.

The reader half (``read_registry``, ``resolve_alias``) is consumed by the
daemon to resolve alias → repo on every ``model.load`` / ``model.list`` /
``model.status``. The writer half (``write_registry`` plus the mutation
helpers ``upsert_model`` / ``remove_model`` / ``set_default_for_role``) is
consumed by the ``chirp models`` Typer subcommands to atomically persist
registry mutations.
"""

from __future__ import annotations

import os
import re
import tomllib
import uuid
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, Field, ValidationError

from chirpd.paths import MODELS_TOML_PATH
from llm.exceptions import LLMError, LLMMalformedResponse, LLMModelNotFound

SUPPORTED_SCHEMA_VERSION = 1

ModelRole = Literal["chat", "embed"]

HEADER_COMMENT = (
    "# chirp models registry — schema_version 1\n"
    "# Managed by `chirp models {add,remove,default,pull}`. Hand edits OK;\n"
    "# re-run `chirp models list` to sanity-check. Alpha note: no automatic\n"
    "# migrations across schema_version changes — delete this file if a\n"
    "# future chirp release rejects it, then `chirp models add` to\n"
    "# re-register.\n"
)

_ALIAS_RE = re.compile(r"^[a-z0-9._-]+$")


class RegistryEntry(BaseModel):
    hf_repo: str
    role: ModelRole
    options: dict[str, Any] = Field(default_factory=dict)


class Registry(BaseModel):
    schema_version: int
    default_chat: str | None = None
    default_embed: str | None = None
    models: dict[str, RegistryEntry] = Field(default_factory=dict)


def read_registry(path: Path | None = None) -> Registry:
    """Parse the model registry TOML.

    Returns an empty registry when the file is absent so the daemon is runnable
    before the user adds any model. Raises :class:`LLMMalformedResponse` on
    invalid TOML or shape, and :class:`LLMModelNotFound` on an unsupported
    schema version (per the alpha-stage no-migrations rule).
    """
    target = path if path is not None else MODELS_TOML_PATH
    try:
        with target.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        return Registry(schema_version=SUPPORTED_SCHEMA_VERSION, models={})
    except tomllib.TOMLDecodeError as err:
        raise LLMMalformedResponse(
            f"models.toml at {target} is not valid TOML: {err}",
            details={"path": str(target), "error": str(err)},
        ) from err

    schema_version = raw.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise LLMModelNotFound(
            f"models.toml schema_version {schema_version!r} not supported; "
            "delete the file and re-register your models with `chirp models add`",
            details={
                "path": str(target),
                "schema_version": schema_version,
                "supported_schema_version": SUPPORTED_SCHEMA_VERSION,
            },
        )

    try:
        return Registry(**raw)
    except ValidationError as err:
        raise LLMMalformedResponse(
            f"models.toml at {target} failed validation: {err}",
            details={"path": str(target), "errors": err.errors()},
        ) from err


def resolved_chat_model(fallback: str | None = None, path: Path | None = None) -> str:
    """Best-effort display name for the active chat model.

    Returns the registry's configured default chat alias when present, so user
    surfaces (``chirp about``, ``chirp config --list``, generated ``meta.toml``)
    reflect the model that actually serves requests rather than the legacy
    ``settings.models.llm`` value. Falls back to ``fallback`` when the registry
    has no usable default and never raises — display sites must tolerate a
    missing or unreadable registry.
    """
    try:
        registry = read_registry(path)
    except LLMError:
        return fallback or "unset"
    alias = registry.default_chat
    if alias and alias in registry.models:
        return alias
    return fallback or "unset"


def resolve_alias(
    registry: Registry,
    identifier: str,
    role: ModelRole,
) -> RegistryEntry:
    """Resolve ``identifier`` to a registry entry per FR35/FR36 precedence."""
    if identifier == "default":
        default_alias = (
            registry.default_chat if role == "chat" else registry.default_embed
        )
        if default_alias is None:
            raise LLMModelNotFound(
                f"no default {role} model configured in registry",
                details={"role": role},
            )
        entry = registry.models.get(default_alias)
        if entry is None:
            raise LLMModelNotFound(
                f"default {role} alias {default_alias!r} is not present in registry",
                details={"role": role, "alias": default_alias},
            )
        _require_matching_role(default_alias, entry, role)
        return entry

    entry = registry.models.get(identifier)
    if entry is not None:
        _require_matching_role(identifier, entry, role)
        return entry

    if "/" in identifier:
        return RegistryEntry(hf_repo=identifier, role=role, options={})

    raise LLMModelNotFound(
        f"unknown model alias {identifier!r}",
        details={"identifier": identifier, "role": role},
    )


def _require_matching_role(
    alias: str, entry: RegistryEntry, requested_role: ModelRole
) -> None:
    if entry.role == requested_role:
        return
    raise LLMModelNotFound(
        f"alias {alias!r} is registered as {entry.role!r}, not {requested_role!r}",
        details={
            "alias": alias,
            "registered_role": entry.role,
            "requested_role": requested_role,
        },
    )


class RegistryWriteError(OSError):
    """Raised when the registry cannot be persisted to disk."""


def write_registry(registry: Registry, *, path: Path | None = None) -> None:
    """Atomically persist ``registry`` to ``path`` (or :data:`MODELS_TOML_PATH`).

    Renders TOML via :func:`tomli_w.dumps` with a stable :data:`HEADER_COMMENT`
    prepended, writes to a sibling temp file named uniquely per invocation
    (pid + random suffix, fsynced), then swaps via :func:`os.replace`. The
    unique name keeps concurrent writers from clobbering each other's temp
    file. A failure during the dump-or-replace sequence leaves any
    pre-existing registry file untouched and removes this writer's partial
    temp file so concurrent or retrying writers don't see stale state.

    A best-effort directory fsync runs after the rename succeeds; on macOS
    APFS this is documented as a no-op (Apple recommends ``F_FULLFSYNC`` on
    the file fd for true durability), so failures are swallowed rather than
    promoted to :class:`RegistryWriteError` — the registry has already been
    written by the time it runs.
    """
    target = path if path is not None else MODELS_TOML_PATH
    payload = registry.model_dump(exclude_none=True, mode="python")
    rendered = HEADER_COMMENT + tomli_w.dumps(payload)
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as handle:
            handle.write(rendered.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(target)
    except OSError as err:
        _safe_unlink(tmp_path)
        raise RegistryWriteError(
            f"failed to write models.toml at {target}: {err}"
        ) from err
    _fsync_directory_best_effort(target.parent)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Best-effort cleanup of the temp file; a failure here is intentionally
        # ignored so it can't mask the original write error being propagated.
        pass


def _fsync_directory_best_effort(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Directory fsync is best-effort durability (a documented no-op on
        # macOS APFS); the registry is already renamed into place, so ignore.
        return
    finally:
        os.close(fd)


def upsert_model(registry: Registry, alias: str, entry: RegistryEntry) -> Registry:
    """Return a copy of ``registry`` with ``alias`` set to ``entry``."""
    _require_valid_alias(alias)
    next_models = {**registry.models, alias: entry}
    return Registry(
        schema_version=registry.schema_version,
        default_chat=registry.default_chat,
        default_embed=registry.default_embed,
        models=next_models,
    )


def remove_model(registry: Registry, alias: str) -> Registry:
    """Return a copy of ``registry`` with ``alias`` removed.

    Clears the role's default if the removed alias was that default.
    """
    if alias not in registry.models:
        raise KeyError(alias)
    next_models = {k: v for k, v in registry.models.items() if k != alias}
    next_default_chat = (
        None if registry.default_chat == alias else registry.default_chat
    )
    next_default_embed = (
        None if registry.default_embed == alias else registry.default_embed
    )
    return Registry(
        schema_version=registry.schema_version,
        default_chat=next_default_chat,
        default_embed=next_default_embed,
        models=next_models,
    )


def set_default_for_role(registry: Registry, alias: str) -> Registry:
    """Return a copy of ``registry`` with the role default pointing at ``alias``."""
    entry = registry.models.get(alias)
    if entry is None:
        raise KeyError(alias)
    if entry.role == "chat":
        return Registry(
            schema_version=registry.schema_version,
            default_chat=alias,
            default_embed=registry.default_embed,
            models={**registry.models},
        )
    if entry.role == "embed":
        return Registry(
            schema_version=registry.schema_version,
            default_chat=registry.default_chat,
            default_embed=alias,
            models={**registry.models},
        )
    raise ValueError(f"alias {alias!r} has unsupported role {entry.role!r}")


def alias_for_repo(repo: str) -> str:
    """Derive the default alias for ``repo`` per the locked rule.

    Strips a single ``<org>/`` prefix, lowercases, and validates the result
    matches :data:`_ALIAS_RE`. Raises :class:`ValueError` otherwise.
    """
    parts = repo.split("/")
    if len(parts) == 1:
        candidate = parts[0]
    elif len(parts) == 2:
        candidate = parts[1]
    else:
        raise ValueError(f"repo {repo!r} has nested slashes; cannot infer alias")
    candidate = candidate.lower()
    if not candidate or not _ALIAS_RE.match(candidate):
        raise ValueError(f"repo {repo!r} does not yield a valid alias")
    return candidate


def _require_valid_alias(alias: str) -> None:
    if not alias:
        raise ValueError("alias must be non-empty")
    if not _ALIAS_RE.match(alias):
        raise ValueError(
            f"alias {alias!r} must match {_ALIAS_RE.pattern} "
            "(lowercase alphanumerics plus '.', '_', '-')"
        )
