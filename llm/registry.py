"""Read-only model-registry parser for ``models.toml``.

The writer side (``chirp models add`` and friends) lives in EPIC-MODEL-REGISTRY;
this module gives the daemon a typed view of the on-disk registry so it can
resolve alias → repo on every ``model.load`` / ``model.list`` / ``model.status``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from chirpd.paths import MODELS_TOML_PATH
from llm.exceptions import LLMMalformedResponse, LLMModelNotFound

SUPPORTED_SCHEMA_VERSION = 1

ModelRole = Literal["chat", "embed"]


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
        with open(target, "rb") as handle:
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
            "re-init your registry with `chirp models init` "
            "(coming in EPIC-MODEL-REGISTRY)",
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
