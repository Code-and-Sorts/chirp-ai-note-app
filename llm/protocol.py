"""NDJSON wire-protocol encode/decode, op/event constants, and validation.

Both the daemon (``chirpd``) and the client (``llm.client``) import from this
module so that envelope shape, op-name spelling, and the request-id format
never drift.
"""

from __future__ import annotations

import json
import re
import secrets
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final

from llm.exceptions import LLMMalformedResponse

DISTRIBUTION_NAME: Final[str] = "chirp-notes-ai"

# Wire-format contract version. Bumped ONLY when the envelope/op/event grammar
# changes — never on a cosmetic package release. The ``hello`` handshake
# compares this (not ``package_version()``), so a patch/docs/dependency bump
# keeps a warm daemon resident instead of evicting its model on a respawn.
PROTOCOL_VERSION: Final[int] = 1


def package_version() -> str:
    """Return the installed chirp distribution version, falling back to 0.0.0."""
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "0.0.0"


MAX_LINE_BYTES: Final[int] = 1_048_576

REQUEST_ID_RE: Final[re.Pattern[str]] = re.compile(r"^r-[0-9a-f]{12}$")

OP_HELLO: Final[str] = "hello"
OP_HEALTH: Final[str] = "health"
OP_CHAT: Final[str] = "chat"
OP_EMBED: Final[str] = "embed"
OP_CANCEL: Final[str] = "cancel"
OP_MODEL_LIST: Final[str] = "model.list"
OP_MODEL_LOAD: Final[str] = "model.load"
OP_MODEL_UNLOAD: Final[str] = "model.unload"
OP_MODEL_STATUS: Final[str] = "model.status"

EVENT_READY: Final[str] = "ready"
EVENT_LOADING: Final[str] = "loading"
EVENT_DELTA: Final[str] = "delta"
EVENT_DONE: Final[str] = "done"
EVENT_ERROR: Final[str] = "error"
EVENT_VERSION_MISMATCH: Final[str] = "version_mismatch"
EVENT_STATUS: Final[str] = "status"

_KNOWN_OPS: Final[frozenset[str]] = frozenset(
    {
        OP_HELLO,
        OP_HEALTH,
        OP_CHAT,
        OP_EMBED,
        OP_CANCEL,
        OP_MODEL_LIST,
        OP_MODEL_LOAD,
        OP_MODEL_UNLOAD,
        OP_MODEL_STATUS,
    }
)


def new_request_id() -> str:
    """Generate a fresh client-side request id (``r-`` + 12 hex chars)."""
    return f"r-{secrets.token_hex(6)}"


def _encode(envelope: dict[str, Any]) -> bytes:
    payload = json.dumps(envelope, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(payload) > MAX_LINE_BYTES:
        raise LLMMalformedResponse(
            f"encoded envelope exceeds MAX_LINE_BYTES ({len(payload)} > "
            f"{MAX_LINE_BYTES})",
            details={"size": len(payload), "limit": MAX_LINE_BYTES},
        )
    return payload


def encode_request(envelope: dict[str, Any]) -> bytes:
    """Encode a client-→-daemon request envelope as a single NDJSON line."""
    return _encode(envelope)


def encode_event(envelope: dict[str, Any]) -> bytes:
    """Encode a daemon-→-client event envelope as a single NDJSON line."""
    return _encode(envelope)


def decode_line(line: bytes) -> dict[str, Any]:
    """Decode a single NDJSON line into a top-level dict envelope.

    Raises:
        LLMMalformedResponse: on non-UTF-8 bytes, invalid JSON, JSON that is
            not a top-level object, or lines exceeding ``MAX_LINE_BYTES``.
    """
    if len(line) > MAX_LINE_BYTES:
        raise LLMMalformedResponse(
            f"line exceeds MAX_LINE_BYTES ({len(line)} > {MAX_LINE_BYTES})",
            details={"size": len(line), "limit": MAX_LINE_BYTES},
        )
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as err:
        raise LLMMalformedResponse(
            f"line is not valid UTF-8: {err}",
            details={"error": str(err)},
        ) from err
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        raise LLMMalformedResponse(
            f"line is not valid JSON: {err.msg}",
            details={"error": err.msg, "pos": err.pos},
        ) from err
    if not isinstance(parsed, dict):
        raise LLMMalformedResponse(
            f"envelope must be a JSON object, got {type(parsed).__name__}",
            details={"type": type(parsed).__name__},
        )
    return parsed


def validate_request(envelope: dict[str, Any]) -> None:
    """Validate a request envelope's shape and op-name.

    Raises:
        LLMMalformedResponse: when ``op`` is missing or unknown, when ``id`` is
            required but missing (every op except ``hello``), or when a present
            ``id`` does not match :data:`REQUEST_ID_RE`.
    """
    if not isinstance(envelope, dict):
        raise LLMMalformedResponse(
            f"request envelope must be a dict, got {type(envelope).__name__}",
            details={"type": type(envelope).__name__},
        )
    op = envelope.get("op")
    if op is None:
        raise LLMMalformedResponse(
            "request envelope is missing required key 'op'",
        )
    if not isinstance(op, str) or op not in _KNOWN_OPS:
        raise LLMMalformedResponse(
            f"unknown op {op!r}",
            details={"op": op},
        )
    request_id = envelope.get("id")
    if request_id is None:
        if op != OP_HELLO:
            raise LLMMalformedResponse(
                f"request {op!r} is missing required key 'id'",
                details={"op": op},
            )
        return
    if not isinstance(request_id, str) or not REQUEST_ID_RE.match(request_id):
        raise LLMMalformedResponse(
            f"request id {request_id!r} does not match {REQUEST_ID_RE.pattern}",
            details={"id": request_id},
        )
