"""Exception hierarchy for the chirp LLM client/daemon boundary.

Concrete subclasses carry a class-level ``code`` attribute that points at the
matching wire constant in :mod:`llm.error_codes`. The ``CODE_TO_EXCEPTION``
mapping at the bottom of this module is the inverse lookup used by the client's
NDJSON reader to raise the correct typed exception for an incoming
``{event: "error", code: ...}`` envelope.
"""

from __future__ import annotations

from typing import Any, ClassVar

from llm import error_codes


class LLMError(Exception):
    """Base class for every chirp LLM-layer exception."""

    code: ClassVar[str | None] = None

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class LLMTransportError(LLMError):
    """Socket / connection-level failure between client and daemon."""


class LLMDaemonUnreachable(LLMTransportError):
    """The daemon socket exists but refuses the connection or is absent."""


class LLMConnectionLost(LLMTransportError):
    """An in-flight request was interrupted by a broken pipe / EOF."""


class LLMDaemonSpawnFailed(LLMTransportError):
    """Lazy-spawn of the daemon process could not produce a working daemon."""


class LLMInferenceTimeout(LLMTransportError):
    """No event arrived from the daemon within the inference read budget.

    Client-local: raised when ``asyncio.wait_for`` around a per-event socket
    read elapses. It never travels the wire, so it carries no class-level
    ``code`` and is absent from :data:`CODE_TO_EXCEPTION`.
    """


class LLMProtocolError(LLMError):
    """Protocol-level violation: malformed envelope, version mismatch, etc."""


class LLMVersionMismatch(LLMProtocolError):
    """Client and daemon protocol versions disagree (retry exhausted)."""

    code: ClassVar[str] = error_codes.PROTOCOL_VERSION_MISMATCH


class LLMMalformedResponse(LLMProtocolError):
    """An envelope that cannot be produced or parsed as a valid NDJSON line."""

    code: ClassVar[str] = error_codes.PROTOCOL_MALFORMED


class LLMRequestConflict(LLMProtocolError):
    """A request id was reused while the original is still in flight."""

    code: ClassVar[str] = error_codes.PROTOCOL_REQUEST_CONFLICT


class LLMRequestNotFound(LLMProtocolError):
    """A cancel named a target id with no matching in-flight request."""

    code: ClassVar[str] = error_codes.REQUEST_NOT_FOUND


class LLMModelError(LLMError):
    """Model-side failure surfaced from the daemon back to the client."""


class LLMModelNotFound(LLMModelError):
    """Alias is not present in the registry or its weights cannot be located."""

    code: ClassVar[str] = error_codes.MODEL_NOT_FOUND


class LLMModelLoadFailed(LLMModelError):
    """``mlx-lm`` could not load the model (architecture, OOM, weights)."""

    code: ClassVar[str] = error_codes.MODEL_LOAD_FAILED


class LLMGenerationFailed(LLMModelError):
    """Inference raised mid-generation."""

    code: ClassVar[str] = error_codes.MODEL_GENERATION_FAILED


class LLMCancelled(LLMModelError):
    """A cancel op succeeded; the caller should treat this as user intent."""

    code: ClassVar[str] = error_codes.MODEL_CANCELLED


class LLMModelCapacityExceeded(LLMModelError):
    """The resident-model set is full and no model is free to evict."""

    code: ClassVar[str] = error_codes.MODEL_CAPACITY_EXCEEDED


CODE_TO_EXCEPTION: dict[str, type[LLMError]] = {
    error_codes.PROTOCOL_VERSION_MISMATCH: LLMVersionMismatch,
    error_codes.PROTOCOL_MALFORMED: LLMMalformedResponse,
    error_codes.PROTOCOL_REQUEST_CONFLICT: LLMRequestConflict,
    error_codes.REQUEST_NOT_FOUND: LLMRequestNotFound,
    error_codes.MODEL_NOT_FOUND: LLMModelNotFound,
    error_codes.MODEL_LOAD_FAILED: LLMModelLoadFailed,
    error_codes.MODEL_GENERATION_FAILED: LLMGenerationFailed,
    error_codes.MODEL_CANCELLED: LLMCancelled,
    error_codes.MODEL_CAPACITY_EXCEEDED: LLMModelCapacityExceeded,
}
