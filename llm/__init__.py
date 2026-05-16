"""chirp LLM client + shared wire-protocol primitives.

Op/event constants are *not* re-exported here — callers import them from
:mod:`llm.protocol`. Only the exception hierarchy is surfaced at the package
level for ergonomic ``from llm import LLMModelError`` usage.
"""

from __future__ import annotations

from llm.exceptions import (
    LLMCancelled,
    LLMConnectionLost,
    LLMDaemonSpawnFailed,
    LLMDaemonUnreachable,
    LLMError,
    LLMGenerationFailed,
    LLMMalformedResponse,
    LLMModelError,
    LLMModelLoadFailed,
    LLMModelNotFound,
    LLMProtocolError,
    LLMTransportError,
    LLMVersionMismatch,
)

__all__ = [
    "LLMCancelled",
    "LLMConnectionLost",
    "LLMDaemonSpawnFailed",
    "LLMDaemonUnreachable",
    "LLMError",
    "LLMGenerationFailed",
    "LLMMalformedResponse",
    "LLMModelError",
    "LLMModelLoadFailed",
    "LLMModelNotFound",
    "LLMProtocolError",
    "LLMTransportError",
    "LLMVersionMismatch",
]
