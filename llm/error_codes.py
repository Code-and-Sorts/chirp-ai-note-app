"""Wire-level error code constants and the code → exception mapping.

This module is the single source of truth for error code string literals.
No other module in the codebase declares these strings directly — daemon and
client both import the constants by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

PROTOCOL_VERSION_MISMATCH = "PROTOCOL_VERSION_MISMATCH"
PROTOCOL_MALFORMED = "PROTOCOL_MALFORMED"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
MODEL_GENERATION_FAILED = "MODEL_GENERATION_FAILED"
MODEL_CANCELLED = "MODEL_CANCELLED"

ALL_CODES: frozenset[str] = frozenset(
    {
        PROTOCOL_VERSION_MISMATCH,
        PROTOCOL_MALFORMED,
        MODEL_NOT_FOUND,
        MODEL_LOAD_FAILED,
        MODEL_GENERATION_FAILED,
        MODEL_CANCELLED,
    }
)

if TYPE_CHECKING:
    from llm.exceptions import CODE_TO_EXCEPTION as CODE_TO_EXCEPTION  # noqa: F401


def __getattr__(name: str) -> Any:
    if name == "CODE_TO_EXCEPTION":
        from llm.exceptions import CODE_TO_EXCEPTION

        return CODE_TO_EXCEPTION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
