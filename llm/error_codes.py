"""Wire-level error code constants.

This module is the single source of truth for error code string literals.
No other module in the codebase declares these strings directly — daemon and
client both import the constants by name. The matching ``code -> exception``
table lives in :mod:`llm.exceptions` to keep the dependency one-directional.
"""

from __future__ import annotations

PROTOCOL_VERSION_MISMATCH = "PROTOCOL_VERSION_MISMATCH"
PROTOCOL_MALFORMED = "PROTOCOL_MALFORMED"
PROTOCOL_REQUEST_CONFLICT = "PROTOCOL_REQUEST_CONFLICT"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
MODEL_GENERATION_FAILED = "MODEL_GENERATION_FAILED"
MODEL_CANCELLED = "MODEL_CANCELLED"
MODEL_CAPACITY_EXCEEDED = "MODEL_CAPACITY_EXCEEDED"
REQUEST_NOT_FOUND = "REQUEST_NOT_FOUND"

ALL_CODES: frozenset[str] = frozenset(
    {
        PROTOCOL_VERSION_MISMATCH,
        PROTOCOL_MALFORMED,
        PROTOCOL_REQUEST_CONFLICT,
        MODEL_NOT_FOUND,
        MODEL_LOAD_FAILED,
        MODEL_GENERATION_FAILED,
        MODEL_CANCELLED,
        MODEL_CAPACITY_EXCEEDED,
        REQUEST_NOT_FOUND,
    }
)
