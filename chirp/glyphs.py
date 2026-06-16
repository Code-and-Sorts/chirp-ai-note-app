"""Shared status glyphs for the ``chirp`` CLI.

Glyphs drifted across commands — ``devices`` used the emoji ``✅`` / ``❌`` while
``init`` and ``search`` already used the terser text marks ``✓`` / ``✗`` / ``—``.
This module is the single source so every command speaks the same visual
vocabulary; the house voice is lowercase and terse (see the story 1.8 ``search``
copy), so messages alongside these marks stay sentence-case, not Title Case.
"""

from __future__ import annotations

SUCCESS = "✓"
FAILURE = "✗"
PENDING = "—"
DEFAULT_MARKER = "★"
ACTIVE_MARKER = "●"
INPUT_ARROW = "›"

__all__ = [
    "ACTIVE_MARKER",
    "DEFAULT_MARKER",
    "FAILURE",
    "INPUT_ARROW",
    "PENDING",
    "SUCCESS",
]
