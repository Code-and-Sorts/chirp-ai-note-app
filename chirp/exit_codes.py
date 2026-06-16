"""Centralized exit codes for the ``chirp`` CLI.

One source of truth for every ``typer.Exit(code=...)`` site so the numbers stay
a stable, documented contract instead of scattered literals. The observable
codes ``2``/``3``/``4``/``5``/``7`` predate this module (stories 3.7, 5.x, 7.1)
and must not be renumbered — they are gathered here, not changed.

================  ====  ===========================================================
Name              Code  Meaning
================  ====  ===========================================================
SUCCESS           0     Command completed.
RUNTIME_ERROR     1     Generic runtime failure (also Ctrl-C / interrupted).
USAGE_ERROR       2     Bad arguments or flags (Click's own convention).
DAEMON_UNREACHABLE 3    The chirpd daemon is not running and could not be started.
MODEL_LOAD_FAILED 4     A model was found but could not be loaded.
MODEL_NOT_FOUND   5     No model is registered for the requested role.
NOT_APPLE_SILICON 7     ``chirp init`` ran on a non-arm64 machine.
================  ====  ===========================================================

``6`` is intentionally unused (reserved). Ctrl-C / EOF keeps the current
behavior of exiting ``1`` rather than claiming a dedicated "interrupted" slot.

``MODEL_LOAD_FAILED`` (4) is the LLM-daemon (chirpd) path. Whisper transcription
model-load failures are deliberately separate: live ``record --live-transcribe``
surfaces them as ``4``, but batch ``transcribe`` keeps ``1`` by existing
contract (``test_transcribe_reports_model_load_error``) — the difference is
intentional, not an oversight.
"""

from __future__ import annotations

SUCCESS = 0
RUNTIME_ERROR = 1
USAGE_ERROR = 2
DAEMON_UNREACHABLE = 3
MODEL_LOAD_FAILED = 4
MODEL_NOT_FOUND = 5
NOT_APPLE_SILICON = 7

__all__ = [
    "DAEMON_UNREACHABLE",
    "MODEL_LOAD_FAILED",
    "MODEL_NOT_FOUND",
    "NOT_APPLE_SILICON",
    "RUNTIME_ERROR",
    "SUCCESS",
    "USAGE_ERROR",
]
