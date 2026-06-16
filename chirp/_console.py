"""Shared stdout/stderr Rich consoles for the first-party ``chirp`` commands.

The newer ``llm/cli`` subsystem already splits its output the clig.dev way —
data on stdout, diagnostics on stderr — via :mod:`llm.cli._console`. Rather than
re-create that pattern, the older commands (``record``, ``transcribe``,
``notes``, ``ask``, ``search``, the note editor) re-export the *same* instances
here so width, theme, and terminal detection stay consistent across every chirp
surface.

- ``stderr_console`` — progress, status, hints, prompts, and errors.
- ``stdout_console`` — data and rendered artifacts (tables, note bodies).

``--no-color`` / ``--plain`` and ``NO_COLOR`` flow through :func:`apply_color_mode`,
which rebuilds both consoles with color forced off. Rich already honors
``NO_COLOR`` / ``FORCE_COLOR`` and non-TTY detection on construction, so the only
job here is to stop overriding that and to expose a single switch for the flag.
"""

from __future__ import annotations

import os

from llm.cli._console import console as stderr_console
from llm.cli._console import stdout_console

__all__ = ["apply_color_mode", "no_color_active", "stderr_console", "stdout_console"]


def no_color_active() -> bool:
    """True when ``NO_COLOR`` is set to any non-empty value (de-facto contract)."""
    return bool(os.environ.get("NO_COLOR"))


def apply_color_mode(no_color: bool) -> None:
    """Force color off on both shared consoles for the rest of the run.

    Called from the top-level callback when ``--no-color`` / ``--plain`` is
    passed. ``NO_COLOR`` in the environment is honored by Rich automatically, so
    this only handles the explicit flag; passing ``no_color=False`` is a no-op so
    a normal run keeps Rich's auto-detection intact.
    """
    if not no_color:
        return
    stderr_console.no_color = True
    stdout_console.no_color = True
