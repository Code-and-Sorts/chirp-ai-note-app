"""Shared Rich consoles for the ``chirp models`` subcommands.

Progress, status, and error output from ``chirp models *`` goes to **stderr**
(``console``) so stdout stays clean for structured output (architecture §CLI
Output Patterns). Commands whose primary result is a rendered artifact — e.g.
the ``chirp models list`` table — write that artifact to **stdout**
(``stdout_console``) while keeping diagnostic preamble on stderr. Sharing single
console instances across the subcommand modules and the progress adapter keeps
width, theme, and terminal detection consistent.
"""

from __future__ import annotations

from rich.console import Console

console = Console(stderr=True)
stdout_console = Console()
