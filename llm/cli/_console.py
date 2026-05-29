"""Shared Rich console for the ``chirp models`` subcommands.

All progress, status, and error output from ``chirp models *`` goes to
**stderr** so stdout stays clean for future structured output (architecture
§CLI Output Patterns). A single console instance is shared across the
subcommand modules and the progress adapter rather than each call site building
its own, so width, theme, and terminal detection stay consistent.
"""

from __future__ import annotations

from rich.console import Console

console = Console(stderr=True)
