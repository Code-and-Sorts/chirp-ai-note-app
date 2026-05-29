"""Rich progress adapter bridging ``llm.hf`` downloads to the shared console.

:class:`RichProgressCallback` implements the :class:`llm.hf.ProgressCallback`
Protocol. On an interactive terminal it drives a Rich progress bar with
bytes-downloaded, transfer speed, and ETA. Off a terminal (CI, pipes) it falls
back to two plain stderr lines — one at start, one at done — so logs stay
readable without ANSI churn. ``chirp models add`` (story 4.3) and
``chirp models pull`` (story 4.5) share this single adapter.
"""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from llm.cli._console import console as shared_console


class RichProgressCallback:
    """Drive a Rich progress bar (TTY) or stderr status lines (non-TTY)."""

    def __init__(self, repo_id: str, *, console: Console = shared_console) -> None:
        self._repo_id = repo_id
        self._console = console
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._bytes_downloaded = 0

    def on_start(self, total_bytes: int | None) -> None:
        if self._console.is_terminal:
            self._progress = Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=self._console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                f"Downloading {self._repo_id}", total=total_bytes
            )
        else:
            self._console.print(
                f"Downloading {self._repo_id}...", markup=False, soft_wrap=True
            )

    def on_progress(self, bytes_downloaded: int, total_bytes: int | None) -> None:
        self._bytes_downloaded = bytes_downloaded
        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id, completed=bytes_downloaded, total=total_bytes
            )

    def on_done(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        else:
            self._console.print(
                f"Downloaded {self._repo_id} ({self._bytes_downloaded} bytes)",
                markup=False,
                soft_wrap=True,
            )
