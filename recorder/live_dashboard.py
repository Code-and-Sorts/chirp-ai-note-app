from __future__ import annotations

import queue
import select
import sys
import threading
import time
from contextlib import contextmanager

try:
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from recorder.live_types import DashboardEvent, TranscriptSegment


class LiveDashboard:
    def __init__(
        self,
        console: Console,
        event_queue: queue.Queue[DashboardEvent],
        stop_event: threading.Event,
        start_time: float,
    ):
        self.console = console
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.start_time = start_time

        self._transcripts: list[TranscriptSegment] = []
        self._language: str | None = None
        self._total_words = 0
        self._latest_level = 0.0
        self._lock = threading.Lock()
        self._chunk_count = 0
        self._sample_rate = 0
        self._channels = 0
        self._frames = 0
        self._processed_chunks = 0
        self._segments_last = 0
        self._vad_frames = 0
        self._vad_speech_frames = 0
        self._vad_triggered = False
        self._vad_chunks_emitted = 0
        self._scroll_offset = 0
        self._auto_scroll = True
        self._stdin_fd: int | None = None
        self._old_settings = None
        self._keyboard_enabled = _HAS_TERMIOS and sys.stdin.isatty()
        if self._keyboard_enabled:
            try:
                self._stdin_fd = sys.stdin.fileno()
            except (OSError, AttributeError):
                self._keyboard_enabled = False

    @contextmanager
    def _raw_mode(self):
        if not self._keyboard_enabled:
            yield
            return

        try:
            self._old_settings = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            yield
        except (termios.error, OSError):
            yield
        finally:
            if self._old_settings:
                try:
                    termios.tcsetattr(
                        self._stdin_fd, termios.TCSADRAIN, self._old_settings
                    )
                except (termios.error, OSError):
                    pass

    def _check_keyboard_input(self):
        if not self._keyboard_enabled:
            return

        if select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if char == "\x1b":
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    next_chars = sys.stdin.read(2)
                    if next_chars == "[A":
                        self._handle_scroll_up()
                    elif next_chars == "[B":
                        self._handle_scroll_down()
                    elif next_chars == "[5":
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            sys.stdin.read(1)
                        self._handle_page_up()
                    elif next_chars == "[6":
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            sys.stdin.read(1)
                        self._handle_page_down()
            elif char in [" ", "\n"]:
                self._handle_scroll_to_bottom()

    def _handle_scroll_up(self):
        with self._lock:
            max_lines = self._estimate_visible_lines()
            max_offset = max(0, len(self._transcripts) - max_lines)
            self._scroll_offset = min(self._scroll_offset + 5, max_offset)
            self._auto_scroll = False

    def _handle_scroll_down(self):
        with self._lock:
            self._scroll_offset = max(0, self._scroll_offset - 5)
            if self._scroll_offset == 0:
                self._auto_scroll = True

    def _handle_page_up(self):
        with self._lock:
            max_lines = self._estimate_visible_lines()
            max_offset = max(0, len(self._transcripts) - max_lines)
            self._scroll_offset = min(self._scroll_offset + max_lines, max_offset)
            self._auto_scroll = False

    def _handle_page_down(self):
        with self._lock:
            max_lines = self._estimate_visible_lines()
            self._scroll_offset = max(0, self._scroll_offset - max_lines)
            if self._scroll_offset == 0:
                self._auto_scroll = True

    def _handle_scroll_to_bottom(self):
        with self._lock:
            self._scroll_offset = 0
            self._auto_scroll = True

    def run(self):
        layout = self._render_layout()
        with self._raw_mode():
            with Live(
                layout,
                console=self.console,
                refresh_per_second=4,
                screen=True,
            ) as live:
                last_render = time.monotonic()
                while not self.stop_event.is_set():
                    self._check_keyboard_input()

                    try:
                        event = self.event_queue.get(timeout=0.1)
                    except queue.Empty:
                        now = time.monotonic()
                        if now - last_render >= 0.25:
                            live.update(self._render_layout(), refresh=True)
                            last_render = now
                        continue

                    self._handle_event(event)
                    live.update(self._render_layout(), refresh=True)
                    last_render = time.monotonic()

    def _handle_event(self, event: DashboardEvent):
        if event.type == "transcript":
            segments = event.payload.get("segments", [])
            with self._lock:
                self._transcripts.extend(segments)
                self._language = event.payload.get("language", self._language)
                self._total_words = event.payload.get("total_words", self._total_words)
        elif event.type == "level":
            value = event.payload.get("value", 0.0)
            with self._lock:
                self._latest_level = float(value)
        elif event.type == "chunk":
            with self._lock:
                self._chunk_count += 1
        elif event.type == "debug":
            with self._lock:
                self._frames = int(event.payload.get("frames", self._frames))
                self._chunk_count = int(event.payload.get("chunks", self._chunk_count))
        elif event.type == "transcriber":
            with self._lock:
                self._processed_chunks = int(event.payload.get("processed", 0))
                self._segments_last = int(event.payload.get("new_segments", 0))
        elif event.type == "info":
            with self._lock:
                self._sample_rate = int(event.payload.get("sample_rate", 0))
                self._channels = int(event.payload.get("channels", 0))
        elif event.type == "vad_status":
            with self._lock:
                self._vad_frames = int(event.payload.get("frames", 0))
                self._vad_speech_frames = int(event.payload.get("speech_frames", 0))
                self._vad_triggered = bool(event.payload.get("triggered", False))
                self._vad_chunks_emitted = int(event.payload.get("chunks_emitted", 0))
        elif event.type == "chunk_emitted":
            with self._lock:
                self._vad_chunks_emitted = int(event.payload.get("chunk_id", 0))

    def _render_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=3),
            Layout(name="footer", size=5),
        )
        layout["body"].split_row(
            Layout(name="transcript", ratio=3),
            Layout(name="status", ratio=1),
        )

        layout["header"].update(self._render_header())
        layout["transcript"].update(self._render_transcript())
        layout["status"].update(self._render_status())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        elapsed = time.monotonic() - self.start_time
        elapsed_text = self._format_elapsed(elapsed)
        header = Text(
            f" LIVE TRANSCRIPTION • {elapsed_text} ", style="bold white on blue"
        )
        return Panel(header, style="white on blue")

    def _render_transcript(self) -> Panel:
        with self._lock:
            segments = list(self._transcripts)
            scroll_offset = self._scroll_offset
            auto_scroll = self._auto_scroll

        if not segments:
            message = Text("Waiting for speech…", style="dim")
            return Panel(message, title="Transcript", border_style="cyan")

        max_lines = self._estimate_visible_lines()
        lines = Text()

        if auto_scroll:
            visible_segments = segments[-max_lines:]
            start_idx = max(0, len(segments) - max_lines)
        else:
            end_idx = len(segments) - scroll_offset
            start_idx = max(0, end_idx - max_lines)
            visible_segments = segments[start_idx:end_idx]

        for segment in visible_segments:
            timestamp = self._format_elapsed(segment.start)
            lines.append(f"[{timestamp}] ", style="cyan")
            sanitized_text = self._sanitize_text(segment.text)
            lines.append(sanitized_text)
            lines.append("\n")

        if auto_scroll:
            title = f"Transcript ({len(segments)} segments)"
        else:
            title = f"Transcript (showing {start_idx + 1}-{start_idx + len(visible_segments)} of {len(segments)})"

        return Panel(lines, title=title, border_style="cyan")

    def _render_status(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")

        elapsed = time.monotonic() - self.start_time
        level_bar = self._render_level_bar(self._latest_level)

        language = self._language or "Detecting…"
        speech_state = "🟢 Speaking" if self._vad_triggered else "⚫ Silent"

        table.add_row("Status ", speech_state)
        table.add_row("Duration ", self._format_elapsed(elapsed))
        table.add_row("Language ", language)
        table.add_row("Words ", str(self._total_words))
        table.add_row("Audio ", level_bar)

        return Panel(table, title="Status", border_style="magenta", box=box.ROUNDED)

    def _render_footer(self) -> Panel:
        with self._lock:
            auto_scroll = self._auto_scroll

        instructions = Text()
        instructions.append("Ctrl+C", style="bold")
        instructions.append(" to stop  •  ")

        if auto_scroll:
            instructions.append("↑↓", style="bold cyan")
            instructions.append(" or ")
            instructions.append("PgUp/PgDn", style="bold cyan")
            instructions.append(" to scroll", style="dim")
        else:
            instructions.append("↑ SCROLLED", style="bold yellow")
            instructions.append(" - Press ")
            instructions.append("SPACE", style="bold cyan")
            instructions.append(" to resume auto-scroll", style="dim")

        return Panel(instructions, border_style="dim")

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total_seconds = int(max(0, seconds))
        minutes, sec = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _estimate_visible_lines(self) -> int:
        try:
            terminal_height = self.console.height
            if terminal_height is None:
                return 40

            header_height: int = 3
            footer_height: int = 5
            borders_and_padding: int = 4
            available_height: int = (
                terminal_height - header_height - footer_height - borders_and_padding
            )
            max_lines: int = max(10, available_height)
            return max_lines
        except (AttributeError, ValueError, TypeError):
            return 40

    @staticmethod
    def _sanitize_text(text: str) -> str:
        if not text:
            return ""
        sanitized = text.replace("\x1b", "")
        sanitized_result = "".join(
            char for char in sanitized if char.isprintable() or char in "\n\t"
        )
        return sanitized_result

    @staticmethod
    def _render_level_bar(level: float) -> str:
        buckets = 10
        filled = min(buckets, int(level * buckets))
        bar = "█" * filled + "░" * (buckets - filled)
        return bar
