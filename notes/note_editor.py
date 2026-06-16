"""Terminal-based modal editor for manual notes."""

from __future__ import annotations

import curses
from dataclasses import dataclass
from typing import Any, ClassVar

from rich.console import Console
from rich.markdown import Heading, Markdown
from rich.segment import Segment
from rich.style import Style

_HAS_ITALIC = hasattr(curses, "A_ITALIC")
_ITALIC_ATTR = getattr(curses, "A_ITALIC", 0)


@dataclass
class EditorResult:
    content: str
    saved: bool


class PlainHeading(Heading):
    """Render h1 headings without Rich's default border."""

    def __rich_console__(self, console: Console, options):
        text = self.text
        text.justify = "left"
        yield text


class PlainMarkdown(Markdown):
    """Markdown renderer tuned for the editor view mode."""

    elements = Markdown.elements.copy()
    elements["heading_open"] = PlainHeading


@dataclass
class DisplayLine:
    segments: list[Segment]
    source_line: int


KEYBINDING_HINT = "i insert · Esc normal · :wq save · :q! quit · ? :help"

HELP_LINES = (
    "  chirp note editor — help",
    "",
    "  modes",
    "    i           enter insert mode",
    "    Esc         return to normal/view mode",
    "",
    "  commands (normal mode, type with :)",
    "    :wq         save and quit",
    "    :q!         quit without saving",
    "    :help       show this help",
    "",
    "  keys",
    "    ?           show this help (normal/view mode)",
    "    arrows/hjkl move the cursor",
    "    PgUp/PgDn   scroll a page",
    "",
    "  press any key to close this help",
)


class ManualNoteEditor:
    """A minimal modal text editor with view/insert modes."""

    _COLOR_NAME_MAP: ClassVar[dict[str, int]] = {
        "black": curses.COLOR_BLACK,
        "red": curses.COLOR_RED,
        "green": curses.COLOR_GREEN,
        "yellow": curses.COLOR_YELLOW,
        "blue": curses.COLOR_BLUE,
        "magenta": curses.COLOR_MAGENTA,
        "cyan": curses.COLOR_CYAN,
        "white": curses.COLOR_WHITE,
    }

    def __init__(
        self,
        title: str,
        initial_content: str,
        readonly: bool = False,
        start_in_insert: bool = False,
    ):
        self.title = title
        self.lines = self._initialise_lines(initial_content)
        self.cursor_y = 0
        self.cursor_x = 0
        self.top_line = 0
        self.mode = "insert" if (start_in_insert and not readonly) else "view"
        self.command_active = False
        self.command_buffer = ""
        self.message = ""
        self.saved = False
        self.dirty = False
        self.readonly = readonly
        self._readonly_notified = False
        self.show_help = False

        self._has_colors = False
        self._color_pairs: dict[tuple[int, int], int] = {}
        self._text_pairs: dict[str, int] = {"view": 0, "insert": 0}
        self._status_pairs: dict[str, int] = {
            "view": curses.A_REVERSE,
            "insert": curses.A_REVERSE,
        }
        self._default_fg = {"view": curses.COLOR_BLACK, "insert": curses.COLOR_WHITE}
        self._default_bg = {"view": -1, "insert": -1}

        self._view_cache_dirty = True
        self._view_cache_width: int | None = None
        self._display_lines: list[DisplayLine] = []
        self._view_cursor_index: int | None = None
        self._view_top_index: int | None = None
        self._last_text_height = 1

    def run(self) -> EditorResult:
        curses.wrapper(self._main)
        content = "\n".join(self.lines)
        if not content.endswith("\n"):
            content += "\n"
        return EditorResult(content=content, saved=self.saved)

    def _main(self, stdscr: Any) -> None:
        curses.curs_set(1)
        stdscr.keypad(True)
        self._init_colors()
        while True:
            self._render(stdscr)
            try:
                get_wch = stdscr.get_wch
                key = get_wch()
            except curses.error:
                continue

            if self.show_help:
                # The help overlay is modal: any key dismisses it.
                self.show_help = False
                continue

            if self.command_active:
                if self._handle_command_input(key):
                    break
                continue

            if self._handle_global_keys(key):
                break

        stdscr.clear()
        stdscr.refresh()

    def _initialise_lines(self, content: str) -> list[str]:
        if not content:
            return [""]

        lines = content.split("\n")
        if lines and lines[-1] == "":
            return lines

        lines.append("")
        return lines

    def _render(self, stdscr: curses.window) -> None:
        height, width = stdscr.getmaxyx()
        # Reserve the bottom-most row for the status line and the row above it for
        # the persistent keybinding hint, so neither overwrites the text body.
        hint_row = max(0, height - 2)
        text_height = max(1, height - 2)
        text_width = max(1, width - 1)
        self._last_text_height = text_height

        if self.show_help:
            self._render_help_overlay(stdscr, height, width)
            return

        if self.mode == "view":
            self._ensure_view_cache(text_width)
            self._ensure_view_positions(text_height)
        else:
            self._ensure_cursor_within_bounds()
            self._adjust_insert_viewport(text_height)

        stdscr.erase()

        if self.mode == "view":
            self._render_view_mode(stdscr, text_height, text_width)
        else:
            self._render_insert_mode(stdscr, text_height, text_width)

        self._render_keybinding_hint(stdscr, hint_row, width)

        status = self._status_line(width)
        status_attr = self._status_pairs.get(self.mode, curses.A_REVERSE)
        try:
            stdscr.addstr(height - 1, 0, " " * max(1, width - 1), status_attr)
            stdscr.addstr(height - 1, 0, status, status_attr)
        except curses.error:
            pass

        if self.mode == "view":
            cursor_screen_y = 0
            if self._view_cursor_index is not None and self._view_top_index is not None:
                cursor_screen_y = self._view_cursor_index - self._view_top_index
        else:
            cursor_screen_y = self.cursor_y - self.top_line

        cursor_screen_x = max(0, min(self.cursor_x, text_width - 1))

        try:
            stdscr.move(cursor_screen_y, cursor_screen_x)
        except curses.error:
            pass

        stdscr.refresh()

    def _render_keybinding_hint(
        self, stdscr: curses.window, hint_row: int, width: int
    ) -> None:
        attr = curses.A_DIM if hasattr(curses, "A_DIM") else curses.A_NORMAL
        hint = KEYBINDING_HINT[: max(1, width - 1)]
        try:
            stdscr.addstr(hint_row, 0, " " * max(1, width - 1))
            stdscr.addstr(hint_row, 0, hint, attr)
        except curses.error:
            pass

    def _render_help_overlay(
        self, stdscr: curses.window, height: int, width: int
    ) -> None:
        stdscr.erase()
        for row, line in enumerate(HELP_LINES):
            if row >= height:
                break
            try:
                stdscr.addstr(row, 0, line[: max(1, width - 1)])
            except curses.error:
                continue
        stdscr.refresh()

    def _render_insert_mode(
        self, stdscr: curses.window, text_height: int, text_width: int
    ) -> None:
        attr = self._text_pairs.get("insert", curses.A_NORMAL)
        for idx in range(text_height):
            line_index = self.top_line + idx
            try:
                stdscr.move(idx, 0)
                stdscr.addstr(idx, 0, " " * text_width, attr)
            except curses.error:
                continue

            if line_index >= len(self.lines):
                continue

            line = self.lines[line_index]
            display_line = line[:text_width]
            try:
                stdscr.addstr(idx, 0, display_line, attr)
            except curses.error:
                pass

    def _render_view_mode(
        self, stdscr: curses.window, text_height: int, text_width: int
    ) -> None:
        start = self._view_top_index or 0
        total_lines = len(self._display_lines)
        for screen_row in range(text_height):
            view_index = start + screen_row
            try:
                stdscr.move(screen_row, 0)
                stdscr.clrtoeol()
            except curses.error:
                continue

            if view_index >= total_lines:
                continue

            display_line = self._display_lines[view_index]
            self._draw_segments(stdscr, screen_row, display_line.segments, text_width)

    def _draw_segments(
        self,
        stdscr: curses.window,
        screen_row: int,
        segments: list[Segment],
        text_width: int,
    ) -> None:
        col = 0
        for segment in segments:
            text = segment.text
            if not text:
                continue
            remaining = text_width - col
            if remaining <= 0:
                break
            snippet = text[:remaining]
            attr = self._style_to_attr(segment.style, "view")
            try:
                stdscr.addstr(screen_row, col, snippet, attr)
            except curses.error:
                pass
            col += len(snippet)

    def _status_line(self, width: int) -> str:
        mode_label = "INSERT" if self.mode == "insert" else "VIEW"
        status = f" {mode_label}"
        if self.readonly:
            status += " | read-only"
        status += f" | Ln {self.cursor_y + 1}, Col {self.cursor_x + 1} "
        if self.dirty:
            status += "| * "
        if self.message:
            status += f"| {self.message} "
        if self.command_active:
            status += f"{self.command_buffer}"
        return status[: max(1, width - 1)].ljust(width - 1)

    def _adjust_insert_viewport(self, text_height: int) -> None:
        if self.cursor_y < self.top_line:
            self.top_line = self.cursor_y
        elif self.cursor_y >= self.top_line + text_height:
            self.top_line = self.cursor_y - text_height + 1

    def _ensure_cursor_within_bounds(self) -> None:
        if not self.lines:
            self.lines = [""]

        self.cursor_y = max(0, min(self.cursor_y, len(self.lines) - 1))
        line_length = len(self.lines[self.cursor_y])
        self.cursor_x = max(0, min(self.cursor_x, line_length))

    def _handle_global_keys(self, key) -> bool:
        if key in ("\x1b", 27):
            if self.mode == "insert":
                self.mode = "view"
                self._view_cursor_index = None
                self._view_top_index = None
            else:
                self.command_active = False
                self.command_buffer = ""
                self.message = ""
            return False

        if self.mode == "insert":
            return self._handle_insert_mode(key)

        return self._handle_view_mode(key)

    def _handle_view_mode(self, key) -> bool:
        if key == "i":
            if self.readonly:
                self._notify_readonly()
                return False
            self._sync_insert_positions()
            self.mode = "insert"
            return False

        if key == ":":
            self.command_active = True
            self.command_buffer = ":"
            self.message = ""
            return False

        if key == "?":
            self.show_help = True
            return False

        return self._handle_navigation(key)

    def _handle_insert_mode(self, key) -> bool:
        if isinstance(key, str):
            if key in ("\n", "\r"):
                self._insert_newline()
                return False
            if key == "\t":
                self._insert_text("    ")
                return False
            if key in {"\x7f", "\b"}:
                self._backspace()
                return False
            if key.isprintable():
                self._insert_text(key)
                return False

        if isinstance(key, int) and key in (curses.KEY_BACKSPACE, 263):
            self._backspace()
            return False

        if key in (curses.KEY_ENTER, 10, 13):
            self._insert_newline()
            return False

        if key == curses.KEY_DC:
            self._delete()
            return False

        return self._handle_navigation(key)

    def _handle_navigation(self, key) -> bool:
        if key in (curses.KEY_LEFT, "h"):
            self._move_left()
        elif key in (curses.KEY_RIGHT, "l"):
            self._move_right()
        elif key in (curses.KEY_UP, "k"):
            self._move_up()
        elif key in (curses.KEY_DOWN, "j"):
            self._move_down()
        elif key == curses.KEY_HOME:
            self.cursor_x = 0
        elif key == curses.KEY_END:
            self.cursor_x = len(self.lines[self.cursor_y])
        elif key == curses.KEY_PPAGE:
            self._page_up()
        elif key == curses.KEY_NPAGE:
            self._page_down()
        elif key == curses.KEY_DC and self.mode == "insert":
            self._delete()
        return False

    def _handle_command_input(self, key) -> bool:
        if isinstance(key, str) and key in ("\n", "\r"):
            return self._execute_command()

        if key in (curses.KEY_ENTER, 10, 13):
            return self._execute_command()

        if key in ("\x1b", 27):
            self.command_active = False
            self.command_buffer = ""
            self.message = ""
            return False

        if key in (curses.KEY_BACKSPACE, 263, "\x7f", "\b"):
            if len(self.command_buffer) > 1:
                self.command_buffer = self.command_buffer[:-1]
            else:
                self.command_active = False
                self.command_buffer = ""
            return False

        if isinstance(key, str) and key.isprintable():
            self.command_buffer += key
        return False

    def _execute_command(self) -> bool:
        command = self.command_buffer[1:] if self.command_buffer.startswith(":") else ""
        self.command_active = False

        if command == "wq":
            self.saved = True
            return True

        if command == "q!":
            self.saved = False
            self.dirty = False
            return True

        if command == "help":
            self.show_help = True
            self.command_buffer = ""
            return False

        self.message = f"Unknown command: {command} — type :help" if command else ""
        self.command_buffer = ""
        return False

    def _insert_text(self, text: str) -> None:
        line = self.lines[self.cursor_y]
        self.lines[self.cursor_y] = line[: self.cursor_x] + text + line[self.cursor_x :]
        self.cursor_x += len(text)
        self.dirty = True
        self._invalidate_view_cache()

    def _insert_newline(self) -> None:
        line = self.lines[self.cursor_y]
        new_line = line[self.cursor_x :]
        self.lines[self.cursor_y] = line[: self.cursor_x]
        self.lines.insert(self.cursor_y + 1, new_line)
        self.cursor_y += 1
        self.cursor_x = 0
        self.dirty = True
        self._invalidate_view_cache()

    def _backspace(self) -> None:
        if self.cursor_x > 0:
            line = self.lines[self.cursor_y]
            self.lines[self.cursor_y] = (
                line[: self.cursor_x - 1] + line[self.cursor_x :]
            )
            self.cursor_x -= 1
            self.dirty = True
            self._invalidate_view_cache()
            return

        if self.cursor_y == 0:
            return

        prev_line = self.lines[self.cursor_y - 1]
        current_line = self.lines[self.cursor_y]
        new_cursor_x = len(prev_line)
        self.lines[self.cursor_y - 1] = prev_line + current_line
        del self.lines[self.cursor_y]
        self.cursor_y -= 1
        self.cursor_x = new_cursor_x
        self.dirty = True
        self._invalidate_view_cache()

    def _delete(self) -> None:
        line = self.lines[self.cursor_y]
        if self.cursor_x < len(line):
            self.lines[self.cursor_y] = (
                line[: self.cursor_x] + line[self.cursor_x + 1 :]
            )
            self.dirty = True
            self._invalidate_view_cache()
            return

        if self.cursor_y < len(self.lines) - 1:
            next_line = self.lines.pop(self.cursor_y + 1)
            self.lines[self.cursor_y] = line + next_line
            self.dirty = True
            self._invalidate_view_cache()

    def _move_left(self) -> None:
        if self.mode == "view":
            self.cursor_x = max(0, self.cursor_x - 1)
            return

        if self.cursor_x > 0:
            self.cursor_x -= 1
            return

        if self.cursor_y > 0:
            self.cursor_y -= 1
            self.cursor_x = len(self.lines[self.cursor_y])

    def _move_right(self) -> None:
        line_length = len(self.lines[self.cursor_y])
        if self.cursor_x < line_length:
            self.cursor_x += 1
            return

        if self.mode == "view":
            return

        if self.cursor_y < len(self.lines) - 1:
            self.cursor_y += 1
            self.cursor_x = 0

    def _move_up(self) -> None:
        if self.mode == "view":
            self._move_view_cursor(-1)
            return

        if self.cursor_y > 0:
            self.cursor_y -= 1
            self.cursor_x = min(self.cursor_x, len(self.lines[self.cursor_y]))

    def _move_down(self) -> None:
        if self.mode == "view":
            self._move_view_cursor(1)
            return

        if self.cursor_y < len(self.lines) - 1:
            self.cursor_y += 1
            self.cursor_x = min(self.cursor_x, len(self.lines[self.cursor_y]))

    def _page_up(self) -> None:
        if self.mode == "view":
            self._move_view_cursor(-self._last_text_height)
        else:
            self.cursor_y = max(0, self.cursor_y - self._last_text_height)

    def _page_down(self) -> None:
        if self.mode == "view":
            self._move_view_cursor(self._last_text_height)
        else:
            self.cursor_y = min(
                len(self.lines) - 1, self.cursor_y + self._last_text_height
            )

    def _move_view_cursor(self, delta: int) -> None:
        if not self._display_lines:
            return

        if self._view_cursor_index is None:
            self._view_cursor_index = 0

        new_index = max(
            0, min(len(self._display_lines) - 1, self._view_cursor_index + delta)
        )
        self._view_cursor_index = new_index

        display_line = self._display_lines[new_index]
        self.cursor_y = display_line.source_line
        self.cursor_x = min(self.cursor_x, len(self.lines[self.cursor_y]))

        if self._view_top_index is None:
            self._view_top_index = 0

        height = max(1, self._last_text_height)
        if self._view_cursor_index < self._view_top_index:
            self._view_top_index = self._view_cursor_index
        elif self._view_cursor_index >= self._view_top_index + height:
            self._view_top_index = self._view_cursor_index - height + 1

    def _init_colors(self) -> None:
        if not curses.has_colors():
            self._has_colors = False
            return

        self._has_colors = True
        self._color_pairs.clear()
        curses.start_color()
        try:
            curses.use_default_colors()
        except curses.error:
            pass

        insert_attr = self._register_color_pair(
            self._default_fg["insert"], self._default_bg["insert"]
        )
        view_attr = self._register_color_pair(
            self._default_fg["view"], self._default_bg["view"]
        )
        self._text_pairs["insert"] = insert_attr
        self._text_pairs["view"] = view_attr
        self._status_pairs["insert"] = self._register_color_pair(
            curses.COLOR_WHITE, curses.COLOR_MAGENTA
        )
        self._status_pairs["view"] = self._register_color_pair(
            curses.COLOR_BLACK, curses.COLOR_WHITE
        )

    def _register_color_pair(self, fg: int | None, bg: int | None) -> int:
        if not self._has_colors:
            return curses.A_NORMAL

        fg_code = fg if fg is not None else -1
        bg_code = bg if bg is not None else -1
        key = (fg_code, bg_code)
        existing = self._color_pairs.get(key)
        if existing is not None:
            return curses.color_pair(existing)

        pair_id = len(self._color_pairs) + 1
        if pair_id >= curses.COLOR_PAIRS:
            pair_id = next(iter(self._color_pairs.values()), 0)
        else:
            curses.init_pair(pair_id, fg_code, bg_code)
        self._color_pairs[key] = pair_id
        return curses.color_pair(pair_id)

    def _style_to_attr(self, style: Style | None, mode: str) -> int:
        base_attr = self._text_pairs.get(mode, curses.A_NORMAL)
        if not self._has_colors or style is None:
            return base_attr

        attr = base_attr
        if style.bold:
            attr |= curses.A_BOLD
        if style.underline:
            attr |= curses.A_UNDERLINE
        if _HAS_ITALIC and style.italic:
            attr |= _ITALIC_ATTR
        if style.reverse:
            attr |= curses.A_REVERSE

        fg = self._style_color(style.color)
        bg = self._style_color(style.bgcolor)
        if style.reverse:
            fg, bg = bg, fg

        if fg is not None or bg is not None:
            attr = (attr & ~curses.A_COLOR) | self._register_color_pair(
                fg if fg is not None else self._default_fg.get(mode),
                bg if bg is not None else self._default_bg.get(mode),
            )
        return attr

    def _style_color(self, color: object | None) -> int | None:
        if color is None:
            return None

        name = color.name if hasattr(color, "name") else None
        if name:
            name = name.lower()
            if name.startswith("bright_"):
                name = name.replace("bright_", "", 1)
            if name in self._COLOR_NAME_MAP:
                return self._COLOR_NAME_MAP[name]

        number = getattr(color, "number", None)
        if isinstance(number, int) and 0 <= number <= 7:
            return number
        return None

    def _ensure_view_cache(self, text_width: int) -> None:
        if not self._view_cache_dirty and self._view_cache_width == text_width:
            return

        # ``force_terminal=True`` keeps render_lines producing width-correct
        # segments offscreen (the editor re-emits them through curses), but it
        # must not override NO_COLOR — so drop the color system entirely when
        # color is disabled while keeping the same width measurement.
        console = (
            Console(width=text_width, color_system=None, force_terminal=True)
            if self._no_color_active()
            else Console(width=text_width, color_system="standard", force_terminal=True)
        )
        options = console.options.update(width=text_width)

        self._display_lines = []
        for index, line in enumerate(self.lines or [""]):
            renderable = PlainMarkdown(line or " ")
            rendered_rows = list(console.render_lines(renderable, options))
            cleaned_rows: list[list[Segment]] = []
            for row in rendered_rows:
                combined = "".join(segment.text for segment in row)
                if not combined.strip():
                    if cleaned_rows:
                        cleaned_rows.append([Segment("", None)])
                    continue
                cleaned_rows.append(row)

            if not cleaned_rows:
                cleaned_rows = [[Segment("", None)]]

            for row in cleaned_rows:
                trimmed = self._trim_segments(row, text_width)
                self._display_lines.append(DisplayLine(trimmed, index))

        self._view_cache_width = text_width
        self._view_cache_dirty = False
        self._view_cursor_index = None
        self._view_top_index = None

    def _trim_segments(self, segments: list[Segment], text_width: int) -> list[Segment]:
        trimmed: list[Segment] = []
        remaining = text_width
        for segment in segments:
            text = segment.text
            if not text:
                continue
            if remaining <= 0:
                break
            snippet = text[:remaining]
            trimmed.append(Segment(snippet, segment.style))
            remaining -= len(snippet)
        if not trimmed:
            trimmed.append(Segment("", None))
        return trimmed

    def _ensure_view_positions(self, text_height: int) -> None:
        if not self._display_lines:
            self._display_lines = [DisplayLine([Segment("", None)], 0)]

        if self._view_cursor_index is None:
            target_line = min(
                max(self.cursor_y, 0), len(self.lines) - 1 if self.lines else 0
            )
            self._view_cursor_index = next(
                (
                    i
                    for i, disp in enumerate(self._display_lines)
                    if disp.source_line == target_line
                ),
                0,
            )

        if self._view_top_index is None:
            self._view_top_index = max(
                0, min(self._view_cursor_index, len(self._display_lines) - 1)
            )

        max_top = max(0, len(self._display_lines) - text_height)
        self._view_top_index = max(0, min(self._view_top_index, max_top))

        if self._view_cursor_index < self._view_top_index:
            self._view_top_index = self._view_cursor_index
        elif self._view_cursor_index >= self._view_top_index + text_height:
            self._view_top_index = self._view_cursor_index - text_height + 1

        display_line = self._display_lines[self._view_cursor_index]
        self.cursor_y = display_line.source_line
        self.cursor_x = (
            min(self.cursor_x, len(self.lines[self.cursor_y])) if self.lines else 0
        )

    def _invalidate_view_cache(self) -> None:
        self._view_cache_dirty = True

    def _sync_insert_positions(self) -> None:
        if not self._display_lines or self._view_top_index is None:
            return

        top_index = min(self._view_top_index, len(self._display_lines) - 1)
        top_display = self._display_lines[top_index]
        self.top_line = top_display.source_line

    def _no_color_active(self) -> bool:
        """Whether color is disabled via NO_COLOR or the --no-color/--plain flag."""
        import os

        if os.environ.get("NO_COLOR"):
            return True
        try:
            from chirp._console import stdout_console

            return bool(stdout_console.no_color)
        except Exception:  # noqa: BLE001 - color detection must never break the editor
            return False

    def _notify_readonly(self) -> None:
        if not self._readonly_notified:
            self.message = "Read-only note: edits disabled"
            self._readonly_notified = True
            try:
                curses.beep()
            except curses.error:
                pass
        else:
            self.message = "Read-only note"
