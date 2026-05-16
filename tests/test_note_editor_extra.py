"""Extended unit tests for notes/note_editor.py targeting >= 80% coverage."""

from __future__ import annotations

import curses
from unittest.mock import MagicMock, patch

from rich.segment import Segment
from rich.style import Style

from notes.note_editor import (
    DisplayLine,
    EditorResult,
    ManualNoteEditor,
    PlainHeading,
    PlainMarkdown,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_editor(
    content: str = "Hello\nWorld\n", readonly: bool = False
) -> ManualNoteEditor:
    return ManualNoteEditor("Test", content, readonly=readonly)


# ---------------------------------------------------------------------------
# EditorResult / PlainHeading / PlainMarkdown basic construction
# ---------------------------------------------------------------------------


def test_editor_result_fields():
    result = EditorResult(content="abc", saved=True)
    assert result.content == "abc"
    assert result.saved is True


def test_display_line_fields():
    segs = [Segment("hi", None)]
    dl = DisplayLine(segments=segs, source_line=3)
    assert dl.source_line == 3
    assert dl.segments is segs


def test_plain_markdown_elements_override():
    assert "heading_open" in PlainMarkdown.elements
    assert PlainMarkdown.elements["heading_open"] is PlainHeading


# ---------------------------------------------------------------------------
# _initialise_lines
# ---------------------------------------------------------------------------


def test_initialise_lines_empty_content():
    editor = ManualNoteEditor("T", "")
    assert editor.lines == [""]


def test_initialise_lines_with_trailing_newline():
    editor = ManualNoteEditor("T", "a\nb\n")
    assert editor.lines == ["a", "b", ""]


def test_initialise_lines_without_trailing_newline():
    editor = ManualNoteEditor("T", "a\nb")
    assert editor.lines == ["a", "b", ""]


def test_initialise_lines_single_line():
    editor = ManualNoteEditor("T", "hello")
    assert editor.lines == ["hello", ""]


# ---------------------------------------------------------------------------
# __init__ modes
# ---------------------------------------------------------------------------


def test_start_in_insert_respects_readonly():
    editor = ManualNoteEditor("T", "x", readonly=True, start_in_insert=True)
    assert editor.mode == "view"


def test_start_in_insert_sets_insert_mode():
    editor = ManualNoteEditor("T", "x", start_in_insert=True)
    assert editor.mode == "insert"


def test_default_mode_is_view():
    editor = make_editor()
    assert editor.mode == "view"


# ---------------------------------------------------------------------------
# run() — curses.wrapper patch
# ---------------------------------------------------------------------------


def test_run_appends_newline_if_missing():
    editor = ManualNoteEditor("T", "abc")
    editor.lines = ["abc"]

    with patch("notes.note_editor.curses.wrapper", lambda fn: None):
        result = editor.run()

    assert result.content.endswith("\n")


def test_run_preserves_existing_newline():
    editor = ManualNoteEditor("T", "abc\n")
    editor.saved = True

    with patch("notes.note_editor.curses.wrapper", lambda fn: None):
        result = editor.run()

    assert result.saved is True
    assert result.content.endswith("\n")


# ---------------------------------------------------------------------------
# _status_line
# ---------------------------------------------------------------------------


def test_status_line_insert_mode():
    editor = make_editor()
    editor.mode = "insert"
    status = editor._status_line(80)
    assert "INSERT" in status


def test_status_line_view_mode():
    editor = make_editor()
    editor.mode = "view"
    status = editor._status_line(80)
    assert "VIEW" in status


def test_status_line_dirty_flag():
    editor = make_editor()
    editor.dirty = True
    status = editor._status_line(80)
    assert "*" in status


def test_status_line_message_shown():
    editor = make_editor()
    editor.message = "saved!"
    status = editor._status_line(80)
    assert "saved!" in status


def test_status_line_command_active():
    editor = make_editor()
    editor.command_active = True
    editor.command_buffer = ":wq"
    status = editor._status_line(80)
    assert ":wq" in status


def test_status_line_truncated_to_width():
    editor = make_editor()
    status = editor._status_line(10)
    assert len(status) <= 9


# ---------------------------------------------------------------------------
# _adjust_insert_viewport
# ---------------------------------------------------------------------------


def test_adjust_viewport_scrolls_down():
    editor = make_editor()
    editor.cursor_y = 10
    editor.top_line = 0
    editor._adjust_insert_viewport(5)
    assert editor.top_line == 6


def test_adjust_viewport_scrolls_up():
    editor = make_editor()
    editor.cursor_y = 2
    editor.top_line = 5
    editor._adjust_insert_viewport(10)
    assert editor.top_line == 2


def test_adjust_viewport_no_change_needed():
    editor = make_editor()
    editor.cursor_y = 3
    editor.top_line = 0
    editor._adjust_insert_viewport(10)
    assert editor.top_line == 0


# ---------------------------------------------------------------------------
# _ensure_cursor_within_bounds
# ---------------------------------------------------------------------------


def test_ensure_cursor_clamps_y_above_max():
    editor = make_editor("a\nb\n")
    editor.cursor_y = 100
    editor._ensure_cursor_within_bounds()
    assert editor.cursor_y <= len(editor.lines) - 1


def test_ensure_cursor_clamps_x():
    editor = make_editor("hi\n")
    editor.cursor_y = 0
    editor.cursor_x = 100
    editor._ensure_cursor_within_bounds()
    assert editor.cursor_x <= len(editor.lines[0])


def test_ensure_cursor_with_empty_lines():
    editor = make_editor()
    editor.lines = []
    editor._ensure_cursor_within_bounds()
    assert editor.lines == [""]


# ---------------------------------------------------------------------------
# _handle_global_keys
# ---------------------------------------------------------------------------


def test_global_keys_escape_from_insert_switches_to_view():
    editor = make_editor()
    editor.mode = "insert"
    result = editor._handle_global_keys("\x1b")
    assert editor.mode == "view"
    assert result is False


def test_global_keys_escape_from_view_clears_command():
    editor = make_editor()
    editor.mode = "view"
    editor.command_active = True
    editor.command_buffer = ":w"
    result = editor._handle_global_keys("\x1b")
    assert editor.command_active is False
    assert editor.command_buffer == ""
    assert result is False


def test_global_keys_integer_escape():
    editor = make_editor()
    editor.mode = "insert"
    result = editor._handle_global_keys(27)
    assert editor.mode == "view"
    assert result is False


def test_global_keys_delegates_to_insert_mode():
    editor = make_editor()
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 0
    result = editor._handle_global_keys("a")
    assert editor.lines[0].startswith("a")
    assert result is False


def test_global_keys_delegates_to_view_mode():
    editor = make_editor("hello\n")
    editor.mode = "view"
    result = editor._handle_global_keys(":")
    assert editor.command_active is True
    assert result is False


# ---------------------------------------------------------------------------
# _handle_view_mode
# ---------------------------------------------------------------------------


def test_view_mode_enter_insert():
    editor = make_editor()
    result = editor._handle_view_mode("i")
    assert editor.mode == "insert"
    assert result is False


def test_view_mode_colon_activates_command():
    editor = make_editor()
    result = editor._handle_view_mode(":")
    assert editor.command_active is True
    assert editor.command_buffer == ":"
    assert result is False


def test_view_mode_readonly_blocks_i():
    editor = make_editor(readonly=True)
    with patch("notes.note_editor.curses.beep"):
        result = editor._handle_view_mode("i")
    assert editor.mode == "view"
    assert result is False


# ---------------------------------------------------------------------------
# _handle_insert_mode
# ---------------------------------------------------------------------------


def test_insert_mode_printable_char():
    editor = make_editor()
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 0
    editor._handle_insert_mode("Z")
    assert editor.lines[0][0] == "Z"
    assert editor.dirty is True


def test_insert_mode_newline_str():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 3
    editor._handle_insert_mode("\n")
    assert editor.lines[0] == "abc"
    assert editor.lines[1] == ""


def test_insert_mode_carriage_return():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 2
    editor._handle_insert_mode("\r")
    assert len(editor.lines) >= 2


def test_insert_mode_tab_inserts_spaces():
    editor = make_editor()
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 0
    editor._handle_insert_mode("\t")
    assert editor.lines[0].startswith("    ")


def test_insert_mode_backspace_delete_str():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 3
    editor._handle_insert_mode("\x7f")
    assert editor.lines[0] == "ab"


def test_insert_mode_backspace_b():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 2
    editor._handle_insert_mode("\b")
    assert editor.lines[0] == "ac"


def test_insert_mode_key_backspace_int():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 1
    editor._handle_insert_mode(curses.KEY_BACKSPACE)
    assert editor.lines[0] == "bc"


def test_insert_mode_key_backspace_263():
    editor = make_editor("xyz\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 3
    editor._handle_insert_mode(263)
    assert editor.lines[0] == "xy"


def test_insert_mode_key_enter():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 1
    editor._handle_insert_mode(curses.KEY_ENTER)
    assert len(editor.lines) >= 2


def test_insert_mode_int_10():
    editor = make_editor("hello\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 2
    editor._handle_insert_mode(10)
    assert editor.cursor_x == 0


def test_insert_mode_key_dc():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 0
    editor._handle_insert_mode(curses.KEY_DC)
    assert editor.lines[0][0] == "b"


# ---------------------------------------------------------------------------
# _handle_navigation
# ---------------------------------------------------------------------------


def test_navigation_key_left():
    editor = make_editor()
    editor.mode = "view"
    editor.cursor_x = 3
    editor._handle_navigation(curses.KEY_LEFT)
    assert editor.cursor_x == 2


def test_navigation_h_key():
    editor = make_editor()
    editor.mode = "view"
    editor.cursor_x = 5
    editor._handle_navigation("h")
    assert editor.cursor_x == 4


def test_navigation_key_right():
    editor = make_editor("abc\n")
    editor.mode = "view"
    editor.cursor_y = 0
    editor.cursor_x = 0
    editor._handle_navigation(curses.KEY_RIGHT)
    assert editor.cursor_x == 1


def test_navigation_l_key():
    editor = make_editor("abc\n")
    editor.mode = "view"
    editor.cursor_y = 0
    editor.cursor_x = 1
    editor._handle_navigation("l")
    assert editor.cursor_x == 2


def test_navigation_key_up():
    editor = make_editor("a\nb\nc\n")
    editor.mode = "insert"
    editor.cursor_y = 2
    editor._handle_navigation(curses.KEY_UP)
    assert editor.cursor_y == 1


def test_navigation_k_key():
    editor = make_editor("a\nb\n")
    editor.mode = "insert"
    editor.cursor_y = 1
    editor._handle_navigation("k")
    assert editor.cursor_y == 0


def test_navigation_key_down():
    editor = make_editor("a\nb\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor._handle_navigation(curses.KEY_DOWN)
    assert editor.cursor_y == 1


def test_navigation_j_key():
    editor = make_editor("a\nb\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor._handle_navigation("j")
    assert editor.cursor_y == 1


def test_navigation_home():
    editor = make_editor()
    editor.cursor_x = 5
    editor._handle_navigation(curses.KEY_HOME)
    assert editor.cursor_x == 0


def test_navigation_end():
    editor = make_editor("hello\n")
    editor.cursor_y = 0
    editor._handle_navigation(curses.KEY_END)
    assert editor.cursor_x == 5


def test_navigation_page_up_view():
    editor = make_editor("a\nb\nc\nd\ne\n")
    editor.mode = "view"
    editor._last_text_height = 2
    editor._ensure_view_cache(80)
    editor._ensure_view_positions(2)
    editor._handle_navigation(curses.KEY_PPAGE)


def test_navigation_page_down_view():
    editor = make_editor("a\nb\nc\nd\ne\n")
    editor.mode = "view"
    editor._last_text_height = 2
    editor._ensure_view_cache(80)
    editor._ensure_view_positions(2)
    editor._handle_navigation(curses.KEY_NPAGE)


def test_navigation_page_up_insert():
    editor = make_editor("a\nb\nc\nd\ne\n")
    editor.mode = "insert"
    editor.cursor_y = 4
    editor._last_text_height = 3
    editor._handle_navigation(curses.KEY_PPAGE)
    assert editor.cursor_y == 1


def test_navigation_page_down_insert():
    editor = make_editor("a\nb\nc\nd\ne\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor._last_text_height = 3
    editor._handle_navigation(curses.KEY_NPAGE)
    assert editor.cursor_y == 3


# ---------------------------------------------------------------------------
# _handle_command_input
# ---------------------------------------------------------------------------


def test_command_input_str_newline_executes():
    editor = make_editor()
    editor.command_buffer = ":wq"
    result = editor._handle_command_input("\n")
    assert result is True
    assert editor.saved is True


def test_command_input_carriage_return_executes():
    editor = make_editor()
    editor.command_buffer = ":q!"
    result = editor._handle_command_input("\r")
    assert result is True
    assert editor.saved is False


def test_command_input_int_enter_executes():
    editor = make_editor()
    editor.command_buffer = ":wq"
    result = editor._handle_command_input(curses.KEY_ENTER)
    assert result is True


def test_command_input_int_10_executes():
    editor = make_editor()
    editor.command_buffer = ":wq"
    result = editor._handle_command_input(10)
    assert result is True


def test_command_input_escape_cancels():
    editor = make_editor()
    editor.command_active = True
    editor.command_buffer = ":w"
    result = editor._handle_command_input("\x1b")
    assert editor.command_active is False
    assert editor.command_buffer == ""
    assert result is False


def test_command_input_escape_int():
    editor = make_editor()
    editor.command_active = True
    editor.command_buffer = ":abc"
    editor._handle_command_input(27)
    assert editor.command_active is False


def test_command_input_backspace_shrinks_buffer():
    editor = make_editor()
    editor.command_buffer = ":wq"
    editor._handle_command_input(curses.KEY_BACKSPACE)
    assert editor.command_buffer == ":w"


def test_command_input_backspace_clears_when_single_char():
    editor = make_editor()
    editor.command_active = True
    editor.command_buffer = ":"
    editor._handle_command_input("\x7f")
    assert editor.command_active is False
    assert editor.command_buffer == ""


def test_command_input_263_backspace():
    editor = make_editor()
    editor.command_buffer = ":abc"
    editor._handle_command_input(263)
    assert editor.command_buffer == ":ab"


def test_command_input_backspace_b():
    editor = make_editor()
    editor.command_buffer = ":xy"
    editor._handle_command_input("\b")
    assert editor.command_buffer == ":x"


def test_command_input_printable_appends():
    editor = make_editor()
    editor.command_buffer = ":"
    editor._handle_command_input("w")
    assert editor.command_buffer == ":w"


def test_command_input_non_printable_ignored():
    editor = make_editor()
    editor.command_buffer = ":"
    result = editor._handle_command_input(999)
    assert editor.command_buffer == ":"
    assert result is False


# ---------------------------------------------------------------------------
# _execute_command
# ---------------------------------------------------------------------------


def test_execute_command_wq_saves():
    editor = make_editor()
    editor.command_buffer = ":wq"
    result = editor._execute_command()
    assert result is True
    assert editor.saved is True


def test_execute_command_q_bang_discards():
    editor = make_editor()
    editor.command_buffer = ":q!"
    editor.dirty = True
    result = editor._execute_command()
    assert result is True
    assert editor.saved is False
    assert editor.dirty is False


def test_execute_command_unknown_sets_message():
    editor = make_editor()
    editor.command_buffer = ":foo"
    result = editor._execute_command()
    assert result is False
    assert "foo" in editor.message


def test_execute_command_empty_clears_message():
    editor = make_editor()
    editor.command_buffer = ":"
    result = editor._execute_command()
    assert result is False
    assert editor.message == ""


# ---------------------------------------------------------------------------
# _insert_text
# ---------------------------------------------------------------------------


def test_insert_text_mid_line():
    editor = make_editor("hello\n")
    editor.cursor_y = 0
    editor.cursor_x = 2
    editor._insert_text("XY")
    assert editor.lines[0] == "heXYllo"
    assert editor.cursor_x == 4
    assert editor.dirty is True


def test_insert_text_invalidates_cache():
    editor = make_editor("abc\n")
    editor._view_cache_dirty = False
    editor.cursor_y = 0
    editor.cursor_x = 0
    editor._insert_text("Z")
    assert editor._view_cache_dirty is True


# ---------------------------------------------------------------------------
# _insert_newline
# ---------------------------------------------------------------------------


def test_insert_newline_splits_line():
    editor = make_editor("abcdef\n")
    editor.cursor_y = 0
    editor.cursor_x = 3
    editor._insert_newline()
    assert editor.lines[0] == "abc"
    assert editor.lines[1] == "def"
    assert editor.cursor_y == 1
    assert editor.cursor_x == 0
    assert editor.dirty is True


def test_insert_newline_at_end():
    editor = make_editor("abc\n")
    editor.cursor_y = 0
    editor.cursor_x = 3
    editor._insert_newline()
    assert editor.lines[0] == "abc"
    assert editor.lines[1] == ""


# ---------------------------------------------------------------------------
# _backspace
# ---------------------------------------------------------------------------


def test_backspace_deletes_char():
    editor = make_editor("hello\n")
    editor.cursor_y = 0
    editor.cursor_x = 3
    editor._backspace()
    assert editor.lines[0] == "helo"
    assert editor.cursor_x == 2


def test_backspace_at_col_zero_joins_lines():
    editor = make_editor("first\nsecond\n")
    editor.cursor_y = 1
    editor.cursor_x = 0
    editor._backspace()
    assert editor.lines[0] == "firstsecond"
    assert editor.cursor_y == 0
    assert editor.cursor_x == 5


def test_backspace_noop_at_start_of_first_line():
    editor = make_editor("hello\n")
    editor.cursor_y = 0
    editor.cursor_x = 0
    editor._backspace()
    assert editor.lines[0] == "hello"
    assert editor.cursor_y == 0
    assert editor.cursor_x == 0


# ---------------------------------------------------------------------------
# _delete
# ---------------------------------------------------------------------------


def test_delete_removes_char_at_cursor():
    editor = make_editor("hello\n")
    editor.cursor_y = 0
    editor.cursor_x = 1
    editor._delete()
    assert editor.lines[0] == "hllo"
    assert editor.dirty is True


def test_delete_at_end_joins_next_line():
    editor = make_editor("abc\ndef\n")
    editor.cursor_y = 0
    editor.cursor_x = 3
    editor._delete()
    assert editor.lines[0] == "abcdef"


def test_delete_noop_at_last_line_end():
    editor = make_editor("only\n")
    # Position cursor at the end of the last non-empty content line.
    # lines are ["only", ""] — cursor_y=1, cursor_x=0 is end of final line.
    editor.cursor_y = 1
    editor.cursor_x = 0
    editor._delete()
    assert editor.lines == ["only", ""]


# ---------------------------------------------------------------------------
# _move_left
# ---------------------------------------------------------------------------


def test_move_left_view_mode():
    editor = make_editor()
    editor.mode = "view"
    editor.cursor_x = 3
    editor._move_left()
    assert editor.cursor_x == 2


def test_move_left_view_clamps_at_zero():
    editor = make_editor()
    editor.mode = "view"
    editor.cursor_x = 0
    editor._move_left()
    assert editor.cursor_x == 0


def test_move_left_insert_wraps_to_previous_line():
    editor = make_editor("abc\ndef\n")
    editor.mode = "insert"
    editor.cursor_y = 1
    editor.cursor_x = 0
    editor._move_left()
    assert editor.cursor_y == 0
    assert editor.cursor_x == 3


def test_move_left_insert_within_line():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 2
    editor._move_left()
    assert editor.cursor_x == 1


def test_move_left_insert_no_wrap_at_start():
    editor = make_editor("abc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 0
    editor._move_left()
    assert editor.cursor_y == 0
    assert editor.cursor_x == 0


# ---------------------------------------------------------------------------
# _move_right
# ---------------------------------------------------------------------------


def test_move_right_within_line():
    editor = make_editor("hello\n")
    editor.cursor_y = 0
    editor.cursor_x = 0
    editor._move_right()
    assert editor.cursor_x == 1


def test_move_right_view_does_not_wrap():
    editor = make_editor("hi\n")
    editor.mode = "view"
    editor.cursor_y = 0
    editor.cursor_x = 2
    editor._move_right()
    assert editor.cursor_y == 0


def test_move_right_insert_wraps_to_next_line():
    editor = make_editor("ab\ncd\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 2
    editor._move_right()
    assert editor.cursor_y == 1
    assert editor.cursor_x == 0


def test_move_right_insert_no_wrap_at_last_line():
    editor = make_editor("hi\n")
    editor.mode = "insert"
    editor.cursor_y = 1
    editor.cursor_x = 0
    before = editor.cursor_y
    editor._move_right()
    assert editor.cursor_y == before


# ---------------------------------------------------------------------------
# _move_up
# ---------------------------------------------------------------------------


def test_move_up_insert_mode():
    editor = make_editor("a\nb\nc\n")
    editor.mode = "insert"
    editor.cursor_y = 2
    editor.cursor_x = 0
    editor._move_up()
    assert editor.cursor_y == 1


def test_move_up_insert_clamps_x():
    editor = make_editor("hi\nlonger\n")
    editor.mode = "insert"
    editor.cursor_y = 1
    editor.cursor_x = 5
    editor._move_up()
    assert editor.cursor_y == 0
    assert editor.cursor_x <= len("hi")


def test_move_up_at_top_noop():
    editor = make_editor("only\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor._move_up()
    assert editor.cursor_y == 0


def test_move_up_view_calls_view_cursor():
    editor = make_editor("a\nb\nc\nd\ne\n")
    editor.mode = "view"
    editor._ensure_view_cache(80)
    editor._ensure_view_positions(10)
    editor._view_cursor_index = 3
    editor._move_up()
    assert editor._view_cursor_index == 2


# ---------------------------------------------------------------------------
# _move_down
# ---------------------------------------------------------------------------


def test_move_down_insert_mode():
    editor = make_editor("a\nb\nc\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor._move_down()
    assert editor.cursor_y == 1


def test_move_down_at_bottom_noop():
    editor = make_editor("only\n")
    editor.mode = "insert"
    editor.cursor_y = len(editor.lines) - 1
    before = editor.cursor_y
    editor._move_down()
    assert editor.cursor_y == before


def test_move_down_view_calls_view_cursor():
    editor = make_editor("a\nb\nc\nd\ne\n")
    editor.mode = "view"
    editor._ensure_view_cache(80)
    editor._ensure_view_positions(10)
    editor._view_cursor_index = 1
    editor._move_down()
    assert editor._view_cursor_index == 2


# ---------------------------------------------------------------------------
# _page_up / _page_down
# ---------------------------------------------------------------------------


def test_page_up_insert():
    editor = make_editor("\n".join(["line"] * 20) + "\n")
    editor.mode = "insert"
    editor.cursor_y = 15
    editor._last_text_height = 5
    editor._page_up()
    assert editor.cursor_y == 10


def test_page_up_insert_clamps_at_zero():
    editor = make_editor("a\nb\n")
    editor.mode = "insert"
    editor.cursor_y = 1
    editor._last_text_height = 10
    editor._page_up()
    assert editor.cursor_y == 0


def test_page_down_insert():
    editor = make_editor("\n".join(["x"] * 20) + "\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor._last_text_height = 5
    editor._page_down()
    assert editor.cursor_y == 5


def test_page_down_insert_clamps_at_last():
    editor = make_editor("a\nb\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor._last_text_height = 50
    editor._page_down()
    assert editor.cursor_y == len(editor.lines) - 1


# ---------------------------------------------------------------------------
# _move_view_cursor
# ---------------------------------------------------------------------------


def test_move_view_cursor_no_display_lines():
    editor = make_editor()
    editor._display_lines = []
    editor._move_view_cursor(1)
    assert editor._view_cursor_index is None


def test_move_view_cursor_initialises_from_none():
    editor = make_editor("a\nb\nc\n")
    editor._ensure_view_cache(80)
    editor._view_cursor_index = None
    editor._view_top_index = None
    editor._move_view_cursor(1)
    assert editor._view_cursor_index == 1


def test_move_view_cursor_clamps_at_zero():
    editor = make_editor("a\nb\n")
    editor._ensure_view_cache(80)
    editor._view_cursor_index = 0
    editor._view_top_index = 0
    editor._move_view_cursor(-10)
    assert editor._view_cursor_index == 0


def test_move_view_cursor_scrolls_top_index():
    editor = make_editor("\n".join(["line"] * 10) + "\n")
    editor._ensure_view_cache(80)
    editor._view_cursor_index = 0
    editor._view_top_index = 0
    editor._last_text_height = 3
    editor._move_view_cursor(5)
    assert editor._view_top_index >= 0


# ---------------------------------------------------------------------------
# _notify_readonly
# ---------------------------------------------------------------------------


def test_notify_readonly_first_call():
    editor = make_editor(readonly=True)
    with patch("notes.note_editor.curses.beep"):
        editor._notify_readonly()
    assert editor._readonly_notified is True
    assert "Read-only note" in editor.message
    assert "edits disabled" in editor.message


def test_notify_readonly_second_call():
    editor = make_editor(readonly=True)
    with patch("notes.note_editor.curses.beep"):
        editor._notify_readonly()
    editor._notify_readonly()
    assert editor.message == "Read-only note"


def test_notify_readonly_beep_error_suppressed():
    editor = make_editor(readonly=True)
    with patch("notes.note_editor.curses.beep", side_effect=curses.error):
        editor._notify_readonly()
    assert editor._readonly_notified is True


# ---------------------------------------------------------------------------
# _invalidate_view_cache
# ---------------------------------------------------------------------------


def test_invalidate_view_cache_sets_dirty():
    editor = make_editor()
    editor._view_cache_dirty = False
    editor._invalidate_view_cache()
    assert editor._view_cache_dirty is True


# ---------------------------------------------------------------------------
# _sync_insert_positions
# ---------------------------------------------------------------------------


def test_sync_insert_positions_no_display_lines():
    editor = make_editor()
    editor._display_lines = []
    editor._view_top_index = 0
    editor.top_line = 99
    editor._sync_insert_positions()
    assert editor.top_line == 99


def test_sync_insert_positions_none_top_index():
    editor = make_editor("a\nb\n")
    editor._ensure_view_cache(80)
    editor._view_top_index = None
    original_top = editor.top_line
    editor._sync_insert_positions()
    assert editor.top_line == original_top


def test_sync_insert_positions_sets_top_line():
    editor = make_editor("a\nb\nc\nd\n")
    editor._ensure_view_cache(80)
    editor._view_top_index = 2
    editor._sync_insert_positions()
    assert editor.top_line == editor._display_lines[2].source_line


# ---------------------------------------------------------------------------
# _ensure_view_cache
# ---------------------------------------------------------------------------


def test_ensure_view_cache_builds_display_lines():
    editor = make_editor("## Heading\nSome text\n")
    editor._ensure_view_cache(80)
    assert len(editor._display_lines) > 0
    assert editor._view_cache_dirty is False
    assert editor._view_cache_width == 80


def test_ensure_view_cache_skips_rebuild_if_clean():
    editor = make_editor("hello\n")
    editor._ensure_view_cache(80)
    original = editor._display_lines
    editor._view_cache_dirty = False
    editor._ensure_view_cache(80)
    assert editor._display_lines is original


def test_ensure_view_cache_rebuilds_on_width_change():
    editor = make_editor("hello\n")
    editor._ensure_view_cache(80)
    editor._view_cache_dirty = False
    editor._ensure_view_cache(40)
    assert editor._view_cache_width == 40


def test_ensure_view_cache_empty_content():
    editor = ManualNoteEditor("T", "")
    editor._ensure_view_cache(80)
    assert len(editor._display_lines) > 0


# ---------------------------------------------------------------------------
# _trim_segments
# ---------------------------------------------------------------------------


def test_trim_segments_basic():
    editor = make_editor()
    segments = [Segment("hello world", None)]
    result = editor._trim_segments(segments, 5)
    assert "".join(s.text for s in result) == "hello"


def test_trim_segments_empty_segment_skipped():
    editor = make_editor()
    segments = [Segment("", None), Segment("abc", None)]
    result = editor._trim_segments(segments, 10)
    assert "".join(s.text for s in result) == "abc"


def test_trim_segments_no_segments_returns_empty_segment():
    editor = make_editor()
    result = editor._trim_segments([], 10)
    assert len(result) == 1
    assert result[0].text == ""


def test_trim_segments_preserves_style():
    editor = make_editor()
    style = Style(bold=True)
    segments = [Segment("hi", style)]
    result = editor._trim_segments(segments, 10)
    assert result[0].style == style


# ---------------------------------------------------------------------------
# _ensure_view_positions
# ---------------------------------------------------------------------------


def test_ensure_view_positions_initialises_indices():
    editor = make_editor("a\nb\nc\n")
    editor._ensure_view_cache(80)
    editor._view_cursor_index = None
    editor._view_top_index = None
    editor._ensure_view_positions(10)
    assert editor._view_cursor_index is not None
    assert editor._view_top_index is not None


def test_ensure_view_positions_adjusts_top_index():
    editor = make_editor("\n".join(["x"] * 20) + "\n")
    editor._ensure_view_cache(80)
    editor._view_cursor_index = 15
    editor._view_top_index = 0
    editor._ensure_view_positions(5)
    assert editor._view_top_index <= editor._view_cursor_index


def test_ensure_view_positions_empty_display_lines():
    editor = make_editor("x\n")
    editor._display_lines = []
    editor._view_cursor_index = None
    editor._view_top_index = None
    editor._ensure_view_positions(10)
    assert len(editor._display_lines) == 1


# ---------------------------------------------------------------------------
# _style_color
# ---------------------------------------------------------------------------


def test_style_color_none_returns_none():
    editor = make_editor()
    assert editor._style_color(None) is None


def test_style_color_by_name():
    editor = make_editor()
    color = MagicMock()
    color.name = "red"
    color.number = None
    result = editor._style_color(color)
    assert result == curses.COLOR_RED


def test_style_color_bright_stripped():
    editor = make_editor()
    color = MagicMock()
    color.name = "bright_green"
    color.number = None
    result = editor._style_color(color)
    assert result == curses.COLOR_GREEN


def test_style_color_unknown_name_falls_to_number():
    editor = make_editor()
    color = MagicMock()
    color.name = "unknown_color"
    color.number = 3
    result = editor._style_color(color)
    assert result == 3


def test_style_color_number_out_of_range():
    editor = make_editor()
    color = MagicMock()
    color.name = None
    color.number = 100
    result = editor._style_color(color)
    assert result is None


def test_style_color_number_zero():
    editor = make_editor()
    color = MagicMock()
    color.name = None
    color.number = 0
    result = editor._style_color(color)
    assert result == 0


def test_style_color_no_name_attr():
    editor = make_editor()

    class NoName:
        number = 5

    result = editor._style_color(NoName())
    assert result == 5


# ---------------------------------------------------------------------------
# _style_to_attr (no colors)
# ---------------------------------------------------------------------------


def test_style_to_attr_no_colors_returns_base():
    editor = make_editor()
    editor._has_colors = False
    result = editor._style_to_attr(None, "view")
    assert result == editor._text_pairs.get("view", curses.A_NORMAL)


def test_style_to_attr_none_style_returns_base():
    editor = make_editor()
    editor._has_colors = True
    result = editor._style_to_attr(None, "view")
    assert result == editor._text_pairs.get("view", curses.A_NORMAL)


def test_style_to_attr_bold():
    editor = make_editor()
    editor._has_colors = True
    style = Style(bold=True)
    result = editor._style_to_attr(style, "view")
    assert result & curses.A_BOLD


def test_style_to_attr_underline():
    editor = make_editor()
    editor._has_colors = True
    style = Style(underline=True)
    result = editor._style_to_attr(style, "view")
    assert result & curses.A_UNDERLINE


def test_style_to_attr_reverse():
    editor = make_editor()
    editor._has_colors = True
    style = Style(reverse=True)
    result = editor._style_to_attr(style, "view")
    assert result & curses.A_REVERSE


# ---------------------------------------------------------------------------
# _register_color_pair
# ---------------------------------------------------------------------------


def test_register_color_pair_no_colors():
    editor = make_editor()
    editor._has_colors = False
    result = editor._register_color_pair(0, 0)
    assert result == curses.A_NORMAL


def test_register_color_pair_returns_existing():
    editor = make_editor()
    editor._has_colors = True
    editor._color_pairs = {(1, 2): 5}
    with patch("notes.note_editor.curses.color_pair", return_value=42):
        result = editor._register_color_pair(1, 2)
    assert result == 42


def test_register_color_pair_creates_new():
    import curses as _curses

    editor = make_editor()
    editor._has_colors = True
    editor._color_pairs = {}
    with (
        patch("notes.note_editor.curses.init_pair"),
        patch("notes.note_editor.curses.color_pair", return_value=77),
    ):
        _curses.COLOR_PAIRS = 256
        try:
            result = editor._register_color_pair(1, 2)
        finally:
            try:
                del _curses.COLOR_PAIRS
            except AttributeError:
                pass
    assert result == 77


def test_register_color_pair_exceeds_color_pairs():
    import curses as _curses

    editor = make_editor()
    editor._has_colors = True
    editor._color_pairs = {(0, 0): 255}
    with patch("notes.note_editor.curses.color_pair", return_value=55):
        _curses.COLOR_PAIRS = 1
        try:
            result = editor._register_color_pair(3, 4)
        finally:
            try:
                del _curses.COLOR_PAIRS
            except AttributeError:
                pass
    assert result == 55


# ---------------------------------------------------------------------------
# _init_colors
# ---------------------------------------------------------------------------


def test_init_colors_no_color_support():
    editor = make_editor()
    with patch("notes.note_editor.curses.has_colors", return_value=False):
        editor._init_colors()
    assert editor._has_colors is False


def test_init_colors_with_color_support():
    import curses as _curses

    editor = make_editor()
    _curses.COLOR_PAIRS = 256
    try:
        with (
            patch("notes.note_editor.curses.has_colors", return_value=True),
            patch("notes.note_editor.curses.start_color"),
            patch("notes.note_editor.curses.use_default_colors"),
            patch("notes.note_editor.curses.init_pair"),
            patch("notes.note_editor.curses.color_pair", return_value=1),
        ):
            editor._init_colors()
    finally:
        try:
            del _curses.COLOR_PAIRS
        except AttributeError:
            pass
    assert editor._has_colors is True


def test_init_colors_use_default_colors_error_suppressed():
    import curses as _curses

    editor = make_editor()
    _curses.COLOR_PAIRS = 256
    try:
        with (
            patch("notes.note_editor.curses.has_colors", return_value=True),
            patch("notes.note_editor.curses.start_color"),
            patch(
                "notes.note_editor.curses.use_default_colors", side_effect=curses.error
            ),
            patch("notes.note_editor.curses.init_pair"),
            patch("notes.note_editor.curses.color_pair", return_value=1),
        ):
            editor._init_colors()
    finally:
        try:
            del _curses.COLOR_PAIRS
        except AttributeError:
            pass
    assert editor._has_colors is True


# ---------------------------------------------------------------------------
# PlainHeading.__rich_console__
# ---------------------------------------------------------------------------


def test_plain_heading_renders_without_border():
    from rich.console import Console

    console = Console(width=40, force_terminal=True, color_system=None)
    md = PlainMarkdown("# Hello World")
    with console.capture() as capture:
        console.print(md)
    output = capture.get()
    assert "Hello World" in output
    assert "─" not in output


# ---------------------------------------------------------------------------
# Integration-style: multiple edits then execute :wq
# ---------------------------------------------------------------------------


def test_full_edit_and_save_flow():
    editor = ManualNoteEditor("Doc", "line1\nline2\n")
    editor.mode = "insert"
    editor.cursor_y = 0
    editor.cursor_x = 5
    editor._insert_text("!")
    assert editor.lines[0] == "line1!"
    editor.command_buffer = ":wq"
    editor._execute_command()
    assert editor.saved is True
