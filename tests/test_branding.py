"""Behavior tests for chirp.branding module."""

from rich.text import Text

from chirp.branding import (
    BEAK_CLOSED,
    BEAK_OPEN,
    COMPACT_LOGO,
    LOGO_ACCENT,
    LOGO_NOTE,
    LOGO_ROWS,
    LOGO_YELLOW,
    TAGLINE,
    LogoCell,
    logo_line_plain,
    logo_plain_lines,
    render_logo,
)


def test_constants_are_nonempty_strings():
    for value in (LOGO_YELLOW, LOGO_ACCENT, LOGO_NOTE, TAGLINE, COMPACT_LOGO):
        assert isinstance(value, str)
        assert value


def test_beak_glyphs_are_distinct():
    assert BEAK_CLOSED != BEAK_OPEN


def test_logo_rows_is_nonempty_tuple():
    assert isinstance(LOGO_ROWS, tuple)
    assert len(LOGO_ROWS) > 0


def test_exactly_one_beak_row():
    beak_rows = [row for row in LOGO_ROWS if row.has_beak]
    assert len(beak_rows) == 1


def test_beak_row_also_has_note():
    beak_rows = [row for row in LOGO_ROWS if row.has_beak]
    assert beak_rows[0].has_note


# --- logo_line_plain ---


def test_logo_line_plain_beak_row_closed():
    row = LogoCell("A", "B", has_beak=True, has_note=True)
    result = logo_line_plain(row, beak_open=False)
    assert BEAK_CLOSED in result
    assert BEAK_OPEN not in result
    assert "♪" not in result


def test_logo_line_plain_beak_row_open():
    row = LogoCell("A", "B", has_beak=True, has_note=True)
    result = logo_line_plain(row, beak_open=True)
    assert BEAK_OPEN in result
    assert BEAK_CLOSED not in result
    assert "♪" in result


def test_logo_line_plain_no_beak_no_note():
    row = LogoCell("prefix", "suffix")
    result = logo_line_plain(row, beak_open=True)
    assert result == "prefixsuffix"
    assert "♪" not in result


def test_logo_line_plain_note_suppressed_when_beak_closed():
    row = LogoCell("X", has_beak=True, has_note=True)
    result = logo_line_plain(row, beak_open=False)
    assert "♪" not in result


def test_logo_line_plain_note_only_when_beak_open_and_has_note():
    row_with_note = LogoCell("X", has_beak=True, has_note=True)
    row_without_note = LogoCell("X", has_beak=True, has_note=False)
    assert "♪" in logo_line_plain(row_with_note, beak_open=True)
    assert "♪" not in logo_line_plain(row_without_note, beak_open=True)


# --- render_logo ---


def test_render_logo_returns_rich_text():
    assert isinstance(render_logo(), Text)


def test_render_logo_default_has_beak_closed():
    text = render_logo(beak_open=False)
    plain = text.plain
    assert BEAK_CLOSED in plain


def test_render_logo_beak_open_contains_open_glyph():
    text = render_logo(beak_open=True)
    plain = text.plain
    assert BEAK_OPEN in plain


def test_render_logo_beak_open_with_note_contains_musical_note():
    text = render_logo(beak_open=True, show_note=True)
    assert "♪" in text.plain


def test_render_logo_show_note_false_suppresses_note():
    text = render_logo(beak_open=True, show_note=False)
    assert "♪" not in text.plain


def test_render_logo_note_absent_when_beak_closed():
    text = render_logo(beak_open=False, show_note=True)
    assert "♪" not in text.plain


def test_render_logo_row_count_matches_logo_rows():
    text = render_logo()
    line_count = text.plain.count("\n") + 1
    assert line_count == len(LOGO_ROWS)


def test_render_logo_applies_yellow_style_to_prefixes():
    text = render_logo()
    spans = [s for s in text._spans if s.style == LOGO_YELLOW]
    assert len(spans) > 0


def test_render_logo_applies_accent_style_to_beak():
    text = render_logo(beak_open=False)
    beak_style = f"bold {LOGO_ACCENT}"
    beak_spans = [s for s in text._spans if s.style == beak_style]
    assert len(beak_spans) == 1


# --- logo_plain_lines ---


def test_logo_plain_lines_returns_list_of_strings():
    lines = logo_plain_lines()
    assert isinstance(lines, list)
    assert all(isinstance(line, str) for line in lines)


def test_logo_plain_lines_length_matches_logo_rows():
    assert len(logo_plain_lines()) == len(LOGO_ROWS)


def test_logo_plain_lines_beak_closed_default():
    lines = logo_plain_lines()
    combined = "".join(lines)
    assert BEAK_CLOSED in combined


def test_logo_plain_lines_beak_open():
    lines = logo_plain_lines(beak_open=True)
    combined = "".join(lines)
    assert BEAK_OPEN in combined
    assert "♪" in combined
