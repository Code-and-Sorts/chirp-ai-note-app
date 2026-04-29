"""Shared ASCII chick logo and palette for the Chirp CLI.

The detailed chick comes from the design handoff (Logo C — "full lockup").
Only the beak glyph animates — < (closed) ↔ v (open) — so consumers render
the body once and update just the beak cell on each frame.
"""

from dataclasses import dataclass

from rich.text import Text

LOGO_YELLOW = "#dcb84a"
LOGO_ACCENT = "#d97a3a"
LOGO_NOTE = "#e08072"

TAGLINE = "AI notes for the terminal"
CREDIT = "made with ♥ by Colby Timm"
REPO = "github.com/colbytimm/chirp-ai-note-app"

BEAK_CLOSED = "<"
BEAK_OPEN = "v"


@dataclass(frozen=True)
class LogoCell:
    """A logo row with optional beak placement.

    ``prefix`` is the left side of the body (always yellow). ``suffix`` is
    the right side of the body (yellow). If ``has_beak`` is True, the beak
    glyph is inserted between prefix and suffix; callers swap it each frame.
    """

    prefix: str
    suffix: str = ""
    has_beak: bool = False
    has_note: bool = False


LOGO_ROWS: tuple[LogoCell, ...] = (
    LogoCell("           .---."),
    LogoCell("    .---. '     \\ _"),
    LogoCell("  /`     `    o  |", "", has_beak=True, has_note=True),
    LogoCell(",_.' _.---.       / `"),
    LogoCell("`\\  `\\         ;.-'"),
    LogoCell("  \\   '._.'    /"),
    LogoCell("   '.        .'"),
    LogoCell("     \\_/--;`"),
    LogoCell("      |   _\\_"),
    LogoCell("    --;--.  \\`--"),
    LogoCell("       '.    `"),
)


def logo_line_plain(row: LogoCell, beak_open: bool) -> str:
    """Return the row as a plain string — useful for tests and width math."""
    beak = ""
    if row.has_beak:
        beak = BEAK_OPEN if beak_open else BEAK_CLOSED
    note = "  ♪" if row.has_note and beak_open else ""
    return f"{row.prefix}{beak}{row.suffix}{note}"


def render_logo(beak_open: bool = False, show_note: bool = True) -> Text:
    """Render the full detailed chick as Rich Text.

    ``beak_open`` toggles the beak glyph. ``show_note`` hides the ♪
    accent for places that want a quieter variant.
    """
    text = Text()
    for idx, row in enumerate(LOGO_ROWS):
        text.append(row.prefix, style=LOGO_YELLOW)
        if row.has_beak:
            beak = BEAK_OPEN if beak_open else BEAK_CLOSED
            text.append(beak, style=f"bold {LOGO_ACCENT}")
        if row.suffix:
            text.append(row.suffix, style=LOGO_YELLOW)
        if show_note and row.has_note and beak_open:
            text.append("  ♪", style=LOGO_ACCENT)
        if idx < len(LOGO_ROWS) - 1:
            text.append("\n")
    return text


def logo_plain_lines(beak_open: bool = False) -> list[str]:
    """Plain-text rendering — used by the about animation's line-by-line paint."""
    return [logo_line_plain(row, beak_open) for row in LOGO_ROWS]


COMPACT_LOGO = "chirp ~>"
