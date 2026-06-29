"""Smoke test for the chirp about animation.

The animation uses rich.live.Live plus time.sleep — we pass a no-op
sleeper and capture output to verify the three phases each emit the
expected content without blocking the test run.
"""

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from chirp import about
from chirp.branding import LOGO_ROWS, TAGLINE


def _fake_settings(tmp_path):
    notes_root = tmp_path / "chirp"
    notes_root.mkdir()
    for slug in ("first-2026-04-20", "second-2026-04-20"):
        note_dir = notes_root / slug
        note_dir.mkdir()
        (note_dir / "notes.md").write_text("hi")
    return SimpleNamespace(
        directories=SimpleNamespace(notes_root=notes_root),
        models=SimpleNamespace(llm="llama3.1:8b"),
        notes_chat=SimpleNamespace(
            semantic_enabled=False, recommended_embed_model="bge-small-en-v1.5"
        ),
    )


def _no_sleep(_seconds: float) -> None:
    return None


def test_run_about_paints_all_logo_rows(tmp_path):
    buffer = StringIO()
    console = Console(file=buffer, width=120, force_terminal=False)

    about.run_about(
        console,
        _fake_settings(tmp_path),
        sleeper=_no_sleep,
    )

    output = buffer.getvalue()
    assert "chirp about" in output
    for row in LOGO_ROWS:
        snippet = row.prefix.strip().split(" ")[0]
        if snippet:
            assert snippet in output, f"logo snippet {snippet!r} missing"
    assert "chirp" in output
    assert "Colby Timm" in output
    assert TAGLINE in output
