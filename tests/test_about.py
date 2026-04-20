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
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "a.md").write_text("hi")
    (notes_dir / "b.md").write_text("hi")
    return SimpleNamespace(
        directories=SimpleNamespace(notes=notes_dir),
        models=SimpleNamespace(llm="llama3.1:8b"),
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
