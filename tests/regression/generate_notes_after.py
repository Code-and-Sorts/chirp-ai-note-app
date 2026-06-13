"""Regenerate the regression corpus's notes through the MLX pipeline (story 6.6).

For each slug under ``tests/regression/notes_quality/``, routes the committed
``transcript.txt`` through the production note-generation path —
``NoteGenerator._generate_structured_notes`` → ``_call_llm`` →
``LLMClient().chat_stream_sync(..., model="default")`` → the same
``template_engine`` rendering and front-matter helpers ``chirp transcribe``
uses — and writes the result to ``<slug>/notes_after.md``.

Idempotent (overwrites ``notes_after.md`` on re-run, e.g. after a model
escalation per AC-9). The only files this script writes are the
``<slug>/notes_after.md`` outputs under the corpus directory — it never writes
to ``~/Documents/chirp/`` (the real notes root). Like any chirp command it
loads the user's settings via ``get_settings()`` and constructs an
``LLMClient``, which resolves the daemon socket via env/config and may read
``~/.chirp/config.toml`` — so generation honours the same config the
production ``chirp transcribe`` path uses.

Usage:
    uv run python tests/regression/generate_notes_after.py [slug ...]

With no arguments, every slug in the corpus is regenerated; pass slugs to
re-roll specific pairs (document any re-roll in the results file).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from config.settings import get_settings
from notes.constants import DEFAULT_MEETING_NAME
from notes.note_generator import NoteGenerator

CORPUS_DIR = Path(__file__).parent / "notes_quality"
OUTPUT_FILENAME = "notes_after.md"


def corpus_slugs() -> list[str]:
    return sorted(
        path.name
        for path in CORPUS_DIR.iterdir()
        if path.is_dir() and (path / "transcript.txt").is_file()
    )


def generate_for_slug(generator: NoteGenerator, slug: str) -> None:
    transcript_text = (
        (CORPUS_DIR / slug / "transcript.txt").read_text(encoding="utf-8").strip()
    )

    structured_notes = generator._generate_structured_notes(
        transcript_text, provided_title=None
    )
    if not structured_notes:
        raise RuntimeError("LLM returned no parseable structured notes")

    # Mirror NoteGenerator._generate_for_record's assembly. note_date is the
    # day of regeneration: the corpus replay has no recording timestamp in
    # scope (story 6.6 AC-2 accepts this).
    note_date = datetime.now()
    meeting_notes = {
        "meeting_title": structured_notes.get("meeting_title", DEFAULT_MEETING_NAME),
        "executive_summary": structured_notes.get(
            "executive_summary", "No summary available"
        ),
        "agenda": structured_notes.get("agenda", []),
        "action_items": structured_notes.get("action_items", []),
        "next_steps": structured_notes.get("next_steps", []),
        "decisions": structured_notes.get("decisions", []),
        "open_questions": structured_notes.get("open_questions", []),
        "discussion_highlights": structured_notes.get("discussion_highlights", []),
        "metadata": {"date": note_date.isoformat()},
    }

    body = generator.template_engine.render_meeting_section(meeting_notes)
    content = generator._format_generated_note(body, note_date)
    (CORPUS_DIR / slug / OUTPUT_FILENAME).write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "slugs",
        nargs="*",
        help="corpus slugs to regenerate (default: every slug)",
    )
    args = parser.parse_args()

    known = corpus_slugs()
    unknown = sorted(set(args.slugs) - set(known))
    if unknown:
        parser.error(f"unknown slugs: {unknown}; corpus has: {known}")
    targets = args.slugs or known

    generator = NoteGenerator(get_settings())
    failures = 0
    for slug in targets:
        try:
            generate_for_slug(generator, slug)
        except Exception as exc:  # noqa: BLE001 - report per-slug, keep batch going
            failures += 1
            print(f"✗ {slug}: {exc}")
        else:
            print(f"✓ {slug}")

    print(f"\n{len(targets) - failures}/{len(targets)} regenerated")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
