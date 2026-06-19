"""End-to-end happy path: record -> transcribe -> notes over the on-disk layout.

The record and transcribe stages need CoreAudio + MLX/Whisper, which are not
available off Apple Silicon, so they are represented here by the canonical
artifacts they write (``audio.wav`` and ``transcript.txt``). The test then
drives the *real* note-generation entry point (``generate_from_notes_root``)
with a faked ``LLMClient`` and asserts that a complete note folder
(``audio.wav``, ``transcript.txt``, ``notes.md``, ``meta.toml``) exists and
parses. This is the pipeline gap called out in the production-readiness review.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest import mock

import pytest
import tomli_w

from config.settings import ChirpSettings, DirectoriesConfig, NotesChatConfig
from notes.note_generator import NoteGenerator

pytestmark = pytest.mark.integration

# A valid structured-notes XML document, split into streamed tokens so the fake
# client exercises NoteGenerator._call_llm's token-join path.
_XML_TOKENS = [
    '<?xml version="1.0" encoding="UTF-8"?>\n',
    "<MEETING_NOTES>\n",
    "<MEETING_TITLE>Quarterly Planning</MEETING_TITLE>\n",
    "<EXECUTIVE_SUMMARY>The team aligned on Q3 priorities and owners."
    "</EXECUTIVE_SUMMARY>\n",
    "<AGENDA><ITEM>Review roadmap</ITEM><ITEM>Assign owners</ITEM></AGENDA>\n",
    '<ACTION_ITEMS><ITEM task="Draft spec" owner="Sam" deadline="2026-07-01"/>'
    "</ACTION_ITEMS>\n",
    "<NEXT_STEPS><ITEM>Circulate notes</ITEM></NEXT_STEPS>\n",
    "<DECISIONS><ITEM>Ship the alpha</ITEM></DECISIONS>\n",
    "<OPEN_QUESTIONS>None</OPEN_QUESTIONS>\n",
    "<DISCUSSION_HIGHLIGHTS><ITEM>Strong demo feedback</ITEM>"
    "</DISCUSSION_HIGHLIGHTS>\n",
    "</MEETING_NOTES>\n",
]


def _seed_recorded_and_transcribed_note(notes_root: Path) -> Path:
    """Write the artifacts the record + transcribe stages produce."""
    note_dir = notes_root / "quarterly-planning-2026-06-19"
    note_dir.mkdir(parents=True)

    # `record` stage: an audio.wav plus the meta.toml it seeds.
    (note_dir / "audio.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake-audio")
    with (note_dir / "meta.toml").open("wb") as fh:
        tomli_w.dump(
            {
                "title": "Quarterly Planning",
                "date": "2026-06-19T09:00:00",
                "tags": ["planning"],
            },
            fh,
        )

    # `transcribe` stage: the transcript.txt note generation consumes (>= 50 chars).
    (note_dir / "transcript.txt").write_text(
        "We reviewed the Q3 roadmap, assigned owners, and agreed to ship the "
        "alpha after one more round of demo feedback.",
        encoding="utf-8",
    )
    return note_dir


def test_record_transcribe_notes_produces_all_artifacts(
    tmp_path, fake_llm_client, fake_chat_tokens
):
    notes_root = tmp_path / "chirp-notes"
    note_dir = _seed_recorded_and_transcribed_note(notes_root)

    settings = ChirpSettings(
        directories=DirectoriesConfig(notes_root=notes_root),
        # Keep the assertion on artifacts; the embedding index needs Chroma and a
        # live embed model, which are out of scope for this happy-path check.
        notes_chat=NotesChatConfig(auto_index=False),
    )

    fake = fake_llm_client(chat_stream_sync=fake_chat_tokens(_XML_TOKENS))

    with mock.patch("notes.note_generator.PopupManager"):
        generator = NoteGenerator(settings, llm_client=fake)
        result = generator.generate_from_notes_root()

    assert result["success"], result

    # All four canonical artifacts exist after the pipeline.
    for name in ("audio.wav", "transcript.txt", "notes.md", "meta.toml"):
        assert (note_dir / name).exists(), f"missing {name}"

    notes_md = (note_dir / "notes.md").read_text(encoding="utf-8")
    assert notes_md.strip()
    assert "Quarterly Planning" in notes_md

    with (note_dir / "meta.toml").open("rb") as fh:
        meta = tomllib.load(fh)
    assert meta["whisper_model"] == settings.models.whisper
    # Review item 3: llm_model reflects a resolved name, not an empty value.
    assert meta["llm_model"]
    assert meta["title"] == "Quarterly Planning"

    # The fake was actually exercised by note generation.
    assert fake.chat_stream_sync.calls
