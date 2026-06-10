import tomllib
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import tomli_w

from notes.note_generator import NoteGenerator
from utils.file_utils import NoteRecord


@pytest.fixture
def mock_settings():
    settings = Mock()
    models = Mock()
    models.llm = "llama3.1:8b"
    models.whisper = "large-v3-turbo"
    models.num_predict = 4096
    settings.models = models
    settings.notes_chat = Mock()
    settings.notes_chat.auto_index = False
    directories = Mock()
    directories.notes_root = None
    settings.directories = directories
    return settings


def _seed_record(tmp_path, title: str, transcript: str) -> NoteRecord:
    note_dir = tmp_path / "sample-2026-04-20"
    note_dir.mkdir()
    transcript_path = note_dir / "transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    meta_path = note_dir / "meta.toml"
    with meta_path.open("wb") as fh:
        tomli_w.dump(
            {
                "title": title,
                "date": "2026-04-20T09:00:00",
                "tags": [],
            },
            fh,
        )
    return NoteRecord(
        slug=note_dir.name,
        dir=note_dir,
        audio=None,
        transcript=transcript_path,
        notes=None,
        meta=meta_path,
        created_at=datetime(2026, 4, 20, 9, 0, 0),
        tags=[],
        title=title,
    )


class TestNoteGenerator:
    def test_initialization(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            assert generator.settings is mock_settings

    def test_parse_xml_response_valid(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)

            xml_response = """<?xml version="1.0" encoding="UTF-8"?>
<MEETING_NOTES>
    <MEETING_TITLE>Project Alpha Sync</MEETING_TITLE>
    <EXECUTIVE_SUMMARY>Discussed project timeline and resource allocation.</EXECUTIVE_SUMMARY>
    <AGENDA>
        <ITEM>Review Q1 goals</ITEM>
        <ITEM>Discuss budget</ITEM>
    </AGENDA>
    <ACTION_ITEMS>
        <ITEM task="Review budget proposal" owner="John" deadline="2024-01-15"/>
        <ITEM task="Update timeline" owner="Sarah" deadline=""/>
    </ACTION_ITEMS>
    <NEXT_STEPS>
        <ITEM>Follow up meeting next week</ITEM>
    </NEXT_STEPS>
    <DECISIONS>
        <ITEM>Approved new feature X</ITEM>
    </DECISIONS>
    <OPEN_QUESTIONS>
        <ITEM>What is the final budget?</ITEM>
    </OPEN_QUESTIONS>
    <DISCUSSION_HIGHLIGHTS>
        <ITEM>Team discussed resource constraints</ITEM>
    </DISCUSSION_HIGHLIGHTS>
</MEETING_NOTES>"""

            result = generator._parse_xml_response(xml_response)

            assert result is not None
            assert result["meeting_title"] == "Project Alpha Sync"
            assert "Discussed project timeline" in result["executive_summary"]
            assert len(result["agenda"]) == 2
            assert len(result["action_items"]) == 2
            assert "Review budget proposal" in result["action_items"][0]
            assert len(result["decisions"]) == 1
            assert len(result["open_questions"]) == 1
            assert len(result["discussion_highlights"]) == 1

    def test_parse_xml_response_with_none_sections(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)

            xml_response = """<?xml version="1.0" encoding="UTF-8"?>
<MEETING_NOTES>
    <MEETING_TITLE>Brief Check-in</MEETING_TITLE>
    <EXECUTIVE_SUMMARY>Short informal chat.</EXECUTIVE_SUMMARY>
    <AGENDA>None</AGENDA>
    <ACTION_ITEMS>None</ACTION_ITEMS>
    <NEXT_STEPS>None</NEXT_STEPS>
    <DECISIONS>None</DECISIONS>
    <OPEN_QUESTIONS>None</OPEN_QUESTIONS>
    <DISCUSSION_HIGHLIGHTS>
        <ITEM>Casual discussion</ITEM>
    </DISCUSSION_HIGHLIGHTS>
</MEETING_NOTES>"""

            result = generator._parse_xml_response(xml_response)

            assert result is not None
            assert result["meeting_title"] == "Brief Check-in"
            assert result["agenda"] == []
            assert result["action_items"] == []
            assert result["next_steps"] == []
            assert result["decisions"] == []
            assert result["open_questions"] == []
            assert len(result["discussion_highlights"]) == 1

    def test_generate_for_record_writes_notes_and_updates_meta(
        self, mock_settings, tmp_path
    ):
        with (
            patch("notes.note_generator.TemplateEngine") as mock_template_engine,
            patch("notes.note_generator.PopupManager"),
        ):
            template_instance = mock_template_engine.return_value
            template_instance.render_meeting_section.return_value = (
                "## Project Alpha\n\nBody"
            )

            generator = NoteGenerator(mock_settings)
            record = _seed_record(
                tmp_path,
                title="Project Alpha",
                transcript=(
                    "This is a long-enough transcript to clear the minimum "
                    "length threshold used by the generator (over 50 chars)."
                ),
            )

            with patch.object(
                generator,
                "_generate_structured_notes",
                return_value={
                    "meeting_title": "Project Alpha",
                    "executive_summary": "Summary",
                    "agenda": [],
                    "action_items": [],
                    "next_steps": [],
                    "decisions": [],
                    "open_questions": [],
                    "discussion_highlights": [],
                },
            ):
                result = generator._generate_for_record(record, force=False)

        assert result["success"] is True
        notes_path = record.dir / "notes.md"
        assert notes_path.exists()

        content = notes_path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "chirp_source: generated" in content
        assert "Project Alpha" in content

        with (record.dir / "meta.toml").open("rb") as fh:
            meta = tomllib.load(fh)
        assert meta["whisper_model"] == "large-v3-turbo"
        assert meta["llm_model"] == "llama3.1:8b"
        assert "indexed_at" in meta

    def test_generate_for_record_skips_short_transcript(self, mock_settings, tmp_path):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            record = _seed_record(tmp_path, title="Quick", transcript="hi")

            result = generator._generate_for_record(record, force=False)

        assert result["success"] is False
        assert "Insufficient" in result["error"]
