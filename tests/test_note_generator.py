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
    models.context_window = 32768
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

    def test_parse_recovers_from_unescaped_special_chars(self, mock_settings):
        """Unescaped `&`/`<` break ET, but the known tags are still recovered."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)

            xml_response = (
                "<MEETING_NOTES>"
                "<MEETING_TITLE>Q&A / R&D sync</MEETING_TITLE>"
                "<EXECUTIVE_SUMMARY>Revenue < target; P&L reviewed</EXECUTIVE_SUMMARY>"
                "<AGENDA><ITEM>Discuss P&L</ITEM></AGENDA>"
                '<ACTION_ITEMS><ITEM task="Fix Q&A doc" owner="Jo" deadline="Fri"/>'
                "</ACTION_ITEMS>"
                "</MEETING_NOTES>"
            )
            import xml.etree.ElementTree as ET

            with pytest.raises(ET.ParseError):
                ET.fromstring(xml_response)

            result = generator._parse_xml_response(xml_response)

        assert result is not None
        assert result["meeting_title"] == "Q&A / R&D sync"
        assert result["executive_summary"] == "Revenue < target; P&L reviewed"
        assert result["agenda"] == ["Discuss P&L"]
        assert result["action_items"] == ["Fix Q&A doc — Owner: Jo — Deadline: Fri"]

    def test_parse_recovers_truncated_output(self, mock_settings):
        """Output cut off at the token cap still yields the completed tags."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            truncated = (
                "<MEETING_NOTES><MEETING_TITLE>Planning</MEETING_TITLE>"
                "<EXECUTIVE_SUMMARY>We discussed the road"
            )

            result = generator._parse_xml_response(truncated)

        assert result is not None
        assert result["meeting_title"] == "Planning"

    def test_parse_returns_none_when_nothing_recoverable(self, mock_settings):
        """A marker with no usable tags still degrades — no empty shell note."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)

            result = generator._parse_xml_response("<MEETING_NOTES>&&& < > junk")

        assert result is None

    def test_generate_for_record_writes_notes_and_updates_meta(
        self, mock_settings, tmp_path
    ):
        with (
            patch("notes.note_generator.TemplateEngine") as mock_template_engine,
            patch("notes.note_generator.PopupManager"),
            patch(
                "notes.note_generator.resolved_chat_model",
                lambda fallback=None, *a, **k: fallback,
            ),
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
        assert result["skipped"] is True
        assert "Insufficient" in result["error"]

    def test_generate_for_records_marks_all_short_as_skipped(
        self, mock_settings, tmp_path
    ):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            record = _seed_record(tmp_path, title="Quick", transcript="hi")

            result = generator.generate_for_records([record])

        assert result["success"] is False
        assert result["skipped"] is True
        assert "Insufficient" in result["error"]

    def test_generate_for_records_surfaces_real_llm_error(
        self, mock_settings, tmp_path, fake_llm_client, raise_llm_error
    ):
        from llm.exceptions import LLMModelError

        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            client = fake_llm_client(
                chat_stream_sync=raise_llm_error(LLMModelError, "model fell over")
            )
            generator = NoteGenerator(mock_settings, llm_client=client)
            record = _seed_record(
                tmp_path,
                title="Sync",
                transcript=(
                    "a transcript comfortably longer than the fifty character floor"
                ),
            )

            result = generator.generate_for_records([record], force=True)

        assert result["success"] is False
        assert "model fell over" in result["error"]

    def test_resolve_duration_reads_duration_s_from_meta(self, mock_settings, tmp_path):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            record = _seed_record(tmp_path, title="X", transcript="hi")
            meta_path = record.dir / "meta.toml"
            with meta_path.open("rb") as fh:
                meta = tomllib.load(fh)
            meta["duration_s"] = 91.99
            with meta_path.open("wb") as fh:
                tomli_w.dump(meta, fh)

            assert generator._resolve_duration_seconds(record) == pytest.approx(91.99)

    def test_resolve_duration_zero_when_no_meta_or_audio(self, mock_settings, tmp_path):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            record = _seed_record(tmp_path, title="X", transcript="hi")  # audio=None
            assert generator._resolve_duration_seconds(record) == 0.0

    def _record_with_audio(self, tmp_path, audio) -> NoteRecord:
        note_dir = audio.parent
        (note_dir / "meta.toml").write_text('title = "W"\n', encoding="utf-8")
        return NoteRecord(
            slug=note_dir.name,
            dir=note_dir,
            audio=audio,
            transcript=None,
            notes=None,
            meta=note_dir / "meta.toml",
            created_at=datetime(2026, 4, 20),
        )

    def test_resolve_duration_falls_back_to_wav_header(self, mock_settings, tmp_path):
        import wave

        note_dir = tmp_path / "wav-note"
        note_dir.mkdir()
        audio = note_dir / "audio.wav"
        with wave.open(str(audio), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16000)
            fh.writeframes(b"\x00\x00" * 8000)  # 8000 frames @ 16 kHz = 0.5s
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            record = self._record_with_audio(tmp_path, audio)
            assert generator._resolve_duration_seconds(record) == pytest.approx(0.5)

    def test_resolve_duration_zero_on_unreadable_audio(self, mock_settings, tmp_path):
        note_dir = tmp_path / "bad-audio"
        note_dir.mkdir()
        audio = note_dir / "audio.wav"
        audio.write_bytes(b"not a wav file")  # triggers wave.Error → logged, falls to 0
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            record = self._record_with_audio(tmp_path, audio)
            assert generator._resolve_duration_seconds(record) == 0.0

    def test_parse_failure_retries_once(self, mock_settings):
        """Unparsable output is retried once before giving up (AC-12)."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            with patch.object(
                generator, "_call_llm", return_value="not xml at all"
            ) as mock_llm:
                result = generator._generate_structured_notes("a transcript")

        assert result is None
        assert mock_llm.call_count == 2

    def test_parse_failure_does_not_write_or_index_junk_note(
        self, mock_settings, tmp_path
    ):
        """A persistent parse failure writes no note and triggers no auto-index."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            record = _seed_record(
                tmp_path,
                title="Broken",
                transcript=(
                    "This transcript is long enough to clear the 50-char minimum "
                    "but the model keeps returning unparsable garbage."
                ),
            )

            with (
                patch.object(generator, "_call_llm", return_value="garbage, no xml"),
                patch.object(generator, "_auto_index_note") as mock_index,
            ):
                result = generator._generate_for_record(record, force=False)

        assert result["success"] is False
        assert result.get("degraded") is True
        assert not (record.dir / "notes.md").exists()
        mock_index.assert_not_called()

    def test_transcript_char_budget_scales_with_context_window(self, mock_settings):
        """The transcript budget tracks context_window (not a fixed constant)."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)

            mock_settings.models.context_window = 32768
            big = generator._transcript_char_budget("")
            mock_settings.models.context_window = 8192
            small = generator._transcript_char_budget("")

        assert big > small
        assert big > 24000

    def test_transcript_char_budget_floors_when_no_input_room(self, mock_settings):
        """num_predict >= context leaves no input room — fall back to the floor,
        not an inflated budget that would overflow the window."""
        from notes.note_generator import _MIN_TRANSCRIPT_BUDGET_CHARS

        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            mock_settings.models.context_window = 4096
            mock_settings.models.num_predict = 4096

            assert generator._transcript_char_budget("") == _MIN_TRANSCRIPT_BUDGET_CHARS

    def test_whole_transcript_used_single_shot_when_it_fits(self, mock_settings):
        """A transcript within budget is sent verbatim — no truncation, no warning."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            console = Mock()
            generator = NoteGenerator(mock_settings, console=console)
            transcript = "We discussed the quarterly roadmap. " * 500

            with (
                patch.object(generator, "_call_llm", return_value="") as mock_llm,
                patch.object(
                    generator,
                    "_parse_xml_response",
                    return_value={"meeting_title": "T"},
                ),
            ):
                generator._generate_structured_notes(transcript)

            sent_prompt = mock_llm.call_args.args[0]

        assert transcript in sent_prompt
        assert not any(
            "summarizing the first" in str(call.args[0]).lower()
            for call in console.print.call_args_list
        )

    def test_long_transcript_is_windowed_with_warning(self, mock_settings):
        """A transcript past the (dynamic) budget is windowed, not silently cut."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            console = Mock()
            generator = NoteGenerator(mock_settings, console=console)
            budget = generator._transcript_char_budget("")
            long_transcript = "word " * ((budget // 5) + 1000)

            with (
                patch.object(generator, "_call_llm", return_value=None) as mock_llm,
                patch.object(
                    generator,
                    "_parse_xml_response",
                    return_value={"meeting_title": "T"},
                ),
            ):
                generator._generate_structured_notes(long_transcript)

            sent_prompt = mock_llm.call_args.args[0]

        assert len(long_transcript) > budget
        assert len(sent_prompt) < len(long_transcript)
        warned = any(
            "summarizing the first" in str(call.args[0]).lower()
            for call in console.print.call_args_list
        )
        assert warned
