import tomllib
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import tomli_w

from notes.note_generator import NoteGenerator
from notes.note_templates import TemplateLoader, build_system_prompt
from utils.file_utils import NoteRecord

MEETING_TEMPLATE = TemplateLoader(
    user_dir=Path("/nonexistent-chirp-templates")
).load_default()
MEETING_PROMPT = build_system_prompt(MEETING_TEMPLATE)


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
<NOTES>
    <TITLE>Project Alpha Sync</TITLE>
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
</NOTES>"""

            result = generator._parse_xml_response(xml_response)

            assert result is not None
            assert result["title"] == "Project Alpha Sync"
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
<NOTES>
    <TITLE>Brief Check-in</TITLE>
    <EXECUTIVE_SUMMARY>Short informal chat.</EXECUTIVE_SUMMARY>
    <AGENDA>None</AGENDA>
    <ACTION_ITEMS>None</ACTION_ITEMS>
    <NEXT_STEPS>None</NEXT_STEPS>
    <DECISIONS>None</DECISIONS>
    <OPEN_QUESTIONS>None</OPEN_QUESTIONS>
    <DISCUSSION_HIGHLIGHTS>
        <ITEM>Casual discussion</ITEM>
    </DISCUSSION_HIGHLIGHTS>
</NOTES>"""

            result = generator._parse_xml_response(xml_response)

            assert result is not None
            assert result["title"] == "Brief Check-in"
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
                "<NOTES>"
                "<TITLE>Q&A / R&D sync</TITLE>"
                "<EXECUTIVE_SUMMARY>Revenue < target; P&L reviewed</EXECUTIVE_SUMMARY>"
                "<AGENDA><ITEM>Discuss P&L</ITEM></AGENDA>"
                '<ACTION_ITEMS><ITEM task="Fix Q&A doc" owner="Jo" deadline="Fri"/>'
                "</ACTION_ITEMS>"
                "</NOTES>"
            )
            import xml.etree.ElementTree as ET

            with pytest.raises(ET.ParseError):
                ET.fromstring(xml_response)

            result = generator._parse_xml_response(xml_response)

        assert result is not None
        assert result["title"] == "Q&A / R&D sync"
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
                "<NOTES><TITLE>Planning</TITLE>"
                "<EXECUTIVE_SUMMARY>We discussed the road"
            )

            result = generator._parse_xml_response(truncated)

        assert result is not None
        assert result["title"] == "Planning"

    def test_parse_returns_none_when_nothing_recoverable(self, mock_settings):
        """A marker with no usable tags still degrades — no empty shell note."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)

            result = generator._parse_xml_response("<NOTES>&&& < > junk")

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
            template_instance.render_note.return_value = (
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
                    "title": "Project Alpha",
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
            big = generator._transcript_char_budget("", MEETING_PROMPT)
            mock_settings.models.context_window = 8192
            small = generator._transcript_char_budget("", MEETING_PROMPT)

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

            assert generator._transcript_char_budget("", MEETING_PROMPT) == _MIN_TRANSCRIPT_BUDGET_CHARS

    def test_whole_transcript_used_single_shot_when_it_fits(self, mock_settings):
        """A transcript within budget is sent verbatim through one prompt."""
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
                    return_value={"title": "T"},
                ),
            ):
                generator._generate_structured_notes(transcript)

        assert mock_llm.call_count == 1
        assert transcript in mock_llm.call_args.args[0]
        assert not any(
            "summarizing" in str(call.args[0]).lower()
            for call in console.print.call_args_list
        )

    def test_long_transcript_is_chunked_and_consolidated(self, mock_settings):
        """A transcript past the budget is chunked (map) then consolidated (reduce)."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            console = Mock()
            generator = NoteGenerator(mock_settings, console=console)
            budget = generator._transcript_char_budget("", MEETING_PROMPT)
            long_transcript = " ".join(
                f"Sentence {i} discusses a distinct topic in some detail."
                for i in range((budget // 40) + 500)
            )

            with (
                patch.object(generator, "_call_llm", return_value="") as mock_llm,
                patch.object(
                    generator,
                    "_parse_xml_response",
                    return_value={
                        "title": "T",
                        "executive_summary": "part",
                        "action_items": ["A"],
                    },
                ),
            ):
                result = generator._generate_structured_notes(long_transcript)

        assert len(long_transcript) > budget
        assert result is not None
        assert result["action_items"] == ["A"]
        assert result["executive_summary"] != "No summary available"
        assert mock_llm.call_count >= 2
        assert any(
            "summarizing in" in str(call.args[0]).lower()
            for call in console.print.call_args_list
        )

    def test_chunk_bounds_size_for_multiline_and_single_line(self, mock_settings):
        """Every chunk stays within budget — including a real single-line transcript
        (whisper joins segments with spaces, no newlines)."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)

        multiline = "\n".join(
            f"[00:{i:02d}] S{i % 3}: Point number {i} was discussed at length."
            for i in range(400)
        )
        ml_chunks = generator._chunk_transcript(multiline, 8000)
        assert len(ml_chunks) > 1
        assert all(len(c) <= 8000 for c in ml_chunks)

        single_line = " ".join(
            f"Sentence {i} covers a distinct topic in detail." for i in range(2000)
        )
        assert "\n" not in single_line
        sl_chunks = generator._chunk_transcript(single_line, 8000)
        assert len(sl_chunks) > 1
        assert all(len(c) <= 8000 for c in sl_chunks)

    def test_chunk_transcript_overlaps_consecutive_chunks(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        single_line = " ".join(
            f"Sentence {i} covers a distinct topic in detail." for i in range(2000)
        )
        chunks = generator._chunk_transcript(single_line, 8000)

        assert len(chunks) >= 2
        assert set(chunks[0].split()[-5:]) & set(chunks[1].split()[:10])

    def test_reduce_preserves_every_item_across_chunks(self, mock_settings):
        """No structured item is dropped when consolidating chunk notes."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        notes = [
            {"executive_summary": "s1", "action_items": ["A1"], "decisions": ["D1"]},
            {"executive_summary": "s2", "action_items": ["A2"], "decisions": ["D2"]},
            {
                "executive_summary": "s3",
                "action_items": ["A3"],
                "open_questions": ["Q3"],
            },
        ]
        with patch.object(generator, "_call_llm", return_value="whole summary"):
            result = generator._reduce_chunk_notes(notes, 95000, MEETING_TEMPLATE)

        assert result["action_items"] == ["A1", "A2", "A3"]
        assert result["decisions"] == ["D1", "D2"]
        assert result["open_questions"] == ["Q3"]
        assert result["executive_summary"] == "whole summary"

    def test_reduce_summary_synthesizes_from_chunk_summaries(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        notes = [{"executive_summary": "a"}, {"executive_summary": "b"}]
        with patch.object(generator, "_call_llm", return_value="synthesized") as call:
            result = generator._reduce_chunk_notes(notes, 95000, MEETING_TEMPLATE)

        call.assert_called_once()
        assert result["executive_summary"] == "synthesized"

    def test_reduce_summary_falls_back_to_joined_when_llm_empty(self, mock_settings):
        """An empty model reply keeps the joined chunk summaries, never 'No summary
        available'."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        notes = [
            {"executive_summary": "first half"},
            {"executive_summary": "second half"},
        ]
        with patch.object(generator, "_call_llm", return_value="   "):
            result = generator._reduce_chunk_notes(notes, 95000, MEETING_TEMPLATE)

        assert result["executive_summary"] == "first half second half"

    def test_reduce_summary_falls_back_to_content_when_no_chunk_summaries(
        self, mock_settings
    ):
        """With no chunk summaries, synthesize from decisions/highlights/actions."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        notes = [
            {"decisions": ["Adopt plan X"]},
            {"action_items": ["Ship the release — Owner: Sam"]},
        ]
        with patch.object(generator, "_call_llm", return_value="synth") as call:
            result = generator._reduce_chunk_notes(notes, 95000, MEETING_TEMPLATE)

        call.assert_called_once()
        assert result["executive_summary"] == "synth"

    def test_reduce_summary_never_empty_when_content_exists(self, mock_settings):
        """No chunk summaries + empty LLM reply still yields a summary from the
        extracted content — never 'No summary available'."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        notes = [
            {"decisions": ["Adopt plan X"]},
            {"action_items": ["Ship it — Owner: Sam"]},
        ]
        with patch.object(generator, "_call_llm", return_value="  "):
            result = generator._reduce_chunk_notes(notes, 95000, MEETING_TEMPLATE)

        assert result["executive_summary"] != "No summary available"
        assert result["executive_summary"].strip()

    def test_merge_notes_unions_and_dedups(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        merged = generator._merge_notes(
            [
                {
                    "title": "M",
                    "executive_summary": "a",
                    "action_items": ["Task — Owner: Jo", "shared"],
                },
                {
                    "title": "M",
                    "executive_summary": "b",
                    "action_items": ["  task — owner: jo  ", "shared"],
                    "decisions": ["D"],
                },
            ],
            MEETING_TEMPLATE,
        )
        assert merged["action_items"] == ["Task — Owner: Jo", "shared"]
        assert merged["decisions"] == ["D"]
        assert merged["executive_summary"] == "a b"
        assert merged["title"] == "M"

    def test_dedup_merges_reordered_and_subset_phrasings(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        result = generator._dedup(
            [
                "Migration of audio capture to sounddevice",
                "audio capture migration to sounddevice",
                "Examine hiring plans for two backend engineers",
                "hiring plans for two backend engineers",
                "Ship the release on Friday",
            ]
        )

        assert len(result) == 3
        assert "Ship the release on Friday" in result

    def test_dedup_keeps_items_distinct_by_number_or_identifier(self, mock_settings):
        """Fuzzy-similar but genuinely distinct items must never collapse."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        result = generator._dedup(
            [
                "Launch feature X on June 1",
                "Launch feature X on June 15",
                "hire 2 engineers",
                "hire 3 engineers",
                "Approve vendor A",
                "Approve vendor B",
            ]
        )

        assert len(result) == 6

    def test_dedup_merges_plural_variant(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        result = generator._dedup(["hire backend engineer", "hire backend engineers"])

        assert len(result) == 1

    def test_consolidate_skips_when_prompt_too_large(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        items = ["x" * 5000 + str(i) for i in range(10)]
        with patch.object(generator, "_call_llm") as call:
            result = generator._consolidate_items(items, "action items")

        call.assert_not_called()
        assert result == items

    def test_consolidate_skips_short_lists(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        with patch.object(generator, "_call_llm") as call:
            result = generator._consolidate_items(["one", "two"], "decisions")

        call.assert_not_called()
        assert result == ["one", "two"]

    def test_consolidate_never_drops_or_fabricates_items(self, mock_settings):
        """Coverage guard: model-omitted items stay as singletons; canonical is
        always a verbatim input."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        items = ["alpha first", "alpha the first", "beta", "gamma", "delta"]
        with patch.object(generator, "_call_llm", return_value="1=a\n2=a"):
            result = generator._consolidate_items(items, "action items")

        assert all(r in items for r in result)
        for distinct in ["beta", "gamma", "delta"]:
            assert distinct in result
        assert len(result) == 4


STANDUP_TEMPLATE = TemplateLoader(
    user_dir=Path("/nonexistent-chirp-templates")
).load("standup")

STANDUP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<NOTES>
    <TITLE>Daily Sync</TITLE>
    <YESTERDAY><ITEM>Shipped the exporter</ITEM></YESTERDAY>
    <TODAY><ITEM>Review open PRs</ITEM></TODAY>
    <BLOCKERS>None</BLOCKERS>
</NOTES>"""


class TestCustomTemplates:
    def test_parse_xml_custom_sections_strict(self, mock_settings):
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            result = generator._parse_xml_response(STANDUP_XML, STANDUP_TEMPLATE)

        assert result == {
            "title": "Daily Sync",
            "yesterday": ["Shipped the exporter"],
            "today": ["Review open PRs"],
            "blockers": [],
        }

    def test_parse_xml_custom_sections_lenient(self, mock_settings):
        """Unescaped `&` breaks ET, yet the template's tags are recovered."""
        malformed = (
            "<NOTES><TITLE>R&D Sync</TITLE>"
            "<YESTERDAY><ITEM>Debugged Q&A flow</ITEM></YESTERDAY>"
            "<TODAY>None</TODAY><BLOCKERS><ITEM>CI < flaky</ITEM></BLOCKERS>"
            "</NOTES>"
        )
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
            result = generator._parse_xml_response(malformed, STANDUP_TEMPLATE)

        assert result is not None
        assert result["title"] == "R&D Sync"
        assert result["yesterday"] == ["Debugged Q&A flow"]
        assert result["today"] == []
        assert result["blockers"] == ["CI < flaky"]

    def test_reduce_chunk_notes_with_custom_list_keys(self, mock_settings):
        """Chunked reduce merges the template's own list sections; a template
        with no prose section triggers no summary LLM call."""
        with (
            patch("notes.note_generator.TemplateEngine"),
            patch("notes.note_generator.PopupManager"),
        ):
            generator = NoteGenerator(mock_settings)
        notes = [
            {"title": "T", "yesterday": ["a"], "today": ["b"], "blockers": []},
            {"title": "T", "yesterday": ["c"], "today": ["b"], "blockers": ["x"]},
        ]
        with patch.object(generator, "_call_llm") as call:
            result = generator._reduce_chunk_notes(notes, 95000, STANDUP_TEMPLATE)

        call.assert_not_called()
        assert result["yesterday"] == ["a", "c"]
        assert result["today"] == ["b"]
        assert result["blockers"] == ["x"]

    def test_generate_for_record_renders_tag_matched_template(
        self, mock_settings, tmp_path
    ):
        loader = TemplateLoader(user_dir=tmp_path / "templates")
        loader.scaffold()
        standup_path = loader.user_dir / "standup.md"
        standup_path.write_text(
            "---\ntags: [standup, dsu]\n---\n"
            + standup_path.read_text(encoding="utf-8").split("---\n", 2)[2],
            encoding="utf-8",
        )

        with (
            patch("notes.note_generator.PopupManager"),
            patch(
                "notes.note_generator.resolved_chat_model",
                lambda fallback=None, *a, **k: fallback,
            ),
        ):
            generator = NoteGenerator(mock_settings, template_loader=loader)
            record = _seed_record(
                tmp_path,
                title="Daily Sync",
                transcript=(
                    "Yesterday the exporter shipped; today the team reviews open "
                    "pull requests and nothing is blocked."
                ),
            )
            record.tags = ["standup"]

            with patch.object(generator, "_call_llm", return_value=STANDUP_XML):
                result = generator._generate_for_record(record, force=True)

        assert result["success"] is True
        content = (record.dir / "notes.md").read_text(encoding="utf-8")
        assert "### Yesterday\n\n- Shipped the exporter" in content
        assert "### Blockers\n\nNone" in content
        assert "Executive Summary" not in content

    def test_generate_for_record_honors_template_override(
        self, mock_settings, tmp_path
    ):
        loader = TemplateLoader(user_dir=tmp_path / "templates")
        with (
            patch("notes.note_generator.PopupManager"),
            patch(
                "notes.note_generator.resolved_chat_model",
                lambda fallback=None, *a, **k: fallback,
            ),
        ):
            generator = NoteGenerator(mock_settings, template_loader=loader)
            record = _seed_record(
                tmp_path,
                title="Daily Sync",
                transcript=(
                    "Yesterday the exporter shipped; today the team reviews open "
                    "pull requests and nothing is blocked."
                ),
            )

            with patch.object(generator, "_call_llm", return_value=STANDUP_XML):
                result = generator._generate_for_record(
                    record, force=True, template_override="standup"
                )

        assert result["success"] is True
        content = (record.dir / "notes.md").read_text(encoding="utf-8")
        assert "### Today\n\n- Review open PRs" in content
