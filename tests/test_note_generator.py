from unittest.mock import Mock, patch

import pytest

from notes.note_generator import NoteGenerator


def _make_streaming_response(text: str):
    import json

    lines = []
    for char in text:
        lines.append(json.dumps({"response": char, "done": False}).encode())
    lines.append(json.dumps({"response": "", "done": True}).encode())

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines = Mock(return_value=iter(lines))
    return mock_response


class TestNoteGenerator:
    @pytest.fixture
    def mock_settings(self):
        settings = Mock()
        models = Mock()
        models.ollama_url = "http://localhost:11434"
        models.llm = "llama3.1:8b"
        models.num_predict = 500
        settings.models = models
        return settings

    def test_initialization(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)
                        assert generator.settings == mock_settings

    def test_parse_xml_response_valid(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
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
                        assert (
                            "Discussed project timeline" in result["executive_summary"]
                        )
                        assert len(result["agenda"]) == 2
                        assert len(result["action_items"]) == 2
                        assert "Review budget proposal" in result["action_items"][0]
                        assert len(result["decisions"]) == 1
                        assert len(result["open_questions"]) == 1
                        assert len(result["discussion_highlights"]) == 1

    def test_parse_xml_response_with_none_sections(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
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

    def test_parse_fallback_response(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        response = "Test response"
                        result = generator._parse_fallback_response(response)

                        assert isinstance(result, dict)
                        assert "meeting_title" in result
                        assert "executive_summary" in result
                        assert "agenda" in result
                        assert "action_items" in result
                        assert "next_steps" in result
                        assert "decisions" in result
                        assert "open_questions" in result
                        assert "discussion_highlights" in result

    def test_call_ollama_success(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        with patch("requests.post") as mock_post:
                            mock_post.return_value = _make_streaming_response(
                                "Test response"
                            )

                            generator = NoteGenerator(mock_settings)
                            result = generator._call_ollama("test prompt")

                            assert result == "Test response"
                            mock_post.assert_called_once()

    def test_call_ollama_connection_error(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        with patch(
                            "requests.post", side_effect=Exception("Connection error")
                        ):
                            generator = NoteGenerator(mock_settings)

                            with pytest.raises(Exception, match="Connection error"):
                                generator._call_ollama("test prompt")

    def test_call_ollama_streams_response(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        xml = "<MEETING_NOTES><MEETING_TITLE>Test</MEETING_TITLE></MEETING_NOTES>"
                        mock_resp = _make_streaming_response(xml)

                        with patch(
                            "notes.note_generator.requests.post",
                            return_value=mock_resp,
                        ):
                            result = generator._call_ollama("test prompt")

                        assert "MEETING_NOTES" in result
                        assert "Test" in result
                        mock_resp.iter_lines.assert_called_once()

    def test_generate_meeting_notes_uses_provided_title(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        with patch.object(
                            generator, "_generate_structured_notes"
                        ) as mock_structured:
                            mock_structured.return_value = {
                                "meeting_title": "Generated Title",
                                "executive_summary": "Test summary",
                                "agenda": ["Agenda item 1"],
                                "action_items": ["Action 1"],
                                "next_steps": ["Next step 1"],
                                "decisions": ["Decision 1"],
                                "open_questions": ["Question 1"],
                                "discussion_highlights": ["Highlight 1"],
                            }

                            transcription_data = {
                                "full_text": "This is a test transcript with sufficient length to pass validation",
                                "metadata": {
                                    "title": "Provided Meeting Title",
                                    "duration": 300,
                                },
                            }

                            result = generator._generate_meeting_notes(
                                transcription_data
                            )

                            assert result is not None
                            assert result["meeting_title"] == "Provided Meeting Title"

    def test_generate_meeting_notes_generates_title_when_none_provided(
        self, mock_settings
    ):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        with patch.object(
                            generator, "_generate_structured_notes"
                        ) as mock_structured:
                            mock_structured.return_value = {
                                "meeting_title": "Generated Meeting Title",
                                "executive_summary": "Test summary",
                                "agenda": ["Agenda item 1"],
                                "action_items": ["Action 1"],
                                "next_steps": ["Next step 1"],
                                "decisions": ["Decision 1"],
                                "open_questions": ["Question 1"],
                                "discussion_highlights": ["Highlight 1"],
                            }

                            transcription_data = {
                                "full_text": "This is a test transcript with sufficient length to pass validation",
                                "metadata": {"duration": 300},
                            }

                            result = generator._generate_meeting_notes(
                                transcription_data
                            )

                            assert result is not None
                            assert result["meeting_title"] == "Generated Meeting Title"

    def test_generate_meeting_notes_generates_title_when_metadata_missing(
        self, mock_settings
    ):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        with patch.object(
                            generator, "_generate_structured_notes"
                        ) as mock_structured:
                            mock_structured.return_value = {
                                "meeting_title": "Generated Meeting Title",
                                "executive_summary": "Test summary",
                                "agenda": ["Agenda item 1"],
                                "action_items": ["Action 1"],
                                "next_steps": ["Next step 1"],
                                "decisions": ["Decision 1"],
                                "open_questions": ["Question 1"],
                                "discussion_highlights": ["Highlight 1"],
                            }

                            transcription_data = {
                                "full_text": "This is a test transcript with sufficient length to pass validation"
                            }

                            result = generator._generate_meeting_notes(
                                transcription_data
                            )

                            assert result is not None
                            assert result["meeting_title"] == "Generated Meeting Title"

    def test_generate_daily_notes_with_filename_override(self, mock_settings):
        from datetime import datetime
        from pathlib import Path

        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator") as mock_aggregator:
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        test_date = datetime(2024, 1, 15)
                        mock_transcription_file = Path("/test/transcription.json")
                        mock_aggregator_instance = mock_aggregator.return_value
                        mock_aggregator_instance.group_transcriptions_by_day.return_value = {
                            test_date: [mock_transcription_file]
                        }

                        with patch.object(
                            generator, "_generate_notes_for_day"
                        ) as mock_generate_day:
                            mock_generate_day.return_value = {
                                "success": True,
                                "filename": "custom-notes.md",
                                "path": "/notes/custom-notes.md",
                                "date": test_date.isoformat(),
                            }

                            result = generator.generate_daily_notes(
                                [mock_transcription_file],
                                force=False,
                                filename_override="custom-notes",
                            )

                            mock_generate_day.assert_called_once_with(
                                test_date,
                                [mock_transcription_file],
                                False,
                                "custom-notes",
                            )

                            assert result["success"] is True
                            assert result["filename"] == "custom-notes.md"

    def test_generate_notes_for_day_filename_override_with_extension(
        self, mock_settings
    ):
        from datetime import datetime
        from pathlib import Path

        mock_settings.directories = Mock()
        mock_settings.directories.notes = Path("/test/notes")

        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        test_date = datetime(2024, 1, 15)
                        mock_transcription_file = Path("/test/transcription.json")

                        with patch("pathlib.Path.exists", return_value=False):
                            with patch.object(
                                generator, "_generate_meeting_notes", return_value=None
                            ):
                                result = generator._generate_notes_for_day(
                                    test_date,
                                    [mock_transcription_file],
                                    False,
                                    "custom-notes.md",
                                )

                                assert result["success"] is False

    def test_generate_notes_for_day_filename_override_without_extension(
        self, mock_settings
    ):
        from datetime import datetime
        from pathlib import Path

        mock_settings.directories = Mock()
        mock_settings.directories.notes = Path("/test/notes")

        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        test_date = datetime(2024, 1, 15)
                        mock_transcription_file = Path("/test/transcription.json")

                        with patch("pathlib.Path.exists", return_value=False):
                            with patch.object(
                                generator, "_generate_meeting_notes", return_value=None
                            ):
                                result = generator._generate_notes_for_day(
                                    test_date,
                                    [mock_transcription_file],
                                    False,
                                    "custom-notes",
                                )

                                assert result["success"] is False

    def test_generated_note_includes_front_matter(self, mock_settings, tmp_path):
        from datetime import datetime
        from pathlib import Path

        mock_settings.directories = Mock()
        mock_settings.directories.notes = tmp_path

        with patch("notes.note_generator.TemplateEngine") as mock_template_engine:
            template_instance = mock_template_engine.return_value
            template_instance.render_daily_notes.return_value = "# Daily Summary\n"
            template_instance.render_meeting_section.return_value = "Section"

            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor") as mock_compressor:
                    mock_compressor.return_value.decompress_json.return_value = {
                        "success": True
                    }

                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        with patch.object(
                            generator,
                            "_generate_meeting_notes",
                            return_value={"dummy": "data"},
                        ):
                            transcription_file = tmp_path / "example.json"
                            transcription_file.write_text("{}", encoding="utf-8")

                            result = generator._generate_notes_for_day(
                                datetime(2024, 1, 1), [transcription_file], force=True
                            )

        assert result["success"] is True
        note_path = Path(result["path"])
        content = note_path.read_text(encoding="utf-8")

        assert content.startswith("---\n")
        assert "chirp_source: generated" in content.splitlines()
        assert "readonly: true" in content.splitlines()
        assert "# Daily Summary" in content
