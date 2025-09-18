from unittest.mock import Mock, patch

import pytest

from notes.note_generator import NoteGenerator


class TestNoteGenerator:
    @pytest.fixture
    def mock_settings(self):
        settings = Mock()
        models = Mock()
        models.ollama_url = "http://localhost:11434"
        models.llm = "llama3.1:8b"
        settings.models = models
        return settings

    def test_initialization(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)
                        assert generator.settings == mock_settings

    def test_parse_fallback_response(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        generator = NoteGenerator(mock_settings)

                        response = "Test response"
                        result = generator._parse_fallback_response(response)

                        assert isinstance(result, dict)

    def test_call_ollama_success(self, mock_settings):
        with patch("notes.note_generator.TemplateEngine"):
            with patch("notes.note_generator.DailyAggregator"):
                with patch("notes.note_generator.JSONCompressor"):
                    with patch("notes.note_generator.PopupManager"):
                        with patch("requests.post") as mock_post:
                            mock_response = Mock()
                            mock_response.json.return_value = {
                                "response": "Test response"
                            }
                            mock_post.return_value = mock_response

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
                                "participants": "Test participant",
                                "executive_summary": "Test summary",
                                "key_points": ["Key point 1"],
                                "decisions": ["Decision 1"],
                                "action_items": ["Action 1"],
                                "next_steps": ["Next step 1"],
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
                            with patch.object(
                                generator, "_generate_meeting_title"
                            ) as mock_title:
                                mock_structured.return_value = {
                                    "participants": "Test participant",
                                    "executive_summary": "Test summary",
                                    "key_points": ["Key point 1"],
                                    "decisions": ["Decision 1"],
                                    "action_items": ["Action 1"],
                                    "next_steps": ["Next step 1"],
                                }
                                mock_title.return_value = "Generated Meeting Title"

                                transcription_data = {
                                    "full_text": "This is a test transcript with sufficient length to pass validation",
                                    "metadata": {"duration": 300},
                                }

                                result = generator._generate_meeting_notes(
                                    transcription_data
                                )

                                assert result is not None
                                assert (
                                    result["meeting_title"] == "Generated Meeting Title"
                                )
                                mock_title.assert_called_once()

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
                            with patch.object(
                                generator, "_generate_meeting_title"
                            ) as mock_title:
                                mock_structured.return_value = {
                                    "participants": "Test participant",
                                    "executive_summary": "Test summary",
                                    "key_points": ["Key point 1"],
                                    "decisions": ["Decision 1"],
                                    "action_items": ["Action 1"],
                                    "next_steps": ["Next step 1"],
                                }
                                mock_title.return_value = "Generated Meeting Title"

                                transcription_data = {
                                    "full_text": "This is a test transcript with sufficient length to pass validation"
                                }

                                result = generator._generate_meeting_notes(
                                    transcription_data
                                )

                                assert result is not None
                                assert (
                                    result["meeting_title"] == "Generated Meeting Title"
                                )
                                mock_title.assert_called_once()
