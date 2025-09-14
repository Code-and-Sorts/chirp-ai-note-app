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
