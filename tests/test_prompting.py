from unittest.mock import Mock, patch

from config.settings import ChirpSettings
from notes_chat.prompting import (
    generate_answer,
    generate_conversational_response,
    is_search_query,
    validate_ollama_connection,
)


class TestPrompting:
    @patch("requests.post")
    def test_prompt_includes_instruction_and_sources(self, mock_post):
        """Test that prompt includes proper instruction and source headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test answer"}
        mock_post.return_value = mock_response

        config = ChirpSettings()
        question = "What was decided?"
        context = "2025-01-15 · meeting.md\nWe decided to implement the new feature."

        result = generate_answer(config, question, context)

        assert result["success"]
        assert result["answer"] == "Test answer"

        call_args = mock_post.call_args
        prompt = call_args[1]["json"]["prompt"]

        assert "based ONLY on the provided meeting notes" in prompt
        assert question in prompt
        assert context in prompt
        assert "temperature" in call_args[1]["json"]
        assert call_args[1]["json"]["temperature"] == 0

    @patch("requests.post")
    def test_budget_cap_enforced(self, mock_post):
        """Test that context budget cap is enforced."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer"}
        mock_post.return_value = mock_response

        config = ChirpSettings()
        question = "Test question"

        context = "X" * 20000

        result = generate_answer(config, question, context)

        assert result["success"]

        call_args = mock_post.call_args
        prompt = call_args[1]["json"]["prompt"]
        assert len(prompt) > 10000

    @patch("requests.post")
    def test_not_found_ambiguous_responses(self, mock_post):
        """Test handling of 'not found' and 'ambiguous' responses."""
        config = ChirpSettings()
        question = "What happened?"
        context = "Some meeting notes"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "I don't have enough information to answer that question."
        }
        mock_post.return_value = mock_response

        result = generate_answer(config, question, context)

        assert not result["success"]
        assert "No relevant information found" in result["error"]
        assert "I don't have enough information" in result["answer"]

    @patch("requests.post")
    def test_empty_context_handling(self, mock_post):
        """Test handling of empty context."""
        config = ChirpSettings()
        question = "What happened?"
        context = ""

        result = generate_answer(config, question, context)

        assert not result["success"]
        assert "Empty context" in result["error"]
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_api_error_handling(self, mock_post):
        """Test handling of various API errors."""
        config = ChirpSettings()
        question = "Test question"
        context = "Test context"

        mock_response = Mock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        result = generate_answer(config, question, context)

        assert not result["success"]
        assert "Model" in result["error"]
        assert "not found" in result["error"]
        assert "ollama pull" in result["error"]

    @patch("requests.post")
    def test_connection_error_handling(self, mock_post):
        """Test handling of connection errors."""
        config = ChirpSettings()
        question = "Test question"
        context = "Test context"

        mock_post.side_effect = ConnectionError("Connection failed")

        result = generate_answer(config, question, context)

        assert not result["success"]
        assert "Cannot connect to Ollama" in result["error"]
        assert "ollama serve" in result["error"]

    @patch("requests.get")
    def test_ollama_validation_success(self, mock_get):
        """Test successful Ollama validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.1:8b"}, {"name": "nomic-embed-text"}]
        }
        mock_get.return_value = mock_response

        config = ChirpSettings()
        result = validate_ollama_connection(config)

        assert result["success"]

    @patch("requests.get")
    def test_ollama_validation_missing_model(self, mock_get):
        """Test Ollama validation with missing model."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "other-model"}]}
        mock_get.return_value = mock_response

        config = ChirpSettings()
        result = validate_ollama_connection(config)

        assert not result["success"]
        assert "llama3.1:8b" in result["error"]
        assert "not found" in result["error"]

    @patch("requests.post")
    def test_deterministic_settings(self, mock_post):
        """Test that deterministic settings are applied."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Answer"}
        mock_post.return_value = mock_response

        config = ChirpSettings()
        question = "Test question"
        context = "Test context"

        generate_answer(config, question, context)

        call_args = mock_post.call_args[1]["json"]
        assert call_args["temperature"] == 0
        assert call_args["top_p"] == 1
        assert call_args["stream"] is False

    @patch("requests.post")
    def test_conversational_response_success(self, mock_post):
        """Test successful conversational response generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Hello! I'm Chirp, nice to meet you!"
        }
        mock_post.return_value = mock_response

        config = ChirpSettings()
        question = "Hi there"

        result = generate_conversational_response(config, question)

        assert result["success"]
        assert "Hello! I'm Chirp" in result["answer"]

        call_args = mock_post.call_args
        prompt = call_args[1]["json"]["prompt"]

        assert "Chirp, a friendly AI assistant" in prompt
        assert question in prompt
        assert call_args[1]["json"]["temperature"] == 0.3
        assert call_args[1]["json"]["top_p"] == 0.9

    @patch("requests.post")
    def test_conversational_response_api_error(self, mock_post):
        """Test conversational response handling API errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        config = ChirpSettings()
        question = "Hello"

        result = generate_conversational_response(config, question)

        assert not result["success"]
        assert "Model" in result["error"]
        assert "not found" in result["error"]

    @patch("requests.post")
    def test_conversational_response_connection_error(self, mock_post):
        """Test conversational response handling connection errors."""
        mock_post.side_effect = ConnectionError("Connection failed")

        config = ChirpSettings()
        question = "Hello"

        result = generate_conversational_response(config, question)

        assert not result["success"]
        assert "Cannot connect to Ollama" in result["error"]

    def test_is_search_query_conversational_patterns(self):
        """Test that conversational patterns are detected correctly."""
        conversational_queries = [
            "hi",
            "hello",
            "Hi there",
            "Hey",
            "thanks",
            "thank you",
            "how are you",
            "what can you do",
            "help",
            "what are you",
            "who are you",
            "good morning",
            "goodbye",
            "bye",
        ]

        for query in conversational_queries:
            assert not is_search_query(query), f"'{query}' should be conversational"

    def test_is_search_query_search_patterns(self):
        """Test that search patterns are detected correctly."""
        search_queries = [
            "what did we discuss",
            "who said that",
            "when did the meeting happen",
            "what was discussed in the standup",
            "tell me about the project updates",
            "find information about the budget",
            "search for action items",
            "show me what happened yesterday",
            "what were the meeting topics",
            "any decisions made about the feature",
            "meeting summary from last week",
        ]

        for query in search_queries:
            assert is_search_query(query), f"'{query}' should be a search query"

    def test_is_search_query_long_questions(self):
        """Test that long questions default to search mode."""
        long_question = "Can you please help me understand what the team discussed about the new feature implementation timeline"
        assert is_search_query(long_question)

    def test_is_search_query_short_ambiguous(self):
        """Test handling of short, ambiguous questions."""
        short_questions = ["what?", "how?", "why?", "really?", "okay"]

        for query in short_questions:
            result = is_search_query(query)
            assert isinstance(result, bool), f"'{query}' should return a boolean"
