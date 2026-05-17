import json
from unittest.mock import Mock, patch

import requests

from config.settings import ChirpSettings
from notes_chat.prompting import (
    enhanced_search_and_answer,
    fast_search_and_answer,
    fast_search_and_answer_stream,
    generate_answer,
    generate_conversational_response,
    generate_conversational_response_stream,
    is_obvious_search,
    is_search_query,
    is_simple_conversational,
    orchestrate_search,
    stream_llm_response,
    validate_ollama_connection,
)


class TestPrompting:
    def test_generate_answer_routes_through_llm_client(self):
        """generate_answer hands the templated prompt to LLMClient.chat_sync."""

        class _StubClient:
            def __init__(self) -> None:
                self.calls: list[tuple[list[dict], str]] = []

            def chat_sync(self, messages, model="default", **_):
                self.calls.append((messages, model))
                return "Test answer"

        client = _StubClient()
        config = ChirpSettings()
        question = "What was decided?"
        context = "2025-01-15 · meeting.md\nWe decided to implement the new feature."

        result = generate_answer(config, question, context, client=client)

        assert result == {"success": True, "answer": "Test answer"}
        assert len(client.calls) == 1
        messages, model = client.calls[0]
        assert model == "default"
        assert messages[-1]["role"] == "user"
        prompt = messages[-1]["content"]
        assert "based ONLY on the provided meeting notes" in prompt
        assert question in prompt
        assert context in prompt

    def test_generate_answer_low_confidence_answer_is_returned(self):
        class _StubClient:
            def chat_sync(self, *a, **kw):
                return "I don't have enough information to answer that question."

        result = generate_answer(
            ChirpSettings(), "hi", "Some meeting notes", client=_StubClient()
        )
        assert result["success"] is True
        assert "I don't have enough information" in result["answer"]

    def test_generate_answer_empty_context_handling(self):
        """Empty context is rejected before reaching the LLM."""

        class _BoomClient:
            def chat_sync(self, *a, **kw):
                raise AssertionError("chat_sync must not be called for empty context")

        result = generate_answer(ChirpSettings(), "What?", "", client=_BoomClient())
        assert not result["success"]
        assert "Empty context" in result["error"]

    def test_generate_answer_empty_response_handling(self):
        class _StubClient:
            def chat_sync(self, *a, **kw):
                return "   "

        result = generate_answer(ChirpSettings(), "Q?", "ctx", client=_StubClient())
        assert not result["success"]
        assert "Empty response" in result["error"]

    def test_generate_answer_propagates_llm_error(self):
        from llm.exceptions import LLMGenerationFailed

        class _BoomClient:
            def chat_sync(self, *a, **kw):
                raise LLMGenerationFailed("inference failed", details={})

        import pytest

        with pytest.raises(LLMGenerationFailed):
            generate_answer(ChirpSettings(), "Q?", "ctx", client=_BoomClient())

    def test_stream_answer_tokens_yields_through_client(self):
        class _StubClient:
            def chat_stream_sync(self, messages, model="default", **_):
                assert messages[-1]["role"] == "user"
                yield "Hello "
                yield "world"

        from notes_chat.prompting import stream_answer_tokens

        tokens = list(
            stream_answer_tokens(ChirpSettings(), "q?", "ctx", client=_StubClient())
        )
        assert tokens == ["Hello ", "world"]

    def test_stream_answer_tokens_skips_empty_context(self):
        class _BoomClient:
            def chat_stream_sync(self, *a, **kw):
                raise AssertionError("must not be called for empty context")
                yield  # pragma: no cover

        from notes_chat.prompting import stream_answer_tokens

        tokens = list(
            stream_answer_tokens(ChirpSettings(), "q?", "", client=_BoomClient())
        )
        assert tokens == []

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

    @patch("requests.post")
    def test_conversational_response_500_error(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = generate_conversational_response(ChirpSettings(), "Hello")

        assert not result["success"]
        assert "ollama serve" in result["error"]

    @patch("requests.post")
    def test_conversational_response_empty_response(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": ""}
        mock_post.return_value = mock_response

        result = generate_conversational_response(ChirpSettings(), "Hello")

        assert not result["success"]
        assert "Empty response" in result["error"]

    @patch("requests.post")
    def test_conversational_response_requests_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError()

        result = generate_conversational_response(ChirpSettings(), "Hello")

        assert not result["success"]
        assert "Cannot connect to Ollama" in result["error"]

    @patch("requests.post")
    def test_conversational_response_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout()

        result = generate_conversational_response(ChirpSettings(), "Hello")

        assert not result["success"]
        assert "timed out" in result["error"]

    @patch("requests.post")
    def test_conversational_response_generic_exception(self, mock_post):
        mock_post.side_effect = ValueError("nope")

        result = generate_conversational_response(ChirpSettings(), "Hello")

        assert not result["success"]
        assert "nope" in result["error"]

    @patch("requests.post")
    def test_orchestrate_search_success_with_json_embedded_in_text(self, mock_post):
        search_plan = {
            "search_terms": ["budget"],
            "time_filter": None,
            "search_strategy": "find budget discussions",
            "requires_search": True,
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": f"Here is my analysis:\n{json.dumps(search_plan)}"
        }
        mock_post.return_value = mock_response

        result = orchestrate_search(ChirpSettings(), "what about the budget?")

        assert result["success"]
        assert result["search_plan"]["search_terms"] == ["budget"]

    @patch("requests.post")
    def test_orchestrate_search_success_raw_json(self, mock_post):
        search_plan = {
            "search_terms": ["action items"],
            "time_filter": "last week",
            "search_strategy": "find action items from last week",
            "requires_search": True,
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": json.dumps(search_plan)}
        mock_post.return_value = mock_response

        result = orchestrate_search(ChirpSettings(), "action items from last week?")

        assert result["success"]
        assert result["search_plan"]["time_filter"] == "last week"

    @patch("requests.post")
    def test_orchestrate_search_api_error(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 503
        mock_post.return_value = mock_response

        result = orchestrate_search(ChirpSettings(), "what happened?")

        assert not result["success"]
        assert "503" in result["error"]

    @patch("requests.post")
    def test_orchestrate_search_empty_response(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": ""}
        mock_post.return_value = mock_response

        result = orchestrate_search(ChirpSettings(), "what happened?")

        assert not result["success"]
        assert "Empty response" in result["error"]

    @patch("requests.post")
    def test_orchestrate_search_invalid_json(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "not valid json at all!!!"}
        mock_post.return_value = mock_response

        result = orchestrate_search(ChirpSettings(), "what happened?")

        assert not result["success"]
        assert "Failed to parse" in result["error"]

    @patch("requests.post")
    def test_orchestrate_search_missing_required_keys(self, mock_post):
        incomplete_plan = {"search_terms": ["budget"]}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": json.dumps(incomplete_plan)}
        mock_post.return_value = mock_response

        result = orchestrate_search(ChirpSettings(), "budget discussion?")

        assert not result["success"]
        assert "Invalid response format" in result["error"]

    @patch("requests.post")
    def test_orchestrate_search_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError()

        result = orchestrate_search(ChirpSettings(), "what happened?")

        assert not result["success"]
        assert "Cannot connect to Ollama" in result["error"]

    @patch("requests.post")
    def test_orchestrate_search_generic_exception(self, mock_post):
        mock_post.side_effect = RuntimeError("unexpected")

        result = orchestrate_search(ChirpSettings(), "what happened?")

        assert not result["success"]
        assert "unexpected" in result["error"]

    @patch("requests.post")
    def test_stream_llm_response_yields_tokens(self, mock_post):
        lines = [
            json.dumps({"response": "Hello", "done": False}).encode(),
            json.dumps({"response": " world", "done": True}).encode(),
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_response

        tokens = list(stream_llm_response(ChirpSettings(), "test prompt"))

        assert tokens == ["Hello", " world"]

    @patch("requests.post")
    def test_stream_llm_response_api_error(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 503
        mock_post.return_value = mock_response

        tokens = list(stream_llm_response(ChirpSettings(), "test prompt"))

        assert len(tokens) == 1
        assert "503" in tokens[0]

    @patch("requests.post")
    def test_stream_llm_response_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError()

        tokens = list(stream_llm_response(ChirpSettings(), "test prompt"))

        assert len(tokens) == 1
        assert "Cannot connect to Ollama" in tokens[0]

    @patch("requests.post")
    def test_stream_llm_response_generic_exception(self, mock_post):
        mock_post.side_effect = RuntimeError("stream broke")

        tokens = list(stream_llm_response(ChirpSettings(), "test prompt"))

        assert len(tokens) == 1
        assert "stream broke" in tokens[0]

    @patch("requests.post")
    def test_stream_llm_response_skips_invalid_json_lines(self, mock_post):
        lines = [
            b"not-json",
            json.dumps({"response": "ok", "done": True}).encode(),
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_response

        tokens = list(stream_llm_response(ChirpSettings(), "test"))

        assert tokens == ["ok"]

    @patch("requests.post")
    def test_stream_llm_response_nonzero_temperature_sets_top_p(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter([])
        mock_post.return_value = mock_response

        list(stream_llm_response(ChirpSettings(), "test", temperature=0.5))

        payload = mock_post.call_args[1]["json"]
        assert payload["top_p"] == 0.9
        assert payload["temperature"] == 0.5

    @patch("requests.post")
    def test_generate_conversational_response_stream_delegates(self, mock_post):
        lines = [json.dumps({"response": "Hi!", "done": True}).encode()]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_response

        tokens = list(generate_conversational_response_stream(ChirpSettings(), "Hello"))

        assert tokens == ["Hi!"]
        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.3

    @patch("requests.post")
    def test_fast_search_and_answer_stream_delegates(self, mock_post):
        lines = [json.dumps({"response": "Found it.", "done": True}).encode()]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_response

        tokens = list(
            fast_search_and_answer_stream(
                ChirpSettings(), "what happened?", "some context"
            )
        )

        assert tokens == ["Found it."]
        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0
        assert "what happened?" in payload["prompt"]
        assert "some context" in payload["prompt"]

    def test_is_simple_conversational_exact_matches(self):
        for phrase in [
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "what can you do",
            "help",
            "what are you",
            "who are you",
        ]:
            assert is_simple_conversational(phrase), (
                f"'{phrase}' should be simple conversational"
            )

    def test_is_simple_conversational_two_word_questions(self):
        assert is_simple_conversational("thanks a lot") is False
        assert is_simple_conversational("quick question") is True

    def test_is_simple_conversational_case_insensitive(self):
        assert is_simple_conversational("Hi")
        assert is_simple_conversational("HELLO")

    def test_is_simple_conversational_returns_false_for_search_question(self):
        assert not is_simple_conversational(
            "what did we decide about the project budget?"
        )

    def test_is_obvious_search_matches_patterns(self):
        for phrase in [
            "what did we decide",
            "who said that",
            "when did it happen",
            "what was discussed",
            "tell me about the meeting",
            "find the notes",
            "search for budget",
            "show me the agenda",
            "what happened",
            "meeting recap",
            "discussed the roadmap",
            "action item assigned",
            "decision was made",
            "summary of today",
            "agenda for tomorrow",
        ]:
            assert is_obvious_search(phrase), f"'{phrase}' should be obvious search"

    def test_is_obvious_search_returns_false_for_casual_chat(self):
        assert not is_obvious_search("hi there")
        assert not is_obvious_search("yes")

    def test_is_obvious_search_case_insensitive(self):
        assert is_obvious_search("MEETING notes please")

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("requests.post")
    def test_fast_search_and_answer_success(
        self, mock_post, mock_retrieve, mock_get_cached, mock_cache_answer
    ):
        mock_retrieve.return_value = {
            "success": True,
            "context": "meeting notes context",
            "retrieved_ids": ["id1"],
            "sources": ["source1"],
        }
        mock_get_cached.return_value = None
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "The answer is X"}
        mock_post.return_value = mock_response

        result = fast_search_and_answer(ChirpSettings(), "what was decided?")

        assert result["success"]
        assert result["answer"] == "The answer is X"
        assert result["search_strategy"] == "fast search"
        mock_cache_answer.assert_called_once()

    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_fast_search_and_answer_cache_hit(self, mock_retrieve, mock_get_cached):
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = "cached answer here"

        result = fast_search_and_answer(ChirpSettings(), "what happened?")

        assert result["success"]
        assert result["answer"] == "cached answer here"

    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.generate_conversational_response")
    def test_fast_search_and_answer_retrieval_failure(self, mock_conv, mock_retrieve):
        mock_retrieve.return_value = {"success": False}
        mock_conv.return_value = {"success": True, "answer": "sorry, could not find"}

        result = fast_search_and_answer(ChirpSettings(), "what happened?")

        assert result["success"]
        mock_conv.assert_called_once()

    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("requests.post")
    def test_fast_search_and_answer_api_error(
        self, mock_post, mock_retrieve, mock_get_cached
    ):
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
        }
        mock_get_cached.return_value = None
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = fast_search_and_answer(ChirpSettings(), "what happened?")

        assert not result["success"]
        assert "500" in result["error"]

    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("requests.post")
    def test_fast_search_and_answer_empty_llm_response(
        self, mock_post, mock_retrieve, mock_get_cached
    ):
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
        }
        mock_get_cached.return_value = None
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": ""}
        mock_post.return_value = mock_response

        result = fast_search_and_answer(ChirpSettings(), "what happened?")

        assert not result["success"]
        assert result["error"] == "Empty response"

    @patch("notes_chat.retrieval.retrieve_context")
    def test_fast_search_and_answer_exception_fallback(self, mock_retrieve):
        mock_retrieve.side_effect = RuntimeError("db gone")

        result = fast_search_and_answer(ChirpSettings(), "what happened?")

        assert not result["success"]
        assert "db gone" in result["error"]

    @patch("notes_chat.prompting.generate_conversational_response")
    def test_enhanced_search_and_answer_simple_conversational(self, mock_conv):
        mock_conv.return_value = {"success": True, "answer": "Hi there!"}

        result = enhanced_search_and_answer(ChirpSettings(), "hi")

        mock_conv.assert_called_once_with(ChirpSettings(), "hi")
        assert result["answer"] == "Hi there!"

    @patch("notes_chat.prompting.fast_search_and_answer")
    def test_enhanced_search_and_answer_obvious_search_routes_to_fast(self, mock_fast):
        mock_fast.return_value = {"success": True, "answer": "fast answer"}

        result = enhanced_search_and_answer(
            ChirpSettings(), "what did we discuss about the budget?"
        )

        mock_fast.assert_called_once()
        assert result["answer"] == "fast answer"

    @patch("notes_chat.prompting.fast_search_and_answer")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_orchestration_failure_falls_back(
        self, mock_orchestrate, mock_fast
    ):
        mock_orchestrate.return_value = {"success": False, "error": "LLM down"}
        mock_fast.return_value = {"success": True, "answer": "fallback answer"}

        result = enhanced_search_and_answer(
            ChirpSettings(), "complex ambiguous question here now"
        )

        mock_fast.assert_called_once()
        assert result["answer"] == "fallback answer"

    @patch("notes_chat.prompting.generate_conversational_response")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_no_search_required(
        self, mock_orchestrate, mock_conv
    ):
        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": [],
                "time_filter": None,
                "search_strategy": "casual greeting",
                "requires_search": False,
            },
        }
        mock_conv.return_value = {"success": True, "answer": "casual response"}

        result = enhanced_search_and_answer(
            ChirpSettings(), "complex ambiguous question here now"
        )

        mock_conv.assert_called_once()
        assert result["answer"] == "casual response"

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    @patch("requests.post")
    def test_enhanced_search_and_answer_full_path_success(
        self, mock_post, mock_orchestrate, mock_retrieve, mock_get_cached, mock_cache
    ):
        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["roadmap"],
                "time_filter": None,
                "search_strategy": "find roadmap discussions",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {
            "success": True,
            "context": "roadmap context",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "The roadmap is X"}
        mock_post.return_value = mock_response

        result = enhanced_search_and_answer(
            ChirpSettings(), "complex question about the roadmap please"
        )

        assert result["success"]
        assert result["answer"] == "The roadmap is X"
        assert result["search_strategy"] == "find roadmap discussions"
        mock_cache.assert_called_once()

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_cache_hit(
        self, mock_orchestrate, mock_retrieve, mock_get_cached, mock_cache
    ):
        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["roadmap"],
                "time_filter": None,
                "search_strategy": "find roadmap",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = "previously cached"

        result = enhanced_search_and_answer(
            ChirpSettings(), "complex question about the roadmap please"
        )

        assert result["success"]
        assert result["answer"] == "previously cached"
        mock_cache.assert_not_called()

    @patch("notes_chat.prompting.fast_search_and_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_retrieve_failure_falls_back(
        self, mock_orchestrate, mock_retrieve, mock_fast
    ):
        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["topic"],
                "time_filter": None,
                "search_strategy": "find topic",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {"success": False}
        mock_fast.return_value = {"success": True, "answer": "fast fallback"}

        result = enhanced_search_and_answer(
            ChirpSettings(), "complex question about some topic here"
        )

        mock_fast.assert_called_once()
        assert result["answer"] == "fast fallback"

    @patch("notes_chat.prompting.fast_search_and_answer")
    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    @patch("requests.post")
    def test_enhanced_search_and_answer_empty_llm_falls_back(
        self,
        mock_post,
        mock_orchestrate,
        mock_retrieve,
        mock_get_cached,
        mock_cache,
        mock_fast,
    ):
        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["topic"],
                "time_filter": None,
                "search_strategy": "find topic",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
        }
        mock_get_cached.return_value = None
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": ""}
        mock_post.return_value = mock_response
        mock_fast.return_value = {"success": True, "answer": "fast fallback again"}

        result = enhanced_search_and_answer(
            ChirpSettings(), "complex question about some topic here"
        )

        mock_fast.assert_called_once()
        assert result["answer"] == "fast fallback again"

    @patch("notes_chat.prompting.fast_search_and_answer")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_exception_falls_back(
        self, mock_orchestrate, mock_fast
    ):
        mock_orchestrate.side_effect = RuntimeError("crash")
        mock_fast.return_value = {"success": True, "answer": "exception fallback"}

        result = enhanced_search_and_answer(
            ChirpSettings(), "complex question about some topic here"
        )

        mock_fast.assert_called_once()
        assert result["answer"] == "exception fallback"

    @patch("notes_chat.prompting.fast_search_and_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    @patch("requests.post")
    def test_enhanced_search_and_answer_api_error_falls_back(
        self, mock_post, mock_orchestrate, mock_retrieve, mock_fast
    ):
        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["budget"],
                "time_filter": None,
                "search_strategy": "find budget",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
        }
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        mock_fast.return_value = {"success": True, "answer": "api error fallback"}

        enhanced_search_and_answer(
            ChirpSettings(), "complex question about the budget here"
        )

        mock_fast.assert_called_once()

    @patch("notes_chat.prompting.orchestrate_search")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_null_time_filter_parsed(
        self, mock_retrieve, mock_get_cached, mock_orchestrate
    ):
        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["budget"],
                "time_filter": "null",
                "search_strategy": "find budget",
                "requires_search": True,
            },
        }
        mock_get_cached.return_value = "cached"
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": [],
        }

        enhanced_search_and_answer(
            ChirpSettings(), "complex question about the budget here"
        )

        call_args = mock_retrieve.call_args[0]
        assert call_args[2] is None

    @patch("requests.get")
    def test_validate_ollama_connection_non_200(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = validate_ollama_connection(ChirpSettings())

        assert not result["success"]
        assert "not responding" in result["error"]

    @patch("requests.get")
    def test_validate_ollama_connection_missing_embedding_model(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_get.return_value = mock_response

        result = validate_ollama_connection(ChirpSettings())

        assert not result["success"]
        assert "nomic-embed-text" in result["error"]

    @patch("requests.get")
    def test_validate_ollama_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = validate_ollama_connection(ChirpSettings())

        assert not result["success"]
        assert "Cannot connect to Ollama" in result["error"]

    @patch("requests.get")
    def test_validate_ollama_connection_generic_exception(self, mock_get):
        mock_get.side_effect = RuntimeError("unexpected")

        result = validate_ollama_connection(ChirpSettings())

        assert not result["success"]
        assert "unexpected" in result["error"]

    def test_is_search_query_long_no_pattern_match(self):
        question = "a completely neutral sentence with six or more words"
        assert is_search_query(question)

    @patch("notes_chat.prompting.generate_conversational_response_stream")
    def test_enhanced_search_and_answer_stream_simple_conversational(self, mock_stream):
        mock_stream.return_value = iter(["Hi", " there!"])

        events = list(
            __import__(
                "notes_chat.prompting", fromlist=["enhanced_search_and_answer_stream"]
            ).enhanced_search_and_answer_stream(ChirpSettings(), "hi")
        )

        types = [e["type"] for e in events]
        assert "thinking" in types
        assert "token" in types
        assert events[-1]["type"] == "complete"
        assert events[-1]["answer"] == "Hi there!"

    @patch("notes_chat.prompting.generate_conversational_response_stream")
    def test_enhanced_search_and_answer_stream_conversational_error_token(
        self, mock_stream
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_stream.return_value = iter(["Error: boom"])

        events = list(enhanced_search_and_answer_stream(ChirpSettings(), "hello"))

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_stream_obvious_search_cache_hit(
        self, mock_retrieve, mock_get_cached, mock_cache
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = "cached result"

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "what did we discuss?")
        )

        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["answer"] == "cached result"
        assert complete_events[0].get("from_cache") is True

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.prompting.fast_search_and_answer_stream")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_stream_obvious_search_streams_tokens(
        self, mock_retrieve, mock_fsa_stream, mock_get_cached, mock_cache
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        mock_fsa_stream.return_value = iter(["The answer", " is here."])

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "what did we discuss?")
        )

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["The answer", " is here."]
        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "The answer is here."
        assert complete[0]["search_strategy"] == "fast search"
        mock_cache.assert_called_once()

    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.prompting.fast_search_and_answer_stream")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_stream_obvious_search_error_token(
        self, mock_retrieve, mock_fsa_stream, mock_get_cached
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
        }
        mock_get_cached.return_value = None
        mock_fsa_stream.return_value = iter(["Error: stream failed"])

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "what did we discuss?")
        )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_stream_obvious_search_retrieve_fails_falls_through(
        self, mock_retrieve
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_retrieve.side_effect = RuntimeError("db offline")

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "what did we discuss?")
        )

        assert any(e["type"] == "thinking" for e in events)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_orchestration_failure_retrieve_fails(
        self, mock_orchestrate
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {"success": False, "error": "LLM down"}

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = {"success": False}

            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(), "totally ambiguous neutral question is here now"
                )
            )

        assert any(e["type"] == "error" for e in events)
        error_msgs = [e["message"] for e in events if e["type"] == "error"]
        assert any("Could not find" in m for m in error_msgs)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_orchestration_failure_retrieve_exception(
        self, mock_orchestrate
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {"success": False, "error": "LLM down"}

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.side_effect = RuntimeError("db gone")

            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(), "totally ambiguous neutral question is here now"
                )
            )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.prompting.generate_conversational_response_stream")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_no_search_required(
        self, mock_orchestrate, mock_conv_stream
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": [],
                "time_filter": None,
                "search_strategy": "casual greeting",
                "requires_search": False,
            },
        }
        mock_conv_stream.return_value = iter(["Hey there!"])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "totally ambiguous neutral question is here now"
            )
        )

        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "Hey there!"

    @patch("notes_chat.prompting.generate_conversational_response_stream")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_no_search_required_error_token(
        self, mock_orchestrate, mock_conv_stream
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": [],
                "time_filter": None,
                "search_strategy": "casual",
                "requires_search": False,
            },
        }
        mock_conv_stream.return_value = iter(["Error: conv failed"])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "totally ambiguous neutral question is here now"
            )
        )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_retrieve_fails_no_results(
        self, mock_orchestrate, mock_retrieve, mock_get_cached
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["roadmap"],
                "time_filter": None,
                "search_strategy": "find roadmap",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {"success": False}

        with patch(
            "notes_chat.prompting.generate_conversational_response_stream"
        ) as mock_conv:
            mock_conv.return_value = iter(["Sorry, nothing found."])
            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(), "totally ambiguous neutral question is here now"
                )
            )

        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "Sorry, nothing found."

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_cache_hit(
        self, mock_orchestrate, mock_retrieve, mock_get_cached, mock_cache
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["budget"],
                "time_filter": None,
                "search_strategy": "find budget",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = "cached stream answer"

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "totally ambiguous neutral question is here now"
            )
        )

        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "cached stream answer"
        assert complete[0].get("from_cache") is True
        mock_cache.assert_not_called()

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.prompting.stream_llm_response")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_full_path(
        self, mock_orchestrate, mock_retrieve, mock_stream, mock_get_cached, mock_cache
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["budget"],
                "time_filter": None,
                "search_strategy": "find budget",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {
            "success": True,
            "context": "budget context",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        mock_stream.return_value = iter(["Budget", " is $100k"])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "totally ambiguous neutral question is here now"
            )
        )

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["Budget", " is $100k"]
        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "Budget is $100k"
        assert complete[0]["search_strategy"] == "find budget"
        mock_cache.assert_called_once()

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.prompting.stream_llm_response")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_full_path_error_token(
        self, mock_orchestrate, mock_retrieve, mock_stream, mock_get_cached, mock_cache
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["budget"],
                "time_filter": None,
                "search_strategy": "find budget",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {
            "success": True,
            "context": "budget context",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        mock_stream.return_value = iter(["Error: stream died"])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "totally ambiguous neutral question is here now"
            )
        )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.prompting.stream_llm_response")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_empty_response(
        self, mock_orchestrate, mock_retrieve, mock_stream, mock_get_cached, mock_cache
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["budget"],
                "time_filter": None,
                "search_strategy": "find budget",
                "requires_search": True,
            },
        }
        mock_retrieve.return_value = {
            "success": True,
            "context": "budget context",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        mock_stream.return_value = iter([])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "totally ambiguous neutral question is here now"
            )
        )

        assert any(e["type"] == "error" and "Empty" in e["message"] for e in events)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_exception_yields_error(
        self, mock_orchestrate
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.side_effect = RuntimeError("total failure")

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "totally ambiguous neutral question is here now"
            )
        )

        assert any(
            e["type"] == "error" and "total failure" in e["message"] for e in events
        )

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_orchestrate_failure_streams_context(
        self, mock_orchestrate
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {"success": False, "error": "down"}

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = {
                "success": True,
                "context": "fallback ctx",
                "retrieved_ids": ["id1"],
                "sources": [],
            }
            with patch(
                "notes_chat.prompting.fast_search_and_answer_stream"
            ) as mock_fsa:
                mock_fsa.return_value = iter(["Fallback answer."])
                events = list(
                    enhanced_search_and_answer_stream(
                        ChirpSettings(),
                        "totally ambiguous neutral question is here now",
                    )
                )

        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "Fallback answer."

    @patch("notes_chat.prompting.generate_conversational_response_stream")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_retrieve_no_results_error_token(
        self, mock_orchestrate, mock_conv_stream
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": ["roadmap"],
                "time_filter": None,
                "search_strategy": "find roadmap",
                "requires_search": True,
            },
        }
        mock_conv_stream.return_value = iter(["Error: conv died"])

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = {"success": False}
            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(), "totally ambiguous neutral question is here now"
                )
            )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_orchestrate_failure_stream_error_token(
        self, mock_orchestrate
    ):
        from notes_chat.prompting import enhanced_search_and_answer_stream

        mock_orchestrate.return_value = {"success": False, "error": "down"}

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = {
                "success": True,
                "context": "ctx",
                "retrieved_ids": ["id1"],
                "sources": [],
            }
            with patch(
                "notes_chat.prompting.fast_search_and_answer_stream"
            ) as mock_fsa:
                mock_fsa.return_value = iter(["Error: fsa died"])
                events = list(
                    enhanced_search_and_answer_stream(
                        ChirpSettings(),
                        "totally ambiguous neutral question is here now",
                    )
                )

        assert any(e["type"] == "error" for e in events)
