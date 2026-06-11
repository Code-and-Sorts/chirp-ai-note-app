import json
import re
from unittest.mock import Mock, patch

import requests

from config.settings import ChirpSettings
from llm.exceptions import LLMGenerationFailed
from notes_chat.prompting import (
    enhanced_search_and_answer,
    enhanced_search_and_answer_stream,
    fast_search_and_answer,
    generate_answer,
    generate_conversational_response,
    is_obvious_search,
    is_search_query,
    is_simple_conversational,
    orchestrate_search,
    validate_ollama_connection,
)


class _FakeStreamClient:
    """Scripts ``chat_stream_sync`` output for the streaming-router tests.

    Records each call's ``request_id`` so tests can assert the run-level id is
    threaded to the daemon. Raises ``error`` (if provided) on iteration to
    mimic a daemon-side ``LLMError``; ``error_after`` controls how many tokens
    stream first (0 = before any token, the default).
    """

    def __init__(self, tokens=None, error=None, error_after=0):
        self._tokens = list(tokens or [])
        self._error = error
        self._error_after = error_after
        self.calls: list[dict] = []

    def chat_stream_sync(
        self, messages, model="default", options=None, request_id=None
    ):
        self.calls.append(
            {"messages": messages, "model": model, "request_id": request_id}
        )
        for i, tok in enumerate(self._tokens):
            if self._error is not None and i >= self._error_after:
                raise self._error
            yield tok
        if self._error is not None:
            raise self._error


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

    def test_enhanced_search_and_answer_stream_first_event_is_request_started(self):
        client = _FakeStreamClient(tokens=["x"])

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "hi", client=client)
        )

        assert events[0]["type"] == "request_started"
        assert re.fullmatch(r"r-[0-9a-f]{12}", events[0]["req_id"])
        # The surfaced id is the same one threaded to the daemon for cancellation.
        assert client.calls[0]["request_id"] == events[0]["req_id"]

    def test_enhanced_search_and_answer_stream_simple_conversational(self):
        client = _FakeStreamClient(tokens=["Hi", " there!"])

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "hi", client=client)
        )

        types = [e["type"] for e in events]
        assert types[0] == "request_started"
        assert "thinking" in types
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["Hi", " there!"]
        assert events[-1]["type"] == "complete"
        assert events[-1]["answer"] == "Hi there!"

    def test_enhanced_search_and_answer_stream_conversational_error_event(self):
        client = _FakeStreamClient(error=LLMGenerationFailed("boom"))

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "hello", client=client)
        )

        assert any(e["type"] == "error" for e in events)
        assert not any(e["type"] == "complete" for e in events)

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_stream_obvious_search_cache_hit(
        self, mock_retrieve, mock_get_cached, mock_cache
    ):
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = "cached result"
        client = _FakeStreamClient()

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what did we discuss?", client=client
            )
        )

        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["answer"] == "cached result"
        assert complete_events[0].get("from_cache") is True
        # Cache hits never reach the daemon.
        assert client.calls == []

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_stream_obvious_search_streams_tokens(
        self, mock_retrieve, mock_get_cached, mock_cache
    ):
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        client = _FakeStreamClient(tokens=["The answer", " is here."])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what did we discuss?", client=client
            )
        )

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["The answer", " is here."]
        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "The answer is here."
        assert complete[0]["search_strategy"] == "fast search"
        mock_cache.assert_called_once()
        # The run-level id is threaded to the daemon on this branch too.
        assert client.calls[0]["request_id"] == events[0]["req_id"]

    def test_enhanced_search_and_answer_stream_error_after_partial_tokens(self):
        client = _FakeStreamClient(
            tokens=["partial ", "answer "],
            error=LLMGenerationFailed("died mid-stream"),
            error_after=2,
        )

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "hi", client=client)
        )

        # Tokens stream up to the failure, then exactly one error, no complete.
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["partial ", "answer "]
        assert [e["type"] for e in events].count("error") == 1
        assert not any(e["type"] == "complete" for e in events)

    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_stream_obvious_search_error_event(
        self, mock_retrieve, mock_get_cached
    ):
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
        }
        mock_get_cached.return_value = None
        client = _FakeStreamClient(error=LLMGenerationFailed("stream failed"))

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what did we discuss?", client=client
            )
        )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.prompting.orchestrate_search")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_enhanced_search_and_answer_stream_obvious_search_retrieve_fails_falls_through(
        self, mock_retrieve, mock_orchestrate
    ):
        mock_retrieve.side_effect = RuntimeError("db offline")
        mock_orchestrate.return_value = {"success": False, "error": "down"}

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what did we discuss?", client=_FakeStreamClient()
            )
        )

        # The obvious-search branch fails on retrieval and falls through to the
        # orchestration path, which emits its own "Analyzing question..." thinking.
        thinking = [e["message"] for e in events if e["type"] == "thinking"]
        assert any("Analyzing" in m for m in thinking)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_orchestration_failure_retrieve_fails(
        self, mock_orchestrate
    ):
        mock_orchestrate.return_value = {"success": False, "error": "LLM down"}

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = {"success": False}

            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(),
                    "totally ambiguous neutral question is here now",
                    client=_FakeStreamClient(),
                )
            )

        assert any(e["type"] == "error" for e in events)
        error_msgs = [e["message"] for e in events if e["type"] == "error"]
        assert any("Could not find" in m for m in error_msgs)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_orchestration_failure_retrieve_exception(
        self, mock_orchestrate
    ):
        mock_orchestrate.return_value = {"success": False, "error": "LLM down"}

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.side_effect = RuntimeError("db gone")

            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(),
                    "totally ambiguous neutral question is here now",
                    client=_FakeStreamClient(),
                )
            )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_no_search_required(
        self, mock_orchestrate
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
        client = _FakeStreamClient(tokens=["Hey there!"])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(),
                "totally ambiguous neutral question is here now",
                client=client,
            )
        )

        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "Hey there!"

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_no_search_required_error_event(
        self, mock_orchestrate
    ):
        mock_orchestrate.return_value = {
            "success": True,
            "search_plan": {
                "search_terms": [],
                "time_filter": None,
                "search_strategy": "casual",
                "requires_search": False,
            },
        }
        client = _FakeStreamClient(error=LLMGenerationFailed("conv failed"))

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(),
                "totally ambiguous neutral question is here now",
                client=client,
            )
        )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_retrieve_fails_no_results(
        self, mock_orchestrate, mock_retrieve, mock_get_cached
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
        mock_retrieve.return_value = {"success": False}
        client = _FakeStreamClient(tokens=["Sorry, nothing found."])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(),
                "totally ambiguous neutral question is here now",
                client=client,
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
        client = _FakeStreamClient()

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(),
                "totally ambiguous neutral question is here now",
                client=client,
            )
        )

        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "cached stream answer"
        assert complete[0].get("from_cache") is True
        mock_cache.assert_not_called()
        assert client.calls == []

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_full_path(
        self, mock_orchestrate, mock_retrieve, mock_get_cached, mock_cache
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
            "context": "budget context",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        client = _FakeStreamClient(tokens=["Budget", " is $100k"])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(),
                "totally ambiguous neutral question is here now",
                client=client,
            )
        )

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["Budget", " is $100k"]
        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "Budget is $100k"
        assert complete[0]["search_strategy"] == "find budget"
        mock_cache.assert_called_once()
        # The grounded answer streamed through the daemon with the run-level id.
        assert client.calls[0]["request_id"] == events[0]["req_id"]

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_full_path_error_event(
        self, mock_orchestrate, mock_retrieve, mock_get_cached, mock_cache
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
            "context": "budget context",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        client = _FakeStreamClient(error=LLMGenerationFailed("stream died"))

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(),
                "totally ambiguous neutral question is here now",
                client=client,
            )
        )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_empty_response(
        self, mock_orchestrate, mock_retrieve, mock_get_cached, mock_cache
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
            "context": "budget context",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        client = _FakeStreamClient(tokens=[])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(),
                "totally ambiguous neutral question is here now",
                client=client,
            )
        )

        assert any(e["type"] == "error" and "Empty" in e["message"] for e in events)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_exception_yields_error(
        self, mock_orchestrate
    ):
        mock_orchestrate.side_effect = RuntimeError("total failure")

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(),
                "totally ambiguous neutral question is here now",
                client=_FakeStreamClient(),
            )
        )

        assert any(
            e["type"] == "error" and "total failure" in e["message"] for e in events
        )

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_orchestrate_failure_streams_context(
        self, mock_orchestrate
    ):
        mock_orchestrate.return_value = {"success": False, "error": "down"}

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = {
                "success": True,
                "context": "fallback ctx",
                "retrieved_ids": ["id1"],
                "sources": [],
            }
            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(),
                    "totally ambiguous neutral question is here now",
                    client=_FakeStreamClient(tokens=["Fallback answer."]),
                )
            )

        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "Fallback answer."

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_retrieve_no_results_error_event(
        self, mock_orchestrate
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

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = {"success": False}
            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(),
                    "totally ambiguous neutral question is here now",
                    client=_FakeStreamClient(error=LLMGenerationFailed("conv died")),
                )
            )

        assert any(e["type"] == "error" for e in events)

    @patch("notes_chat.prompting.orchestrate_search")
    def test_enhanced_search_and_answer_stream_orchestrate_failure_stream_error_event(
        self, mock_orchestrate
    ):
        mock_orchestrate.return_value = {"success": False, "error": "down"}

        with patch("notes_chat.retrieval.retrieve_context") as mock_retrieve:
            mock_retrieve.return_value = {
                "success": True,
                "context": "ctx",
                "retrieved_ids": ["id1"],
                "sources": [],
            }
            events = list(
                enhanced_search_and_answer_stream(
                    ChirpSettings(),
                    "totally ambiguous neutral question is here now",
                    client=_FakeStreamClient(error=LLMGenerationFailed("fsa died")),
                )
            )

        assert any(e["type"] == "error" for e in events)
