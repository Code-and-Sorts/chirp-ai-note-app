import re
from unittest.mock import patch

import pytest

from config.settings import ChirpSettings
from llm.exceptions import LLMGenerationFailed
from notes_chat.prompting import (
    enhanced_search_and_answer_stream,
    generate_answer,
    is_simple_conversational,
    stream_answer_tokens,
)


class TestPrompting:
    @pytest.fixture(autouse=True)
    def _llm_fakes(
        self, fake_llm_client, fake_chat_tokens, fake_chat_text, raise_llm_error
    ):
        """Bind the shared `llm.client` fakes (story 6.5) for every test."""
        self.fake_llm_client = fake_llm_client
        self.fake_chat_tokens = fake_chat_tokens
        self.fake_chat_text = fake_chat_text
        self.raise_llm_error = raise_llm_error

    def _stream_client(self, tokens=None, error=None):
        """A fake client whose ``chat_stream_sync`` yields ``tokens`` then
        raises ``error`` (if given), mimicking a daemon that dies mid-stream.
        Calls are recorded on ``client.chat_stream_sync.calls``."""
        return self.fake_llm_client(
            chat_stream_sync=self.fake_chat_tokens(list(tokens or []), error=error)
        )

    # --- generate_answer (chirpd one-shot) -----------------------------------

    def test_generate_answer_routes_through_llm_client(self):
        """generate_answer hands the templated prompt to LLMClient.chat_sync."""
        client = self.fake_llm_client(chat_sync=self.fake_chat_text("Test answer"))
        config = ChirpSettings()
        question = "What was decided?"
        context = "2025-01-15 · meeting.md\nWe decided to implement the new feature."

        result = generate_answer(config, question, context, client=client)

        assert result == {"success": True, "answer": "Test answer"}
        assert len(client.chat_sync.calls) == 1
        call = client.chat_sync.calls[0]
        assert call["model"] == "default"
        messages = call["messages"]
        assert messages[-1]["role"] == "user"
        prompt = messages[-1]["content"]
        assert "based ONLY on the provided meeting notes" in prompt
        assert question in prompt
        assert context in prompt

    def test_generate_answer_low_confidence_answer_is_returned(self):
        client = self.fake_llm_client(
            chat_sync=self.fake_chat_text(
                "I don't have enough information to answer that question."
            )
        )
        result = generate_answer(
            ChirpSettings(), "hi", "Some meeting notes", client=client
        )
        assert result["success"] is True
        assert "I don't have enough information" in result["answer"]

    def test_generate_answer_empty_context_handling(self):
        """Empty context is rejected before reaching the LLM.

        The fake exposes no ``chat_sync`` at all, so reaching the LLM would
        fail loudly with ``AttributeError``."""
        result = generate_answer(
            ChirpSettings(), "What?", "", client=self.fake_llm_client()
        )
        assert not result["success"]
        assert "Empty context" in result["error"]

    def test_generate_answer_empty_response_handling(self):
        client = self.fake_llm_client(chat_sync=self.fake_chat_text("   "))
        result = generate_answer(ChirpSettings(), "Q?", "ctx", client=client)
        assert not result["success"]
        assert "Empty response" in result["error"]

    def test_generate_answer_propagates_llm_error(self):
        client = self.fake_llm_client(
            chat_sync=self.raise_llm_error(LLMGenerationFailed, "inference failed")
        )
        with pytest.raises(LLMGenerationFailed):
            generate_answer(ChirpSettings(), "Q?", "ctx", client=client)

    # --- stream_answer_tokens (chirpd one-shot stream) -----------------------

    def test_stream_answer_tokens_yields_through_client(self):
        client = self._stream_client(tokens=["Hello ", "world"])
        tokens = list(stream_answer_tokens(ChirpSettings(), "q?", "ctx", client=client))
        assert tokens == ["Hello ", "world"]
        assert client.chat_stream_sync.calls[0]["messages"][-1]["role"] == "user"

    def test_stream_answer_tokens_skips_empty_context(self):
        client = self.fake_llm_client()  # no chat_stream_sync: must not be called
        tokens = list(stream_answer_tokens(ChirpSettings(), "q?", "", client=client))
        assert tokens == []

    # --- is_simple_conversational --------------------------------------------

    def test_is_simple_conversational_exact_matches(self):
        for phrase in [
            "hi",
            "hello",
            "thanks",
            "good morning",
            "how are you",
            "what can you do",
            "who are you",
        ]:
            assert is_simple_conversational(phrase), f"'{phrase}' should be simple"

    def test_is_simple_conversational_only_matches_greetings(self):
        # Short but searchy inputs must NOT be treated as conversational —
        # they route to the notes search, not the chat path.
        assert is_simple_conversational("thanks a lot") is False
        assert is_simple_conversational("quick question") is False
        assert is_simple_conversational("budget") is False
        assert is_simple_conversational("roadmap") is False

    def test_is_simple_conversational_case_insensitive(self):
        assert is_simple_conversational("Hi")
        assert is_simple_conversational("HELLO")

    def test_is_simple_conversational_returns_false_for_search_question(self):
        assert not is_simple_conversational(
            "what did we decide about the project budget?"
        )

    # --- enhanced_search_and_answer_stream (interactive chirpd path) ---------

    def test_stream_first_event_is_request_started(self):
        client = self._stream_client(tokens=["x"])

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "hi", client=client)
        )

        assert events[0]["type"] == "request_started"
        assert re.fullmatch(r"r-[0-9a-f]{12}", events[0]["req_id"])
        # The surfaced id is the same one threaded to the daemon for cancellation.
        assert client.chat_stream_sync.calls[0]["request_id"] == events[0]["req_id"]

    def test_stream_simple_conversational(self):
        client = self._stream_client(tokens=["Hi", " there!"])

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

    def test_stream_conversational_error_event(self):
        client = self._stream_client(error=LLMGenerationFailed("boom"))

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "hello", client=client)
        )

        assert any(e["type"] == "error" for e in events)
        assert not any(e["type"] == "complete" for e in events)

    def test_stream_conversational_empty_response(self):
        # An empty conversational stream yields an error, not a silent complete.
        client = self._stream_client(tokens=[])

        events = list(
            enhanced_search_and_answer_stream(ChirpSettings(), "hi", client=client)
        )

        assert any(e["type"] == "error" and "Empty" in e["message"] for e in events)
        assert not any(e["type"] == "complete" for e in events)

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_stream_search_cache_hit(self, mock_retrieve, mock_get_cached, mock_cache):
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = "cached result"
        client = self._stream_client()

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what did we discuss?", client=client
            )
        )

        complete = [e for e in events if e["type"] == "complete"]
        assert len(complete) == 1
        assert complete[0]["answer"] == "cached result"
        assert complete[0].get("from_cache") is True
        # Cache hits never reach the daemon.
        assert client.chat_stream_sync.calls == []

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_stream_search_streams_grounded_answer(
        self, mock_retrieve, mock_get_cached, mock_cache
    ):
        mock_retrieve.return_value = {
            "success": True,
            "context": "budget ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        client = self._stream_client(tokens=["The budget", " is $100k."])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what about the budget?", client=client
            )
        )

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["The budget", " is $100k."]
        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "The budget is $100k."
        assert complete[0]["search_strategy"] == "notes search"
        assert complete[0]["sources"] == ["src"]
        mock_cache.assert_called_once()
        # The grounded prompt and run-level id reached the daemon.
        call = client.chat_stream_sync.calls[0]
        assert call["request_id"] == events[0]["req_id"]
        assert "budget ctx" in call["messages"][-1]["content"]

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_stream_search_no_results_streams_helpful_reply(
        self, mock_retrieve, mock_get_cached, mock_cache
    ):
        mock_retrieve.return_value = {"success": False}
        client = self._stream_client(tokens=["Sorry, ", "nothing found."])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what about the budget?", client=client
            )
        )

        # Falls back to a conversational "nothing found" stream, not an error.
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["Sorry, ", "nothing found."]
        complete = [e for e in events if e["type"] == "complete"]
        assert complete[0]["answer"] == "Sorry, nothing found."
        mock_cache.assert_not_called()

    @patch("notes_chat.retrieval.retrieve_context")
    def test_stream_search_retrieval_raises_yields_error(self, mock_retrieve):
        mock_retrieve.side_effect = RuntimeError("db offline: /secret/path")
        client = self._stream_client(tokens=["unused"])

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what about the budget?", client=client
            )
        )

        errors = [e for e in events if e["type"] == "error"]
        assert errors, "expected an error event"
        # Stable, user-friendly message — raw exception detail is NOT leaked.
        assert errors[0]["message"] == "Search failed. Please try again."
        assert "db offline" not in errors[0]["message"]
        assert "/secret/path" not in errors[0]["message"]
        assert not any(e["type"] == "complete" for e in events)
        assert client.chat_stream_sync.calls == []  # never reached the daemon

    @patch("notes_chat.cache.cache_answer")
    @patch("notes_chat.cache.get_cached_answer")
    @patch("notes_chat.retrieval.retrieve_context")
    def test_stream_search_grounded_error_mid_stream(
        self, mock_retrieve, mock_get_cached, mock_cache
    ):
        mock_retrieve.return_value = {
            "success": True,
            "context": "ctx",
            "retrieved_ids": ["id1"],
            "sources": ["src"],
        }
        mock_get_cached.return_value = None
        client = self._stream_client(
            tokens=["partial "], error=LLMGenerationFailed("died mid-stream")
        )

        events = list(
            enhanced_search_and_answer_stream(
                ChirpSettings(), "what about the budget?", client=client
            )
        )

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == ["partial "]
        assert [e["type"] for e in events].count("error") == 1
        assert not any(e["type"] == "complete" for e in events)
        mock_cache.assert_not_called()
