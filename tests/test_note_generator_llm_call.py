"""Tests for the note-generator LLM call site after the chirpd cutover (6.2).

`_call_llm` routes note generation through `llm.client` instead of the Ollama
HTTP API. These tests assert the call shape, the streamed-token aggregation,
and that LLM errors propagate into the existing best-effort `return None` path.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch

import pytest

from llm.exceptions import LLMModelError
from notes.note_generator import SYSTEM_PROMPT, NoteGenerator


class FakeLLMClient:
    """Stand-in for ``llm.client.LLMClient`` that records the chat call."""

    def __init__(
        self,
        tokens: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.tokens = tokens or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat_stream_sync(
        self,
        messages: list[dict[str, Any]],
        model: str = "default",
        options: dict[str, Any] | None = None,
        keep_alive: int | None = None,
    ) -> Iterator[str]:
        self.calls.append({"messages": messages, "model": model, "options": options})
        if self.error is not None:
            raise self.error
        return iter(self.tokens)


@pytest.fixture
def mock_settings():
    settings = Mock()
    models = Mock()
    models.llm = "default_chat"
    models.whisper = "large-v3-turbo"
    models.num_predict = 4096
    settings.models = models
    return settings


def _make_generator(settings, client):
    with (
        patch("notes.note_generator.TemplateEngine"),
        patch("notes.note_generator.PopupManager"),
    ):
        return NoteGenerator(settings, llm_client=client)


def test_call_llm_uses_default_model_and_single_user_message(mock_settings):
    client = FakeLLMClient(tokens=["Hello", " ", "world", "  "])
    generator = _make_generator(mock_settings, client)
    prompt = f"{SYSTEM_PROMPT}\n\nTranscript:\nhello there"

    result = generator._call_llm(prompt)

    assert result == "Hello world"  # joined token stream, stripped
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "default"
    assert call["messages"] == [{"role": "user", "content": prompt}]
    assert call["messages"][0]["content"].startswith(SYSTEM_PROMPT)


def test_call_llm_passes_max_tokens_budget(mock_settings):
    client = FakeLLMClient(tokens=["ok"])
    generator = _make_generator(mock_settings, client)

    generator._call_llm("prompt")

    options = client.calls[0]["options"]
    assert options == {"max_tokens": 4096}
    # temperature / top_p are intentionally not sent: the MLX backend splats
    # options into mlx_lm.stream_generate, which only honours max_tokens.
    assert "temperature" not in options
    assert "top_p" not in options


def test_generate_structured_notes_returns_none_on_llm_error(mock_settings):
    client = FakeLLMClient(error=LLMModelError("model fell over"))
    generator = _make_generator(mock_settings, client)

    result = generator._generate_structured_notes(
        "a transcript that is comfortably longer than the fifty character floor",
        "Some Title",
    )

    assert result is None
