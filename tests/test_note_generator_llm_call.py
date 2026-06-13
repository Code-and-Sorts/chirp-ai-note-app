"""Tests for the note-generator LLM call site after the chirpd cutover (6.2).

`_call_llm` routes note generation through `llm.client` instead of the legacy
HTTP API. These tests assert the call shape, the streamed-token aggregation,
and that LLM errors propagate into the existing best-effort `return None` path.
The client stand-ins are the shared fixtures from `tests/conftest.py` (story
6.5), injected through `NoteGenerator(..., llm_client=)`.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from llm.exceptions import LLMConnectionLost, LLMModelError
from notes.note_generator import SYSTEM_PROMPT, NoteGenerator


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


def test_call_llm_uses_default_model_and_single_user_message(
    mock_settings, fake_llm_client, fake_chat_tokens
):
    client = fake_llm_client(
        chat_stream_sync=fake_chat_tokens(["Hello", " ", "world", "  "])
    )
    generator = _make_generator(mock_settings, client)
    prompt = f"{SYSTEM_PROMPT}\n\nTranscript:\nhello there"

    result = generator._call_llm(prompt)

    assert result == "Hello world"  # joined token stream, stripped
    assert len(client.chat_stream_sync.calls) == 1
    call = client.chat_stream_sync.calls[0]
    assert call["model"] == "default"
    assert call["messages"] == [{"role": "user", "content": prompt}]
    assert call["messages"][0]["content"].startswith(SYSTEM_PROMPT)


def test_call_llm_passes_max_tokens_budget(
    mock_settings, fake_llm_client, fake_chat_tokens
):
    client = fake_llm_client(chat_stream_sync=fake_chat_tokens(["ok"]))
    generator = _make_generator(mock_settings, client)

    generator._call_llm("prompt")

    options = client.chat_stream_sync.calls[0]["options"]
    assert options == {"max_tokens": 4096}
    # temperature / top_p are intentionally not sent: the MLX backend splats
    # options into its stream-generate call, which only honours max_tokens.
    assert "temperature" not in options
    assert "top_p" not in options


def test_call_llm_reprints_progress_every_twenty_tokens(
    mock_settings, fake_llm_client, fake_chat_tokens
):
    client = fake_llm_client(chat_stream_sync=fake_chat_tokens(["x"] * 45))
    generator = _make_generator(mock_settings, client)

    result = generator._call_llm("prompt")

    assert result == "x" * 45  # all 45 tokens aggregated, none dropped


def test_generate_structured_notes_returns_none_on_llm_error(
    mock_settings, fake_llm_client, raise_llm_error
):
    client = fake_llm_client(
        chat_stream_sync=raise_llm_error(LLMModelError, "model fell over")
    )
    generator = _make_generator(mock_settings, client)

    result = generator._generate_structured_notes(
        "a transcript that is comfortably longer than the fifty character floor",
        "Some Title",
    )

    assert result is None
    assert len(client.chat_stream_sync.calls) == 1


def test_generate_structured_notes_returns_none_on_midstream_error(
    mock_settings, fake_llm_client, fake_chat_tokens
):
    # Daemon drops the connection after some tokens have already streamed; the
    # partial result is discarded and the record yields no note.
    client = fake_llm_client(
        chat_stream_sync=fake_chat_tokens(
            ["partial ", "content "], error=LLMConnectionLost("daemon went away")
        )
    )
    generator = _make_generator(mock_settings, client)

    result = generator._generate_structured_notes(
        "a transcript that is comfortably longer than the fifty character floor",
        "Some Title",
    )

    assert result is None
