import time
from unittest.mock import Mock, patch

import pytest

from config.settings import ChirpSettings
from notes_chat.interactive import InteractiveChatSession


@pytest.fixture
def mock_config():
    return Mock(spec=ChirpSettings)


@pytest.fixture
def chat_session(mock_config):
    return InteractiveChatSession(mock_config)


def test_interactive_chat_initialization(mock_config):
    session = InteractiveChatSession(mock_config)
    assert session.config == mock_config
    assert session.last_interrupt_time is None
    assert session.interrupt_timeout == 2.0


def test_interrupt_timer_starts_unset(chat_session):
    assert chat_session.last_interrupt_time is None


def test_interrupt_timer_can_be_set(chat_session):
    interrupt_time = time.time()
    chat_session.last_interrupt_time = interrupt_time
    assert chat_session.last_interrupt_time == interrupt_time


def test_interrupt_timer_within_timeout_window(chat_session):
    chat_session.last_interrupt_time = time.time()
    time.sleep(0.1)
    time_diff = time.time() - chat_session.last_interrupt_time
    assert time_diff < chat_session.interrupt_timeout


def test_interrupt_timer_can_be_reset(chat_session):
    chat_session.last_interrupt_time = time.time()
    chat_session.last_interrupt_time = None
    assert chat_session.last_interrupt_time is None


def test_hide_hint_clears_interrupt_timer(chat_session):
    chat_session.last_interrupt_time = time.time()
    chat_session._hide_hint()
    assert chat_session.last_interrupt_time is None


def test_toolbar_shows_nothing_when_no_interrupt(chat_session):
    result = chat_session._toolbar()
    assert result == ""


def test_toolbar_shows_hint_within_timeout(chat_session):
    chat_session.last_interrupt_time = time.time()
    result = chat_session._toolbar()
    assert "Ctrl+C" in result


def test_toolbar_shows_nothing_after_timeout(chat_session):
    chat_session.last_interrupt_time = time.time() - 3.0
    result = chat_session._toolbar()
    assert result == ""


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console.print")
def test_handle_question_search_success(mock_print, mock_stream, chat_session):
    mock_stream.return_value = [
        {"type": "thinking", "message": "Searching..."},
        {"type": "token", "content": "test"},
        {"type": "token", "content": " answer"},
        {
            "type": "complete",
            "answer": "test answer",
            "sources": ["source1", "source2"],
        },
    ]

    chat_session.handle_question("what was discussed in the meeting")

    mock_stream.assert_called_once_with(
        chat_session.config, "what was discussed in the meeting", tags=None
    )


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console.print")
def test_handle_question_conversational_success(mock_print, mock_stream, chat_session):
    mock_stream.return_value = [
        {"type": "thinking", "message": "Having a chat..."},
        {"type": "token", "content": "Hello!"},
        {"type": "token", "content": " I'm Chirp"},
        {"type": "complete", "answer": "Hello! I'm Chirp"},
    ]

    chat_session.handle_question("hi")

    mock_stream.assert_called_once_with(chat_session.config, "hi", tags=None)


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console.print")
def test_handle_question_error_handling(mock_print, mock_stream, chat_session):
    mock_stream.return_value = [
        {"type": "thinking", "message": "Processing..."},
        {"type": "error", "message": "Connection failed"},
    ]

    chat_session.handle_question("hello")

    mock_stream.assert_called_once_with(chat_session.config, "hello", tags=None)


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console.print")
def test_handle_question_no_documents_found(mock_print, mock_stream, chat_session):
    mock_stream.return_value = [
        {"type": "thinking", "message": "Searching..."},
        {
            "type": "thinking",
            "message": "No results found, generating helpful response...",
        },
        {"type": "token", "content": "I couldn't find"},
        {"type": "token", "content": " any relevant information"},
        {"type": "complete", "answer": "I couldn't find any relevant information"},
    ]

    chat_session.handle_question("test question")
    mock_stream.assert_called_once_with(chat_session.config, "test question", tags=None)


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console.print")
def test_handle_question_cached_answer(mock_print, mock_stream, chat_session):
    mock_stream.return_value = [
        {"type": "thinking", "message": "Searching..."},
        {
            "type": "complete",
            "answer": "cached answer",
            "sources": ["source1"],
            "from_cache": True,
        },
    ]

    chat_session.handle_question("test question")
    mock_stream.assert_called_once_with(chat_session.config, "test question", tags=None)
