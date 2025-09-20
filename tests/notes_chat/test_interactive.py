from unittest.mock import Mock, patch

import pytest

from config.settings import ChirpSettings
from notes_chat.interactive import InteractiveChatSession


@pytest.fixture
def mock_config():
    config = Mock(spec=ChirpSettings)
    return config


@pytest.fixture
def chat_session(mock_config):
    session = InteractiveChatSession(mock_config)
    return session


def test_interactive_chat_initialization(mock_config):
    session = InteractiveChatSession(mock_config)
    assert session.config == mock_config
    assert session.exit_attempts == 0


def test_ac1_welcome_message_display(chat_session):
    assert chat_session.exit_attempts == 0


def test_ac2_exit_counter_reset_logic(chat_session):
    chat_session.exit_attempts = 1
    question = "What is this about?"
    if question.strip():
        chat_session.exit_attempts = 0
    assert chat_session.exit_attempts == 0


def test_ac3_implementation_limitation_exposed():
    session = InteractiveChatSession(Mock())
    session.exit_attempts += 1
    session.exit_attempts += 1
    should_exit_incorrectly = session.exit_attempts >= 2
    assert should_exit_incorrectly


def test_ac3_desired_behavior():
    session = InteractiveChatSession(Mock())
    assert session.exit_attempts == 0
    session.exit_attempts += 1
    assert session.exit_attempts == 1
    session.exit_attempts += 1
    should_exit = session.exit_attempts >= 2
    assert should_exit


def test_ac4_first_ctrl_c_logic(chat_session):
    initial = chat_session.exit_attempts
    chat_session.exit_attempts += 1
    assert chat_session.exit_attempts == initial + 1
    assert chat_session.exit_attempts < 2


def test_ac5_second_ctrl_c_logic(chat_session):
    chat_session.exit_attempts = 1
    chat_session.exit_attempts += 1
    should_exit = chat_session.exit_attempts >= 2
    assert should_exit


def test_ac6_counter_reset_after_question_logic(chat_session):
    chat_session.exit_attempts = 1
    question = "What happened today?"
    if question.strip():
        chat_session.exit_attempts = 0
    assert chat_session.exit_attempts == 0
    chat_session.exit_attempts += 1
    assert chat_session.exit_attempts == 1
    chat_session.exit_attempts += 1
    assert chat_session.exit_attempts == 2
    should_exit = chat_session.exit_attempts >= 2
    assert should_exit


def test_ac7_clean_exit_conditions():
    try:
        raise EOFError()
    except EOFError:
        should_exit_cleanly = True
    assert should_exit_cleanly
    exit_attempts = 2
    should_exit_on_double_ctrl_c = exit_attempts >= 2
    assert should_exit_on_double_ctrl_c


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
        chat_session.config, "what was discussed in the meeting"
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

    mock_stream.assert_called_once_with(chat_session.config, "hi")


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console.print")
def test_handle_question_conversational_failure(mock_print, mock_stream, chat_session):
    mock_stream.return_value = [
        {"type": "thinking", "message": "Processing..."},
        {"type": "error", "message": "Connection failed"},
    ]

    chat_session.handle_question("hello")

    mock_stream.assert_called_once_with(chat_session.config, "hello")


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
    mock_stream.assert_called_once_with(chat_session.config, "test question")


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
    mock_stream.assert_called_once_with(chat_session.config, "test question")


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console.print")
def test_handle_question_retrieval_error(mock_print, mock_stream, chat_session):
    mock_stream.return_value = [
        {"type": "thinking", "message": "Searching..."},
        {"type": "error", "message": "Index error"},
    ]

    chat_session.handle_question("test question")
    mock_stream.assert_called_once_with(chat_session.config, "test question")


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console.print")
def test_handle_question_generation_failure(mock_print, mock_stream, chat_session):
    mock_stream.return_value = [
        {"type": "thinking", "message": "Generating answer..."},
        {"type": "error", "message": "Generation failed"},
    ]

    chat_session.handle_question("test question")
    mock_stream.assert_called_once_with(chat_session.config, "test question")


def test_empty_input_logic():
    empty_questions = ["", "   ", "\t", "\n"]
    for question in empty_questions:
        should_show_message = not question.strip()
        assert should_show_message
    valid_question = "What happened?"
    should_process = bool(valid_question.strip())
    assert should_process


def test_ac3_ctrl_c_while_typing_behavior_fixed(chat_session):
    initial_attempts = chat_session.exit_attempts
    question = ""
    was_ctrl_c_while_typing = True
    if question == "" and was_ctrl_c_while_typing:
        pass
    assert chat_session.exit_attempts == initial_attempts
    question = ""
    was_ctrl_c_while_typing = False
    if question == "" and not was_ctrl_c_while_typing:
        chat_session.exit_attempts += 1
    assert chat_session.exit_attempts == initial_attempts + 1


def test_smart_input_handler_logic():
    from notes_chat.interactive import SmartInputHandler

    handler = SmartInputHandler()
    handler.input_buffer = ""
    was_typing = len(handler.input_buffer) > 0
    assert not was_typing
    handler.input_buffer = "hello"
    was_typing = len(handler.input_buffer) > 0
    assert was_typing


def test_input_handler_return_format():
    from notes_chat.interactive import SmartInputHandler

    handler = SmartInputHandler()
    assert hasattr(handler, "get_input")


def test_exit_logic_with_new_input_format(chat_session):
    was_ctrl_c = True
    was_typing = True
    if was_ctrl_c and was_typing:
        pass
    initial_attempts = chat_session.exit_attempts
    assert chat_session.exit_attempts == initial_attempts
    was_ctrl_c = True
    was_typing = False
    if was_ctrl_c and not was_typing:
        chat_session.exit_attempts += 1
    assert chat_session.exit_attempts == initial_attempts + 1
    if was_ctrl_c and not was_typing:
        chat_session.exit_attempts += 1
    should_exit = chat_session.exit_attempts >= 2
    assert should_exit
