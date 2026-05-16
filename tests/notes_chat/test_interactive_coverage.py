"""Additional tests to raise notes_chat/interactive.py coverage to >= 80%."""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from notes_chat.interactive import InteractiveChatSession


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.directories.notes_root.exists.return_value = False
    config.models.llm = "test-model"
    return config


@pytest.fixture
def chat_session(mock_config):
    return InteractiveChatSession(mock_config)


@pytest.fixture
def plaintext_session(mock_config):
    return InteractiveChatSession(mock_config, markdown=False)


# ---------------------------------------------------------------------------
# _on_user_activity
# ---------------------------------------------------------------------------


def test_on_user_activity_clears_hint_when_interrupt_set(chat_session):
    chat_session.last_interrupt_time = time.time()
    chat_session._on_user_activity(None)
    assert chat_session.last_interrupt_time is None


def test_on_user_activity_noop_when_no_interrupt(chat_session):
    chat_session._on_user_activity(None)
    assert chat_session.last_interrupt_time is None


# ---------------------------------------------------------------------------
# _hide_hint
# ---------------------------------------------------------------------------


def test_hide_hint_cancels_existing_timer(chat_session):
    timer = Mock()
    chat_session._hint_timer = timer
    chat_session.last_interrupt_time = time.time()
    chat_session._hide_hint()
    timer.cancel.assert_called_once()
    assert chat_session._hint_timer is None


def test_hide_hint_tolerates_invalidate_runtime_error(chat_session):
    app = Mock()
    app.invalidate.side_effect = RuntimeError("torn down")
    chat_session._session.app = app
    chat_session.last_interrupt_time = time.time()
    chat_session._hide_hint()
    assert chat_session.last_interrupt_time is None


def test_hide_hint_tolerates_invalidate_attribute_error(chat_session):
    del chat_session._session.app
    chat_session.last_interrupt_time = time.time()
    chat_session._hide_hint()
    assert chat_session.last_interrupt_time is None


# ---------------------------------------------------------------------------
# _show_hint_then_auto_clear
# ---------------------------------------------------------------------------


def test_show_hint_starts_timer(chat_session):
    with patch("notes_chat.interactive.threading.Timer") as mock_timer_cls:
        fake_timer = Mock()
        mock_timer_cls.return_value = fake_timer
        chat_session._show_hint_then_auto_clear()
        mock_timer_cls.assert_called_once()
        assert fake_timer.daemon is True
        fake_timer.start.assert_called_once()


def test_show_hint_cancels_previous_timer(chat_session):
    old_timer = Mock()
    chat_session._hint_timer = old_timer
    with patch("notes_chat.interactive.threading.Timer") as mock_timer_cls:
        mock_timer_cls.return_value = Mock()
        chat_session._show_hint_then_auto_clear()
    old_timer.cancel.assert_called_once()


def test_show_hint_tolerates_invalidate_runtime_error(chat_session):
    app = Mock()
    app.invalidate.side_effect = RuntimeError
    chat_session._session.app = app
    with patch("notes_chat.interactive.threading.Timer") as mock_timer_cls:
        mock_timer_cls.return_value = Mock()
        chat_session._show_hint_then_auto_clear()


def test_show_hint_tolerates_invalidate_attribute_error(chat_session):
    del chat_session._session.app
    with patch("notes_chat.interactive.threading.Timer") as mock_timer_cls:
        mock_timer_cls.return_value = Mock()
        chat_session._show_hint_then_auto_clear()


# ---------------------------------------------------------------------------
# _count_notes
# ---------------------------------------------------------------------------


def test_count_notes_returns_zero_when_dir_missing():
    config = MagicMock()
    config.directories.notes_root.exists.return_value = False
    session = InteractiveChatSession(config)
    assert session._count_notes() == 0


def test_count_notes_counts_notes_md_files(tmp_path):
    for i in range(3):
        note_dir = tmp_path / f"note-{i}"
        note_dir.mkdir()
        (note_dir / "notes.md").touch()
    config = MagicMock()
    config.directories.notes_root = tmp_path
    session = InteractiveChatSession(config)
    assert session._count_notes() == 3


def test_count_notes_returns_zero_on_oserror():
    config = MagicMock()
    config.directories.notes_root.exists.side_effect = OSError("permission denied")
    session = InteractiveChatSession(config)
    assert session._count_notes() == 0


# ---------------------------------------------------------------------------
# _handle_slash
# ---------------------------------------------------------------------------


@patch("notes_chat.interactive.console")
def test_handle_slash_exit(mock_console, chat_session):
    assert chat_session._handle_slash("/exit") is False
    mock_console.print.assert_called_with("[dim]bye![/dim]")


@patch("notes_chat.interactive.console")
def test_handle_slash_quit(mock_console, chat_session):
    assert chat_session._handle_slash("/quit") is False


@patch("notes_chat.interactive.console")
def test_handle_slash_q(mock_console, chat_session):
    assert chat_session._handle_slash("/q") is False


@patch("notes_chat.interactive.console")
def test_handle_slash_help(mock_console, chat_session):
    result = chat_session._handle_slash("/help")
    assert result is True
    assert mock_console.print.call_count >= 4


@patch("notes_chat.interactive.console")
def test_handle_slash_clear(mock_console, chat_session):
    result = chat_session._handle_slash("/clear")
    assert result is True
    mock_console.clear.assert_called_once()


@patch("notes_chat.interactive.console")
def test_handle_slash_unknown(mock_console, chat_session):
    result = chat_session._handle_slash("/unknown")
    assert result is True
    printed = mock_console.print.call_args[0][0]
    assert "unknown command" in printed


# ---------------------------------------------------------------------------
# start() loop branches
# ---------------------------------------------------------------------------


def _make_prompt_side_effects(*values):
    """Return a side_effect list for PromptSession.prompt that raises StopIteration
    after returning all values so the while-True loop exits cleanly via EOFError."""
    return list(values) + [EOFError()]


@patch("notes_chat.interactive.console")
def test_start_quits_on_quit_sentinel(mock_console, mock_config):
    session = InteractiveChatSession(mock_config)
    session._session = Mock()
    session._session.prompt.side_effect = [session._QUIT]

    with patch("sys.stdout"):
        session.start()

    printed_args = [str(c) for c in mock_console.print.call_args_list]
    assert any("bye" in a for a in printed_args)


@patch("notes_chat.interactive.console")
def test_start_skips_blank_input(mock_console, mock_config):
    session = InteractiveChatSession(mock_config)
    session._session = Mock()
    session._session.prompt.side_effect = ["", "   ", EOFError()]

    with patch("sys.stdout"):
        session.start()


@patch("notes_chat.interactive.console")
def test_start_handles_slash_command_exit(mock_console, mock_config):
    session = InteractiveChatSession(mock_config)
    session._session = Mock()
    session._session.prompt.side_effect = ["/exit"]

    with patch("sys.stdout"):
        session.start()


@patch("notes_chat.interactive.console")
def test_start_handles_slash_command_continue(mock_console, mock_config):
    session = InteractiveChatSession(mock_config)
    session._session = Mock()
    session._session.prompt.side_effect = ["/help", EOFError()]

    with patch("sys.stdout"):
        session.start()


@patch("notes_chat.interactive.console")
def test_start_dispatches_plain_question(mock_console, mock_config):
    session = InteractiveChatSession(mock_config)
    session._session = Mock()
    session._session.prompt.side_effect = ["hello chirp", EOFError()]

    with patch.object(session, "handle_question") as mock_hq, patch("sys.stdout"):
        session.start()

    mock_hq.assert_called_once_with("hello chirp")


@patch("notes_chat.interactive.console")
def test_start_handles_eoferror_with_oserror_on_write(mock_console, mock_config):
    session = InteractiveChatSession(mock_config)
    session._session = Mock()
    session._session.prompt.side_effect = [EOFError()]

    with patch("sys.stdout") as mock_stdout:
        mock_stdout.write.side_effect = OSError("broken pipe")
        session.start()

    printed_args = [str(c) for c in mock_console.print.call_args_list]
    assert any("Goodbye" in a for a in printed_args)


@patch("notes_chat.interactive.console")
def test_start_handles_keyboard_interrupt(mock_console, mock_config):
    session = InteractiveChatSession(mock_config)
    session._session = Mock()
    session._session.prompt.side_effect = [KeyboardInterrupt("oops"), EOFError()]

    with patch("sys.stdout"):
        session.start()

    printed_args = [str(c) for c in mock_console.print.call_args_list]
    assert any("Error" in a for a in printed_args)


@patch("notes_chat.interactive.console")
def test_start_handles_runtime_error(mock_console, mock_config):
    session = InteractiveChatSession(mock_config)
    session._session = Mock()
    session._session.prompt.side_effect = [RuntimeError("boom"), EOFError()]

    with patch("sys.stdout"):
        session.start()


# ---------------------------------------------------------------------------
# handle_question — plaintext (markdown=False) path
# ---------------------------------------------------------------------------


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console")
def test_handle_question_plaintext_mode(mock_console, mock_stream, plaintext_session):
    mock_stream.return_value = [
        {"type": "token", "content": "hello"},
        {"type": "complete", "answer": "hello"},
    ]
    plaintext_session.handle_question("hi")
    mock_stream.assert_called_once_with(plaintext_session.config, "hi")


# ---------------------------------------------------------------------------
# handle_question — error event when live is active
# ---------------------------------------------------------------------------


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console")
def test_handle_question_error_with_active_live(
    mock_console, mock_stream, chat_session
):
    mock_stream.return_value = [
        {"type": "token", "content": "partial"},
        {"type": "error", "message": "crash"},
    ]
    chat_session.handle_question("test")


# ---------------------------------------------------------------------------
# handle_question — complete event with no answer (empty break)
# ---------------------------------------------------------------------------


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console")
def test_handle_question_complete_with_no_answer(
    mock_console, mock_stream, chat_session
):
    mock_stream.return_value = [
        {"type": "complete", "answer": "", "sources": None, "from_cache": False},
    ]
    chat_session.handle_question("test")


# ---------------------------------------------------------------------------
# handle_question — complete event with live already None (direct live start)
# ---------------------------------------------------------------------------


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console")
def test_handle_question_complete_starts_live_when_none(
    mock_console, mock_stream, chat_session
):
    mock_stream.return_value = [
        {"type": "complete", "answer": "fresh answer", "sources": None},
    ]
    chat_session.handle_question("test")


# ---------------------------------------------------------------------------
# handle_question — exception during streaming (generic except)
# ---------------------------------------------------------------------------


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console")
def test_handle_question_stream_raises_exception(
    mock_console, mock_stream, chat_session
):
    mock_stream.side_effect = RuntimeError("unexpected failure")
    chat_session.handle_question("test")
    printed = [str(c) for c in mock_console.print.call_args_list]
    assert any("Query failed" in a for a in printed)


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console")
def test_handle_question_exception_with_active_live(
    mock_console, mock_stream, chat_session
):
    def _stream(*_):
        yield {"type": "token", "content": "partial"}
        raise RuntimeError("mid-stream failure")

    mock_stream.side_effect = _stream
    chat_session.handle_question("test")
    printed = [str(c) for c in mock_console.print.call_args_list]
    assert any("Query failed" in a for a in printed)


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console")
def test_handle_question_progress_active_when_stream_ends(
    mock_console, mock_stream, chat_session
):
    """Covers the progress.stop() after the for loop (line 261)."""
    mock_stream.return_value = [
        {"type": "thinking", "message": "Thinking..."},
    ]
    chat_session.handle_question("test")


@patch("notes_chat.interactive.enhanced_search_and_answer_stream")
@patch("notes_chat.interactive.console")
def test_handle_question_exception_with_active_progress(
    mock_console, mock_stream, chat_session
):
    """Covers progress.stop() in the except block (line 275) and finally (line 281)."""

    def _stream(*_):
        yield {"type": "thinking", "message": "Working..."}
        raise RuntimeError("progress was active when exception hit")

    mock_stream.side_effect = _stream
    chat_session.handle_question("test")
    printed = [str(c) for c in mock_console.print.call_args_list]
    assert any("Query failed" in a for a in printed)
