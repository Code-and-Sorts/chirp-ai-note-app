import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config.settings import ChirpSettings
from notes_chat.search import LiveSearchSession


@pytest.fixture(autouse=True)
def enable_testing_mode():
    """Automatically enable testing mode for all tests"""
    from notes_chat.search import set_testing_mode

    set_testing_mode(True)

    def no_subprocess(*args, **kwargs):
        """Mock function that prevents any subprocess execution"""
        return None

    with (
        patch("subprocess.run", side_effect=no_subprocess) as mock_run,
        patch(
            "notes_chat.search.subprocess.run", side_effect=no_subprocess
        ) as mock_search_run,
        patch("subprocess.Popen", side_effect=no_subprocess) as mock_popen,
        patch("os.system", side_effect=no_subprocess) as mock_system,
    ):
        yield mock_run, mock_search_run, mock_popen, mock_system

    set_testing_mode(False)


@pytest.fixture
def temp_notes_dir():
    """Create a temporary directory with test notes"""
    with tempfile.TemporaryDirectory() as temp_dir:
        notes_dir = Path(temp_dir) / "notes"
        notes_dir.mkdir()

        note1_content = """# Meeting Notes - September 13, 2025

## Team Standup - Daily Sync - 9:00 AM

**Duration:** 15m
**Participants:** Alice, Bob, Charlie

### Executive Summary
Daily standup discussion about project progress.

## Sprint Planning - Q4 Goals - 2:00 PM

**Duration:** 60m
**Participants:** Full team

### Executive Summary
Planning session for upcoming sprint.
"""

        note2_content = """# Meeting Notes - September 14, 2025

## Client Demo - Product Showcase - 10:00 AM

**Duration:** 30m
**Participants:** Sales team, Client

### Executive Summary
Demonstration of new features to potential client.

## Bug Triage - Critical Issues - 3:00 PM

**Duration:** 45m
**Participants:** Dev team

### Executive Summary
Review and prioritization of critical bugs.
"""

        note3_content = """# Meeting Notes - September 15, 2025

## All Hands - Company Update - 11:00 AM

**Duration:** 45m
**Participants:** Everyone

### Executive Summary
Quarterly company updates and announcements.
"""

        (notes_dir / "meetings_2025_09_13.md").write_text(note1_content)
        (notes_dir / "meetings_2025_09_14.md").write_text(note2_content)
        (notes_dir / "meetings_2025_09_15.md").write_text(note3_content)

        yield notes_dir


@pytest.fixture
def mock_config(temp_notes_dir):
    """Create mock config with temporary notes directory"""
    config = Mock(spec=ChirpSettings)
    config.directories = Mock()
    config.directories.notes = str(temp_notes_dir)
    return config


@pytest.fixture
def search_session(mock_config):
    """Create LiveSearchSession with mock config"""
    return LiveSearchSession(mock_config)


class TestLiveSearchSession:
    def test_initialization(self, mock_config):
        session = LiveSearchSession(mock_config)
        assert session.config == mock_config
        assert session.search_term == ""
        assert session.selected_index == 0
        assert isinstance(session.notes, list)
        assert isinstance(session.filtered_notes, list)

    def test_load_notes(self, search_session):
        """Test that notes are loaded correctly from markdown files"""
        expected_meeting_titles = [
            "Meeting Notes - September 13, 2025",
            "Meeting Notes - September 14, 2025",
            "Meeting Notes - September 15, 2025",
        ]

        loaded_meeting_titles = [
            meeting_title for meeting_title, _, _ in search_session.notes
        ]

        assert len(loaded_meeting_titles) == 3
        for expected_title in expected_meeting_titles:
            assert expected_title in loaded_meeting_titles

    def test_load_notes_with_nonexistent_directory(self):
        """Test handling of non-existent notes directory"""
        config = Mock(spec=ChirpSettings)
        config.directories = Mock()
        config.directories.notes = "/nonexistent/path"
        session = LiveSearchSession(config)

        assert session.notes == []
        assert session.filtered_notes == []

    def test_filter_notes_empty_search(self, search_session):
        """Test filtering with empty search term returns no notes"""
        search_session.search_term = ""
        search_session.filter_notes()

        assert len(search_session.filtered_notes) == 0

    def test_filter_notes_case_insensitive(self, search_session):
        """Test case-insensitive filtering"""
        search_session.search_term = "SEPTEMBER"
        search_session.filter_notes()

        filtered_titles = [
            meeting_title for meeting_title, _, _ in search_session.filtered_notes
        ]
        assert len(filtered_titles) == 3
        assert "Meeting Notes - September 13, 2025" in filtered_titles

    def test_filter_notes_partial_match(self, search_session):
        """Test partial string matching"""
        search_session.search_term = "13"
        search_session.filter_notes()

        filtered_titles = [
            meeting_title for meeting_title, _, _ in search_session.filtered_notes
        ]
        assert len(filtered_titles) == 1
        assert "Meeting Notes - September 13, 2025" in filtered_titles

    def test_filter_notes_multiple_matches(self, search_session):
        """Test filtering with multiple matches"""
        search_session.search_term = "2025"
        search_session.filter_notes()

        filtered_titles = [
            meeting_title for meeting_title, _, _ in search_session.filtered_notes
        ]
        assert len(filtered_titles) == 3

    def test_filter_notes_no_matches(self, search_session):
        """Test filtering with no matches"""
        search_session.search_term = "nonexistent"
        search_session.filter_notes()

        assert len(search_session.filtered_notes) == 0

    def test_filter_notes_resets_selection_index(self, search_session):
        """Test that selection index is reset when filtered results change"""
        search_session.selected_index = len(search_session.notes) - 1

        search_session.search_term = "13"
        search_session.filter_notes()

        assert search_session.selected_index < len(search_session.filtered_notes)

    def test_handle_key_printable_characters(self, search_session):
        """Test handling of printable characters"""
        initial_term = search_session.search_term

        # Add character 'a' (ASCII 97)
        result = search_session.handle_key(97)

        assert result is True
        assert search_session.search_term == initial_term + "a"

    def test_handle_key_backspace(self, search_session):
        """Test backspace handling"""
        search_session.search_term = "test"

        # Backspace (ASCII 127)
        result = search_session.handle_key(127)

        assert result is True
        assert search_session.search_term == "tes"

    def test_handle_key_backspace_empty_term(self, search_session):
        """Test backspace with empty search term"""
        search_session.search_term = ""

        result = search_session.handle_key(127)

        assert result is True
        assert search_session.search_term == ""

    def test_handle_key_escape(self, search_session):
        """Test escape key handling"""
        result = search_session.handle_key(27)  # ESC
        assert result is False

    def test_handle_key_enter(self, search_session):
        """Test enter key handling"""
        result = search_session.handle_key(13)  # Enter
        assert result is False

    def test_handle_key_ctrl_c(self, search_session):
        """Test Ctrl+C handling"""
        result = search_session.handle_key(3)  # Ctrl+C
        assert result is False

    def test_handle_arrow_keys_up(self, search_session):
        """Test up arrow key navigation"""
        search_session.search_term = "2025"
        search_session.filter_notes()
        search_session.selected_index = 2

        result = search_session.handle_arrow_keys("\x1b[A")  # Up arrow

        assert result is True
        assert search_session.selected_index == 1

    def test_handle_arrow_keys_up_at_top(self, search_session):
        """Test up arrow at top of list"""
        search_session.selected_index = 0

        result = search_session.handle_arrow_keys("\x1b[A")  # Up arrow

        assert result is True
        assert search_session.selected_index == 0

    def test_handle_arrow_keys_down(self, search_session):
        """Test down arrow key navigation"""
        search_session.search_term = "2025"
        search_session.filter_notes()
        search_session.selected_index = 0

        result = search_session.handle_arrow_keys("\x1b[B")  # Down arrow

        assert result is True
        assert search_session.selected_index == 1

    def test_handle_arrow_keys_down_at_bottom(self, search_session):
        """Test down arrow at bottom of list"""
        search_session.selected_index = len(search_session.filtered_notes) - 1

        result = search_session.handle_arrow_keys("\x1b[B")  # Down arrow

        assert result is True
        assert search_session.selected_index == len(search_session.filtered_notes) - 1

    def test_handle_arrow_keys_empty_results(self, search_session):
        """Test arrow key handling with empty results"""
        search_session.filtered_notes = []
        search_session.selected_index = 0

        # Should not crash with empty results
        result = search_session.handle_arrow_keys("\x1b[A")  # Up arrow
        assert result is True

        result = search_session.handle_arrow_keys("\x1b[B")  # Down arrow
        assert result is True

    @patch("notes_chat.search.console.print")
    def test_open_selected_note_success(self, mock_print, search_session):
        """Test successfully opening a selected note"""
        # Add a search term to get results
        search_session.search_term = "2025"
        search_session.filter_notes()
        search_session.selected_index = 0

        search_session.open_selected_note()

        mock_print.assert_called()
        call_args = mock_print.call_args[0][0]
        assert "Testing mode: Would open" in call_args

    @patch("notes_chat.search.console.print")
    def test_open_selected_note_failure(self, mock_print, search_session):
        """Test handling of failed note opening"""
        search_session.search_term = "2025"
        search_session.filter_notes()
        search_session.selected_index = 0

        search_session.open_selected_note()

        mock_print.assert_called()
        call_args = mock_print.call_args[0][0]
        assert "Testing mode: Would open" in call_args

    def test_open_selected_note_empty_results(self, search_session):
        """Test opening note with empty filtered results"""
        search_session.filtered_notes = []

        search_session.open_selected_note()

    def test_open_selected_note_invalid_index(self, search_session):
        """Test opening note with invalid selection index"""
        search_session.selected_index = len(search_session.filtered_notes) + 10

        search_session.open_selected_note()

    @patch("notes_chat.search.console.print")
    def test_open_selected_note_general_exception(self, mock_print, search_session):
        """Test handling of general exceptions during note opening"""
        search_session.search_term = "2025"
        search_session.filter_notes()
        search_session.selected_index = 0

        search_session.open_selected_note()

        mock_print.assert_called()
        call_args = mock_print.call_args[0][0]
        assert "Testing mode: Would open" in call_args

    @patch("notes_chat.search.console.print")
    def test_start_with_no_notes(self, mock_print):
        """Test starting search session with no notes"""
        config = Mock(spec=ChirpSettings)
        config.directories = Mock()
        config.directories.notes = "/empty/path"
        session = LiveSearchSession(config)

        session.start()

        mock_print.assert_called()
        call_args = mock_print.call_args[0][0]
        assert "No notes found" in call_args


class TestSearchIntegration:
    @patch("notes_chat.search.LiveSearchSession")
    def test_search_command_success(self, mock_session_class):
        """Test search command execution"""
        from chirp.cli import search

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        search()

        mock_session_class.assert_called_once()
        mock_session.start.assert_called_once()

    @patch("notes_chat.search.LiveSearchSession")
    @patch("chirp.cli.console.print")
    def test_search_command_keyboard_interrupt(self, mock_print, mock_session_class):
        """Test search command handling KeyboardInterrupt"""
        from chirp.cli import search

        mock_session = Mock()
        mock_session.start.side_effect = KeyboardInterrupt()
        mock_session_class.return_value = mock_session

        # Should handle KeyboardInterrupt gracefully
        search()

        mock_print.assert_called()
        call_args = mock_print.call_args[0][0]
        assert "Search cancelled" in call_args

    @patch("notes_chat.search.LiveSearchSession")
    def test_search_command_exception(self, mock_session_class):
        """Test search command handling general exceptions"""
        import typer

        from chirp.cli import search

        mock_session = Mock()
        mock_session.start.side_effect = Exception("Test error")
        mock_session_class.return_value = mock_session

        with pytest.raises(typer.Exit):
            search()


class TestSearchFiltering:
    def test_complex_search_scenarios(self, search_session):
        """Test various complex search scenarios"""
        search_session.search_term = "14"
        search_session.filter_notes()
        filtered_titles = [
            meeting_title for meeting_title, _, _ in search_session.filtered_notes
        ]
        assert any("14" in title for title in filtered_titles)

        search_session.search_term = "September"
        search_session.filter_notes()
        filtered_titles = [
            meeting_title for meeting_title, _, _ in search_session.filtered_notes
        ]
        assert len(filtered_titles) == 3

        search_session.search_term = "meeting-notes"
        search_session.filter_notes()
        filtered_titles = [
            meeting_title for meeting_title, _, _ in search_session.filtered_notes
        ]

    def test_search_term_edge_cases(self, search_session):
        """Test edge cases for search terms"""
        search_session.search_term = "meeting notes"
        search_session.filter_notes()
        filtered_titles = [
            meeting_title for meeting_title, _, _ in search_session.filtered_notes
        ]
        assert len(filtered_titles) == 3

        search_session.search_term = "SePtEmBeR"
        search_session.filter_notes()
        filtered_titles = [
            meeting_title for meeting_title, _, _ in search_session.filtered_notes
        ]
        assert len(filtered_titles) == 3
