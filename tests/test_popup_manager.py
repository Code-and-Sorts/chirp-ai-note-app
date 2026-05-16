import subprocess
from unittest.mock import MagicMock, patch

import pytest

from utils.popup_manager import PopupManager


@pytest.fixture()
def macos_popup():
    with patch("utils.popup_manager.platform.system", return_value="Darwin"):
        yield PopupManager()


@pytest.fixture()
def linux_popup():
    with patch("utils.popup_manager.platform.system", return_value="Linux"):
        yield PopupManager()


class TestPopupManagerInit:
    def test_is_macos_true_on_darwin(self, macos_popup):
        assert macos_popup.is_macos is True

    def test_is_macos_false_on_linux(self, linux_popup):
        assert linux_popup.is_macos is False


class TestShowRecordingWarning:
    def test_delegates_to_show_notification(self, macos_popup):
        macos_popup.show_notification = MagicMock(return_value=True)
        result = macos_popup.show_recording_warning(30)
        macos_popup.show_notification.assert_called_once()
        title, message = macos_popup.show_notification.call_args[0]
        assert "30 minutes" in message
        assert result is True


class TestShowRecordingComplete:
    def test_includes_filename_and_duration(self, macos_popup):
        macos_popup.show_notification = MagicMock(return_value=True)
        macos_popup.show_recording_complete("meeting.wav", "5:00")
        _, message = macos_popup.show_notification.call_args[0]
        assert "meeting.wav" in message
        assert "5:00" in message


class TestShowTranscriptionComplete:
    def test_singular_file(self, macos_popup):
        macos_popup.show_notification = MagicMock(return_value=True)
        macos_popup.show_transcription_complete(1)
        _, message = macos_popup.show_notification.call_args[0]
        assert "1 audio file" in message
        assert message.endswith("file")

    def test_plural_files(self, macos_popup):
        macos_popup.show_notification = MagicMock(return_value=True)
        macos_popup.show_transcription_complete(3)
        _, message = macos_popup.show_notification.call_args[0]
        assert "3 audio files" in message


class TestShowNotesGenerated:
    def test_includes_filename(self, macos_popup):
        macos_popup.show_notification = MagicMock(return_value=True)
        macos_popup.show_notes_generated("notes.md")
        _, message = macos_popup.show_notification.call_args[0]
        assert "notes.md" in message


class TestShowError:
    def test_passes_error_message_through(self, macos_popup):
        macos_popup.show_notification = MagicMock(return_value=False)
        result = macos_popup.show_error("something went wrong")
        _, message = macos_popup.show_notification.call_args[0]
        assert message == "something went wrong"
        assert result is False


class TestShowNotification:
    def test_routes_to_macos_on_darwin(self, macos_popup):
        macos_popup._show_macos_notification = MagicMock(return_value=True)
        result = macos_popup.show_notification("T", "M")
        macos_popup._show_macos_notification.assert_called_once_with("T", "M")
        assert result is True

    def test_routes_to_generic_on_non_darwin(self, linux_popup):
        linux_popup._show_generic_notification = MagicMock(return_value=True)
        result = linux_popup.show_notification("T", "M")
        linux_popup._show_generic_notification.assert_called_once_with("T", "M")
        assert result is True

    def test_returns_false_on_subprocess_error(self, macos_popup):
        macos_popup._show_macos_notification = MagicMock(
            side_effect=subprocess.SubprocessError
        )
        assert macos_popup.show_notification("T", "M") is False

    def test_returns_false_on_os_error(self, macos_popup):
        macos_popup._show_macos_notification = MagicMock(side_effect=OSError)
        assert macos_popup.show_notification("T", "M") is False


class TestShowMacosNotification:
    def test_returns_true_on_success(self, macos_popup):
        with patch("utils.popup_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            assert macos_popup._show_macos_notification("T", "M") is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "osascript"

    def test_returns_false_on_called_process_error(self, macos_popup):
        with patch(
            "utils.popup_manager.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "osascript"),
        ):
            assert macos_popup._show_macos_notification("T", "M") is False


class TestShowGenericNotification:
    def test_returns_true_on_success(self, linux_popup):
        with patch("utils.popup_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            assert linux_popup._show_generic_notification("T", "M") is True
            args = mock_run.call_args[0][0]
            assert args[0] == "notify-send"

    def test_returns_false_on_called_process_error(self, linux_popup):
        with patch(
            "utils.popup_manager.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "notify-send"),
        ):
            assert linux_popup._show_generic_notification("T", "M") is False

    def test_returns_false_when_notify_send_missing(self, linux_popup):
        with patch(
            "utils.popup_manager.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert linux_popup._show_generic_notification("T", "M") is False


class TestAskYesNo:
    def test_returns_none_on_non_macos(self, linux_popup):
        assert linux_popup.ask_yes_no("Title", "Question?") is None

    def test_returns_true_when_yes_clicked(self, macos_popup):
        mock_result = MagicMock()
        mock_result.stdout = "button returned:Yes"
        with patch("utils.popup_manager.subprocess.run", return_value=mock_result):
            assert macos_popup.ask_yes_no("Title", "Question?") is True

    def test_returns_false_when_no_clicked(self, macos_popup):
        mock_result = MagicMock()
        mock_result.stdout = "button returned:No"
        with patch("utils.popup_manager.subprocess.run", return_value=mock_result):
            assert macos_popup.ask_yes_no("Title", "Question?") is False

    def test_returns_none_on_called_process_error(self, macos_popup):
        with patch(
            "utils.popup_manager.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "osascript"),
        ):
            assert macos_popup.ask_yes_no("Title", "Question?") is None

    def test_osascript_receives_both_buttons(self, macos_popup):
        mock_result = MagicMock()
        mock_result.stdout = "button returned:Yes"
        with patch(
            "utils.popup_manager.subprocess.run", return_value=mock_result
        ) as mock_run:
            macos_popup.ask_yes_no("Title", "Question?")
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "osascript"
