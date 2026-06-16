import platform
import subprocess


def _escape_applescript(value: str) -> str:
    """Escape a string for safe interpolation inside an AppleScript "..." literal.

    Backslashes and double quotes are the two characters that break (or could be
    used to inject into) a quoted AppleScript string, so both are backslash-
    escaped. Backslash must be escaped first to avoid double-escaping.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


class PopupManager:
    def __init__(self):
        self.is_macos = platform.system() == "Darwin"

    def show_recording_warning(self, duration_minutes: int) -> bool:
        title = "Chirp - Recording Warning"
        message = f"You've been recording for {duration_minutes} minutes. Don't forget to stop when your meeting ends!"
        return self.show_notification(title, message)

    def show_recording_complete(self, filename: str, duration: str) -> bool:
        title = "Chirp - Recording Complete"
        message = f"Recording saved as {filename}\nDuration: {duration}"
        return self.show_notification(title, message)

    def show_transcription_complete(self, count: int) -> bool:
        title = "Chirp - Transcription Complete"
        message = (
            f"Successfully transcribed {count} audio file{'s' if count != 1 else ''}"
        )
        return self.show_notification(title, message)

    def show_notes_generated(self, filename: str) -> bool:
        title = "Chirp - Notes Generated"
        message = f"Meeting notes saved to {filename}"
        return self.show_notification(title, message)

    def show_error(self, error_message: str) -> bool:
        title = "Chirp - Error"
        return self.show_notification(title, error_message)

    def show_notification(self, title: str, message: str) -> bool:
        try:
            if self.is_macos:
                return self._show_macos_notification(title, message)
            return self._show_generic_notification(title, message)
        except (subprocess.SubprocessError, OSError):
            return False

    def _show_macos_notification(self, title: str, message: str) -> bool:
        safe_message = _escape_applescript(message)
        safe_title = _escape_applescript(title)
        applescript = f"""
        display notification "{safe_message}" with title "{safe_title}"
        """
        try:
            subprocess.run(
                ["osascript", "-e", applescript], check=True, capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _show_generic_notification(self, title: str, message: str) -> bool:
        try:
            subprocess.run(
                ["notify-send", title, message], check=True, capture_output=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def ask_yes_no(self, title: str, question: str) -> bool | None:
        if not self.is_macos:
            return None

        safe_question = _escape_applescript(question)
        safe_title = _escape_applescript(title)
        applescript = f"""
        display dialog "{safe_question}" with title "{safe_title}" buttons {{"No", "Yes"}} default button "Yes"
        """
        try:
            result = subprocess.run(
                ["osascript", "-e", applescript],
                capture_output=True,
                text=True,
                check=True,
            )
            return "Yes" in result.stdout
        except subprocess.CalledProcessError:
            return None
