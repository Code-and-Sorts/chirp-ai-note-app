import os
import subprocess
import sys
import termios
import tty
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from config.settings import ChirpSettings

console = Console()

_TESTING_MODE = False


def set_testing_mode(enabled: bool):
    """Enable/disable testing mode to prevent subprocess calls"""
    global _TESTING_MODE
    _TESTING_MODE = enabled


class LiveSearchSession:
    def __init__(self, config: ChirpSettings):
        self.config = config
        self.notes: list[tuple[str, Path]] = []
        self.filtered_notes: list[tuple[str, Path]] = []
        self.selected_index = 0
        self.search_term = ""
        self.live = None
        self.load_notes()

    def load_notes(self):
        """Load meeting names from the notes directory"""
        notes_dir = Path(self.config.directories.notes)
        if not notes_dir.exists():
            return

        for note_file in notes_dir.glob("*.md"):
            try:
                with open(note_file, encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if line.startswith("# "):
                        meeting_title = line[2:].strip()
                        self.notes.append((meeting_title, note_file.name, note_file))
                        break

            except Exception:
                continue

        self.filtered_notes = []

    def filter_notes(self):
        """Filter notes based on current search term"""
        import shutil

        if not self.search_term:
            self.filtered_notes = []
        else:
            self.filtered_notes = [
                (meeting_title, filename, full_path)
                for meeting_title, filename, full_path in self.notes
                if self.search_term.lower() in meeting_title.lower()
                or self.search_term.lower() in filename.lower()
            ]

        terminal_height = shutil.get_terminal_size().lines
        max_results = min(10, terminal_height - 4)
        max_visible_index = min(len(self.filtered_notes), max_results) - 1

        if self.selected_index > max_visible_index:
            self.selected_index = max(0, max_visible_index)

    def create_display(self):
        """Create the display content for Rich Live"""
        import shutil

        terminal_height = shutil.get_terminal_size().lines
        max_results = min(8, terminal_height - 6)

        search_text = Text()
        search_text.append("Search: ", style="bold blue")
        search_text.append(self.search_term)
        search_text.append("█", style="bold")

        header_text = Text()
        header_text.append("File Name".ljust(30), style="bold blue")
        header_text.append("Meeting Name", style="bold blue")

        results_text = Text()
        if self.filtered_notes:
            visible_notes = self.filtered_notes[:max_results]
            last_index = len(visible_notes) - 1

            for i, (meeting_title, filename, _full_path) in enumerate(visible_notes):
                line = Text()

                if i == self.selected_index:
                    line.append(f"{filename}".ljust(30), style="bold cyan on magenta")
                    line.append(meeting_title, style="bold white on magenta")
                else:
                    line.append(f"{filename}".ljust(30), style="cyan")
                    line.append(meeting_title)

                results_text.append(line)

                if i < last_index:
                    results_text.append("\n")

        display_items = [search_text, Text(""), header_text, results_text]
        return Group(*display_items)

    def update_display(self):
        """Update the live display"""
        if self.live:
            self.live.update(self.create_display())

    def open_selected_note(self):
        """Open the selected note file"""
        if not self.filtered_notes or self.selected_index >= len(self.filtered_notes):
            return

        _, _, note_path = self.filtered_notes[self.selected_index]

        if _TESTING_MODE:
            console.print(f"\n[dim]Testing mode: Would open {note_path}[/dim]")
            return

        try:
            if os.name == "posix":  # macOS/Linux
                subprocess.run(["open", str(note_path)], check=True)
            elif os.name == "nt":  # Windows
                subprocess.run(["start", str(note_path)], shell=True, check=True)
        except subprocess.CalledProcessError:
            console.print(f"\n[green]Note location:[/green] {note_path}")
        except Exception as e:
            console.print(f"\n[yellow]Could not open file: {e}[/yellow]")
            console.print(f"[green]Note location:[/green] {note_path}")

    def handle_key(self, key_code: int) -> bool:
        """Handle keyboard input. Returns True to continue, False to exit"""

        if key_code == 27:  # Esc
            return False
        elif key_code == 13 or key_code == 10:  # Enter
            self.open_selected_note()
            return False
        elif key_code == 127 or key_code == 8:  # Backspace
            if self.search_term:
                self.search_term = self.search_term[:-1]
                self.filter_notes()
        elif key_code == 3:  # Ctrl+C
            return False
        elif 32 <= key_code <= 126:  # Printable characters
            self.search_term += chr(key_code)
            self.filter_notes()

        return True

    def handle_arrow_keys(self, sequence: str) -> bool:
        """Handle arrow key sequences"""
        import shutil

        terminal_height = shutil.get_terminal_size().lines
        max_results = min(10, terminal_height - 4)
        max_visible_index = min(len(self.filtered_notes), max_results) - 1

        if sequence == "\x1b[A":  # Up arrow
            if self.filtered_notes and self.selected_index > 0:
                self.selected_index -= 1
        elif sequence == "\x1b[B":  # Down arrow
            if self.filtered_notes and self.selected_index < max_visible_index:
                self.selected_index += 1

        return True

    def start(self):
        """Start the interactive search session"""
        if not self.notes:
            console.print(
                "[yellow]No notes found. Generate some notes first with 'chirp generate-notes'[/yellow]"
            )
            return

        if not sys.stdin.isatty() or not sys.stdout.isatty():
            console.print(
                "[yellow]Interactive search requires a terminal. Please run from a terminal.[/yellow]"
            )
            return

        try:
            original_settings = termios.tcgetattr(sys.stdin)
        except termios.error as e:
            console.print(
                f"[yellow]Terminal error: {e}. Please run from an interactive terminal.[/yellow]"
            )
            return

        try:
            tty.setcbreak(sys.stdin.fileno())

            self.live = Live(
                self.create_display(), console=console, refresh_per_second=10
            )
            self.live.start()

            while True:
                char = sys.stdin.read(1)
                key_code = ord(char)

                # Handle escape sequences (arrow keys)
                if key_code == 27:  # Esc sequence start
                    try:
                        seq = char + sys.stdin.read(2)
                        if len(seq) == 3 and seq[1] == "[":
                            if not self.handle_arrow_keys(seq):
                                break
                        else:
                            break
                    except:
                        break
                else:
                    if not self.handle_key(key_code):
                        break

                self.update_display()

        except KeyboardInterrupt:
            pass
        finally:
            if self.live:
                self.live.stop()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
            console.clear()
            console.print("\n[dim]Search ended[/dim]")
