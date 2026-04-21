import sys
import termios
import tty
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from config.settings import ChirpSettings
from notes.note_editor import ManualNoteEditor

console = Console()

_TESTING_MODE = False


def set_testing_mode(enabled: bool):
    """Enable/disable testing mode to prevent subprocess calls"""
    global _TESTING_MODE
    _TESTING_MODE = enabled


class LiveSearchSession:
    def __init__(self, config: ChirpSettings):
        self.config = config
        self.notes: list[tuple[str, str, Path]] = []
        self.filtered_notes: list[tuple[str, str, Path]] = []
        self.selected_index = 0
        self.search_term = ""
        self.live: Live | None = None
        self.pending_note_path: Path | None = None
        self.load_notes()

    def load_notes(self) -> None:
        """Load meeting names from the notes directory"""
        notes_root = Path(self.config.directories.notes_root)
        if not notes_root.exists():
            return

        for note_file in notes_root.glob("*/notes.md"):
            try:
                with open(note_file, encoding="utf-8") as f:
                    lines = f.readlines()

                slug = note_file.parent.name
                for line in lines:
                    line = line.strip()
                    if line.startswith("# "):
                        meeting_title = line[2:].strip()
                        self.notes.append((meeting_title, slug, note_file))
                        break

            except Exception:
                continue

        self.filtered_notes = []

    def filter_notes(self) -> None:
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

    def create_display(self) -> Group:
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

    def update_display(self) -> None:
        """Update the live display"""
        if self.live:
            self.live.update(self.create_display())

    def open_selected_note(self) -> None:
        """Open the selected note file"""
        if not self.filtered_notes or self.selected_index >= len(self.filtered_notes):
            return

        _, _, note_path = self.filtered_notes[self.selected_index]
        self.pending_note_path = note_path

        if _TESTING_MODE:
            console.print(f"\n[dim]Testing mode: Would open {note_path}[/dim]")

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

    def start(self) -> None:
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

        note_to_open: Path | None = None
        try:
            tty.setcbreak(sys.stdin.fileno())

            self.live = Live(
                self.create_display(), console=console, refresh_per_second=10
            )
            self.live.start()

            while True:
                char = sys.stdin.read(1)
                key_code = ord(char)

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

            note_to_open = self.pending_note_path

        except KeyboardInterrupt:
            pass
        finally:
            if self.live:
                self.live.stop()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
            console.clear()

        self.pending_note_path = None

        if note_to_open and not _TESTING_MODE:
            self._open_note_in_editor(note_to_open)

        console.print("\n[dim]Search ended[/dim]")

    def _open_note_in_editor(self, note_path: Path) -> None:
        try:
            original_content = note_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            console.print(f"[yellow]Note not found: {note_path}[/yellow]")
            return
        except Exception as exc:
            console.print(f"[red]Failed to read note: {exc}[/red]")
            return

        metadata, body = self._parse_front_matter(original_content)
        readonly = self._is_readonly(metadata, note_path)
        display_content = body if metadata else original_content
        title = self._extract_title(display_content, note_path)

        editor = ManualNoteEditor(title, display_content, readonly=readonly)

        try:
            result = editor.run()
        except KeyboardInterrupt:
            console.print("\n[dim]Editor cancelled[/dim]")
            return
        except Exception as exc:
            console.print(f"[red]❌ Editor error: {exc}[/red]")
            return

        if readonly or not result.saved:
            if readonly:
                console.print("[dim]Closed read-only note without changes.[/dim]")
            else:
                console.print("[dim]No changes saved.[/dim]")
            return

        updated_content = result.content
        if metadata:
            updated_content = self._apply_front_matter(updated_content, metadata)

        try:
            note_path.write_text(updated_content, encoding="utf-8")
            console.print(f"[green]✅ Saved changes to {note_path.name}[/green]")
        except Exception as exc:
            console.print(f"[red]❌ Failed to write note: {exc}[/red]")

    def _parse_front_matter(self, content: str) -> tuple[list[tuple[str, str]], str]:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return [], content

        metadata_lines: list[str] = []
        closing_index = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                closing_index = idx
                break
            metadata_lines.append(lines[idx])

        if closing_index is None:
            return [], content

        body_lines = lines[closing_index + 1 :]
        body = "\n".join(body_lines)
        if content.endswith("\n"):
            body += "\n"

        metadata: list[tuple[str, str]] = []
        for line in metadata_lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata.append((key.strip(), value.strip()))

        return metadata, body

    def _apply_front_matter(self, body: str, metadata: list[tuple[str, str]]) -> str:
        if not metadata:
            return body

        header_lines = ["---"]
        header_lines.extend(f"{key}: {value}" for key, value in metadata)
        header_lines.append("---")

        stripped_body = body.lstrip("\n")
        content = "\n".join(header_lines)
        if stripped_body:
            content = f"{content}\n\n{stripped_body}"
        else:
            content = f"{content}\n"

        if not content.endswith("\n"):
            content += "\n"

        return content

    def _is_readonly(self, metadata: list[tuple[str, str]], note_path: Path) -> bool:
        if not metadata:
            return False

        normalized = {key.lower(): value for key, value in metadata}
        readonly_value = normalized.get("readonly")
        if readonly_value and readonly_value.lower() in {"true", "1", "yes"}:
            return True

        source = normalized.get("chirp_source")
        if source and source.lower() == "generated":
            return True

        return False

    def _extract_title(self, content: str, note_path: Path) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip() or note_path.stem
        return note_path.stem
