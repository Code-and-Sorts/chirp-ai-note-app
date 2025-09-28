import signal
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import ChirpSettings
from notes_chat.prompting import enhanced_search_and_answer_stream

console = Console()


class SmartInputHandler:
    """
    Input handler that can distinguish between Ctrl+C while typing vs on empty prompt.

    This works by tracking input state and using signal handling to detect interrupts.
    """

    def __init__(self):
        self.input_buffer = ""
        self.input_started = False
        self.ctrl_c_received = False
        self.original_handler = None

    def _signal_handler(self, signum, frame):
        """Handle SIGINT (Ctrl+C) signal"""
        self.ctrl_c_received = True

    def get_input(self, prompt_text: str) -> tuple[str, bool, bool]:
        """
        Get input with Ctrl+C context detection.

        Returns:
            tuple: (input_text, was_ctrl_c, was_typing)
            - input_text: The user input (empty if Ctrl+C or Enter pressed)
            - was_ctrl_c: True if Ctrl+C was pressed
            - was_typing: True if there was text when Ctrl+C was pressed
        """
        self.original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self.input_buffer = ""
        self.input_started = False
        self.ctrl_c_received = False

        try:
            console.print(prompt_text, end="")

            import sys
            import termios
            import tty

            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())

                while True:
                    if self.ctrl_c_received:
                        was_typing = len(self.input_buffer) > 0
                        if was_typing:
                            print("\r\033[K", end="", flush=True)
                        return "", True, was_typing

                    char = sys.stdin.read(1)

                    if ord(char) == 3:  # Ctrl+C
                        was_typing = len(self.input_buffer) > 0
                        if was_typing:
                            print("\r\033[K", end="", flush=True)
                        return "", True, was_typing
                    elif ord(char) == 13 or ord(char) == 10:  # Enter
                        print()  # Add newline
                        return self.input_buffer, False, False
                    elif ord(char) in (8, 127):  # Backspace (8) or DEL (127)
                        if self.input_buffer:
                            self.input_buffer = self.input_buffer[:-1]
                            print("\b \b", end="", flush=True)
                    elif ord(char) >= 32:  # Printable character
                        self.input_buffer += char
                        print(char, end="", flush=True)
                        self.input_started = True

            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        except Exception:
            try:
                result = input(" ")
                return result, False, False
            except KeyboardInterrupt:
                return "", True, False
        finally:
            if self.original_handler:
                signal.signal(signal.SIGINT, self.original_handler)


input_handler = SmartInputHandler()


class InteractiveChatSession:
    def __init__(self, config: ChirpSettings):
        self.config = config
        self.exit_attempts = 0

    def start(self):
        console.print(
            Panel(
                "[bold blue]Notes Chat[/bold blue]\n"
                "Ask questions about your meeting notes.\n\n"
                "[dim]Press Ctrl+C twice to exit[/dim]",
                border_style="blue",
                padding=(1, 2),
                expand=False,
            )
        )

        while True:
            try:
                question, was_ctrl_c, was_typing = input_handler.get_input(
                    "\n[bold blue]>[/bold blue] "
                )

                if question.strip():
                    self.exit_attempts = 0
                    self.handle_question(question)

                elif question == "" and not was_ctrl_c:
                    console.print("[dim]Please enter a question[/dim]")

                elif was_ctrl_c and was_typing:
                    pass

                elif was_ctrl_c and not was_typing:
                    self.exit_attempts += 1
                    if self.exit_attempts >= 2:
                        console.print("\n[dim]Goodbye![/dim]\n")
                        sys.exit(0)
                    else:
                        console.print("\n[dim]Press Ctrl+C again to exit[/dim]")

            except KeyboardInterrupt:
                self.exit_attempts += 1
                if self.exit_attempts >= 2:
                    console.print("\n[dim]Goodbye![/dim]\n")
                    sys.exit(0)
                else:
                    console.print("\n[dim]Press Ctrl+C again to exit[/dim]")
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]\n")
                sys.exit(0)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def handle_question(self, question: str):
        progress = None
        current_thinking_msg = ""
        console.print("")  # Add spacing

        try:
            answer_parts = []
            sources = None
            from_cache = False

            for stream_event in enhanced_search_and_answer_stream(
                self.config, question
            ):
                event_type = stream_event.get("type", "")

                if event_type == "thinking":
                    thinking_msg = stream_event.get("message", "Processing...")
                    if thinking_msg != current_thinking_msg:
                        if progress:
                            progress.stop()
                        progress = Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console,
                            transient=True,
                        )
                        progress.start()
                        progress.add_task(thinking_msg, total=None)
                        current_thinking_msg = thinking_msg

                elif event_type == "token":
                    if progress:
                        progress.stop()
                        progress = None
                    token = stream_event.get("content", "")
                    answer_parts.append(token)

                    if len(answer_parts) == 1:
                        console.print(
                            "[magenta]>[/magenta] chirp 🐣: ", end="", highlight=False
                        )

                    console.print(token, end="", highlight=False)

                elif event_type == "complete":
                    if progress:
                        progress.stop()
                        progress = None

                    answer = stream_event.get("answer", "")
                    sources = stream_event.get("sources")
                    from_cache = stream_event.get("from_cache", False)

                    # If we didn't stream tokens (cached response), print the full answer
                    if not answer_parts and answer:
                        console.print(
                            "[magenta]>[/magenta] chirp 🐣: ", end="", highlight=False
                        )
                        console.print(answer)

                    break

                elif event_type == "error":
                    if progress:
                        progress.stop()
                        progress = None
                    error_msg = stream_event.get("message", "Unknown error")
                    console.print(f"\n❌ {error_msg}")
                    return

            # Clean up progress if still running
            if progress:
                progress.stop()

            # Print sources if available
            if sources:
                console.print("")  # Add spacing
                sources_text = "📚 Sources:\n" + "\n".join(
                    f"  • {source}" for source in sources
                )
                console.print(f"[dim]{sources_text}[/dim]")

            # Print metadata
            if from_cache:
                console.print("\n[dim]cached[/dim]")

        except Exception as e:
            if progress:
                progress.stop()
            console.print(f"\n❌ Query failed: {e}")
        finally:
            if progress:
                progress.stop()
