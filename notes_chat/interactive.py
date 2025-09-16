import signal
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import ChirpSettings
from notes_chat.cache import cache_answer, get_cached_answer
from notes_chat.prompting import generate_answer
from notes_chat.retrieval import retrieve_context

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
        self.chat_history: list[dict[str, Any]] = []

    def start(self):
        console.print(
            Panel(
                "[bold blue]Notes Chat[/bold blue]\n"
                "Ask questions about your meeting notes.\n\n"
                "[dim]Press Ctrl+C twice to exit[/dim]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        while True:
            try:
                question, was_ctrl_c, was_typing = input_handler.get_input(
                    "\n[bold blue]>[/bold blue] "
                )

                if question.strip():
                    self.exit_attempts = 0
                    self._add_to_history("question", question)
                    self.handle_question(question)

                elif question == "" and not was_ctrl_c:
                    console.print("[dim]Please enter a question[/dim]")

                elif was_ctrl_c and was_typing:
                    pass

                elif was_ctrl_c and not was_typing:
                    self.exit_attempts += 1
                    if self.exit_attempts >= 2:
                        console.print("\n[dim]Goodbye![/dim]")
                        sys.exit(0)
                    else:
                        console.print("\n[dim]Press Ctrl+C again to exit[/dim]")

            except KeyboardInterrupt:
                self.exit_attempts += 1
                if self.exit_attempts >= 2:
                    console.print("\n[dim]Goodbye![/dim]")
                    sys.exit(0)
                else:
                    console.print("\n[dim]Press Ctrl+C again to exit[/dim]")
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                sys.exit(0)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def _add_to_history(self, entry_type, content):
        self.chat_history.append({"type": entry_type, "content": content})

    def _display_chat_in_box(self):
        """Display the current chat history in a bordered box"""
        if not self.chat_history:
            return

        recent_history = self.chat_history[-10:]
        chat_lines = []

        for entry in recent_history:
            if entry["type"] == "question":
                chat_lines.append(f"[bold blue]Q:[/bold blue] {entry['content']}")
            elif entry["type"] == "answer":
                chat_lines.append(f"[green]A:[/green] {entry['content']}")
            elif entry["type"] == "message":
                chat_lines.append(f"[dim]{entry['content']}[/dim]")

        chat_content = "\n".join(chat_lines)
        console.print(
            Panel(
                chat_content, border_style="cyan", title="Chat History", padding=(1, 2)
            )
        )

    def handle_question(self, question: str):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Searching...", total=None)

            try:
                context_result = retrieve_context(self.config, question)

                if not context_result.get("success"):
                    error = context_result.get("error", "Unknown error")
                    if "no documents found" in error.lower():
                        self._add_to_history(
                            "message", "📭 No relevant documents found."
                        )
                        if context_result.get("suggestion"):
                            self._add_to_history(
                                "message", f"💡 Try: {context_result['suggestion']}"
                            )
                    else:
                        self._add_to_history(
                            "message", f"❌ Context retrieval failed: {error}"
                        )
                    self._display_chat_in_box()
                    return

                context = context_result["context"]
                retrieved_ids = context_result["retrieved_ids"]

                cached_answer = get_cached_answer(question, retrieved_ids)
                if cached_answer:
                    progress.update(task, description="Using cached answer...")
                    answer = cached_answer
                else:
                    progress.update(task, description="Generating answer...")
                    answer_result = generate_answer(self.config, question, context)

                    if not answer_result.get("success"):
                        self._add_to_history(
                            "message",
                            f"❌ Answer generation failed: {answer_result.get('error', 'Unknown error')}",
                        )
                        self._display_chat_in_box()
                        return

                    answer = answer_result["answer"]
                    cache_answer(question, retrieved_ids, answer)

                self._add_to_history("answer", answer)
                self._display_chat_in_box()

                if context_result.get("sources"):
                    sources_text = "📚 Sources:\n" + "\n".join(
                        f"  • {source}" for source in context_result["sources"]
                    )
                    self._add_to_history("message", sources_text)
                    console.print(f"\n[dim]{sources_text}[/dim]")

            except Exception as e:
                self._add_to_history("message", f"❌ Query failed: {e}")
                self._display_chat_in_box()
