import sys
import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import ChirpSettings
from notes_chat.prompting import enhanced_search_and_answer_stream

console = Console()


class InteractiveChatSession:
    def __init__(self, config: ChirpSettings):
        self.config = config
        self.last_interrupt_time = None
        self.interrupt_timeout = 2.0

        toolbar_style = Style.from_dict(
            {
                "bottom-toolbar": "bg:default fg:default noreverse",
                "bottom-toolbar.text": "bg:default fg:default noreverse",
            }
        )

        self._session = PromptSession(style=toolbar_style)
        self._kb = KeyBindings()
        self._hint_timer = None
        self._QUIT = "__CHIRP_QUIT__"

        @self._kb.add("c-c")
        def _(event):
            now = time.time()
            if (
                self.last_interrupt_time
                and (now - self.last_interrupt_time) <= self.interrupt_timeout
            ):
                event.current_buffer.reset()
                event.app.exit(result=self._QUIT)
                return
            self.last_interrupt_time = now
            event.current_buffer.reset()
            self._show_hint_then_auto_clear()

        self._session.default_buffer.on_text_insert += self._on_user_activity
        self._session.default_buffer.on_cursor_position_changed += (
            self._on_user_activity
        )

    def _on_user_activity(self, _):
        if self.last_interrupt_time is not None:
            self._hide_hint()

    def _hide_hint(self):
        self.last_interrupt_time = None
        if self._hint_timer:
            self._hint_timer.cancel()
            self._hint_timer = None
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def _show_hint_then_auto_clear(self):
        if self._hint_timer:
            self._hint_timer.cancel()
        self._hint_timer = threading.Timer(self.interrupt_timeout, self._hide_hint)
        self._hint_timer.daemon = True
        self._hint_timer.start()
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def _toolbar(self):
        if self.last_interrupt_time is None:
            return ""
        if (time.time() - self.last_interrupt_time) > self.interrupt_timeout:
            return ""
        return "Press Ctrl+C again to exit"

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
                question = self._session.prompt(
                    "> ",
                    key_bindings=self._kb,
                    bottom_toolbar=self._toolbar,
                )

                if question == self._QUIT:
                    sys.stdout.write("\r\x1b[2K")
                    sys.stdout.flush()
                    console.print("[dim]Goodbye![/dim]")
                    return

                self._hide_hint()

                if question.strip():
                    self.handle_question(question)
                else:
                    console.print("[dim]Please enter a question[/dim]")

            except EOFError:
                try:
                    sys.stdout.write("\r\x1b[2K")
                    sys.stdout.flush()
                except Exception:
                    pass
                console.print("\n[dim]Goodbye![/dim]")
                return
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def handle_question(self, question: str):
        progress = None
        current_thinking_msg = ""

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

            if progress:
                progress.stop()

            if answer_parts:
                console.print()

            if sources:
                console.print("")
                sources_text = "📚 Sources:\n" + "\n".join(
                    f"  • {source}" for source in sources
                )
                console.print(f"[dim]{sources_text}[/dim]")

            if from_cache:
                console.print("\n[dim]cached[/dim]")

        except Exception as e:
            if progress:
                progress.stop()
            console.print(f"\n❌ Query failed: {e}")
        finally:
            if progress:
                progress.stop()
