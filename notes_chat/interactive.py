import sys
import time

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
                question = console.input("\n[bold blue]>[/bold blue] ")
                self.last_interrupt_time = None

                if question.strip():
                    self.handle_question(question)
                else:
                    console.print("[dim]Please enter a question[/dim]")

            except KeyboardInterrupt:
                current_time = time.time()

                if self.last_interrupt_time is not None:
                    time_diff = current_time - self.last_interrupt_time

                    if time_diff <= self.interrupt_timeout:
                        console.print("\n[dim]Goodbye![/dim]\n")
                        sys.exit(0)
                    else:
                        self.last_interrupt_time = None
                else:
                    console.print("\n[dim]Press Ctrl+C again to exit[/dim]")
                    self.last_interrupt_time = current_time

                continue

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

            if progress:
                progress.stop()

            if sources:
                console.print("")  # Add spacing
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
