import logging
import sys
import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from chirp._console import stderr_console, stdout_console
from config.settings import ChirpSettings
from llm.client import LLMClient
from llm.exceptions import LLMError
from notes_chat.prompting import enhanced_search_and_answer_stream

logger = logging.getLogger(__name__)

# Diagnostics (banner, hints, errors, spinners) → stderr; the streamed answer
# body → stdout.
console = stderr_console


class InteractiveChatSession:
    def __init__(
        self,
        config: ChirpSettings,
        markdown: bool = True,
        tags: list[str] | None = None,
    ):
        self.config = config
        self.markdown = markdown
        self.tags = list(tags) if tags else None
        self.last_interrupt_time = None
        self.interrupt_timeout = 2.0
        self._inflight_req_id: str | None = None

        toolbar_style = Style.from_dict(
            {
                "bottom-toolbar": "bg:default fg:default noreverse",
                "bottom-toolbar.text": "bg:default fg:default noreverse",
            }
        )

        self._session: PromptSession = PromptSession(style=toolbar_style)
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
        except (RuntimeError, AttributeError):
            # UI session may be torn down already; invalidate is fire-and-forget.
            pass

    def _show_hint_then_auto_clear(self):
        if self._hint_timer:
            self._hint_timer.cancel()
        self._hint_timer = threading.Timer(self.interrupt_timeout, self._hide_hint)
        self._hint_timer.daemon = True
        self._hint_timer.start()
        try:
            self._session.app.invalidate()
        except (RuntimeError, AttributeError):
            # Same as above: invalidate is best-effort during teardown / re-entry.
            pass

    def _toolbar(self):
        if self.last_interrupt_time is None:
            return ""
        if (time.time() - self.last_interrupt_time) > self.interrupt_timeout:
            return ""
        return "Press Ctrl+C again to exit"

    def _count_notes(self) -> int:
        try:
            notes_root = self.config.directories.notes_root
            if not notes_root.exists():
                return 0
            return sum(1 for _ in notes_root.glob("*/notes.md"))
        except OSError:
            return 0

    def start(self):
        note_count = self._count_notes()
        model = getattr(self.config.models, "llm", "local")
        tag_scope = f" · tag: {', '.join(self.tags)}" if self.tags else ""
        header = (
            f"[cyan bold]Chirp[/cyan bold] [dim]· chat over {note_count} notes"
            f"{tag_scope} · {model} (local)[/dim]"
        )
        console.print()
        console.print(header)
        console.print("[dim]type your question, or /help · ctrl+d to exit[/dim]")
        console.print()

        while True:
            try:
                question = self._session.prompt(
                    ANSI("\x1b[1;32myou ›\x1b[0m "),
                    key_bindings=self._kb,
                    bottom_toolbar=self._toolbar,
                )

                if question == self._QUIT:
                    sys.stdout.write("\r\x1b[2K")
                    sys.stdout.flush()
                    console.print("[dim]bye![/dim]")
                    return

                self._hide_hint()

                stripped = question.strip()
                if not stripped:
                    continue
                if stripped.startswith("/"):
                    if not self._handle_slash(stripped):
                        return
                    continue
                self.handle_question(stripped)

            except EOFError:
                try:
                    sys.stdout.write("\r\x1b[2K")
                    sys.stdout.flush()
                except OSError:
                    # Terminal may have closed mid-prompt; goodbye banner still prints below.
                    pass
                console.print("\n[dim]Goodbye![/dim]")
                return
            except (KeyboardInterrupt, RuntimeError, ValueError) as e:
                console.print(f"[red]Error: {e}[/red]")

    def _handle_slash(self, command: str) -> bool:
        """Return False when the loop should exit."""
        cmd = command.lstrip("/").strip().lower()
        if cmd in {"exit", "quit", "q"}:
            console.print("[dim]bye![/dim]")
            return False
        if cmd == "help":
            console.print()
            console.print(" [bold]slash commands[/bold]")
            console.print(" [dim]/help[/dim]    show this list")
            console.print(" [dim]/clear[/dim]   clear the scrollback")
            console.print(" [dim]/exit[/dim]    quit (or ctrl+d)")
            console.print()
            return True
        if cmd == "clear":
            console.clear()
            return True
        console.print(f"[yellow]unknown command: /{cmd}[/yellow]")
        return True

    def handle_question(self, question: str):
        from rich.live import Live
        from rich.text import Text

        progress = None
        current_thinking_msg = ""
        live: Live | None = None
        answer_parts: list[str] = []

        def _start_live() -> Live:
            console.print("[magenta bold]chirp ›[/magenta bold]")
            initial = Markdown("") if self.markdown else Text("")
            new_live = Live(
                initial,
                console=stdout_console,
                refresh_per_second=10,
                vertical_overflow="visible",
            )
            new_live.start()
            return new_live

        def _update_live(target: Live, text: str) -> None:
            if self.markdown:
                target.update(Markdown(text))
            else:
                target.update(Text(text))

        stream_gen = None

        try:
            sources = None
            from_cache = False

            stream_gen = enhanced_search_and_answer_stream(
                self.config, question, tags=self.tags
            )
            for stream_event in stream_gen:
                event_type = stream_event.get("type", "")

                if event_type == "request_started":
                    self._inflight_req_id = stream_event.get("req_id")
                    continue

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
                    if live is None:
                        live = _start_live()
                    _update_live(live, "".join(answer_parts))

                elif event_type == "complete":
                    if progress:
                        progress.stop()
                        progress = None

                    answer = stream_event.get("answer", "") or "".join(answer_parts)
                    sources = stream_event.get("sources")
                    from_cache = stream_event.get("from_cache", False)

                    if not answer:
                        break

                    if live is None:
                        live = _start_live()
                    _update_live(live, answer)
                    break

                elif event_type == "error":
                    if progress:
                        progress.stop()
                        progress = None
                    if live is not None:
                        live.stop()
                        live = None
                    error_msg = stream_event.get("message", "Unknown error")
                    console.print(f"\n[red]{error_msg}[/red]")
                    return

            if progress:
                progress.stop()
            if live is not None:
                live.stop()
                live = None

            if sources:
                console.print("[dim]sources: " + ", ".join(sources) + "[/dim]")

            if from_cache:
                console.print("[dim]cached[/dim]")

        except KeyboardInterrupt:
            # Mid-stream Ctrl-C cancels this answer (not the session). Tell the
            # daemon to stop generating, unwind the generator so its cleanup
            # runs, and return to the prompt. This does NOT touch
            # last_interrupt_time, so it never counts toward the two-press exit.
            if self._inflight_req_id is not None:
                try:
                    LLMClient().cancel_sync(self._inflight_req_id)
                except LLMError as exc:
                    logger.debug("Cancel request failed: %s", exc)
            if stream_gen is not None:
                stream_gen.close()
            if progress:
                progress.stop()
                progress = None
            if live is not None:
                live.stop()
                live = None
            console.print("[dim]— cancelled[/dim]")
        except Exception as e:  # noqa: BLE001 - LLM streaming or UI; many failure modes
            logger.debug("Query failed: %s", e, exc_info=True)
            if progress:
                progress.stop()
            if live is not None:
                live.stop()
            console.print(f"\n[red]Query failed: {e}[/red]")
        finally:
            self._inflight_req_id = None
            if progress:
                progress.stop()
            if live is not None:
                live.stop()
