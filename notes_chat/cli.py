import json
import logging
import sys

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from chirp import exit_codes
from chirp._console import stderr_console, stdout_console
from llm.exceptions import (
    LLMCancelled,
    LLMDaemonSpawnFailed,
    LLMDaemonUnreachable,
    LLMError,
    LLMModelLoadFailed,
    LLMModelNotFound,
)
from notes_chat.config import get_notes_config

logger = logging.getLogger(__name__)

# Diagnostics/progress/errors → stderr; the answer body and JSON → stdout.
console = stderr_console
app = typer.Typer()


@app.command()
def index(
    force: bool = typer.Option(False, "--force", help="Force full rebuild of index"),
):
    """Build or rebuild the notes search index."""
    config = get_notes_config()
    console.print("[blue]Building notes index…[/blue]")

    if force:
        console.print("[yellow]--force specified, rebuilding entire index[/yellow]")
        from notes_chat.cache import clear_cache

        clear_cache()

    try:
        from notes_chat.index import build_index

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Indexing notes...", total=None)
            files_indexed = 0

            def on_progress():
                nonlocal files_indexed
                files_indexed += 1
                progress.update(
                    task, description=f"Indexing notes... ({files_indexed} files)"
                )

            result = build_index(config, force=force, progress_callback=on_progress)

        if result.get("success"):
            console.print(
                f"[green]Index built successfully: {result.get('files_processed', 0)} files processed[/green]"
            )
        else:
            console.print(
                f"[red]Index build failed: {result.get('error', 'Unknown error')}[/red]"
            )
            raise typer.Exit(exit_codes.RUNTIME_ERROR)

    except Exception as e:  # noqa: BLE001 - top-level CLI handler for index build
        logger.debug("Index build failed: %s", e, exc_info=True)
        console.print(f"[red]Index build failed: {e}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)


@app.command()
def ask(
    question: str | None = typer.Argument(
        None,
        help="Question to ask about your meetings (omit for interactive chat).",
    ),
    question_option: str | None = typer.Option(
        None,
        "--question",
        "-q",
        help="Same as the positional argument; kept for backwards compatibility.",
    ),
    when: str | None = typer.Option(
        None,
        "--when",
        help="Time range filter (e.g., 'last week', 'on:2025-01-15', '2025-01-01:2025-01-31')",
    ),
    sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Show source information"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be sent to LLM without calling it"
    ),
    markdown: bool = typer.Option(
        True,
        "--markdown/--no-markdown",
        help="Render answers as markdown (code blocks, bullets, bold).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the answer and sources as JSON to stdout (one-shot only).",
    ),
):
    """Ask questions about your meeting notes. Run without a question for interactive chat."""
    config = get_notes_config()

    if question is None:
        question = question_option

    if question is None:
        from notes_chat.interactive import InteractiveChatSession

        session = InteractiveChatSession(config, markdown=markdown)
        session.start()
        return

    try:
        from notes_chat.cache import cache_answer, get_cached_answer
        from notes_chat.prompting import generate_answer, stream_answer_tokens
        from notes_chat.retrieval import retrieve_context

        if not json_output:
            console.print(f"[dim]searching for: {question}[/dim]")

        context_result = retrieve_context(config, question, when_filter=when)

        if not context_result.get("success"):
            error = context_result.get("error", "Unknown error")
            if "no documents found" in error.lower():
                console.print("[yellow]No relevant documents found.[/yellow]")
                if context_result.get("suggestion"):
                    console.print(f"[dim]try: {context_result['suggestion']}[/dim]")
                raise typer.Exit(exit_codes.USAGE_ERROR)
            console.print(f"[red]Context retrieval failed: {error}[/red]")
            if context_result.get("suggestion"):
                console.print(f"[dim]{context_result['suggestion']}[/dim]")
            raise typer.Exit(exit_codes.RUNTIME_ERROR)

        context = context_result["context"]
        retrieved_ids = context_result["retrieved_ids"]

        if dry_run:
            if json_output:
                payload = {
                    "dry_run": True,
                    "question": question,
                    "context": context,
                    "retrieved_chunks": len(retrieved_ids),
                }
                sys.stdout.write(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
                )
                return
            console.print("[yellow]dry run — showing context and prompt:[/yellow]")
            console.print(f"[dim]Context length: {len(context)} characters[/dim]")
            console.print(f"[dim]Retrieved chunks: {len(retrieved_ids)}[/dim]")
            console.print("\n[bold]Context:[/bold]")
            console.print(context[:1000] + "..." if len(context) > 1000 else context)
            return

        cached_answer = get_cached_answer(question, retrieved_ids)
        if cached_answer:
            answer = cached_answer
            if not json_output:
                console.print("[dim]using cached answer[/dim]")
                console.print("\n[magenta bold]chirp ›[/magenta bold]")
                if markdown:
                    from rich.markdown import Markdown

                    stdout_console.print(Markdown(cached_answer))
                else:
                    stdout_console.print(cached_answer)
        elif json_output or markdown:
            # Under --json the answer is collected, not streamed/rendered, so no
            # Markdown is ever instantiated; the raw JSON is emitted at the end.
            answer_result = generate_answer(config, question, context)
            if not answer_result.get("success"):
                console.print(
                    f"[red]Answer generation failed: {answer_result.get('error', 'Unknown error')}[/red]"
                )
                raise typer.Exit(exit_codes.RUNTIME_ERROR)
            answer = answer_result["answer"]
            cache_answer(question, retrieved_ids, answer)
            if not json_output:
                console.print("\n[magenta bold]chirp ›[/magenta bold]")
                from rich.markdown import Markdown

                stdout_console.print(Markdown(answer))
        else:
            console.print("\n[magenta bold]chirp ›[/magenta bold]")
            tokens: list[str] = []
            for token in stream_answer_tokens(config, question, context):
                sys.stdout.write(token)
                sys.stdout.flush()
                tokens.append(token)
            sys.stdout.write("\n")
            sys.stdout.flush()
            answer = "".join(tokens).strip()
            if not answer:
                console.print("[red]Answer generation failed: empty response[/red]")
                raise typer.Exit(exit_codes.RUNTIME_ERROR)
            cache_answer(question, retrieved_ids, answer)

        answer_sources = (
            list(context_result["sources"]) if context_result.get("sources") else []
        )

        if json_output:
            payload = {
                "question": question,
                "answer": answer,
                "sources": answer_sources if sources else [],
            }
            sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            return

        if sources and answer_sources:
            joined = ", ".join(answer_sources)
            console.print(f"\n[dim]sources: {joined}[/dim]")

    except typer.Exit:
        raise
    except (LLMDaemonUnreachable, LLMDaemonSpawnFailed) as exc:
        console.print(
            f"[red]chirpd is not running and could not be started: {exc.message}[/red]"
        )
        console.print("[dim]run `chirp daemon status` for diagnostics[/dim]")
        raise typer.Exit(exit_codes.DAEMON_UNREACHABLE) from exc
    except LLMModelNotFound as exc:
        console.print(f"[red]{exc.message}[/red]")
        console.print(
            "[dim]run `chirp models add <hf-repo>` to register a chat model[/dim]"
        )
        raise typer.Exit(exit_codes.MODEL_NOT_FOUND) from exc
    except LLMModelLoadFailed as exc:
        console.print(f"[red]{exc.message}[/red]")
        raise typer.Exit(exit_codes.MODEL_LOAD_FAILED) from exc
    except LLMCancelled:
        console.print("[yellow]Interrupted.[/yellow]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR) from None
    except LLMError as exc:
        console.print(f"[red]{exc.message}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR) from exc
    except Exception as e:  # noqa: BLE001 - top-level CLI handler for ask command
        logger.debug("Query failed: %s", e, exc_info=True)
        console.print(f"[red]Query failed: {e}[/red]")
        raise typer.Exit(exit_codes.RUNTIME_ERROR)


if __name__ == "__main__":
    app()
