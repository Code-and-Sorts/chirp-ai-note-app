from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from notes_chat.config import get_notes_config

console = Console()
app = typer.Typer()


@app.command()
def index(
    force: bool = typer.Option(False, "--force", help="Force full rebuild of index"),
):
    """Build or rebuild the notes search index."""
    config = get_notes_config()
    console.print("[blue]🔍 Building notes index...[/blue]")

    if force:
        console.print("[yellow]--force specified, rebuilding entire index[/yellow]")

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
                f"[green]✅ Index built successfully: {result.get('files_processed', 0)} files processed[/green]"
            )
        else:
            console.print(
                f"[red]❌ Index build failed: {result.get('error', 'Unknown error')}[/red]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Index build failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ask(
    question: Optional[str] = typer.Option(
        None,
        "--question",
        "-q",
        help="Question to ask about your meetings (omit for interactive chat)",
    ),
    when: Optional[str] = typer.Option(
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
    markdown: bool = True,
):
    """Ask questions about your meeting notes. Run without a question for interactive chat."""
    config = get_notes_config()

    if question is None:
        from notes_chat.interactive import InteractiveChatSession

        session = InteractiveChatSession(config, markdown=markdown)
        session.start()
        return

    try:
        from notes_chat.cache import cache_answer, get_cached_answer
        from notes_chat.prompting import generate_answer
        from notes_chat.retrieval import retrieve_context

        console.print(f"[blue]🤔 Searching for: {question}[/blue]")

        context_result = retrieve_context(config, question, when_filter=when)

        if not context_result.get("success"):
            error = context_result.get("error", "Unknown error")
            if "no documents found" in error.lower():
                console.print("[yellow]📭 No relevant documents found.[/yellow]")
                if context_result.get("suggestion"):
                    console.print(f"[dim]💡 Try: {context_result['suggestion']}[/dim]")
                raise typer.Exit(2)
            else:
                console.print(f"[red]❌ Context retrieval failed: {error}[/red]")
                if context_result.get("suggestion"):
                    console.print(f"[dim]💡 {context_result['suggestion']}[/dim]")
                raise typer.Exit(1)

        context = context_result["context"]
        retrieved_ids = context_result["retrieved_ids"]

        if dry_run:
            console.print(
                "[yellow]🧪 Dry run mode - showing context and prompt:[/yellow]"
            )
            console.print(f"[dim]Context length: {len(context)} characters[/dim]")
            console.print(f"[dim]Retrieved chunks: {len(retrieved_ids)}[/dim]")
            console.print("\n[bold]Context:[/bold]")
            console.print(context[:1000] + "..." if len(context) > 1000 else context)
            return

        cached_answer = get_cached_answer(question, retrieved_ids)
        if cached_answer:
            console.print("[dim]📋 Using cached answer[/dim]")
            answer = cached_answer
        else:
            answer_result = generate_answer(config, question, context)

            if not answer_result.get("success"):
                console.print(
                    f"[red]❌ Answer generation failed: {answer_result.get('error', 'Unknown error')}[/red]"
                )
                raise typer.Exit(1)

            answer = answer_result["answer"]
            cache_answer(question, retrieved_ids, answer)

        console.print("\n[magenta bold]chirp ›[/magenta bold]")
        if markdown:
            from rich.markdown import Markdown

            console.print(Markdown(answer))
        else:
            console.print(answer)

        if sources and context_result.get("sources"):
            console.print("\n[dim]📚 Sources:[/dim]")
            for source in context_result["sources"]:
                console.print(f"[dim]  • {source}[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Query failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
