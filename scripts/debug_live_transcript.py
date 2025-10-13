from __future__ import annotations

from pathlib import Path

import typer

from config.settings import get_settings
from transcriber.whisper_transcriber import WhisperTranscriber

app = typer.Typer(help="Debug live transcription chunks")


@app.command()
def transcribe_chunk(
    chunk_path: Path = typer.Argument(..., exists=True, readable=True),
    sample_rate: int = typer.Option(16000, help="Sample rate of the chunk"),
):
    """Transcribe a single debug chunk using the whisper pipeline."""
    settings = get_settings()
    transcriber = WhisperTranscriber(settings)

    result = transcriber.transcribe_file(chunk_path)
    if result["success"]:
        typer.secho(f"Transcript for {chunk_path.name}:", fg=typer.colors.GREEN)
        typer.echo(result["full_text"])
    else:
        typer.secho(
            f"Failed to transcribe {chunk_path.name}: {result['error']}",
            fg=typer.colors.RED,
        )


if __name__ == "__main__":
    app()
