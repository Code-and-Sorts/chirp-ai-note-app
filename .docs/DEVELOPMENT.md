# Development Guide

This document is for contributors working on Chirp. End users should start with the top-level `README.md`.

## Prerequisites

- Python 3.11+
- macOS for local audio-capture development
- Homebrew
- Git
- Ollama for note-generation and retrieval flows

## Setup

```bash
git clone <repository-url>
cd chirp-ai-note-app
make dev-install
```

This installs system dependencies, syncs the Python environment with `uv`, installs the package editable, and enables pre-commit hooks.

If you only need the editable install in an already-prepared environment:

```bash
make install-venv
```

## Quality checks

```bash
make check
make test
make test-coverage
make lint-fix
make format
make type-check
```

Targeted test helpers are also available:

```bash
make test-file FILE=tests/test_settings.py
make test-match PATTERN=slugify
make test-failed
```

## Useful CLI checks

```bash
uv run chirp --help
uv run chirp init --recheck
make verify-deps
```

## Project structure

```text
chirp-ai-note-app/
├── chirp/           # Typer CLI entrypoint and high-level flows
├── config/          # Pydantic settings and config-path helpers
├── recorder/        # Audio recording, device handling, live transcription
├── transcriber/     # Whisper transcription and batch processing
├── notes/           # Note generation, templates, manual editing
├── notes_chat/      # Retrieval, keyword search, chat flows, indexing
├── utils/           # Shared filesystem and time helpers
├── templates/       # Prompt and note templates
├── scripts/         # Dev/debug helper scripts
├── tests/           # Pytest suite
├── AGENTS.md        # Canonical contributor guidance
└── README.md        # Canonical user-facing readme
```

## Notes storage

- Config file: `~/.chirp/config.toml`
- Default notes root: `~/Documents/chirp`
- Each note lives in its own folder with `audio.wav`, `transcript.txt`, `notes.md`, and `meta.toml`

## Contributing

1. Branch from `main`.
2. Keep changes scoped.
3. Run the relevant tests plus `make check`.
4. Open a PR with context about behavior changes and any CLI-facing output changes.
