# Development Guide

This document is for contributors working on Chirp. End users should see the top-level `README.md`.

## Prerequisites

- Python 3.9+
- macOS (BlackHole support for recording)
- Homebrew
- Git

## Setup

```bash
git clone <repository-url>
cd chirp-ai-note-app
make dev-install
```

This installs dependencies, sets up pre-commit hooks, and installs the package in development mode.

Install system deps:

```bash
make install-deps
```

## Running tests and checks

```bash
make test           # Unit tests
make test-coverage  # HTML coverage
make lint           # Ruff linting
make format         # Auto-format
make type-check     # mypy type checks
make check          # All quality checks
```

## Dev workflow

```bash
make dev-workflow   # Style, type-check, test, build
```

Individual commands are available via the Makefile for linting, formatting, spelling, and validation.

## Project structure

```text
chirp-ai-note-app/
├── chirp/                 # Main CLI application
│   ├── cli.py            # CLI interface with Typer
│   └── exceptions.py     # Custom exceptions
├── config/               # Configuration management
│   └── settings.py      # Pydantic settings with YAML
├── recorder/             # Audio recording modules
├── transcriber/          # Transcription and processing
├── notes/               # Note generation
├── notes_chat/          # Search and query functionality
├── utils/               # Utilities
├── templates/           # Markdown templates
├── .docs/               # Documentation
├── tests/               # Test suite
├── pyproject.toml       # Project configuration
├── Makefile             # Development commands
└── .pre-commit-config.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make check`
5. Open a pull request with context
