# Repository Guidelines

## Project Structure & Module Organization
Chirp is a Python CLI whose runtime code lives under `chirp/` with Typer commands in `chirp/cli.py`. Recorder, transcription, note generation, and chat pipelines live in sibling packages (`recorder/`, `transcriber/`, `notes/`, `notes_chat/`, `utils/`) and share Pydantic config from `config/settings.py`. Markdown templates sit in `templates/`, while runtime artifacts land in `to-transcribe/`, `transcription-out/`, and `notes-out/`. Tests mirror the domains in `tests/` (e.g., `tests/test_note_generator.py`), so add new coverage alongside the feature.

## Build, Test & Development Commands
Use `make dev-install` once to sync dependencies with `uv` and install pre-commit hooks. Day-to-day, run `make lint`, `make format`, and `make type-check` before pushing; `make check` chains the validation helpers. Run the full suite with `make test` or `make test-coverage` when you need HTML coverage reports. Use `uv run chirp status` to verify your environment, and `make process` for an end-to-end functional smoke.

## Coding Style & Naming Conventions
Ruff enforces formatting (88-char lines, double quotes, spaces for indent), so run `make format` after major edits. Prefer descriptive module-level functions and keep CLI command names terse (see `chirp/cli.py`). Name files and functions after the domain action (`*_manager`, `*_processor`). Follow "Clean Code" guidance: keep logic readable, avoid unnecessary comments, and document only intent that code cannot express. Type hints are encouraged; mypy runs against key packages, so silence warnings with actual annotations instead of `type: ignore`.

## Testing Guidelines
Write `pytest` tests in `tests/` with filenames starting `test_` and functions mirroring user-facing behavior. Use fixtures to stub audio and Ollama integrations; see `tests/test_audio_recorder.py` and `tests/notes_chat/` for patterns. When adding asynchronous or long-running flows, include a fast unit test and, if needed, a skipped integration marked with a reason. Keep coverage above existing baselines by running `make test-coverage` locally.

## Commit & Pull Request Guidelines
Commit subjects follow the short, imperative style already in history (e.g., `Add file name override option`). Group related changes and include meaningful bodies when context is not obvious. Before opening a PR, ensure `make check` and `make test` pass, link any GitHub issues, and describe runtime impacts. Screenshots or sample CLI output help reviewers validate UX changes; attach them when modifying prompts or note templates.
