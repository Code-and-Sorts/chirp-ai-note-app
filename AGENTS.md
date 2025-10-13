# Repository Guidelines

## Project Structure & Module Organization
Chirp is a Python CLI whose runtime code lives under `chirp/` with Typer commands in `chirp/cli.py`. Recorder, transcription, note generation, and chat pipelines live in sibling packages (`recorder/`, `transcriber/`, `notes/`, `notes_chat/`, `utils/`) and share Pydantic config from `config/settings.py`. Markdown templates sit in `templates/`, while runtime artifacts land in `to-transcribe/`, `transcription-out/`, and `notes-out/`. Tests mirror the domains in `tests/` (e.g., `tests/test_note_generator.py`), so add new coverage alongside the feature.

## Build, Test & Development Commands
**Setup:** `make dev-install` (installs deps with `uv` and pre-commit hooks).
**Quality checks:** `make check` chains `validate`, `lint`, `format-check`, `spell-check`, and `type-check`. Run `make lint-fix` and `make format` to auto-fix issues before pushing.
**Testing:** `make test` runs pytest; `make test-coverage` generates HTML reports. Run a single test with `uv run pytest tests/test_note_generator.py` or a specific function with `uv run pytest tests/test_note_generator.py::test_function_name`. Use `make test-failed` to re-run only failures.
**Validation:** `uv run chirp status` verifies environment; `make process` smoke-tests the full pipeline (record → transcribe → notes).

## Coding Style & Naming Conventions
**Formatting:** Ruff enforces 88-char lines, double quotes, space indents (run `make format` after edits). Pre-commit hooks auto-fix ruff, codespell, trailing whitespace, and YAML.
**Imports:** Group stdlib, third-party, and first-party (`chirp`, `config`, `notes`, etc.) via isort. Use absolute imports; avoid star imports except for exceptions.
**Naming:** Files and functions follow domain actions (`*_manager`, `*_processor`). CLI commands stay terse (see `chirp/cli.py`). Classes use PascalCase; functions/vars use snake_case.
**Types:** Type hints encouraged; mypy checks `chirp`, `config`, `notes`, `notes_chat`, `recorder`, `transcriber`, `utils`. Silence warnings with annotations, not `type: ignore`.
**Comments:** Avoid unnecessary comments; write self-documenting code. Document only non-obvious intent that code cannot express.
**Error handling:** Raise custom exceptions from `chirp.exceptions` for domain errors; use generic exceptions sparingly.

## Testing Guidelines
Write `pytest` tests in `tests/` with filenames starting `test_` and functions mirroring user-facing behavior. Use fixtures to stub audio and Ollama integrations; see `tests/test_audio_recorder.py` and `tests/notes_chat/` for patterns. Mark slow or integration tests with `@pytest.mark.slow` or `@pytest.mark.integration`; skip with `@pytest.mark.skip(reason="...")` if needed. Keep coverage above existing baselines by running `make test-coverage` locally.

## Commit & Pull Request Guidelines
Commit subjects follow the short, imperative style (e.g., `Add file name override option`). Group related changes; include meaningful bodies when context is not obvious. Before opening a PR, ensure `make check` and `make test` pass, link any GitHub issues, and describe runtime impacts. Screenshots or sample CLI output help reviewers validate UX changes; attach them when modifying prompts or note templates.
