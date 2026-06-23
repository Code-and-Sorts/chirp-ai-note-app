# Repository Guidelines

## Project Structure & Module Organization
Chirp is a Typer-based CLI in `chirp/cli.py`. Runtime domains live in sibling packages: `recorder/` for capture and live transcription, `transcriber/` for Whisper processing, `notes/` for note generation and editing, `notes_chat/` for search/chat, `config/` for Pydantic settings, and `utils/` for shared helpers. Notes are stored under `~/Documents/chirp` by default as per-note directories containing `audio.wav`, `transcript.txt`, `notes.md`, and `meta.toml`. Tests mirror runtime domains in `tests/`, and contributor docs live in this file plus `.docs/DEVELOPMENT.md`.

## Build, Test & Development Commands
**Setup:** `make dev-install` installs system deps, syncs Python dependencies with `uv`, installs the package editable, and enables pre-commit hooks. Use `make install-venv` if the virtualenv already exists and you only need the editable install.
**macOS toolchain:** building the bundled `Chirp.app` helper needs `swiftc` (Swift 5.9+, macOS 13+), shipped with `xcode-select --install`. No separate C toolchain is required — the disclaim shim is consolidated into `capture_audio.swift`. Run `uv run python -m audio_capture.build` to (re)build the helper when you change Swift sources.

**Signing for local dev:** the helper is ad-hoc signed by default, so each rebuild changes its code hash and macOS re-prompts for Microphone / Screen Recording. To keep the grant across rebuilds, run `scripts/create-dev-signing-cert.sh` once and `export CHIRP_CODESIGN_IDENTITY="Chirp Dev"` before building. CI and distribution wheels stay ad-hoc.
**Quality:** `make check` runs `validate`, `lint`, `format-check`, `spell-check`, and `type-check`. Use `make lint-fix` and `make format` before pushing.
**Testing:** `make test`, `make test-coverage`, `make test-file FILE=tests/test_settings.py`, `make test-match PATTERN=slugify`, and `make test-failed`.
**CLI verification:** `uv run chirp --help`, `uv run chirp init --recheck`, and `make verify-deps` are the current repo-supported smoke checks.

## Coding Style & Naming Conventions
**Formatting:** Ruff enforces 88-char lines, double quotes, and import sorting. Use absolute imports and keep stdlib / third-party / first-party groups clean.
**Naming:** Public CLI commands are `record`, `transcribe`, `notes`, `ask`, `search`, `init`, and `about`. Hidden maintenance commands such as `config`, `devices`, `index`, and `daemon` exist, but they are not the primary user workflow.
**Types:** Add type hints where practical; mypy checks `chirp`, `config`, `notes`, `notes_chat`, `recorder`, `transcriber`, and `utils`.
**Errors:** Prefer domain exceptions from `chirp.exceptions` and user-facing messages that explain the next recovery step.
**Comments:** Default to **zero**. A comment is justified only for a *why* the code can't express (workaround, subtle invariant, footgun, spec/security reason) — ideally with a reference. NEVER add comments that restate the code (`# loop over notes`), narrate steps (`# Step 1:`, `# Now build …`), re-describe a well-named symbol, or head sections. Test before adding: "would a competent engineer be misled without it?" — if not, cut it. Fix the code instead: a clearer name, a smaller helper, or a named constant beats a comment. Terse tooling markers (`# noqa`, `# type: ignore`, `# pragma: no cover`) are fine.

## Testing Guidelines
Write `pytest` tests under `tests/` with `test_*.py` names and behavior-focused test functions. Reuse fixtures to isolate chirpd / `llm.client`, audio devices, and filesystem state. Mark slow or integration coverage explicitly with `@pytest.mark.slow` and `@pytest.mark.integration`. When changing CLI output or flows, update the closest focused tests first, then run the smallest relevant test slice before broader checks.

## Documentation Guidelines
`README.md` is the canonical user-facing readme for both GitHub and package metadata. Keep command examples aligned with live `chirp --help` output and with `config/settings.py`. If shared contributor guidance changes, update `AGENTS.md` first and keep `CLAUDE.md` as a thin compatibility file rather than duplicating long-form instructions.

## Commit & Pull Request Guidelines
Use short imperative commit subjects. Keep changes scoped, run `make check` and `make test` before opening a PR, and include runtime-facing context when behavior or prompts change. For CLI UX changes, sample terminal output is more useful than screenshots.
