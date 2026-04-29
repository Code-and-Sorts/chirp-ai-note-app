# Epic: Align Chirp CLI with Wireframe (Direction A — Classic Subcommands)

- **Epic ID:** EPIC-WF-ALIGN
- **Owner:** Colby
- **Status:** Draft
- **Created:** 2026-04-20
- **Design source:** Chirp CLI Wireframes — Direction A ("Classic subcommands"); locked design
- **Related branch (current work):** `claude/implement-chirp-cli-IoXWx`

## 1. Goal

Bring the Chirp CLI's surface, storage layout, and transcribe pipeline into exact alignment with the locked Direction A wireframe so that all future docs, demos, and the README line up with one canonical design. No back-compat required (WIP).

## 2. Why now

The CLI has drifted in three structural ways that make the tool hard to demo against the wireframe and noisy in `--help`:

1. **Surface bloat** — 17 top-level commands across 3 help panels (`MAIN_PANEL`, `SETUP_PANEL`, `INFO_PANEL`) vs. the wireframe's 7 commands in a single panel.
2. **`notes` vs `note` split** — two separate top-level commands vs. the wireframe's single `notes` entry point with a `view N` / `edit N` subcommand hint.
3. **Storage split-by-kind** — current writes `~/Documents/Chirp/{recordings,transcripts,notes}/…`; wireframe locked decision 5 is one folder per note: `~/Documents/chirp/<slug>/` containing `audio.wav`, `transcript.txt`, `notes.md`, `meta.toml`.

Additionally, the transcribe pipeline needs to become queue-driven (FIFO, all-untranscribed by default; optional `N` batch arg) with a 6-step per-note status checklist.

## 3. Locked decisions from the wireframe

| # | Decision | Source |
|---|----------|--------|
| 1 | Command surface: `record · transcribe · notes · ask · search · init · about` (7 total, single panel) | A1 --help |
| 2 | `notes` is the entry point; `chirp notes view <id>` / `edit <id>` / `delete <id>` subcommands with `--tag` filter | A4 + product decision |
| 3 | `transcribe` renders 5 stages as a checklist (loaded audio → transcribe → generate notes → index → save); no language stage, no progress bar, no streaming | A6 + product decision |
| 4 | `transcribe` processes FIFO queue by default; `transcribe N` processes the N oldest untranscribed | A6 header |
| 5 | Storage is one folder per note: `~/Documents/chirp/<slug>/{audio.wav,transcript.txt,notes.md,meta.toml}` | Locked decision 5 |
| 6 | Root folder is lowercase `chirp/`, not `Chirp/` | Locked decision 5 |
| 7 | Config moves to `~/.chirp/config.toml` (TOML, not YAML) with Chroma at `~/.chirp/chroma/` | Init Phase 4 |
| 8 | `init` is a 4-phase flow: verify → install → pick → pull/finalize; `--recheck` and `--switch-model` flags; Phase 2 BlackHole prompt uses `[o]/[s]` key picker | Init wireframe A2 |
| 9 | Language is not a user-facing concern (Whisper auto-detects); removed from the checklist. Multi-language UX deferred. | product decision |
| 10 | `record` prompts title (required) → timeframe (optional) → tags (optional, comma-separated) | A3 + product decision |
| 11 | `ask` is a standalone top-level command, not a `notes` subcommand | product decision |

## 4. Research findings — what already exists vs. what is missing

Validated against `main` at commit `a2552bc`:

### Already implemented (verify/polish only)
- **`init_flow.run_init`** with 4 phases (`chirp/init_flow.py:472`); supports `--recheck` and `--switch-model`.
- **`ask`** with `--markdown/--no-markdown` default-on, interactive chat mode, `--sources`, `--when` (`chirp/cli.py:556`).
- **`search`** as a live keyword-search session (`chirp/cli.py:457`).
- **`about`** command with 3-frame animation (`chirp/about.py`).
- **Live recording dashboard** with scrollable transcript, VAD, pause/resume (`recorder/live_dashboard.py`, `recorder/live_session.py`).
- **Manual note editor** with view / edit modes (`notes/note_editor.py`, `notes/manual_note_manager.py`).
- **`BatchProcessor`** does sequential + concurrent file processing with force flag (`transcriber/batch_processor.py:16`).
- **Whisper transcriber** with streaming / `on_segment` callback (`transcriber/whisper_transcriber.py`).

### Needs to change
- **`chirp/cli.py`** — 17 `@app.command` decorators across 3 panels; need prune to 7 visible + hide diagnostics.
- **`config/settings.py`** — `DirectoriesConfig` uses flat-by-kind; config lives at `user_config_dir/config.yaml`.
- **`utils/file_utils.py`** — `get_audio_files` / `get_transcription_files` / `get_notes_files` all scan separate dirs.
- **`transcriber/batch_processor.py`** — output path is `<transcription_id>.json.gz` + sibling `metadata.json`; no 6-step checklist UI; no FIFO queue awareness; `process_concurrently` exists but wireframe says sequential.
- **`recorder/*`** — audio written to `settings.directories.raw_audio/…`; needs to write to `notes_root/<slug>/audio.wav` and drop initial `meta.toml`.
- **`notes/note_generator.py`** — writes to `settings.directories.notes`; needs to write `notes.md` next to the audio and update `meta.toml`.
- **`notes_chat/index.py`** — scans the old `notes` dir; needs `notes_root/*/notes.md`.
- **`chirp/init_flow.py`** — Phase 4 writes YAML to `user_config_dir`; needs TOML at `~/.chirp/config.toml`; Phase 2 needs to absorb BlackHole audio-routing prompt from deleted `setup` command.
- **Notes list hint** (`chirp/cli.py:453`) — prints `chirp note [NAME]`; needs to say `chirp notes view N` / `chirp notes edit N`.

## 5. Stories

Execution order matters: storage is the foundation — most other stories depend on the new `NoteRecord` shape.

| ID | Title | Depends on | File |
|----|-------|------------|------|
| 1.1 | Storage rewrite — one folder per note, TOML config | — | [stories/1.1-storage-rewrite.md](stories/1.1-storage-rewrite.md) |
| 1.2 | Command-surface prune to 7 visible commands | 1.1 | [stories/1.2-command-surface-prune.md](stories/1.2-command-surface-prune.md) |
| 1.3 | Merge `note` into `notes` sub-app (`view`/`edit`/`delete N`) | 1.1, 1.2 | [stories/1.3-notes-sub-app.md](stories/1.3-notes-sub-app.md) |
| 1.4 | Queue-driven `transcribe` with 6-step checklist | 1.1 | [stories/1.4-transcribe-queue-checklist.md](stories/1.4-transcribe-queue-checklist.md) |
| 1.5 | `init` polish — absorb BlackHole prompt, write TOML | 1.1 | [stories/1.5-init-polish.md](stories/1.5-init-polish.md) |
| 1.6 | Verify `record` copy + flow against wireframe A3 | 1.1 | [stories/1.6-record-verify.md](stories/1.6-record-verify.md) |
| 1.7 | Verify `ask` copy + flow against wireframe A5 | 1.1 | [stories/1.7-ask-verify.md](stories/1.7-ask-verify.md) |
| 1.8 | `chirp search` keyword rewrite over transcripts + notes (A7) | 1.1, 1.3, 1.4 | [stories/1.8-search-keyword-rewrite.md](stories/1.8-search-keyword-rewrite.md) |

## 6. Epic-level acceptance criteria

- `uv run chirp --help` shows exactly 7 commands (`record`, `transcribe`, `notes`, `ask`, `search`, `init`, `about`) under one panel titled "Commands".
- `uv run chirp notes` lists notes; `chirp notes view 1` opens read-only; `chirp notes edit 1` opens editable; `chirp notes delete 1` prompts and removes the whole `<slug>/` folder.
- After `chirp record`, `ls ~/Documents/chirp/<slug>/` shows `audio.wav` and `meta.toml`.
- After `chirp transcribe`, that same folder additionally contains `transcript.txt` and `notes.md`; `meta.toml` is updated with `indexed_at`, `whisper_model`, `llm_model`.
- `cat ~/.chirp/config.toml` is valid TOML; `~/.chirp/chroma/` exists after `chirp init`.
- `uv run chirp transcribe` with 3 untranscribed notes prints `1 of 3 · …` header, the 6-step checklist per note, and completes sequentially.
- `uv run chirp transcribe 2` with 3 untranscribed processes the oldest 2 only.
- `uv run chirp transcribe --force` re-runs all 6 stages on an already-finished note.
- `uv run chirp init --recheck` shows the verify phase only and does not install anything.
- `uv run chirp about` renders the 3-frame animation unchanged.
- `uv run pytest` passes the full suite; `uv run ruff check .` reports no issues.

## 7. Out of scope / deferred

- `chirp notes --tag meeting` (wireframe hint): defer until a tag system exists.
- Stage pipelining across notes in `transcribe` (note B in Whisper while note A is in Ollama): revisit only if profiling shows idle GPU time.
- Language auto-detect in `transcribe`: hard-coded `en` for now; expose `--language` when we support more.
- Migration script from the old `~/Documents/Chirp/{recordings,transcripts,notes}/` layout: explicitly not needed (WIP, no back-compat).

## 8. Risks

- **Dependency cascade from storage rewrite (1.1):** every downstream module resolves paths through `config/settings.py` today. Mitigation: land 1.1 behind a single commit with a complete test-suite pass before starting 1.2+.
- **Whisper output loss:** current code writes compressed `.json.gz` with segment-level timestamps. The new `transcript.txt` drops timestamps. Confirm with user that plain text is the accepted trade-off (wireframe implies yes). If timestamps are needed later, add `transcript.srt` in the same folder without changing the contract.
- **Init Phase 4 config path change:** moving from `user_config_dir/config.yaml` to `~/.chirp/config.toml` breaks any existing user's config. Acceptable per the no-back-compat directive.
