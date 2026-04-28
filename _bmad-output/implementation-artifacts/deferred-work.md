# Deferred Work

## Deferred from: code review of 1.1-storage-rewrite (2026-04-20)

- Live session writes `transcript.live.txt` instead of canonical `transcript.txt`. Intentional — live transcript is a watch-along preview; canonical transcript comes from `chirp transcribe`. Revisit if users expect live-only recordings to flow into notes/index. [recorder/live_session.py]
- ~~`list_notes` uncaught on PermissionError / `notes_root` being a file not a dir~~ — addressed 2026-04-27: `iterdir()` is wrapped and returns `[]` on `PermissionError` / `NotADirectoryError` / `OSError`. [utils/file_utils.py]
- ~~`BatchProcessor.process_directory(directory)` parameter untyped~~ — already addressed in a prior commit; signature is `directory: Path`. [transcriber/batch_processor.py]
- ~~Unicode/punctuation-only titles collapse to bare `note` base before the date suffix~~ — addressed 2026-04-27: `_kebab_case` now NFKD-folds to ASCII so accented titles (e.g. `Café résumé` → `cafe-resume`) survive; truly non-alphanumeric titles still fall back to `note` and rely on the collision counter. [utils/file_utils.py:_kebab_case]
- Missing end-to-end integration test from spec's Testing section: record→transcribe→generate asserting `audio.wav`, `transcript.txt`, `notes.md`, `meta.toml` and `whisper_model`/`llm_model`/`indexed_at` keys. [tests/]
- AC-3 partial: `notes_chat.index_dir` default is `~/.chirp`, and the chroma dir resolves as `index_dir/chroma`. Functionally correct but the spec wording suggests `index_dir` should itself point at `~/.chirp/chroma/`. Consider renaming `index_dir` to `chirp_home` or moving the `chroma/` suffix into the default. [config/settings.py]

## Deferred from: code review of 1.2-command-surface-prune (2026-04-27)

- `ManualNoteManager` / `ManualNoteEditor` are orphans in the active code path after `note` was removed. Tests exercise them in isolation, but no CLI command calls them. Expected to be revived by story 1.3's `notes view` / `notes edit`. [notes/manual_note_manager.py, notes/note_editor.py] — **resolved by 1.3:** `notes view` / `notes edit` now drive `ManualNoteEditor` via the `notes` sub-app.

## Deferred from: code review of 1.3-notes-sub-app (2026-04-27)

- `_drop_from_index` and `_reindex_after_edit` reach into `IndexManager._remove_from_index`, `_load_manifest`, `_save_manifest`, `_rebuild_bm25` (private API). Pre-existing pattern from the deleted `note` command; expose a public `IndexManager.delete(path)` / `add(path)` and have CLI use those. [chirp/cli.py, notes_chat/index.py]
- No happy-path test for `notes view` / `notes edit` because they invoke the interactive `ManualNoteEditor` (curses/TTY). Acceptable gap; resolution paths are tested directly via `_resolve_note`. [tests/test_cli_commands.py]

## Deferred from: code review of 1.4-transcribe-queue-checklist (2026-04-27)

- `IndexManager` private-method reach-around continues in stage 4 (`_add_to_index`, `_load_manifest`, `_save_manifest`, `_rebuild_bm25`). Same root cause flagged in 1.3 — fix once, fix in both call sites. [transcriber/batch_processor.py:_stage_index]
- `WhisperTranscriber._read_audio_metadata` is a private call from stage 1. Promote it (or extract `utils/audio_meta.py`) so callers don't depend on Whisper internals. [transcriber/batch_processor.py:_stage_load_audio]
- No KeyboardInterrupt handling in `run_queue` — Ctrl-C mid-batch skips the popup notification and the `done · N ok · M failed` summary. Acceptable for now; revisit if users complain. [transcriber/batch_processor.py:run_queue]

## Deferred from: code review of 1.5-init-polish (2026-04-27)

- No test for the "garbage input retries" branch in `_prompt_blackhole_routing`. Feed `["x", "s"]` to assert the loop prints `please type o or s` then exits cleanly. [chirp/init_flow.py:_prompt_blackhole_routing]
- `_ollama_models` is called in both `verify` (Phase 1) and `keep_or_pick` (Phase 3) — two redundant `requests.get` calls per healthy `init` run. Cache the list in Phase 1 and pass it down. [chirp/init_flow.py]
- Path display in `_print_path_summary` uses absolute paths (`/Users/<you>/.chirp/…`) where the wireframe shows `~/.chirp/…`. Cosmetic; revisit when polishing the finalize panel. [chirp/init_flow.py:_path_line]

## Deferred from: code review of 1.6-record-verify (2026-04-27)

- AC-11 integration test "discard leaves no `<slug>/` folder" not added — requires non-trivial PyAudio mocking to exercise the `x`-key + recorder path end-to-end. The discard rendering branch is unit-tested via `_RecordViewState`. [tests/test_record_view.py, tests/test_cli_commands.py]
- `stopped_by_cap` rendering branch never set — when the timer fires the `Live` loop exits before any tick can render `■ stopped`. Either remove the dead branch or have the timer flip the flag through one final tick. [chirp/cli.py:_render_record_view]
- Discard path writes `audio.wav` to disk and immediately `rmtree`s the slug folder. Wasteful but correct. Adding a "skip save" signal to `AudioRecorder` is invasive; defer until a real user complains. [chirp/cli.py:record, recorder/audio_recorder.py]
