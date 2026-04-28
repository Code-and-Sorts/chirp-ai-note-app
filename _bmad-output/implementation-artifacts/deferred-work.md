# Deferred Work

## Deferred from: code review of 1.1-storage-rewrite (2026-04-20)

- Live session writes `transcript.live.txt` instead of canonical `transcript.txt`. Intentional — live transcript is a watch-along preview; canonical transcript comes from `chirp transcribe`. Revisit if users expect live-only recordings to flow into notes/index. [recorder/live_session.py]
- ~~`list_notes` uncaught on PermissionError / `notes_root` being a file not a dir~~ — addressed 2026-04-27: `iterdir()` is wrapped and returns `[]` on `PermissionError` / `NotADirectoryError` / `OSError`. [utils/file_utils.py]
- ~~`BatchProcessor.process_directory(directory)` parameter untyped~~ — already addressed in a prior commit; signature is `directory: Path`. [transcriber/batch_processor.py]
- ~~Unicode/punctuation-only titles collapse to bare `note` base before the date suffix~~ — addressed 2026-04-27: `_kebab_case` now NFKD-folds to ASCII so accented titles (e.g. `Café résumé` → `cafe-resume`) survive; truly non-alphanumeric titles still fall back to `note` and rely on the collision counter. [utils/file_utils.py:_kebab_case]
- Missing end-to-end integration test from spec's Testing section: record→transcribe→generate asserting `audio.wav`, `transcript.txt`, `notes.md`, `meta.toml` and `whisper_model`/`llm_model`/`indexed_at` keys. [tests/]
- AC-3 partial: `notes_chat.index_dir` default is `~/.chirp`, and the chroma dir resolves as `index_dir/chroma`. Functionally correct but the spec wording suggests `index_dir` should itself point at `~/.chirp/chroma/`. Consider renaming `index_dir` to `chirp_home` or moving the `chroma/` suffix into the default. [config/settings.py]
