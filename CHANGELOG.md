# Changelog

All notable changes to Chirp are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions track the
releases published to PyPI as `chirp-notes-ai`.

## [Unreleased]

### Added

- `CHANGELOG.md`, PyPI classifiers, license metadata, and project URLs.
- Model downloads and daemon loads are pinned to the HuggingFace commit SHA
  captured at `chirp models add` time (`revision` field in `models.toml`);
  `chirp models pull` re-pins to the current upstream head.
- Integration smoke test in CI: the built wheel is installed into a clean
  environment and exercised via `chirp --version` and CLI startup checks.

### Changed

- All user-data writes (`transcript.txt`, `notes.md`, `meta.toml`,
  `config.toml`, search-index manifests) are atomic: a crash or full disk can
  no longer leave silently truncated notes or transcripts.
- `~/.chirp` and newly created notes roots are owner-only (`0700`) since they
  hold verbatim meeting content.
- launchd's captured stdout/stderr moved to `chirpd.launchd.log` so rotation
  of `chirpd.log` no longer loses daemon output or bypasses the size cap.
- Renovate now auto-merges only dev-tooling and patch-level updates; runtime
  dependency minor/major bumps require review.
- CI tests run on Python 3.11, 3.12, and 3.13.
- Publishing to PyPI now fails hard when the Developer ID signing certificate
  is unavailable instead of silently shipping an ad-hoc-signed helper.

### Fixed

- Errors during `chirp record --live-transcribe` (for example an audio device
  disappearing) now surface as friendly messages instead of raw tracebacks.

## [0.0.5-alpha] — 2026-07-01

### Added

- Chunked (map-reduce) summarization for long meetings, replacing
  single-prompt note generation that degraded past the model context window.

### Fixed

- Note-generation resilience, first-run model install, and daemon freshness
  checks.

## [0.0.4-alpha] — 2026-06-30

### Added

- First-run setup: register, download, and warm a chat model during
  `chirp init`.

### Changed

- Replaced PyAudio with sounddevice (PortAudio ships via pip; no brew
  dependency).
- Decode WAVs natively, dropping the ffmpeg dependency.
- Polished the transcribe pipeline UI and quieted HuggingFace log noise.

### Fixed

- Use the OS trust store for HTTPS so corporate TLS proxies work.
- Stopped an MLX memory leak during transcription.

## [0.0.3-alpha] — 2026-06-29

### Added

- Lexical-first retrieval: BM25 is the out-of-the-box index; semantic search
  became opt-in via `chirp config --semantic`.
- mlx-whisper transcription engine (Metal-accelerated on Apple Silicon).

### Changed

- Production hardening (epic 8): platform-tagged wheel, hybrid retrieval
  fusion, daemon/client timeout hardening, CLI conventions (`--version`,
  `--json`, stdout/stderr split), tolerant config loading, and a CI coverage
  floor.

## Earlier releases

Versions before 0.0.3-alpha were internal alphas that established the core
record → transcribe → notes pipeline, the chirpd daemon (MLX inference over a
local socket), the model registry, and the guided `chirp init` flow.
