# Epic: Production hardening — close the v0.1.0 readiness gaps

- **Epic ID:** EPIC-PRODUCTION-HARDENING
- **Owner:** Colby
- **Status:** In-progress — stories 8.1–8.6 created from the 2026-06-14 production-readiness audit. Closes the packaging blocker, the RAG correctness/scale gaps, the daemon/client reliability gaps, the audio/transcription reliability gaps, the CLI/TUI best-practice gaps, and the config/CI quality-gate gaps that stand between the current `0.0.1a0` alpha and a defensible `0.1.0`.
- **Created:** 2026-06-14
- **Design source:** Production-readiness audit (2026-06-14), conducted across all nine runtime packages + tests/CI/packaging. Two headline RAG bugs verified directly in source. References [`prd.md`](../prd.md), [`architecture.md`](../architecture.md), and the `.docs/` design notes (`hybrid-retrieval.md`, `embeddings.md`, `chunking.md`).
- **Related branch (current work):** TBD

## 1. Goal

After this epic, `pip install chirp-notes-ai` installs only where it can run, fails loudly off-platform at install time (not at runtime), and the published artifacts are buildable from source. The headline `chirp ask` retrieval feature delivers what its design docs promise (true hybrid fusion + correct time filtering) and scales past a demo-sized corpus. The chirpd daemon and its client never hang, never evict warm models on cosmetic releases, and degrade predictably under memory pressure. Audio capture stops leaking the resource that re-triggers macOS permission prompts, and transcription surfaces actionable errors instead of generic crashes. The user-facing commands meet baseline CLI conventions (stream separation, `--json`, `--version`, signal handling, color contract) already established by the newer `llm/cli` subsystem. Malformed config can't brick the tool, and CI actually enforces the quality bar it measures.

## 2. Why now

The app is an honestly-versioned, high-quality alpha (`0.0.1a0`) whose hard parts — the ScreenCaptureKit/AVAudioEngine Swift helper, the daemon IPC + cancellation, the local-first privacy posture — are its strongest. The remaining work is targeted hardening of the glue layer plus fixing two verified RAG correctness bugs. The single packaging Critical (a `py3-none-any` wheel bundling an arm64-only Mach-O binary) must be fixed before any real PyPI release because it mis-ships to every platform. The rest are the must-fix reliability and UX debts that separate "works on the maintainer's machine" from "production."

## 3. Findings → story map

All audit findings are owned by exactly one story. Severities: CRIT/HIGH/MED/LOW.

| Story | Title | Findings covered |
|---|---|---|
| 8.1 | Packaging & distribution correctness | CRIT wheel platform tag; CRIT sdist source-build (Swift in build hook); wheel-tag regression test; drop unused `requests` direct dep; bundled-binary reproducibility/identity |
| 8.2 | Retrieval & note-generation correctness and scale | HIGH hybrid fusion (RRF/normalization); HIGH time-range date extraction + tz consistency; MED query-time scaling (BM25 reload/rebuild, full stat sweep); MED cache keying + eviction; MED atomic `--force` rebuild; MED manifest/chroma drift on partial failure; MED embedding-dimension fingerprint guard; MED transcript-size cap + prompt-injection hardening; MED XML-parse fallback junk-note handling |
| 8.3 | chirpd daemon & LLM client reliability | HIGH socket read/inference timeout; HIGH PROTOCOL_VERSION vs package version; HIGH socket-isolation lock keyed to socket path; HIGH cancellation request-id trust; MED memory-pressure backpressure + embed idle policy; MED idle-unload vs long-generation race; MED late-cancel graceful classification; LOW override-socket dir perms |
| 8.4 | Audio capture & transcription reliability | HIGH PyAudio/DeviceManager leak + dead-code removal; MED Whisper load/download typed error + actionable message + load-before-capture ordering; MED live-transcript finalize race (join timeout / lock-guarded export); MED queue-full drop surfacing; LOW explicit Whisper model teardown |
| 8.5 | CLI/TUI best-practices conformance | HIGH stdout/stderr split back-port; HIGH `--json` for `notes`/`ask`; HIGH `--version` flag; HIGH uniform Ctrl-C/EOF prompt-escape; MED `NO_COLOR`/`--no-color` contract + editor `force_terminal` fix; MED unified exit-code table; MED SIGTERM/SIGWINCH handling + terminal restore; MED editor in-app help + `$EDITOR`/`$VISUAL`; LOW glyph/voice consistency + completion reconsideration; LOW AppleScript escaping in popup path |
| 8.6 | Config resilience & CI/release quality gates | HIGH tolerant config load (malformed value can't brick CLI) + config schema version; HIGH coverage floor (`--cov-fail-under`); MED align `--cov` package sets across workflows; MED mypy in shared build + tighten; HIGH/MED raise `chirp/cli.py` coverage; LOW reconsider Renovate automerge given untested inference path; LOW remove stale `.coverage`/`htmlcov` artifacts |

## 4. Sequencing & dependencies

- **8.1 lands first.** It is the only hard release blocker; nothing else can ship to PyPI safely without it.
- **8.6 should land early** alongside the others — the coverage floor and tolerant config load are guardrails that protect the other five stories' changes.
- **8.2, 8.3, 8.4, 8.5 are largely independent** and can be implemented in parallel; they touch disjoint subsystems (`notes_chat`+`notes`, `chirpd`+`llm`, `recorder`+`transcriber`+`audio_capture`, `chirp`+`utils`+`llm/cli` respectively). Where a finding touches a shared seam (e.g. the daemon read timeout in 8.3 benefits note-gen/ask/index), the owning story is the one that owns the seam.
- **8.5's new tests feed 8.6's coverage floor** — set the `--cov-fail-under` threshold after 8.5 raises `chirp/cli.py` coverage, or set it conservatively first and ratchet.

## 5. Definition of done (epic)

- `pip install chirp-notes-ai` refused by pip off-platform; sdist builds the Swift helper from source.
- `chirp ask` retrieval uses RRF (or normalized fusion) and correct meeting-date filtering, with un-mocked tests for both.
- No code path can hang the CLI on a stalled daemon; cosmetic version bumps do not evict warm models.
- `chirp record` no longer leaks the PyAudio handle; Whisper load failures produce actionable messages.
- `record`/`transcribe`/`notes`/`ask`/`search` separate stdout/stderr, support `--json` where it matters, and the app has `--version`.
- A malformed `~/.chirp/config.toml` value cannot crash every command; CI fails on coverage regressions.
- `make check` and `make test` pass; new behavior is covered by tests.
