# Epic: Sam can keep the daemon alive across reboots and diagnose issues without reading source code

- **Epic ID:** EPIC-DAEMON-LIFECYCLE
- **Owner:** Colby
- **Status:** Draft
- **Created:** 2026-05-15
- **Design source:** `_bmad-output/planning-artifacts/prd.md` (FR39–FR45, NFR-O1/O2/O3, NFR-R2/R3); `_bmad-output/planning-artifacts/architecture.md` § Daemon Lifecycle Integration with launchd, § Implementation Patterns → Logging Discipline
- **Related branch (current work):** TBD

## 1. Goal

After this epic, the troubleshooter user (Sam, per PRD §User Journeys) can:

- **Survive login and crashes without thinking about it.** Run `chirp daemon enable` once, reboot the Mac, and have `chirp ask` hit a warm daemon on the very next invocation. If `chirpd` crashes mid-day, launchd restarts it before the next CLI command notices.
- **Look inside the daemon when something feels off.** Run `chirp daemon status` and see the daemon's PID, uptime, version, loaded models with per-model RSS, and the idle countdown on each model — enough to triage "is it running?" / "is the model still warm?" / "is RAM leaking?" without reading source code.
- **Tail the log.** Run `chirp daemon logs -f` and watch logfmt lines stream as requests land — operation, model alias, duration, error code. Sensitive content (prompts, chat messages, note text) never appears.
- **Manually control the process when needed.** `chirp daemon start | stop | restart` cover the rare cases where Sam wants explicit lifecycle control (after editing `models.toml`, after a `pip install -U`, when debugging).
- **Roll the LaunchAgent back without artifacts.** `chirp daemon disable` removes the plist cleanly and the daemon stops auto-starting at login.

The epic is gated on `chirp daemon status` reporting accurate truth (PID, uptime, loaded models, idle countdowns), `chirp daemon enable` producing a LaunchAgent that survives logout/login on a real Mac, and `chirp daemon logs` working with the same UX a developer expects from `tail -f` — including across the rotation boundary.

What this epic does **not** ship: the daemon itself, model load/unload behavior, the `chirp models` subcommand group, or the `chirp init` LaunchAgent prompt. Those live in EPIC-CHIRPD-CORE, EPIC-MODEL-REGISTRY, and EPIC-INIT-AND-MIGRATION respectively. This epic builds on top of CHIRPD-CORE's `model.status` and `health` ops and exposes them in a CLI surface tuned for diagnosis and lifecycle control.

## 2. Why now

The PRD's whole "users never need to install or manage a separate daemon" pitch only holds if **when** they do need to manage the daemon — for diagnosis or for opt-in auto-start — chirp gives them honest, terminal-native tooling. Three concrete forces pull this epic into the same release as CHIRPD-CORE:

1. **Auto-start at login is part of the "no Ollama" parity story.** Ollama runs via Homebrew services and is up at login. If chirp's MLX daemon required an explicit `chirp daemon start` the first time the user runs `chirp ask` after a reboot, the perceived UX would regress against the very thing the PRD set out to improve. `chirp daemon enable` + a LaunchAgent with `KeepAlive` closes that gap.
2. **Diagnostics are the only escape valve users have when models misbehave.** The PRD's NFR-M5 ("diagnostics surface adequate to triage daemon-won't-start, model-won't-load, version-mismatch, idle-unload-not-firing without reading source code") is satisfied entirely by the commands in this epic. There is no other introspection surface in MVP.
3. **Sequencing.** CHIRPD-CORE exposes the `model.status` and `health` ops this epic consumes. INTEGRATION-CUTOVER will route `notes` and `notes_chat` through the daemon — at which point any latent lifecycle bug becomes a user-visible bug rather than a developer-only annoyance. Shipping the diagnostic surface before cutover means we have something to point users at when the inevitable first-week issues land.

This epic can develop in parallel with EPIC-MODEL-REGISTRY (independent paths — the registry CLI works without LaunchAgent integration; this epic works without the registry CLI). It depends on EPIC-CHIRPD-CORE only for the daemon's `model.status` / `health` op contract and the `llm.client` library.

## 3. Locked decisions from the architecture

| # | Decision | Source |
|---|----------|--------|
| 1 | Daemon logs are logfmt-style key=value lines (`ts=... level=... component=chirpd op=chat req_id=r-... model=<alias> duration_ms=...`). One line per event. Greppable with `awk` / `grep`; parseable with any logfmt library. | Architecture § Implementation Patterns → Logging Discipline; PRD OQ3 resolution. |
| 2 | Log file path: `~/Library/Logs/chirp/chirpd.log`. Rotation at ~10 MB via `logging.handlers.RotatingFileHandler`; one prior generation retained (`chirpd.log.1`). | PRD FR16, NFR-O2; architecture § Project Structure → Runtime File Layout. |
| 3 | Log redaction is a hard constraint (NFR-S5). No user prompts, chat messages, note content, embed input text, or transcript text ever appear in logs. Logged metadata only: op name, model alias, request id, token counts, durations, error class, error message (which must itself be redacted of user content). | Architecture § Logging Discipline → Forbidden in logs; PRD NFR-S5. |
| 4 | LaunchAgent plist generated via Python's stdlib `plistlib`. Plist installed at `~/Library/LaunchAgents/com.chirp.chirpd.plist`. `launchctl load <plist>` and `launchctl unload <plist>` run via `subprocess`. Verification probe: `launchctl list \| grep com.chirp.chirpd`. | Architecture § Daemon Lifecycle Integration with launchd; PRD FR43/FR44. |
| 5 | `ProgramArguments` in the plist is the absolute path to the `chirpd` console script resolved at `chirp daemon enable` time via `shutil.which("chirpd")`. The user re-runs `enable` if they relocate their Python environment. | Architecture § Daemon Lifecycle Integration with launchd. |
| 6 | LaunchAgent `KeepAlive` policy: `<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>`. launchd restarts the daemon on crash (non-zero exit) but **not** on intentional clean exit. The version-mismatch immediate-exit (exit 0) is intentional — launchd will not respawn it. The next CLI invocation lazy-spawns the new version. | Architecture § Daemon Lifecycle Integration with launchd; PRD NFR-R2, NFR-R5. |
| 7 | Daemon never auto-exits on idle. Only **models** unload on idle; the process stays up indefinitely. `chirp daemon status` reports both daemon uptime (process lifetime) and per-model idle countdowns (model lifetime). | Architecture § Process & Concurrency Model; PRD FR4, FR39. |
| 8 | `chirp daemon stop` does not drain in-flight requests (alpha simplification). The daemon exits promptly via a shutdown op, and any in-flight chat fails with `MODEL_GENERATION_FAILED` cause `daemon_shutdown`. The CLI surfaces this clearly. | PRD §Alpha-stage simplifications; PRD NFR-R6; PRD FR41. |
| 9 | `chirp daemon` is a **hidden** Typer subcommand group — not visible in `chirp --help`, registered via `app.add_typer(..., hidden=True)`. Joins the existing hidden tier alongside `config`, `devices`, `index`. The visible 7-command surface (`record`, `transcribe`, `notes`, `ask`, `search`, `init`, `about`) is unchanged. | PRD §CLI Tool Specific Requirements; CLAUDE.md hidden-command list. |
| 10 | `chirp daemon status --json` schema: `{"running": bool, "pid": int, "uptime_seconds": float, "version": str, "loaded_models": [{"alias": str, "role": "chat"\|"embed", "rss_bytes": int, "last_used": iso-8601, "idle_countdown_seconds": float\|null}], "last_request_at": iso-8601\|null, "total_rss_bytes": int}`. Same fields rendered as a Rich table on a TTY. | PRD FR39, FR31, NFR-O3. |

## 4. Research findings — what already exists vs. what is missing

Validated against the current `main` branch and the architecture document.

### Already exists (referenced, not modified)

- **Hidden-command Typer pattern.** `chirp/cli.py` already registers hidden maintenance commands (`config`, `devices`, `index`) — the daemon group joins this tier using the same `hidden=True` mechanism. Locked decision 9.
- **Rich console rendering and `--json` flag conventions.** `chirp/init_flow.py`, `recorder/live_dashboard.py`, and the existing `chirp ask --json` shape establish how to switch between Rich rendering on a TTY and pure JSON on stdout. The `daemon status` command follows the same pattern.
- **Domain-exception → exit-code mapping.** `chirp/exceptions.py` plus the existing Typer outer wrapper convert typed errors into the exit codes documented in PRD §Scripting Support. The daemon CLI raises `LLMTransportError` / `LLMModelError` from `llm.exceptions` (delivered by EPIC-CHIRPD-CORE) and lets the wrapper produce exit codes 3/4/5.
- **Swift `.app` bundle pattern from EPIC-AUDIO-CAPTURE.** Referenced for the macOS permissions parallel — that work taught the codebase how to bundle, install, and verify a macOS helper. We borrow none of the code for this epic; the LaunchAgent install path is purely Python + `launchctl`. The pattern is the precedent, not the implementation.

### Built by EPIC-CHIRPD-CORE; consumed here

- **`llm.client.LLMClient`.** Used by `chirp daemon status` to issue `health` and `model.status` ops, and by `chirp daemon stop` to issue the shutdown op. CHIRPD-CORE delivers the connect-with-lazy-spawn-and-retry surface; this epic uses it.
- **`chirpd` console script entry.** `pyproject.toml`'s `[project.scripts]` table provides `chirpd` on PATH (added in CHIRPD-CORE). `chirp daemon enable` resolves this via `shutil.which("chirpd")`; `chirp daemon start` invokes it via `subprocess.Popen`.
- **Daemon ops `health` and `model.status`.** CHIRPD-CORE's story for these ops exposes:
  - `health` returns `{"status": "ok", "version": "<str>", "uptime_seconds": <float>, "pid": <int>}`.
  - `model.status` returns the loaded-models payload defined in locked decision 10 above. **This epic depends on `model.status` exposing per-model RSS** — a contract that must be validated against CHIRPD-CORE's implementation when both land.
- **Basic logfmt formatter.** CHIRPD-CORE delivers a minimal `chirpd/logging_setup.py` for dev-mode stderr logging. This epic extends it with production paths, rotation, and the redaction-discipline test coverage.

### Missing — built by this epic

- **`chirpd/logging_setup.py` rotation + production config.** Today CHIRPD-CORE has logfmt-to-stderr only. This epic adds the rotating file handler, log-directory creation under `~/Library/Logs/chirp/`, and the logfmt-helper functions used by callers throughout `chirpd/`. Story 5.1.
- **`llm/cli/daemon.py` — the entire Typer subapp.** No `chirp daemon` command exists today. This epic introduces the subapp file, the seven subcommands, and the Typer registration in `chirp/cli.py`. Stories 5.2 / 5.3 / 5.4 / 5.5 / 5.6.
- **`chirpd/launchd.py` — LaunchAgent install/uninstall.** No LaunchAgent infrastructure exists in the codebase. This epic adds the plist generator, the `launchctl` wrappers, and the verification probe. **Reusable functions** — EPIC-INIT-AND-MIGRATION will call `install_launch_agent()` from `chirp init`'s opt-in prompt; factor cleanly so the CLI subcommand is one consumer, not the only consumer. Story 5.4.
- **Log-tailing UX for chirp.** No existing chirp command tails a file. `chirp daemon logs` builds the `cat` / `-n` / `-f` semantics from scratch. Story 5.5.
- **Hidden-tier subapp registration.** `chirp/cli.py` needs one new `app.add_typer(daemon_app, name="daemon", hidden=True)` call. Story 5.6.

### Net code delta (rough)

- **Add:** ~600 lines Python across `chirpd/logging_setup.py` (extensions), `chirpd/launchd.py` (new), `llm/cli/daemon.py` (new), plus ~400 lines of tests across `tests/chirpd/test_logging_setup.py`, `tests/chirpd/test_launchd.py` (unit-only — `launchctl` subprocess seam mocked), and `tests/llm/test_cli_daemon.py`.
- **Modify:** ~10 lines in `chirp/cli.py` (the Typer registration).
- **Untouched:** the daemon process itself, the `model.status` / `health` op implementations, the `llm.client` library, the model registry, `notes`, `notes_chat`, `chirp init`.

## 5. Stories

Execution order matters: logging lands first because every other story emits log lines (and we want the redaction tests + rotation in place from the start). The `status` command lands next as the standalone diagnostic surface. `start`/`stop`/`restart` are grouped — they share a Typer subapp file. `enable`/`disable` land together for the same reason and depend only on the daemon entrypoint being on PATH. `logs` lands after the file path is locked by story 5.1. The Typer registration (5.6) is intentionally last so that the hidden command group lights up in `chirp --help` only when every subcommand is real.

| ID | Title | Depends on | File |
|----|-------|------------|------|
| 5.1 | Logfmt logging setup + rotation in `chirpd/logging_setup.py` | EPIC-CHIRPD-CORE (basic logfmt formatter exists) | [stories/5.1-logfmt-logging-and-rotation.md](stories/5.1-logfmt-logging-and-rotation.md) |
| 5.2 | `chirp daemon status` (with `--json`) — diagnostic snapshot | 5.1; EPIC-CHIRPD-CORE (`health` + `model.status` ops, `llm.client`) | [stories/5.2-daemon-status.md](stories/5.2-daemon-status.md) |
| 5.3 | `chirp daemon start \| stop \| restart` — manual lifecycle controls | 5.2 (Typer subapp file exists); EPIC-CHIRPD-CORE (shutdown op, lazy-spawn) | [stories/5.3-daemon-start-stop-restart.md](stories/5.3-daemon-start-stop-restart.md) |
| 5.4 | `chirp daemon enable \| disable` — LaunchAgent install/uninstall via `chirpd/launchd.py` | 5.3 (subapp pattern); CHIRPD-CORE (`chirpd` console script on PATH) | [stories/5.4-daemon-enable-disable.md](stories/5.4-daemon-enable-disable.md) |
| 5.5 | `chirp daemon logs` (with `-f` / `--follow`, `-n`) | 5.1 (log file path locked) | [stories/5.5-daemon-logs.md](stories/5.5-daemon-logs.md) |
| 5.6 | Hidden Typer registration in `chirp/cli.py` | 5.2, 5.3, 5.4, 5.5 (all subcommands real) | [stories/5.6-typer-registration.md](stories/5.6-typer-registration.md) |

## 6. Sequencing & dependencies

**Upstream:** EPIC-CHIRPD-CORE must land before any story in this epic ships. Specifically, story 5.2 depends on the daemon's `health` and `model.status` op contract; stories 5.3 and 5.4 depend on the `chirpd` console script being installed by `pip install`; all stories depend on the `llm.client` library.

**Sibling:** EPIC-MODEL-REGISTRY can develop in parallel — there is no code overlap. The two epics share the `llm/` package directory but touch different files (`llm/cli/models.py` vs. `llm/cli/daemon.py`).

**Downstream:** This epic does **not block** EPIC-INTEGRATION-CUTOVER. The cutover work routes `notes` and `notes_chat` through `llm.client`, which lazy-spawns `chirpd` if no LaunchAgent is installed. The LaunchAgent is opt-in convenience, not a precondition. Diagnostic commands are similarly nice-to-have for cutover but don't gate it.

EPIC-INIT-AND-MIGRATION consumes `chirpd/launchd.install_launch_agent()` from story 5.4 (when `chirp init` offers LaunchAgent installation). Story 5.4 must factor the install logic into a reusable function, not solely as a Typer command body. This is enforced by story 5.4's AC and called out in §Out of scope.

## 7. Success criteria

- **Sam's "Monday morning" scenario.** Run `chirp daemon enable` on a Friday. Close the laptop. On Monday morning, reboot. Run `chirp ask "what did I record last Thursday?"` without first running anything else. Expectation: the LaunchAgent has spawned the daemon at login; `chirp ask` connects to a daemon with uptime > 0 and (per CHIRPD-CORE) lazy-loads the model. First-token latency lines up with NFR-P2 cold-path (≤ 5 s for a 4-bit 7B on M2). No explicit daemon command needed between reboot and `chirp ask`.
- **`chirp daemon status` accuracy.** On a developer machine: run `chirp daemon status` with no model loaded → output reports `running: yes`, `loaded_models: []`, `total_rss_bytes` ≤ 150 MB (NFR-resource budget). Then run `chirp ask "hello"` to warm a chat model. Re-run `chirp daemon status` → output reports the model alias, `role: chat`, `rss_bytes` within 10% of the model's on-disk size, and `idle_countdown_seconds` ~ 300 (default 5-minute timer, FR5). Re-run after 1 minute → countdown ~ 240. `chirp daemon status --json` emits valid JSON matching locked decision 10's schema (parseable with `jq`).
- **`chirp daemon logs -f` UX.** With the daemon idle, run `chirp daemon logs -f` in one terminal. In another, run `chirp ask "hello"`. The first terminal streams the chat-op logfmt line (with `op=chat`, `model=<alias>`, `duration_ms=<int>`) within 1 second of the request landing. Hit Ctrl-C → tail exits cleanly with exit 0.
- **`chirp daemon enable` round-trip.** On a fresh checkout: run `chirp daemon enable` → command reports success; `~/Library/LaunchAgents/com.chirp.chirpd.plist` exists; `launchctl list | grep com.chirp.chirpd` shows the agent loaded. Reboot. After login, before running any chirp command: `launchctl list | grep com.chirp.chirpd` still shows it; `~/Library/Application Support/chirp/chirpd.sock` exists (daemon spawned at login). Run `chirp daemon disable` → command reports success; the plist is gone; `launchctl list | grep com.chirp.chirpd` returns no match.
- **Log redaction holds under adversarial input.** A test sends a `chat` request whose `messages` payload contains a known token (e.g. `"REDACTION_CANARY_42"`). After the request completes, `grep -F REDACTION_CANARY_42 ~/Library/Logs/chirp/chirpd.log*` returns zero matches. Same for embed input text and note slugs. (Implemented as a test in story 5.1; rerun in CI.)
- **Rotation works.** Write 12 MB of log lines synthetically (test harness, not real traffic). Verify `chirpd.log` rotates to `chirpd.log.1` and a fresh `chirpd.log` begins. No prior generations beyond `.1` are retained.
- **`chirp daemon stop` semantics with the version handshake.** With the daemon running and a chat request in-flight, run `chirp daemon stop` from another terminal. The chat request fails with a typed `LLMModelError` whose `code` is `MODEL_GENERATION_FAILED` and `details.cause` is `daemon_shutdown`. The user-facing error message reads cleanly (not a stack trace). The daemon exits within 2 seconds.
- **`make check` and `make test`** pass with the new modules. Coverage on `chirpd/logging_setup.py`, `chirpd/launchd.py`, and `llm/cli/daemon.py` is ≥ 90% line (per NFR-M1). The LaunchAgent subprocess seam (`launchctl` invocation) is mocked in unit tests; manual install/uninstall is verified per story 5.4's manual testing checklist.
- **Hidden command group is invisible by default.** `uv run chirp --help` does not list `daemon`. `uv run chirp daemon --help` lists all seven subcommands.

## 8. Out of scope / deferred

- **Daemon process and inference.** `chirpd` itself, its asyncio server, the `model.load` / `model.unload` / `chat` / `embed` / `cancel` ops, and the per-model state machine — EPIC-CHIRPD-CORE.
- **Model registry CLI.** `chirp models add | list | remove | default | pull | show` — EPIC-MODEL-REGISTRY. Daemon status reports loaded models by alias; how aliases come to be registered is the other epic's concern.
- **Existing-module cutover.** Routing `notes`, `notes_chat`, `chirp ask`, `chirp search` through `llm.client` — EPIC-INTEGRATION-CUTOVER. This epic adds the diagnostic commands; the wiring is elsewhere.
- **`chirp init` LaunchAgent prompt.** `chirp init` offering LaunchAgent installation as part of first-run is EPIC-INIT-AND-MIGRATION's job. That epic imports `install_launch_agent()` from `chirpd/launchd.py` (this epic's deliverable, story 5.4); the prompt UX itself belongs there, not here.
- **Graceful drain on `chirp daemon stop`.** PRD §Alpha-stage simplifications removes this. `stop` is prompt; in-flight requests fail with a typed error. If a future user actually wants drain semantics, revisit post-MVP.
- **Hot-reload of `models.toml`.** Listed in PRD §Out of Scope. Editing `models.toml` requires `chirp daemon restart`.
- **JSON log output.** PRD OQ3 resolution defers a `chirp daemon logs --json` flag to a Growth-tier story. This epic emits logfmt only; `daemon logs` cats the file as-is.
- **Telemetry, crash reporting, update checks.** Hard constraint per PRD NFR-S3/S4. Logs stay local; no analytics anywhere.
- **Multi-user / multi-daemon.** Single user, single host, single daemon. `chirp daemon enable` for the current user only; no system-wide LaunchDaemon variant.
- **Apple Silicon enforcement.** Done elsewhere — `chirp init` and `chirpd/__main__.py` perform the `platform.machine() == "arm64"` check (CHIRPD-CORE and INIT-AND-MIGRATION). This epic's commands inherit that fail-fast behavior; no additional checks needed.

## 9. Risks

- **`model.status` per-model RSS contract drift between epics.** Story 5.2's `chirp daemon status` depends on CHIRPD-CORE's `model.status` op returning `rss_bytes` per loaded model. If CHIRPD-CORE ships without this field, `chirp daemon status` cannot satisfy NFR-O3. Mitigation: story 5.2's AC names the exact JSON shape it consumes; coordinate with CHIRPD-CORE's `model.status` story before that PR merges.
- **LaunchAgent install fails silently on macOS variants.** `launchctl load` returns non-zero on plist syntax errors, already-loaded labels, and missing files. Mitigation: story 5.4's AC requires capturing `launchctl` stderr and surfacing it to the user, plus running `launchctl list` afterward as a positive verification probe.
- **Version-mismatch + LaunchAgent interaction is counterintuitive.** The version-mismatch exit is `exit(0)`, which `KeepAlive { SuccessfulExit: false }` correctly treats as "do not respawn." Users may expect the LaunchAgent to immediately restart the daemon after a version-mismatch event. It does not — the next CLI invocation lazy-spawns the new version. Mitigation: story 5.4 documents this in the plist comment / dev notes; story 5.2's `chirp daemon status` works against both a launchd-spawned daemon and a lazy-spawned daemon so users can't tell the difference (and shouldn't need to).
- **`shutil.which("chirpd")` resolves to a different interpreter than expected.** A user with multiple Python environments could resolve `chirpd` to a stale path. Mitigation: story 5.4 documents that re-running `chirp daemon enable` refreshes the path, and `chirp daemon status` reports the resolved interpreter path under `--json` so users can spot the mismatch.
- **Log rotation race with `tail -f`.** The architecture uses `RotatingFileHandler`, which renames `chirpd.log` → `chirpd.log.1` and opens a new `chirpd.log`. A naive `tail -f` keeps reading the old inode and goes silent. Mitigation: story 5.5 uses `tail -F` semantics (re-open by name) rather than holding the original file descriptor; tests cover the rotation boundary explicitly.
- **30-day daemon survival not directly verifiable inside this epic.** NFR-R1's 30-day target requires real elapsed time. Mitigation: this epic delivers the *visibility* (RSS in `chirp daemon status`, rotation cadence in logs) needed to observe drift in production; the 30-day target is an observation gate, not a story-level AC.
- **Test-coverage of `launchctl` subprocess seam.** Per the project memory file `feedback_unit_test_mocking.md`, OS-touching code that requires real permissions is not unit-tested in this codebase. The `launchctl` boundary in `chirpd/launchd.py` is small and isolatable — story 5.4 mocks the subprocess call for unit tests and exercises the real path via the manual testing checklist. Same pattern as audio capture.
