---
stepsCompleted:
  - step-01-init
  - step-02-context
  - step-03-starter
  - step-04-decisions
  - step-05-patterns
  - step-06-structure
  - step-07-validation
  - step-08-complete
lastStep: 8
status: 'complete'
completedAt: '2026-05-12'
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/epic-audio-capture/epic.md
  - _bmad-output/planning-artifacts/epic-wireframe-alignment/epic.md
  - AGENTS.md
  - CLAUDE.md
workflowType: 'architecture'
project_name: 'chirp-ai-note-app'
user_name: 'Colby'
date: '2026-05-12'
prd_under_design: '_bmad-output/planning-artifacts/prd.md'
---

# Architecture Decision Document

**Project:** chirp-ai-note-app
**Author:** Colby
**Date:** 2026-05-12
**PRD under design:** [`prd.md`](./prd.md) — Ollama → MLX migration (`chirpd` daemon)

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements** — 56 FRs across 8 capability areas in the PRD. Architecturally significant groupings:

- **Two-process system (FR1–FR23).** A long-lived inference daemon (`chirpd`) and many short-lived CLI client invocations. The daemon owns model resident state; clients are stateless aside from a socket connection per invocation.
- **Stateful resident model lifecycle (FR5–FR10).** Lazy load → idle-unload (default 5 min) → pin override (`keep_alive`). Embed-role models are always-resident, chat-role models churn. This is the core complexity in the daemon: a small state machine per model with timers and concurrency.
- **Version-tolerant client/daemon coupling (FR13, FR14, FR20).** The `hello` handshake plus immediate-exit-and-respawn plus transparent one-shot client retry are the architectural answer to "user upgrades chirp while daemon is running." (Alpha-simplified — no drain; see §Core Architectural Decisions.)
- **Single-instance enforcement (FR12).** Two CLI invocations racing to lazy-spawn must resolve to exactly one daemon. File-lock based.
- **Configuration & persistence on disk (FR37, FR38).** `models.toml` (registry), HuggingFace cache (weights), `~/Library/Logs/chirp/chirpd.log` (logs), `~/Library/Application Support/chirp/` (socket, lockfile, registry). No database.
- **Integration touch points (FR46–FR48).** Existing `notes`, `notes_chat`, `chirp ask`, `chirp search` call sites: replace Ollama client calls with `chirp.llm` client calls. Prompt assembly, retrieval, and templating stay in their current modules.

**Non-Functional Requirements** — 33 NFRs that materially shape architecture:

- **Performance budgets (NFR-P1–NFR-P7).** Warm first-token ≤ 500 ms, cold first-token ≤ 5 s, ≥ 30 tokens/sec, cancellation ≤ 200 ms. These force streaming-first IPC and constrain how much logic can sit between client and `mlx_lm.stream_generate()`.
- **Reliability budgets (NFR-R1–NFR-R7).** 30-day daemon stability, version-drift recovery ≤ 2 s, prompt shutdown on `stop` or mismatch (no drain — alpha). Forces explicit lifecycle state tracking but a simpler exit path.
- **Privacy / no-network (NFR-S1–NFR-S6).** Daemon emits zero network traffic. CLI emits HTTP only to `huggingface.co`, only on user-initiated `models add`/`pull`. No telemetry. Constrains every component to avoid "phone home" patterns.
- **Maintainability (NFR-M1–NFR-M5).** ≥ 90% coverage on new modules. Mypy clean. Reinforces small, well-bounded modules over clever abstraction layers.
- **Compatibility (NFR-C1–NFR-C4).** macOS 13+, Apple Silicon only. Lets the design assume `launchctl`, `flock`, unix sockets, and MLX without portability hedges.

### Scale & Complexity

- **Primary domain:** local-process IPC + on-device ML inference (Python).
- **Complexity level:** medium. Domain is single-user CLI (low), but the daemon-lifecycle + version-handshake + streaming-cancellation surface introduces real coordination complexity.
- **Estimated architectural components:** 5 — `chirpd` (daemon process), `chirp.llm` (client library), `chirp.models` (registry + `chirp models` CLI), `chirp.daemon` (lifecycle + `chirp daemon` CLI), integration shims inside existing `notes`/`notes_chat`.
- **No multi-tenancy, no auth, no real-time collaboration, no horizontal scaling.** Single user, single host, single daemon.
- **No database.** Registry is TOML on disk; logs are rotating files; weights are in HF cache.

### Technical Constraints & Dependencies

| Constraint | Source | Architectural implication |
|---|---|---|
| Apple Silicon only | NFR-C2, user instruction | No fallback inference path; `mlx-lm` is the only engine in MVP. `chirp init` fails fast on Intel. |
| macOS 13+ | NFR-C1, aligns with EPIC-AUDIO-CAPTURE | LaunchAgent available; `os.fork`/POSIX socket semantics consistent. |
| No network from daemon | NFR-S3 | Daemon must not embed HTTP client. Model downloads happen in the CLI process only (`chirp models add`/`pull`). |
| No telemetry | NFR-S4 | No analytics, crash reporting, or update checks anywhere in the stack. |
| Single-developer, sequential implementation | PRD §Resource Scope | Avoid unnecessary abstraction layers. Concrete simple modules over plugin frameworks. |
| Daemon protocol + client lands before integration cutover | PRD §Resource Scope | Architecture must let CHIRPD-CORE be useful standalone (vertical-slice demonstrable per readiness review). |
| Existing storage layout (`~/Documents/chirp/<slug>/`, `~/.chirp/chroma/`, `~/.chirp/config.toml`) unchanged | FR47 | This work touches **none** of the on-disk data layout; only the LLM call sites. |
| `mlx-lm` exact pin pre-1.0 | OQ5 resolution | Architecture should assume `mlx-lm`'s API surface is the version pinned in `pyproject.toml` and not abstract behind a "plug any inference library" interface unless required by a Vision-tier item. |
| HuggingFace cache reuse | FR38 | Architecture defers model file management to `huggingface_hub`; no custom download/cache layer. |
| Logfmt logging | OQ3 resolution | One logging facade across `chirpd` and CLI components emitting `key=value` lines. |

### Cross-Cutting Concerns

- **Streaming.** Chat ops produce token-by-token output. Affects: IPC framing (NDJSON line-per-event), client API shape (iterator vs. blocking), CLI stdout flushing (per-token flush to avoid block-buffering when piped), cancellation propagation (client closes connection or sends `cancel` op → daemon halts generation mid-loop).
- **Cancellation.** Required for `chat` op (NFR-P4 ≤ 200 ms). Implementation requires `mlx_lm.stream_generate` to be cancellable cooperatively — typically via a `should_stop` callback or async task cancellation.
- **Error model.** Three distinguishable error classes for clients: (1) transport (socket gone, broken pipe, daemon dead), (2) protocol (version mismatch, malformed request), (3) model (load failed, unsupported architecture). FR23 makes this explicit. Maps to a typed exception hierarchy in `chirp.llm`.
- **Lifecycle observability.** `chirp daemon status` must report enough state to diagnose the most common failure modes without reading logs: PID, uptime, version, loaded models with per-model RSS, last-request timestamps, idle countdowns. Implies the daemon exposes a `model.status` op with rich metadata.
- **Logging discipline.** Logfmt key=value lines, redacted (no prompts, no chat content), correlated by request id, rotated at 10 MB. Single logger configuration shared between `chirpd` and the CLI's client-side diagnostic output.
- **Concurrency in the daemon.** Multiple concurrent client connections must be serviceable (e.g., chat-in-flight + parallel embed). Architecture decision needed: single async event loop (asyncio) vs. threadpool vs. process-per-request. Settled in step-04.
- **Configuration precedence.** Env var → `~/.chirp/config.toml` → `models.toml` → built-in defaults. Established by FR / PRD §Config Schema; the architecture should consolidate this into one settings module rather than ad-hoc lookups.
- **Testability.** Coverage target NFR-M1 = 90% on new modules. The daemon must be testable without spawning a real subprocess (in-process socket; injectable inference backend for unit tests). Implies a thin `LLMBackend` boundary between protocol-handling code and `mlx-lm` calls.
- **Brownfield integration.** Existing `notes`, `notes_chat`, `chirp ask`, `chirp search` modules must compile and pass tests at every step of the migration. No long-lived migration branch (PRD §Resource Scope). Implies the `chirp.llm` client must be drop-in-replaceable for the Ollama client at the call site level.

## Starter Template Evaluation

### Decision: No starter template — brownfield extension

This is a brownfield Python project with an established codebase. The toolchain is already locked by `AGENTS.md` and the existing `pyproject.toml`:

| Concern | Established choice (source) |
|---|---|
| CLI framework | Typer (existing `chirp/cli.py`) |
| Configuration | Pydantic models loaded from TOML (existing `config/settings.py`) |
| Package manager | `uv` (AGENTS.md "Build, Test & Development Commands") |
| Test framework | `pytest` (AGENTS.md "Testing Guidelines") |
| Linter / formatter | `ruff` 88-char lines, double quotes (AGENTS.md "Coding Style") |
| Type checker | `mypy` (AGENTS.md "Coding Style → Types") |
| Build / install entry | `make dev-install`, `make check`, `make test` (AGENTS.md) |
| TTY rendering | Rich (existing `recorder/live_dashboard.py`, `chirp/init_flow.py`) |
| Pre-existing helper-bundle pattern | Swift `.app` bundle built by `python -m audio_capture.build` (EPIC-AUDIO-CAPTURE) |

### New dependencies introduced by this PRD

| Package | Pin | Role |
|---|---|---|
| `mlx-lm` | exact pin pre-1.0 (per OQ5 resolution) | MLX inference engine consumed by `chirpd` |
| `huggingface_hub` | minimum pin compatible with `mlx-lm`'s requirement | Model download and HF cache management |

No new test framework, CLI framework, or build system is introduced. New modules follow the existing project structure conventions described in `AGENTS.md` § "Project Structure & Module Organization."

### Note for the first implementation story

The first story under EPIC-CHIRPD-CORE does not need to "set up a project" — it adds `mlx-lm` and `huggingface_hub` to `pyproject.toml`, runs `make dev-install`, and creates the skeleton of the new `chirpd/` package. The Ollama removal is deferred to EPIC-INIT-AND-MIGRATION per the readiness review's recommendation to split `pyproject.toml` work across two epics.

## Core Architectural Decisions

The template's default decision categories (Data Architecture / Auth & Security / API / Frontend / Infrastructure) don't map cleanly to this PRD. The actual decision surface is:

- Process & concurrency model
- IPC protocol framing & semantics
- State machines (version handshake, model lifecycle)
- Error & exception taxonomy
- Configuration & persistence
- Daemon lifecycle integration with launchd

Decisions below are recorded with their rationale and the FRs/NFRs they support.

### Decision Priority Overview

| Priority | Decision | Status |
|---|---|---|
| Critical | Concurrency: asyncio with inference in a worker thread | ✅ Decided |
| Critical | IPC: unix socket + NDJSON, one event per line | ✅ Decided (PRD-locked) |
| Critical | Version handshake: explicit `hello` op, immediate-exit on mismatch + client respawn-and-retry | ✅ Decided (PRD-locked, mechanics defined here) |
| Critical | Single in-flight chat per model; embed parallel | ✅ Decided |
| Critical | Single-instance enforcement: `flock` on lockfile | ✅ Decided (PRD-locked) |
| Critical | Exception taxonomy: `LLMTransportError` / `LLMProtocolError` / `LLMModelError` | ✅ Decided |
| Important | Idle-unload timer: per-model asyncio task, rescheduled on activity | ✅ Decided |
| Important | Chat template handling: defer to model's tokenizer; cache per-model | ✅ Decided |
| Important | Configuration loading: Pydantic models from TOML | ✅ Decided (extends existing pattern) |
| Important | LaunchAgent plist: generated via `plistlib`; `launchctl load/unload` shelled | ✅ Decided |
| Important | Daemon entrypoint: console script `chirpd` in `pyproject.toml [project.scripts]` | ✅ Decided |
| Deferred | Multi-loaded-models concurrency (Growth feature) | Out of MVP |
| Deferred | Pluggable backend protocol beyond MLX | Vision tier |

### Process & Concurrency Model

**Daemon process model.**

- Single OS process per host (enforced by `flock` on `chirpd.lock`).
- Started lazily by any CLI client, or eagerly by LaunchAgent (opt-in).
- Exits only on: explicit `daemon stop`, version-mismatch (immediate), or fatal error.
- No auto-exit on idle. Idle behavior is *model* unload, not *process* exit.

**Concurrency: asyncio, single event loop.**

- The daemon runs one `asyncio` event loop hosting: a unix-socket server, per-connection request handler tasks, the idle-unload scheduler, and the inference worker.
- Inference (`mlx_lm.stream_generate`, `embed`) is CPU/GPU-bound and would block the loop. **Mitigation:** wrap each inference call in `asyncio.to_thread()` so the loop stays responsive for control messages (cancel, status, hello) and other connections.
- Token streaming from the worker thread back to the event loop uses a thread-safe `asyncio.Queue` (via `loop.call_soon_threadsafe(queue.put_nowait, ...)` from the worker).

**Why not threading-only:** the streaming + cancellation surface needs cancellable async iteration, which asyncio expresses naturally. Threading would force manual coordination via Events and Locks. **Why not multi-process:** wasteful — the model weights are the dominant resource and must stay resident.

**Why not put inference on the event loop directly:** even one second of blocked event loop breaks NFR-P4 (cancellation ≤ 200 ms) and breaks parallel embed+chat (FR concurrency expectation).

**Request execution model.**

- **Chat ops:** Single in-flight chat per chat-model at a time. Concurrent chat requests against the same model are queued FIFO. Cancel only affects the currently-running request; queued requests can be cancelled by closing the connection.
- **Embed ops:** Parallel with chat. Embed model is pinned and small; multiple embed requests can run concurrently (each in its own `to_thread()`). No queue.
- **Model.load / unload:** Mutually exclusive with all other ops on that model. Implemented via a per-model `asyncio.Lock`.

**Affects:** FR1–FR10 (daemon ops), FR21–FR23 (client streaming + non-streaming), NFR-P3 (throughput), NFR-P4 (cancellation), NFR-R6 (prompt shutdown).

### IPC Protocol — NDJSON over Unix Socket

**Already locked by the PRD.** Decisions captured here add mechanics.

- **Socket path:** `~/Library/Application Support/chirp/chirpd.sock`, mode `0600`.
- **Wire format:** newline-delimited JSON. One JSON object per line. UTF-8.
- **Framing:** plain `\n` line termination. No length prefix. Lines exceeding 1 MB rejected with `error`.
- **Connection lifecycle:**
  1. Client connects.
  2. Client sends one `hello` message.
  3. Daemon responds with `ready` (or `version_mismatch` — see state machine below).
  4. Client sends a single request message and reads streamed events until `done` or `error`.
  5. Either side may close the connection at any time; daemon treats client-close as cancel.
- **One request per connection.** Simplifies cancellation semantics and connection lifetime. Cheap on a unix socket.
- **Request envelope (client → daemon):**
  ```json
  {"id": "<client-generated-id>", "op": "<chat|embed|cancel|model.load|model.unload|model.list|model.status|health|hello>", "...op-specific fields..."}
  ```
- **Event envelope (daemon → client):**
  ```json
  {"id": "<same-id>", "event": "<ready|loading|delta|done|error|version_mismatch|status>", "...event-specific fields..."}
  ```
- **No id reuse, no out-of-order events.** Each request line gets a stream of events tagged with the same id, terminated by `done` or `error`.

**Affects:** FR11 (NDJSON over unix socket), FR1–FR4 (ops), NFR-S1 (local-only IPC), NFR-S2 (socket mode 0600).

### State Machine — Version Handshake & Drain

**Triggers:**
- Client sends `{op: "hello", client_version: "X.Y.Z"}` on every new connection.
- Daemon compares to its own version (from the same `chirp` package's `__version__`).

**Daemon state (alpha — simplified, no drain):**

```
[Running] ──hello matches──→ emits {event: "ready"} → handles request
[Running] ──hello mismatch──→ emits {event: "version_mismatch", daemon_version} → exits 0 immediately
```

Any requests in-flight at the moment of mismatch fail with `MODEL_GENERATION_FAILED` (cause `daemon_shutdown`). The client maps these to `LLMConnectionLost` on the socket-close side.

**Client behavior on `version_mismatch`:**

1. Close current socket.
2. Poll for socket-not-accepting (daemon is exiting immediately). Timeout: 1 s.
3. Once socket is gone OR connection refused, lazy-spawn a fresh daemon process via subprocess.
4. Poll for socket-accepting. Timeout: 5 s.
5. Reconnect and retry the *same* request exactly once. If the retry also fails on version (shouldn't happen but defensive): raise `LLMVersionMismatch`.

**Why immediate-exit instead of drain (alpha simplification):** chirp is in alpha; cross-upgrade in-flight requests are not a supported scenario. Removing the drain state machine simplifies `chirpd/lifecycle.py` materially (no `Draining` state, no 10 s deadline timer, no "reject new connections during drain" branch). The user-visible behavior is essentially identical for the common case (idle daemon at upgrade time), and clearer in the rare case (mid-request at upgrade time fails loudly).

**Why client respawn rather than daemon self-respawn:** the daemon doesn't know where the new package version's entrypoint lives; the client (running under the user's current shell) does. Self-respawn would require executing `which chirpd` from inside the daemon, which is brittle across uv/pip/editable installs.

**Why client respawn rather than daemon self-respawn:** the daemon doesn't know where the new package version's entrypoint lives; the client (running under the user's current shell) does. Self-respawn would require executing `which chirpd` from inside the daemon, which is brittle across uv/pip/editable installs.

**Affects:** FR13 (hello handshake), FR14 (immediate exit on mismatch — alpha-simplified), FR20 (transparent one-shot retry), NFR-R5 (drift recovery ≤ 2 s user-visible), NFR-R6 (prompt shutdown — alpha-simplified, no drain).

### Single-Instance Enforcement & Lazy-Spawn Race Resolution

**Lockfile:** `~/Library/Application Support/chirp/chirpd.lock`. Daemon acquires `fcntl.flock(LOCK_EX | LOCK_NB)` at startup; on failure, exits 0 immediately.

**Lazy-spawn race scenario.** Two CLI invocations simultaneously detect no socket:

1. Both run `subprocess.Popen(["chirpd"], start_new_session=True)`.
2. Both new daemon processes attempt `flock`. One acquires; the other exits 0.
3. The winner creates the socket and starts listening.
4. Both clients are in a connect-with-retry loop, polling the socket path with bounded backoff (50 ms × 100 attempts = 5 s).
5. Both connect to the same daemon.

**Affects:** FR12 (single-instance), FR19 (lazy-spawn), NFR-R4 (single-instance determinism within 1 s).

### Model Lifecycle State Machine

For each registered model:

```
[Unloaded] ──load request OR first op────→ [Loading]
[Loading]  ──mlx_lm.load completes────────→ [Ready]
[Ready]    ──op arrives───────────────────→ [Busy] → returns to [Ready] when done
[Ready]    ──idle timer fires (chat only)─→ [Unloading] → [Unloaded]
[Ready]    ──unload request───────────────→ [Unloading] → [Unloaded]
[Any]      ──load failure─────────────────→ [Unloaded] (with last error captured)
```

**Per-model state held in the daemon:**

```python
class LoadedModel:
    alias: str
    role: Literal["chat", "embed"]
    handle: Any  # the mlx-lm model object + tokenizer
    last_used: datetime
    idle_unload_task: asyncio.Task | None  # only set for chat models
    lock: asyncio.Lock  # mutex for load/unload/op
```

**Idle-unload scheduler.** When a chat op completes:
- Cancel any existing `idle_unload_task` for that model.
- Schedule a new one: `asyncio.create_task(_unload_after(timeout))`.
- The task sleeps `timeout` seconds, re-checks `last_used` (in case a request arrived during sleep), then unloads.
- `keep_alive=-1` skips scheduling. `keep_alive=0` triggers immediate unload after the response completes.

**Embed models** never get `idle_unload_task` scheduled. Pinned for process lifetime.

**Affects:** FR5 (lazy load), FR6 (embed pinned), FR7 (keep_alive override), FR9/FR10 (explicit load/unload), NFR-P5 (idle-unload precision ± 10%).

### Error & Exception Taxonomy

**`chirp.llm` exception hierarchy:**

```
LLMError (base)
├── LLMTransportError              # socket/connection failures
│   ├── LLMDaemonUnreachable       # daemon won't accept connection
│   ├── LLMConnectionLost          # mid-request broken pipe
│   └── LLMDaemonSpawnFailed       # lazy-spawn could not produce a working daemon
├── LLMProtocolError               # protocol-level issues
│   ├── LLMVersionMismatch         # raised only if retry fails (rare)
│   └── LLMMalformedResponse       # daemon emitted invalid JSON
└── LLMModelError                  # model-side issues
    ├── LLMModelNotFound           # alias not in registry, repo not resolvable
    ├── LLMModelLoadFailed         # mlx-lm load error (architecture, OOM, weights)
    ├── LLMGenerationFailed        # inference threw mid-generation
    └── LLMCancelled               # cancel succeeded, caller should treat as user intent
```

**Wire-level error event:** `{event: "error", code: "<machine-readable>", message: "<human-readable>", details: {...}}`. The client's NDJSON reader maps `code` to the exception class.

**Affects:** FR23 (distinguish load failures from transport), NFR-R7 (typed missing-weights error pointing to `chirp models pull`).

### Configuration & Persistence

**Two TOML files** (PRD-locked):

- `~/.chirp/config.toml` — existing chirp config, extended with `[llm]` section for backend/socket/idle.
- `~/Library/Application Support/chirp/models.toml` — new model registry.

**Pydantic models** for each file. Read via `tomllib` (stdlib, Python 3.11+). Write via `tomli_w` (small dependency; only needed by `chirp models` mutations and `chirp daemon enable`). Both models include a `schema_version` field. Reader rejects unknown versions outright with a typed error pointing users to re-init the registry (alpha-simplified — no migration pipeline).

**Precedence:** env var → `~/.chirp/config.toml` → built-in defaults (matches PRD §Config Schema → Environment variable overrides).

**Settings module:** one `chirp.settings` (or extend the existing `config/settings.py`) consolidates all environment + file lookups. No ad-hoc `os.environ.get()` calls scattered across the codebase.

**Affects:** FR37 (models.toml with schema version), FR38 (HF cache reuse), NFR-M4 (dependency hygiene).

### Daemon Lifecycle Integration with launchd

**`chirp daemon enable`:**

1. Writes `~/Library/LaunchAgents/com.chirp.chirpd.plist` via Python's `plistlib`. Contents (sketch):
   ```xml
   <key>Label</key>                  <string>com.chirp.chirpd</string>
   <key>ProgramArguments</key>       <array><string>/full/path/to/chirpd</string></array>
   <key>RunAtLoad</key>              <true/>
   <key>KeepAlive</key>              <dict><key>SuccessfulExit</key><false/></dict>
   <key>StandardOutPath</key>        <string>~/Library/Logs/chirp/chirpd.log</string>
   <key>StandardErrorPath</key>      <string>~/Library/Logs/chirp/chirpd.log</string>
   <key>EnvironmentVariables</key>   <dict>... (PATH, HF_HOME if set) ...</dict>
   ```
2. Runs `launchctl load <plist>` and verifies the agent appears with `launchctl list | grep com.chirp.chirpd`.
3. Reports success or surfaces stderr from `launchctl` on failure.

**`chirp daemon disable`:** `launchctl unload <plist>`; remove plist file.

**`ProgramArguments` resolution:** `shutil.which("chirpd")` at the time `enable` is run. The resolved absolute path is written into the plist. If the user later moves their Python environment, they re-run `chirp daemon enable` to refresh the path.

**Affects:** FR43/FR44 (enable/disable), NFR-R2 (LaunchAgent restart-on-crash), NFR-C1 (macOS 13+).

### Daemon Entrypoint

`pyproject.toml [project.scripts]`:

```toml
chirpd = "chirpd.__main__:main"
```

This means `chirpd` is on PATH after `pip install chirp` and can be invoked directly by LaunchAgent, lazy-spawn (`subprocess.Popen(["chirpd"])`), or the user. `python -m chirpd` also works for development.

**Affects:** FR15 (health endpoint usable by `chirp init`), FR40 (`chirp daemon start`), LaunchAgent plist `ProgramArguments`.

### Chat Template Handling

Each MLX model's tokenizer carries its own chat template (Jinja string in `tokenizer_config.json`). Apply via `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.

- The tokenizer is loaded alongside the model in `mlx_lm.load(...)` — no separate fetch.
- Cache the tokenizer per loaded model; it's part of the `LoadedModel` handle.
- No custom template parsing in chirp. If a model lacks a chat template, the registry validation step rejects it during `chirp models add`.

**Affects:** FR8 (apply chat template).

### Cross-Component Dependencies

| Component | Depends on | Why |
|---|---|---|
| `chirp.llm` client | NDJSON protocol spec | Encode/decode envelopes |
| `chirp.llm` client | Exception taxonomy | Map wire errors to typed exceptions |
| `chirpd` daemon | NDJSON protocol spec | Same encoder/decoder, ideally shared module |
| `chirpd` daemon | Model registry (`models.toml` reader) | Resolve alias → repo on `model.load` |
| `chirpd` daemon | `mlx-lm`, `huggingface_hub` (read-only) | Inference; weights discovery |
| `chirp.models` (CLI + registry writer) | `huggingface_hub` (download) | `chirp models add`/`pull` |
| `chirp.models` (CLI + registry writer) | `chirp.llm` client | Optional warm-on-add; `chirp models list` queries daemon for loaded state |
| `chirp.daemon` (CLI + lifecycle) | `chirp.llm` client | `chirp daemon status` / `health` |
| `chirp.daemon` (CLI + lifecycle) | `launchctl` (subprocess) | `enable` / `disable` |
| Existing `notes`, `notes_chat` | `chirp.llm` client | Drop-in replacement for Ollama client |

**Implementation sequence implied by dependencies (matches PRD §Resource Scope):**

1. Wire protocol module + exception taxonomy (no IO, pure)
2. `chirpd` skeleton: process startup, flock, socket listener, hello/health ops
3. Model registry reader (read-only against `models.toml`)
4. `chirpd` core ops: `model.load`, `model.unload`, `model.list`, `model.status`
5. `chirpd` inference ops: `chat` streaming, `embed`, `cancel`
6. `chirp.llm` client: connection management, lazy-spawn, version-mismatch retry
7. **Vertical-slice gate:** wire one existing command (`chirp ask`) through the new client end-to-end before declaring CHIRPD-CORE done.
8. `chirp models` CLI (writer side of the registry, HF integration)
9. `chirp daemon` CLI + LaunchAgent
10. Integration cutover for remaining existing commands
11. `chirp init` updates, dependency removal, docs sweep

## Implementation Patterns & Consistency Rules

The template's web-app conflict areas (DB naming, REST endpoints, JSON casing across browser/server boundaries) don't apply. The conflict surfaces that matter for this PRD:

1. **Wire protocol field & code naming.** Multiple modules emit/consume the same NDJSON envelope; drift would break parsing.
2. **Request ID generation.** Used as an event correlation key in logs and in protocol envelopes.
3. **Async patterns.** Easy to introduce subtle event-loop blocking or missed cancellation.
4. **Exception construction & re-raising.** A typed hierarchy is only useful if it's used consistently.
5. **Logging discipline.** Logfmt fields and what's safe to log.
6. **Configuration access.** Where settings come from and how they're read.
7. **Testing structure.** Backend boundary so unit tests don't require MLX or real subprocesses.

### Python / Project Conventions (Pre-Locked)

These come from AGENTS.md and existing chirp code. Listed here only to remove ambiguity, not to re-decide.

- **Formatting:** ruff, 88-char lines, double quotes, sorted imports (stdlib / third-party / first-party blocks).
- **Naming:** `snake_case` for vars/functions, `PascalCase` for classes, `lower_case` module names. Public CLI commands as locked-7 + new `models` (visible) + new `daemon` (hidden).
- **Type hints:** required on public functions in `chirp`, `chirp.llm`, `chirpd`, `chirp.models`, `chirp.daemon`. mypy must pass.
- **Comments:** rare. Prefer descriptive names over commentary. (Per user CLAUDE.md and AGENTS.md.)
- **Imports:** absolute imports only.

### Wire Protocol Naming Conventions

**Top-level envelope keys:** `id`, `op`, `event`, `error` — lowercase, no underscores, no camelCase. Same set on both directions.

**Request `op` values** (machine-readable, dot-namespaced):

```
hello | health | chat | embed | cancel
model.list | model.load | model.unload | model.status
```

**Event `event` values** (machine-readable, lowercase, no namespacing):

```
ready | loading | delta | done | error | version_mismatch | status
```

**Op-specific field names: snake_case.** Examples:

- `chat`: `model`, `messages`, `options`, `stream`, `keep_alive`, `max_tokens`
- `embed`: `model`, `inputs`
- `delta`: `text` (the new token text)
- `done`: `usage` (with `prompt_tokens`, `completion_tokens`, `ms`)
- `loading`: `model`, `progress` (0.0–1.0, optional)
- `error`: `code`, `message`, `details` (object)

**Error `code` values: SCREAMING_SNAKE_CASE**, namespaced by class:

```
TRANSPORT_*       — never emitted by daemon; constructed client-side
PROTOCOL_VERSION_MISMATCH
PROTOCOL_MALFORMED
MODEL_NOT_FOUND
MODEL_LOAD_FAILED
MODEL_GENERATION_FAILED
MODEL_CANCELLED
```

These codes map directly to the exception hierarchy in `chirp.llm`. **The mapping table must live in exactly one place** — `chirp.llm._error_codes` — and both daemon and client import from it.

### Request ID Convention

- Client generates request id per request.
- Format: `r-` prefix + 12 lowercase hex chars (e.g., `r-7c4a91b3de02`). Uses `secrets.token_hex(6)`.
- Daemon never invents ids; it echoes the client's `id` on every event for that request.
- Logged in every line associated with that request: `req_id=r-7c4a91b3de02`.

### Async Patterns

**Mandatory:**

- **Never** call blocking inference functions directly on the event loop. **Always** wrap in `await asyncio.to_thread(blocking_fn, ...)`. Holds for `mlx_lm.load`, `mlx_lm.stream_generate`, and `embed` calls.
- **Never** acquire `asyncio.Lock` and then call a blocking function that could span seconds without yielding. The per-model `Lock` is acquired, the inference is dispatched to `to_thread`, and the lock is released after `to_thread` completes (the lock spans the inference, but doesn't block other event-loop work).
- **Cancellation must propagate.** Inference threads check a `should_stop: asyncio.Event` via a callback passed to `stream_generate`. On `cancel` op or connection close, set the event; `stream_generate` exits at the next token boundary.
- **`asyncio.Queue` for streaming.** Token producer (worker thread) uses `loop.call_soon_threadsafe(queue.put_nowait, token)`. Token consumer (request handler coroutine) `await queue.get()`s and writes to the socket.

**Forbidden:**

- `.result()` or `.wait()` on `Future`/`Task` from inside coroutines (use `await`).
- `time.sleep()` in coroutines (use `asyncio.sleep`).
- Bare `asyncio.create_task()` without storing the task reference (causes silent cancellation).

### Exception Construction & Re-Raising

- **Always raise the most specific subclass.** Don't raise the base `LLMError` from production code; reserve it for `except` clauses.
- **Always include the wire `code` constant in the exception's `code` attribute.** Tests assert against the constant, not the message string.
- **Daemon-side:** catch all unexpected exceptions at the request handler boundary and convert to `error` events with code `MODEL_GENERATION_FAILED` (with the original exception chained in `details.exception_type` and `details.message`). The handler does not let exceptions propagate out of `handle_request` and crash the connection silently.
- **Client-side:** the NDJSON reader raises the typed exception immediately on `event: error`. Callers wrap calls in `try/except LLMModelError, LLMTransportError, LLMProtocolError` as needed; **never** catch bare `Exception`.

### Logging Discipline

**Format:** logfmt key=value, one line per event. Required keys (every line):

```
ts=2026-05-12T14:32:01.234Z level=info component=<chirpd|chirp.llm|...> msg=<short>
```

Optional keys, used when relevant:

```
req_id=r-... op=<...> model=<alias> duration_ms=<int> tokens=<int>
err_code=<...> err_type=<exception class name>
```

**Quoting:** values with spaces or special chars are double-quoted with backslash escaping. Use a small `logfmt` helper rather than hand-formatting.

**Forbidden in logs (privacy hard constraint NFR-S5):**

- User prompt content, chat messages, note content, transcript text.
- Embed input text.
- Full file paths under `~/Documents/` (use the bare slug if needed).

**Logger setup:** single `chirp.logging` module configures the root logger with the logfmt formatter and the rotating file handler. Imported once at process start. No per-module `logging.basicConfig`.

### Configuration Access

- All configuration reads go through `chirp.settings.get_settings()` (or the existing `config.settings` equivalent).
- **No `os.environ.get(...)` calls** outside the settings module.
- **No `tomllib.load(...)` calls** outside the settings module and the model-registry module.
- Settings are immutable after process start; the daemon doesn't watch files. Re-reading requires daemon restart (the version-mismatch + lazy-respawn cycle covers this implicitly).

### Model Registry Read/Write

- **Reads (daemon, `chirp models list/show`):** via a Pydantic model loaded from `models.toml`. Single shared module: `chirp.models.registry`.
- **Writes (`chirp models add/remove/default`):** the same Pydantic model, dumped via `tomli_w`. Atomic write: write to `models.toml.tmp` in the same directory, then `os.replace` to `models.toml`. Never partial writes.
- **Schema versioning (alpha-simplified):** `schema_version` field exists in the file. Reader accepts only the current version; an unknown version raises a typed error pointing users to re-init. No migration pipeline in MVP — chirp is alpha and the user re-creates the registry across breaking schema changes.

### Testing Patterns

**Backend abstraction for unit tests.** The daemon's inference layer sits behind a `LLMBackend` protocol:

```python
class LLMBackend(Protocol):
    async def load(self, repo: str, role: Literal["chat", "embed"]) -> ModelHandle: ...
    async def stream_generate(self, handle: ModelHandle, messages: list[dict],
                              options: GenOptions, should_stop: asyncio.Event) -> AsyncIterator[str]: ...
    async def embed(self, handle: ModelHandle, inputs: list[str]) -> list[list[float]]: ...
    async def unload(self, handle: ModelHandle) -> None: ...
```

Production implementation: `MLXBackend` (wraps `mlx-lm` calls in `to_thread`).
Test implementation: `FakeBackend` (deterministic, no MLX, returns scripted token sequences).

**Unit tests use `FakeBackend`.** No `mlx-lm` import required to run the daemon's unit tests. This is the only abstraction layer the codebase needs — explicitly *not* a "plug any inference library" framework.

**Integration tests use an in-process socket.** Start the daemon in the same Python process via `asyncio.start_unix_server(...)` against a temp socket; connect from the test with `chirp.llm.client(socket_path=...)`. No subprocess, no LaunchAgent.

**Real-model tests** (slow, opt-in): marked `@pytest.mark.slow` and `@pytest.mark.integration`. Skipped by default in `make test`. Used in CI on a dedicated job with a tiny test model (e.g., a 1B model) to verify the MLXBackend itself.

**Memory of prior project policy** (per `feedback_unit_test_mocking.md`): no unit tests for high-risk OS-touching code that requires permissions or real devices. **This rule does not extend to `chirpd`** — its unit-testable surface (protocol, state machines, registry, lifecycle without LaunchAgent) is large, and the OS-touching surface (LaunchAgent install/uninstall via `launchctl`) is small and isolatable behind subprocess seams. **The LaunchAgent install/uninstall code path is excluded from unit-test coverage** the same way audio capture is — covered by manual or integration tests only.

### CLI Output Patterns

- **`Rich` console** instance is shared across the codebase, not re-instantiated per command.
- **`--json` flag on `models list`, `models show`, `daemon status`:** when present, suppress all Rich rendering and emit a single JSON document to stdout. Diagnostic notices go to stderr in plain text.
- **Streaming output for `chat`/`ask`:** `sys.stdout.write(token); sys.stdout.flush()` per delta. Don't use `print(end="")` without `flush=True`.
- **Exit codes** match the table in PRD §Scripting Support. Codes 3/4/5 must be raised via typed exceptions caught by the Typer outer wrapper, not via `sys.exit(N)` scattered through the code.

### Enforcement

**All AI agents and human contributors MUST:**

- Use the typed exception hierarchy (`chirp.llm` exceptions) — never raise `ValueError`/`RuntimeError` for LLM-layer failures.
- Use the request id format `r-<12-hex>` everywhere, including in tests.
- Route all configuration reads through the settings module.
- Wrap all blocking inference in `asyncio.to_thread`.
- Use the logfmt helper for logs; never `f"timestamp={ts} ..."` ad-hoc.
- Use the shared error-code constants from `chirp.llm._error_codes`.

**Pattern violations are caught by:**

- `ruff` for formatting and import order.
- `mypy` for type contract.
- Code review (single-developer project, but `make check` is the gate).
- A small list of custom checks: a `tests/test_conventions.py` that asserts e.g., "no `os.environ` outside settings module" via AST grep.

### Anti-Patterns to Avoid

- **Mock the MLX module in unit tests** — don't. Use `FakeBackend` at the `LLMBackend` protocol boundary instead. Mocking `mlx_lm.load` directly couples tests to the third-party API surface.
- **Logging the full request payload "for debugging."** Violates NFR-S5. Use `req_id` for correlation and tail the daemon to see protocol-level events.
- **Catching `Exception` in request handlers** — except at the outermost boundary where we convert to a typed `error` event. Inside the handler, let typed exceptions propagate to that boundary.
- **Schedulers using `asyncio.create_task` without retaining the reference** — the task gets garbage collected and silently cancelled. Always assign to a stored variable (e.g., on the `LoadedModel`).
- **Re-implementing model-cache file management.** `huggingface_hub.snapshot_download` handles atomic downloads, partial-recovery, and cache locking. Don't write a custom downloader.
- **Mixing `print()` and the `logging` module** in the daemon — only `logging` is permitted from `chirpd`. The CLI surface may use `print()` (or Rich) for user-facing output, with `logging` reserved for diagnostics to stderr.
- **Adding a `--ollama-fallback` flag.** Reject all variants of this. The PRD explicitly removes Ollama; a fallback re-introduces every problem the PRD set out to fix.

## Project Structure & Boundaries

The existing chirp codebase uses **sibling top-level packages** (`chirp/`, `notes/`, `notes_chat/`, `recorder/`, `transcriber/`, `config/`, `utils/`, `audio_capture/`). This PRD's new code follows the same convention rather than nesting under `chirp/`.

### New Top-Level Packages

Two new packages are added; one existing package is extended; several existing modules get one-line call-site changes.

| Package | Purpose | Touched by |
|---|---|---|
| `chirpd/` (new) | The daemon process, its async server, state, and inference backend | EPIC-CHIRPD-CORE |
| `llm/` (new) | Client library, wire protocol, exceptions, model registry, HF integration, CLI subcommands (`chirp models`, `chirp daemon`) | EPIC-CHIRPD-CORE, EPIC-MODEL-REGISTRY, EPIC-DAEMON-LIFECYCLE |
| `chirp/` (existing) | CLI entry point — `chirp/cli.py` registers the new `models` and `daemon` Typer subcommand groups | EPIC-MODEL-REGISTRY, EPIC-DAEMON-LIFECYCLE, EPIC-INIT-AND-MIGRATION |
| `notes/`, `notes_chat/` (existing) | LLM call-site changes — swap Ollama client for `llm.client` | EPIC-INTEGRATION-CUTOVER |
| `config/` (existing) | Settings module gains an `[llm]` section for daemon socket, idle timeout, backend selection | EPIC-CHIRPD-CORE |

### Complete Project Tree (New + Touched)

```
chirp-ai-note-app/
├── pyproject.toml                      # adds mlx-lm, huggingface_hub; removes ollama at cutover
├── Makefile                            # unchanged
├── AGENTS.md                           # updated: Ollama references removed
├── CLAUDE.md                           # unchanged
├── README.md                           # updated: Ollama references removed
│
├── chirp/                              # EXISTING — CLI entry point package
│   ├── __init__.py
│   ├── cli.py                          # MODIFIED — registers `models` + `daemon` typer apps
│   ├── about.py
│   ├── branding.py
│   ├── exceptions.py                   # MODIFIED — no LLM-layer exceptions added here; see llm/exceptions.py
│   └── init_flow.py                    # MODIFIED — drops Ollama checks; adds daemon-readiness check
│
├── llm/                                # NEW — client library + registry + HF + CLI subcommands
│   ├── __init__.py                     # re-exports `client`, `LLMError` hierarchy
│   ├── client.py                       # LLMClient class: connect, lazy-spawn, retry, exception mapping
│   ├── protocol.py                     # NDJSON envelope encode/decode (shared with chirpd)
│   ├── exceptions.py                   # LLMError, LLMTransportError, LLMProtocolError, LLMModelError + subclasses
│   ├── error_codes.py                  # Wire-level error code constants (single source of truth)
│   ├── registry.py                     # models.toml Pydantic models, atomic write, alias resolution
│   ├── hf.py                           # HuggingFace integration: snapshot_download wrappers, role inference, validation
│   └── cli/
│       ├── __init__.py
│       ├── models.py                   # `chirp models` typer subcommand: add, list, remove, default, pull, show
│       └── daemon.py                   # `chirp daemon` typer subcommand: status, start, stop, restart, enable, disable, logs
│
├── chirpd/                             # NEW — the daemon
│   ├── __init__.py
│   ├── __main__.py                     # `python -m chirpd` entrypoint; also wired as `chirpd` console script
│   ├── server.py                       # asyncio.start_unix_server; per-connection request handler; cancel plumbing
│   ├── dispatcher.py                   # routes parsed ops to handlers (chat, embed, model.*, hello, health)
│   ├── state.py                        # LoadedModel dataclass; daemon-wide state container
│   ├── lifecycle.py                    # flock, version handshake (immediate-exit on mismatch), idle-unload scheduler
│   ├── backend.py                      # LLMBackend protocol + MLXBackend implementation
│   ├── launchd.py                      # plistlib-based plist gen; launchctl load/unload subprocess wrappers
│   └── logging_setup.py                # logfmt formatter, rotating handler, log directory creation
│
├── config/                             # EXISTING — settings package
│   ├── __init__.py
│   └── settings.py                     # MODIFIED — adds [llm] section: backend, daemon_socket, idle_timeout_seconds
│
├── notes/                              # EXISTING — call-site changes only
│   ├── __init__.py
│   ├── note_generator.py               # MODIFIED — replaces ollama client with llm.client
│   ├── note_editor.py
│   ├── manual_note_manager.py
│   ├── template_engine.py
│   ├── constants.py
│   └── ... (other existing files unchanged)
│
├── notes_chat/                         # EXISTING — call-site changes only
│   ├── __init__.py
│   ├── retrieval.py                    # MODIFIED — replaces ollama embed client with llm.client.embed
│   ├── cli.py                          # MODIFIED — chat/ask now routes through llm.client
│   ├── prompting.py                    # unchanged (prompts stay client-side)
│   ├── index.py                        # unchanged (chroma layout unchanged per FR47)
│   └── ... (other existing files unchanged)
│
├── recorder/                           # EXISTING — UNCHANGED
├── transcriber/                        # EXISTING — UNCHANGED (Whisper local; not LLM-routed)
├── utils/                              # EXISTING — UNCHANGED
├── audio_capture/                      # EXISTING — UNCHANGED (Swift helper bundle)
├── templates/                          # EXISTING — UNCHANGED
├── scripts/                            # EXISTING — UNCHANGED
│
└── tests/
    ├── chirpd/                         # NEW
    │   ├── test_server.py              # in-process socket integration tests
    │   ├── test_dispatcher.py
    │   ├── test_lifecycle.py           # version handshake state machine, flock race, idle-unload timing
    │   ├── test_state.py
    │   ├── test_backend_fake.py        # FakeBackend behavior; sanity for the protocol contract
    │   └── conftest.py                 # fixtures: temp socket, FakeBackend, daemon-under-test
    ├── llm/                            # NEW
    │   ├── test_client.py              # lazy-spawn, retry, exception mapping
    │   ├── test_protocol.py            # envelope encode/decode, malformed input handling
    │   ├── test_registry.py            # models.toml round-trip, alias resolution, atomic write
    │   ├── test_hf.py                  # role inference (with HF API mocked at the boundary)
    │   └── test_cli_models.py          # `chirp models` CLI behavior
    ├── chirp/
    │   └── test_init_flow.py           # MODIFIED — drops Ollama assertions, adds daemon-readiness checks
    ├── notes/                          # MODIFIED — fixtures swap from Ollama mocks to FakeBackend / mocked llm.client
    ├── notes_chat/                     # MODIFIED — same fixture swap
    ├── regression/
    │   └── notes_quality/              # NEW (manual capture, see PRD §Domain Requirements)
    │       ├── README.md
    │       ├── <slug>/
    │       │   ├── transcript.txt
    │       │   ├── notes_before.md     # baseline (Ollama era)
    │       │   └── notes_after.md      # to be generated post-migration
    │       └── ...
    └── test_conventions.py             # NEW — small custom-rule checks (no raw os.environ in modules, etc.)
```

### Runtime File Layout (User Machine)

```
~/                                          (user's home)
├── .chirp/
│   ├── config.toml                         # EXISTING — extended with [llm] section
│   └── chroma/                             # EXISTING — chroma index, untouched per FR47
├── Documents/chirp/                        # EXISTING — notes, untouched per FR47
│   └── <slug>/
│       ├── audio.wav
│       ├── transcript.txt
│       ├── notes.md
│       └── meta.toml
└── Library/
    ├── Application Support/chirp/          # NEW
    │   ├── chirpd.sock                     # Unix socket (mode 0600); cleaned up on graceful exit
    │   ├── chirpd.lock                     # flock target
    │   └── models.toml                     # model registry
    ├── Logs/chirp/                         # NEW
    │   ├── chirpd.log                      # logfmt key=value, ~10MB rotation, 1 prior gen retained
    │   └── chirpd.log.1
    └── LaunchAgents/                       # NEW (opt-in)
        └── com.chirp.chirpd.plist          # written by `chirp daemon enable`; absent otherwise

~/.cache/huggingface/hub/                   # SHARED with other MLX tools; weights live here
└── models--mlx-community--gemma-4-4b-it-4bit/ ...
```

### Architectural Boundaries

**Process boundary: `chirpd` ↔ CLI clients.**

- The unix-domain socket at `~/Library/Application Support/chirp/chirpd.sock` is the **only** runtime communication path.
- No shared in-memory state. No shared files (except read-only `models.toml`, which both sides read; only `chirp models` mutations write it, via atomic replace).
- Both sides import `llm.protocol` and `llm.error_codes` for envelope/code consistency. **These are the only shared modules between daemon and client.**

**Package boundary: `chirpd/` ↔ `llm/`.**

- `chirpd/` imports from `llm.protocol` and `llm.error_codes` only.
- `chirpd/` does **not** import from `llm.client`, `llm.cli`, `llm.hf`, or `llm.registry`'s write paths. (The daemon reads `models.toml` via a small `llm.registry.read_registry()` function — itself a read-only API. The writer side stays out of the daemon's import surface to avoid pulling `tomli_w` into the daemon.)
- `llm/` does **not** import from `chirpd/` directly. The client connects to the daemon via socket; it never spawns the daemon's internals in-process for production. (Tests do spawn `chirpd.server.serve(...)` in-process — that's the integration test boundary.)

**Module boundary: `llm.client` ↔ existing chirp modules.**

- `notes.note_generator`, `notes_chat.retrieval`, `notes_chat.cli` import `from llm.client import client` (or `from llm import client`). They never import from `chirpd/` or open the socket directly.
- The Ollama Python client import is the only thing being replaced at these sites. No prompt logic, retrieval logic, or template logic moves.

**CLI boundary: `chirp/cli.py` ↔ subcommand groups.**

- `chirp/cli.py` imports the new typer subapps from `llm.cli.models` and `llm.cli.daemon` and registers them via `app.add_typer(...)`. No business logic in `chirp/cli.py` for the new groups.

**HuggingFace boundary: `llm.hf` only.**

- Any `huggingface_hub` import lives in `llm.hf`. Other modules go through `llm.hf` functions. This isolates the only network-touching dependency to one module.
- `chirpd/` does **not** depend on `huggingface_hub` at runtime. The daemon loads weights from disk via `mlx_lm.load(repo_or_path)`; the path comes from the HF cache layout, which `huggingface_hub` is responsible for populating earlier (via `chirp models add`).

### Requirements-to-Structure Mapping

| FR range | Implementing modules |
|---|---|
| FR1–FR10 (daemon ops) | `chirpd/dispatcher.py`, `chirpd/backend.py`, `chirpd/state.py` |
| FR11–FR17 (protocol, lifecycle, handshake) | `chirpd/server.py`, `chirpd/lifecycle.py`, `chirpd/logging_setup.py`, `llm/protocol.py` |
| FR18–FR23 (client library) | `llm/client.py`, `llm/exceptions.py`, `llm/error_codes.py` |
| FR24–FR31 (registry user-facing) | `llm/cli/models.py`, `llm/registry.py`, `llm/hf.py` |
| FR32–FR38 (registry system behaviors) | `llm/registry.py`, `llm/hf.py`, `chirpd/state.py` (load-time resolution) |
| FR39–FR45 (daemon lifecycle CLI) | `llm/cli/daemon.py`, `chirpd/launchd.py`, `chirpd/logging_setup.py` |
| FR46–FR48 (integration) | call-site edits in `notes/note_generator.py`, `notes_chat/retrieval.py`, `notes_chat/cli.py` |
| FR49–FR54 (init, migration) | `chirp/init_flow.py`, `llm/cli/daemon.py` (LaunchAgent prompt) |
| FR55–FR56 (deps, docs) | `pyproject.toml`, `README.md`, `AGENTS.md`, doc sweep across all modules |

### Data Flow Examples

**`chirp ask "..."` (warm path):**

```
chirp ask → chirp/cli.py → notes_chat/cli.py → llm/client.py
  → unix socket → chirpd/server.py → chirpd/dispatcher.py
  → chirpd/backend.py (MLXBackend.stream_generate in to_thread)
  → tokens stream back via asyncio.Queue → socket → llm/client.py
  → notes_chat/cli.py prints to stdout
```

**`chirp models add <repo>` (no daemon required to register; warming requires daemon):**

```
chirp models add → llm/cli/models.py
  → llm/hf.py (validate repo, snapshot_download with progress)
  → llm/registry.py (atomic write to models.toml)
  → llm/client.py (issue model.load to warm; lazy-spawn daemon if absent)
```

**`chirp daemon enable`:**

```
chirp daemon enable → llm/cli/daemon.py
  → chirpd/launchd.py (write plist via plistlib, then subprocess: launchctl load <plist>)
  → verify with launchctl list → report result
```

### Integration Points (External)

- **HuggingFace Hub** — outbound HTTPS to `huggingface.co`, only from `llm.hf` during user-initiated `models add`/`pull`. Standard `huggingface_hub` library handles cache and resume.
- **launchctl** — local subprocess invocations from `chirpd/launchd.py`, only on `chirp daemon enable`/`disable`.
- **mlx-lm** — Python-API binding from `chirpd/backend.py` only. Daemon-side import; not imported in the CLI client process.
- **No other external integrations.** No databases, no message queues, no observability backends.

### Configuration Files

- `pyproject.toml` — dependencies; entry-point declarations (`chirpd` console script).
- `~/.chirp/config.toml` — user-editable runtime config; existing file extended.
- `~/Library/Application Support/chirp/models.toml` — registry; written by `chirp models *`.
- `~/Library/LaunchAgents/com.chirp.chirpd.plist` — only if user opted into LaunchAgent.

### Development Workflow Integration

- **Editable install:** `make dev-install` (existing). Now also pulls `mlx-lm` and `huggingface_hub`.
- **Run the daemon in dev:** `python -m chirpd` (foreground; prints logfmt to stderr in dev mode). Logs also go to `~/Library/Logs/chirp/chirpd.log`.
- **Run tests:** `make test`. Slow / integration tests gated behind `@pytest.mark.slow` and `@pytest.mark.integration` (existing convention).
- **CLI verification:** `uv run chirp --help`, `uv run chirp models --help`, `uv run chirp daemon --help`. Adds to existing `chirp init --recheck` and `make verify-deps`.
- **Make targets:** no new targets required; `make check` and `make test` cover the new modules.

## Architecture Validation Results

### Coherence Validation ✅

**Decision compatibility:**

- `asyncio` + `mlx-lm` via `asyncio.to_thread` is the standard pattern for sync-generator inference behind an async event loop. No conflict.
- `flock` (sync, called once at startup before the event loop starts) coexists cleanly with the asyncio listener that comes after.
- LaunchAgent-spawned daemon and lazy-spawned daemon resolve via the same `flock` lockfile. Deterministic single survivor.
- Per-model `asyncio.Lock` serializes load/unload/op against one model; embed and other-model ops proceed in parallel coroutines without blocking the loop.
- Pydantic v2 + `tomllib` (read) + `tomli_w` (write) is a well-established pattern. Atomic writes via `os.replace` are POSIX-safe on macOS.
- Version-handshake exit is clean: the daemon emits `version_mismatch` on the offending connection, then closes its listener and exits the asyncio loop. Any tasks for other in-flight connections raise on next `await` and surface to those clients as `LLMConnectionLost`.

**Pattern consistency:**

- Wire envelope field names (`id`, `op`, `event`, `error`) used identically on both sides via the shared `llm.protocol` module.
- Error code constants live in one place (`llm.error_codes`) and are imported by both `chirpd` (emit) and `llm.client` (raise typed exception).
- Logfmt format used consistently across `chirpd/logging_setup.py` and any CLI-side diagnostic logging.
- Request ID format (`r-<12-hex>`) generated only by the client; daemon echoes only.

**Structure alignment:**

- Sibling-top-level-package convention matches existing chirp layout exactly.
- Daemon (`chirpd/`) and client/registry (`llm/`) are in separate packages with a documented thin shared surface (`llm.protocol`, `llm.error_codes`, `llm.registry.read_registry`).
- HuggingFace dependency boundary (`llm.hf`) contains all outbound HTTP. Daemon does not import `huggingface_hub` directly (see refinement below).

### Requirements Coverage Validation ✅

**Functional Requirements (56 total) — full traceability:**

| FR range | Implementing module(s) | Status |
|---|---|---|
| FR1–FR10 | `chirpd/dispatcher.py`, `chirpd/backend.py`, `chirpd/state.py` | ✅ |
| FR11 | `chirpd/server.py`, `llm/protocol.py` | ✅ |
| FR12 | `chirpd/lifecycle.py` (flock) | ✅ |
| FR13–FR14 | `chirpd/lifecycle.py` (handshake state machine) | ✅ |
| FR15 | `chirpd/dispatcher.py` (`health` op) | ✅ |
| FR16 | `chirpd/logging_setup.py` (rotating handler) | ✅ |
| FR17 | structural boundary: daemon does not import `huggingface_hub` | ✅ (see refinement) |
| FR18–FR23 | `llm/client.py`, `llm/exceptions.py`, `llm/error_codes.py` | ✅ |
| FR24–FR31 | `llm/cli/models.py`, `llm/registry.py`, `llm/hf.py` | ✅ |
| FR32 | `llm/hf.py` (role inference) | ✅ |
| FR33 | `llm/cli/models.py` (auto-default) | ✅ |
| FR34 | `llm/cli/models.py` (post-add `model.load`) | ✅ |
| FR35–FR36 | `chirpd/dispatcher.py` (alias / raw-repo resolution) | ✅ |
| FR37 | `llm/registry.py` (Pydantic, `schema_version` field) | ✅ |
| FR38 | `llm/hf.py` + daemon load uses HF cache layout | ✅ |
| FR39–FR45 | `llm/cli/daemon.py`, `chirpd/launchd.py`, `chirpd/logging_setup.py` | ✅ |
| FR46–FR48 | call-site edits in `notes/note_generator.py`, `notes_chat/retrieval.py`, `notes_chat/cli.py` | ✅ |
| FR49–FR54 | `chirp/init_flow.py`, `llm/cli/daemon.py` | ✅ |
| FR55–FR56 | `pyproject.toml` (two-phase landing), `README.md`, `AGENTS.md` | ✅ |

**Non-Functional Requirements (33 total) — architectural support:**

- **Performance (NFR-P1–P7):** asyncio + `to_thread` keeps the loop responsive; streaming via `asyncio.Queue` minimizes per-token overhead; lazy-spawn polls socket every 50 ms (under spawn-to-ready budget); cancellation propagation via `should_stop: asyncio.Event` checked per token (under 200 ms budget).
- **Reliability (NFR-R1–R7):** LaunchAgent `KeepAlive` covers crash recovery; flock covers single-instance race; version-handshake immediate-exit plus client respawn covers drift (alpha-simplified, no drain); typed exception for missing weights covers self-heal path.
- **Security (NFR-S1–S6):** unix socket only (no TCP); 0600 permissions enforced at `bind`; HF imports isolated to `llm.hf` in CLI process; logfmt redaction discipline enforced by code review and convention checks.
- **Maintainability (NFR-M1–M5):** `LLMBackend` protocol enables FakeBackend in unit tests → high coverage attainable; mypy clean; new modules small enough to stay testable.
- **Compatibility (NFR-C1–C4):** Apple Silicon required → checked at `chirpd/__main__.py` startup and at `chirp init`; macOS 13+ allows POSIX-y assumptions and LaunchAgent.
- **Observability (NFR-O1–O3):** logfmt resolved per OQ3; rotation in `chirpd/logging_setup.py`; rich `model.status` op feeds `chirp daemon status`.
- **Accessibility (NFR-A1):** stdout discipline (TTY vs non-TTY) in `llm/cli/*` modules.

### Implementation Readiness Validation ✅

**Decision completeness:** Critical decisions (process model, IPC, concurrency, version handshake, error taxonomy, single-instance) all locked. Important decisions (idle-unload, chat template, config, LaunchAgent plist, entrypoint) all locked. Deferred decisions (multi-loaded-models, pluggable backend) explicitly out of MVP.

**Structure completeness:** Full directory tree provided; every new file has a named purpose; every modified existing file has a one-line description of the change. Runtime file layout on the user's machine is explicit.

**Pattern completeness:** Wire-protocol naming, error code naming, request ID format, async patterns, exception construction, logging fields, configuration access, registry I/O, testing structure, CLI output discipline all specified with examples and anti-patterns.

### Gap Analysis — Findings and Refinements

**🟠 One refinement to lock before EPIC-CHIRPD-CORE story authoring:**

**Daemon's HF-cache lookup mode.** The daemon must call `mlx_lm.load(...)` with a *local path*, not a repo id, to avoid any chance of transitive network access from `huggingface_hub`. Refinement:

- The daemon imports `huggingface_hub.snapshot_download` solely for the purpose of resolving repo id → local cache path with `local_files_only=True`. This call performs no network I/O; it raises if the cache is empty.
- On `LocalEntryNotFoundError`, the daemon returns `MODEL_LOAD_FAILED` with a clear message pointing to `chirp models pull <alias>`.
- This means `huggingface_hub` IS imported in the daemon process, but only the cache-lookup APIs are used. NFR-S3 ("no outbound traffic from `chirpd`") is preserved at the *behavioral* level. Documented explicitly.
- Alternative considered: have the CLI resolve the local path and pass it to the daemon in the `model.load` op payload. Rejected — adds unnecessary coupling; the daemon's job is to load by alias, and the alias-to-path resolution is naturally daemon-side.

This refinement is reflected in:
- `chirpd/backend.py` — uses `snapshot_download(repo, local_files_only=True)` then `mlx_lm.load(local_path)`.
- Documentation: the daemon-side architectural boundary statement updated to "no outbound network from daemon" rather than "no `huggingface_hub` import at all."

**🟡 Minor clarifications (not blocking):**

- **Apple Silicon enforcement points.** Two places (one is just-in-case redundancy): `chirp init` checks `platform.machine() == "arm64"` and exits with a clear message if not; `chirpd/__main__.py` runs the same check at process start (defensive — `chirp init` should have caught it, but if the daemon is launched on the wrong arch via LaunchAgent after a hardware transplant, fail fast).
- **`models.toml` consistency between daemon and CLI.** The daemon re-reads on every `model.load`/`model.list`/`model.status`. The CLI writes atomically via `os.replace`. Race window: CLI is mid-write while daemon is mid-read. `os.replace` is atomic so the daemon sees either the old or new state — never a partial file. Acceptable.
- **Regression corpus storage.** `tests/regression/notes_quality/` holds transcripts and `notes_before.md` / `notes_after.md` only. **Audio files are NOT committed** — too large; users capture their own. `tests/regression/notes_quality/README.md` documents the capture procedure (referring back to PRD §Domain Requirements validation methodology).
- **`mlx-lm` direct dependency on `huggingface_hub`.** Confirmed `mlx-lm` already depends on `huggingface_hub` transitively. Adding `huggingface_hub` as a direct dependency in `pyproject.toml` is appropriate — we depend on it explicitly for `snapshot_download` and `HfApi`. Pin its minimum version compatible with `mlx-lm`'s requirement.

**No critical gaps.** No blockers identified for EPIC-CHIRPD-CORE story authoring beyond the refinement above (which is a clarification, not new design work).

### Architecture Completeness Checklist

**Requirements Analysis:**
- [x] Project context thoroughly analyzed
- [x] Scale and complexity assessed (medium technical, single-user)
- [x] Technical constraints identified (Apple Silicon only, macOS 13+, no network from daemon, exact mlx-lm pin)
- [x] Cross-cutting concerns mapped (streaming, cancellation, error model, logging, concurrency, config precedence, testability)

**Architectural Decisions:**
- [x] Process model and concurrency decided (asyncio + to_thread)
- [x] IPC framing decided (NDJSON, line-per-event)
- [x] Version handshake state machine specified
- [x] Single-instance enforcement specified (flock)
- [x] Idle-unload model lifecycle specified
- [x] Error taxonomy and wire-code mapping specified
- [x] Configuration & persistence specified (TOML + Pydantic)
- [x] LaunchAgent integration specified (plistlib + launchctl shell)
- [x] Daemon entrypoint specified (console script)
- [x] Chat template handling specified (defer to model tokenizer)

**Implementation Patterns:**
- [x] Wire protocol field/code naming
- [x] Request ID format
- [x] Async patterns and forbidden constructs
- [x] Exception construction discipline
- [x] Logging discipline (logfmt fields, redaction rules)
- [x] Configuration access (single settings module)
- [x] Registry read/write (atomic via os.replace)
- [x] Testing patterns (`LLMBackend` boundary; FakeBackend; integration in-process)
- [x] CLI output discipline (TTY/non-TTY, `--json`, streaming flush)
- [x] Enforcement mechanisms (ruff, mypy, `tests/test_conventions.py`)

**Project Structure:**
- [x] Complete directory tree (new + touched existing)
- [x] Runtime file layout on user's machine
- [x] Module-to-FR mapping
- [x] Data flow examples for the three primary scenarios (`chirp ask`, `chirp models add`, `chirp daemon enable`)
- [x] Integration boundaries (process, package, module, CLI, HF)
- [x] External integration points (HuggingFace, launchctl, mlx-lm — and no others)

### Architecture Readiness Assessment

**Overall Status:** READY FOR IMPLEMENTATION

**Confidence Level:** HIGH

**Key strengths:**

1. **Reuses existing project conventions exhaustively.** Sibling top-level packages, Typer, Pydantic, ruff, mypy, pytest, uv, the Swift-helper bundle pattern from EPIC-AUDIO-CAPTURE for the LaunchAgent equivalent. No new tooling.
2. **Boundaries that prevent the most likely mistakes.** Single source of truth for wire codes (`llm.error_codes`); HF import isolated; daemon and client share only the protocol module; config access funneled through one settings module.
3. **Testable surface despite OS coupling.** `LLMBackend` protocol with FakeBackend enables 90% unit coverage without MLX or subprocesses. LaunchAgent's launchctl seam is the only meaningful untestable area, and it's small.
4. **Version-drift recovery is a real state machine, not a hope.** The immediate-exit-and-respawn cycle is specified end-to-end with timeouts, retry policy, and user-visible behavior. Alpha-simplified (no graceful drain), but the upgrade UX is preserved for the common case.
5. **No new external services or infrastructure.** No database, no message queue, no observability backend. The whole system runs on a unix socket and two TOML files.

**Areas for future enhancement (out of MVP scope, captured in PRD Growth/Vision):**

- Per-task model overrides (Growth)
- Multi-loaded-models concurrency (Growth)
- Background pre-warm on `chirp init` (Growth, gated by OQ4)
- Telemetry surface in `chirp daemon status` (Growth)
- Custom prompt profiles in registry (Growth)
- Apple Foundation Models backend (Vision)
- Menu bar app sharing the daemon (Vision, separate epic)

### Implementation Handoff

**AI agent / contributor guidelines:**

- Follow the wire envelope conventions and error code constants exactly as specified.
- Wrap every blocking inference call in `asyncio.to_thread`. No exceptions.
- Use the typed `LLMError` hierarchy at every LLM-layer call site; never raise raw `ValueError`/`RuntimeError`.
- Route all configuration reads through the settings module.
- Route HuggingFace network calls through `llm.hf` only.
- Use logfmt for all daemon logs; never log user-supplied content (prompts, messages, notes).
- Use `FakeBackend` for unit tests; never mock `mlx_lm.*` directly.

**First implementation priority** (sequenced per PRD §Resource Scope and §Cross-Component Dependencies):

1. Add `mlx-lm` (exact pin) and `huggingface_hub` to `pyproject.toml`; run `make dev-install`.
2. Create `llm/protocol.py`, `llm/error_codes.py`, `llm/exceptions.py` (no IO, pure modules).
3. Create `chirpd/` skeleton: `__main__.py` with Apple Silicon check + flock + asyncio entry; `server.py` with socket setup; `dispatcher.py` with `hello` and `health` only.
4. Create `llm/client.py` minimum: connect, send `hello`, handle `ready`/`version_mismatch`; expose `client.health()`.
5. Verify the loop with a smoke test: spawn `chirpd`, call `client.health()`, assert OK.

Subsequent steps continue the sequence in §Cross-Component Dependencies → "Implementation sequence implied by dependencies."
