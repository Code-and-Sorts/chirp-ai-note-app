# Epic: `chirp ask` runs against the new MLX-backed `chirpd` daemon

- **Epic ID:** EPIC-CHIRPD-CORE
- **Owner:** Colby
- **Status:** Draft
- **Created:** 2026-05-15
- **Design source:** [`prd.md`](../prd.md) — Ollama → MLX migration; [`architecture.md`](../architecture.md) — daemon, protocol, client decisions
- **Related branch (current work):** TBD (off `main`; predecessor branch `story/2.3-blackhole-removal`)

## 1. Goal

Stand up the in-process MLX inference daemon (`chirpd`), the NDJSON wire protocol between it and chirp CLI clients, and the `llm.client` library that talks to it — then prove the whole stack works by routing the existing `chirp ask` command through it end-to-end.

After this epic, a developer can run `chirp ask "hello"` and watch a streamed answer arrive from a locally-spawned `chirpd` process loading an MLX model from the HuggingFace cache. The Ollama daemon is no longer involved in the `ask` flow. The Ollama Python client remains in `pyproject.toml` for the rest of chirp's surfaces — its removal (and the `notes` / `notes_chat` broader cutover) lives in later epics. This epic's "done" criterion is a working user command, not "library exists."

## 2. Why now

The PRD decomposes into five epics: **CHIRPD-CORE**, **MODEL-REGISTRY**, **DAEMON-LIFECYCLE**, **INTEGRATION-CUTOVER**, **INIT-AND-MIGRATION**. Every one of the other four depends on the daemon being a real process that accepts NDJSON over its socket and on `llm.client` being importable from existing modules:

1. **MODEL-REGISTRY's** writer side (`chirp models add`) calls `client.warm()` after registering, which requires the daemon and client surface to exist.
2. **DAEMON-LIFECYCLE's** `chirp daemon status` is a thin wrapper over a `health` / `model.status` op against the daemon.
3. **INTEGRATION-CUTOVER's** swap of `notes/note_generator.py` and `notes_chat/retrieval.py` is a call-site change to `llm.client`.
4. **INIT-AND-MIGRATION's** Ollama removal from `pyproject.toml` cannot land until every chirp call site is already on `llm.client`.

Until CHIRPD-CORE ships a working vertical slice, none of the other epics can land their first story. Sequencing this epic first turns the implementation order locked in [`architecture.md` § Cross-Component Dependencies](../architecture.md) into shipped scope.

The readiness review ([`implementation-readiness-report-2026-05-12.md`](../implementation-readiness-report-2026-05-12.md) § Epic Quality Review) called out the specific risk that CHIRPD-CORE becomes a "technical milestone" epic with no user-observable demonstration. The vertical-slice gate in §7 below is the explicit remediation: the epic does not land until `chirp ask "..."` runs against the new daemon.

## 3. Locked decisions from the architecture

| # | Decision | Source |
|---|----------|--------|
| 1 | Daemon is a single OS process, single asyncio event loop. Blocking inference (`mlx_lm.load`, `mlx_lm.stream_generate`, `embed`) is wrapped in `asyncio.to_thread`. Tokens stream back over `asyncio.Queue` using `loop.call_soon_threadsafe`. | architecture § Process & Concurrency Model |
| 2 | Single-instance via `fcntl.flock(LOCK_EX \| LOCK_NB)` on `~/Library/Application Support/chirp/chirpd.lock`. Acquired **before** the asyncio loop starts (sync). Losers `exit(0)` cleanly. | architecture § Single-Instance Enforcement |
| 3 | IPC is NDJSON over unix socket at `~/Library/Application Support/chirp/chirpd.sock` (mode `0600`), one JSON object per line, one request per connection. Envelope keys: `id`, `op`, `event`, `error` (lowercase, no underscores). | architecture § IPC Protocol |
| 4 | Version handshake is **alpha-simplified — immediate exit on mismatch, no drain**. Daemon receives `hello`, compares versions, on mismatch emits `{event: "version_mismatch", daemon_version}` and `exit(0)`. Client polls until socket gone, lazy-spawns fresh daemon, retries the same request exactly once. | architecture § State Machine — Version Handshake; PRD FR14, NFR-R6 (alpha simplification) |
| 5 | Request ID format: `r-` + 12 lowercase hex chars (e.g., `r-7c4a91b3de02`). Generated client-side via `secrets.token_hex(6)`. Daemon echoes only — never invents an id. | architecture § Request ID Convention |
| 6 | Error code constants live in **exactly one place** — `llm/error_codes.py` — imported by both daemon (to emit) and client (to map to exceptions). Code form: `SCREAMING_SNAKE_CASE` namespaced by class (`PROTOCOL_*`, `MODEL_*`). | architecture § Wire Protocol Naming Conventions |
| 7 | Exception hierarchy: `LLMError` → `LLMTransportError` / `LLMProtocolError` / `LLMModelError`, each with concrete subclasses. **Never** raise the base class from production code; **never** raise `ValueError`/`RuntimeError` at the LLM layer. | architecture § Error & Exception Taxonomy |
| 8 | Per-model state lives in a `LoadedModel` dataclass (alias, role, handle, last_used, optional idle-unload task, `asyncio.Lock`). The lock serializes load/unload/op against one model; embed ops on other models proceed in parallel. | architecture § Model Lifecycle State Machine |
| 9 | Inference is testable via an `LLMBackend` Protocol (`load`, `stream_generate`, `embed`, `unload`). Production: `MLXBackend`. Unit tests: `FakeBackend`. **`mlx_lm.*` must not be mocked directly anywhere.** | architecture § Testing Patterns |
| 10 | The daemon resolves alias → local path via `huggingface_hub.snapshot_download(repo, local_files_only=True)` then loads from the resolved local path. No network from the daemon. On `LocalEntryNotFoundError`, emit `MODEL_LOAD_FAILED` pointing to `chirp models pull <alias>`. | architecture § Gap Analysis → HF-cache-lookup refinement |
| 11 | Logging: logfmt key=value lines (per OQ3). Required keys per line: `ts`, `level`, `component`, `msg`. Request-correlated lines add `req_id`, `op`, `model`. Forbidden: prompts, chat content, transcripts, embed inputs. Configured once in `chirpd/logging_setup.py` with a rotating file handler at `~/Library/Logs/chirp/chirpd.log` (~10 MB cap, ≥1 prior generation retained). | architecture § Logging Discipline; NFR-O1, NFR-O2, NFR-S5 |
| 12 | Apple Silicon check happens at `chirpd/__main__.py` startup (defensive, in case launched on the wrong arch via LaunchAgent after hardware transplant). The user-facing check at `chirp init` is owned by EPIC-INIT-AND-MIGRATION. | architecture § Gap Analysis → Apple Silicon enforcement points |
| 13 | Daemon entrypoint: console script `chirpd = "chirpd.__main__:main"` in `pyproject.toml`. `python -m chirpd` also works for development. | architecture § Daemon Entrypoint |
| 14 | `mlx-lm` is pinned **exact** (`==X.Y.Z`) per OQ5 resolution; `huggingface_hub` carries a compatible minimum pin. The Ollama Python client stays in `pyproject.toml` for now — its removal belongs to EPIC-INIT-AND-MIGRATION after cutover. | architecture § Starter Template Evaluation → New dependencies; PRD §Open Questions → OQ5 |

## 4. Research findings — what exists vs. what is missing

Validated against the current `main` branch and the `story/2.3-blackhole-removal` predecessor branch.

### What does not yet exist

- **`chirpd/` package** — none of the daemon files exist. No `chirpd/__main__.py`, no `chirpd/server.py`, no `chirpd/dispatcher.py`, no `chirpd/state.py`, no `chirpd/lifecycle.py`, no `chirpd/backend.py`, no `chirpd/logging_setup.py`.
- **`llm/` package** — does not exist. No `llm/protocol.py`, no `llm/client.py`, no `llm/exceptions.py`, no `llm/error_codes.py`, no `llm/registry.py` (read-only side).
- **`mlx-lm` and `huggingface_hub` dependencies** — not in `pyproject.toml`.
- **`chirpd` console script** — not declared in `pyproject.toml [project.scripts]`.
- **`~/Library/Application Support/chirp/`** runtime directory — not created by any existing code path (created by this epic the first time the daemon starts).
- **`~/Library/Logs/chirp/`** runtime directory — not created by any existing code path.
- **`tests/chirpd/`** and **`tests/llm/`** test trees — do not exist.

### Ollama call sites that this epic touches (or that remain for later epics)

Captured via `grep -n "ollama" notes_chat/*.py notes/*.py` on the predecessor branch:

| File | Lines | Story handling it |
|---|---|---|
| `notes_chat/cli.py` | the `ask` flow (one-shot + interactive) — currently calls into `notes_chat.prompting.*` which talks to Ollama over HTTP | **Story 3.7** (vertical-slice gate; reroute the `ask` flow through `llm.client`) |
| `notes_chat/prompting.py` | 15+ Ollama URLs across `_call_ollama_*` variants, `validate_ollama_connection` | **Story 3.7** rewires the `ask`-reachable subset; the rest stays on Ollama under EPIC-INTEGRATION-CUTOVER |
| `notes_chat/retrieval.py:396` | embeddings via Ollama HTTP — used by `ask`'s retrieval | **Story 3.7** if it is on the `ask` hot path; otherwise EPIC-INTEGRATION-CUTOVER |
| `notes_chat/index.py:334` | indexer embeddings via Ollama HTTP | EPIC-INTEGRATION-CUTOVER (not on the `ask` hot path) |
| `notes/note_generator.py:349-356` | `_call_ollama` for note generation during `chirp transcribe` | EPIC-INTEGRATION-CUTOVER |
| `pyproject.toml` | `ollama` Python client still listed | EPIC-INIT-AND-MIGRATION (after every call site is migrated) |

### Already implemented that this epic reuses

- **`config/settings.py`** — Pydantic `ChirpSettings` with TOML loading; this epic extends it with an `[llm]` section (`backend`, `daemon_socket`, `idle_timeout_seconds`).
- **`chirp/cli.py`** — Typer entry point. This epic does **not** add visible commands; the new `chirpd` console script is a separate top-level entry. `chirp models` and `chirp daemon` Typer groups are EPIC-MODEL-REGISTRY and EPIC-DAEMON-LIFECYCLE.
- **`notes_chat/cli.py` ask command** — the entry point for the vertical-slice cutover.
- **Rich rendering helpers** in `chirp/init_flow.py` and elsewhere — the streaming token output in story 3.7 uses the same stdout flushing convention.
- **Pytest + uv toolchain** locked by AGENTS.md — no new test framework, lint tool, or build system in this epic.

### Net code delta (rough)

- **Add:** ~1500–1800 lines Python across `chirpd/` and `llm/`, plus ~600 lines tests. Two `pyproject.toml` dependency lines and one console-script line.
- **Modify:** ~50 lines in `notes_chat/cli.py` (story 3.7 vertical slice — swap the `ask` LLM call to `llm.client`); one new `[llm]` section in `config/settings.py`.
- **Remove:** nothing yet. Ollama removal lives downstream.

## 5. Story breakdown

| ID | Title | FRs covered | Depends on | File |
|----|-------|-------------|------------|------|
| 3.1 | Add `mlx-lm` + `huggingface_hub` dependencies; scaffold `chirpd/` and `llm/` packages; declare `chirpd` console script | (enabling work; no FR delivered) | — | [stories/3.1-deps-and-scaffold.md](stories/3.1-deps-and-scaffold.md) |
| 3.2 | Wire protocol module, error-code constants, exception taxonomy | FR11, FR23 | 3.1 | [stories/3.2-protocol-and-exceptions.md](stories/3.2-protocol-and-exceptions.md) |
| 3.3 | Daemon skeleton: startup, flock, Apple-Silicon check, asyncio socket listener, `hello` + `health` ops, logfmt logging | FR11, FR12, FR13, FR15, FR16, FR17 | 3.2 | [stories/3.3-daemon-skeleton.md](stories/3.3-daemon-skeleton.md) |
| 3.4 | Client library: connect, lazy-spawn, version-mismatch handling, exception mapping | FR18, FR19, FR20, FR23, FR13 (client side), FR14 (client side) | 3.3 | [stories/3.4-client-library.md](stories/3.4-client-library.md) |
| 3.5 | Model lifecycle: state machine, registry reader, `model.load` / `model.unload` / `model.list` / `model.status` ops, idle-unload scheduler | FR4, FR5, FR6, FR7, FR9, FR10, FR35, FR36, FR37 (read), FR38 | 3.4 | [stories/3.5-model-lifecycle.md](stories/3.5-model-lifecycle.md) |
| 3.6 | Backend abstraction + MLX implementation: `chat` (streaming), `embed`, `cancel` ops, chat-template handling, cancellation event plumbing | FR1, FR2, FR3, FR8, FR21, FR22 | 3.5 | [stories/3.6-backend-and-inference.md](stories/3.6-backend-and-inference.md) |
| 3.7 | Vertical slice: route `chirp ask` through `llm.client` end-to-end (epic completion gate) | FR46 (partial — `ask` only); validates FR1–FR23 in production | 3.6 | [stories/3.7-vertical-slice-chirp-ask.md](stories/3.7-vertical-slice-chirp-ask.md) |

## 6. Sequencing & dependencies

**Within-epic ordering is strictly sequential:** 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7. Each story has an independently demonstrable end-state:

- **3.1** — `make dev-install` succeeds with new deps; `python -c "import chirpd; import llm"` returns; `chirpd --help` shows a (still-empty) Typer / argparse stub.
- **3.2** — `pytest tests/llm/test_protocol.py tests/llm/test_exceptions.py` passes; pure-module unit tests only.
- **3.3** — Manual smoke: `chirpd` starts, accepts a netcat-style connection, returns `{event: "ready"}` for matching version, `{event: "version_mismatch"}` for non-matching, and exits in the latter case.
- **3.4** — `pytest tests/llm/test_client.py` passes; integration tests use in-process `asyncio.start_unix_server` against a temp socket.
- **3.5** — `client.model_list()` returns the registered set; `client.model_load("alias")` transitions a `LoadedModel` through `Loading → Ready`; idle-unload fires within ± 10% of the configured timeout in tests with a `FakeBackend`.
- **3.6** — `client.chat(messages=[...])` streams deltas from `FakeBackend` end-to-end through the socket; `client.cancel(id)` halts within 200 ms; `client.embed([...])` returns the right vector count.
- **3.7** — `chirp ask "hello"` prints a streamed answer from the new daemon. **This is the epic's "done" gate.**

**What this epic blocks:**

- **EPIC-MODEL-REGISTRY** — `chirp models add` calls `client.model_load("alias")` to warm; requires 3.5 + 3.4 landed.
- **EPIC-DAEMON-LIFECYCLE** — `chirp daemon status` wraps the daemon's `health` + `model.status` ops; requires 3.3 + 3.5 landed.
- **EPIC-INTEGRATION-CUTOVER** — `notes/note_generator.py` and `notes_chat/retrieval.py` migration call-site changes; requires the full client surface from 3.4 + 3.6.
- **EPIC-INIT-AND-MIGRATION** — Ollama removal from `pyproject.toml` cannot land until every call site is on `llm.client`, which closes after EPIC-INTEGRATION-CUTOVER.

## 7. Success criteria

The epic is "done" when **all** of the following hold:

- **SC-1 (vertical-slice gate):** `uv run chirp ask "hello"` on a developer machine with `mlx-community/gemma-4-4b-it-4bit` (or any other 4-bit chat MLX model) in `~/.cache/huggingface/hub/` and a manually-authored `models.toml` referencing it produces a streamed answer over stdout. The daemon was lazy-spawned by the client (not pre-started). No Ollama process was contacted during the run.
- **SC-2 (FR coverage):** FR1, FR2, FR3, FR4, FR5, FR6, FR7, FR8, FR9, FR10, FR11, FR12, FR13, FR15, FR16, FR17, FR18, FR19, FR20, FR21, FR22, FR23 are implemented and demonstrated by tests. FR14 (immediate exit on version mismatch) is demonstrated by an integration test that spawns a daemon, then connects with a deliberately bumped client version, and asserts `version_mismatch` + daemon process exits within 1 s.
- **SC-3 (performance budgets, smoke-tested):** On the same machine used for SC-1, a warm-path `chirp ask` (chat model already loaded) produces its first token within 500 ms (NFR-P1, hand-timed acceptable for this epic — automated benchmarking is deferred). A cold-path `chirp ask` (no model loaded) produces its first token within 5 s on M2 / 16 GB or 8 s on M1 (NFR-P2).
- **SC-4 (cancellation):** A test that issues a long-running `chat` and then sends `cancel` halts generation and frees the model lock within 200 ms (NFR-P4). Verified with `FakeBackend` in unit tests; smoke-tested with the real backend in manual verification.
- **SC-5 (concurrency):** A test that launches a long-running streaming `chat` plus a parallel `embed` (against a pinned embed model) sees both complete correctly. The chat model stays resident through both.
- **SC-6 (version handshake):** A test that spawns a daemon with version `X`, then a client with version `Y ≠ X`, observes the daemon exit, the client respawn a daemon (now version `Y`), and the original request complete on retry. User-visible pause ≤ 2 s on M2 (NFR-R5).
- **SC-7 (single-instance):** A test that races two `chirpd` subprocesses sees exactly one acquire the flock and survive; the other exits 0 within 1 s (NFR-R4).
- **SC-8 (no daemon network traffic):** A test (or manual `lsof -i -P -n -p <chirpd-pid>`) confirms `chirpd` opens no network sockets after startup (NFR-S3). The daemon's `huggingface_hub.snapshot_download(..., local_files_only=True)` call is verified to not hit the network.
- **SC-9 (socket permissions):** After daemon startup, `stat -f "%Lp" ~/Library/Application\ Support/chirp/chirpd.sock` returns `600` (NFR-S2).
- **SC-10 (coverage):** Line coverage on `chirpd/` and `llm/` ≥ 90% via `make test-coverage` (NFR-M1), measured against the unit-test surface only. LaunchAgent code is not in this epic; no exclusion needed here.
- **SC-11 (quality gates):** `make check` and `make test` pass on every commit that touches code in scope (NFR-M2). `mypy` passes against `chirpd/` and `llm/` (NFR-M3).
- **SC-12 (logs are clean and rotating):** `~/Library/Logs/chirp/chirpd.log` contains logfmt lines after a smoke run, contains zero prompt/message/note content (NFR-S5), and a synthetic test that writes >10 MB of log output rotates to a `.1` generation (NFR-O2).

## 8. Out of scope (deferred to other epics)

- **`chirp models` Typer subcommand group, `chirp models add/list/remove/default/pull/show`, HuggingFace downloads with progress bars, role inference, atomic registry writes.** Owned by **EPIC-MODEL-REGISTRY**. This epic only delivers the read-only `llm.registry.read_registry()` function so the daemon can resolve aliases. The registry file itself must exist on disk for the vertical slice — story 3.7 documents the temporary manual-authoring step (a single example `models.toml` committed in the smoke-test instructions).
- **`chirp daemon` Typer subcommand group (`status`, `start`, `stop`, `restart`, `enable`, `disable`, `logs`), LaunchAgent install/uninstall via `plistlib` + `launchctl`, log rotation policy beyond the basic rotating handler, `chirp daemon logs -f` tailing UX.** Owned by **EPIC-DAEMON-LIFECYCLE**. This epic sets up a basic logfmt-formatted rotating file handler in story 3.3 (per NFR-O1, NFR-O2). The user-facing daemon-management CLI is out of scope here.
- **Migration of `notes/note_generator.py`, `notes_chat/retrieval.py`, `notes_chat/index.py`, full `notes_chat/prompting.py` to `llm.client`. Regression-corpus capture and blinded A/B comparison.** Owned by **EPIC-INTEGRATION-CUTOVER**. Story 3.7 cuts over a single command (`chirp ask`) as the vertical-slice gate; the broader sweep across `notes` / `notes_chat` is out of scope here.
- **Removal of the Ollama Python client from `pyproject.toml`, removal of `ollama_url` config field, `chirp init` updates (Ollama detection, migration plan, Apple-Silicon fail-fast at init time, LaunchAgent prompt), README / AGENTS.md / `chirp --help` docs sweep.** Owned by **EPIC-INIT-AND-MIGRATION**. This epic *adds* `mlx-lm` and `huggingface_hub` in story 3.1 but does **not** remove `ollama` — the Ollama client must remain installable for the parallel call sites that have not yet been cut over.
- **`chirp models add` / `pull` HuggingFace network calls.** `llm/hf.py` (network-touching wrapper around `huggingface_hub.snapshot_download` with full download + progress) is EPIC-MODEL-REGISTRY's surface. The daemon's read-only HF cache lookup via `local_files_only=True` is this epic's only `huggingface_hub` integration.
- **`chirp models add`'s post-add warm via `client.warm()`.** The client surface this would call (`client.model_load`) exists after story 3.5, but the warm-on-add flow itself is EPIC-MODEL-REGISTRY.
- **Apple-Silicon fail-fast at `chirp init`.** The defensive check at `chirpd/__main__.py` startup is in scope here (story 3.3); the user-facing `chirp init` check that catches the case earlier (FR54) is EPIC-INIT-AND-MIGRATION.
