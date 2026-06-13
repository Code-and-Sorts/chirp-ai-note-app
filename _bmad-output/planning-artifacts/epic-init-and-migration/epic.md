# Epic: Clean `chirp init` of Ollama and finish the migration (init flow, Apple-Silicon gate, deps, docs)

- **Epic ID:** EPIC-INIT-AND-MIGRATION
- **Owner:** Colby
- **Status:** Done — all five stories complete (7.1 init rewrite, 7.2 migration plan, 7.3 LaunchAgent offer, 7.4 ollama dep removal, 7.5 docs sweep + the deferred prompting.py cutover). `chirp init` is Ollama-free with the Apple-Silicon gate, chirpd readiness, and registry-backed model checks; the `ollama` dependency is gone; docs and CLI surfaces no longer reference Ollama. The Ollama → MLX migration is complete.
- **Created:** 2026-05-15
- **Design source:** [`prd.md`](../prd.md) §First-Run/Init/Migration (FR49–FR54) + §Documentation and Dependency Surface (FR55–FR56); [`architecture.md`](../architecture.md) §Project Structure → `chirp/init_flow.py — MODIFIED`; readiness review (2026-05-12) §`pyproject.toml` split decision
- **Related branch (current work):** TBD

## 1. Goal

After this epic, `chirp init` no longer mentions Ollama in any phase or branch. Fresh users on Apple Silicon Macs are guided from a clean `pip install chirp` to a working `chirp ask` in two visible steps — `chirp init` and `chirp models add <recommended>` — without ever installing or starting a third-party daemon.

Users migrating from a pre-`chirpd` chirp version see a friendly, multi-line migration plan on `chirp init --recheck` that detects their existing Ollama install, points at `chirp models add` for the new model, and **leaves their Ollama install untouched** (explicit PRD out-of-scope).

Intel Macs fail loudly at `chirp init` with a distinct exit code and a message naming the constraint, instead of getting half-way through a setup that will never work.

The `ollama` Python client is removed from `pyproject.toml`, and every Ollama reference in README, AGENTS.md, CLAUDE.md, `chirp --help`, and the Makefile is replaced with the new `chirp models` / `chirp daemon` story.

## 2. Why now

This epic lands **last** in the Ollama → MLX migration sequence. Two of its outcomes — removing the `ollama` Python client from `pyproject.toml` (story 7.4) and the documentation sweep (story 7.5) — are final-cleanup work that must wait until EPIC-INTEGRATION-CUTOVER has cut every Ollama call site over to `llm.client`. Removing the dependency before then would break `make check` and `make test`.

The init-flow updates themselves (stories 7.1, 7.2, 7.3) can land slightly earlier in the sequence, but only once `llm.client.health()` exists (EPIC-CHIRPD-CORE), `chirp models add` exists (EPIC-MODEL-REGISTRY), and `chirp daemon enable` exists (EPIC-DAEMON-LIFECYCLE), so that the new init flow has working commands to point users at.

Until this epic closes, the PRD's headline outcome — "zero install steps beyond `pip install chirp`" — is not yet visible to users. That visibility lives entirely in this epic's surface area: the init flow, the dependency manifest, and the docs.

## 3. Locked decisions from PRD/architecture

| # | Decision | Source |
|---|----------|--------|
| 1 | `chirp init` drops every Ollama branch — verify, install, model-pick, model-pull, `ollama serve` startup. None of `_ollama_installed`, `_ollama_models`, `_model_installed`, `_pull_model`, the brew-install-ollama task, the `ollama isn't running` error, or the CHAT/EMBEDDING `ModelOption` lists survive. | PRD FR49; architecture §Project Structure ("MODIFIED — drops Ollama checks") |
| 2 | `chirp init` runs an Apple-Silicon check (`platform.machine() == "arm64"`) at the **start** of Phase 1 before any other check. On non-arm64, fail fast with a clear message and a **distinct exit code (`7`)** so scripts can distinguish. | PRD FR54, NFR-C2; architecture §Architecture Validation Results → "Apple Silicon enforcement points" |
| 3 | `chirp init` adds a daemon-readiness check via `llm.client.health()`. Lazy-spawn handles the "no daemon running yet" case automatically; the check returns OK or surfaces a typed error if the daemon couldn't be spawned. **No "is the daemon process running" logic outside `llm.client`** — the client encapsulates this. | PRD FR51; architecture §Decision: Daemon entrypoint + §Cross-Component Dependencies |
| 4 | `chirp init` detects whether a default chat model is registered via `llm.registry.read_registry()`. If `default_chat` is unset (or its alias is missing from the registry), surface a clear next-step prompt referencing `chirp models add <recommended-hf-repo>`. | PRD FR50, OQ1 (recommend a specific model in the prompt) |
| 5 | `chirp init` offers (but does not require) LaunchAgent install via a single yes/no prompt. Default to **no** so users who just want to try chirp aren't surprised by a new login agent. Persist the user's choice so `--recheck` doesn't re-prompt unless asked. | PRD FR52 |
| 6 | `chirp init --recheck` detects a pre-existing Ollama install (heuristic: `shutil.which("ollama")`, `~/.ollama` directory, or `ollama` Python client importable) and prints a **loud / multi-line** migration plan. Open Question OQ6 is resolved in favor of "loud" per Devon's journey. | PRD FR53, Journey 2 (Devon), OQ6 (resolved here) |
| 7 | The migration plan is informational only. `chirp init` **does not** run `brew uninstall ollama`, **does not** `rm -rf ~/.ollama`, **does not** modify `OLLAMA_HOST`, and **does not** stop the user's Ollama service. The recommended manual cleanup is documented in the printed plan. | PRD §Project Scoping → Out of Scope ("auto-uninstall of Ollama"); Journey 2 ("Ollama itself is left installed") |
| 8 | The `--switch-model` flag on `chirp init` is preserved but routes through `chirp models default <alias>` semantics (or `chirp models add <hf-repo>` when the registry is empty). It no longer touches Ollama's model-pull path. | EPIC-WF-ALIGN locked decision 8; PRD §CLI Tool Specific Requirements |
| 9 | The `ollama` Python client is removed from `pyproject.toml` **only after** EPIC-INTEGRATION-CUTOVER has cut every import site. Story 7.4 declares this dependency explicitly; the lockfile (`uv lock`) is refreshed in the same commit. Adding `mlx-lm` and `huggingface_hub` is **not** in this epic — that lands in EPIC-CHIRPD-CORE story 3.1. | Readiness review §Recommendations item 6 ("`pyproject.toml` work to land in two places"); PRD FR55 |
| 10 | Documentation sweep covers Markdown (README, AGENTS.md, CLAUDE.md), the Makefile (Ollama-mentioning targets and help text), `chirp --help` text in `chirp/cli.py` (including the `config` command's `--ollama-url` option and the `ollama_url` field surfaced in `config --list`), and any module-level docstrings. Verified by `grep -ri "ollama" .` returning zero hits outside `.git`, `_bmad-output/`, and history/changelog locations. | PRD FR56; AGENTS.md "Validate doc updates against live CLI help before finishing" |
| 11 | OQ1's recommended chat model in the README and `chirp init` next-step prompt is `mlx-community/gemma-4-4b-it-4bit` as the **primary** recommendation, with `mlx-community/gemma-4-e2b-it-8bit` offered as a **smaller-footprint alternative** (Gemma effective-2B architecture, 8-bit quant — lower resident memory and activation memory than the 4-bit 4B) for tighter-RAM users. Exposed as `RECOMMENDED_CHAT_REPO` and `SMALLER_CHAT_REPO` constants in `chirp/init_flow.py` so future updates are a single-line change. | PRD OQ1 (resolved here for init-prompt consistency; revisit when the model landscape shifts) |

## 4. Research findings — what exists vs. what is missing

Validated against `chirp/init_flow.py`, `chirp/cli.py`, `pyproject.toml`, `README.md`, `AGENTS.md`, and `Makefile` at the current commit on `story/2.3-blackhole-removal`.

### Ollama-touching code that goes away

This mirrors what story 2.3 did for BlackHole — strip Ollama-specific verify/install/prompt logic from `init_flow.py` and surrounding modules. Net delta in `init_flow.py` is comparable to the ~60-line BlackHole strip from story 2.3.

- `chirp/init_flow.py:9-13` — Phase 1–4 docstring referencing Ollama (`verify homebrew, ffmpeg, Ollama, …`, `start Ollama`, `let the user pick chat + embedding models`, `ollama pull the models`).
- `chirp/init_flow.py:63-73` — `CHAT_MODELS` / `EMBEDDING_MODELS` lists of Ollama tags (`llama3.1:8b`, `qwen2.5:7b`, `phi3:mini`, `nomic-embed-text`, …) and the `ModelOption` dataclass that backs them. Replaced by a single recommended-model next-step prompt (decision 4).
- `chirp/init_flow.py:113-138` — `_ollama_installed()` (uses `requests` to hit `http://localhost:11434/api/version`).
- `chirp/init_flow.py:140-149` — `_ollama_models()` (`/api/tags` enumeration).
- `chirp/init_flow.py:152-158` — `_model_installed()` (Ollama-tag prefix matching).
- `chirp/init_flow.py:218-253` — Ollama row in `verify()`'s status list; `ollama_up`/`available` derivation; the per-chat-model and per-embed-model `DependencyStatus` rows; the "will check after ollama is installed" placeholder.
- `chirp/init_flow.py:343-345` — `ollama` brew-install task and `brew services start ollama` task in `install_missing()`.
- `chirp/init_flow.py:382-461` — `pick_models()`, `_pick()`, `keep_or_pick()` (Ollama-tag picker UX) and their helpers. Replaced by the registry-aware next-step prompt.
- `chirp/init_flow.py:463-510` — `pull_and_finalize()`'s `ollama pull` loop and the `_pull_model()`/`_parse_percent()` helpers (L613-663). Replaced by the lazy-spawn + `health()` flow plus a "run `chirp models add <recommended>`" hint when the registry is empty.
- `chirp/init_flow.py:698-703` — the `ollama isn't running. start it with brew services start ollama` red-error branch in `run_init`.
- `chirp/cli.py:1025-1027` — the `--ollama-url` option on the hidden `config` command.
- `chirp/cli.py:1045` — the `Ollama URL: {settings.models.ollama_url}` line in `config --list`.
- `chirp/cli.py:1074-1075` — the `if ollama_url: settings.models.ollama_url = ollama_url` setter branch.
- `config/settings.py:40` — `ollama_url: str = "http://localhost:11434"` (settings field is removed if no other call site survives after EPIC-INTEGRATION-CUTOVER; otherwise, story 7.5 verifies it is gone).
- `Makefile:117` — the `notes_chat.index` smoke-import message `Notes chat may need Ollama running`.
- `Makefile:126-168` — `verify-deps` target's Ollama mentions; `setup-ollama` target (delete entirely); the post-install hint `Install Ollama: make setup-ollama`.

### Ollama references in user-facing / contributor-facing docs

- `README.md:16` — "Generate structured notes with Ollama" feature bullet.
- `README.md:26` — "[Ollama](https://ollama.com) for note generation and retrieval" prerequisite bullet.
- `README.md:79` — `chirp init` row in the command-overview table mentions "model selection" — copy is fine, but verify against new init UX.
- `README.md:132` — "It verifies Homebrew, `ffmpeg`, Ollama, and your configured models …" setup-details paragraph.
- `README.md:139-148` — the manual-setup block (`brew install ffmpeg ollama`, `ollama serve`, `ollama pull llama3.1:8b`, `ollama pull nomic-embed-text`).
- `README.md:185` — "Make sure Ollama is running and the configured models are installed." troubleshooting paragraph.
- `AGENTS.md:21` — "Reuse fixtures to isolate Ollama, audio devices, and filesystem state."
- `CLAUDE.md` — currently no Ollama references (verified). The Claude-specific reminders section may need a one-line nudge toward the new model-management story, but no deletions.

### Existing `pyproject.toml` line to remove

- `pyproject.toml:19` — `"ollama>=0.1.0",` in the `[project] dependencies` list. Removed in story 7.4 after EPIC-INTEGRATION-CUTOVER lands.

### Already implemented (verify only, no rewrite)

- `chirp init`'s 4-phase scaffold with `--recheck` and `--switch-model` flags lives in `chirp/init_flow.py:run_init` (EPIC-WF-ALIGN story 1.5 done). The phase orchestration is reused; only the per-phase content changes.
- The screen-recording-permission Phase 1 check (`_screen_recording_permission`, EPIC-AUDIO-CAPTURE story 2.3 done) stays as-is. Its position in `verify()` is preserved — Apple-Silicon check goes **before** it (architecture-level gate), Ollama-related rows go **away** entirely.
- `llm.client.health()` and `llm.registry.read_registry()` ship in EPIC-CHIRPD-CORE and EPIC-MODEL-REGISTRY respectively; this epic consumes them, does not define them.
- `chirpd.launchd.install_launch_agent()` (or equivalent) ships in EPIC-DAEMON-LIFECYCLE; story 7.3 calls it directly rather than shelling to `chirp daemon enable`.

### Net code delta (rough)

- **Remove:** ~250 lines from `chirp/init_flow.py` (Ollama checks, model-pickers, `ollama pull` flow, "ollama isn't running" branch). Plus `pyproject.toml` line, `Makefile` Ollama mentions (~25 lines), README Ollama paragraphs (~15 lines), AGENTS.md fixture mention (~half-line), the `config --ollama-url` option and `--list` line in `cli.py` (~5 lines).
- **Add:** ~80 lines to `chirp/init_flow.py` (Apple-Silicon gate, `llm.client.health()` probe, registry-aware next-step prompt, Ollama-migration detection block, LaunchAgent yes/no prompt and persistence). Plus README replacement copy (~20 lines) and Makefile-target replacements (~10 lines).
- **Net change:** ~−180 lines of code + ~−10 lines of docs.

## 5. Stories

Execution order: 7.1 → 7.2 → 7.3 can sequence in either order against each other once their dependencies land, but the init-flow shape stabilises fastest if 7.1 lands first. 7.4 is **strictly blocked** on EPIC-INTEGRATION-CUTOVER. 7.5 lands last as the final docs sweep.

| ID | Title | FR coverage | Depends on | File |
|----|-------|-------------|------------|------|
| 7.1 | `chirp init`: drop Ollama branches; add Apple-Silicon and daemon-readiness checks | FR49, FR50, FR51, FR54 | EPIC-CHIRPD-CORE (`llm.client.health()`); EPIC-MODEL-REGISTRY (`llm.registry.read_registry()`, `chirp models add` exists by the time the prompt is printed) | [stories/7.1-init-drop-ollama-add-arm64-and-daemon-checks.md](stories/7.1-init-drop-ollama-add-arm64-and-daemon-checks.md) |
| 7.2 | `chirp init --recheck`: Ollama-detection and loud migration plan | FR53 | 7.1 | [stories/7.2-recheck-ollama-detect-and-migration-plan.md](stories/7.2-recheck-ollama-detect-and-migration-plan.md) |
| 7.3 | `chirp init`: offer LaunchAgent install (opt-in, default no) | FR52 | 7.1; EPIC-DAEMON-LIFECYCLE (`chirpd.launchd` install function and `chirp daemon enable` semantics) | [stories/7.3-init-offer-launchagent.md](stories/7.3-init-offer-launchagent.md) |
| 7.4 | Remove `ollama` from `pyproject.toml`, `uv lock` refresh, prune `make verify-deps` | FR55 (removal half) | **EPIC-INTEGRATION-CUTOVER (must complete in full)** — last `ollama` import gone | [stories/7.4-remove-ollama-from-pyproject.md](stories/7.4-remove-ollama-from-pyproject.md) |
| 7.5 | Docs sweep: README, AGENTS.md, CLAUDE.md, `chirp --help`, Makefile | FR56 | 7.1, 7.2, 7.3, 7.4 | [stories/7.5-docs-sweep-no-ollama-references.md](stories/7.5-docs-sweep-no-ollama-references.md) |

## 6. Sequencing & dependencies

**External epic dependencies:**

- **EPIC-CHIRPD-CORE** — delivers `llm.client` with `health()` and lazy-spawn. Consumed by story 7.1. Story 3.1 of EPIC-CHIRPD-CORE adds `mlx-lm` and `huggingface_hub` to `pyproject.toml`; this epic's story 7.4 only handles the removal of `ollama`.
- **EPIC-MODEL-REGISTRY** — delivers `chirp models add`, `chirp models default`, and `llm.registry.read_registry()`. Consumed by story 7.1 (registry probe + next-step prompt) and story 7.3 (LaunchAgent prompt fires after the user has a working model).
- **EPIC-DAEMON-LIFECYCLE** — delivers `chirpd/launchd.py` install function and the `chirp daemon enable` subcommand. Consumed by story 7.3.
- **EPIC-INTEGRATION-CUTOVER** — **blocks story 7.4**. Until every `import ollama` is gone from the runtime code and the test fixture migration is complete, removing the dependency will break `make check` and `make test`. Story 7.4 explicitly gates on this; the dependency is unmissable in the story header.

**Intra-epic ordering:**

1. **7.1** lands first — establishes the new init shape with the Apple-Silicon gate, the daemon-readiness probe, and the registry-aware model prompt. Most Ollama deletion in `init_flow.py` happens here.
2. **7.2** layers the Ollama-detection migration plan onto `--recheck`. Independent of 7.3.
3. **7.3** adds the LaunchAgent prompt. Sequencing with 7.2 doesn't matter functionally, but 7.3 needs EPIC-DAEMON-LIFECYCLE to be landed first.
4. **7.4** lands once integration cutover completes — removing the `ollama` dep and refreshing the lockfile.
5. **7.5** lands last — sweeps remaining doc references across README/AGENTS/CLAUDE/Makefile/`chirp --help`. Doing this after 7.4 is intentional: removing the dep first guarantees the codebase can't accidentally re-introduce an `import ollama` while the doc sweep is in flight.

## 7. Success criteria

- **Fresh-box (Maya's journey).** On a clean Apple Silicon Mac with no prior chirp install: `pip install chirp-notes-ai && chirp init` prints the verify table with no Ollama row, no model-picker phase, exits Phase 1 cleanly, prompts the user to run `chirp models add mlx-community/gemma-4-4b-it-4bit` (or equivalent), and on a subsequent `chirp init --recheck` shows all rows green and the suggested LaunchAgent prompt (default no). Running the suggested `chirp models add` then `chirp ask "test"` produces a streamed response.
- **Intel Mac fail-fast.** On an Intel Mac: `chirp init` exits with code `7` after printing a single-line message naming the Apple-Silicon requirement. No verify table is printed; no temporary files are created; no daemon is lazy-spawned.
- **Migrating user (Devon's journey).** On a Mac with Ollama installed via Homebrew: `chirp init --recheck` prints the standard verify table **and** a multi-line migration plan that names: (a) the recommended replacement model (`chirp models add mlx-community/gemma-4-4b-it-4bit`), (b) that the user's existing notes / `~/.chirp/config.toml` / chroma index are unchanged, (c) the manual cleanup steps (`brew uninstall ollama`, optional removal of `~/.ollama` GGUF files), and (d) that chirp itself has not modified the user's Ollama install. Re-running `chirp init --recheck` after `brew uninstall ollama` no longer prints the migration plan.
- **No Ollama in dep manifest after cutover.** `grep -i "ollama" pyproject.toml uv.lock` returns zero hits (excluding lockfile comment blocks). `python -c "import ollama"` fails with `ModuleNotFoundError`.
- **No Ollama in user-facing or contributor-facing docs.** `grep -ri --exclude-dir=.git --exclude-dir=_bmad-output --exclude-dir=.docs/history "ollama" .` returns zero hits in `README.md`, `AGENTS.md`, `CLAUDE.md`, `Makefile`, `chirp/`, `config/`, `notes/`, `notes_chat/`, `recorder/`, `transcriber/`, `utils/`, `audio_capture/`, and `tests/`.
- **`chirp --help` is Ollama-free.** `uv run chirp --help`, `uv run chirp init --help`, `uv run chirp config --help`, `uv run chirp config --list`, `uv run chirp models --help`, and `uv run chirp daemon --help` contain zero occurrences of "ollama" (case-insensitive).
- **Quality gates.** `make check` and `make test` pass on every story merge. Coverage on `chirp/init_flow.py` ≥ 90% line per NFR-M1.
- **`make verify-deps` still works.** Now exercises only the Apple-Silicon check, daemon readiness via `llm.client.health()`, and the screen-recording probe — no Ollama mention.

## 8. Out of scope

- **The daemon process itself** (`chirpd` startup, IPC, model lifecycle, version handshake) — EPIC-CHIRPD-CORE.
- **The model registry CLI** (`chirp models add/list/remove/default/pull/show`, `models.toml` writer, HuggingFace integration) — EPIC-MODEL-REGISTRY.
- **The daemon-lifecycle CLI** (`chirp daemon status/start/stop/restart/enable/disable/logs`, LaunchAgent plist generation) — EPIC-DAEMON-LIFECYCLE. Story 7.3 *consumes* the LaunchAgent install function; it does not implement it.
- **Call-site cutover for `notes`, `notes_chat`, and `chirp ask`** (replacing `import ollama` with `from llm.client import client`) — EPIC-INTEGRATION-CUTOVER.
- **Adding `mlx-lm` and `huggingface_hub` to `pyproject.toml`** — EPIC-CHIRPD-CORE story 3.1. This epic only handles the *removal* of `ollama`.
- **Auto-uninstall of the user's Ollama install.** Explicit PRD out-of-scope (§Project Scoping → Out of Scope). The migration plan is informational only — no `brew uninstall ollama`, no `rm -rf ~/.ollama`, no touching `OLLAMA_HOST`, no stopping `brew services`.
- **Migrating GGUF models to MLX.** Users re-download an MLX-format equivalent via `chirp models add`. No conversion tooling.
- **`chirp init` background pre-warm** of the default chat model after registration. Tracked as a Growth feature (OQ4); not adopted in MVP.
- **Hot-reload of `models.toml`** without daemon restart. Daemon reads on `model.*` ops; explicit `chirp daemon restart` is the documented escape valve. Out of MVP per PRD §Project Scoping → Out of Scope.

## 9. Risks

- **`llm.client.health()` not yet available when 7.1 starts.** Mitigation: 7.1's dependency on EPIC-CHIRPD-CORE is hard. If parallel work is needed, gate behind a feature flag or stub the import locally — but the story acceptance criteria require the real call.
- **Ollama-detection heuristic produces a false positive.** A user who once ran a curl-installed Ollama and removed it but still has `~/.ollama` lying around would get the migration plan printed unnecessarily. Mitigation: the plan is informational, prints once per `--recheck`, and references `brew uninstall ollama` only conditionally (story 7.2 details). Worst case: a one-time noisy message on `--recheck`.
- **Apple-Silicon gate misfires under Rosetta.** Running a Rosetta-translated Python on an Apple Silicon Mac would report `platform.machine() == "x86_64"`. Mitigation: error message points the user toward installing an arm64 Python and explains why; documented in the story (7.1).
- **Coverage regressions on `init_flow.py`.** The current file has good coverage; deleting ~250 lines and adding ~80 will shift coverage hotspots. Mitigation: tests for the new flow ship in 7.1 (Apple-Silicon gate, daemon-readiness, registry-empty prompt) and 7.2 (Ollama-detection branches, migration plan output) and 7.3 (LaunchAgent prompt yes/no/persisted-no). Existing init tests are deleted alongside the deleted code paths.
- **`make verify-deps` breaking on partial landing.** If 7.1 lands but 7.4 hasn't, `pyproject.toml` still declares `ollama` as a dep. That's fine — the dep is installed but unused by init. No regression. The reverse (7.4 lands before integration cutover) is the actual hazard and is gated by the explicit story dep.
- **Docs sweep missing an Ollama reference in a generated artifact.** Mitigation: 7.5's verification is a `grep -ri` sweep against the working tree (excluding history and planning artifacts), run as the last task in the story.
