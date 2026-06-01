# Epic: `chirp models` — HuggingFace-backed model registry for the MLX daemon

- **Epic ID:** EPIC-MODEL-REGISTRY
- **Owner:** Colby
- **Status:** Done — all six stories (4.1–4.6) complete; SC-1..SC-13 verified end-to-end on Apple Silicon (macOS 26.5, arm64) on 2026-05-31. See **§7 → Smoke evidence**. Pending merge of PR #72 (story 4.6 wiring) to land on `main`.
- **Created:** 2026-05-15
- **Design source:** `_bmad-output/planning-artifacts/prd.md` (§Functional Requirements FR24–FR38, §CLI Tool Specific Requirements, §User Journeys — Maya, Priya); `_bmad-output/planning-artifacts/architecture.md` (§Configuration & Persistence, §Project Structure → `llm/` package, §Implementation Patterns → CLI Output Patterns, §Model Registry Read/Write); `_bmad-output/planning-artifacts/implementation-readiness-report-2026-05-12.md` (Minor concerns: `models.toml` created on first use)
- **Related branch (current work):** `feat/story-4.6-models-typer-wiring` (PR #72)

## 1. Goal

Give Chirp users a single, complete command surface for the LLM half of the product: `chirp models add <hf-repo>` downloads an MLX model from HuggingFace, registers it in a local TOML registry, sets it as the default for its role if none is configured, and warms it on the daemon — all in one command. Once registered, models are inspected with `chirp models list` / `show`, swapped with `chirp models default <alias>`, refreshed with `chirp models pull <alias>`, and dropped with `chirp models remove <alias>`.

After this epic, the Maya journey from the PRD is shippable: a fresh user runs `chirp models add mlx-community/gemma-4-4b-it-4bit` and ends in a usable state without any follow-up commands. The Priya journey is shippable too: a power user registers a second model with `--alias fast`, flips the default with `chirp models default fast`, and inspects both with `chirp models list`.

The epic delivers the **writer half** of the `models.toml` registry, the **HuggingFace integration boundary** (`llm.hf`), and the **`chirp models` Typer subcommand group**. The reader half of the registry, the daemon, and the `model.load`/`model.list` daemon ops are owned by EPIC-CHIRPD-CORE — this epic consumes those contracts via the `llm.client` library.

## 2. Why now

`chirp models` is the only user-facing way to install a model after Ollama is gone. Until this epic lands, a chirp user with the new daemon has no path from "I ran `pip install chirp`" to "I have a model I can `ask` against." It is independent of CHIRPD-CORE's daemon implementation work in the sense that the registry file, the HF download path, and the CLI commands can all be written and unit-tested with `FakeBackend` or a mocked `llm.client` — but it depends on CHIRPD-CORE's `model.load` and `model.list` daemon-op contracts being stable so that auto-warm and loaded-state rendering work end-to-end.

Sequencing this epic in parallel with CHIRPD-CORE (after the protocol + client + `model.load` contract are landed but possibly before the daemon's inference path is finalized) keeps the critical path short and lets the integration-cutover epic consume both deliverables together. Both Maya's and Priya's journeys in the PRD become testable as soon as this epic completes; until then, no journey involving the new daemon can be exercised end-to-end on a clean machine.

## 3. Locked decisions from the architecture

| # | Decision | Source |
|---|----------|--------|
| 1 | Registry format is TOML, parsed with stdlib `tomllib` (read) and written with `tomli_w`. Pydantic models gate the file shape. | Architecture §Configuration & Persistence |
| 2 | Registry lives at `~/Library/Application Support/chirp/models.toml`. The file (and its parent directory) is created on first write — there is no separate "init the registry" step. | Architecture §Configuration & Persistence; Readiness report §Minor concerns |
| 3 | Writes are atomic: `tomli_w.dumps(...)` to `models.toml.tmp` in the same directory, then `os.replace(...)` to `models.toml`. Never partial writes. | Architecture §Model Registry Read/Write |
| 4 | Schema versioning is alpha-simplified. The `schema_version` field is written (`= 1`) for forward use, but the reader rejects unknown versions with a typed error — no migration pipeline in MVP. | Architecture §Configuration & Persistence; PRD §Alpha-stage simplifications |
| 5 | All HuggingFace network calls live in `llm/hf.py`. No other module imports `huggingface_hub` directly. The daemon's local-cache lookup (also `huggingface_hub` but `local_files_only=True`) is in CHIRPD-CORE's scope, not this epic's. | Architecture §HuggingFace boundary |
| 6 | Role inference (`chat` vs. `embed`) reads HF repo metadata (`HfApi.repo_info` tags and `config.json` `model_type`/architecture markers); falls back to requiring `--role` when ambiguous. | PRD FR32; Architecture §Requirements-to-Structure Mapping |
| 7 | Alias is inferred from the repo's basename, lowercased, with `<org>/` stripped. `--alias` overrides. Example: `mlx-community/gemma-4-4b-it-4bit` → `gemma-4-4b-it-4bit`. | PRD §User Journeys (Maya, Priya); FR24/FR25 |
| 8 | `chirp models add` validates the repo, infers role, downloads, writes the registry, then warms the model. Failure in any pre-write step aborts before the registry is touched. Warm failure does **not** roll back the registry write — the user can retry with `chirp models pull` or diagnose with `chirp daemon logs`. | PRD §User Journeys (Maya); FR34 |
| 9 | Default-for-role is set automatically when a newly-added model is the first of its role (FR33). `chirp models default <alias>` flips the default for whichever role the alias is registered as — setting a chat alias as `default_embed` (or vice versa) is rejected. | FR27, FR33 |
| 10 | `chirp models list`, `chirp models show`, and any other reader-side command emits Rich on TTY and a single JSON document on stdout when `--json` is set. No log lines on stdout in JSON mode. | Architecture §Implementation Patterns → CLI Output Patterns; PRD FR31 |
| 11 | Tab completion on alias arguments uses Typer's `shell_complete` callback. The callback reads `models.toml` directly (the same Pydantic reader CHIRPD-CORE ships) and returns the list of registered aliases. | PRD §Scripting Support |
| 12 | The daemon — not the CLI — owns raw `<org>/<repo>` resolution (FR36, "alias not in registry"). The registry does not need a special row for raw repos. | Architecture §Requirements-to-Structure Mapping |
| 13 | `tomli_w` is added to `pyproject.toml` as a direct dependency. `huggingface_hub` is added as a direct dependency (rather than relied on transitively through `mlx-lm`) so the version pin is explicit. | Architecture §Gap Analysis — Minor clarifications |
| 14 | HF cache layout is **not** owned by chirp. `llm/hf.py` calls `huggingface_hub.snapshot_download` with default cache discovery (`~/.cache/huggingface/hub`, respecting `HF_HOME`). Chirp does not move, mirror, or curate the cache. | PRD FR38 |

## 4. Research findings — what already exists vs. what is missing

Validated against `story/2.3-blackhole-removal` at the current HEAD.

### Today's model-management surface (Ollama-era)

- **`config/settings.py`** carries an `[llm]` section keyed for Ollama (`OLLAMA_HOST`, model id strings like `llama3.2`). After EPIC-CHIRPD-CORE rewrites this to point at the daemon socket and idle timer, model-id strings still exist but mean "alias in `models.toml`."
- **`notes/note_generator.py`** and **`notes_chat/retrieval.py`** call the Ollama Python client with a hard-coded model name from settings. They are owned by EPIC-INTEGRATION-CUTOVER, not this epic.
- **No `chirp models` command exists today.** No code path writes a registry file. No code path downloads HuggingFace weights. No code path infers a model's role.
- **No `models.toml` exists** on any user's machine. The file is introduced by this epic the first time `chirp models add` runs.

### What this epic delivers (net new)

- **`llm/registry.py`** — Pydantic models for `models.toml`, the `write_registry(...)` atomic-write helper, and the alias-inference / role-validation helpers used by the CLI. The **reader** (`read_registry()`) lives in CHIRPD-CORE story 3.5; this epic consumes that reader.
- **`llm/hf.py`** — single module that owns all `huggingface_hub` calls. Two responsibilities: validate-and-fetch (`HfApi.repo_info` + `snapshot_download` with progress) and role inference (architecture tags / `config.json` heuristic).
- **`llm/cli/models.py`** — the six `chirp models` subcommands (`add`, `list`, `remove`, `default`, `pull`, `show`) plus a Typer completion callback that reads aliases from `models.toml`.
- **`chirp/cli.py`** registration — one `app.add_typer(...)` call adds the `models` subapp to the visible `MODELS_PANEL`.

### What this epic does **not** deliver

- The daemon process, the inference path, the protocol envelope, the `model.load` daemon op (CHIRPD-CORE).
- The `llm.client` library itself (CHIRPD-CORE) — this epic *uses* it for auto-warm and loaded-state queries.
- The `chirp daemon` subcommand group (DAEMON-LIFECYCLE).
- `chirp init` updates that prompt the user to run `chirp models add` when no chat default exists (INIT-AND-MIGRATION; consumes the registry reader to make that check).
- Routing existing `notes` / `notes_chat` modules through `llm.client` (INTEGRATION-CUTOVER).
- Removing the Ollama Python dependency (INIT-AND-MIGRATION; lands after cutover).

### Net code delta (rough)

- **Add:** ~250 lines Python across `llm/registry.py` (Pydantic + atomic write + helpers), `llm/hf.py` (HF wrapper + role inference), and `llm/cli/models.py` (six subcommands + completion); ~3 lines in `chirp/cli.py` to register the subapp; one new test module per added module (~400 lines tests total).
- **Modify:** `pyproject.toml` (add `tomli_w`, `huggingface_hub` direct dep).
- **Remove:** nothing in this epic — the Ollama removal lives in INIT-AND-MIGRATION after cutover.

## 5. Stories

Execution order matters: the registry writer is the foundation everything else needs; the HF module is the second foundation; the `add` command pulls both together; `list`/`show`/`default`/`remove`/`pull` are simpler reads + small writes; the Typer wiring closes the epic by making the surface visible on `chirp --help`.

| ID | Title | Depends on | File |
|----|-------|------------|------|
| 4.1 | `models.toml` Pydantic model + atomic write helper | CHIRPD-CORE 3.5 (reader contract) | [stories/4.1-registry-writer.md](stories/4.1-registry-writer.md) |
| 4.2 | `llm/hf.py` — HuggingFace download wrapper + role inference | — | [stories/4.2-hf-integration.md](stories/4.2-hf-integration.md) |
| 4.3 | `chirp models add` — validate, download, register, warm | 4.1, 4.2, CHIRPD-CORE 3.6 (`llm.client` + `model.load` op) | [stories/4.3-models-add.md](stories/4.3-models-add.md) |
| 4.4 | `chirp models list` — render registry + loaded state | 4.1, CHIRPD-CORE 3.7 (`model.list` op) | [stories/4.4-models-list.md](stories/4.4-models-list.md) |
| 4.5 | `chirp models show` / `default` / `remove` / `pull` | 4.1, 4.2 (for `pull`), CHIRPD-CORE 3.6 (for `pull` auto-warm) | [stories/4.5-models-show-default-remove-pull.md](stories/4.5-models-show-default-remove-pull.md) |
| 4.6 | Typer registration + alias tab completion | 4.3, 4.4, 4.5 | [stories/4.6-typer-wiring-and-completion.md](stories/4.6-typer-wiring-and-completion.md) |

CHIRPD-CORE story numbers (3.5, 3.6, 3.7) are placeholders for the corresponding deliverables in that epic; they are referenced here so the dependency is explicit. If CHIRPD-CORE renumbers, this epic's "depends on" pointers update without changing the work.

## 6. Sequencing & dependencies

**Upstream** (must land before stories under this epic become demonstrable):

- **EPIC-CHIRPD-CORE** ships the `llm.client` library with lazy-spawn and version-mismatch retry, the `model.load` daemon op (for auto-warm), the `model.list` daemon op (for `chirp models list` loaded-state rendering), and the `llm.registry.read_registry()` reader function. Once those four pieces are stable (not necessarily fully implemented end-to-end against MLX — `FakeBackend` is enough for this epic's tests), 4.1 through 4.6 can land in order. Story 4.1 can in principle start in parallel with CHIRPD-CORE's protocol work; the reader contract just needs to be agreed on the Pydantic schema before either side codes against it.

**Downstream** (consumes this epic's deliverables):

- **EPIC-INTEGRATION-CUTOVER** uses the registry indirectly: when `notes` / `notes_chat` ask the daemon to chat or embed, the daemon resolves aliases from `models.toml` (CHIRPD-CORE's reader path) which this epic's writers populated.
- **EPIC-INIT-AND-MIGRATION** uses `llm.registry.read_registry()` (CHIRPD-CORE's API) to check whether a default chat model is registered during `chirp init`; if none, the init flow prints the `chirp models add` prompt the Maya journey requires.

**Internal sequencing rule:** every story under this epic must be independently demonstrable. 4.1 demonstrates by writing a hand-rolled registry from a pytest fixture and round-tripping it through the reader. 4.2 demonstrates against a mocked `HfApi` and `snapshot_download`. 4.3 demonstrates end-to-end with `FakeBackend` substituted for MLX. 4.4 / 4.5 / 4.6 all demonstrate against a `FakeBackend`-backed daemon spawned in-process by tests.

## 7. Success criteria

End-to-end on a clean Apple Silicon Mac with CHIRPD-CORE landed and `make dev-install` complete:

- **SC-1** `uv run chirp --help` shows a `Models` panel containing `models` as a visible subcommand group. `uv run chirp models --help` shows six subcommands: `add`, `list`, `remove`, `default`, `pull`, `show`.
- **SC-2** `uv run chirp models add mlx-community/gemma-4-4b-it-4bit` on a machine with no prior chirp model installed:
  1. Validates the repo exists on HuggingFace,
  2. Streams a progress bar while `huggingface_hub.snapshot_download` runs,
  3. Writes `~/Library/Application Support/chirp/models.toml` (creating the parent directory) with the alias `gemma-4-4b-it-4bit`, role `chat`, `default_chat = "gemma-4-4b-it-4bit"`,
  4. Lazy-spawns `chirpd` and issues `model.load`, surfacing the load result,
  5. Exits 0 with a one-line "Ready." status.
- **SC-3** `cat ~/Library/Application Support/chirp/models.toml` is valid TOML. `schema_version = 1`, `default_chat` and `[models."gemma-4-4b-it-4bit"]` keys present with the expected values.
- **SC-4** `uv run chirp models list` on the resulting state prints a Rich table with at least the columns `alias`, `role`, `default`, `loaded`, `hf_repo`, with the registered alias marked as the chat default and (if the daemon is still warm) loaded.
- **SC-5** `uv run chirp models list --json` prints a single JSON document to stdout with no Rich rendering, parseable by `jq -e .`.
- **SC-6** `uv run chirp models add mlx-community/Llama-3.2-3B-Instruct-4bit --alias fast` registers a second chat model. Because a chat default already exists, the new model is **not** auto-promoted. `uv run chirp models default fast` flips `default_chat` to `fast`. `uv run chirp models list` confirms.
- **SC-7** `uv run chirp models show fast --json` prints alias, hf_repo, role, options (if any), and resolved cache path on disk.
- **SC-8** `uv run chirp models remove fast --purge` removes the entry from `models.toml` **and** deletes the model's HF cache directory. The other model remains registered; `default_chat` is unchanged (still `gemma-4-4b-it-4bit`).
- **SC-9** `uv run chirp models pull gemma-4-4b-it-4bit` re-runs `snapshot_download` against the registered repo; on a healthy cache it's a no-op (HF prints "files already up to date"); on a corrupted/missing cache it re-downloads.
- **SC-10** Tab completion: `uv run chirp models default <TAB><TAB>` (assuming Typer completion is installed via `chirp --install-completion`) lists `gemma-4-4b-it-4bit` (and any other registered alias).
- **SC-11** Atomic write resilience: a pytest test that kills the writer mid-flight (between `tomli_w.dumps` and `os.replace`) leaves the existing `models.toml` intact, with no partial `.tmp` artifact visible to the reader.
- **SC-12** `uv run pytest tests/llm/test_registry.py tests/llm/test_hf.py tests/llm/test_cli_models.py` passes with the new modules at ≥ 90% line coverage (NFR-M1).
- **SC-13** `uv run ruff check .` and `uv run mypy chirp llm` (or whatever the project's mypy invocation is for `llm/`) report no issues.

### Smoke evidence

Verified end-to-end on 2026-05-31, Apple Silicon (arm64, macOS 26.5), `mlx_lm` present, HuggingFace reachable, branch `feat/story-4.6-models-typer-wiring`, starting from no `models.toml`. **All 13 SCs passed.**

Because the PRD's `mlx-community/gemma-4-4b-it-4bit` does not exist on HF and `Llama-3.2-3B-Instruct-4bit` was not cached, the run substituted already-cached models to exercise the same code paths without multi-GB downloads: primary chat = `mlx-community/Llama-3.2-1B-Instruct-4bit` (alias `llama-3.2-1b-instruct-4bit`); second model = `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (alias `fast`).

| SC | Evidence |
|----|----------|
| SC-1 | `chirp models --help` lists `add`/`list`/`show`/`default`/`remove`/`pull`; `chirp --help` shows the `Models` panel. |
| SC-2 | `chirp models add …Llama-3.2-1B…` → validate → cache-hit download → lazy-spawned `chirpd` → `model.load` → `Ready.` in ~3.6 s. |
| SC-3 | `models.toml` written with header comment, `schema_version = 1`, `default_chat`, and the `[models."…"]` entry; valid TOML. |
| SC-4 | `chirp models list` Rich table with `alias`/`role`/`default`/`loaded`/`hf_repo`; default ★, loaded ●. |
| SC-5 | `chirp models list --json` single JSON doc, `jq -e .` exit 0, `daemon_reachable`/`loaded` true. |
| SC-6 | Adding a second model with a chat default already set did **not** auto-promote; `chirp models default fast` flipped `default_chat`. |
| SC-7 | `chirp models show fast --json` returned alias/hf_repo/role/options/resolved `cache_path`. |
| SC-8 | `chirp models remove fast --purge` dropped the entry and deleted the HF cache dir; `llama` entry and `default_chat` unchanged. |
| SC-9 | `chirp models pull` on a healthy cache was a ~0.33 s no-op (`cache hit`), then re-warmed. |
| SC-10 | `_complete_alias` returned the registered aliases against the live registry; the real zsh completion protocol (`_CHIRP_COMPLETE=complete_zsh`) returned both aliases with prefix filtering. |
| SC-11 | `test_write_is_atomic_on_replace_failure` passes (pre-existing file intact, no `.tmp` leftover). |
| SC-12 | `pytest test_registry.py test_hf.py test_cli_models.py` → 154 passed; coverage `registry` 96% / `hf` 98% / `cli.models` 100% (99% total). |
| SC-13 | `ruff check .` clean; `mypy chirp llm` clean. |

## 8. Out of scope / deferred

- **Daemon process, protocol, inference, `model.load`/`model.list`/`model.status` daemon ops, `llm.client` library, `llm.registry.read_registry()` reader, `LLMBackend` protocol, `FakeBackend`.** Owned by EPIC-CHIRPD-CORE.
- **`chirp daemon` subcommand group** (`status`, `start`, `stop`, `restart`, `enable`, `disable`, `logs`) and LaunchAgent plumbing. Owned by EPIC-DAEMON-LIFECYCLE.
- **Cutover of `notes` / `notes_chat` / `chirp ask` to route through `llm.client`.** Owned by EPIC-INTEGRATION-CUTOVER. This epic does not edit `notes/note_generator.py` or `notes_chat/retrieval.py`.
- **`chirp init` changes**, including the "no chat default registered" prompt that points users to `chirp models add`, the Ollama detection in `--recheck`, and the Intel fail-fast. Owned by EPIC-INIT-AND-MIGRATION (it consumes the reader path from CHIRPD-CORE story 3.5 to make the "is a chat default registered" check).
- **Schema migrations for `models.toml`.** Alpha-simplified: the reader rejects unknown `schema_version` with a typed error. Users re-init the registry across breaking schema changes. (PRD §Alpha-stage simplifications.)
- **Per-task model overrides** (different model for `ask` vs. `transcribe`). Growth feature per PRD §Product Scope.
- **Multi-model concurrent loading.** Growth feature per PRD §Product Scope. `chirp models default` flips the single default; the daemon's loaded-state surface (FR4, `model.list` op) is owned by CHIRPD-CORE.
- **Custom prompt profiles in the registry** (named `[options]` presets). Growth feature per PRD §Product Scope. This epic supports a per-model `options` table in `models.toml` because the PRD §Config Schema specifies it, but the CLI for editing options is deferred — users hand-edit `models.toml` in MVP and re-run `chirp models add` overwrites only the keys it sets.
- **Removing the Ollama Python client from `pyproject.toml`.** Owned by EPIC-INIT-AND-MIGRATION (lands after cutover so a side-by-side comparison is possible).
- **Hot-reload of `models.toml` while the daemon is running.** Explicitly out of scope per PRD §Explicit Out of Scope. The daemon re-reads on `model.*` ops; manual edits between ops are visible to the next op without requiring restart, but file-watcher complexity is not introduced.

## 9. Risks

- **HuggingFace API churn.** `huggingface_hub.HfApi` and `snapshot_download` are reasonably stable, but call signatures occasionally shift between minor versions. Mitigation: pin a known-good minimum version in `pyproject.toml` (story 4.2 sets this); unit tests mock at the `HfApi` / `snapshot_download` call seam, so API surface changes break tests loudly rather than silently.
- **Role-inference ambiguity.** Some HF repos lack the architecture tags or `config.json` markers we need to distinguish chat from embed (especially community uploads). Mitigation: when inference is ambiguous, require `--role` explicitly with a clear error pointing the user at the flag. Story 4.2 lists the heuristic explicitly; unit tests cover both branches (clear chat, clear embed, ambiguous-requires-flag).
- **Auto-warm failure UX.** If `chirp models add` writes the registry successfully but `model.load` fails (e.g., unsupported MLX architecture), the user is left with a registered model that can't be used. Per the locked decision in §3, this does **not** roll back the registry — the user can `chirp models pull` to repair weights or `chirp models remove` to drop it. The CLI must distinguish download-success-but-load-failure clearly. Story 4.3 owns this error-message clarity.
- **Atomic-write race vs. daemon read.** Daemon may re-read `models.toml` on a `model.load` while the CLI is in the middle of writing. Mitigation already in §3: `tomli_w.dumps` to `.tmp` in the same directory then `os.replace` makes the swap atomic on macOS (POSIX). Daemon sees either the old or new state, never a half-written file. Story 4.1 includes a test that asserts a partial write (kill between dumps and replace) leaves the old file intact.
- **HF cache permissions / disk full.** `snapshot_download` can fail with `OSError` mid-download. Mitigation: `llm.hf` catches and re-raises with a typed `LLMModelError` subclass (or a registry-side equivalent) that the CLI converts to a clear stderr message. Story 4.2 lists the error mapping.
- **Schema version pin.** Setting `schema_version = 1` now without a migration path means any v2 schema change will require users to delete and re-create their registry. Acceptable for alpha (PRD §Alpha-stage simplifications) but worth a one-line note in the file header that says so. Story 4.1 emits a TOML header comment explaining the alpha constraint.
- **Tab completion file read latency.** The completion callback runs synchronously on every `<TAB>` press and reads `models.toml` from disk. For a registry with a handful of aliases this is sub-millisecond; for pathological cases (~10k aliases) it could lag. Out of scope: no realistic user will register more than a dozen models. Story 4.6 caps the file size sanity-check in the reader (CHIRPD-CORE's reader is the gate) at something reasonable like 1 MB.
