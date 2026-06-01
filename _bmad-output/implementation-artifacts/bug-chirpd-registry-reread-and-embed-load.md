# BUG: chirpd does not re-read the registry per op, and cannot load embed models

- **Status:** Both fixed. Issue 2 — `chirpd/state.py` (PR #70). Issue 1 — story 3.8 on branch `feat/story-3.8-embed-model-load`. See "Resolution" under each issue below.
- **Severity:** High (blocks the Maya/Priya end-to-end journeys: a freshly `add`ed model can't be warmed; no embed model can be warmed at all)
- **Owning epic:** EPIC-CHIRPD-CORE
- **Owning stories:** 3.5-model-lifecycle (Issue 2), 3.6-backend-and-inference (Issue 1) — both currently marked **Done**
- **Found during:** manual smoke of story 4.4 (`chirp models list`) on 2026-05-28, Apple Silicon (arm64), against `mlx-community/bge-small-en-v1.5-bf16` and `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- **Not a defect in:** story 4.4 — `chirp models list` behaved correctly throughout (rendered `loaded: false`/`null` faithfully, never spawned a daemon under `spawn_if_absent=False`). These are escaped defects in already-Done CHIRPD-CORE stories, surfaced by exercising `list` against a real daemon.

## Summary

Two independent daemon-side defects, both discovered while validating `chirp models list` against a live `chirpd`:

1. **Embed models cannot be loaded.** `MLXBackend.load` routes every role through `mlx_lm.load`, which only supports causal LMs. Warming a `bert`-class embed model fails.
2. **The daemon never re-reads `models.toml` after startup.** It caches the registry once at construction, so a model registered (or re-roled) after the daemon started is invisible to `model.load`/`model.list`/`model.status` until the daemon restarts — directly contradicting the architecture contract.

---

## Issue 1 — `MLXBackend.load` cannot load embed (`bert`) models  ✅ FIXED via STORY 3.8

- **Resolution:** Fixed via [EPIC-CHIRPD-CORE story 3.8](../planning-artifacts/epic-chirpd-core/stories/3.8-embed-model-load-and-pooled-inference.md) on branch `feat/story-3.8-embed-model-load`. `MLXBackend.load` now branches on role: chat → `mlx_lm.load` (unchanged); embed → `mlx_embeddings.load` (new arm64-gated `mlx-embeddings==0.1.0` dependency). `MLXBackend.embed` was reworked to return pooled sentence vectors via `mlx_embeddings.generate(model, processor, texts=...).text_embeds`, fixing the second latent defect (the old `_invoke_embed` returned token-level lookup vectors, not sentence embeddings — now deleted). Verified by an opt-in `@slow @integration` test that passes against real MLX (`mlx-community/bge-small-en-v1.5-bf16`): correct vector count, uniform dimensionality, determinism, and cosine-similarity sanity. EPIC-INTEGRATION-CUTOVER 6.3 remains the first real consumer and will exercise this end-to-end through the retrieval/index pipeline.
- **Owning story:** 3.6-backend-and-inference (embed inference path)
- **Symptom:**
  ```
  $ chirp models add mlx-community/bge-small-en-v1.5-bf16
  Warming bge-small-en-v1.5-bf16...
  Error: Model registered but warm failed: mlx_lm.load failed for
  'mlx-community/bge-small-en-v1.5-bf16': Model type bert not supported.
  ```
- **Root cause:** `chirpd/backend.py:49` — `MLXBackend.load` calls `mlx_lm.load(local_path)` (`chirpd/backend.py:82`) for **both** `chat` and `embed` roles. `mlx_lm.load` only constructs causal-LM architectures, so any embedding model (`bert`, sentence-transformers, etc.) raises `Model type <x> not supported`. The downstream embed path (`MLXBackend.embed` / `_invoke_embed`, `chirpd/backend.py:170`/`:189`) is never reached because `load()` fails first.
- **Expected (per spec):**
  - Story 3.5 deferred inference to 3.6: *"`stream_generate` and `embed` raise `NotImplementedError` in this story; story 3.6 fills them in"* (`stories/3.5-model-lifecycle.md:105`).
  - The epic assigns embed to 3.6: *"3.6 | Backend abstraction + MLX implementation: `chat` (streaming), `embed`, `cancel` ops"* (`epic-chirpd-core/epic.md:98`), with an acceptance bullet that `client.embed([...])` returns the right vector count (`epic.md:110`).
  So `embed`-role loading was in 3.6's scope but the `load()` side was never branched for it.
- **Proposed fix:** branch `MLXBackend.load` on `role`. For `role == "embed"`, use an embedding-capable loader (an MLX embedding model loader / sentence-transformers-style path) instead of `mlx_lm.load`, returning a handle whose `embed`/`embed_tokens` callable satisfies `_invoke_embed`. Add a `@slow @integration` test that loads and embeds against a small real `bge` model.

---

## Issue 2 — daemon caches the registry at startup; never re-reads per op  ✅ FIXED (PR #70)

- **Resolution:** Fixed directly in `chirpd/state.py` (PR #70, branch `fix/chirpd-registry-reread`). `DaemonState` now takes an optional `registry_reader`; the daemon passes `read_registry`, and `model.load` / `model.list` / `resolve_canonical_alias` re-read `models.toml` per op via `DaemonState._refresh_registry()` before resolving (refresh happens before lock acquisition in `load`, preserving the `_registry_locks` discipline). Plain per-op read, no file watcher (hot-reload stays out of scope). Regression tests: `tests/chirpd/test_state.py::test_load_sees_alias_added_after_startup`, `::test_list_models_reflects_registry_changes_after_startup`, `::test_registry_reader_absent_keeps_in_memory_registry`. Not story-tracked (small, isolated contract fix).
- **Owning story:** 3.5-model-lifecycle (per-op registry read)
- **Symptom:** after a daemon is already running, registering a new alias and warming it fails until the daemon is restarted:
  ```
  $ chirp models add mlx-community/Qwen2.5-0.5B-Instruct-4bit   # daemon already up
  Warming qwen2.5-0.5b-instruct-4bit...
  Error: Model registered but warm failed: unknown model alias
  'qwen2.5-0.5b-instruct-4bit'. ...
  # kill chirpd, retry → "Ready." (fresh daemon re-read the registry)
  ```
- **Root cause:** `chirpd/__main__.py` (`main()`) calls `read_registry()` exactly once at startup and hands the result to `DaemonState`. `DaemonState._registry` (`chirpd/state.py:46`) stores that snapshot for the daemon's lifetime; `resolve_alias` always consults the frozen copy (`chirpd/state.py:78`, `:88`) and `list_models` iterates it (`chirpd/state.py:145`). Nothing re-reads `models.toml`.
- **Expected (per spec) — this is a contract violation, not a missing feature:**
  - Story 3.5 dev notes: *"Reading `models.toml` happens **on every `model.load`/`model.list`/`model.status`** per architecture § Configuration & Persistence (no hot-reload; daemon doesn't watch the file)"* (`stories/3.5-model-lifecycle.md:289`).
  - Model-registry epic: *"The daemon re-reads on `model.*` ops; manual edits between ops are visible to the next op without requiring restart, but file-watcher complexity is not introduced"* (`epic-model-registry/epic.md:138`).
- **Proposed fix:** re-read `models.toml` at the start of each `model.load` / `model.list` / `model.status` handler (or via a `DaemonState` accessor that re-reads), replacing the cached `self._registry`. Keep it a plain per-op read — **not** a file-watcher (hot-reload is explicitly out of scope per PRD §Explicit Out of Scope and `epic.md:138`). Mind the existing `_registry_locks` / per-model lock discipline when swapping the registry reference. Add a test: construct `DaemonState`, mutate the on-disk `models.toml`, assert the next `model.list`/`model.load` sees the new alias without reconstruction.

---

## Evidence (4.4 smoke, isolated `HOME`, real `chirpd`, real MLX)

- `list` with no daemon → `(daemon not running — loaded state unknown)`, table with `★`/`—`, `--json` → `daemon_reachable: false`, `loaded: null`. No socket created (confirmed `list` does not spawn). ✅ (4.4 correct)
- Warming embed `bge-small-en-v1.5-bf16` → Issue 1 (`Model type bert not supported`).
- Warming newly-added `Qwen2.5-0.5B-Instruct-4bit` against an already-running daemon → Issue 2 (`unknown model alias`); succeeded after daemon restart (`Ready.`).
- Post-restart `list` (daemon up, qwen warmed) → `daemon_reachable: true`, qwen `loaded ●` / `true`, bge `loaded —` / `false`; `list --json | jq -e .` parsed OK. ✅ (4.4 correct)

## Recommended handling

**Resolved as follows:**

- **Issue 2** — fixed directly (PR #70). It broke the core "`add` then use" loop for chat models too, was a small isolated contract fix in `chirpd/state.py`, and was on a live path, so it did not warrant a story.
- **Issue 1** — fixed via **EPIC-CHIRPD-CORE story 3.8** (`feat/story-3.8-embed-model-load`). Initially scoped to defer to cutover, but implemented now: the loader branch + pooled-inference rework land with an opt-in `@slow @integration` test that passes against a real `bge` model, so the daemon-side embed capability is verified ahead of its first consumer. **EPIC-INTEGRATION-CUTOVER 6.3** still depends on 3.8 and provides the end-to-end pipeline verification.
