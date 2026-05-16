# Epic: Route existing chirp surfaces through `llm.client` with no notes-quality regression

- **Epic ID:** EPIC-INTEGRATION-CUTOVER
- **Owner:** Colby
- **Status:** Draft
- **Created:** 2026-05-15
- **Design source:** [`prd.md`](../prd.md) — FR46–FR48, §Domain-Specific Requirements → Validation Methodology, §Success Criteria → User Success → "No regression in note quality"; [`architecture.md`](../architecture.md) — §Project Structure (call-site changes in `notes/note_generator.py`, `notes_chat/retrieval.py`, `notes_chat/cli.py`; regression corpus at `tests/regression/notes_quality/`), §Module boundary "llm.client ↔ existing chirp modules", §Implementation Patterns → Testing Patterns (FakeBackend); [`implementation-readiness-report-2026-05-12.md`](../implementation-readiness-report-2026-05-12.md) — §Epic Quality Review (corpus is precondition gate)
- **Related branch (current work):** TBD (off `main`; predecessor branch `story/2.3-blackhole-removal`)

## 1. Goal

After this epic, the existing user-facing chirp commands that touch a language model — `chirp transcribe` (note generation), `chirp ask` (interactive chat mode), and `chirp search`'s retrieval embedding path — all run against `chirpd` via `llm.client` instead of the Ollama HTTP API. Prompt templates, retrieval logic, chroma index layout, and notes directory layout are unchanged (FR47, FR48). A regression corpus of ≥10 representative recordings, captured against the current Ollama-backed pipeline before any cutover code is written, is replayed against the new MLX pipeline and scored blindly per the PRD's validation methodology; the new pipeline rates equal-or-better on ≥80% of cases (FR46, PRD §Success Criteria → User Success).

The vertical-slice cutover for `chirp ask` (one-shot mode) was already landed by EPIC-CHIRPD-CORE story 3.7. This epic extends that pattern to the rest of the LLM-touching surface and proves that the swap preserves note quality.

## 2. Why now

The five-epic decomposition orders this epic fourth:

1. **EPIC-CHIRPD-CORE** delivers `chirpd`, the NDJSON protocol, `llm.client` (streaming + non-streaming chat, embed, cancel), and a working `chirp ask` vertical slice.
2. **EPIC-MODEL-REGISTRY** delivers `chirp models add/list/default/...` so the operator running the regression run can register the model used for "after" generation.
3. **EPIC-DAEMON-LIFECYCLE** delivers `chirp daemon status` / `logs` — useful for diagnosing the regression run but not on the critical path.
4. **EPIC-INTEGRATION-CUTOVER (this epic)** — routes the remaining Ollama call sites through `llm.client` and runs the quality regression.
5. **EPIC-INIT-AND-MIGRATION** finishes the migration by removing the `ollama` Python client from `pyproject.toml` and updating docs / `chirp init`. That removal cannot happen until this epic completes, because the regression-corpus baseline capture in story 6.1 must run against the live Ollama-backed pipeline.

Two practical reasons this epic must follow CHIRPD-CORE and MODEL-REGISTRY:

- The `llm.client` API surface (streaming `chat`, batched `embed`, `cancel`) must exist before any call-site can be migrated.
- The "after" half of the regression comparison requires a registered MLX model the daemon can load. Without `chirp models add` shipped, the operator has no way to register the regression model.

The PRD's "no regression in note quality" bar (§Success Criteria → User Success) and the §Domain Requirements validation methodology are quantitative gates with a defined corpus, defined comparison protocol, and a numeric pass threshold (≥80%). This epic is where those gates are enforced; if the threshold is not met, the migration does not ship.

## 3. Locked decisions from PRD/architecture

| # | Decision | Source |
|---|----------|--------|
| 1 | Prompt templates, retrieval logic, note-format conventions, and chroma index layout stay in their existing Python modules. Only the LLM call site changes. | PRD FR47, FR48; architecture §Module boundary → "llm.client ↔ existing chirp modules" |
| 2 | Regression corpus is ≥10 representative recordings split across length and speaker-count buckets: 3 short (<2 min), 4 medium (5–15 min), 3 long (>20 min); mix of single-speaker, two-person, and multi-speaker; mix of technical and non-technical content. | PRD §Domain-Specific Requirements → Validation Methodology |
| 3 | Regression corpus is captured against the **current Ollama-backed pipeline** before any cutover code is written. Story 6.1 is the precondition gate; no other story in this epic may merge until 6.1 lands. | Readiness report §Epic Quality Review → Major Issues |
| 4 | Regression corpus storage layout: `tests/regression/notes_quality/<slug>/{transcript.txt, notes_before.md, notes_after.md}`. Audio files are **not** committed (gitignored — too large; users capture their own). A README at `tests/regression/notes_quality/README.md` documents the capture procedure, the comparison methodology, the blinding protocol, and the ≥80% pass threshold. | Architecture §Gap Analysis → Regression corpus storage; PRD §Domain Requirements |
| 5 | Comparison protocol is blinded: the developer running the comparison shuffles `before`/`after` pairs and hides which is which before scoring. Scores are recorded in `tests/regression/notes_quality/results-<YYYY-MM-DD>.md`. | PRD §Domain-Specific Requirements → Validation Methodology |
| 6 | Pass threshold: ≥80% of compared pairs rated "equal or better" for the new MLX pipeline. If the threshold is not met, escalate model size (e.g., 8-bit 7B or 4-bit 14B), re-register the model via `chirp models add`, regenerate `notes_after.md`, and re-score. The epic does not close until the threshold is met. | PRD §Success Criteria → User Success; §Scoping Risks → MLX quality regression |
| 7 | Model selection in MVP uses the user's default chat model (alias resolved via the literal `"default"` per FR35). No per-task model selection — that is a PRD Growth feature. | PRD FR35; PRD §Growth Features → Per-task model overrides |
| 8 | Cancellation of an in-flight chat (e.g., Ctrl-C during interactive chat) maps to `client.cancel(req_id)` followed by clean stdout cleanup. Cancellation budget is ≤200 ms per NFR-P4. | Architecture §Async Patterns; PRD NFR-P4 |
| 9 | Streaming output for the chat surfaces: each delta from `llm.client.chat(stream=True)` is written via `sys.stdout.write(token); sys.stdout.flush()` (or the equivalent Rich `Live` update) per architecture §CLI Output Patterns. No print-with-buffering. | Architecture §CLI Output Patterns; PRD §Output Formats |
| 10 | Test fixtures that previously mocked the Ollama HTTP client are replaced with `FakeBackend` driven through an in-process daemon-under-test, **or** with a mocked `llm.client` at the call-site boundary — pick the simpler option per test. `mlx_lm.*` is **never** mocked directly. | Architecture §Testing Patterns; §Anti-Patterns to Avoid |
| 11 | Ollama Python client / `requests` calls to `OLLAMA_HOST` remain in `pyproject.toml` for the duration of this epic so story 6.1 can capture the baseline. The dependency removal lands later under EPIC-INIT-AND-MIGRATION. | PRD §Resource Scope; epic-chirpd-core §3 decision 14 |
| 12 | Embedding API shape change: Ollama's `/api/embeddings` returns one vector per call (`{"embedding": [...]}`); `llm.client.embed(inputs=[...])` is batched and returns a list-of-vectors. Each call site adapts to the new shape but preserves the existing per-input semantics — batching beyond one input per call is an optimization opportunity, **not required** by this epic. | Architecture §Async Patterns; FR2; FR22 |
| 13 | Each story commits a working state — `make check` and `make test` must pass at every story boundary. No long-lived migration branch. | PRD §Resource Scope |

## 4. Research findings — what exists vs. what is missing

Validated against `story/2.3-blackhole-removal` at the current HEAD plus the in-flight epic plan for EPIC-CHIRPD-CORE story 3.7.

### Ollama call sites still touching production code after CHIRPD-CORE lands

EPIC-CHIRPD-CORE story 3.7 rewires the `chirp ask` one-shot flow (`notes_chat/cli.py:ask` → `notes_chat/prompting.generate_answer`) and any `retrieval.retrieve_context` path it traverses. The following call sites remain on Ollama after 3.7 lands and are scoped to **this** epic:

| File | Lines (current `main`) | Surface | Story handling it |
|---|---|---|---|
| `notes/note_generator.py` | `_call_ollama` at L355–401; called from `_generate_structured_notes` at L329–353 | `chirp transcribe` note-generation prompt → Ollama `/api/generate` streaming | **Story 6.2** |
| `notes_chat/retrieval.py` | `_get_query_embedding` at L392–410 | `chirp ask` and `chirp search` query-time embedding → Ollama `/api/embeddings` | **Story 6.3** |
| `notes_chat/index.py` | embedding calls inside `_add_to_index` / `_embed_chunks` (~L334) | Indexer-time embedding when notes are added → Ollama `/api/embeddings` | **Story 6.3** (same module boundary; embedding API surface) |
| `notes_chat/cli.py` | interactive chat path: `ask()` with `question is None` delegates to `notes_chat.interactive.InteractiveChatSession` at L99–103 | `chirp ask` (no argument) → interactive chat → `enhanced_search_and_answer_stream` → Ollama streaming `/api/generate` | **Story 6.4** |
| `notes_chat/interactive.py` | `handle_question` at L171–283 consumes `enhanced_search_and_answer_stream` events | streamed token rendering + Ctrl-C handling | **Story 6.4** |
| `notes_chat/prompting.py` | `_call_ollama_*` variants, `enhanced_search_and_answer_stream`, `validate_ollama_connection`, and the per-Ollama-failure error-message helpers (~15+ call sites across the file) | Helper functions shared by `notes_chat/cli.py` and `notes_chat/interactive.py` | **Stories 6.2 / 6.4** — touched at each consumer; prompt strings themselves are not moved (FR48) |

Story 3.7 leaves any helper in `notes_chat/prompting.py` that is not on the `ask` one-shot hot path untouched. This epic completes the migration of those helpers as their consumers are cut over.

### What `llm.client` provides as a drop-in replacement

From EPIC-CHIRPD-CORE story 3.4 and 3.6 (per the epic doc and architecture §Client Library):

- `client.chat(messages=[...], model="default", options={...}, stream=False) -> ChatResponse` — non-streaming, returns the full string under `.text`.
- `client.chat(messages=[...], model="default", options={...}, stream=True) -> Iterator[Delta]` — streaming; each `Delta.text` is one or more tokens appended to the response.
- `client.embed(inputs=[str], model="default") -> list[list[float]]` — batched embedding; one vector per input.
- `client.cancel(req_id)` — best-effort cancellation of an in-flight `chat` request. Bound by NFR-P4 (≤200 ms).
- `client.health() -> bool` — readiness probe.
- Lazy-spawn and one-shot transparent retry on version mismatch are handled inside the client (FR19, FR20). Callers do not implement them.

The literal alias `"default"` resolves to `default_chat` for `chat` ops and `default_embed` for `embed` ops (FR35). Existing chirp call sites pass `"default"` and never touch model-id strings directly — this insulates the call sites from the model registry's contents.

### What regression baseline already exists

**Nothing.** No directory exists at `tests/regression/notes_quality/`. No `transcripts.txt` / `notes_before.md` pairs have been captured. The readiness report calls this out explicitly:

> **Regression corpus** — ❌ NOT CAPTURED. §Domain Requirements requires ≥10 recording before/after pairs. Must be captured *before* migration code is written.

Story 6.1 is the precondition that closes this gap. Until it lands, stories 6.2 / 6.3 / 6.4 / 6.5 must not be authored, started, or merged.

### Net code delta (rough)

- **Add:** ~150 lines under `tests/regression/notes_quality/` (README + ≥10 per-recording subdirectories with transcripts and `notes_before.md` baselines + one results markdown file from story 6.6).
- **Modify:** ~80 lines of call-site swap across `notes/note_generator.py`, `notes_chat/retrieval.py`, `notes_chat/index.py`, `notes_chat/cli.py`, `notes_chat/interactive.py`, and the relevant helpers in `notes_chat/prompting.py`. Test files under `tests/notes/`, `tests/notes_chat/` get fixture rewrites (~200 lines moved from Ollama-mock-style to `FakeBackend` / mocked `llm.client`).
- **Remove:** nothing yet — the `import ollama` / `import requests`-against-`OLLAMA_HOST` blocks are deleted as their consumers cut over, but the `ollama` Python client dependency stays in `pyproject.toml` for the duration of this epic (it gets pulled out by EPIC-INIT-AND-MIGRATION afterwards).

## 5. Story breakdown

| ID | Title | FRs covered | Depends on | File |
|----|-------|-------------|------------|------|
| 6.1 | Regression corpus capture (precondition gate) | FR46 (precondition) | — | [stories/6.1-regression-corpus-capture.md](stories/6.1-regression-corpus-capture.md) |
| 6.2 | Cut over `notes/note_generator.py` to `llm.client` | FR46 (note generation), FR48 | 6.1; EPIC-CHIRPD-CORE 3.6 | [stories/6.2-note-generator-cutover.md](stories/6.2-note-generator-cutover.md) |
| 6.3 | Cut over `notes_chat/retrieval.py` + `notes_chat/index.py` embeddings to `llm.client.embed` | FR47, FR48 | 6.1; EPIC-CHIRPD-CORE 3.6; EPIC-MODEL-REGISTRY (embed alias registered) | [stories/6.3-retrieval-embed-cutover.md](stories/6.3-retrieval-embed-cutover.md) |
| 6.4 | Cut over `notes_chat` interactive chat (`cli.py` + `interactive.py`) to `llm.client` streaming | FR46 (chat), FR48 | 6.1; EPIC-CHIRPD-CORE 3.6 + 3.7 | [stories/6.4-interactive-chat-cutover.md](stories/6.4-interactive-chat-cutover.md) |
| 6.5 | Migrate `tests/notes/` and `tests/notes_chat/` fixtures from Ollama mocks to FakeBackend / mocked `llm.client` | NFR-M1, NFR-M2 | 6.2, 6.3, 6.4 | [stories/6.5-test-fixture-migration.md](stories/6.5-test-fixture-migration.md) |
| 6.6 | Run the regression comparison; verify ≥80% equal-or-better; close the epic | FR46 (acceptance), PRD §Success Criteria → User Success | 6.1, 6.2, 6.3, 6.4, 6.5; EPIC-MODEL-REGISTRY (chat alias registered) | [stories/6.6-regression-comparison-run.md](stories/6.6-regression-comparison-run.md) |

## 6. Sequencing & dependencies

**Within-epic ordering:**

- **6.1 is the precondition gate.** No other story in this epic may merge until 6.1 lands with ≥10 `notes_before.md` baselines committed. This is explicit and load-bearing: once 6.2/6.3/6.4 are merged, the Ollama-backed code paths are gone and the "before" baseline can no longer be reproduced.
- **6.2, 6.3, and 6.4 are independent** of each other after 6.1 lands and may be merged in any order. Each story commits a working state — `make check` and `make test` pass between merges. The choice of merge order is driven by reviewer availability, not technical sequencing.
- **6.5 follows 6.2/6.3/6.4** because the fixture migration removes the Ollama mocking infrastructure that those stories temporarily rely on. Splitting 6.5 across the cutover stories would force the same fixture-shape decisions to be re-litigated three times.
- **6.6 is the closer.** It depends on every cutover story (the `notes_after.md` artifacts cannot be regenerated until the new pipeline is in place) and on EPIC-MODEL-REGISTRY (the operator must `chirp models add` the chosen MLX model before running 6.6).

**Cross-epic dependencies (incoming):**

- **EPIC-CHIRPD-CORE** must be complete (stories 3.1 through 3.7 merged). Specifically:
  - Story 3.4 (`llm.client` with lazy-spawn, retry, exception mapping) — required by every cutover story.
  - Story 3.6 (`chat` streaming, `embed`, `cancel` ops) — required by every cutover story.
  - Story 3.7 (the `chirp ask` one-shot vertical slice) — establishes the pattern this epic extends to the rest of the surface; also de-risks the streaming token-flush conventions before 6.4.
- **EPIC-MODEL-REGISTRY** must be complete enough that the operator running stories 6.1 and 6.6 can register both a chat model (`chirp models add <mlx-chat-repo>`) and an embedding model (`chirp models add <mlx-embed-repo>`) before the runs. Specifically: `chirp models add`, `chirp models default`, and `chirp models list` must work.

**Cross-epic dependencies (outgoing — what this epic blocks):**

- **EPIC-INIT-AND-MIGRATION** removes the `ollama` Python client from `pyproject.toml`, removes any remaining Ollama references in `chirp init` / `README.md` / `AGENTS.md`, and updates `chirp init` to surface the `chirp models add` next-step prompt (FR50). That epic cannot land its dependency-removal story until **every** call site in this epic is on `llm.client` and the regression bar (6.6) has been met.

**Not depended on:**

- **EPIC-DAEMON-LIFECYCLE** (`chirp daemon status` / `logs` / LaunchAgent). Useful for diagnostics during 6.6 but not on the critical path; this epic can proceed without it. If a regression run reveals a daemon-side problem, `chirp daemon logs` from DAEMON-LIFECYCLE makes triage easier, but the absence of that subcommand does not block this epic.

## 7. Success criteria

- **Regression corpus committed.** `tests/regression/notes_quality/` contains a README plus ≥10 subdirectories, each with `transcript.txt` and `notes_before.md`. The README documents the capture procedure and the blinded comparison methodology. Audio files are gitignored, and the README explains how an operator on a fresh machine can regenerate them locally if needed.
- **All Ollama call sites in `notes/` and `notes_chat/` route through `llm.client`.** `rg -n "ollama|OLLAMA_HOST|/api/generate|/api/embeddings" notes/ notes_chat/` returns zero matches outside of error messages that name the old setup for migration users (which are removed by EPIC-INIT-AND-MIGRATION). The `import requests` lines used solely for Ollama traffic are removed from `notes/note_generator.py`, `notes_chat/retrieval.py`, `notes_chat/index.py`, `notes_chat/prompting.py`, and `notes_chat/interactive.py`.
- **Existing chroma index and notes-on-disk layout are unchanged.** Re-running `chirp search` against a chroma index built before this epic returns the same chunks (modulo embedding-model differences, which are expected and acceptable — chunk-store contents and IDs do not change). Notes generated under `~/Documents/chirp/<slug>/notes.md` retain the same front-matter shape and section structure (FR47, FR48).
- **Quality regression bar met.** Story 6.6's blinded scoring shows ≥80% of pairs rated equal-or-better for the new MLX pipeline. `tests/regression/notes_quality/results-<YYYY-MM-DD>.md` records the per-pair scores, the aggregate percentage, the model alias used for "after" generation, and the scoring date. If the bar is not met on first run, the escalation path (8-bit 7B or 4-bit 14B) is documented in the same results file and the threshold is re-evaluated against the escalated run.
- **Streaming behavior preserved.** `chirp ask` (interactive mode) renders tokens incrementally as they arrive, with no perceptible delay between daemon delta and stdout flush. Ctrl-C during an in-flight answer cancels the request within NFR-P4's 200 ms budget and leaves the terminal in a clean state.
- **Quality gates green at every boundary.** `make check` and `make test` pass on every story merge in this epic (NFR-M2). Coverage on the modified files in `notes/` and `notes_chat/` does not regress from the pre-epic baseline (NFR-M1, per PRD §Coverage on touched existing modules).
- **No new exception types leak past the call-site boundary.** The typed `LLMError` hierarchy from `llm.exceptions` is caught at the same outer boundaries that previously caught `requests.exceptions.ConnectionError` / `requests.exceptions.Timeout`. User-facing error messages on transport/protocol/model failures are at least as informative as today's "Cannot connect to Ollama" copy.
- **Manual smoke test.** A fresh-checkout developer runs: `chirp models add <chat-repo>`, `chirp models add <embed-repo>` (if not already), `chirp record "smoke"`, `chirp transcribe`, `chirp ask "smoke"`. All three commands complete end-to-end against `chirpd`. The note generated by `chirp transcribe` is rendered correctly under `~/Documents/chirp/<slug>/notes.md` with the XML-parsed structure intact.

## 8. Out of scope / deferred

- **`chirpd` daemon process, the NDJSON protocol, the `llm.client` library implementation.** Owned by EPIC-CHIRPD-CORE. This epic *consumes* `llm.client`; it does not implement it.
- **`chirp models` CLI (add / list / default / remove / pull / show), HuggingFace download integration, `llm/hf.py`, `llm/registry.py` writer side, `models.toml` schema.** Owned by EPIC-MODEL-REGISTRY. Stories 6.1 and 6.6 in this epic assume that subcommand group works.
- **`chirp daemon` CLI (status / logs / start / stop / enable / disable), LaunchAgent integration, `chirp daemon logs` tailing.** Owned by EPIC-DAEMON-LIFECYCLE.
- **`chirp init` updates** (drop Ollama branches, surface `chirp models add` next-step prompt, detect pre-existing Ollama installs, Intel fail-fast), `pyproject.toml` Ollama Python-client removal, README / AGENTS.md / `chirp --help` Ollama sweep. Owned by EPIC-INIT-AND-MIGRATION; lands after this epic completes.
- **Per-task model selection** (different model for `notes` vs. `ask` vs. quick-title generation). PRD §Growth Features. MVP uses one default chat model resolved via the `"default"` alias (FR35).
- **Batched embedding optimization at index time.** The new `client.embed(inputs=[...])` is batched, but story 6.3 preserves the existing one-input-per-call semantics. Batching `n>1` is an opportunity to speed up `chirp index` builds; out of scope here and trackable as a follow-up.
- **Re-indexing existing chroma collections** against the new MLX embedding model. The PRD's quality bar is "no regression in note quality," not "identical vector spaces"; existing chroma indices keep working because chunk IDs and store contents are unchanged. If a user notices retrieval quality drift after migration, the documented remedy is `chirp index --force`. Not blocking this epic.
- **A formal evaluation harness beyond the 10-recording corpus.** PRD §Out of Scope (MVP) → "Inference quality benchmarks beyond regression corpus." The corpus is sufficient; comprehensive eval suites are deferred.
- **Streaming-mode `chirp ask` one-shot.** Already cut over by EPIC-CHIRPD-CORE story 3.7. This epic does not re-touch that flow except where unifying helpers in `notes_chat/prompting.py` would otherwise create drift.
- **GGUF-to-MLX or Ollama-to-MLX model file conversion.** PRD §Out of Scope. Users re-download an MLX-format equivalent via `chirp models add`.

## 9. Risks

- **The regression corpus is not representative of real user content.** A 10-recording corpus captured by one developer cannot cover every length / topic / speaker-count combination users will produce. Mitigation: story 6.1 enforces the bucket split from the PRD (3 short, 4 medium, 3 long; mix of speakers; mix of technical and non-technical) so the corpus is at least *structurally* diverse. The corpus is also versioned alongside the codebase — future epics can extend it.
- **Quality bar fails on first run.** PRD §Scoping Risks → MLX quality regression names this as Medium likelihood / High impact. Mitigation: story 6.6 documents an explicit escalation path (8-bit 7B → 4-bit 14B) with concrete next steps; the epic does not close until the bar is met.
- **Blinded scoring is hard to do honestly when one developer is both the implementer and the scorer.** Mitigation: story 6.6's blinding procedure shuffles `before`/`after` filenames into anonymized pairs (`A` / `B`) before scoring; the mapping is written to a sealed file that is opened only after all scores are recorded. This does not eliminate self-influence but materially reduces it.
- **Streaming cancellation in interactive chat doesn't propagate cleanly.** Story 6.4's existing `InteractiveChatSession` uses prompt-toolkit's Ctrl-C handling on top of a generator-based `enhanced_search_and_answer_stream`. Mapping the existing two-press-to-exit UX to `llm.client.cancel(req_id)` without regressing the "first press = clear hint" / "second press = exit" semantics is fiddly. Mitigation: story 6.4 includes explicit AC for the cancel-during-streaming path and references NFR-P4's 200 ms budget so the test can assert it.
- **Embedding API shape change introduces a silent semantic shift.** Ollama's `/api/embeddings` returns one vector per call; chirp's `client.embed([...])` returns a list-of-vectors. A naive port that loops one input per call will be correct but slow; a careful port that batches must preserve insertion order and handle empty-input lists. Mitigation: story 6.3's AC asserts insertion-order preservation and zero-length-input handling explicitly.
- **`make check` / `make test` is brittle across the cutover.** Each cutover story (6.2 / 6.3 / 6.4) leaves the test suite green on its own merge, but the test fixtures used in those intermediate states overlap with the ones story 6.5 cleans up. Mitigation: 6.5 lands soon after 6.4 to retire any temporary mocking helpers; the intermediate-state fixtures are explicitly tagged as transitional in the relevant story's Dev notes.
- **The Ollama Python client / `requests` import stays in the codebase longer than needed.** Per locked decision 11, the `ollama` Python client and `requests`-against-`OLLAMA_HOST` blocks remain importable through story 6.1's baseline run. If 6.1 is delayed but 6.2 / 6.3 / 6.4 are not, the cutover stories must not preemptively remove those imports — that work is owned by EPIC-INIT-AND-MIGRATION. Mitigation: each cutover story's "out of scope" section calls this out explicitly.
- **Operator capacity for the regression corpus capture is real work, not just engineering.** Story 6.1 takes the operator-author roughly half a day of recording and pipeline-running (per readiness report estimate). It can stall the epic if not scheduled. Mitigation: story 6.1 lists candidate recording topics so the operator does not need to invent material from scratch; pre-existing development recordings are acceptable if they fit the bucket distribution.
