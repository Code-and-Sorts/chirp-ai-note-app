---
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
workflowType: 'implementation-readiness'
date: 2026-05-12
project: chirp-ai-note-app
prdUnderReview: _bmad-output/planning-artifacts/prd.md
---

# Implementation Readiness Assessment Report

**Date:** 2026-05-12
**Project:** chirp-ai-note-app
**PRD Under Review:** Ollama → MLX migration (`chirpd` daemon)

## Document Inventory

### PRD

**Whole Documents:**
- `_bmad-output/planning-artifacts/prd.md` — 627 lines, 12 level-2 sections, created 2026-05-12

**Sharded Documents:** none

### Architecture

⚠️ **WARNING: No architecture document found.**

No `*architecture*.md` files exist under `_bmad-output/planning-artifacts/`. Architecture work for this PRD has not yet been started.

### Epics & Stories

**Whole Documents:**
- `_bmad-output/planning-artifacts/epic-audio-capture/epic.md` — **prior unrelated work** (BlackHole removal, completed/in-progress)
- `_bmad-output/planning-artifacts/epic-wireframe-alignment/epic.md` — **prior unrelated work** (CLI surface lock, completed)

**Sharded Documents:** none

**Stories:** 11 stories across the two prior epics, none related to this PRD.

⚠️ **WARNING: No epic exists for the Ollama → MLX migration PRD.** Implementation cannot begin until at least one epic decomposes the PRD's FRs into stories.

### UX

Not applicable — chirp is a CLI tool. EPIC-WF-ALIGN locks the visible command surface (`record · transcribe · notes · ask · search · init · about`). New subcommands defined in this PRD (`models`, `daemon`) are documented in §CLI Tool Specific Requirements; no separate UX document is expected.

## Duplicates

None.

## Critical Issues Flagged for Downstream Steps

1. **No architecture document** — readiness assessment will flag this as a gap.
2. **No epic for this PRD** — readiness assessment will flag this as a gap.

These are expected at this point in the workflow (PRD just completed). The assessment will produce a punch list of what needs to exist before implementation can start.

## Documents to Use for This Assessment

- `_bmad-output/planning-artifacts/prd.md`

## PRD Analysis

PRD loaded and read in full. Source of truth for verbatim text: `_bmad-output/planning-artifacts/prd.md`. Below is the requirements inventory keyed for coverage validation in subsequent steps.

### Functional Requirements

**Total FRs: 56**, organized in 8 capability areas.

| Area | IDs | Count | Summary |
|---|---|---|---|
| Inference Daemon (`chirpd`) — Core Operations | FR1–FR10 | 10 | chat streaming, embed, cancel, status, lazy-load + idle-unload, embed-pinning, `keep_alive` override, chat-template application, explicit load/unload |
| Daemon Protocol, Lifecycle, Version Handshake | FR11–FR17 | 7 | unix socket + NDJSON, single-instance lock, version-stamped `hello`, graceful drain, health endpoint, local logs, no network from daemon |
| Client Library (`chirp.llm`) | FR18–FR23 | 6 | socket auto-discover, lazy-spawn, one-shot retry, streaming + non-streaming chat, batched embed, typed exception classes |
| Model Registry — User-Facing | FR24–FR31 | 8 | `chirp models {add, list, default, remove, pull, show}`, alias/role flags, `--json` |
| Model Registry — System Behaviors | FR32–FR38 | 7 | role inference, auto-default-on-first-of-role, auto-warm, `"default"` resolution, raw `org/repo` resolution, `models.toml` w/ schema version, HF cache reuse |
| Daemon Lifecycle Management — User-Facing | FR39–FR45 | 7 | `chirp daemon {status, start, stop, restart, enable, disable, logs}` |
| Integration with Existing Modules | FR46–FR48 | 3 | quality parity, existing chroma/notes layout unchanged, prompts stay in existing packages |
| First-Run, Init, Migration | FR49–FR54 | 6 | drop Ollama checks, surface model-add prompt, daemon-readiness check, LaunchAgent opt-in, Ollama-detect migration plan, fail-fast on Intel |
| Documentation and Dependency Surface | FR55–FR56 | 2 | dependency swap in pyproject.toml, docs reference Ollama nowhere |

**Full FR text:** see `prd.md` §Functional Requirements (lines ≈436–504 in the source).

### Non-Functional Requirements

**Total NFRs: 33**, organized in 7 categories.

| Category | IDs | Count | Key targets |
|---|---|---|---|
| Performance | NFR-P1–NFR-P7 | 7 | Warm first-token ≤ 500 ms; cold ≤ 5 s on M2 / 8 s on M1; ≥ 30 tok/s gen; cancel ≤ 200 ms; idle-unload ±10%; spawn ≤ 3 s; embed ≥ 50/s |
| Reliability | NFR-R1–NFR-R7 | 7 | 30-day stability; LaunchAgent restart-on-crash; single-instance determinism; version-drift ≤ 2 s user-visible; graceful drain ≤ 10 s; typed missing-weights error |
| Security & Privacy | NFR-S1–NFR-S6 | 6 | Local-only IPC; socket 0600; daemon emits no network; no telemetry; log redaction (no prompts/content); HF cache scoping |
| Maintainability | NFR-M1–NFR-M5 | 5 | ≥ 90% coverage on new modules; `make check` + `make test` green; mypy clean; deps pinned; diagnostic surface adequate |
| Compatibility & Portability | NFR-C1–NFR-C4 | 4 | macOS 13+; Apple Silicon required (Intel fails loudly); Python floor unchanged; HF cache shared with MLX ecosystem |
| Observability | NFR-O1–NFR-O3 | 3 | Structured logs (format deferred — Open Question 3); rotation ~10 MB; rich `daemon status --json` |
| Accessibility | NFR-A1 | 1 | Terminal accessibility; clean fallback in non-TTY |

**Full NFR text:** see `prd.md` §Non-Functional Requirements (lines ≈506–559 in the source).

### Additional Requirements and Constraints

These are not labeled FR/NFR but are binding for implementation:

- **Hard constraints (§Domain-Specific Requirements):**
  - On-device privacy — no remote inference fallback, no telemetry, daemon emits no network.
  - Validation methodology — ≥10 recording regression corpus, blinded comparison, ≥80% equal-or-better threshold, artifact committed under `tests/regression/notes_quality/`.
  - Resource budgets — daemon idle ≤ 150 MB; loaded model RSS matches on-disk ± 10%; embed model RSS ≤ 200 MB.
- **Explicit Out-of-Scope (§Project Scoping → Out of Scope):**
  - Menu bar app (separate epic)
  - Intel Mac support
  - Ollama → MLX model migration tooling
  - Auto-uninstall of Ollama
  - OpenAI-compatible HTTP endpoint
  - Per-task model selection in MVP
  - Remote / cloud inference fallback
  - GPU offload to external accelerators
  - App Store packaging
  - Inference quality benchmarks beyond regression corpus
  - Hot-reload of `models.toml` without restart
- **Scoping risks (§Project Scoping → Scoping Risks):** 7 risks with mitigations, including MLX quality regression (M/H), cold-start exceedance (M/M), `mlx-lm` API churn (M/M), LaunchAgent silent failure (L/M), daemon hang on shutdown (L/M), HF cache corruption (M/L), long-uptime memory leak (M/L).
- **Resource scope (§Project Scoping → Resource Scope):** solo developer, sequential phasing — daemon protocol first (blocking), registry + `chirp models` in parallel, init updates last.
- **Open Questions (§Open Questions):** 7 deferrable decisions including default chat/embed model choices, log format, pre-warm on init, mlx-lm pin granularity, migration messaging, idle-unload default.

### PRD Completeness Assessment (Initial)

**Strengths:**
- Capability contract is dense and testable. 56 FRs with clear actor/capability structure; every FR is implementation-agnostic.
- NFRs are quantitative (latency numbers, RSS budgets, coverage thresholds) — not vague.
- Out-of-scope list is explicit, with rationale per item. Reduces re-litigation cost.
- User journeys (Maya, Devon, Priya, Sam) map cleanly to capability requirements via the journey→FR table in the PRD.
- Domain constraints (privacy, validation methodology, resource budgets) are named hard constraints, not aspirations.
- Risks have mitigation strategies attached, not just risk-naming.

**Gaps / concerns flagged for downstream steps:**
- **Architecture document is absent.** Several FRs imply non-trivial design decisions (protocol framing, version-handshake state machine, lazy-spawn race resolution, chat-template caching, idle-unload timer ownership) that should be locked in an architecture doc before stories land. Flag for §Final Assessment.
- **No epic exists.** No story-level decomposition of FR1–FR56 has been done. This is the single biggest readiness gap; this step's mandate covers it in §Step 3 (Epic Coverage Validation).
- **Open Question 3 (log format)** is a NFR-O1 decision that affects diagnostic UX (`chirp daemon logs`). Should be resolved before story-level work begins on the daemon.
- **Open Question 5 (`mlx-lm` pin granularity)** affects the dependency surface (FR55) and CI matrix (NFR-M2/M4). Should be resolved at epic-planning time.
- **FR46 (quality parity)** depends on the regression corpus existing. Worth tracking as a precondition for the integration-cutover work, not just a verification step.

## Epic Coverage Validation

### Epics Reviewed

Two epics exist under `_bmad-output/planning-artifacts/`:

1. **EPIC-AUDIO-CAPTURE** (`epic-audio-capture/epic.md`) — BlackHole removal via Swift `CaptureAudio.app` helper. Stories: 2.1, 2.2, 2.3. **Subject: audio capture pipeline.** No overlap with the Ollama → MLX PRD's capability surface.
2. **EPIC-WF-ALIGN** (`epic-wireframe-alignment/epic.md`) — CLI command surface alignment, storage layout, transcribe queue. Stories: 1.1–1.8. **Subject: CLI structure and storage.** No overlap with the Ollama → MLX PRD's capability surface.

### FR Coverage Extracted from Epics

**None of FR1–FR56 are covered in any existing epic.** The two existing epics predate this PRD and address unrelated work streams.

### Coverage Matrix (Compressed)

| FR Range | PRD Capability Area | Epic Coverage | Status |
|---|---|---|---|
| FR1–FR10 | Inference Daemon core ops | None | ❌ MISSING |
| FR11–FR17 | Daemon protocol, lifecycle, handshake | None | ❌ MISSING |
| FR18–FR23 | Client library | None | ❌ MISSING |
| FR24–FR31 | Model registry — user-facing | None | ❌ MISSING |
| FR32–FR38 | Model registry — system behaviors | None | ❌ MISSING |
| FR39–FR45 | Daemon lifecycle CLI | None | ❌ MISSING |
| FR46–FR48 | Integration with existing modules | None | ❌ MISSING |
| FR49–FR54 | First-run, init, migration | None | ❌ MISSING |
| FR55–FR56 | Dependency surface, docs | None | ❌ MISSING |

**Note:** FR47 ("existing chroma index, notes directory layout, and `~/.chirp/config.toml` schema continue to function unchanged") interacts with **EPIC-WF-ALIGN locked decision 5** (one-folder-per-note storage at `~/Documents/chirp/<slug>/`) and **decision 7** (`~/.chirp/config.toml`). EPIC-WF-ALIGN does not *cover* FR47, but FR47 is a non-regression contract on EPIC-WF-ALIGN's deliverables — call this out as a precondition note when the migration epic is authored.

### Missing FR Coverage — All 56 FRs

All 56 functional requirements are uncovered. Listing individually would be noise; the appropriate action is to author a new epic that decomposes them.

**Recommended epic structure for the new work:**

Based on the PRD's capability areas and the §Resource Scope note ("daemon protocol + client wrapper must land before existing modules can be migrated (blocking)"), a clean decomposition is:

| Suggested Epic | FR Coverage | Rationale |
|---|---|---|
| **EPIC-CHIRPD-CORE** — Daemon process, protocol, client library | FR1–FR23 (40% of FRs) | Foundational; everything else depends on this landing first. Includes ops, protocol, lifecycle, single-instance, version handshake, client library. |
| **EPIC-MODEL-REGISTRY** — `chirp models` subcommand + `models.toml` + HF integration | FR24–FR38, FR55 (parts) | Independent track from daemon core; can develop in parallel once `chirpd`'s `model.load` contract is stable. |
| **EPIC-DAEMON-LIFECYCLE** — `chirp daemon` subcommand + LaunchAgent + logging | FR39–FR45, NFR-O1–O3 | Depends on EPIC-CHIRPD-CORE for daemon-status integration; can develop in parallel with EPIC-MODEL-REGISTRY. |
| **EPIC-INTEGRATION-CUTOVER** — Route existing `notes`/`notes_chat` through `chirp.llm`; regression corpus; quality parity | FR46–FR48 | Depends on both EPIC-CHIRPD-CORE and EPIC-MODEL-REGISTRY. Includes the §Domain Requirements validation methodology. |
| **EPIC-INIT-AND-MIGRATION** — `chirp init` updates, Ollama-detection messaging, Intel fail-fast, dependency surface, docs | FR49–FR56 | Lands last in the sequence per §Resource Scope. |

5 epics, ~5–10 stories each, sequenced as: CHIRPD-CORE → (REGISTRY ‖ LIFECYCLE) → INTEGRATION-CUTOVER → INIT-AND-MIGRATION.

### Coverage Statistics

- **Total PRD FRs:** 56
- **FRs covered in epics:** 0
- **Coverage percentage:** **0%**
- **Existing epics relevant to this PRD:** 0
- **New epics required:** ~5 (per recommended structure)

## UX Alignment Assessment

### UX Document Status

**Not Found** — no `*ux*.md` or `*ux*/index.md` exists under `_bmad-output/planning-artifacts/`.

### Is UX Implied?

**No.** Chirp is a terminal CLI tool with no GUI surface. The PRD explicitly excludes the menu bar app from scope (§Project Scoping → Out of Scope) and defers it to a separate future epic. EPIC-WF-ALIGN already locks the visible command surface; this PRD adds two subcommand groups (`models`, `daemon`) whose interaction model is documented in §CLI Tool Specific Requirements → Command Structure / Output Formats / Scripting Support.

### Alignment Findings

No misalignment to report. The PRD's CLI-tool-specific requirements section substitutes for a UX document:

- **Command shape** is fully specified (FR24–FR45 plus the §Command Structure table).
- **Output discipline** (TTY vs non-TTY, stdout vs stderr, `--json` semantics) is specified in §Output Formats.
- **Error UX** (exit codes 0/1/2/3/4/5) is specified in §Scripting Support.
- **Interactive flows** (`chirp init`'s recheck migration messaging, post-add model warm) are specified across FR49–FR54 and Open Question 6.

### Warnings

- **OQ6 (migration messaging strategy)** is a UX-class decision that should be resolved before the EPIC-INIT-AND-MIGRATION work begins, since it affects the user-visible output of `chirp init --recheck`.
- **Progress feedback semantics** for `chirp models add` / `pull` (Rich progress bar on TTY, start/done lines on stderr in non-TTY) are specified in §Output Formats but should be validated against the existing chirp UX conventions (Rich rendering is already used in `recorder/live_dashboard.py` and `chirp/init_flow.py`) during implementation.
- **No formal user-test plan** for the new subcommand surface. Acceptable for a single-developer project, but worth noting that the four journey narratives (Maya, Devon, Priya, Sam) in the PRD are the de facto acceptance scenarios — they should be referenced as test inputs during story authoring.

### Conclusion

UX coverage is **adequate via PRD §CLI Tool Specific Requirements**. No separate UX document required for this PRD's scope.

## Epic Quality Review

No epics exist for this PRD to review against the checklist. Pivoting to a **pre-authoring quality review** of the 5-epic structure recommended in §Epic Coverage Validation, flagging issues to address when those epics are actually written.

### Pre-Authoring Review of Proposed Epic Structure

Recap of the recommended structure:

1. **EPIC-CHIRPD-CORE** — Daemon process, protocol, client library (FR1–FR23)
2. **EPIC-MODEL-REGISTRY** — `chirp models` subcommand + `models.toml` + HF integration (FR24–FR38)
3. **EPIC-DAEMON-LIFECYCLE** — `chirp daemon` subcommand + LaunchAgent + logging (FR39–FR45, NFR-O1–O3)
4. **EPIC-INTEGRATION-CUTOVER** — Route existing modules through `chirp.llm`; regression corpus; quality parity (FR46–FR48)
5. **EPIC-INIT-AND-MIGRATION** — `chirp init` updates, Ollama detection, Intel fail-fast, dependency surface, docs (FR49–FR56)

### 🔴 Critical Concerns (Address Before Epic Authoring)

- **EPIC-CHIRPD-CORE risks being a "technical milestone" epic.** As described, it builds infrastructure (daemon, protocol, client library) with no user-facing demonstration. This violates the "epic delivers user value" rule.
  - **Remediation:** redefine the epic to include a vertical slice — wire at least one user-visible command (e.g., `chirp ask`) end-to-end through the new daemon before the epic is "done." Name should reflect user outcome: e.g., **"chirp ask runs against the new MLX-backed daemon"** rather than "Daemon process, protocol, client library."
  - **Alternative remediation:** fold a minimum subset of EPIC-INTEGRATION-CUTOVER into EPIC-CHIRPD-CORE so the epic's "done" criterion is a working user command.

- **EPIC-DAEMON-LIFECYCLE has marginal standalone user value.** Diagnostic-only UX (`chirp daemon status`/`logs`). Most users (per the Maya, Devon, Priya journeys) never run these commands.
  - **Remediation:** consider folding into EPIC-CHIRPD-CORE. The LaunchAgent piece is the only part with directly observable user value (auto-start at login = no first-of-day cold start). Diagnostic commands can be sized as a single story under EPIC-CHIRPD-CORE rather than a standalone epic.
  - **Alternative:** if kept standalone, rename the epic to user-outcome language: "Daemon survives login and reboots, with diagnostics for troubleshooting" — and explicitly justify why this needs to be its own epic.

### 🟠 Major Issues to Watch When Authoring

- **EPIC-INTEGRATION-CUTOVER depends on a regression corpus existing.** Per §Domain Requirements, the corpus must be captured *before* the migration code is written (so the "before" baseline is preserved). This is an upstream dependency on operator work, not engineering work. Should be either (a) the first story under EPIC-INTEGRATION-CUTOVER, with a clear gate ("corpus exists and committed"), or (b) a precondition story under EPIC-CHIRPD-CORE so it's done before any integration code is touched.
- **`models.toml` creation timing.** Per the "Database/Entity Creation Timing" check — `models.toml` should be created by the first story under EPIC-MODEL-REGISTRY that needs it (likely `chirp models add`), not in a separate "init the registry file" story upfront. Schema migration scaffolding (`schema_version` field) can be deferred until/unless a v2 schema lands.
- **Story-level forward dependencies inside EPIC-CHIRPD-CORE.** The natural decomposition (daemon process → unix socket → NDJSON handler → chat op → version handshake → client wrapper → lazy spawn → vertical slice) has many internal dependencies. Each story must declare an independently demonstrable end-state — e.g., the "NDJSON handler" story is demonstrable via `nc` against the socket, even before any op is implemented. Resist the temptation to write a single "build the daemon" mega-story.
- **OQ3 (log format) blocks NFR-O1, NFR-O2, and the user-visible `chirp daemon logs` UX.** Must be resolved before the LaunchAgent / logging stories are authored, or else the work bifurcates.
- **OQ5 (`mlx-lm` pin granularity)** affects FR55 (dependency surface) and NFR-M4 (dependency hygiene). Resolve before EPIC-INIT-AND-MIGRATION's `pyproject.toml` story lands.

### 🟡 Minor Concerns

- Epic naming convention. Existing epics use `EPIC-<TOPIC>` (e.g., `EPIC-AUDIO-CAPTURE`, `EPIC-WF-ALIGN`). The proposed names follow that pattern but lean technical (`EPIC-CHIRPD-CORE`). Consider user-outcome framing in line with the BMAD "user value" principle.
- Cross-epic FR mapping has slight ambiguity at the boundary:
  - **FR55** (dep removal) is assigned to EPIC-INIT-AND-MIGRATION but logically *enables* EPIC-CHIRPD-CORE (you can't import `mlx-lm` until it's in `pyproject.toml`). Recommendation: split into "add mlx-lm to deps" (lands with EPIC-CHIRPD-CORE) and "remove Ollama client from deps" (lands with EPIC-INIT-AND-MIGRATION after cutover is complete).
  - **FR56** (docs reference Ollama nowhere) is implicitly cross-cutting: every epic that touches user-facing surfaces should update the relevant doc fragment, with a final "docs sweep" story under EPIC-INIT-AND-MIGRATION. Not a single story.
- No architecture document exists (flagged in §Document Inventory). Several FRs (FR9 single-instance enforcement → `flock` semantics; FR11 protocol framing; FR13 version handshake state machine; FR15 health endpoint contract) imply design decisions that should be locked before story authoring.

### Brownfield Indicators ✓

Both correctly accounted for in the proposed structure:

- **Integration points with existing systems:** EPIC-INTEGRATION-CUTOVER explicitly handles `notes`, `notes_chat`, and the chroma index.
- **Migration/compatibility stories:** EPIC-INIT-AND-MIGRATION explicitly handles Ollama detection in `chirp init --recheck` (FR53) and the non-destructive migration message.

### Best-Practices Compliance Checklist (Pre-Authoring)

For each of the 5 proposed epics, the checklist below is the test that will be applied at authoring time:

- [ ] Epic delivers user-observable value (not "infrastructure built")
- [ ] Epic can function independently (Epic N uses only Epics 1..N−1)
- [ ] Stories appropriately sized (each independently demonstrable)
- [ ] No forward dependencies between stories
- [ ] Entities (`models.toml`, log files, socket file) created when first needed, not preemptively
- [ ] Clear acceptance criteria per story (Given/When/Then)
- [ ] Traceability from each story → at least one FR or NFR
- [ ] Epic name uses user-outcome language

**Current state:** 0 of 5 epics authored; checklist applies on first authoring pass.

### Recommendations Summary

1. **Author an architecture document** before authoring epics. Lock the protocol framing, version-handshake state machine, single-instance approach, and chat-template caching. Several FRs imply non-trivial design decisions that should not be deferred to story-authoring time.
2. **Resolve Open Questions 3 and 5** (log format, `mlx-lm` pin granularity) before authoring EPIC-CHIRPD-CORE and EPIC-INIT-AND-MIGRATION respectively.
3. **Reframe EPIC-CHIRPD-CORE** to include a user-observable vertical slice (one command through the new daemon end-to-end). Or fold a minimum integration scope into it.
4. **Reconsider EPIC-DAEMON-LIFECYCLE as standalone.** Likely better folded into EPIC-CHIRPD-CORE.
5. **Capture the regression corpus** (per §Domain Requirements validation methodology) as the first story under EPIC-INTEGRATION-CUTOVER, or as a precondition to it. Don't author migration code before the corpus exists.
6. **Plan `pyproject.toml` work to land in two places** — add `mlx-lm` early (EPIC-CHIRPD-CORE), remove `ollama` late (EPIC-INIT-AND-MIGRATION after cutover).

## Summary and Recommendations

### Overall Readiness Status

**NEEDS WORK** — the PRD itself is strong, but it cannot be implemented yet because no architecture document and no epics exist for it. These are normal gaps at this workflow stage; the PRD has just been completed. The work to bridge them is well-scoped below.

### Readiness Scorecard

| Artifact | Status | Notes |
|---|---|---|
| **PRD** | ✅ READY | 56 FRs, 33 NFRs, explicit out-of-scope, validated against journeys. High information density. |
| **Architecture document** | ❌ NOT STARTED | Multiple FRs imply non-trivial design decisions (protocol framing, version-handshake state machine, single-instance, chat-template caching). Should be authored before stories. |
| **Epics for this PRD** | ❌ NOT STARTED | 0 of 56 FRs covered by any existing epic. ~5 new epics needed per the recommended structure. |
| **UX document** | ➖ N/A | CLI tool; PRD §CLI Tool Specific Requirements substitutes. |
| **Regression corpus** | ❌ NOT CAPTURED | §Domain Requirements requires ≥10 recording before/after pairs. Must be captured *before* migration code is written. |
| **Open decisions** | ⚠️ 2 BLOCKING | OQ3 (log format) blocks daemon stories; OQ5 (mlx-lm pin granularity) blocks dependency surface work. Other open questions (OQ1, OQ2, OQ4, OQ6, OQ7) can be resolved later. |

### Critical Issues Requiring Immediate Action

1. **Author the architecture document.** The PRD specifies the *what* well; several FRs (FR9, FR11, FR13, FR15) need *how* decisions locked before story authoring. Use `/bmad-create-architecture`.
2. **Resolve Open Question 3 (logging format)** — affects NFR-O1, NFR-O2, and the user-visible `chirp daemon logs` UX. Cheap to decide; expensive if deferred past story-writing.
3. **Resolve Open Question 5 (`mlx-lm` pin granularity)** — affects FR55 and NFR-M4. Single-line decision.
4. **Capture the validation regression corpus.** Per §Domain Requirements, ≥10 representative recordings with their existing Ollama-generated `notes.md`. Must happen *before* migration work begins so the "before" baseline is preserved. This is operator work, not engineering work, and is easy to forget.
5. **Author the 5 epics** per the recommended structure in §Epic Coverage Validation, applying the pre-authoring quality remediations in §Epic Quality Review (specifically: reframe EPIC-CHIRPD-CORE to include a user-observable vertical slice; reconsider folding EPIC-DAEMON-LIFECYCLE into EPIC-CHIRPD-CORE).

### Recommended Next Steps (Sequenced)

1. **Resolve OQ3 and OQ5** in a short note appended to the PRD's Open Questions section or in a follow-up commit. ~30 minutes.
2. **Capture the regression corpus** — record (or designate from existing recordings) ≥10 representative inputs, run them through the current Ollama-backed pipeline, commit transcripts + generated `notes.md` to `tests/regression/notes_quality/`. Half a day.
3. **Run `/bmad-create-architecture`** to produce the architecture document covering: protocol framing, version-handshake state machine, single-instance enforcement, chat-template handling, lazy-spawn race resolution, idle-unload timer ownership, error model. Output: `_bmad-output/planning-artifacts/architecture.md`.
4. **Run `/bmad-create-epics-and-stories`** with the recommended 5-epic structure as input, applying the quality remediations from §Epic Quality Review. Output: 5 epic folders with stories.
5. **Re-run `/bmad-check-implementation-readiness`** to verify the gaps closed before implementation begins.

### Issues Identified by Severity

- 🔴 **Critical (2):** missing architecture document; missing epics.
- 🟠 **Major (3):** regression corpus not captured; EPIC-CHIRPD-CORE risks being a technical-milestone epic if authored as proposed; EPIC-DAEMON-LIFECYCLE has marginal standalone user value.
- 🟡 **Minor (5):** epic-naming convention leans technical; `pyproject.toml` work needs split-epic landing; FR47 vs EPIC-WF-ALIGN precondition not explicit; OQ6 migration-messaging affects init UX; `models.toml` creation timing should be tied to first use, not preemptive.

### Final Note

This assessment identified **10 issues across 5 categories** (PRD/architecture/epics/UX/data-capture). The PRD is high-quality and ready to drive downstream work. The critical issues — missing architecture, missing epics, missing regression corpus — are the expected next stages of the workflow rather than defects in what's been done. Address them in the sequence above before implementation begins.

These findings can be used to improve the artifacts or you may choose to proceed as-is. If proceeding without resolving the critical issues, expect rework when those gaps surface during implementation (architecture decisions made implicitly during coding tend to drift between stories; epics authored without quality remediation tend to produce stories with forward dependencies).

---

**Report generated:** 2026-05-12
**Assessor:** Claude (PM facilitator)
**PRD reviewed:** `_bmad-output/planning-artifacts/prd.md`
