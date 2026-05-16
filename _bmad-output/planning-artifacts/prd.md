---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02b-vision
  - step-02c-executive-summary
  - step-03-success
  - step-04-journeys
  - step-05-domain
  - step-06-innovation
  - step-07-project-type
  - step-08-scoping
  - step-09-functional
  - step-10-nonfunctional
  - step-11-polish
  - step-12-complete
inputDocuments:
  - _bmad-output/planning-artifacts/epic-audio-capture/epic.md
  - _bmad-output/planning-artifacts/epic-wireframe-alignment/epic.md
  - AGENTS.md
  - CLAUDE.md
documentCounts:
  briefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 4
workflowType: 'prd'
projectType: brownfield
classification:
  projectType: cli_tool
  domain: scientific
  complexity: medium
  projectContext: brownfield
---

# Product Requirements Document - chirp-ai-note-app

**Author:** Colby
**Date:** 2026-05-12

## Executive Summary

Chirp is a local-first voice notes CLI for macOS. It records audio, transcribes locally with Whisper, generates structured notes with a local LLM, and lets users search and chat over those notes — all without sending data off-device. Target users are knowledge workers, researchers, and developers who want personal note-taking that is private by default and works offline.

Today, chirp depends on Ollama running as a separately-installed background daemon to host its LLM. Ollama works, but it forces every new user through a third-party install, adds a process the user must manage independently of chirp's lifecycle, and ties the project to whichever models Ollama supports well. This PRD replaces that dependency with a chirp-owned inference daemon (`chirpd`) that loads MLX models in-process and ships inside the pip package.

### What Makes This Special

After this work, chirp has zero third-party install requirements beyond `pip install chirp`. EPIC-AUDIO-CAPTURE removed BlackHole; this PRD removes Ollama. Users still bring their own models — they pick any MLX-format model from HuggingFace via `chirp models add <hf-repo>` — but they never install or configure a separate inference runtime. Daemon behavior mirrors what users already understand from Ollama (always-on lightweight daemon, lazy-loaded model, idle unload after 5 minutes) so the mental model is familiar, but the daemon ships inside chirp and respects chirp's lifecycle, logging, and version.

The core insight: MLX on Apple Silicon has crossed the threshold where Python can host inference directly with competitive performance (~2× llama.cpp Metal on the same 4-bit quant), and the HuggingFace cache solves model distribution for free. The Ollama process is no longer pulling its weight as a separate concern.

## Project Classification

| Attribute | Value |
|---|---|
| Project Type | CLI tool (Typer-based, Python, macOS-targeted) |
| Domain | Scientific / on-device ML tooling |
| Complexity | Medium — low-complexity domain, real technical complexity in daemon lifecycle (IPC, version handshake, LaunchAgent, MLX model lifecycle) |
| Project Context | Brownfield — replaces Ollama integration in an established CLI |
| Platform | macOS only; Apple Silicon required (Intel support explicitly dropped) |

## Success Criteria

### User Success

- **Zero-install LLM stack.** A new user runs `pip install chirp` then `chirp init` on a clean Apple Silicon Mac and reaches a working `chirp ask` without ever invoking `brew`, downloading a `.pkg`, or running a third-party setup tool. `chirp init` no longer mentions Ollama in any branch.
- **Single-command model add.** `chirp models add mlx-community/gemma-4-4b-it-4bit` downloads, registers, and (if no chat default exists) sets the model as the default, ending in a usable state with no follow-up commands required.
- **First-token latency feels familiar.**
  - Cold path (model not loaded): first streamed token within 5 seconds for a 4-bit 7B on M2 / 16 GB.
  - Warm path (model resident): first streamed token within 500 ms.
- **No regression in note quality.** On a held-out regression set of ≥10 representative recordings, blinded comparison of notes generated before vs. after migration shows MLX outputs rated equal-or-better on at least 80% of cases. (Same prompt templates, equivalent-class model, same Whisper transcript input.)
- **Daemon is invisible when things work.** Users running `record`, `transcribe`, `notes`, `ask`, `search` never need to interact with `chirp daemon` directly. The daemon-management subcommand exists for diagnostics; happy-path users never see it.

### Business Success

(Single-developer / open-source project — "business" reads as project health and support burden.)

- **Support burden drops.** Zero new issues filed related to Ollama install, version mismatch, daemon connection, or `OLLAMA_HOST` configuration in the 30 days following release. Existing Ollama-related open issues close as "resolved by chirpd migration."
- **Documentation surface shrinks.** README, AGENTS.md, and `chirp init` help text contain zero references to Ollama post-merge. `pyproject.toml` removes the Ollama Python client dependency.
- **Install-step count.** Number of human-visible install steps from clean Mac to first `chirp ask` reduces from current N (pip install chirp + brew install ollama + ollama serve + ollama pull + chirp init) to 2 (pip install chirp + chirp init).

### Technical Success

- **Daemon idle footprint.** `chirpd` resident memory with no model loaded ≤ 150 MB.
- **Loaded-model footprint matches model.** RSS with a model loaded equals the model's on-disk size ±10%. No leaks beyond first 30 minutes of normal use.
- **Cold model load time.** Loading a 4-bit 7B model from HF cache to ready-for-inference ≤ 5 s on M2 / 16 GB; ≤ 8 s on M1.
- **Idle unload reliability.** With the default 5-minute idle timer, an idle model is unloaded within configured interval ± 10%. RSS returns to baseline after unload.
- **Version handshake.** Every client connect issues `hello` and detects daemon/CLI version skew. On mismatch, the client transparently triggers a daemon restart and retries exactly once; user observes a one-shot pause, never a stuck connection or stale-behavior bug.
- **Concurrency.** A long-running streaming `chat` request and a one-shot `embed` request issued in parallel both complete correctly (the embed model is pinned; the chat model is held resident through both).
- **Cancellation.** Sending `cancel` for an in-flight `chat` halts generation within 200 ms and frees the model for the next request.
- **Survival.** `chirpd` survives a 30-day continuous run on a developer machine without resident memory growing beyond `baseline + (sum of loaded models)`.
- **Test + quality gates.** `make check` and `make test` pass. Coverage on `chirp/llm`, `chirpd`, and the model-registry module ≥ 90% line coverage.

### Measurable Outcomes

| Metric | Target |
|---|---|
| Install steps from clean Mac to first `chirp ask` | 2 (was 5+) |
| First-token latency, cold | ≤ 5 s (M2, 4-bit 7B) |
| First-token latency, warm | ≤ 500 ms |
| Cold model load | ≤ 5 s (M2, 4-bit 7B) |
| Daemon idle RAM | ≤ 150 MB |
| Idle-unload accuracy | ± 10% of configured timeout |
| Notes-quality regression (blinded A/B) | ≥ 80% equal-or-better |
| Coverage on new modules | ≥ 90% line |
| 30-day issue count for Ollama-class bugs post-launch | 0 |

## Product Scope

### MVP — Minimum Viable Product

Everything needed to remove Ollama from the codebase and the user's machine, with feature parity for the existing CLI surface.

1. **`chirpd` daemon process.** Long-running Python process, unix-domain socket at `~/Library/Application Support/chirp/chirpd.sock`, NDJSON line protocol, `flock`-based single-instance.
2. **Daemon ops:** `chat` (streaming), `embed`, `cancel`, `model.list`, `model.load`, `model.unload`, `model.status`, `health`, `hello` (version handshake).
3. **`chirp.llm` client library.** Auto-discovers socket, lazy-spawns daemon if absent, transparent one-retry on version mismatch or broken pipe, returns sync or streaming responses.
4. **Model registry** at `~/Library/Application Support/chirp/models.toml`. Aliases mapped to HF repos, role = `chat` or `embed`, per-model option defaults, `default_chat` and `default_embed` keys, `schema_version`.
5. **Model resolution rules:** alias → `default_chat`/`default_embed` → raw `org/repo` fallback (for testing).
6. **HuggingFace cache reuse.** Model files live in `~/.cache/huggingface/hub/`; respect `HF_HOME`.
7. **CLI subcommand: `chirp models`** with `list`, `add`, `remove`, `default`, `pull`, `show`. `add` validates repo, infers role, downloads, registers, sets as default if first of its role, and warms the model.
8. **CLI subcommand: `chirp daemon`** (hidden / maintenance tier) with `status`, `start`, `stop`, `restart`, `enable`, `disable`, `logs`. `enable`/`disable` manage the LaunchAgent.
9. **Lifecycle.**
   - Hybrid start: lazy-spawn from any client + opt-in `~/Library/LaunchAgents/com.chirp.chirpd.plist` (LaunchAgent created by `chirp daemon enable`).
   - Daemon process: indefinite (no auto-exit).
   - Model resident lifetime: configurable idle-unload, default 5 minutes (matches Ollama).
   - Embed-role models pinned (never auto-unloaded).
10. **Version handshake.** Every client connect sends `hello {client_version}`. Daemon compares against own version; on mismatch returns `{event: "version_mismatch"}` and exits immediately. Client transparently respawns and retries once.
11. **Integration into existing modules.** `notes`, `notes_chat`, and any other Ollama-touching code routes through `chirp.llm` instead of the Ollama client. Prompts and templates stay in their existing packages.
12. **`chirp init` updated.** Drops all Ollama checks; adds daemon-readiness check (socket exists or lazy-spawn succeeds, `health` op returns OK); offers to install LaunchAgent.
13. **Logging.** `~/Library/Logs/chirp/chirpd.log`, rotating, ~10 MB cap. `chirp daemon logs` tails it.
14. **Dependency removal.** Ollama Python client removed from `pyproject.toml`. `mlx-lm` and `huggingface_hub` added.
15. **Docs.** README, AGENTS.md, and `chirp --help` output updated to reflect new model-management story. No Ollama references remain.

### Growth Features (Post-MVP)

- **Per-task model overrides.** Different models for `notes` vs. `ask` vs. quick-title-gen, configured in `models.toml`.
- **Multi-model concurrent loading.** Hold multiple chat models resident when RAM permits, controlled by a `max_loaded_models` setting analogous to `OLLAMA_MAX_LOADED_MODELS`.
- **Background pre-warm.** Optional `chirp init` step that loads the default model on first run so the first `ask` is warm.
- **Telemetry surface.** `chirp daemon status` exposes per-op latency, tokens/sec, recent request log.
- **Custom prompt profiles in the registry.** Per-model system-prompt and option presets users can name and select.

### Vision (Future)

- **Menu bar app** sharing the daemon — explicitly out of scope here, captured as a separate future epic. The daemon protocol is designed to support it.
- **Apple Foundation Models backend.** Opt-in second backend behind the same `LLMBackend` protocol for users on macOS 26 with capable hardware, providing a zero-download baseline.
- **Pluggable backend protocol.** Generalize beyond MLX so external tools or remote inference endpoints can be wired in.

## User Journeys

### Journey 1 — Maya, the fresh installer (Primary, happy path)

Maya is a PhD candidate on an M3 MacBook Air. She heard about chirp from a Mastodon thread that called it "the local Whisper-to-notes CLI." She has never installed Ollama and doesn't want to.

**Opening.** She runs `pip install chirp`. The install completes in under a minute. She types `chirp init`. The init flow walks her through screen-recording and microphone permission prompts (carried over from the BlackHole-removal work), then says: "No language model installed yet. Add the recommended model with `chirp models add mlx-community/gemma-4-4b-it-4bit`, or the smaller-footprint alternative `mlx-community/gemma-4-e2b-it-8bit` if RAM is tight."

**Rising action.** She runs the suggested command. A progress bar streams the download from HuggingFace. When it finishes, chirp says: "Model registered as `gemma-4-4b-it-4bit`, set as default chat. Warming up..." After a few seconds: "Ready." She runs `chirp record`, talks for two minutes about her thesis, hits Ctrl-C, and runs `chirp transcribe`. The 5-stage checklist completes — load audio, transcribe, generate notes, index, save — and her note appears at `~/Documents/chirp/<slug>/notes.md`.

**Resolution.** Total install-to-first-note time: under five minutes. She never opened the App Store, never touched Homebrew, never read documentation about a second daemon. The only thing she had to make a choice about was which model to download.

**This journey reveals requirements for:** clean `chirp init` flow (no Ollama branches), HF-backed `chirp models add` with progress feedback, auto-set-as-default behavior, auto-warm on add, lazy-spawn daemon (she never started one explicitly).

---

### Journey 2 — Devon, the migrating user (Primary, edge case)

Devon has been using chirp for six months. He has Ollama installed via Homebrew, `llama3.2` pulled, and a workflow he likes. He sees a release note: "Chirp 2.0 replaces Ollama with an in-process MLX daemon. Run `chirp init --recheck` to migrate."

**Opening.** He runs `pip install -U chirp`. Then `chirp init --recheck`. Chirp detects: Ollama installed; no chirp model registered yet; existing notes and config present. It prints a migration plan: "We're replacing Ollama with a bundled inference daemon. Your existing notes and config are unchanged. You'll need to pick a new model (MLX format, not GGUF). Recommended: `mlx-community/gemma-4-4b-it-4bit`. Ollama itself is left installed — uninstall it manually with `brew uninstall ollama` once you're satisfied with the new setup."

**Rising action.** He picks the suggested model. Chirp downloads it, registers it, sets it as default, warms it. He runs `chirp ask "what did I discuss with my advisor last week"` — his existing notes index, generated against the old model's prompts, still works (prompts and RAG live in chirp, not in Ollama). The first answer streams back in under five seconds.

**Resolution.** Three days later, satisfied nothing broke, he runs `brew uninstall ollama`. His Mac is now down one daemon and one menu-bar icon. Disk reclaimed: ~5 GB of unused GGUF files (he deletes those manually from `~/.ollama`).

**This journey reveals requirements for:** Ollama-detection in `chirp init --recheck`, a clear migration message, **no auto-uninstall of Ollama** (we don't touch the user's system), reuse of existing chroma index and notes layout, friendly UX around the existing `--switch-model` flag.

---

### Journey 3 — Priya, the power user / model swapper (Primary, exploratory)

Priya is a frontend engineer who installed chirp last month. She finds the default 4B great for quick chat but wants richer notes on longer recordings, so she's adding a 7B as a second model and flipping defaults per workflow.

**Opening.** She runs `chirp models list`. Output: the registered `gemma-4-4b-it-4bit` as default, marked loaded. She runs `chirp models add mlx-community/Qwen2.5-7B-Instruct-4bit --alias quality`. Download completes in ~90 seconds.

**Rising action.** Her default 4B handles `chirp ask "summarize today's note in three bullets"` snappily — first token in under half a second. But when she runs `chirp transcribe` on a 30-minute recording, the notes come out shallow. She runs `chirp models default quality` to switch to the 7B before her next long transcript, then `chirp models default gemma-4-4b-it-4bit` to flip back for quick ad-hoc questions.

**Resolution.** She has both models registered. The fast 4B is her default; she flips to `quality` when she needs deeper notes. She mentally files this as "I wish I could set per-command model defaults" — captured for Growth scope.

**This journey reveals requirements for:** `chirp models default <alias>` flip, `--alias` flag on `add`, the `chirp models list` showing which model is currently resident, and the Growth feature of per-task model overrides.

---

### Journey 4 — Sam, the troubleshooter (Recovery path)

Sam upgraded chirp last night. This morning, `chirp ask` returns: "Daemon version mismatch detected. Restarting... done. Retrying request..." and then the answer streams normally. The whole pause is about a second. He doesn't think much of it — the handshake worked.

**Later that day**, Sam tries an unusual model — a freshly-uploaded HF repo. `chirp models add <repo>` succeeds (download works) but `chirp ask` fails with: "Model failed to load: unsupported architecture. Run `chirp daemon logs` for details." He runs `chirp daemon logs` and sees a stack trace from `mlx_lm.load`. He removes the broken entry with `chirp models remove <alias>` and goes back to his prior default. Chirp's CLI surfaced enough to diagnose this without him learning MLX internals.

**Another scenario**: Sam closes his laptop for the weekend. Monday morning, the LaunchAgent (which he enabled with `chirp daemon enable`) is running again — `chirp daemon status` shows: "Running, uptime 4m 12s, no model loaded, idle 4m." His first `ask` takes the cold-start pause, then runs normally.

**This journey reveals requirements for:** version handshake with single transparent retry, clear error messages distinguishing "download" vs "load" failures, `chirp daemon logs` and `chirp daemon status` as diagnostic tools, LaunchAgent reliability, idle-unload visibility in `status` output.

---

### Journey Requirements Summary

| Capability | Required by journey |
|---|---|
| Lazy-spawn daemon from any CLI invocation | Maya, Priya |
| `chirp models add <hf-repo>` with progress, auto-register, auto-set-default, auto-warm | Maya, Priya |
| `chirp models list` showing default, registered, loaded state | Priya |
| `chirp models default <alias>` flip | Priya |
| `chirp models remove <alias>` cleanup | Sam |
| Distinguish download failure vs. load failure in errors | Sam |
| Version-stamped `hello` handshake with one-shot transparent restart-and-retry | Sam |
| `chirp daemon logs` and `chirp daemon status` for diagnostics | Sam |
| `chirp daemon enable` writes LaunchAgent, persists across reboots | Sam |
| `chirp init --recheck` detects pre-existing Ollama, prints migration plan, **does not touch** Ollama | Devon |
| Existing notes and chroma index continue to work post-migration | Devon |
| Idle-unload visible in `daemon status` | Sam |
| Streaming first-token latency cold ≤ 5s / warm ≤ 500ms | Maya, Devon, Priya |

## Domain-Specific Requirements

### On-Device Privacy (Hard Constraint)

Chirp's core value proposition is that voice, transcripts, and notes never leave the device. The Ollama → MLX migration must not introduce any code path that uploads user audio, transcripts, notes, or LLM prompts to a remote service.

- **No telemetry from `chirpd`.** The daemon emits logs locally to `~/Library/Logs/chirp/chirpd.log`. It never opens a network socket to any host other than HuggingFace (for model downloads, user-initiated only).
- **No remote-inference fallback.** The architecture must not include "if local inference fails, try a cloud endpoint" logic in MVP. Pluggable remote backends are a Vision-tier item explicitly gated behind user opt-in.
- **Model downloads from HuggingFace are user-initiated.** Only `chirp models add` and `chirp models pull` make outbound HTTP requests, and only against `huggingface.co`. `chirpd` itself never reaches out.
- **No background analytics, crash reporting, or update checks.** Updates are the user's responsibility via `pip install -U chirp`.

### Validation Methodology (Notes-Quality Regression)

The migration replaces the model that produces notes. We need a defined methodology to confirm note quality is preserved.

- **Regression corpus.** A held-out set of ≥10 representative recordings (varying length, topic, speaker count) is captured before this work begins. Each recording is paired with its existing chirp-generated `notes.md` produced by the current Ollama-backed pipeline.
- **Comparison protocol.** For each recording in the corpus, generate notes via the new MLX pipeline using the same Whisper transcript as input and equivalent prompt templates. Score `before` vs. `after` notes blindly (author shuffles, hides which is which) against the qualitative criteria the prompts target (structure, faithfulness to source, completeness, brevity).
- **Pass threshold.** ≥80% of comparisons rated "equal or better" for the new pipeline (recorded in §Success Criteria → Measurable Outcomes).
- **Tracked artifact.** The regression corpus and scores live under `tests/regression/notes_quality/` and are committed (transcripts and scores, not audio).

### Computational Resource Budget

The product runs on consumer Apple Silicon Macs. Resource budgets are part of the domain contract.

- **Daemon idle RSS** ≤ 150 MB (Python interpreter + mlx-lm imports + small embedding model resident).
- **Loaded-chat-model RSS** matches model's on-disk size ± 10%. No leak beyond first 30 minutes of steady-state use.
- **Embed model RSS** ≤ 200 MB (pinned, always resident).
- **Disk** — chirp's own footprint is unchanged; model weights live in the HF cache, which the user manages.
- **Network** — only on `chirp models add`/`pull`. No background usage.

### Domain Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MLX quality regression vs. equivalent Ollama/GGUF model on identical prompts | Validation methodology above; ship hidden behind a `CHIRP_BACKEND` env override during transition so power users can A/B if needed |
| User picks a model with unsupported MLX architecture | `chirp models add` validates against `mlx_lm` loadability before registering; clear error message distinguishes download success vs. load failure |
| HuggingFace cache lives outside chirp's control (user could wipe it) | Daemon detects missing weights at load time; CLI command `chirp models pull <alias>` re-downloads |
| Apple changes MLX APIs between minor versions | Pin `mlx-lm` minimum version in `pyproject.toml`; CI runs against the pinned version |
| Long-running daemon accumulates memory over weeks | Documented `chirp daemon restart` and the 30-day stability target; logs RSS in `chirp daemon status` so users can spot drift |

## CLI Tool Specific Requirements

### Project-Type Overview

Chirp is a Typer-based Python CLI. Existing user-visible commands (locked by EPIC-WF-ALIGN): `record · transcribe · notes · ask · search · init · about`. Hidden maintenance commands: `config · devices · index`. This PRD adds two subcommand groups and modifies one existing command:

- **New (visible):** `chirp models` — model registry management. User-facing because users must add a model before chirp is functional.
- **New (hidden / maintenance tier):** `chirp daemon` — daemon lifecycle and diagnostics. Hidden because happy-path users never need it; surfaced when troubleshooting.
- **Modified:** `chirp init` — drops Ollama branches; adds daemon-readiness check; offers LaunchAgent install.

### Command Structure

```
chirp models
  list                              Show registered models, defaults, loaded state
        [--json]                    Machine-readable output
  add <hf-repo>                     Download, register, set default if first of role, warm
        [--alias <name>]            Override inferred alias
        [--role chat|embed]         Force role (otherwise inferred)
        [--no-warm]                 Skip post-add load
  remove <alias>                    Remove from registry
        [--purge]                   Also delete files from HF cache
  default <alias>                   Set as default for whichever role this alias has
  pull <alias>                      Force re-download / repair
  show <alias>                      Print resolved config (alias, hf_repo, role, options)
        [--json]

chirp daemon                        Hidden command group (not in main --help)
  status                            Running? Model loaded? RAM? Uptime? Idle timer?
        [--json]
  start                             Explicit spawn (rarely needed; lazy handles it)
  stop                              Stop daemon (in-flight requests fail with a typed error)
  restart                           stop + start
  enable                            Install LaunchAgent (~/Library/LaunchAgents/)
  disable                           Uninstall LaunchAgent
  logs                              Tail ~/Library/Logs/chirp/chirpd.log
        [-f | --follow]
        [-n <lines>]
```

The existing 7-command surface (`record`, `transcribe`, `notes`, `ask`, `search`, `init`, `about`) remains visible and unchanged in shape. `daemon` joins the hidden tier alongside `config`/`devices`/`index`. `models` joins the visible tier — model management is a primary workflow.

### Output Formats

- **Default:** human-readable text with Rich rendering when stdout is a TTY (matches existing chirp conventions).
- **`--json` flag** on `models list`, `models show`, `daemon status`: emits a single JSON document to stdout, suitable for `jq` piping. No log lines on stdout in JSON mode.
- **Streaming output** for `ask` / chat-mode flows: tokens written to stdout as they arrive. The CLI client transforms NDJSON `{event: "delta", text: "..."}` frames from the daemon into a clean text stream — JSON framing stays internal. Stderr carries diagnostic notices ("model loading...", "daemon respawned for version handshake").
- **Progress feedback** for `models add` / `models pull`: a Rich progress bar with bytes-downloaded and ETA, fed by `huggingface_hub` callbacks. TTY only; JSON/non-TTY mode emits start/done lines on stderr.

### Config Schema

Two TOML files; both human-editable.

**`~/.chirp/config.toml`** (existing — unchanged shape):

```toml
notes_root = "~/Documents/chirp"
chroma_path = "~/.chirp/chroma"

[llm]
backend = "chirpd"                  # was "ollama"; default flips post-migration
daemon_socket = "~/Library/Application Support/chirp/chirpd.sock"
idle_timeout_seconds = 300          # model unload after idle; 0 = never unload
```

**`~/Library/Application Support/chirp/models.toml`** (new):

```toml
schema_version = 1
default_chat = "gemma-4-4b-it-4bit"
default_embed = "bge-small-en-v1.5"

[models."gemma-4-4b-it-4bit"]
hf_repo = "mlx-community/gemma-4-4b-it-4bit"
role = "chat"
options = { temperature = 0.7, top_p = 0.9, max_tokens = 2048 }

[models."bge-small-en-v1.5"]
hf_repo = "mlx-community/bge-small-en-v1.5"
role = "embed"
```

**Environment variable overrides** (precedence: env > config file > built-in default):

| Variable | Purpose |
|---|---|
| `CHIRP_DAEMON_SOCKET` | Override socket path (testing, multi-user) |
| `CHIRP_MODEL_IDLE_TIMEOUT` | Override idle-unload timer in seconds |
| `CHIRP_BACKEND` | Force backend selection (`chirpd` MVP; future: `apple-foundation`) |
| `HF_HOME` | Standard HuggingFace cache relocation; respected automatically |

### Scripting Support

- **Exit codes:**
  - `0` — success
  - `1` — generic error (with stderr message)
  - `2` — usage error / invalid argument (Typer default)
  - `3` — daemon unreachable after lazy-spawn + retry
  - `4` — model load failure
  - `5` — model not found / not registered
- **Stdin:** `chirp ask -` reads the prompt from stdin (useful for `echo "summarize this" | chirp ask -`). Existing convention is preserved.
- **Stdout / Stderr discipline:** primary command output goes to stdout; diagnostics and progress go to stderr. `--json` mode guarantees only valid JSON on stdout.
- **Tab completion:** Typer's built-in shell completion (`chirp --install-completion`) covers new subcommands automatically. The `<alias>` argument on `models default|remove|show|pull` uses a completion callback that reads `models.toml` to suggest registered aliases.
- **Non-interactive safety:** none of the new commands prompt interactively. `models add` does not ask for confirmation; failures are loud and unambiguous.

### Implementation Considerations

- **Typer panels:** add a new `MODELS_PANEL` next to `MAIN_PANEL` (visible) for `models` subcommands; the `daemon` group joins the existing hidden-command set (no panel registration).
- **Streaming buffer.** The CLI client must flush stdout after each token to avoid block-buffering when piped; use `sys.stdout.write(token); sys.stdout.flush()` or `print(token, end="", flush=True)`.
- **Lazy-spawn race.** When two CLI invocations start simultaneously and both lazy-spawn, the `flock` on `chirpd.lock` guarantees only one daemon survives; the loser exits cleanly and the second client retries connecting to the surviving socket.
- **Help text discipline.** Per AGENTS.md "validate doc updates against live CLI help before finishing" — every new flag and subcommand description must be checked against `uv run chirp --help` and `uv run chirp models --help` etc. before sign-off.

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

This is a **substitution MVP** — replace one well-understood component (Ollama) with a chirp-owned equivalent (`chirpd` + MLX), preserving feature parity for every existing user-facing surface. Success is defined by what users *don't notice* (notes quality unchanged, ask/search still work) plus what they *do notice* (no Ollama install step). It is explicitly not an opportunity to redesign the prompt layer, the retrieval layer, or the CLI surface beyond the two new subcommand groups.

**MVP boundary test.** A feature is in MVP if removing it would either:
1. Leave Ollama referenced anywhere in the codebase, dependencies, or docs, or
2. Break a journey from §User Journeys (Maya, Devon, Priya, or Sam).

Everything that fails both tests is Growth or Vision.

### MVP Feature Set (Phase 1)

The MVP feature list lives under §Success Criteria → Product Scope → MVP. Summary:

- `chirpd` daemon + NDJSON protocol + `chirp.llm` client
- Model registry (`models.toml`) and `chirp models` subcommand group
- Daemon lifecycle: hybrid lazy-spawn + LaunchAgent, idle-unload, embed-pinning
- Version-stamped handshake with auto-restart
- `chirp daemon` hidden subcommand group
- `chirp init` updates (no Ollama branches)
- Existing modules (`notes`, `notes_chat`) routed through `chirp.llm`
- Dependency swap in `pyproject.toml`; docs updated

### Post-MVP Features

Phase 2 (Growth) and Phase 3 (Vision) lists live under §Success Criteria → Product Scope. Highlights:

- **Phase 2:** per-task model overrides, multi-model concurrent loading, background pre-warm, telemetry surface, custom prompt profiles.
- **Phase 3:** menu bar app sharing the daemon (separate epic), Apple Foundation Models backend, pluggable remote backends.

### Explicit Out of Scope (MVP)

These have been considered and deliberately excluded. Listing them prevents re-litigation.

| Out of Scope | Why |
|---|---|
| **Menu bar app** | Separate future epic. Daemon protocol is designed to support it, but UI work is out of scope here. |
| **Intel Mac support** | User explicitly dropped it. MLX is Apple Silicon only; no fallback engine in MVP. `chirp init` on Intel must fail loudly with a clear message. |
| **Ollama-to-MLX model conversion / migration** | GGUF and MLX are different formats. Users re-download an MLX equivalent of their preferred model. Migration tooling is high effort, low value. |
| **Auto-uninstall of Ollama** | Don't touch the user's system. Surface a recommendation in `chirp init --recheck`; user uninstalls manually. |
| **OpenAI-compatible HTTP endpoint** | Ollama provides this; nobody in chirp's userbase is using it. Adding it expands surface for no current user. |
| **Per-task model selection in MVP** | Single default chat + single default embed is sufficient for parity. Per-task overrides land as a Growth feature once users actually request it. |
| **Remote / cloud inference fallback** | Violates on-device privacy hard constraint (§Domain Requirements). |
| **GPU offload to external eGPU / non-Apple-Silicon accelerators** | MLX targets Apple Silicon unified memory; this is the supported architecture. |
| **App Store packaging** | Out of scope; this PRD's goal is removing third-party install friction, not changing distribution. Future-compatible (no kernel extensions, no daemons-by-pkg) but not delivered. |
| **Inference quality benchmarks beyond regression corpus** | The 10-recording held-out set is sufficient. Comprehensive eval suites are deferred. |
| **Hot-reload of `models.toml` without daemon restart** | Daemon reads on startup and on `model.*` ops; users editing the file manually must `chirp daemon restart`. Edge case; not worth the file-watcher complexity. |

### Scoping Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MLX 4-bit 7B notes quality lags Ollama's equivalent | Medium | High — breaks no-regression bar | Validation corpus runs before merge. If quality fails, escalate model size (8-bit 7B or 4-bit 14B) before declaring the migration done. |
| Cold-start exceeds 5s on M1 with large models | Medium | Medium | Document expected cold-start per model size in `chirp models show`; surface load-progress events in CLI; allow `keep_alive=-1` to pin. |
| `mlx-lm` minor-version API churn during build | Medium | Medium | Pin a tested minimum version in `pyproject.toml`; CI runs against the pinned version. |
| LaunchAgent setup fails silently on macOS variants | Low | Medium | `chirp daemon enable` runs `launchctl load` synchronously and verifies process appears; report exit code clearly. |
| Daemon hangs blocking shutdown during `pip install -U` | Low | Medium | Version-handshake-triggered exit is immediate (alpha-simplified, no drain). Defensive: if daemon doesn't exit within 2 s, client sends SIGTERM, then SIGKILL after 3 s. |
| HF cache corruption (partial download, network drop) | Medium | Low | `chirp models pull <alias>` is the documented repair. `models add` retries downloads on transient HTTP failure. |
| `chirpd` memory leak surfaces only at 7+ day uptime | Medium | Low | 30-day stability target is observation, not gate — ship with documented `chirp daemon restart` as escape valve. |

### Resource Scope

Solo developer (Colby). No parallel implementers. Implication for sequencing:
- Daemon protocol + client wrapper must land before existing modules can be migrated (blocking).
- Model registry + `chirp models` subcommand can land in parallel with daemon work (independent paths).
- `chirp init` updates and dependency removal land last (cleanup).
- Each phase should pass `make check` and `make test` before the next merges — no long-lived migration branch.

## Functional Requirements

The FR list below is the capability contract for this PRD. Every feature delivered by the Ollama→MLX migration must trace back to an FR here. Anything not listed is out of scope unless an FR is added explicitly.

### Inference Daemon (chirpd) — Core Operations

- **FR1:** `chirpd` accepts streaming chat-completion requests against a configured chat model, returning incremental token deltas terminated by a `done` event.
- **FR2:** `chirpd` accepts embedding requests over one or more text inputs and returns one vector per input.
- **FR3:** `chirpd` accepts cancel requests that terminate an in-flight chat request and free the model for the next request within a bounded latency.
- **FR4:** `chirpd` accepts a model-status request that reports currently-loaded models, last-used timestamps, idle countdown, RSS, and process uptime.
- **FR5:** `chirpd` loads a registered chat model on demand and unloads it after a configurable idle period (default: 5 minutes).
- **FR6:** `chirpd` keeps embedding-role models resident indefinitely; the idle-unload timer does not apply to embedding models.
- **FR7:** `chirpd` honors a per-request `keep_alive` override (`-1` = pin indefinitely; `0` = unload immediately after request completes).
- **FR8:** `chirpd` applies each model's chat template (sourced from the model's tokenizer config) to incoming message arrays before generation.
- **FR9:** `chirpd` accepts an explicit `model.load` request that loads a registered model without issuing a chat or embed.
- **FR10:** `chirpd` accepts an explicit `model.unload` request that frees a currently-loaded model from RAM.

### Daemon Protocol, Lifecycle, and Version Handshake

- **FR11:** `chirpd` communicates with clients over a unix-domain socket using newline-delimited JSON messages.
- **FR12:** `chirpd` enforces a single-instance invariant — only one daemon process owns the socket at a time.
- **FR13:** `chirpd` accepts a version-stamped `hello` request from each connecting client and reports a version mismatch when the client's version differs from its own.
- **FR14:** `chirpd` exits immediately after a version mismatch is reported. In-flight requests at the moment of mismatch fail with a clear typed error; the client's one-shot retry against the respawned daemon succeeds. **(Simplified for alpha — no backwards-compat obligation; users running long requests across a mid-flight upgrade is not a supported scenario.)**
- **FR15:** `chirpd` exposes a health endpoint suitable for use by `chirp init` doctor checks.
- **FR16:** `chirpd` writes structured logs to a rotating local logfile under `~/Library/Logs/chirp/`.
- **FR17:** `chirpd` emits no network traffic except as part of explicit user-initiated HuggingFace model downloads.

### Client Library (`chirp.llm`)

- **FR18:** The `chirp.llm` client auto-discovers the daemon socket at the configured path, honoring environment overrides.
- **FR19:** The `chirp.llm` client lazy-spawns `chirpd` when no socket is present and waits for the socket to accept connections up to a bounded timeout.
- **FR20:** The `chirp.llm` client transparently retries exactly once on a detected version mismatch or broken pipe; subsequent failures surface to the caller.
- **FR21:** The `chirp.llm` client exposes both streaming and non-streaming chat APIs.
- **FR22:** The `chirp.llm` client exposes a batched embedding API accepting a list of inputs.
- **FR23:** The `chirp.llm` client distinguishes model-load failures from transport failures in the exception types it raises.

### Model Registry — User-Facing Capabilities

- **FR24:** User can register a model by HuggingFace repo identifier via `chirp models add <hf-repo>`.
- **FR25:** User can override the inferred alias and/or role via `--alias` and `--role` flags on `chirp models add`.
- **FR26:** User can list all registered models with role, default-flag, downloaded-status, and currently-loaded state via `chirp models list`.
- **FR27:** User can set the default chat or embed model via `chirp models default <alias>`; the system selects which default to set based on the alias's registered role.
- **FR28:** User can remove a registered model from the registry, optionally deleting its weights from the HuggingFace cache via `--purge`.
- **FR29:** User can force re-download or repair a model's weights via `chirp models pull <alias>`.
- **FR30:** User can inspect a model's resolved configuration (alias, repo, role, options, current location on disk) via `chirp models show <alias>`.
- **FR31:** User can request machine-readable output via `--json` on `models list`, `models show`, and `daemon status`.

### Model Registry — System Behaviors

- **FR32:** The system infers `role = chat` or `role = embed` from the model's architecture when registering, falling back to requiring `--role` when ambiguous.
- **FR33:** The system automatically sets a newly-added model as the default for its role when no default for that role is currently set.
- **FR34:** The system warms (loads) a newly-added chat model after registration unless `--no-warm` is passed.
- **FR35:** The system resolves the literal model identifier `"default"` to the configured `default_chat` or `default_embed` based on the operation type.
- **FR36:** The system resolves a raw `<org>/<repo>` identifier (containing a slash) as a one-off model load when no matching alias exists in the registry.
- **FR37:** The system stores model registry state at `~/Library/Application Support/chirp/models.toml` with a schema-version field.
- **FR38:** The system reuses the HuggingFace cache (`~/.cache/huggingface/hub` by default, honoring `HF_HOME`) for model weight storage.

### Daemon Lifecycle Management — User-Facing Capabilities

- **FR39:** User can query daemon status (running flag, loaded models, RAM, uptime, idle countdown) via `chirp daemon status`.
- **FR40:** User can explicitly start the daemon via `chirp daemon start`.
- **FR41:** User can stop the daemon via `chirp daemon stop`. In-flight requests at the moment of stop fail with a typed error (no drain — alpha simplification).
- **FR42:** User can restart the daemon via `chirp daemon restart`.
- **FR43:** User can install a LaunchAgent for auto-start at login via `chirp daemon enable`.
- **FR44:** User can remove the LaunchAgent via `chirp daemon disable`.
- **FR45:** User can tail the daemon log file via `chirp daemon logs`, optionally following new output via `-f`.

### Integration with Existing Modules

- **FR46:** Existing note-generation (`chirp transcribe`), ask (`chirp ask`), and search (`chirp search`) workflows produce results of equivalent quality before and after migration when given identical inputs and an equivalent-class model, per the validation methodology in §Domain Requirements.
- **FR47:** Existing chroma index, notes directory layout, and `~/.chirp/config.toml` schema continue to function unchanged after migration.
- **FR48:** All prompt templates, retrieval logic, and note-format conventions remain in their existing Python modules; only the LLM call site routes through `chirp.llm`.

### First-Run, Init, and Migration Experience

- **FR49:** `chirp init` no longer performs or prompts about Ollama installation, configuration, or model pulls.
- **FR50:** `chirp init` detects whether a default chat model is registered and surfaces a clear next-step prompt referencing `chirp models add` when none is found.
- **FR51:** `chirp init` detects daemon readiness as part of its existing doctor-style flow, lazy-spawning the daemon if absent and confirming `health` returns OK.
- **FR52:** `chirp init` offers (but does not require) LaunchAgent installation, recording the user's choice for future runs.
- **FR53:** `chirp init --recheck` detects pre-existing Ollama installations and prints a migration plan without modifying the user's Ollama installation or data.
- **FR54:** `chirp init` on a non-Apple-Silicon Mac fails fast with a message naming the architecture constraint and exits with a distinct exit code.

### Documentation and Dependency Surface

- **FR55:** Project dependencies (`pyproject.toml`) remove the Ollama Python client and add `mlx-lm` and `huggingface_hub` pins.
- **FR56:** Project documentation (`README.md`, `AGENTS.md`, `CLAUDE.md`, `chirp --help` output) contains no references to Ollama post-merge.

## Non-Functional Requirements

Each NFR below is measurable. Targets and methodology are restated here to consolidate the quality contract; the same numbers appear in §Success Criteria → Technical Success / Measurable Outcomes and are authoritative there.

### Performance

- **NFR-P1: First-token latency, warm path.** With a chat model already loaded, `chat` requests return their first streaming delta within 500 ms (median over a 20-request rolling window) on M2 / 16 GB.
- **NFR-P2: First-token latency, cold path.** For a 4-bit 7B chat model loaded from the HF cache on a `chirpd` with no models resident, the first streaming delta arrives within 5 s on M2 / 16 GB. Within 8 s on M1 / 16 GB.
- **NFR-P3: Throughput.** Token generation throughput for a 4-bit 7B model on M2 / 16 GB sustains ≥ 30 tokens/sec during streaming generation.
- **NFR-P4: Cancellation latency.** A `cancel` request for an in-flight chat halts generation and frees the model for the next request within 200 ms.
- **NFR-P5: Idle-unload precision.** With the default 5-minute idle timer, an unused chat model is unloaded within ± 10% of the configured interval, returning RSS to baseline.
- **NFR-P6: Daemon spawn-to-ready.** Lazy-spawn of `chirpd` from a client invocation reaches socket-accepting state within 3 s on M2.
- **NFR-P7: Embedding throughput.** Batch `embed` requests on a pinned BGE-class model process ≥ 50 inputs/sec on M2.

### Reliability

- **NFR-R1: Daemon stability.** `chirpd` survives a 30-day continuous run on a developer machine without resident memory growing beyond `baseline + (sum of currently-loaded models)`.
- **NFR-R2: Crash recovery.** When `chirpd` is launched under LaunchAgent supervision and the process exits non-zero, launchd restarts it without manual intervention.
- **NFR-R3: Crash recovery (without LaunchAgent).** When `chirpd` is not under LaunchAgent supervision and crashes, the next CLI invocation lazy-spawns a replacement and surfaces a transparent one-shot retry.
- **NFR-R4: Single-instance enforcement.** Concurrent attempts to start two daemons resolve deterministically — one acquires the lockfile and survives; the other exits cleanly within 1 s.
- **NFR-R5: Version-drift recovery.** When a CLI invocation's version does not match the running daemon's version, the daemon exits immediately and the client transparently respawns and retries the request exactly once. The user-observed pause is ≤ 2 s on M2.
- **NFR-R6: Shutdown behavior.** On `daemon stop` or version-mismatch exit, `chirpd` terminates promptly. In-flight requests at exit fail with a typed error (`MODEL_GENERATION_FAILED` with cause `daemon_shutdown`); the CLI surfaces this clearly. **(Alpha simplification: no graceful drain. Users running long requests across mid-flight upgrades is not a supported scenario.)**
- **NFR-R7: Self-heal from missing model weights.** When a registered model's weights are missing from the HF cache at load time, the daemon returns a typed error that points the user to `chirp models pull <alias>`.

### Security & Privacy

- **NFR-S1: Local-only IPC.** The daemon listens only on a unix-domain socket within the user's `~/Library/Application Support/chirp/` directory; no TCP port is opened in MVP.
- **NFR-S2: Socket permissions.** The socket file is created with mode `0600` so only the owning user can connect.
- **NFR-S3: No outbound traffic from `chirpd`.** The daemon process makes no network requests of any kind. Outbound HTTP is limited to the CLI process during `chirp models add` / `pull`, targeting `huggingface.co` only.
- **NFR-S4: No telemetry.** No analytics, crash reports, or update pings are sent from any chirp component.
- **NFR-S5: Log redaction.** Daemon logs contain no user prompts, chat content, or note content. Only operation metadata (op name, model alias, token counts, durations, error class).
- **NFR-S6: Model-cache scoping.** The HuggingFace cache used by chirp is the standard cache (`~/.cache/huggingface/hub`) shared with other HF-aware tools; chirp does not write outside its own config and log directories beyond this cache.

### Maintainability

- **NFR-M1: Test coverage.** Line coverage on the new modules (`chirp.llm` client, `chirpd` daemon, model registry) is ≥ 90%. Coverage on touched existing modules does not regress.
- **NFR-M2: CI gating.** `make check` (validate, lint, format-check, spell-check, type-check) and `make test` pass on every commit that touches code in scope.
- **NFR-M3: Type discipline.** All new public APIs have type hints. mypy passes against `chirp`, `chirp.llm`, `chirpd`, and the new model-registry module.
- **NFR-M4: Dependency hygiene.** `pyproject.toml` pins a minimum `mlx-lm` version known-good against the CI matrix. Removed dependencies (Ollama client) are excised, not just unimported.
- **NFR-M5: Diagnostics surface.** `chirp daemon status` and `chirp daemon logs` provide sufficient information to triage the most common failure modes (daemon won't start, model won't load, version mismatch, idle-unload not firing) without reading source code.

### Compatibility & Portability

- **NFR-C1: macOS version floor.** Supported on macOS 13 (Ventura) and later, matching the existing EPIC-AUDIO-CAPTURE floor.
- **NFR-C2: Apple Silicon required.** MVP runs on Apple Silicon (M1, M2, M3, M4, and later) only. Intel Macs fail loudly at `chirp init` with NFR-C2 cited.
- **NFR-C3: Python version.** Matches the project's existing minimum Python version pinned in `pyproject.toml`; the migration does not raise the floor.
- **NFR-C4: HF cache compatibility.** Models downloaded by chirp are usable by other MLX-aware tools (LM Studio, raw `mlx_lm.load()`) without re-download.

### Observability

- **NFR-O1: Structured logs.** Daemon logs are logfmt-style key=value lines (resolved per Open Question 3) with timestamps, op-id correlation, severity, and operation metadata. Greppable with `awk`/`grep`; parseable with any logfmt library.
- **NFR-O2: Log rotation.** `chirpd.log` rotates at ~10 MB with at least one prior generation retained.
- **NFR-O3: Status detail.** `chirp daemon status --json` includes: daemon PID, uptime, daemon version, list of loaded models with per-model RSS, idle countdown per loaded model, total RSS, last request timestamp.

### Accessibility

- **NFR-A1: Terminal accessibility.** The CLI uses standard text output suitable for VoiceOver and other terminal-based screen readers; Rich-rendered tables and progress bars degrade gracefully in non-TTY contexts (no ANSI escape sequences on stdout in non-TTY mode).

## Open Questions

These are decisions deferred to implementation or first-epic planning. They do not block PRD approval; they are tracked so they get resolved deliberately rather than by drift.

- **OQ1: Default chat model for the docs.** ✅ **Resolved 2026-05-15:** `mlx-community/gemma-4-4b-it-4bit` as the primary recommendation; `mlx-community/gemma-4-e2b-it-8bit` offered as a smaller-footprint alternative for users on tighter RAM. Both repos recommended explicitly in the README, `chirp init` next-step prompt, and Devon's `--recheck` migration message. Defined as `RECOMMENDED_CHAT_REPO` and `SMALLER_CHAT_REPO` module constants so all references share one source of truth (EPIC-INIT-AND-MIGRATION story 7.1). The 4B at 4-bit fits comfortably under the 7B-calibrated NFR performance budgets; the E2B-8bit variant uses Gemma's effective-2B architecture so its resident footprint and activation memory are lower than a vanilla 4-bit 4B. Quality-sensitive users can register a 7B-class model as a second alias (per Priya's journey) or escalate via the regression-comparison process (EPIC-INTEGRATION-CUTOVER story 6.6).
- **OQ2: Default embed model.** Same question for embeddings. `mlx-community/bge-small-en-v1.5` is the obvious default for English content, but chirp users may record in other languages (Whisper auto-detects). A multilingual embed model would broaden support but is bigger. Decide before MVP merge.
- **OQ3: Logging format.** ✅ **Resolved 2026-05-12:** logfmt-style key=value lines (e.g., `ts=2026-05-12T14:32:01Z level=info op=chat req_id=r-abc model=gemma-4-4b-it-4bit duration_ms=612`). Easy to tail, easy to grep, easy to machine-parse with `awk` or any logfmt library. JSON output deferred to a Growth-tier `chirp daemon logs --json` flag if a need arises.
- **OQ4: Pre-warm on `chirp init`.** Should `chirp init` issue a `model.load` after registering the default model so the first `ask` is warm? Growth-tier feature in the current scope, but it's cheap to add to MVP if it removes the cold-start surprise from Maya's journey.
- **OQ5: `mlx-lm` version pin granularity.** ✅ **Resolved 2026-05-12:** exact pin in `pyproject.toml` (`mlx-lm == X.Y.Z` against the version CI is green on). Reasoning: mlx-lm is pre-1.0 with non-trivial API churn between minor versions. Cost of bumping the pin deliberately is lower than the risk of silent breakage from a transitive update. Re-evaluate to compatible-release (`~=`) once mlx-lm reaches stable 1.x.
- **OQ6: Migration messaging strategy.** Should `chirp init --recheck` on a Mac with detected Ollama be loud (multi-line migration plan) or quiet (one-line note plus URL)? Devon's journey assumes loud; revisit if testing shows it's noisy.
- **OQ7: Idle-unload default.** Hardcoded 5 minutes matches Ollama. Should the default be different (longer? shorter?) for chirp's typical "open the laptop, talk for 10 minutes, close" usage pattern? Worth observing in real use before changing.

### Alpha-stage simplifications (added 2026-05-15)

Chirp is in alpha. No backwards-compatibility obligation across versions. This rules out a class of "smooth upgrade" complexity that earlier drafts of this PRD reserved:

- **No graceful drain on daemon shutdown.** Daemon exits promptly on `stop` or version mismatch; in-flight requests fail with a typed error. (See revised NFR-R6 and FR14.)
- **No `models.toml` schema migrations in MVP.** `schema_version` field exists in the file for future use, but the registry reader rejects unknown versions outright — users re-init their registry across breaking schema changes.
- **No backwards-compat shims** in the wire protocol, error codes, or registry. Versioning is single-version-current; old clients hitting newer daemons get `version_mismatch`, same as the reverse.

## References

### Prior epics this PRD builds on or relates to

- `_bmad-output/planning-artifacts/epic-audio-capture/epic.md` — **EPIC-AUDIO-CAPTURE**. Established the "remove a third-party install, replace with bundled first-party helper" pattern (BlackHole → ScreenCaptureKit + AVAudioEngine via `CaptureAudio.app`). This PRD applies the same pattern to Ollama → MLX via `chirpd`.
- `_bmad-output/planning-artifacts/epic-wireframe-alignment/epic.md` — **EPIC-WF-ALIGN**. Locks the visible CLI surface to seven commands (`record · transcribe · notes · ask · search · init · about`). This PRD adds one visible group (`models`) and one hidden group (`daemon`) without modifying the locked seven.

### Project documentation

- `AGENTS.md` — canonical contributor guide; build commands, style, testing rules.
- `CLAUDE.md` — Claude-specific contributor notes.
- `README.md` — user-facing readme (must be updated to remove Ollama references per FR56).

### External

- HuggingFace `mlx-community` organization — pre-converted MLX models. The primary source of chat and embed models registered via `chirp models add`.
- `mlx-lm` (Apple, MLX framework) — Python inference library that `chirpd` builds on.
- `huggingface_hub` (Hugging Face, Inc.) — model download and cache management.
- Ollama documentation — referenced for prior-art on idle-unload behavior, `keep_alive` semantics, and `OLLAMA_MAX_LOADED_MODELS` (informs the Growth-tier multi-loaded-models feature).
- Apple `launchd` / LaunchAgent documentation — for `chirp daemon enable`/`disable` implementation.

### Future epics flagged by this PRD

- **Menu bar app sharing the daemon.** Captured as a separate future epic in §Product Scope → Vision. Daemon protocol must remain compatible.
- **Apple Foundation Models backend.** Captured as a Vision-tier item. Requires a `LLMBackend` protocol abstraction that the MVP daemon should anticipate but not block on.
