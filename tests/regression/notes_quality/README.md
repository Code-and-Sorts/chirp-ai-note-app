# Notes-quality regression corpus

A held-out corpus of representative recordings paired with the notes the
**current Ollama-backed pipeline** generated for them. It is the "before"
baseline for the Ollama → MLX migration (EPIC-INTEGRATION-CUTOVER).

## Purpose

PRD §Domain-Specific Requirements → Validation Methodology requires a regression
corpus to confirm the migration preserves note quality. After the cutover lands,
story 6.6 replays the new MLX pipeline against the exact same transcripts and
scores the new `notes_after.md` blindly against the locked `notes_before.md`.

**Pass threshold:** ≥80% of recordings must score equal-or-better under the new
pipeline (recorded at the end of story 6.6).

This corpus is captured **before** any migration code is written. Once the Ollama
call sites in `notes/note_generator.py` and `notes_chat/retrieval.py` are removed
(stories 6.2 and 6.3), regenerating the "before" baseline is impossible without
reverting the branch. Story 6.1 therefore blocks every other story in the epic.

## Layout

```
tests/regression/notes_quality/
├── README.md
├── .gitignore              # blocks audio extensions (audio is never committed)
├── bucket_manifest.toml    # per-slug speaker/domain/bucket metadata
└── <slug>/
    ├── transcript.txt      # verbatim Whisper transcript
    └── notes_before.md     # verbatim Ollama-generated notes (with YAML front-matter)
```

Story 6.6 will later append `notes_after.md` to each `<slug>/` and a
`results-<date>.md` summary; this story does not.

## Required bucket distribution

Per PRD §Domain Requirements, the corpus must contain **≥10** recordings:

| Bucket          | Requirement | Definition (by `transcript.txt` word count) |
| --------------- | ----------- | ------------------------------------------- |
| Short           | ≥3          | < 300 words (< ~2 min audio)                |
| Medium          | ≥4          | 300–3000 words (~5–15 min audio)            |
| Long            | ≥3          | > 3000 words (> ~20 min audio)              |

Plus, spread across the above (a single recording can satisfy several):

- **Speakers:** ≥1 single-speaker, ≥1 two-person, ≥1 multi-speaker (3+).
- **Domain:** ≥1 technical (code/engineering) and ≥1 non-technical (general
  meeting/planning) recording.

Speaker count and domain are not inferable from a transcript alone, so they are
declared per slug in `bucket_manifest.toml`. The smoke test
(`tests/regression/test_corpus_inventory.py`) derives the length buckets from
word count and reads the manifest for speaker/domain coverage.

## Capture procedure

Capture **all** recordings with the **same** Ollama model (whatever
`models.llm` resolves to in `~/.chirp/config.toml`) so the baseline is uniform.
For each recording:

1. **Record or designate audio.** Capture fresh with `chirp record`, or reuse an
   existing recording under `~/Documents/chirp/<name>/`. Good sources for filling
   buckets:
   - Reading aloud from a long-form blog post or paper (single-speaker,
     length-tunable — useful for the long bucket).
   - A real or simulated stand-up with a colleague (two-person, short/medium).
   - A meeting with three or more participants (multi-speaker, medium/long).
   - Explaining a code change you shipped (technical, length-tunable).
   - A casual planning or to-do session (non-technical, short/medium).
2. **Transcribe with the current Ollama pipeline.** Run `chirp transcribe` (or
   `chirp transcribe <N>`) against the recording. Confirm
   `~/Documents/chirp/<name>/transcript.txt` and `notes.md` are written, and note
   `llm_model` from `~/Documents/chirp/<name>/meta.toml`.
3. **Copy into the corpus** under a stable, descriptive slug:
   ```sh
   mkdir tests/regression/notes_quality/<slug>/
   cp ~/Documents/chirp/<name>/transcript.txt tests/regression/notes_quality/<slug>/transcript.txt
   cp ~/Documents/chirp/<name>/notes.md        tests/regression/notes_quality/<slug>/notes_before.md
   ```
   Do **not** copy `audio.wav` or `meta.toml`.
4. **Add a manifest entry** in `bucket_manifest.toml` (see its schema), filling
   `speakers`, `domain`, `word_count_bucket`, `recorded_on`, and
   `notes_before_model` (the `llm_model` value from step 2).

Slugs are **stable** — story 6.6's results file references them. Use descriptive
lowercase-hyphenated names, e.g. `team-standup-short`, `architecture-deep-dive-long`.

If `chirp transcribe` fails on a recording (Ollama down, model missing),
recapture it — never commit a partial subdirectory. If two manifest entries
disagree on `notes_before_model`, recapture the divergent ones with the canonical
model before merging.

## Blinded comparison (story 6.6)

The "after" scoring procedure is owned by **story 6.6**
(`_bmad-output/planning-artifacts/epic-integration-cutover/stories/6.6-regression-comparison-run.md`).
In short: 6.6 generates `notes_after.md` for each slug with the MLX pipeline
(via the committed `../generate_notes_after.py`) and compares them against the
locked `notes_before.md` baselines on structure, faithfulness, completeness,
and brevity. This file does not duplicate that procedure.

**Most recent regression run:** [`results-2026-06-12.md`](results-2026-06-12.md)
— MLX `qwen2.5-7b-instruct-4bit` vs Ollama `llama3.1:8b`, **12/12 equal-or-better
(PASS, ≥80% threshold)**. Note that run recorded a maintainer qualitative
sign-off rather than the full blinded per-pair protocol (see the results file's
methodology section).

## Why audio is gitignored

Audio is **never committed** (`.gitignore` blocks `*.wav`, `*.m4a`, `*.mp3`,
`*.flac`, `*.aiff`, `*.ogg`):

- **Size** — ~10 MB per minute for 16 kHz PCM; multiple GB for a 10-recording
  corpus including long ones.
- **Sensitivity** — some captures contain real meeting or personal content.
- **Not needed for replay** — story 6.6 compares notes generated from the
  committed `transcript.txt`; the audio is not part of the comparison.

The block lives in this directory's `.gitignore`, not the repo-root `.gitignore`,
so other regression corpora can opt in or out independently.

## Corpus inventory

Each slug's topic descriptor — for recapturing a substitute if the audio is ever
needed again. Speaker / domain / length buckets are authoritative in
`bucket_manifest.toml`.

| Slug | Topic descriptor |
| ---- | ---------------- |
| `mouseless-overlay-short` | Demo of the "mouseless" keyboard-driven cursor overlay for macOS |
| `tensor-ops-musing-short` | Spoken musing on tensor ops (keepdim/axis/reshape/permute) |
| `podcast-interview-intro-short` | Talk-show interview intro (two hosts riffing) |
| `podcast-woodchucks-chat-short` | Casual chat about a book project ("Little Woodchucks") |
| `neovim-intro-medium` | Explainer on Neovim's history and design |
| `opengrid-tile-removal-medium` | Maker how-to: removing tiles from the OpenGrid 3D-print system |
| `usb-10gbe-review-medium` | Hardware review of a USB 10GbE network adapter |
| `copilot-statusline-medium` | Tutorial: customizing the GitHub Copilot CLI status line with Oh My Posh |
| `llm-agents-tools-medium` | Explainer on LLM agents, tools/skills, and MCP |
| `parliament-recession-debate-long` | News segment on a parliamentary recession debate (multi-speaker) |
| `karpathy-interview-long` | Long-form interview with an AI researcher (host(s) + guest) |
| `agentic-coding-talk-long` | Conference talk on building software with coding agents |

## How to recapture missing audio

The committed transcript is enough for story 6.6's comparison, so audio normally
never needs recapturing. If a developer needs to re-run a recording end-to-end
(e.g. to test a Whisper-side change), reproduce a substitute matching the slug's
topic descriptor in the inventory above, then re-run the capture procedure.
Substitute recordings need only match the slug's bucket (length / speakers /
domain); exact wording is not required.

## Known baseline characteristics

These `notes_before.md` files are the **genuine, unmodified** output of the
current Ollama pipeline (`llama3.1:8b`). **All three long entries are degraded**,
because the pipeline sends the whole transcript without setting `num_ctx`, so any
transcript over ~4k tokens overflows the default context window:

- `karpathy-interview-long` (5,906 w) and `agentic-coding-talk-long` (6,153 w) —
  **XML-tag bleed**: raw `<transcript>` markup in the summary and an "Unable to
  parse structured notes" fallback.
- `parliament-recession-debate-long` (4,097 w, just over the window) —
  **structurally valid but empty**: "No summary available" and every section
  "None".

The degraded notes are committed deliberately as the honest "before" baseline.
**Story 6.6's blinded scoring should account for all three** when comparing.
Root cause is logged in `_bmad-output/implementation-artifacts/deferred-work.md`.
