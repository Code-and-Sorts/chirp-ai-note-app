# Diarization Roadmap (Local, macOS-first)

This document outlines a pragmatic, phased plan to add speaker detection to Chirp. It keeps everything local, starts simple, and gives a clear upgrade path. No implementation is committed yet—this is a planning guide for when we're ready.

- Goals
  - Tag “You” vs “Others” reliably during meetings
  - Optionally split “Others” into Speaker A/B/C
  - Keep fully local; no cloud calls or gated model requirements by default
  - Minimal impact on the current CLI until we flip it on

- Scope
  - macOS focus to start (bundled Chirp.app: ScreenCaptureKit system audio + mic)
  - Compatible with existing transcription (mlx-whisper)
  - Future-compatible with Linux/Windows if desired

## Phase 0 — Prep and Guardrails

- Config placeholders (no behavior change yet)
  - `diarization.enabled: false`
  - `diarization.backend: "speechbrain" | "pyannote"`
  - `diarization.overlap: false`
  - `diarization.align_words: true`
  - `diarization.mic_label.enabled: true`
- Recorder readiness
  - Dual capture (Mic + System) via the bundled Chirp.app helper
  - Keep sample rate aligned (e.g., 48 kHz)
- Tooling
  - `chirp status` and `chirp devices` surface device hints (Aggregate/Multi-Output)
- Acceptance criteria
  - Config keys exist and are ignored safely when disabled
  - Clear doc pointers; no runtime breakage if diarization is off

## Phase 1 — Mic-based “You” Labeling (No Other-speaker splits)

- Approach
  - Record two synchronized streams: Mic (you) and System (others)
  - Run VAD on Mic to build a "you speaking" time mask
  - Assign transcript words/segments to "You" when they overlap the mask; otherwise "Other"
- Pros: Zero model downloads, fast, robust with headphones
- Limitations: No separation among “Others,” no overlap handling
- Config ideas
  - `diarization.mic_label.threshold: 0.4` (overlap proportion)
  - `diarization.vad: webrtc`
- Acceptance criteria
  - Words you speak are labeled “You” with high precision on headphone setups
  - No regression to existing transcription/notes when disabled

## Phase 2 — SpeechBrain Diarization for “Others”

- Approach
  - VAD on System stream → short speech segments (~1.5–3.0s)
  - ECAPA‑TDNN embeddings (SpeechBrain) for each segment
  - Cluster embeddings (Agglomerative/Spectral, cosine distance) into Speaker A/B/C
  - Map words to speakers by timestamp overlap (±100–200 ms padding)
  - Keep Mic-based “You” override: if Mic is active, label as “You”
- Pros: Fully local, no gated models, Apple Silicon friendly
- Limitations: Not overlap‑aware; best for typical meetings
- Config ideas
  - `diarization.backend: "speechbrain"`
  - `diarization.max_speakers: null` (auto 2–6 with override)
  - `diarization.min_speech_ms: 250`
  - `diarization.max_silence_ms: 400`
  - `diarization.frame_ms: 30`
- Acceptance criteria
  - Two‑speaker meetings label cleanly into “You” and one other speaker
  - Three‑plus speaker meetings are reasonable (some fragmentation acceptable)

## Phase 3 — Overlap Handling

- Approach
  - Basic: If both Mic and System show strong activity, mark words as "Overlap" or choose dominant energy
  - Advanced: Enable overlap‑aware diarization (e.g., pyannote backend) to permit multiple concurrent speakers
  - Word assignment: choose the speaker with highest local activity; if Mic active and dominant, prefer “You”
- Pros: Better attribution during cross‑talk
- Limitations: Heavier models if you enable pyannote (requires HF token)
- Config ideas
  - `diarization.overlap: true`
  - `diarization.energy_ratio_threshold: 2.0`
- Acceptance criteria
  - Cross‑talk segments aren’t misattributed wholesale; predictable tie‑breaking

## Phase 4 — Performance, Reliability, and Fallbacks

- Performance
  - Chunk long audio (2–5 min with overlaps) to bound memory; stitch results
  - CPU acceptable on M‑series; keep offline, non‑realtime
- Reliability
  - Friendly errors if models unavailable; clear instructions in `chirp status/test`
- Fallbacks
  - If `backend: pyannote` is requested but models/HF token absent, fall back to SpeechBrain with a warning
- Acceptance criteria
  - 30–90 minute meetings process reliably with clear messaging and no hard failures

## Phase 5 — Optional Identification and UX

- Identify "Me"
  - Optional enrollment: store a local ECAPA voiceprint; relabel diarized cluster as “You” via cosine similarity
  - Keep others anonymous (Speaker A/B/C)
- CLI/UX
  - Flags: `--diarize`, `--overlap`, `--identify-me`
  - Templates: compact vs detailed speaker labels
- Indexing
  - Include `speaker` and `is_you` metadata for future filters
- Acceptance criteria
  - Easy toggles, predictable output, speaker tags improve note readability

## Risks and Mitigations

- Mic bleed (no headphones): higher false “You” rate → increase thresholds; recommend headphones
- Double‑talk: resolve with overlap heuristics or pyannote backend when needed
- Fragmentation: merge adjacent same‑speaker segments; tune VAD thresholds
- Model weight size/network: default to SpeechBrain; make pyannote optional

## Testing Plan (incremental)

- Unit tests
  - VAD mask generation (Mic/System) with synthetic signals
  - Word‑to‑mask overlap labeling
  - Embedding clustering on toy datasets (2–3 speakers)
- Fixtures
  - Short (2–5 min) two‑speaker and three‑speaker samples
  - Headphones vs. speaker playback scenarios
- CLI smoke
  - `chirp transcribe --diarize` produces labeled utterances; disabled mode unchanged

## Next Steps (when ready)

- Keep diarization disabled by default
- Implement Phase 1 first (mic‑based labeling) for immediate value
- Add Phase 2 (SpeechBrain) behind `backend: speechbrain`
- Consider Phase 3 (overlap) only if needed; pyannote as an optional backend

---

References

- SpeechBrain ECAPA TDNN: <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb>
- WebRTC VAD: <https://webrtc.org/> (Python bindings: `webrtcvad`)
- pyannote (optional backend): <https://github.com/pyannote/pyannote-audio>
