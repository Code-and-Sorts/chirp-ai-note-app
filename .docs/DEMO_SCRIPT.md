# 🎬 Chirp Demo Script for README GIFs

Step-by-step demos for recording the GIFs in `README.md`. Commands here track
the current CLI surface (`record`, `transcribe`, `notes`, `ask`, `search`,
`init`, `about`, plus the hidden `config`, `devices`, `index`, `daemon`,
`models`). Run `chirp COMMAND --help` if anything looks off.

Notes live under `~/Documents/chirp/<note-slug>/`, each a directory containing
`audio.wav`, `transcript.txt`, `notes.md`, and `meta.toml`.

---

## 🎥 Demo 1: Recording a Meeting (30-45 seconds)

**Goal**: Show how easy it is to start and stop a recording.

### Setup
```bash
clear
chirp notes        # show the current (empty) note list
```

### Recording
```bash
# Auto-stop after 30 seconds. --timeframe takes 30s / 5m / 1h.
# (--duration is in MINUTES, so use --timeframe for a short demo.)
chirp record --timeframe 30s --title "Product Demo Discussion"

# On screen while recording:
# - elapsed timer
# - live audio level meter
# - press Ctrl+C to stop early

# Variant worth showing: live transcription as you speak
chirp record --timeframe 30s --title "Product Demo Discussion" --live-transcribe
```

### Verification
```bash
# The new note now appears in the list (still "untranscribed")
chirp notes
```

**Key Highlights**:
- One clean command, no flags required
- Real-time timer and audio levels
- Optional live transcription with `--live-transcribe`
- Graceful Ctrl+C stop

---

## 🎥 Demo 2: Transcribe → Notes (45-60 seconds)

**Goal**: Show the end-to-end pipeline from audio → transcript → AI notes.

### Setup
```bash
clear
chirp notes        # pending note(s) waiting to be transcribed
```

### Transcribe
```bash
# Transcribe pending recordings and generate notes (and index them).
# Optionally cap how many: `chirp transcribe 1` does just the oldest.
chirp transcribe

# On screen:
# 1. transcription progress (mlx-whisper, on-device)
# 2. note generation via the local chirpd model
# 3. search index update
```

### View Results
```bash
# Browse, then open the freshly generated note read-only
chirp notes
chirp notes view 1
```

**Key Highlights**:
- One command: transcript + summary + index
- Fully on-device (mlx-whisper + chirpd)
- `--regen` rebuilds notes from existing transcripts after switching models

---

## 🎥 Demo 3: Interactive Chat (60-90 seconds)

**Goal**: Showcase the interactive chat interface — the star feature.

### Setup
```bash
clear
```

### Interactive Session
```bash
# No question = interactive chat
chirp ask

# A welcome panel appears; type questions at the prompt.
# Ctrl+C handling is shown in the toolbar (press again to exit).
```

### Ask Questions
```text
> what were the key decisions from recent meetings?

# Watch for:
# - "searching your notes..." then "thinking..." spinners
# - streamed answer from chirp 🐣
# - source notes listed under the answer

> who owns the follow-ups from those decisions?
# Conversational follow-up reuses the retrieved context

> press Ctrl+C once  → toolbar shows "press Ctrl+C again to exit"
> press Ctrl+C again → "Goodbye!"
```

**Key Highlights**:
- Clean bordered interface
- Live retrieval + streaming responses
- Source attribution under every answer
- Smart two-step Ctrl+C exit

---

## 🎥 Demo 4: One-Shot Questions (30-45 seconds)

**Goal**: Quick answers without entering interactive mode.

### Setup
```bash
clear
```

### Quick Questions
```bash
# A positional question runs once and exits
chirp ask "what action items were assigned this week?"

# Shows the retrieval step, a streamed answer, and source notes
```

### Advanced Queries
```bash
# Time-range filter
chirp ask -q "budget discussions" --when "last week"

# Inspect the prompt/retrieval without calling the model
chirp ask -q "project timeline" --dry-run

# Machine-readable output (one-shot only)
chirp ask -q "open risks" --json
```

**Key Highlights**:
- Instant answers, no interactive mode
- Time-range filtering with `--when`
- `--dry-run` to debug retrieval; `--json` for scripting

---

## 🎥 Demo 5: Search & Browse (30-45 seconds)

**Goal**: Show fast keyword search and note browsing.

### Keyword Search
```bash
clear

# Lexical (BM25) search across transcripts and notes — no model needed
chirp search "timeline" --since 14d

# Regex search
chirp search "owner: .*" --regex
```

### Browse Notes
```bash
# Filter the note list by tag
chirp notes --tag roadmap,planning

# Open one
chirp notes view 1
```

**Key Highlights**:
- Keyword search works out of the box (no embedding model)
- `--since` time filtering and `--regex` power search
- Tag-filtered browsing; `view` / `edit` / `delete` subcommands

---

## 🎥 Demo 6: Setup, Config & Diagnostics (30-45 seconds)

**Goal**: Show first-run setup, configuration, and health checks.

### Verify the environment
```bash
clear

# Re-run only the verify phase of init (Apple Silicon, daemon, model, perms)
chirp init --recheck

# Daemon health: PID, uptime, version, loaded models
chirp daemon status

# Registered models and which is the default
chirp models list

# Audio input devices
chirp devices
```

### Configuration
```bash
# Show current configuration
chirp config --list

# Point the notes root somewhere else
chirp config --notes-root ~/chirp-demo

# Opt into semantic (vector) search — registers an embed model and
# rebuilds the index so `chirp ask` blends keyword + meaning
chirp config --semantic
```

**Key Highlights**:
- `chirp init --recheck` as a one-shot health check
- Clear daemon/model/device diagnostics
- Lexical by default; semantic search is one opt-in command away

---

## 📋 Recording Tips

### Terminal Setup
1. **Font Size**: Use large, readable font (18-20pt)
2. **Theme**: Use high-contrast theme (dark background recommended)
3. **Window Size**: 100-120 columns wide, 30-40 rows
4. **Clear Screen**: Always start with `clear`

### Recording Settings
- **Resolution**: 1920x1080 or higher
- **FPS**: 30fps minimum
- **Format**: MP4 or MOV (for later conversion to GIF)
- **Tool**: Use `asciinema` for terminal recordings or QuickTime/OBS

### GIF Conversion
```bash
# Convert MP4 to GIF using ffmpeg
ffmpeg -i demo.mp4 -vf "fps=15,scale=1000:-1:flags=lanczos" -c:v gif demo.gif

# Or use gifski for better quality
ffmpeg -i demo.mp4 -vf "fps=30,scale=1000:-1:flags=lanczos" -pix_fmt rgb24 frame%04d.png
gifski -o demo.gif frame*.png --fps 15
rm frame*.png
```

> Note: `ffmpeg` here is just a local tool for making demo GIFs — Chirp itself
> has no ffmpeg dependency.

### Optimize GIFs
```bash
# Use gifsicle to optimize
gifsicle -O3 --colors 256 demo.gif -o demo-optimized.gif

# Keep under 10MB for GitHub
ls -lh demo-optimized.gif
```

---

## 🎯 Feature Priority for GIFs

### Must-Have (Top Priority)
1. **Interactive Chat** — the star feature; streaming, sources, Ctrl+C handling
2. **Transcribe → Notes** — one command: `chirp transcribe`
3. **Recording** — show how simple start/stop is

### Nice-to-Have
4. **One-Shot Questions** — quick queries without interactive mode
5. **Search & Browse** — keyword search and note list

### Optional
6. **Setup, Config & Diagnostics** — if you have extra time

---

## 📝 Caption Ideas for README

### Demo 1: Recording
> "Start recording in seconds with real-time feedback"

### Demo 2: Transcribe → Notes
> "One command turns audio into a transcript, AI notes, and a search index"

### Demo 3: Interactive Chat
> "Interactive chat with streaming answers and source attribution"

### Demo 4: Quick Questions
> "Get instant answers with time-filtered search"

### Demo 5: Search & Browse
> "Fast keyword search across every meeting — no model required"

---

## 🎨 README Structure Suggestion

```markdown
## ✨ See It In Action

### 🎙️ Record Meetings
![Recording Demo](docs/gifs/demo-recording.gif)

### ⚡ Transcribe → Notes
![Transcribe Demo](docs/gifs/demo-transcribe.gif)

### 💬 Interactive Chat (Our Favorite!)
![Chat Demo](docs/gifs/demo-chat.gif)

### 🔍 Quick Questions
![Query Demo](docs/gifs/demo-query.gif)
```

---

## 🔧 Preparation Checklist

Before recording:

- [ ] Point the notes root at a throwaway dir so demos never touch real notes
      (`chirp config --notes-root ~/chirp-demo`)
- [ ] Ensure the daemon is healthy (`chirp daemon status`)
- [ ] Verify a chat model is registered (`chirp models list`)
- [ ] Test audio input is detected (`chirp devices`)
- [ ] Build the search index if needed (`chirp index`)
- [ ] Increase terminal font size
- [ ] Set terminal dimensions (100x30)
- [ ] Clear command history
- [ ] Close unnecessary terminal tabs
- [ ] Practice each demo 2-3 times

---

## 🚀 Quick Reset Script

Reset a **demo** notes root between takes. Set it first so this never deletes
real notes:

```bash
chirp config --notes-root ~/chirp-demo
```

```bash
#!/bin/bash
# reset-demo.sh — only ever run against a throwaway demo notes root.
set -euo pipefail

DEMO_ROOT="${HOME}/chirp-demo"

clear
rm -rf "${DEMO_ROOT:?}/"*   # the :? guard refuses to run on an empty path

chirp notes                 # verify the clean state
echo "✅ Demo environment reset (${DEMO_ROOT})"
```

> **⚠️ WARNING**: Chirp's default notes root is `~/Documents/chirp`. Never point
> this script there — keep demos on a separate `--notes-root` so a reset can't
> touch real recordings.
