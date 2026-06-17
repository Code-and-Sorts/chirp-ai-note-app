# Chirp

<!-- markdownlint-disable MD033 -->
<p align="center">
   <img src=".docs/imgs/chirp-logo.png" alt="Chirp Logo" height="300" />
</p>
<!-- markdownlint-enable MD033 -->

Chirp is a local-first CLI for recording meetings, transcribing audio, generating notes, and searching past conversations from your terminal.

## Features

- Record audio into a new note workspace
- Stream live transcription in a Rich dashboard while recording
- Transcribe recordings with mlx-whisper (Metal-accelerated on Apple Silicon)
- Generate structured notes with a local MLX model (via the bundled chirpd daemon)
- Browse, edit, and delete saved notes from the terminal
- Ask questions or run keyword search across your note history

## Prerequisites

Chirp currently targets **macOS 13.0 (Ventura) or later** for audio capture. The bundled `CaptureAudio.app` helper uses ScreenCaptureKit's audio-only mode, which requires macOS 13+.

- macOS 13.0+ on **Apple Silicon** (M1/M2/M3/M4 or newer — chirpd runs models on MLX)
- Python 3.11+
- Homebrew if you want `chirp init` to install missing macOS dependencies for you

Note generation and retrieval run on-device through the bundled **chirpd** daemon
(MLX) — no separate model server to install. You register a model once with
`chirp models add` (see Setup below).

## Install

```bash
pip install chirp-notes-ai
```

## Quick start

1. Run the guided setup:

   ```bash
   chirp init
   ```

2. Record a meeting:

   ```bash
   chirp record --title "Team Standup" --live-transcribe
   ```

3. Transcribe audio and generate notes:

   ```bash
   chirp transcribe
   ```

4. Browse or edit saved notes:

   ```bash
   chirp notes
   chirp notes view 1
   chirp notes edit 1
   ```

5. Search or chat across your history:

   ```bash
   chirp search "timeline" --since 14d
   chirp ask -q "What action items did we capture?"
   ```

## Command overview

| Command | What it does |
| --- | --- |
| `chirp record` | Capture audio to a new note, optionally with live transcription |
| `chirp transcribe [N]` | Process pending recordings into transcripts and notes |
| `chirp notes` | List saved notes; `view`, `edit`, and `delete` are subcommands |
| `chirp ask` | Ask questions about your meetings, or open interactive chat |
| `chirp search` | Run keyword or regex search across transcripts and notes |
| `chirp init` | Guided setup, daemon readiness, and model recommendation |
| `chirp about` | Show the animated bird and version info |

## Common workflows

### Recording

```bash
# Timed recording
chirp record --duration 30 --title "Customer Interview"

# Auto-stop after a timeframe
chirp record --title "Sprint Planning" --timeframe 45m

# Add tags at capture time
chirp record --title "Roadmap Review" --tag roadmap --tag planning
```

### Transcription and notes

```bash
# Process all pending notes
chirp transcribe

# Process only the oldest 5 pending notes
chirp transcribe 5

# Rebuild notes from existing transcripts
chirp transcribe --regen

# Override the Whisper model for one run
chirp transcribe --model medium
```

### Notes, search, and chat

```bash
# Filter note list by tags
chirp notes --tag roadmap,planning

# Open interactive chat
chirp ask

# Ask with a time filter
chirp ask -q "What changed this week?" --when "last week"

# Regex or JSON search output
chirp search "action item" --since 30d
chirp search "owner: .*" --regex --json
```

## Setup details

`chirp init` is the recommended setup path. It checks for Apple Silicon, verifies Homebrew and `ffmpeg`, confirms the bundled chirpd daemon is reachable, reports whether a default chat model is registered, and checks the screen-recording permission — then helps install anything missing and offers to start chirpd at login. The bundled `CaptureAudio.app` records system audio and microphone directly via ScreenCaptureKit; no virtual audio driver is required.

If you prefer to set things up manually on macOS:

1. Install the only system dependency (chirp itself is pip-installed):

   ```bash
   brew install ffmpeg
   ```

2. Register a chat model and an embedding model (downloaded to the local HF cache, served on-device by chirpd):

   ```bash
   # Chat — balanced quality and speed (~2 GB)
   chirp models add mlx-community/gemma-4-4b-it-4bit
   # ...or a smaller-footprint variant for tighter RAM:
   chirp models add mlx-community/gemma-4-e2b-it-8bit

   # Embeddings (for search / retrieval)
   chirp models add mlx-community/bge-small-en-v1.5 --role embed
   ```

3. Re-check your environment:

   ```bash
   chirp init --recheck
   ```

## Configuration and storage

- Config file: `~/.chirp/config.toml`
- Default notes root: `~/Documents/chirp`

Each note is stored in its own directory:

```text
~/Documents/chirp/<note-slug>/
├── audio.wav
├── transcript.txt
├── notes.md
└── meta.toml
```

For advanced maintenance, Chirp also exposes hidden commands such as:

```bash
chirp config --list
chirp devices
chirp index --force
```

## Troubleshooting

**Recording fails immediately**
Grant Chirp access to **Screen Recording** and **Microphone** in `System Settings → Privacy & Security`, then retry. The first run will prompt for both; later denials require toggling the entries manually.

**Transcription or notes generation fails**
Check the daemon with `chirp daemon status` and your registered models with `chirp models list`. `chirp init --recheck` will show what is missing; `chirp models add <hf-repo>` registers a model if none is set.

**No notes found**
Run `chirp transcribe` first, or check `chirp config --list` to confirm the notes root you are using.

## Development

Contributor docs live in `AGENTS.md` and `.docs/DEVELOPMENT.md`.

<!-- markdownlint-disable MD033 -->
<p align="center">
   <img src=".docs/imgs/chirp-footer.svg" alt="Chirp Footer" />
</p>
<!-- markdownlint-enable MD033 -->
