# Chirp

<!-- markdownlint-disable MD033 -->
<p align="center">
   <img src=".docs/imgs/chirp-logo.png" alt="Chirp Logo" height="300" />
</p>
<!-- markdownlint-enable MD033 -->

A CLI tool that records meetings, transcribes audio, and generates structured notes — all locally.

Chirp captures system audio via BlackHole, transcribes with faster-whisper, and produces summaries with Ollama. No cloud services, no API keys, everything runs on your machine.

## Features

- Record system audio (calls, meetings, lectures)
- Transcribe with faster-whisper (Apple Silicon optimized)
- Generate structured notes with Ollama + Llama 3.1
- Search and chat with your meeting history
- Batch process multiple recordings

## Prerequisites

**macOS only** (Windows/Linux support planned)

- Python 3.11+
- [BlackHole](https://existential.audio/blackhole/) — virtual audio driver
- [Ollama](https://ollama.com) — local LLM runtime

## Install

```bash
pip install chirp-notes-ai
```

## Setup

### Audio

BlackHole routes system audio into Chirp. Run `chirp setup` for a step-by-step guide, or set it up manually:

1. Install BlackHole: `brew install blackhole-2ch`
2. Open **Audio MIDI Setup** → create a Multi-Output Device ("Chirp Output") with your speakers + BlackHole
3. Create an Aggregate Device ("Chirp Input") with your microphone + BlackHole
4. Set system output to Chirp Output, input to Chirp Input

Verify with `chirp devices`.

### Models

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Verify with `chirp test`.

## Usage

```bash
# Record a meeting
chirp record --duration 60 --title "Team Standup"

# Record indefinitely (ESC or Ctrl+C to stop)
chirp record

# Transcribe all new recordings
chirp transcribe

# Generate notes from transcriptions
chirp generate

# Transcribe and generate in one step
chirp transcribe-and-generate
```

### Search & chat

```bash
# Build the search index
chirp index

# Ask a question
chirp ask -q "what was decided about the timeline?"

# Interactive chat mode
chirp ask

# Filter by date
chirp ask -q "hiring updates" --when "last week"
```

### Configuration

```bash
chirp config --list
chirp config --audio-dir ./my-recordings
chirp config --notes-dir ./my-notes
```

### Diagnostics

```bash
chirp devices   # list audio devices
chirp test      # check all dependencies
chirp stats     # view processing stats
```

## Troubleshooting

**BlackHole not detected** — Install it (`brew install blackhole-2ch`), create the Multi-Output Device, and run `chirp devices` to verify.

**Recording fails** — Check microphone permissions in System Settings > Privacy & Security > Microphone. Run `chirp test`.

**Transcription fails** — Make sure the audio file isn't empty or corrupted. Try a shorter recording first.

**Note generation fails** — Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.1:8b`).

**Chat/search not working** — Build the index first: `chirp index`. Check that you have notes in your output directory.

## Roadmap

- Speaker diarization
- Calendar integration (auto-record from macOS Calendar)
- Real-time transcription
- Export to PDF, DOCX, Notion
- Windows and Linux support

## Development

See [.docs/DEVELOPMENT.md](.docs/DEVELOPMENT.md).
