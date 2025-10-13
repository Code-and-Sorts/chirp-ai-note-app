# 🐣 Chirp - AI Meeting Notes CLI

<!-- markdownlint-disable MD033 -->
<p align="center">
   <img src=".docs/imgs/chirp-logo.png" alt="Chirp Logo" height="300" />
</p>
<!-- markdownlint-enable MD033 -->

**Chirp** is a comprehensive CLI application that captures system audio, transcribes meetings using AI, and generates structured meeting notes automatically. Perfect for remote meetings, interviews, lectures, or any audio content you want to transcribe and summarize.

## ✨ Features

- **🎙️ System Audio Recording**: Capture high-quality audio from meetings, calls, or any system audio
- **📺 Live Transcription**: Real-time transcription with scrollable dashboard while recording
- **🤖 AI Transcription**: Powered by faster-whisper with Apple Silicon optimization for accurate speech-to-text
- **📋 Smart Note Generation**: Uses Ollama + Llama 3.1 to generate structured meeting notes with key points, decisions, and action items
- **📁 Batch Processing**: Process multiple audio files with progress indicators
- **🗜️ Efficient Storage**: Compressed JSON storage for transcriptions to save space
- **📝 Template System**: Customizable markdown templates for consistent note formatting
- **🔍 Smart Search**: Query your meeting history with natural language using hybrid search (semantic + keyword)
- **💬 Interactive Chat**: Beautiful chat interface to have conversations about your meeting notes
- **⚡ Modern CLI**: Built with Typer and Rich for a beautiful command-line experience

## 🚀 Quick Start

### Installation (End Users)

**Coming Soon**: Install via Homebrew (recommended for general users):

```bash
# Future release - not yet available
brew tap your-org/chirp
brew install chirp

# Setup audio driver and AI models
chirp setup
```

### Setup Audio Input Device

Chirp uses your system's default input device for recording. You have several options:

**Option 1: Use an Aggregate Device (Recommended for system audio + microphone)**
- Open Audio MIDI Setup (Applications/Utilities)
- Create an Aggregate Device combining:
  - Your microphone (for your voice)
  - BlackHole (for system audio)
- Set this aggregate device as your default input in System Settings → Sound

**Option 2: Use BlackHole Only (System audio only)**
```bash
# Install BlackHole
brew install blackhole-2ch
```
- Set BlackHole as your default input device
- Create a Multi-Output Device to hear audio while recording

**Option 3: Use Built-in Microphone**
- Just use your Mac's built-in microphone (no setup needed)
- Set it as default input in System Settings → Sound

### Setup Ollama and AI Models

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve

# Install required models
ollama pull llama3.1:8b        # LLM for note generation and Q&A
ollama pull nomic-embed-text   # Embedding model for search indexing
```

**Required Models**:

- `llama3.1:8b`: Main language model for generating notes and answering questions
- `nomic-embed-text`: Embedding model for semantic search functionality

### Verify setup

```bash
chirp status
```

### Basic Usage

1. **Record a meeting**:

   ```bash
   chirp record --duration 60 --title "Team Standup"
   ```

2. **Transcribe audio files**:

   ```bash
   chirp transcribe
   ```

3. **Generate meeting notes**:

   ```bash
   chirp notes
   ```

4. **Process everything at once**:

   ```bash
   chirp process
   ```

5. **Build search index and query your meetings**:

   ```bash
   # Build search index for your notes
   chirp notes index

   # Ask specific questions
   chirp notes ask --question "what was decided about the project timeline?"

   # Start interactive chat mode
   chirp notes ask
   ```

6. **Check status**:

   ```bash
   chirp status
   ```

## 📖 User Guide

### Recording Meetings

```bash
# Record with duration and title
chirp record --duration 30 --title "Client Meeting"

# Record indefinitely (stop with Ctrl+C)
chirp record

# Record with live transcription (real-time transcription while recording)
chirp record --live-transcribe --title "Team Meeting"

# Record with custom settings
chirp record -d 45 -t "Project Planning"
```

#### Live Transcription Mode

When using `--live-transcribe`, you get a real-time dashboard showing:

- **Live transcript**: See the transcription as it happens
- **Audio levels**: Visual feedback of recording volume
- **Speaker detection**: See when speech is detected
- **Scrollable view**: Use arrow keys to scroll through the transcript while recording continues

**Keyboard Controls During Live Transcription**:
- `↑/↓` or `PgUp/PgDn`: Scroll through the transcript
- `Space` or `Enter`: Jump back to latest transcription (resume auto-scroll)
- `Ctrl+C`: Stop recording and save

**Note**: Live transcription uses the `large-v3-turbo` Whisper model and provides near real-time results. The final high-quality transcript can be generated afterward with `chirp transcribe`.

### Processing Audio

```bash
# Transcribe all new audio files
chirp transcribe

# Force re-transcribe existing files
chirp transcribe --force

# Process specific directory
chirp transcribe --input /path/to/audio/files
```

### Searching and Querying Meeting History

#### Building the Search Index

Before you can search your meetings, you need to build a search index:

```bash
# Build search index for the first time
chirp notes index

# Force rebuild the index (if you have new notes or want to refresh)
chirp notes index --force
```

**Note**: The index is automatically built/updated when you generate notes, but you can build it manually at any time.

#### Interactive Chat Mode

The interactive chat provides a beautiful, bordered interface for conversing with your meeting notes:

```bash
# Start interactive chat (no question parameter)
chirp notes ask
```

**Features**:

- Beautiful bordered chat interface with animated loading spinners
- Conversation history displayed in organized panels
- Smooth Ctrl+C behavior: clear while typing, double Ctrl+C to exit
- Real-time search with visual feedback
- Professional CLI experience similar to Claude Code

#### One-time Questions

Ask specific questions directly from the command line:

```bash
# Ask specific questions
chirp notes ask --question "what were the key decisions from yesterday?"
chirp notes ask -q "who is responsible for the budget review?"

# Search within specific time ranges
chirp notes ask -q "what was discussed about hiring?" --when "last week"
chirp notes ask -q "any mentions of deadlines?" --when "2024-01-15"
chirp notes ask -q "project updates?" --when "2024-01-01:2024-01-31"

# Show search context without generating an answer (for debugging)
chirp notes ask -q "budget planning" --dry-run

# Hide source attribution
chirp notes ask -q "team updates" --no-sources
```

#### Search Features

- **Hybrid Search**: Combines semantic search (using embeddings) with keyword search (BM25)
- **Time Range Filtering**: Search within specific dates or relative time periods
- **Natural Language**: Ask questions in plain English
- **Source Attribution**: See which meeting notes contain your answers
- **Caching**: Repeated questions use cached results for speed
- **Smart Suggestions**: Get helpful suggestions when no results are found

### Managing Configuration

View current settings:

```bash
chirp config --list
```

Update directories:

```bash
chirp config --audio-dir ./my-recordings
chirp config --notes-dir ./my-notes
```

### Checking System Status

```bash
# Show audio devices and BlackHole status
chirp devices

# Test all dependencies
chirp test

# View processing statistics
chirp status
```

## 🧰 Troubleshooting

### Interactive chat not starting

1. Verify the index exists: `chirp notes index`
2. Check that you have notes in the `notes-out/` directory
3. Ensure all dependencies are installed: `chirp test`
4. Check Ollama is accessible: `curl http://localhost:11434/api/version`

### Audio Recording Issues

**No audio being recorded**:

1. Check your default input device in System Settings → Sound
2. Verify the device is working: `chirp devices`
3. Check audio permissions in System Preferences > Security & Privacy > Microphone

**Recording fails to start**:

1. Check audio permissions in System Preferences > Security & Privacy > Microphone
2. Verify your default input device is set correctly
3. List available devices: `chirp devices`
4. Try a different input device

**Want to record system audio**:

1. Install BlackHole: `brew install blackhole-2ch`
2. Create an Aggregate Device in Audio MIDI Setup
3. Set the aggregate device as your default input

### AI/LLM Issues

**Transcription fails**:

1. Check that faster-whisper is properly installed
2. Verify audio file format is supported
3. Try with a smaller audio file first

**Note generation fails**:

1. Ensure Ollama is running: `ollama serve`
2. Verify the model is installed: `ollama pull llama3.1:8b`
3. Check Ollama logs for errors

### Configuration Issues

**Wrong directories or missing files**:

1. Check current configuration: `chirp config --list`
2. Verify directory structure: `chirp status`
3. Reset to defaults by removing `config/config.yaml`

## 🚧 Future Enhancements

### Near-term

- **Speaker Diarization**: Identify and label different speakers in meetings
- **Streaming Transcription**: Segment-by-segment streaming display

### Long-term

- **Calendar Integration**: Auto-trigger recording from macOS Calendar
- **Multiple Export Formats**: PDF, DOCX, Notion, etc.
- **Meeting Analytics**: Insights and patterns from meeting data
- **OS Support**: Support Windows and Linux

## 🤝 Support

- Check `chirp --help` for command-line help
- Run `chirp test` to diagnose setup issues

## 👩‍💻 For Developers

Developer setup, commands, and contribution guidelines have moved to `.docs/DEVELOPMENT.md`.
