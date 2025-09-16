# Chirp - AI Meeting Notes CLI

🐦 A powerful CLI tool that records meetings, transcribes audio to text, and generates AI-powered meeting notes with semantic search capabilities.

## Quick Start

Install Chirp with pip:

```bash
pip install chirp-notes-ai
```

## Manual Dependencies

Before using Chirp, you'll need to install these system dependencies:

### macOS
```bash
# Install PortAudio for audio recording
brew install portaudio

# Install BlackHole for system audio capture
# Download from: https://existential.audio/blackhole/
```

### Linux (Ubuntu/Debian)
```bash
# Install PortAudio development headers
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev

# Install ALSA utilities for audio
sudo apt-get install alsa-utils
```

### Windows
```bash
# PortAudio is included with PyAudio wheels on Windows
# No additional system dependencies required
```

## AI Model Setup

Chirp requires **Ollama** for AI processing:

1. **Install Ollama**: Download from [https://ollama.ai](https://ollama.ai)
2. **Start Ollama**: Run `ollama serve` in a terminal
3. **Download models**:
   ```bash
   # For meeting note generation
   ollama pull llama3.1:8b

   # For semantic search
   ollama pull nomic-embed-text
   ```

## Usage

```bash
# Record a meeting
chirp record --duration 30 --title "Team Standup"

# Transcribe audio files
chirp transcribe

# Generate AI meeting notes
chirp generate-notes

# Complete workflow (transcribe + generate notes)
chirp transcribe-and-notes

# Ask questions about your notes
chirp ask "What were the action items from today's meeting?"

# Check system status
chirp stats

# Test installation
chirp test
```

## Configuration

On first run, Chirp creates a config file at:
- **macOS/Linux**: `~/.config/chirp/config.yaml`
- **Windows**: `%APPDATA%/chirp/config.yaml`

Edit this file to customize directories, models, and settings.

## Features

- 🎙️ **High-quality audio recording** with system audio capture
- 📝 **Accurate transcription** using OpenAI's Whisper models
- 🧠 **AI-powered meeting notes** with structured summaries
- 🔍 **Semantic search** across all your meeting notes
- 💬 **Interactive chat** to ask questions about your meetings
- ⚙️ **Flexible configuration** with sensible defaults

## Requirements

- Python 3.11+
- Ollama (for AI features)
- System audio setup (PortAudio, BlackHole on macOS)

## Support

- **Documentation**: [GitHub Repository](https://github.com/Code-and-Sorts/chirp-ai-note-app)
- **Issues**: [Report bugs or request features](https://github.com/Code-and-Sorts/chirp-ai-note-app/issues)
