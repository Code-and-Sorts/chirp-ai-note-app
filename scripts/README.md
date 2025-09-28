# 🐦 Chirp - AI Meeting Notes CLI

<!-- markdownlint-disable MD033 -->
<p align="center">
   <img src=".docs/imgs/chirp-logo.png" alt="Chirp Logo" height="300" />
</p>
<!-- markdownlint-enable MD033 -->

🐦 A powerful CLI tool that records meetings, transcribes audio to text, and generates AI-powered meeting notes with semantic search capabilities.

## Installation

### 1. Install System Dependencies

**macOS:**

```bash
# Install required system libraries
brew install portaudio ollama

# Install BlackHole for system audio capture
# Download from: https://existential.audio/blackhole/
```

**Linux (Ubuntu/Debian):**

```bash
# Install required system libraries
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev alsa-utils

# Install Ollama for AI processing
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**

```bash
# PortAudio is included with PyAudio wheels on Windows
# Install Ollama for AI processing from: https://ollama.ai/download
```

### 2. Install Chirp

```bash
pip install chirp-notes-ai
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
