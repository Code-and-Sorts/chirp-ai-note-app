# 🐦 Chirp - Meeting Recorder CLI

**Chirp** is a comprehensive CLI application that captures system audio, transcribes meetings using AI, and generates structured meeting notes automatically. Perfect for remote meetings, interviews, lectures, or any audio content you want to transcribe and summarize.

## ✨ Features

- **🎙️ System Audio Recording**: Capture high-quality audio from meetings, calls, or any system audio using BlackHole
- **🤖 AI Transcription**: Powered by faster-whisper with Apple Silicon optimization for accurate speech-to-text
- **📋 Smart Note Generation**: Uses Ollama + Llama 3.1 to generate structured meeting notes with key points, decisions, and action items
- **📁 Batch Processing**: Process multiple audio files with progress indicators
- **🗜️ Efficient Storage**: Compressed JSON storage for transcriptions to save space
- **📝 Template System**: Customizable markdown templates for consistent note formatting
- **🔍 Smart Search**: Query your meeting history with natural language using hybrid search (semantic + keyword)
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

### Installation (Local Development/Manual)

For developers or users who want to run from source:

1. **Clone and setup everything**:

   ```bash
   git clone <repository-url>
   cd chirp-ai-note-app
   make setup
   ```

   This command automatically:
   - Installs system dependencies (PortAudio)
   - Installs Python dependencies with uv
   - Creates required directories
   - Sets up pre-commit hooks

2. **Install BlackHole audio driver**:
   - Download from [existential.audio/blackhole](https://existential.audio/blackhole/)
   - Follow the installation guide to set up a multi-output device

3. **Setup Ollama**:

   ```bash
   brew install ollama
   ollama serve
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

4. **Verify setup**:

   ```bash
   make check-deps
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

5. **Search your meeting history**:

   ```bash
   notes ask "what was decided about the project timeline?"
   notes ask "who was assigned the API task?" --when "last week"
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

# Record with custom settings
chirp record -d 45 -t "Project Planning"
```

### Processing Audio

```bash
# Transcribe all new audio files
chirp transcribe

# Force re-transcribe existing files
chirp transcribe --force

# Process specific directory
chirp transcribe --input /path/to/audio/files
```

### Searching Meeting History

```bash
# Ask questions about your meetings
notes ask "what were the key decisions from yesterday?"
notes ask "who is responsible for the budget review?"

# Search within specific time ranges
notes ask "what was discussed about hiring?" --when "last week"
notes ask "any mentions of deadlines?" --when "2024-01-15"
notes ask "project updates?" --when "2024-01-01:2024-01-31"

# Build search index manually (optional - auto-builds during note generation)
notes index
notes index --force
```

### Managing Configuration

```bash
# View current settings
chirp config --list

# Update directories
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

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+
- macOS (for BlackHole support)
- Homebrew
- Git

### Setup

1. **Clone and setup**:

   ```bash
   git clone <repository-url>
   cd chirp-ai-note-app
   make dev-install
   ```

   This installs all dependencies, sets up pre-commit hooks, and installs the package in development mode.

2. **Install system dependencies**:

   ```bash
   make install-deps
   ```

3. **Run tests**:

   ```bash
   make test
   make test-coverage
   ```

4. **Code quality checks**:

   ```bash
   make style-check  # Linting, formatting, spell check
   make type-check   # Type checking with mypy
   make check        # All quality checks
   ```

### Development Workflow

```bash
# Complete development workflow
make dev-workflow  # Style, type-check, test, build

# Individual commands
make lint          # Check code with ruff
make format        # Format code with ruff
make spell-check   # Check spelling with codespell
make validate      # Validate imports and compilation
```

### Project Structure

```text
chirp-ai-note-app/
├── chirp/                 # Main CLI application
│   ├── cli.py            # CLI interface with Typer
│   └── exceptions.py     # Custom exceptions
├── config/               # Configuration management
│   └── settings.py      # Pydantic settings with YAML
├── recorder/             # Audio recording modules
│   ├── audio_recorder.py
│   └── device_manager.py
├── transcriber/          # Transcription and processing
│   ├── whisper_transcriber.py
│   ├── batch_processor.py
│   └── compression.py
├── notes/               # Note generation
│   ├── note_generator.py
│   ├── template_engine.py
│   └── daily_aggregator.py
├── notes_chat/          # Search and query functionality
│   ├── index.py
│   ├── retrieval.py
│   ├── prompting.py
│   └── cache.py
├── utils/               # Utilities
│   ├── file_utils.py
│   ├── time_utils.py
│   └── popup_manager.py
├── templates/           # Markdown templates
├── config/             # Configuration files
├── pyproject.toml      # Project configuration
├── Makefile           # Development commands
└── .pre-commit-config.yaml
```

### Available Make Commands

```bash
make help              # Show all commands
make install          # Install production dependencies
make dev-install      # Install development dependencies
make test             # Run tests
make lint             # Run linting
make format           # Format code
make style-check      # Check code style
make build            # Build package
make clean            # Clean build artifacts
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make check` to ensure code quality
5. Submit a pull request

## 🎯 Use Cases

- **Remote Meetings**: Record Zoom, Teams, or any video call audio
- **Interviews**: Transcribe and summarize job interviews or research interviews
- **Lectures**: Convert educational content into structured notes
- **Brainstorming Sessions**: Capture ideas and generate actionable summaries
- **Client Calls**: Create professional meeting summaries automatically
- **Meeting Research**: Quickly find past decisions, action items, or specific discussions

## 🚧 Future Enhancements

### Near-term

- **Speaker Detection**: Identify different speakers in meetings

### Long-term

- **Calendar Integration**: Auto-trigger recording from macOS Calendar
- **Real-time Transcription**: Live transcription during recording
- **Multiple Export Formats**: PDF, DOCX, Notion, etc.
- **Meeting Analytics**: Insights and patterns from meeting data

## 🤝 Support

- Check `chirp --help` for command-line help
- Run `chirp test` to diagnose setup issues
- See the Makefile for all available development commands
