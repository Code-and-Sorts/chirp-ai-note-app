# Chirp AI Note App - Claude Development Rules

## Project Overview

**Chirp** is a CLI tool for recording meetings, transcribing audio to text, and generating AI-powered meeting notes with semantic search capabilities.

### Core Functionality

- **Audio Recording**: High-quality meeting recording with system audio capture
- **Transcription**: Speech-to-text using OpenAI's Whisper models
- **AI Notes**: Generate structured meeting summaries using Ollama LLMs
- **Semantic Search**: Query meeting notes with ChromaDB vector search
- **Interactive Chat**: Ask questions about meeting content

### Package Information

- **PyPI Name**: `chirp-notes-ai`
- **CLI Command**: `chirp`
- **Python Version**: 3.13+ (minimum 3.11)
- **Build System**: Hatchling
- **Dependency Manager**: uv

## Architecture & Dependencies

### Core Components

- **`chirp/`**: Main CLI module with Typer commands
- **`config/`**: Settings management with Pydantic and platform-specific paths
- **`recorder/`**: Audio recording with PyAudio and device management
- **`transcriber/`**: Whisper-based transcription with batch processing
- **`notes/`**: AI note generation using Ollama
- **`notes_chat/`**: Semantic search and interactive chat features
- **`utils/`**: Shared utilities for file operations and time handling

### External Dependencies

- **Audio**: PyAudio, PortAudio (system), BlackHole (macOS)
- **AI Models**: Ollama (llama3.1:8b, nomic-embed-text)
- **Database**: ChromaDB for vector search
- **CLI**: Typer with Rich for beautiful terminal output

## Development Guidelines

### Code Quality Standards

Follow principles from **"Clean Code" by Robert C. Martin**:

#### Naming

- Use **descriptive, intention-revealing names**
- Avoid mental mapping and abbreviations
- Use searchable names for important concepts
- Class names should be nouns, method names should be verbs

#### Functions

- **Small functions** - Do one thing well
- **Few parameters** - Ideally 0-2, maximum 3
- **No side effects** - Functions should be predictable
- **Single level of abstraction** per function

#### Classes

- **Single Responsibility Principle** - One reason to change
- **Open/Closed Principle** - Open for extension, closed for modification
- Small, cohesive classes with clear purposes

#### Comments

- **Code should be self-documenting**
- Only add comments when code cannot express intent
- Avoid redundant, misleading, or outdated comments
- Use docstrings for public APIs

#### Error Handling

- Use exceptions, not error codes
- Create informative error messages
- Don't ignore caught exceptions
- Use custom exception classes for domain-specific errors

### Project-Specific Patterns

#### CLI Command Organization

```python
# Use rich help panels for command grouping
@app.command(rich_help_panel=RECORDING_PANEL)
def record(...):
    """Start recording a meeting"""
```

#### Configuration Management

- Use **platform-specific paths** (`platformdirs`)
- **Auto-create config** on first run with sensible defaults
- **Validate configuration** with Pydantic models

#### Error Handling

- Use **domain-specific exceptions** (AudioDeviceError, RecordingError, etc.)
- Provide **helpful error messages** with suggested fixes
- **Graceful degradation** when possible

#### File Operations

- Use **Path objects** consistently, not strings
- **Atomic operations** where possible
- **Progress indicators** for long-running operations

#### Testing

- **Unit tests** for core logic
- **Integration tests** for workflow verification
- **Mock external dependencies** (Ollama, audio devices)

## Code Style

### Type Hints

- Use **comprehensive type hints** for all public APIs
- Use `Optional[T]` for nullable parameters
- Use `Union` types sparingly, prefer specific types

### Imports

- Group imports: standard library, third-party, local
- Use absolute imports from project root
- Import specific functions/classes, avoid `import *`

### Constants

- Use **ALL_CAPS** for module-level constants
- Group related constants (e.g., help panel names)

### Async/Concurrency

- Use `asyncio` for I/O-bound operations when beneficial
- Prefer explicit over implicit async patterns
- Handle cancellation and cleanup properly

## CI/CD Requirements

### Build Process

- **Linting**: Ruff for code style and quality
- **Testing**: pytest with coverage reporting
- **Building**: uv build for wheel creation
- **Publishing**: Automatic TestPyPI on PRs, manual PyPI releases

### Workflow Structure

- **PR Checks**: Build, test, lint, publish to TestPyPI
- **Production Release**: Manual trigger for PyPI publication
- **Version Management**: Automated dev versions for PRs

### System Dependencies

- **macOS**: PortAudio via Homebrew, cached for CI speed
- **Linux**: PortAudio dev headers, ALSA utilities
- **Windows**: PyAudio wheels include PortAudio

## Security Considerations

- **No secrets in code** - Use environment variables
- **Validate user inputs** - Especially file paths and commands
- **Safe file operations** - Check permissions and paths
- **Audio privacy** - Clear user consent for recording

## Performance Guidelines

- **Lazy loading** for heavy dependencies
- **Streaming processing** for large audio files
- **Caching** for expensive operations (model loading, embeddings)
- **Progress feedback** for operations >2 seconds

## Documentation Standards

- **README**: Focus on quick start and essential information
- **Docstrings**: Follow Google/NumPy style for public APIs
- **Help text**: Clear, concise command descriptions
- **Error messages**: Actionable guidance for users

## Development Workflow

1. **Feature branches** from main
2. **Descriptive commit messages** following conventional commits
3. **PR reviews** required before merging
4. **Automated testing** must pass
5. **Manual testing** of audio features when relevant

## Notes for Claude

- Follow existing **architectural patterns** in the codebase
- Maintain **consistency** with established naming conventions
- **Prefer editing** existing files over creating new ones
- **Test changes** can be verified with the existing test suite
- Consider **platform differences** (macOS, Linux, Windows) for audio features
- Remember the **PyPI package name** is `chirp-notes-ai` but CLI is `chirp`
