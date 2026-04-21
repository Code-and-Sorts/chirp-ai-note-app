.PHONY: help install dev-install test lint format type-check clean run build setup check-deps

# Default help command
help: ## Show this help message
	@echo "🐣 Chirp Development Makefile"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation commands
install-deps: ## Install system dependencies (macOS)
	@echo "📦 Installing system dependencies..."
	@if command -v brew >/dev/null 2>&1; then \
		echo "Installing PortAudio for PyAudio..."; \
		brew install portaudio; \
	else \
		echo "❌ Homebrew not found. Please install manually:"; \
		echo "  - PortAudio: https://portaudio.com/"; \
		echo "  - Or install Homebrew: https://brew.sh/"; \
		exit 1; \
	fi

install: install-deps ## Install production dependencies with uv
	uv sync

dev-install: install-deps ## Install development dependencies
	uv sync --all-extras
	uv pip install -e .
	uv run pre-commit install

install-venv: ## Install chirp to current virtual environment in editable mode
	uv pip install -e .

# Development commands
run: ## Run Chirp in development mode
	uv run python -m chirp.cli

# CLI command shortcuts
record: ## Start recording a meeting (with optional DURATION and TITLE)
	uv run chirp record $(if $(DURATION),--duration $(DURATION)) $(if $(TITLE),--title "$(TITLE)")

transcribe: ## Transcribe audio files
	uv run chirp transcribe $(if $(FORCE),--force)

generate: ## Generate meeting notes from transcripts
	uv run chirp generate $(if $(FORCE),--force)

process: ## Process audio files (transcribe + generate notes)
	uv run chirp transcribe-and-generate $(if $(FORCE),--force)

notes: ## Browse your notes (list)
	uv run chirp notes

note: ## Create or edit a single note (optional NAME)
	uv run chirp note $(NAME)

search: ## Keyword search through your notes
	uv run chirp search

ask: ## Start interactive chat with your notes
	uv run chirp ask

ask-question: ## Ask a question about your notes (requires QUESTION)
	uv run chirp ask --question "$(QUESTION)" $(if $(WHEN),--when "$(WHEN)")

index: ## Build notes search index
	uv run chirp index $(if $(FORCE),--force)

stats: ## Show Chirp statistics
	uv run chirp stats

about: ## Show the animated Chirp logo + version info
	uv run chirp about

version: ## Show the installed Chirp version
	uv run chirp version

config: ## Show Chirp configuration
	uv run chirp config --list

devices: ## List audio devices
	uv run chirp devices

init: ## First-run setup & model picker
	uv run chirp init $(if $(RECHECK),--recheck) $(if $(SWITCH_MODEL),--switch-model)

chirp-setup: ## Step-by-step audio setup guide (chirp setup)
	uv run chirp setup

chirp-test: ## Test Chirp dependencies and configuration (chirp test)
	uv run chirp test

# Testing commands
test: ## Run unit tests
	uv run pytest

test-file: ## Run a single test file (FILE=tests/test_settings.py)
	uv run pytest $(FILE)

test-match: ## Run tests matching a pattern (PATTERN=slugify)
	uv run pytest -k "$(PATTERN)"

test-coverage: ## Run tests with coverage report
	uv run pytest --cov --cov-report=html --cov-report=term

test-failed: ## Run only failed tests from last run
	uv run pytest --lf

test-verbose: ## Run tests with verbose output
	uv run pytest -vv


# Code quality commands
lint: ## Run linting with ruff
	uv run ruff check .

lint-fix: ## Run linting and fix issues
	uv run ruff check . --fix

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check code formatting
	uv run ruff format --check .

style: ## Run linting and formatting together
	@$(MAKE) lint-fix
	@$(MAKE) format

style-check: ## Check linting, formatting, and spelling
	@$(MAKE) lint
	@$(MAKE) format-check
	@$(MAKE) spell-check

type-check: ## Run type checking with mypy
	uv run mypy

spell-check: ## Check spelling with codespell
	uv run codespell

# Validation commands
validate: ## Validate code compiles and imports work
	@echo "🔍 Validating Python code..."
	@uv run python -m py_compile chirp/cli.py
	@uv run python -c "import chirp.cli; print('✅ CLI imports successfully')"
	@uv run python -c "import config.settings; print('✅ Config imports successfully')"
	@uv run python -c "import recorder.audio_recorder; print('✅ Recorder imports successfully')" || echo "⚠️  Recorder may need audio dependencies"
	@uv run python -c "import transcriber.whisper_transcriber; print('✅ Transcriber imports successfully')" || echo "⚠️  Transcriber may need model downloads"
	@uv run python -c "import notes.note_generator; print('✅ Notes generator imports successfully')"
	@uv run python -c "import notes_chat.index; print('✅ Notes chat imports successfully')" || echo "⚠️  Notes chat may need Ollama running"
	@echo "✅ Code validation complete!"

# Quality check combination
check: validate style-check type-check ## Run all quality checks (excluding tests)

ci: lint format-check type-check test ## Run CI checks (lint, format, type-check, test)

# Dependency management
check-deps: ## Check if all dependencies are available (alias for chirp-test)
	uv run chirp test

update: ## Update dependencies
	uv sync --upgrade

# Setup commands
setup: install ## Initial setup for new installations
	uv run pre-commit install
	@echo "✅ Chirp setup complete!"
	@echo "Next steps:"
	@echo "  1. Install BlackHole: make setup-blackhole"
	@echo "  2. Install Ollama: make setup-ollama"
	@echo "  3. Run 'make check-deps' to verify setup"

setup-blackhole: ## Show BlackHole installation instructions
	@echo "🎵 BlackHole Audio Driver Installation"
	@echo ""
	@echo "1. Install BlackHole:"
	@echo "   brew install blackhole-2ch"
	@echo ""
	@echo "2. Configure Audio MIDI Setup:"
	@echo "   - Open Audio MIDI Setup (Applications/Utilities)"
	@echo "   - Create a Multi-Output Device"
	@echo "   - Include both your speakers and BlackHole"
	@echo "   - Set this as your default output device"
	@echo ""
	@echo "4. Test with: make devices"

setup-ollama: ## Show Ollama setup instructions
	@echo "🧠 Ollama Setup Instructions"
	@echo ""
	@echo "1. Install Ollama:"
	@echo "   brew install ollama"
	@echo ""
	@echo "2. Start Ollama service:"
	@echo "   ollama serve"
	@echo ""
	@echo "3. Install required models:"
	@echo "   ollama pull llama3.1:8b"
	@echo "   ollama pull nomic-embed-text"
	@echo ""
	@echo "4. Test with: make check-deps"

# Build commands
build: ## Build package
	uv build


# Cleaning commands
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Clean everything including build artifacts

# Development workflow shortcuts
dev-workflow: ## Complete development workflow (style, type-check, test, build)
	@echo "🔄 Running development workflow..."
	@$(MAKE) style
	@$(MAKE) type-check
	@$(MAKE) test
	@$(MAKE) build
	@echo "✅ Development workflow complete!"

demo: ## Run a complete demo workflow
	@echo "🎬 Chirp Demo Workflow"
	@echo "1. Checking dependencies..."
	@$(MAKE) check-deps
	@echo "2. Showing stats..."
	@$(MAKE) stats
	@echo "3. Listing audio devices..."
	@$(MAKE) devices
	@echo "4. Showing configuration..."
	@$(MAKE) config
	@echo "✅ Demo complete! Ready to record with 'make record'"

# Documentation
docs: ## Show documentation locations
	@echo "📚 Chirp Documentation:"
	@echo "  - README.md - Main documentation"
	@echo "  - .docs/DEVELOPMENT.md - Developer guide"
	@echo "  - CLI help: uv run chirp --help"

# System information
info: ## Show system and version information
	@echo "🐣 Chirp System Information:"
	@echo "Python: $(shell python --version)"
	@echo "UV: $(shell uv --version)"
	@echo "Project: $(shell grep '^version' pyproject.toml | cut -d'"' -f2)"
	@echo "OS: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"
	@echo "Working Directory: $(PWD)"
	@echo ""
	@$(MAKE) stats

# Quick examples
example: ## Example: Record a 1-minute meeting and process it
	@echo "🎬 Running Chirp example workflow:"
	@echo "1. Recording 1-minute meeting titled 'Demo Meeting'..."
	@$(MAKE) record DURATION=1 TITLE="Demo Meeting" || echo "⚠️  Recording failed (audio setup required)"
	@echo "2. Processing audio files..."
	@$(MAKE) process FORCE=true
	@echo "3. Example notes search commands:"
	@echo "   make ask-question QUESTION='what was discussed?'"
	@echo "   make ask-question QUESTION='any action items?' WHEN='yesterday'"
