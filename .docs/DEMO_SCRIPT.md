# 🎬 Chirp Demo Script for README GIFs

This script provides step-by-step demos to showcase Chirp's key features for README.md video recordings.

---

## 🎥 Demo 1: Recording a Meeting (30-45 seconds)

**Goal**: Show how easy it is to start and stop a recording.

### Setup
```bash
# Clear terminal
clear

# Show we're ready
chirp status
```

### Recording
```bash
# Start a short recording (30 seconds for demo)
chirp record --duration 30 --title "Product Demo Discussion"

# Show recording in progress with:
# - Timer counting up
# - Audio level indicators
# - File size growing

# Let it run for ~15 seconds, then Ctrl+C to stop early
# Or let it complete the 30 seconds
```

### Verification
```bash
# Show the new recording was created
chirp status

# Optional: Show the audio file directly
ls -lh audio-in/
```

**Key Highlights**:
- Clean, simple command
- Real-time feedback (timer, audio levels)
- Graceful interruption with Ctrl+C

---

## 🎥 Demo 2: Full Processing Pipeline (45-60 seconds)

**Goal**: Show the end-to-end workflow from audio → transcription → notes.

### Setup
```bash
clear
chirp status
```

### Process Everything
```bash
# Run the full pipeline
chirp process

# This will show:
# 1. Transcribing audio files (progress bar)
# 2. Generating meeting notes (progress bar)
# 3. Building search index (progress indicator)
```

### View Results
```bash
# Show generated notes
ls -lh notes-out/

# Display a note file (pick one with nice formatting)
cat notes-out/2024-01-15_Product_Demo_Discussion.md
```

**Key Highlights**:
- One command does everything
- Clear progress indicators
- Beautiful output formatting

---

## 🎥 Demo 3: Interactive Chat Mode (60-90 seconds)

**Goal**: Showcase the beautiful interactive chat interface - the star feature!

### Setup
```bash
clear

# Ensure index exists
chirp notes index
```

### Interactive Session
```bash
# Start interactive chat
chirp notes ask

# The welcome panel appears:
# ╭──────────────────────────────────╮
# │     Notes Chat                   │
# │ Ask questions about your         │
# │ meeting notes.                   │
# │                                  │
# │ Press Ctrl+C twice to exit       │
# ╰──────────────────────────────────╯
```

### Ask Questions
```bash
# Question 1: Show search in action
> what were the key decisions from recent meetings?

# Wait for:
# - Spinner: "Searching meeting notes..."
# - Spinner: "Generating answer..."
# - Streaming response from chirp 🐣
# - Sources displayed at bottom
# - "cached" indicator if applicable

# Question 2: Follow-up question
> who is responsible for implementing those decisions?

# Shows conversation flow

# Question 3: Try Ctrl+C behavior
> this is a test to show ctr<Ctrl+C>
# Shows: "Press Ctrl+C again to exit" in toolbar
# Buffer clears, hint disappears after 2 seconds

# Exit with double Ctrl+C or type and submit
<Ctrl+C>
<Ctrl+C>
# Shows: "Goodbye!"
```

**Key Highlights**:
- Beautiful bordered interface
- Real-time search feedback
- Streaming AI responses
- Source attribution
- Smart Ctrl+C handling
- Professional CLI experience

---

## 🎥 Demo 4: One-Shot Questions (30-45 seconds)

**Goal**: Show quick question answering without entering interactive mode.

### Setup
```bash
clear
```

### Quick Questions
```bash
# Ask a specific question
chirp notes ask --question "what action items were assigned this week?"

# Shows:
# - Search indicator
# - Answer streaming in
# - Source files listed
```

### Advanced Queries
```bash
# Time-filtered search
chirp notes ask -q "budget discussions" --when "last week"

# Dry run to see search results
chirp notes ask -q "project timeline" --dry-run
```

**Key Highlights**:
- Fast answers without interactive mode
- Time range filtering
- Dry-run debugging option

---

## 🎥 Demo 5: Status & Diagnostics (30-45 seconds)

**Goal**: Show system status and health checks.

### Setup
```bash
clear
```

### Status Check
```bash
# Comprehensive status
chirp status

# Shows:
# - Audio files count
# - Transcriptions count
# - Notes count
# - Index status
# - Total duration
```

### Device Information
```bash
# Show audio devices
chirp devices

# Lists:
# - Available input devices
# - BlackHole status
# - Recommended device
```

### Health Check
```bash
# Run full diagnostic
chirp test

# Tests:
# ✓ BlackHole installation
# ✓ chirpd daemon readiness
# ✓ Registered default chat model
# ✓ Embedding model (nomic-embed-text)
# ✓ Notes index
```

**Key Highlights**:
- Clear system overview
- Easy diagnostics
- Helpful status messages

---

## 🎥 Demo 6: Configuration Management (20-30 seconds)

**Goal**: Show how easy it is to configure Chirp.

### Setup
```bash
clear
```

### View Config
```bash
# List current configuration
chirp config --list

# Shows table with:
# - All settings
# - Current values
# - Organized by category
```

### Update Settings
```bash
# Update directories
chirp config --audio-dir ./my-recordings
chirp config --notes-dir ./my-notes

# Verify changes
chirp config --list
```

**Key Highlights**:
- Simple configuration
- Beautiful table display
- Easy to customize

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
1. **Interactive Chat Mode** - The star feature! Show streaming, sources, Ctrl+C behavior
2. **Full Pipeline** - One command to rule them all (`chirp process`)
3. **Recording** - Show simplicity of starting/stopping

### Nice-to-Have
4. **One-Shot Questions** - Quick queries without interactive mode
5. **Status & Diagnostics** - System health at a glance

### Optional
6. **Configuration** - If you have extra space/time

---

## 📝 Caption Ideas for README

### Demo 1: Recording
> "Start recording in seconds with real-time feedback"

### Demo 2: Processing
> "One command processes everything: transcribe → notes → search index"

### Demo 3: Interactive Chat
> "Beautiful interactive chat with streaming AI responses and source attribution"

### Demo 4: Quick Questions
> "Get instant answers with time-filtered search"

### Demo 5: Status
> "Monitor your meeting library and system health"

---

## 🎨 README Structure Suggestion

```markdown
## ✨ See It In Action

### 🎙️ Record Meetings
![Recording Demo](docs/gifs/demo-recording.gif)

### ⚡ Process Everything
![Processing Demo](docs/gifs/demo-process.gif)

### 💬 Interactive Chat (Our Favorite!)
![Chat Demo](docs/gifs/demo-chat.gif)

### 🔍 Quick Questions
![Query Demo](docs/gifs/demo-query.gif)
```

---

## 🔧 Preparation Checklist

Before recording:

- [ ] Clear all previous recordings/transcriptions for clean demo
- [ ] Ensure the daemon is healthy (`chirp daemon status`)
- [ ] Verify a model is registered (`chirp models list`)
- [ ] Test audio recording works (`chirp devices`)
- [ ] Build search index (`chirp notes index`)
- [ ] Increase terminal font size
- [ ] Set terminal dimensions (100x30)
- [ ] Clear command history
- [ ] Close unnecessary terminal tabs
- [ ] Practice each demo 2-3 times

---

## 🚀 Quick Reset Script

Use this between takes to reset the demo environment:

```bash
#!/bin/bash
# reset-demo.sh

# Clear terminal
clear

# Remove all demo files (CAUTION: backup first!)
rm -rf audio-in/*
rm -rf transcriptions/*
rm -rf notes-out/*
rm -rf search-index/*

# Verify clean state
chirp status

echo "✅ Demo environment reset"
```

**⚠️ WARNING**: Only run this in a demo/test environment, not on production data!
