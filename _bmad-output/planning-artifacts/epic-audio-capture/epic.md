# Epic: Replace BlackHole with native macOS audio capture (ScreenCaptureKit + AVAudioEngine)

- **Epic ID:** EPIC-AUDIO-CAPTURE
- **Owner:** Colby
- **Status:** Done
- **Created:** 2026-04-27
- **Design source:** `audio_capture` technical spec (in-conversation, 2026-04-27); macOS-only v1
- **Related branch (current work):** `claude/implement-chirp-cli-IoXWx`

## 1. Goal

Eliminate the BlackHole virtual audio driver dependency and the manual Audio MIDI Setup multi-output device routing step. Replace the entire system-audio capture path with a first-party macOS solution: a small Swift helper (`capture_audio`) that uses ScreenCaptureKit for system audio and AVAudioEngine for microphone, streaming tagged length-prefixed PCM frames to a Python `audio_capture` module over a subprocess pipe.

After this epic, a fresh-box `chirp init` issues exactly one screen-recording permission prompt and one microphone permission prompt the first time `chirp record` runs — no driver install, no Audio MIDI Setup walk, no `brew install --cask blackhole-2ch`.

## 2. Why now

The locked wireframe direction (EPIC-WF-ALIGN) is shipping the CLI as a polished, low-friction product. BlackHole is the single worst part of the first-run experience:

1. **Manual driver install** — `brew install --cask blackhole-2ch` requires admin password + a kernel-extension approval reboot on some macOS versions.
2. **Manual Audio MIDI Setup walk** — users must open Audio MIDI Setup, create a Multi-Output Device combining Speakers + BlackHole 2ch, and set it as the system output. This is genuinely confusing and is the most-asked support question.
3. **Fragility across macOS updates** — system extensions periodically break or require re-approval. We saw this on macOS 14 → 15.
4. **No App Store path** — even though Chirp ships via PyPI today, kernel-extension dependencies foreclose any future packaging route (Mac App Store, signed `.app`, MDM-managed installs).
5. **Init flow noise** — `chirp/init_flow.py` carries ~60 lines of BlackHole-specific verify/install/prompt logic (`_blackhole_installed`, `_prompt_blackhole_routing`, the `[o]/[s]` picker) that vanish entirely once we replace the underlying capture mechanism.

ScreenCaptureKit (macOS 13+, audio-only support) is Apple's first-party answer. The trade-off — a screen-recording permission scope and the macOS 15 menu-bar recording indicator — is the same trade-off SuperWhisper and other native transcription tools accept. It is strictly better than the BlackHole UX.

## 3. Locked decisions from the spec

| # | Decision | Source |
|---|----------|--------|
| 1 | Capture is macOS-only in v1; no Linux / Windows path. Non-macOS hosts must keep working for non-recording commands (`notes`, `ask`, `search`, `about`). | Spec § "Out of Scope" |
| 2 | System audio uses ScreenCaptureKit (`SCStream` with `capturesAudio = true`, dummy 2x2 video, `excludesCurrentProcessAudio = true`). Microphone uses AVAudioEngine input-node tap. | Spec § "Capture Sources" |
| 3 | Both sources output 16 kHz mono float32 LPCM so they are directly comparable downstream. | Spec § "Audio Format" |
| 4 | Wire format on stdout: `[source: u8][timestamp_us: u64 LE][length: u32 LE][pcm: bytes]`. `0x01` = system, `0x02` = microphone. All writes serialized through a shared serial dispatch queue. | Spec § "Framing Protocol" |
| 5 | Helper ships as a minimal `.app` bundle (`CaptureAudio.app/Contents/{Info.plist, MacOS/capture_audio}`) so `NSMicrophoneUsageDescription` / `NSScreenCaptureUsageDescription` apply. CLI binary outside the bundle will not get prompted for permissions on macOS. | Spec § "Permissions and Bundle Structure" |
| 6 | Python module is a context manager named `AudioCapture` exposing `frames()`, `system_frames()`, `mic_frames()`. Mixing the two streams is **not** the module's responsibility — timestamps are provided so consumers can align. | Spec § "Public Interface" |
| 7 | Permission denial is reported via stderr lines `error: microphone_denied` / `error: screen_recording_denied` and surfaces in Python as `PermissionError` with System-Settings instructions. | Spec § "Startup Sequence" + "Error Handling" |
| 8 | Stdin on the helper is reserved for a future command channel (pause/resume/source-switch). Do **not** use it for anything else in v1. | Spec § "Future Considerations" |
| 9 | v1 ships unsigned. Distribution-side Gatekeeper steps are documented in the README. Codesigning is deferred. | Spec § "Code Signing" |
| 10 | The Swift helper's source, Info.plist, and Makefile live in-repo and are buildable via `python -m audio_capture.build` (which shells out to `swiftc`). The built `.app` bundle ships as package data via `importlib.resources`. | Spec § "Build" + "Packaging" |
| 11 | The `chirp init` flow no longer mentions BlackHole. Phase 1 verify drops `_blackhole_installed`; Phase 2 install drops the `blackhole-2ch` brew cask and the Audio MIDI Setup `[o]/[s]` prompt. A new check verifies `xcrun --find swiftc` returns success (needed only if we trigger the build at install time). | This epic, derived from spec |

## 4. Research findings — what already exists vs. what is missing

Validated against `claude/implement-chirp-cli-IoXWx` at the current commit.

### BlackHole-touching code that goes away

- `chirp/init_flow.py:111` — `_blackhole_installed()` (verify check)
- `chirp/init_flow.py:188` — `_blackhole_installed()` call inside `verify()`'s status list
- `chirp/init_flow.py:308-311` — `blackhole-2ch` brew cask install task in `install_missing()`
- `chirp/init_flow.py:330-331` — `_prompt_blackhole_routing(console)` invocation
- `chirp/init_flow.py:335-371` — the `_prompt_blackhole_routing` function itself (`[o]/[s]` picker, the "open Audio MIDI Setup" branch, the wireframe routing block)
- `recorder/device_manager.py:62-74` — `find_blackhole_device()`
- `recorder/device_manager.py:76-92` — `find_aggregate_device()` (only useful as a BlackHole companion)
- `recorder/device_manager.py:112-116` — `check_blackhole_available()` / `check_aggregate_available()`
- `chirp/cli.py:980-996` — the `devices` command's BlackHole/aggregate branch and "install BlackHole" hint
- `tests/test_device_manager.py:81-134` — `test_find_blackhole_device_*` and `test_find_aggregate_device_*` tests
- `tests/test_init_flow.py` — any test exercising `_blackhole_installed` or `_prompt_blackhole_routing` (verify before deletion)
- `README.md:11, 27, 40-44, 115` — BlackHole references in description, dependency list, manual setup, and troubleshooting
- `CLAUDE.md` — the line `optional BlackHole/aggregate devices (macOS)` under External Dependencies

### Audio capture call sites that move to `audio_capture`

- **`recorder/audio_recorder.py`** — `start_recording` opens a PyAudio stream on `device_manager.get_recommended_device()` (the system default input). Frames are int16, accumulated into `self.frames`, then written via `wave` to `<slug>/audio.wav`. This is the offline `chirp record` path.
- **`recorder/live_audio.py`** — `LiveAudioStream.start()` opens a PyAudio stream against the same recommended device and pushes `AudioFrame` chunks into a queue for the live transcription dashboard. This is the live recording / live-dashboard path.
- **`recorder/live_session.py`** — orchestrates `LiveAudioStream` + `LiveTranscriber`. Reads `audio_stream.sample_rate` / `audio_stream.channels` to drive WAV writing.

Both paths use `device_manager.get_recommended_device()` → `pyaudio.open(...)`. After this epic they go through `AudioCapture(...)` instead. The recorder modules no longer touch PyAudio for capture; PyAudio stays in `device_manager.list_devices()` (the read-only enumeration used by `chirp devices`) since that command remains useful.

### Already implemented (verify only)

- The wireframe-aligned storage layout (`<notes_root>/<slug>/{audio.wav, transcript.txt, notes.md, meta.toml}`) — story 1.1 done. The new capture path writes to the same `audio.wav` location.
- `chirp init` 4-phase flow with `--recheck` / `--switch-model` — story 1.5 done. The capture-permission verification slots into Phase 1 alongside the existing dependency checks.
- `chirp/about.py`, `chirp/cli.py` `transcribe` queue, `notes` sub-app — unrelated, untouched.

### Net code delta (rough)

- **Add:** ~250 lines Swift (`capture_audio.swift`), ~150 lines Python (`audio_capture/__init__.py`, `audio_capture/build.py`), ~50 lines tests/fixtures.
- **Remove:** ~120 lines BlackHole-specific Python in `init_flow.py`, `device_manager.py`, `cli.py`, plus the ~60 lines of BlackHole tests.
- **Rewrite:** the capture half of `recorder/audio_recorder.py` and `recorder/live_audio.py` — same public interface, new backing source.

## 5. Stories

Execution order matters: the capture module must exist (and be installable) before the recorder integration can replace its PyAudio capture path. BlackHole-removal is last so that we can run both paths side by side during development if needed.

| ID | Title | Depends on | File |
|----|-------|------------|------|
| 2.1 | `audio_capture` module — Swift helper + Python wrapper + packaging | — | [stories/2.1-audio-capture-module.md](stories/2.1-audio-capture-module.md) |
| 2.2 | Recorder integration — route `audio_recorder` and `live_audio` through `AudioCapture` | 2.1 | [stories/2.2-recorder-integration.md](stories/2.2-recorder-integration.md) |
| 2.3 | BlackHole removal — strip from init flow, devices command, tests, and docs | 2.1, 2.2 | [stories/2.3-blackhole-removal.md](stories/2.3-blackhole-removal.md) |

## 6. Epic-level acceptance criteria

- A fresh checkout on a clean macOS 13+ machine can `make -C audio_capture/swift build` (or `python -m audio_capture.build`) and produce `audio_capture/CaptureAudio.app/Contents/MacOS/capture_audio`.
- `from audio_capture import AudioCapture; with AudioCapture() as cap: next(cap.frames())` returns a `(source, timestamp_us, np.float32_array)` tuple within 2 seconds of context entry on a permission-granted machine.
- On a machine that has denied microphone permission, the same code raises `PermissionError` whose message names "System Settings → Privacy & Security → Microphone" and the executable that needs the grant (`CaptureAudio.app`).
- `chirp record "smoke test"` produces `~/Documents/chirp/<slug>/audio.wav` containing both the user's speech and any system audio playing during the recording, with no BlackHole or multi-output device installed.
- `chirp record` followed immediately by `chirp transcribe` produces a `transcript.txt` whose word count is non-zero — i.e. the new capture path delivers audio Whisper can transcribe.
- Live recording dashboard (`chirp record` interactive flow) shows VU-level updates and live transcription chunks driven by `AudioCapture` frames, not by PyAudio.
- `rg -i blackhole .` returns matches only inside `_bmad-output/` (planning history) and the `.claude/skills/` research notes — zero matches under `chirp/`, `recorder/`, `tests/`, `README.md`, or `CLAUDE.md`.
- `uv run chirp init` no longer mentions BlackHole or Audio MIDI Setup at any phase. The `[o]/[s]` picker is gone. Phase 1 prints a new `screen-recording permission` row that says either `granted` or `will prompt on first record`.
- `uv run chirp devices` lists input devices but no longer prints "✅ BlackHole detected" or "Install BlackHole from …" branches.
- `uv run pytest` passes the full suite (with the new `tests/test_audio_capture.py` fixture-mocked tests for the framing parser and subprocess lifecycle); `uv run ruff check .` reports no issues.

## 7. Out of scope / deferred

- **Linux and Windows capture paths.** Spec is explicit: macOS only in v1. Non-macOS hosts continue to work for `notes` / `ask` / `search`; `chirp record` raises a clear "macOS-only" error.
- **Stream mixing inside `audio_capture`.** Two timestamps + two iterators are delivered; mixing is the consumer's job. Story 2.2 will do the simplest possible mix (sample-aligned add with clipping) inline in `audio_recorder.py` since that's what offline recording needs.
- **Codesigning the `.app` bundle.** v1 ships unsigned. README documents the `xattr -cr` Gatekeeper workaround for direct PyPI install. Revisit when distribution widens (notably: when a signed installer ships).
- **Per-application audio filtering** (Zoom-only / browser-only via `SCRunningApplication` includes/excludes — macOS 14+). Out of scope for v1; revisit for a "meeting mode" story later.
- **Universal binary (arm64 + x86_64).** v1 builds arm64-only. If Intel Mac users surface, add an `x86_64` target to the Makefile.
- **Stdin command channel.** Reserved (`pause\n`, `resume\n`, `set_source mic\n`) but not implemented. Restart the helper to change capture state.
- **Migration of in-flight recordings.** No back-compat with BlackHole-routed recordings — there is nothing to migrate; old recordings already live as `audio.wav` files and are agnostic to capture source.

## 8. Risks

- **macOS 13 minimum bump.** ScreenCaptureKit's audio-only support is 13+. CLAUDE.md doesn't pin a macOS version today; we need to add `macOS 13.0+` to the README "Requirements" section. Mitigation: low risk — macOS 13 is 3+ years old by 2026.
- **Permission UX confusion.** Users will see "Chirp wants to record your screen" the first time they run `chirp record`, which is alarming if not framed up-front. Mitigation: print a one-line preamble on the first `chirp record` of a session: `you'll see a screen-recording permission prompt — chirp uses it for system audio only, no pixels are captured.` Story 2.2 owns this.
- **macOS 15 menu-bar indicator.** A persistent recording icon appears in the menu bar while `capture_audio` is running. Cosmetic but visible. Mitigation: document in the README troubleshooting section. Cannot be hidden (Apple-imposed).
- **Periodic re-authorization on macOS 15+.** Apple may re-prompt for screen-recording permission roughly monthly. Affects all SCK-based apps. Mitigation: detect the `screen_recording_denied` stderr message on a previously-authorized machine and surface "macOS asked you to re-confirm permission — open System Settings to grant again."
- **Build dependency on Xcode command-line tools.** `swiftc` is only present if `xcode-select --install` has been run. Mitigation: `python -m audio_capture.build` runs `xcrun --find swiftc` first and exits with `xcode command-line tools missing — run \`xcode-select --install\`` if absent. The wheel ships the prebuilt `.app` bundle so end users don't hit this; only developers building from source do.
- **Subprocess lifetime / zombie helpers.** A crashed Python parent could leave `capture_audio` running. Mitigation: `atexit` handler in `audio_capture/__init__.py` sends `SIGTERM`, with a 5-second wait then `SIGKILL`. The Swift helper additionally exits on broken-pipe write failure (after `signal(SIGPIPE, SIG_IGN)`).
- **Cache-warm assumption in tests.** The framing parser is straightforward to unit-test with a `BytesIO` stand-in for `proc.stdout`. The subprocess + permissions path is genuinely hardware-dependent and macOS-dependent; mark those tests `@pytest.mark.integration` and `@pytest.mark.skipif(sys.platform != 'darwin')` so CI on Linux skips them cleanly.
