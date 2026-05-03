"""Tagged dual-source audio capture for macOS.

`AudioCapture` launches a bundled Swift helper (`CaptureAudio.app`) that
streams system audio (via ScreenCaptureKit) and the default microphone (via
AVAudioEngine) simultaneously, framed on stdout as
``[u8 source][u64 LE timestamp_us][u32 LE length][float32 PCM]``.

Example:
    from audio_capture import AudioCapture
    with AudioCapture() as cap:
        for source, timestamp_us, chunk in cap.frames():
            ...  # 1 = system audio, 2 = microphone

`source = 1` is system audio, `source = 2` is the microphone. Audio is
16 kHz mono float32. Timestamps are microseconds since a single shared
anchor captured before either engine starts (monotonic per source).

Requires macOS 13+. The package raises ``RuntimeError`` at import time on
non-Darwin hosts so callers fail loudly rather than silently degrading.

Building from source:
    Either ``make -C audio_capture/swift build`` or
    ``python -m audio_capture.build``. macOS 13+ and Swift 5.9+ required;
    install via ``xcode-select --install``. README-level macOS 13+
    requirement copy is owned by story 2.3.
"""

from __future__ import annotations

import sys

if sys.platform != "darwin":
    raise RuntimeError("audio_capture requires macOS 13+")

import queue
import struct
import subprocess
import threading
import time
import weakref
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import IO

import numpy as np

SOURCE_SYSTEM = 1
SOURCE_MICROPHONE = 2

_FRAME_HEADER_SIZE = 1 + 8 + 4
_STARTUP_TIMEOUT_SECONDS = 5.0
_PROC_WAIT_TIMEOUT = 5.0
_PROC_WAIT_FAILURE_TIMEOUT = 2.0
_STDERR_JOIN_TIMEOUT = 1.0
_CRASH_WAIT_TIMEOUT = 2.0


class AudioCaptureStartTimeout(RuntimeError):
    """Raised when the helper fails to emit `capture: started` in time."""


class AudioCaptureCrashed(RuntimeError):
    """Raised when the helper exits with a non-zero return code."""


def _resolve_binary_path() -> Path:
    package_files = resources.files("audio_capture")
    binary_resource = (
        package_files / "CaptureAudio.app" / "Contents" / "MacOS" / "capture_audio"
    )
    with resources.as_file(binary_resource) as path:
        return Path(path)


def _read_frame(
    stdout: IO[bytes],
) -> tuple[int, int, np.ndarray] | None:
    header = stdout.read(_FRAME_HEADER_SIZE)
    if len(header) < _FRAME_HEADER_SIZE:
        return None
    source = header[0]
    timestamp_us = struct.unpack("<Q", header[1:9])[0]
    length = struct.unpack("<I", header[9:13])[0]
    if length == 0:
        return source, timestamp_us, np.zeros(0, dtype=np.float32)
    payload = stdout.read(length)
    if len(payload) < length:
        return None
    audio = np.frombuffer(payload, dtype=np.float32)
    return source, timestamp_us, audio


class AudioCapture:
    """Context manager that yields tagged PCM frames from a Swift helper."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_queue: queue.Queue[str] = queue.Queue()
        self._stderr_thread: threading.Thread | None = None
        self._recent_stderr: list[str] = []
        self._cleaned_up = False
        self._atexit_finalizer: weakref.finalize | None = None

    def __enter__(self) -> AudioCapture:
        binary_path = _resolve_binary_path()
        if not binary_path.exists():
            raise FileNotFoundError(
                "capture_audio binary not found. Build it with: "
                "python -m audio_capture.build"
            )

        self._proc = subprocess.Popen(
            [str(binary_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        try:
            self._start_stderr_drain()
            self._wait_for_startup()
        except BaseException:
            self._cleanup_after_failed_start()
            raise

        self._atexit_finalizer = weakref.finalize(self, _atexit_cleanup, self._proc)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._shutdown()

    def _start_stderr_drain(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None

        def drain(stderr: IO[bytes], q: queue.Queue[str]) -> None:
            try:
                for raw in iter(stderr.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    q.put(line)
            except Exception:
                pass

        thread = threading.Thread(
            target=drain,
            args=(proc.stderr, self._stderr_queue),
            daemon=True,
            name="audio_capture-stderr",
        )
        thread.start()
        self._stderr_thread = thread

    def _wait_for_startup(self) -> None:
        # The helper emits `capture: awaiting_permission` once before each TCC
        # prompt (mic, then screen recording). Reset the deadline on every
        # such line so a slow second prompt does not blow the budget.
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                tail = "\n".join(self._recent_stderr[-5:])
                raise AudioCaptureStartTimeout(
                    "capture_audio helper did not start within "
                    f"{_STARTUP_TIMEOUT_SECONDS:.1f}s. Recent stderr:\n{tail}"
                )
            try:
                line = self._stderr_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            self._recent_stderr.append(line)
            if line == "capture: started":
                return
            if line == "capture: awaiting_permission":
                deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
                continue
            if line.startswith("error: "):
                code = line[len("error: ") :].strip()
                raise self._permission_error(code)

    def _permission_error(self, code: str) -> Exception:
        if code == "microphone_denied":
            return PermissionError(
                "Chirp needs microphone access. Open System Settings → "
                "Privacy & Security → Microphone and grant access to "
                "CaptureAudio.app, then try again."
            )
        if code == "screen_recording_denied":
            return PermissionError(
                "Chirp needs Screen Recording access. Open System Settings → "
                "Privacy & Security → Screen Recording and grant access to "
                "CaptureAudio.app, then try again."
            )
        return AudioCaptureCrashed(f"capture_audio failed with error: {code}")

    def _cleanup_after_failed_start(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=_PROC_WAIT_FAILURE_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=_PROC_WAIT_FAILURE_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=_STDERR_JOIN_TIMEOUT)
        self._cleaned_up = True

    def _shutdown(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=_PROC_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=_PROC_WAIT_FAILURE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=_STDERR_JOIN_TIMEOUT)
        if self._atexit_finalizer is not None:
            self._atexit_finalizer.detach()

    def _drain_recent_stderr(self) -> None:
        while True:
            try:
                line = self._stderr_queue.get_nowait()
            except queue.Empty:
                return
            self._recent_stderr.append(line)

    def frames(self) -> Iterator[tuple[int, int, np.ndarray]]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        while True:
            frame = _read_frame(proc.stdout)
            if frame is None:
                break
            yield frame
        try:
            proc.wait(timeout=_CRASH_WAIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._drain_recent_stderr()
            tail = "\n".join(self._recent_stderr[-5:])
            raise AudioCaptureCrashed(
                "capture_audio helper stopped streaming but did not exit "
                f"within {_CRASH_WAIT_TIMEOUT:.1f}s. Recent stderr:\n{tail}"
            )
        if proc.returncode not in (0, None):
            self._drain_recent_stderr()
            tail = "\n".join(self._recent_stderr[-5:])
            raise AudioCaptureCrashed(
                "capture_audio helper exited with code "
                f"{proc.returncode}. Recent stderr:\n{tail}"
            )

    def system_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for source, timestamp_us, audio in self.frames():
            if source == SOURCE_SYSTEM:
                yield timestamp_us, audio

    def mic_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for source, timestamp_us, audio in self.frames():
            if source == SOURCE_MICROPHONE:
                yield timestamp_us, audio


def _atexit_cleanup(proc: subprocess.Popen[bytes]) -> None:
    try:
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=_PROC_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except ProcessLookupError:
                    return
                try:
                    proc.wait(timeout=_PROC_WAIT_FAILURE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass
    except Exception:
        pass


__all__ = [
    "AudioCapture",
    "AudioCaptureCrashed",
    "AudioCaptureStartTimeout",
    "SOURCE_MICROPHONE",
    "SOURCE_SYSTEM",
]
