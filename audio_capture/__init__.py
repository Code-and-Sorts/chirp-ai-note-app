"""Tagged dual-source audio capture for macOS.

`AudioCapture` launches a bundled Swift helper (`Chirp.app`) that
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

Requires macOS 13+ at runtime. Importing the package is platform-neutral
so wire-format helpers (``_read_frame``) and exception types stay
introspectable on any host. ``AudioCapture().__enter__`` raises
``RuntimeError`` on non-Darwin so callers fail loudly at first use.

Building from source:
    Either ``make -C audio_capture/swift build`` or
    ``python -m audio_capture.build``. macOS 13+ and Swift 5.9+ required;
    install via ``xcode-select --install``. README-level macOS 13+
    requirement copy is owned by story 2.3.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import platform
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import types
import weakref
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import IO

import numpy as np

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = 1
SOURCE_MICROPHONE = 2

_REQUIRED_MACOS_MAJOR = 13


def check_macos_version() -> None:
    """Raise ``RuntimeError`` unless the host is macOS 13.0+.

    Used by both `AudioCapture.__enter__` and the recorder/live-audio
    entry points so the error message stays in one place. Looks at
    ``sys.platform`` and ``platform.mac_ver()`` so a macOS 12 host (where
    ``sys.platform`` is still ``darwin``) gets the advertised error
    instead of a deeper subprocess-launch failure.
    """
    if sys.platform != "darwin":
        raise RuntimeError("chirp record requires macOS 13 or later")
    version_string = platform.mac_ver()[0]
    if not version_string:
        raise RuntimeError("chirp record requires macOS 13 or later")
    try:
        major = int(version_string.split(".", 1)[0])
    except (TypeError, ValueError):
        raise RuntimeError("chirp record requires macOS 13 or later") from None
    if major < _REQUIRED_MACOS_MAJOR:
        raise RuntimeError("chirp record requires macOS 13 or later")


_FRAME_HEADER_SIZE = 1 + 8 + 4
# Sanity cap on a single frame's payload; well above any realistic 16 kHz
# float32 chunk (a 1-second buffer is 64 KiB). Defends against a
# malformed length prefix from a wedged or corrupted helper.
_MAX_FRAME_PAYLOAD_BYTES = 8 * 1024 * 1024
_STARTUP_TIMEOUT_SECONDS = 5.0
# Post-`capture: started` window during which we drain the diagnostic lines
# the helper emits next (sample_rate, system_audio, microphone=...). The
# Swift side flushes all four lines back-to-back, so this only needs to cover
# pipe + drainer-thread latency.
_POST_START_DRAIN_SECONDS = 0.5
_PROC_WAIT_TIMEOUT = 5.0
_PROC_WAIT_FAILURE_TIMEOUT = 2.0
_STDERR_JOIN_TIMEOUT = 1.0
_CRASH_WAIT_TIMEOUT = 2.0

_MIC_DEVICE_LINE_RE = re.compile(r'^capture: microphone=enabled device="(.+)"$')


class AudioCaptureStartTimeout(RuntimeError):
    """Raised when the helper fails to emit `capture: started` in time."""


class AudioCaptureCrashed(RuntimeError):
    """Raised when the helper exits with a non-zero return code."""


class AudioCaptureCorrupt(RuntimeError):
    """Raised when the helper's framing protocol is violated.

    Distinct from EOF (clean stream end) and from `AudioCaptureCrashed`
    (helper exited non-zero) so callers can tell "stream truncated by
    corruption" apart from "stream ended normally".
    """


def _resolve_binary_path() -> contextlib.AbstractContextManager[Path]:
    package_root = Path(__file__).parent
    package_files = resources.files("audio_capture")
    binary_resource = (
        package_files / "Chirp.app" / "Contents" / "MacOS" / "capture_audio"
    )
    try:
        ctx = resources.as_file(binary_resource)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Chirp.app is not bundled in this installation. "
            "The wheel must be built on macOS via `python -m audio_capture.build` "
            "before installation."
        ) from exc

    class _IntegrityCheckedContext:
        def __enter__(self) -> Path:
            path = ctx.__enter__()
            if not path.is_relative_to(package_root):
                raise RuntimeError(
                    "resolved Chirp.app path is outside the package directory"
                )
            return path

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None,
        ) -> None:
            ctx.__exit__(exc_type, exc_val, exc_tb)

    return _IntegrityCheckedContext()


def _read_exactly(stream: IO[bytes], n: int) -> bytes | None:
    """Read exactly ``n`` bytes from ``stream``, looping on short reads.

    Returns ``None`` only when the very first ``read`` call returns zero
    bytes (clean EOF before any frame data).  If some bytes have already
    arrived when EOF is encountered mid-frame, raises ``AudioCaptureCorrupt``
    because the stream was truncated inside a frame.
    """
    buf = stream.read(n)
    if len(buf) == 0:
        return None
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if len(chunk) == 0:
            raise AudioCaptureCorrupt(
                f"truncated frame: expected {n} bytes, got {len(buf)} before EOF"
            )
        buf += chunk
    return buf


def _read_frame(
    stdout: IO[bytes],
) -> tuple[int, int, np.ndarray] | None:
    header = _read_exactly(stdout, _FRAME_HEADER_SIZE)
    if header is None:
        return None
    source = header[0]
    timestamp_us = struct.unpack("<Q", header[1:9])[0]
    payload_len = struct.unpack("<I", header[9:13])[0]
    if payload_len == 0:
        return source, timestamp_us, np.zeros(0, dtype=np.float32)
    if payload_len > _MAX_FRAME_PAYLOAD_BYTES:
        raise AudioCaptureCorrupt(
            f"payload length {payload_len} exceeds max {_MAX_FRAME_PAYLOAD_BYTES}"
        )
    if payload_len % 4 != 0:
        raise AudioCaptureCorrupt(
            f"payload length {payload_len} is not a multiple of 4 (float32 alignment)"
        )
    payload = _read_exactly(stdout, payload_len)
    if payload is None:
        raise AudioCaptureCorrupt(
            f"truncated frame: expected {payload_len} bytes, got 0 before EOF"
        )
    try:
        audio = np.frombuffer(payload, dtype=np.float32)
    except ValueError as exc:
        raise AudioCaptureCorrupt(
            f"np.frombuffer failed on {payload_len}-byte payload: {exc}"
        ) from exc
    return source, timestamp_us, audio


class AudioCapture:
    """Context manager that yields tagged PCM frames from a Swift helper."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_queue: queue.Queue[str] = queue.Queue()
        self._stderr_thread: threading.Thread | None = None
        self._recent_stderr: collections.deque[str] = collections.deque(maxlen=200)
        self._cleaned_up = False
        self._atexit_finalizer: weakref.finalize | None = None
        self._binary_resource_ctx: contextlib.AbstractContextManager[Path] | None = None
        self.mic_device_name: str | None = None

    def __enter__(self) -> AudioCapture:
        check_macos_version()
        ctx = _resolve_binary_path()
        binary_path = ctx.__enter__()
        self._binary_resource_ctx = ctx
        try:
            if not binary_path.exists():
                raise FileNotFoundError(
                    "capture_audio binary not found. Build it with: "
                    "python -m audio_capture.build"
                )

            # `capture_audio` re-spawns itself with the macOS
            # `responsibility_spawnattrs_setdisclaim` flag set so the
            # screen-recording / mic prompts attribute to the bundled
            # `Chirp.app` (and a Chirp row appears in System
            # Settings) rather than inheriting the parent terminal's TCC
            # state. From Python's perspective the first invocation is a
            # tiny launcher shim; the second invocation does the actual
            # capture.
            self._proc = subprocess.Popen(
                [str(binary_path)],
                stdin=subprocess.DEVNULL,
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
        except BaseException:
            if self._binary_resource_ctx is not None:
                self._binary_resource_ctx.__exit__(None, None, None)
                self._binary_resource_ctx = None
            raise

        self._atexit_finalizer = weakref.finalize(self, _atexit_cleanup, self._proc)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._shutdown()

    def _start_stderr_drain(self) -> None:
        proc = self._proc
        assert proc is not None
        assert proc.stderr is not None

        def drain(stderr: IO[bytes], q: queue.Queue[str]) -> None:
            try:
                for raw in iter(stderr.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    q.put(line)
            except OSError as exc:
                q.put(f"[audio_capture drain error] {exc}")

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
                tail = "\n".join(list(self._recent_stderr)[-5:])
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
                self._drain_post_start_diagnostics()
                return
            if line == "capture: awaiting_permission":
                deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
                continue
            if line.startswith("error: "):
                code = line[len("error: ") :].strip()
                raise self._permission_error(code)

    def _drain_post_start_diagnostics(self) -> None:
        # The helper emits `capture: sample_rate=...`, `capture: system_audio=...`,
        # and `capture: microphone=enabled device="..."` immediately after
        # `capture: started`. Drain them here so `mic_device_name` is set
        # before `__enter__` returns; callers reading it right after entering
        # the context manager get a deterministic value.
        deadline = time.monotonic() + _POST_START_DRAIN_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.debug(
                    "audio_capture: post-start drain timed out without "
                    "microphone diagnostic line; mic_device_name stays None"
                )
                return
            try:
                line = self._stderr_queue.get(timeout=remaining)
            except queue.Empty:
                logger.debug(
                    "audio_capture: post-start drain ran out of lines "
                    "before microphone diagnostic"
                )
                return
            self._recent_stderr.append(line)
            match = _MIC_DEVICE_LINE_RE.match(line)
            if match:
                self.mic_device_name = match.group(1)
                return

    def _permission_error(self, code: str) -> Exception:
        if code == "microphone_denied":
            return PermissionError(
                "Chirp needs microphone access. Open System Settings → "
                "Privacy & Security → Microphone and grant access to "
                "Chirp, then try again."
            )
        if code == "screen_recording_denied":
            return PermissionError(
                "Chirp needs Screen Recording access. Open System Settings → "
                "Privacy & Security → Screen Recording and grant access to "
                "Chirp, then try again."
            )
        return AudioCaptureCrashed(f"capture_audio failed with error: {code}")

    def _cleanup_after_failed_start(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            # Process already exited before terminate() could reach it.
            pass
        try:
            proc.wait(timeout=_PROC_WAIT_FAILURE_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                # Process already exited before kill() could reach it.
                pass
            try:
                proc.wait(timeout=_PROC_WAIT_FAILURE_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Cleanup must not mask the original startup error; best-effort wait.
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
                # SIGTERM is sent to the launcher (outer) PID that Python
                # tracks. The launcher's `disclaimForwardSignal` handler
                # forwards the signal to the disclamed child PID and arms a
                # 2-second SIGKILL escalation via SIGALRM, so the child is
                # reliably terminated even if it ignores SIGTERM. The
                # launcher then re-raises the child's exit signal and exits
                # itself, giving Python a clean process reap.
                proc.terminate()
            except ProcessLookupError:
                # Process already exited between poll() and terminate().
                pass
            try:
                proc.wait(timeout=_PROC_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except ProcessLookupError:
                    # Process already exited between timeout and kill() attempt.
                    pass
                try:
                    proc.wait(timeout=_PROC_WAIT_FAILURE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    # Process did not exit after kill(); best-effort shutdown.
                    pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=_STDERR_JOIN_TIMEOUT)
        if self._atexit_finalizer is not None:
            self._atexit_finalizer.detach()
        if self._binary_resource_ctx is not None:
            self._binary_resource_ctx.__exit__(None, None, None)
            self._binary_resource_ctx = None

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
            tail = "\n".join(list(self._recent_stderr)[-5:])
            raise AudioCaptureCrashed(
                "capture_audio helper stopped streaming but did not exit "
                f"within {_CRASH_WAIT_TIMEOUT:.1f}s. Recent stderr:\n{tail}"
            )
        if proc.returncode not in (0, None):
            self._drain_recent_stderr()
            tail = "\n".join(list(self._recent_stderr)[-5:])
            raise AudioCaptureCrashed(
                "capture_audio helper exited with code "
                f"{proc.returncode}. Recent stderr:\n{tail}"
            )


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
                    # Best-effort wait at interpreter exit; force-continue cleanup.
                    pass
    except Exception:  # noqa: BLE001 - atexit handler; logger may be torn down at shutdown
        # Swallow all errors during interpreter shutdown; cleanup is best-effort.
        pass


def check_permissions() -> dict[str, str]:
    try:
        ctx = _resolve_binary_path()
    except RuntimeError as exc:
        raise FileNotFoundError(str(exc)) from exc
    with ctx as binary_path:
        if not binary_path.exists():
            raise FileNotFoundError(
                "capture_audio binary not found. Build it with: "
                "python -m audio_capture.build"
            )
        try:
            result = subprocess.run(
                [str(binary_path), "--check-permissions"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioCaptureStartTimeout(
                "The capture helper did not respond to --check-permissions "
                "within 3s; it may be wedged. Check System Settings → Privacy "
                "& Security (Microphone and Screen Recording) and try again."
            ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"capture_audio --check-permissions exited {result.returncode}: "
            f"{result.stderr.strip()!r}"
        )

    permissions: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if line.startswith("permission: "):
            payload = line[len("permission: ") :]
            if "=" not in payload:
                raise RuntimeError(
                    f"malformed permission line in helper output: {result.stdout!r}"
                )
            key, _, value = payload.partition("=")
            permissions[key] = value
    if "screen_recording" not in permissions or "microphone" not in permissions:
        raise RuntimeError(
            f"expected screen_recording and microphone permission lines; got: {result.stdout!r}"
        )
    return permissions


__all__ = [
    "SOURCE_MICROPHONE",
    "SOURCE_SYSTEM",
    "AudioCapture",
    "AudioCaptureCorrupt",
    "AudioCaptureCrashed",
    "AudioCaptureStartTimeout",
    "check_macos_version",
    "check_permissions",
]
