"""Tests for the `audio_capture` package.

Parser tests, fake-helper end-to-end tests, and the non-Darwin guard test
run on every platform. Tests that depend on the real built Swift binary
are gated to macOS via ``@pytest.mark.skipif``.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import os
import struct
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from audio_capture import (
    AudioCapture,
    AudioCaptureCrashed,
    AudioCaptureStartTimeout,
    _read_frame,
)


@pytest.fixture(autouse=True)
def _force_darwin_platform(request: pytest.FixtureRequest) -> Iterator[None]:
    if "real_platform" in request.keywords:
        yield
        return
    with mock.patch.object(sys, "platform", "darwin"):
        yield


def _pack_frame(source: int, timestamp_us: int, samples: np.ndarray) -> bytes:
    pcm: bytes = samples.astype(np.float32).tobytes()
    header = struct.pack("<BQI", source, timestamp_us, len(pcm))
    return header + pcm


def test_read_frame_parses_system_audio() -> None:
    samples = np.array([0.1, -0.25, 0.5], dtype=np.float32)
    stream = io.BytesIO(_pack_frame(1, 12345, samples))
    result = _read_frame(stream)
    assert result is not None
    source, timestamp_us, audio = result
    assert source == 1
    assert timestamp_us == 12345
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, samples)


def test_read_frame_parses_microphone() -> None:
    samples = np.array([0.0, 1.0], dtype=np.float32)
    stream = io.BytesIO(_pack_frame(2, 999, samples))
    result = _read_frame(stream)
    assert result is not None
    source, timestamp_us, audio = result
    assert source == 2
    assert timestamp_us == 999
    np.testing.assert_array_equal(audio, samples)


def test_read_frame_returns_none_on_eof() -> None:
    assert _read_frame(io.BytesIO(b"")) is None


def test_read_frame_returns_none_on_partial_header() -> None:
    assert _read_frame(io.BytesIO(b"\x01\x00\x00\x00\x00")) is None


def test_read_frame_returns_none_on_partial_payload() -> None:
    header = struct.pack("<BQI", 1, 0, 16)
    stream = io.BytesIO(header + b"\x00\x00\x00\x00")
    assert _read_frame(stream) is None


def _spawn_fake_helper(
    stderr_lines: list[str], frames: list[bytes], exit_code: int = 0
) -> list[str]:
    return [
        sys.executable,
        "-c",
        textwrap.dedent(
            f"""
            import os, struct, sys, time
            for line in {stderr_lines!r}:
                sys.stderr.write(line + '\\n')
                sys.stderr.flush()
            for chunk in {frames!r}:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            sys.exit({exit_code})
            """
        ),
    ]


def _patch_resolve_to(path: Path):
    return mock.patch(
        "audio_capture._resolve_binary_path",
        return_value=contextlib.nullcontext(path),
    )


def test_permission_denied_raises_permission_error(tmp_path: Path) -> None:
    fake_binary = tmp_path / "capture_audio"
    fake_binary.write_text("#!/bin/sh\necho stub\n")
    fake_binary.chmod(0o755)

    cmd = _spawn_fake_helper(
        stderr_lines=["error: microphone_denied"],
        frames=[],
        exit_code=1,
    )

    real_popen = subprocess.Popen

    def popen_override(args, **kwargs):
        return real_popen(cmd, **kwargs)

    with (
        _patch_resolve_to(fake_binary),
        mock.patch("audio_capture.subprocess.Popen", side_effect=popen_override),
    ):
        with pytest.raises(PermissionError) as excinfo:
            with AudioCapture():
                pass
    assert "Microphone" in str(excinfo.value)


def test_missing_binary_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with _patch_resolve_to(missing):
        with pytest.raises(FileNotFoundError) as excinfo:
            with AudioCapture():
                pass
    assert "python -m audio_capture.build" in str(excinfo.value)


def test_atexit_kills_lingering_process(tmp_path: Path) -> None:
    fake_binary = tmp_path / "capture_audio"
    fake_binary.write_text("#!/bin/sh\nsleep 60\n")
    fake_binary.chmod(0o755)

    real_popen = subprocess.Popen

    # ["sleep", "60"] is a stand-in for the real Swift binary so we can
    # exercise the atexit path without launching capture_audio. The fake
    # never emits diagnostics, so we patch _wait_for_startup to no-op.
    sleep_cmd = ["sleep", "60"]

    def popen_override(args, **kwargs):
        return real_popen(sleep_cmd, **kwargs)

    with (
        _patch_resolve_to(fake_binary),
        mock.patch("audio_capture.subprocess.Popen", side_effect=popen_override),
        mock.patch.object(AudioCapture, "_wait_for_startup", return_value=None),
    ):
        cap = AudioCapture().__enter__()
        proc = cap._proc
        assert proc is not None
        assert proc.poll() is None

        from audio_capture import _atexit_cleanup

        _atexit_cleanup(proc)

    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.05)
    assert proc.poll() is not None


def test_enter_failure_cleans_up_subprocess(tmp_path: Path) -> None:
    fake_binary = tmp_path / "capture_audio"
    fake_binary.write_text("#!/bin/sh\nsleep 30\n")
    fake_binary.chmod(0o755)

    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen[bytes]] = []

    def popen_override(args, **kwargs):
        proc = real_popen(["sleep", "30"], **kwargs)
        spawned.append(proc)
        return proc

    with (
        _patch_resolve_to(fake_binary),
        mock.patch("audio_capture.subprocess.Popen", side_effect=popen_override),
        mock.patch("audio_capture._STARTUP_TIMEOUT_SECONDS", 0.5),
    ):
        with pytest.raises(AudioCaptureStartTimeout):
            with AudioCapture():
                pass

    assert spawned
    proc = spawned[0]
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.05)
    assert proc.poll() is not None


def test_frames_raises_on_nonzero_exit(tmp_path: Path) -> None:
    fake_binary = tmp_path / "capture_audio"
    fake_binary.write_text("#!/bin/sh\necho stub\n")
    fake_binary.chmod(0o755)

    samples = np.array([0.5, 0.25, 0.0, -0.25], dtype=np.float32)
    frame_bytes = _pack_frame(1, 1000, samples)

    cmd = _spawn_fake_helper(
        stderr_lines=["capture: started", "error: screen_recording_denied"],
        frames=[frame_bytes],
        exit_code=1,
    )

    real_popen = subprocess.Popen

    def popen_override(args, **kwargs):
        return real_popen(cmd, **kwargs)

    with (
        _patch_resolve_to(fake_binary),
        mock.patch("audio_capture.subprocess.Popen", side_effect=popen_override),
    ):
        with AudioCapture() as cap:
            iterator = cap.frames()
            first = next(iterator)
            assert first[0] == 1
            assert first[1] == 1000
            np.testing.assert_array_equal(first[2], samples)
            with pytest.raises(AudioCaptureCrashed) as excinfo:
                for _ in iterator:
                    pass
    assert "screen_recording_denied" in str(excinfo.value)


def test_monotonic_timestamps_per_source() -> None:
    samples = np.array([0.1, 0.2], dtype=np.float32)
    payload = b"".join(_pack_frame(1, ts, samples) for ts in (10, 20, 30, 40))
    stream = io.BytesIO(payload)
    timestamps = []
    while True:
        frame = _read_frame(stream)
        if frame is None:
            break
        timestamps.append(frame[1])
    assert timestamps == [10, 20, 30, 40]
    for prev, curr in itertools.pairwise(timestamps):
        assert curr > prev


def test_end_to_end_with_python_fake_helper(tmp_path: Path) -> None:
    fake_binary = tmp_path / "capture_audio"
    fake_binary.write_text("#!/bin/sh\necho stub\n")
    fake_binary.chmod(0o755)

    samples_sys = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    samples_mic = np.array([-0.1, -0.2], dtype=np.float32)
    frames = [
        _pack_frame(1, 100, samples_sys),
        _pack_frame(2, 110, samples_mic),
        _pack_frame(1, 200, samples_sys),
        _pack_frame(2, 210, samples_mic),
    ]

    cmd = _spawn_fake_helper(
        stderr_lines=["capture: started"],
        frames=frames,
        exit_code=0,
    )

    real_popen = subprocess.Popen

    def popen_override(args, **kwargs):
        return real_popen(cmd, **kwargs)

    with (
        _patch_resolve_to(fake_binary),
        mock.patch("audio_capture.subprocess.Popen", side_effect=popen_override),
    ):
        with AudioCapture() as cap:
            collected = list(cap.frames())
    assert [(s, ts) for s, ts, _ in collected] == [
        (1, 100),
        (2, 110),
        (1, 200),
        (2, 210),
    ]

    with (
        _patch_resolve_to(fake_binary),
        mock.patch("audio_capture.subprocess.Popen", side_effect=popen_override),
    ):
        with AudioCapture() as cap:
            sys_only = list(cap.system_frames())
    assert [ts for ts, _ in sys_only] == [100, 200]

    with (
        _patch_resolve_to(fake_binary),
        mock.patch("audio_capture.subprocess.Popen", side_effect=popen_override),
    ):
        with AudioCapture() as cap:
            mic_only = list(cap.mic_frames())
    assert [ts for ts, _ in mic_only] == [110, 210]


def test_wait_for_startup_resets_deadline_on_each_awaiting_permission(
    tmp_path: Path,
) -> None:
    fake_binary = tmp_path / "capture_audio"
    fake_binary.write_text("#!/bin/sh\necho stub\n")
    fake_binary.chmod(0o755)

    cmd = [
        sys.executable,
        "-c",
        textwrap.dedent(
            """
            import sys, time
            sys.stderr.write('capture: awaiting_permission\\n')
            sys.stderr.flush()
            time.sleep(0.6)
            sys.stderr.write('capture: awaiting_permission\\n')
            sys.stderr.flush()
            time.sleep(0.6)
            sys.stderr.write('capture: started\\n')
            sys.stderr.flush()
            sys.stdout.buffer.flush()
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                pass
            """
        ),
    ]

    real_popen = subprocess.Popen

    def popen_override(args, **kwargs):
        return real_popen(cmd, **kwargs)

    with (
        _patch_resolve_to(fake_binary),
        mock.patch("audio_capture.subprocess.Popen", side_effect=popen_override),
        mock.patch("audio_capture._STARTUP_TIMEOUT_SECONDS", 1.0),
    ):
        with AudioCapture() as cap:
            assert cap._proc is not None


@pytest.mark.real_platform
def test_non_darwin_capture_raises() -> None:
    with mock.patch.object(sys, "platform", "linux"):
        with pytest.raises(RuntimeError) as excinfo:
            with AudioCapture():
                pass
    assert "macOS 13+" in str(excinfo.value)


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only")
def test_built_bundle_binary_is_executable(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    bundle_binary = (
        repo_root
        / "audio_capture"
        / "CaptureAudio.app"
        / "Contents"
        / "MacOS"
        / "capture_audio"
    )
    if not bundle_binary.exists():
        pytest.skip("Swift helper not built; run `python -m audio_capture.build`")
    assert os.access(bundle_binary, os.X_OK)


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only wheel build")
def test_wheel_bundles_executable_helper(tmp_path: Path) -> None:
    import zipfile

    repo_root = Path(__file__).resolve().parent.parent
    bundle_binary = (
        repo_root
        / "audio_capture"
        / "CaptureAudio.app"
        / "Contents"
        / "MacOS"
        / "capture_audio"
    )
    if not bundle_binary.exists():
        pytest.skip("Swift helper not built; run `python -m audio_capture.build`")

    result = subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"uv build failed:\n{result.stdout}\n{result.stderr}")

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    wheel_path = wheels[0]

    binary_in_wheel = "audio_capture/CaptureAudio.app/Contents/MacOS/capture_audio"
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()
        assert binary_in_wheel in names, (
            f"{binary_in_wheel} missing from wheel; zipfile contains: "
            f"{[n for n in names if 'audio_capture' in n][:10]}"
        )
        unix_mode = (zf.getinfo(binary_in_wheel).external_attr >> 16) & 0o777
        assert unix_mode & 0o111, (
            f"helper binary in wheel lacks any execute bit: mode={oct(unix_mode)}"
        )
