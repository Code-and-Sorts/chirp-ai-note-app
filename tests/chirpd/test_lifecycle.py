"""Tests for chirpd.lifecycle and chirpd/__main__ Apple-Silicon checks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import pytest

from chirpd import __main__ as chirpd_main
from chirpd.lifecycle import single_instance_lock


def test_ensure_runtime_dirs_creates_both_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from chirpd import lifecycle, paths

    app_support = tmp_path / "AppSupport" / "chirp"
    log_dir = tmp_path / "Logs" / "chirp"
    monkeypatch.setattr(paths, "APP_SUPPORT_DIR", app_support)
    monkeypatch.setattr(paths, "LOG_DIR", log_dir)

    lifecycle.ensure_runtime_dirs()

    assert app_support.is_dir()
    assert log_dir.is_dir()


def test_flock_acquires_when_free(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    with single_instance_lock(lock_path) as acquired:
        assert acquired
        assert lock_path.exists()


def test_flock_yields_false_when_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    with mock.patch("chirpd.lifecycle.fcntl.flock", side_effect=BlockingIOError):
        with single_instance_lock(lock_path) as acquired:
            assert acquired is False


def test_flock_blocks_second_acquirer(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    with single_instance_lock(lock_path) as first:
        assert first
        with single_instance_lock(lock_path) as second:
            assert second is False


def test_different_socket_locks_acquire_concurrently(tmp_path: Path) -> None:
    """AC-3: two daemons on different sockets get independent locks."""
    from chirpd.paths import lock_path_for_socket

    primary_socket = tmp_path / "primary.sock"
    secondary_socket = tmp_path / "secondary.sock"
    primary_lock = lock_path_for_socket(primary_socket)
    secondary_lock = lock_path_for_socket(secondary_socket)
    assert primary_lock != secondary_lock

    with single_instance_lock(primary_lock) as primary:
        assert primary
        with single_instance_lock(secondary_lock) as secondary:
            assert secondary, "a distinct socket must yield a distinct, free lock"


def test_same_socket_locks_mutually_exclude(tmp_path: Path) -> None:
    """AC-3: two daemons on the same socket still contend (loser exits)."""
    from chirpd.paths import lock_path_for_socket

    socket_path = tmp_path / "shared.sock"
    derived = lock_path_for_socket(socket_path)
    with single_instance_lock(derived) as first:
        assert first
        with single_instance_lock(derived) as second:
            assert second is False


def test_lock_releases_on_exit_so_reacquire_succeeds(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    with single_instance_lock(lock_path) as first:
        assert first
    with single_instance_lock(lock_path) as second:
        assert second


def test_lock_release_swallows_oserror_on_unlock(tmp_path: Path) -> None:
    """flock(LOCK_UN) failures during release must not propagate."""
    lock_path = tmp_path / "chirpd.lock"
    real_flock = __import__("fcntl").flock
    call_count = {"n": 0}

    def _flaky_flock(fd: int, op: int) -> None:
        call_count["n"] += 1
        # Let the initial LOCK_EX | LOCK_NB succeed via the real flock; raise
        # only on the LOCK_UN that single_instance_lock issues in its finally.
        if call_count["n"] == 1:
            real_flock(fd, op)
            return
        raise OSError("forced unlock failure")

    with mock.patch("chirpd.lifecycle.fcntl.flock", side_effect=_flaky_flock):
        with single_instance_lock(lock_path) as acquired:
            assert acquired


def _patch_lock(monkeypatch: pytest.MonkeyPatch, acquired: bool) -> None:
    """Replace ``single_instance_lock`` with a contextmanager yielding ``acquired``."""
    import contextlib

    @contextlib.contextmanager
    def _fake_lock(_path: Path | None = None):
        yield acquired

    monkeypatch.setattr("chirpd.__main__.single_instance_lock", _fake_lock)


def test_apple_silicon_check_passes_on_arm64(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("chirpd.__main__.platform.machine", lambda: "arm64")
    _patch_lock(monkeypatch, acquired=False)
    monkeypatch.setattr("chirpd.__main__.configure_logging", lambda **_: None)
    monkeypatch.setattr("chirpd.__main__.ensure_runtime_dirs", lambda: None)
    assert chirpd_main.main() == 0


def test_apple_silicon_check_fails_on_x86_64(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("chirpd.__main__.platform.machine", lambda: "x86_64")
    exit_code = chirpd_main.main()
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "arm64" in captured.err
    assert "x86_64" in captured.err


def test_resolve_socket_path_respects_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "override.sock"
    monkeypatch.setenv("CHIRP_DAEMON_SOCKET", str(override))
    assert chirpd_main._resolve_socket_path() == override


def test_resolve_socket_path_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHIRP_DAEMON_SOCKET", raising=False)
    assert chirpd_main._resolve_socket_path() == chirpd_main.DEFAULT_SOCKET_PATH


def test_main_returns_zero_when_lock_already_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chirpd.__main__.platform.machine", lambda: "arm64")
    monkeypatch.setattr("chirpd.__main__.configure_logging", lambda **_: None)
    monkeypatch.setattr("chirpd.__main__.ensure_runtime_dirs", lambda: None)
    _patch_lock(monkeypatch, acquired=False)
    assert chirpd_main.main() == 0


def test_main_passes_lock_derived_from_resolved_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AC-3: an override socket isolates the lock so a second daemon can run."""
    import contextlib

    from chirpd.paths import lock_path_for_socket

    override = tmp_path / "override.sock"
    monkeypatch.setenv("CHIRP_DAEMON_SOCKET", str(override))
    monkeypatch.setattr("chirpd.__main__.platform.machine", lambda: "arm64")
    monkeypatch.setattr("chirpd.__main__.configure_logging", lambda **_: None)
    monkeypatch.setattr("chirpd.__main__.ensure_runtime_dirs", lambda: None)

    captured: dict[str, Path | None] = {}

    @contextlib.contextmanager
    def _capturing_lock(lock_path: Path | None = None):
        captured["lock_path"] = lock_path
        yield False

    monkeypatch.setattr("chirpd.__main__.single_instance_lock", _capturing_lock)

    assert chirpd_main.main() == 0
    assert captured["lock_path"] == lock_path_for_socket(override)
    assert captured["lock_path"] != chirpd_main.lock_path_for_socket(
        chirpd_main.DEFAULT_SOCKET_PATH
    )


def test_main_runs_serve_until_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="cd-main-", dir="/tmp"))
    socket_path = tmp_dir / "s"
    monkeypatch.setenv("CHIRP_DAEMON_SOCKET", str(socket_path))
    monkeypatch.setattr("chirpd.__main__.platform.machine", lambda: "arm64")
    monkeypatch.setattr("chirpd.__main__.configure_logging", lambda **_: None)
    monkeypatch.setattr("chirpd.__main__.ensure_runtime_dirs", lambda: None)
    _patch_lock(monkeypatch, acquired=True)

    original_run = chirpd_main.asyncio.run

    def _run_with_keyboard_interrupt(coro: object) -> None:
        coro.close()  # type: ignore[attr-defined]
        raise KeyboardInterrupt

    monkeypatch.setattr(chirpd_main.asyncio, "run", _run_with_keyboard_interrupt)
    try:
        assert chirpd_main.main() == 0
    finally:
        monkeypatch.setattr(chirpd_main.asyncio, "run", original_run)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


async def test_run_returns_when_serve_task_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    from chirpd.dispatcher import Dispatcher

    tmp_dir = Path(tempfile.mkdtemp(prefix="cd-run-", dir="/tmp"))
    socket_path = tmp_dir / "s"
    dispatcher = Dispatcher()
    run_task = asyncio.create_task(chirpd_main._run(socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 1.0
    while not socket_path.exists() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    run_task.cancel()
    try:
        await asyncio.wait_for(run_task, timeout=1.0)
    except asyncio.CancelledError:
        pass
    try:
        if socket_path.exists():
            socket_path.unlink()
        tmp_dir.rmdir()
    except OSError:
        pass
