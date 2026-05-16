"""Tests for chirpd.lifecycle and chirpd/__main__ Apple-Silicon checks."""

from __future__ import annotations

import asyncio
import fcntl
from pathlib import Path
from unittest import mock

import pytest

from chirpd import __main__ as chirpd_main
from chirpd.lifecycle import acquire_single_instance_lock, release_lock


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


def test_release_lock_swallows_oserror_on_unlock(tmp_path: Path) -> None:
    from chirpd import lifecycle

    lock_path = tmp_path / "chirpd.lock"
    handle = lifecycle.acquire_single_instance_lock(lock_path)
    assert handle is not None
    with mock.patch("chirpd.lifecycle.fcntl.flock", side_effect=OSError("forced")):
        lifecycle.release_lock(handle)


def test_flock_acquires_when_free(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    handle = acquire_single_instance_lock(lock_path)
    assert handle is not None
    try:
        assert lock_path.exists()
    finally:
        release_lock(handle)


def test_flock_returns_none_when_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    with mock.patch("chirpd.lifecycle.fcntl.flock", side_effect=BlockingIOError):
        handle = acquire_single_instance_lock(lock_path)
    assert handle is None


def test_flock_blocks_second_acquirer(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    first = acquire_single_instance_lock(lock_path)
    assert first is not None
    try:
        second = acquire_single_instance_lock(lock_path)
        assert second is None
    finally:
        release_lock(first)


def test_release_lock_allows_reacquire(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    first = acquire_single_instance_lock(lock_path)
    assert first is not None
    release_lock(first)

    second = acquire_single_instance_lock(lock_path)
    assert second is not None
    release_lock(second)


def test_release_lock_tolerates_already_unlocked(tmp_path: Path) -> None:
    lock_path = tmp_path / "chirpd.lock"
    handle = acquire_single_instance_lock(lock_path)
    assert handle is not None
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    release_lock(handle)


def test_apple_silicon_check_passes_on_arm64(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("chirpd.__main__.platform.machine", lambda: "arm64")
    monkeypatch.setattr("chirpd.__main__.acquire_single_instance_lock", lambda: None)
    monkeypatch.setattr("chirpd.__main__.configure_logging", lambda: None)
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
    monkeypatch.setattr("chirpd.__main__.configure_logging", lambda: None)
    monkeypatch.setattr("chirpd.__main__.ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr("chirpd.__main__.acquire_single_instance_lock", lambda: None)
    assert chirpd_main.main() == 0


def test_main_runs_serve_until_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="cd-main-", dir="/tmp"))
    socket_path = tmp_dir / "s"
    monkeypatch.setenv("CHIRP_DAEMON_SOCKET", str(socket_path))
    monkeypatch.setattr("chirpd.__main__.platform.machine", lambda: "arm64")
    monkeypatch.setattr("chirpd.__main__.configure_logging", lambda: None)
    monkeypatch.setattr("chirpd.__main__.ensure_runtime_dirs", lambda: None)

    class _DummyHandle:
        def close(self) -> None:
            pass

    handle = _DummyHandle()
    monkeypatch.setattr("chirpd.__main__.acquire_single_instance_lock", lambda: handle)
    released: dict[str, object] = {}
    monkeypatch.setattr(
        "chirpd.__main__.release_lock",
        lambda h: released.setdefault("handle", h),
    )

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

    assert released["handle"] is handle


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
