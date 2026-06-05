"""Tests for ``chirpd/launchd.py`` (story 5.4 — LaunchAgent install/uninstall).

The ``launchctl`` seam is subprocess-driven (not permission/device-driven), so it
mocks cleanly: :func:`subprocess.run` is patched to return canned
``CompletedProcess`` results and the module-level ``LAUNCH_AGENT_PLIST_PATH`` is
redirected into ``tmp_path``. No test touches the real ``~/Library/LaunchAgents``
or shells out to ``launchctl``. ``sys.platform`` is forced to ``"darwin"`` so the
suite runs identically on CI Linux.
"""

from __future__ import annotations

import plistlib
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from chirpd import launchd

CHIRPD_PATH = "/usr/local/bin/chirpd"


@pytest.fixture(autouse=True)
def _force_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the platform-gated functions as if on macOS, regardless of host OS."""
    monkeypatch.setattr(launchd.sys, "platform", "darwin")


@pytest.fixture
def plist_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the module plist path into a temp dir (parent intentionally absent)."""
    target = tmp_path / "LaunchAgents" / "com.chirp.chirpd.plist"
    monkeypatch.setattr(launchd, "LAUNCH_AGENT_PLIST_PATH", target)
    return target


def _completed(
    args: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def _patch_launchctl(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> list[list[str]]:
    """Patch ``subprocess.run``; ``handler(args) -> CompletedProcess``.

    Returns a list that accumulates the ``launchctl`` sub-args of every call, so
    tests can assert ordering (e.g. ``unload`` before ``load`` under ``--force``).
    """
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command[1:])  # drop the leading "launchctl"
        return handler(command)

    monkeypatch.setattr(launchd.subprocess, "run", fake_run)
    return calls


def _success_handler(command: list[str]) -> subprocess.CompletedProcess[str]:
    """``load``/``unload`` succeed; ``list`` reports the agent present."""
    if command[1] == "list":
        return _completed(command, returncode=0, stdout="{ ... };\n")
    return _completed(command, returncode=0)


# --- _build_plist -----------------------------------------------------------


def test_build_plist_has_required_keys() -> None:
    payload = launchd._build_plist(Path(CHIRPD_PATH))

    assert payload["Label"] == "com.chirp.chirpd"
    assert payload["ProgramArguments"] == [CHIRPD_PATH]
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["StandardOutPath"] == str(launchd.LAUNCH_AGENT_LOG_PATH)
    assert payload["StandardErrorPath"] == str(launchd.LAUNCH_AGENT_LOG_PATH)
    assert payload["ProcessType"] == "Background"


def test_build_plist_propagates_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/bin")

    payload = launchd._build_plist(Path(CHIRPD_PATH))

    assert payload["EnvironmentVariables"]["PATH"] == "/opt/homebrew/bin:/usr/bin"


def test_build_plist_propagates_hf_home_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "/tmp/hf")

    payload = launchd._build_plist(Path(CHIRPD_PATH))

    assert payload["EnvironmentVariables"]["HF_HOME"] == "/tmp/hf"


def test_build_plist_omits_hf_home_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)

    payload = launchd._build_plist(Path(CHIRPD_PATH))

    assert "HF_HOME" not in payload["EnvironmentVariables"]


# --- install_launch_agent ---------------------------------------------------


def test_install_writes_plist_atomically(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    monkeypatch.setattr(launchd.shutil, "which", lambda _name: CHIRPD_PATH)
    _patch_launchctl(monkeypatch, _success_handler)

    returned = launchd.install_launch_agent()

    assert returned == plist_path
    assert plist_path.exists()
    with plist_path.open("rb") as handle:
        on_disk = plistlib.load(handle)
    assert on_disk["Label"] == "com.chirp.chirpd"
    assert on_disk["ProgramArguments"] == [CHIRPD_PATH]


def test_install_raises_already_installed_without_force(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("stale")
    monkeypatch.setattr(launchd.shutil, "which", lambda _name: CHIRPD_PATH)

    with pytest.raises(launchd.LaunchAgentAlreadyInstalled):
        launchd.install_launch_agent()


def test_install_with_force_replaces_existing(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("stale-content")
    monkeypatch.setattr(launchd.shutil, "which", lambda _name: CHIRPD_PATH)
    calls = _patch_launchctl(monkeypatch, _success_handler)

    launchd.install_launch_agent(force=True)

    with plist_path.open("rb") as handle:
        on_disk = plistlib.load(handle)
    assert on_disk == launchd._build_plist(Path(CHIRPD_PATH))
    # Force must unload the old agent before loading the freshly-written plist.
    verbs = [args[0] for args in calls]
    assert verbs.index("unload") < verbs.index("load")


def test_install_raises_when_chirpd_missing(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    monkeypatch.setattr(launchd.shutil, "which", lambda _name: None)

    with pytest.raises(launchd.LaunchAgentError, match="not found on PATH"):
        launchd.install_launch_agent()


def test_install_raises_launchctl_failed_on_nonzero(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    monkeypatch.setattr(launchd.shutil, "which", lambda _name: CHIRPD_PATH)

    def handler(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "load":
            return _completed(command, returncode=5, stderr="Load failed: 5: I/O error")
        return _success_handler(command)

    _patch_launchctl(monkeypatch, handler)

    with pytest.raises(launchd.LaunchctlFailed) as excinfo:
        launchd.install_launch_agent()

    assert excinfo.value.returncode == 5
    assert "Load failed: 5" in excinfo.value.stderr


def test_install_raises_launchctl_failed_on_empty_list_output(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    monkeypatch.setattr(launchd.shutil, "which", lambda _name: CHIRPD_PATH)

    def handler(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "list":
            return _completed(command, returncode=0, stdout="")
        return _completed(command, returncode=0)

    _patch_launchctl(monkeypatch, handler)

    with pytest.raises(launchd.LaunchctlFailed):
        launchd.install_launch_agent()


def test_write_plist_atomic_cleans_up_tmp_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "com.chirp.chirpd.plist"

    def boom(_src: object, _dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(launchd.os, "replace", boom)

    with pytest.raises(OSError, match="replace failed"):
        launchd._write_plist_atomic({"Label": "x"}, target)

    assert not target.with_name(target.name + ".tmp").exists()
    assert not target.exists()


# --- uninstall_launch_agent -------------------------------------------------


def test_uninstall_runs_unload_and_removes_plist(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("present")

    def handler(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "list":
            return _completed(command, returncode=1)  # agent gone
        return _completed(command, returncode=0)

    _patch_launchctl(monkeypatch, handler)

    launchd.uninstall_launch_agent()

    assert not plist_path.exists()


def test_uninstall_raises_not_installed_when_no_plist(plist_path: Path) -> None:
    with pytest.raises(launchd.LaunchAgentNotInstalled):
        launchd.uninstall_launch_agent()


def test_uninstall_raises_when_still_listed_after_unload(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    # unload returns 0 and the plist is unlinked, but `launchctl list` still
    # reports the agent (macOS edge case) — surface it; the error names the
    # already-removed plist so the user isn't left guessing.
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("present")
    _patch_launchctl(monkeypatch, lambda command: _completed(command, returncode=0))

    with pytest.raises(launchd.LaunchctlFailed, match="already been removed"):
        launchd.uninstall_launch_agent()
    assert not plist_path.exists()


def test_uninstall_raises_launchctl_failed_on_nonzero(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("present")

    def handler(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "unload":
            return _completed(command, returncode=3, stderr="Unload failed")
        return _completed(command, returncode=0)

    _patch_launchctl(monkeypatch, handler)

    with pytest.raises(launchd.LaunchctlFailed):
        launchd.uninstall_launch_agent()
    # Plist must remain on disk so the user can inspect / retry.
    assert plist_path.exists()


# --- is_launch_agent_installed ----------------------------------------------


def test_is_launch_agent_installed_true_when_both_present(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("present")
    _patch_launchctl(monkeypatch, lambda command: _completed(command, returncode=0))

    assert launchd.is_launch_agent_installed() is True


def test_is_launch_agent_installed_false_when_plist_missing(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    called = False

    def handler(command: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return _completed(command, returncode=0)

    _patch_launchctl(monkeypatch, handler)

    assert launchd.is_launch_agent_installed() is False
    assert called is False  # launchctl must not run when the plist is absent


def test_is_launch_agent_installed_false_when_launchctl_missing(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("present")
    _patch_launchctl(monkeypatch, lambda command: _completed(command, returncode=1))

    assert launchd.is_launch_agent_installed() is False


# --- installed_chirpd_path --------------------------------------------------


def test_installed_chirpd_path_reads_baked_value(
    monkeypatch: pytest.MonkeyPatch, plist_path: Path
) -> None:
    monkeypatch.setattr(launchd.shutil, "which", lambda _name: CHIRPD_PATH)
    _patch_launchctl(monkeypatch, _success_handler)
    launchd.install_launch_agent()

    assert launchd.installed_chirpd_path(plist_path) == CHIRPD_PATH
    # Default arg resolves the module-level plist path (monkeypatched here too).
    assert launchd.installed_chirpd_path() == CHIRPD_PATH


def test_installed_chirpd_path_none_when_plist_missing(plist_path: Path) -> None:
    assert launchd.installed_chirpd_path(plist_path) is None


def test_installed_chirpd_path_none_when_plist_malformed(plist_path: Path) -> None:
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("not a plist")

    assert launchd.installed_chirpd_path(plist_path) is None


def test_installed_chirpd_path_none_when_program_arguments_absent(
    plist_path: Path,
) -> None:
    plist_path.parent.mkdir(parents=True)
    with plist_path.open("wb") as handle:
        plistlib.dump({"Label": "com.chirp.chirpd"}, handle)

    assert launchd.installed_chirpd_path(plist_path) is None


def test_installed_chirpd_path_none_when_program_arguments_empty(
    plist_path: Path,
) -> None:
    plist_path.parent.mkdir(parents=True)
    with plist_path.open("wb") as handle:
        plistlib.dump({"Label": "com.chirp.chirpd", "ProgramArguments": []}, handle)

    assert launchd.installed_chirpd_path(plist_path) is None


# --- non-Darwin guard -------------------------------------------------------


def test_non_darwin_raises(monkeypatch: pytest.MonkeyPatch, plist_path: Path) -> None:
    monkeypatch.setattr(launchd.sys, "platform", "linux")

    with pytest.raises(launchd.LaunchAgentError, match="macOS-only"):
        launchd.install_launch_agent()
    with pytest.raises(launchd.LaunchAgentError, match="macOS-only"):
        launchd.uninstall_launch_agent()
    assert launchd.is_launch_agent_installed() is False
