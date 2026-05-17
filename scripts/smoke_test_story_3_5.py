"""Automated smoke test for story 3.5 (AC-29).

Spawns chirpd against a temporary models.toml, exercises the model lifecycle
ops end-to-end against a real MLX backend, and verifies the NFR-S3 / FR5 /
FR9 / NFR-O3 budgets without requiring the user to babysit two terminals.

Run with::

    uv run python scripts/smoke_test_story_3_5.py
    uv run python scripts/smoke_test_story_3_5.py --model mlx-community/Llama-3.2-1B-Instruct-4bit

The script uses CHIRP_MODEL_IDLE_TIMEOUT=10 so the idle-unload assertion
finishes in ~15 s rather than the 5 min production default. Pre-existing
~/Library/Application Support/chirp/models.toml is backed up and restored.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL_REPO = "mlx-community/gemma-4-4b-it-4bit"
IDLE_TIMEOUT_S = 10
DAEMON_START_TIMEOUT_S = 5.0
MODEL_LOAD_TIMEOUT_S = 120.0
SOCKET_POLL_INTERVAL_S = 0.1

APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "chirp"
MODELS_TOML_PATH = APP_SUPPORT_DIR / "models.toml"
SOCKET_PATH = APP_SUPPORT_DIR / "chirpd.sock"
LOG_FILE = Path.home() / "Library" / "Logs" / "chirp" / "chirpd.log"


def _hf_cache_dir_for(repo: str) -> Path:
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return base / "hub" / f"models--{repo.replace('/', '--')}"


def _step(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)


def _ok(message: str) -> None:
    print(f"  PASS: {message}", flush=True)


def _fail(message: str) -> str:
    print(f"  FAIL: {message}", flush=True)
    return message


def _alias_for(repo: str) -> str:
    last = repo.rsplit("/", 1)[-1]
    return last.lower().replace("_", "-")


def _preflight(repo: str) -> str | None:
    if platform.machine() != "arm64":
        return f"requires Apple Silicon (arm64); detected {platform.machine()}"
    cache_dir = _hf_cache_dir_for(repo)
    if not cache_dir.exists():
        return (
            f"model weights not in HF cache at {cache_dir}\n  Run: hf download {repo}"
        )
    return None


def _stage_models_toml(repo: str, alias: str) -> Path | None:
    APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup_path: Path | None = None
    if MODELS_TOML_PATH.exists():
        backup_path = MODELS_TOML_PATH.with_suffix(".toml.smoke-backup")
        shutil.copy2(MODELS_TOML_PATH, backup_path)
        print(f"  backed up existing models.toml → {backup_path}", flush=True)
    MODELS_TOML_PATH.write_text(
        "schema_version = 1\n"
        f'default_chat = "{alias}"\n'
        "\n"
        f'[models."{alias}"]\n'
        f'hf_repo = "{repo}"\n'
        'role = "chat"\n'
    )
    return backup_path


def _restore_models_toml(backup_path: Path | None) -> None:
    if backup_path is None:
        MODELS_TOML_PATH.unlink(missing_ok=True)
        return
    shutil.move(str(backup_path), MODELS_TOML_PATH)


def _wait_for_socket(timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if SOCKET_PATH.is_socket():
            return True
        time.sleep(SOCKET_POLL_INTERVAL_S)
    return False


def _chirpd_binary() -> str:
    candidate = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "chirpd"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("chirpd")
    if found:
        return found
    raise RuntimeError(
        "chirpd console script not found; install with `uv pip install -e .`"
    )


def _spawn_daemon() -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["CHIRP_MODEL_IDLE_TIMEOUT"] = str(IDLE_TIMEOUT_S)
    return subprocess.Popen(
        [_chirpd_binary()],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _terminate_daemon(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _lsof_network_sockets(pid: int) -> list[str]:
    result = subprocess.run(
        ["lsof", "-a", "-iTCP", "-iUDP", "-P", "-n", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[1:] if lines else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_REPO,
        help=f"HF repo to load (default: {DEFAULT_MODEL_REPO})",
    )
    args = parser.parse_args()
    repo = args.model
    alias = _alias_for(repo)

    failures: list[str] = []

    _step("Preflight")
    err = _preflight(repo)
    if err is not None:
        _fail(err)
        return 1
    _ok(f"Apple Silicon ({platform.machine()})")
    _ok(f"model in HF cache: {repo}")

    _step("Stage models.toml")
    backup_path = _stage_models_toml(repo, alias)
    _ok(f"wrote registry → {MODELS_TOML_PATH}")

    _step(f"Spawn chirpd (CHIRP_MODEL_IDLE_TIMEOUT={IDLE_TIMEOUT_S})")
    proc = _spawn_daemon()
    try:
        if not _wait_for_socket(DAEMON_START_TIMEOUT_S):
            _fail(
                f"socket {SOCKET_PATH} did not appear within {DAEMON_START_TIMEOUT_S}s"
            )
            try:
                stdout, stderr = proc.communicate(timeout=2)
                if stdout:
                    print(f"  daemon stdout:\n    {stdout.decode(errors='replace')}")
                if stderr:
                    print(f"  daemon stderr:\n    {stderr.decode(errors='replace')}")
            except subprocess.TimeoutExpired:
                proc.kill()
            return _restore_or(1, proc, backup_path)
        _ok(f"daemon pid {proc.pid} listening on socket")

        from llm.client import LLMClient

        client = LLMClient()

        _step("model.list returns the registered alias")
        models = client.model_list_sync()
        aliases = [m["alias"] for m in models]
        if alias not in aliases:
            failures.append(
                _fail(f"registered alias missing from model_list: {aliases}")
            )
        else:
            _ok(f"registry reports {aliases}")

        _step("model.load against the real MLX backend")
        load_started = time.monotonic()
        client.model_load_sync(alias)
        load_elapsed = time.monotonic() - load_started
        _ok(f"model_load_sync returned in {load_elapsed:.2f}s")

        _step("model.status shows the model loaded with non-zero RSS")
        status = client.model_status_sync()
        loaded_aliases = [m["alias"] for m in status.get("models", [])]
        if alias not in loaded_aliases:
            failures.append(_fail(f"model not in status.models: {loaded_aliases}"))
        else:
            _ok(f"models={loaded_aliases}")
        if status.get("rss_bytes", 0) <= 0:
            failures.append(_fail(f"rss_bytes not positive: {status.get('rss_bytes')}"))
        else:
            rss_mb = status["rss_bytes"] / (1024 * 1024)
            _ok(f"rss_bytes={status['rss_bytes']} ({rss_mb:.1f} MiB)")
        for key in ("pid", "uptime_seconds", "daemon_version"):
            if key not in status:
                failures.append(_fail(f"status missing required key {key!r}"))
        if all(k in status for k in ("pid", "uptime_seconds", "daemon_version")):
            _ok(
                f"status keys present: pid={status['pid']} "
                f"uptime={status['uptime_seconds']:.2f}s "
                f"version={status['daemon_version']}"
            )

        _step("NFR-S3: daemon opens no network sockets")
        network_rows = _lsof_network_sockets(proc.pid)
        if network_rows:
            failures.append(
                _fail(
                    "lsof reports network sockets:\n    " + "\n    ".join(network_rows)
                )
            )
        else:
            _ok("lsof -iTCP -iUDP returned no rows")

        _step(f"FR5: idle-unload fires within {IDLE_TIMEOUT_S}s + buffer")
        wait_total = IDLE_TIMEOUT_S + 5
        print(f"  sleeping {wait_total}s for idle timer...", flush=True)
        time.sleep(wait_total)
        post_status = client.model_status_sync()
        still_loaded = [m["alias"] for m in post_status.get("models", [])]
        if alias in still_loaded:
            failures.append(
                _fail(f"model still loaded after idle window: {still_loaded}")
            )
        else:
            _ok(f"model unloaded; status.models={still_loaded}")

    finally:
        _step("Teardown")
        _terminate_daemon(proc)
        _ok(f"daemon exit code {proc.returncode}")
        _restore_models_toml(backup_path)
        _ok("models.toml restored")
        if LOG_FILE.exists():
            print(f"  daemon log: {LOG_FILE}", flush=True)

    _step("Summary")
    if failures:
        for line in failures:
            print(f"  - {line}", flush=True)
        print(f"\n{len(failures)} check(s) failed.", flush=True)
        return 1
    print("\nAll checks passed.", flush=True)
    return 0


def _restore_or(code: int, proc: subprocess.Popen[bytes], backup: Path | None) -> int:
    _terminate_daemon(proc)
    _restore_models_toml(backup)
    return code


if __name__ == "__main__":
    sys.exit(main())
