"""Automated smoke test for story 3.7 (AC-10 / AC-14).

Drives the full vertical slice: `chirp ask` → `llm.client` → in-process
chirpd → MLX backend → streamed tokens to stdout. Captures the timings,
lsof, socket-mode, and log-slice evidence the AC-14 PR checklist asks for.

Run with::

    uv run python scripts/smoke_test_story_3_7.py
    uv run python scripts/smoke_test_story_3_7.py --question "say hi in five words"
    uv run python scripts/smoke_test_story_3_7.py --check-idle-unload

Prerequisites:
- Apple Silicon arm64 Mac.
- Model weights present in the Hugging Face cache. The default model is
  ``mlx-community/gemma-4-4b-it-4bit``; acquire with ``hf download <repo>``.
- An indexed note corpus already in chroma. Use ``--mock-retrieval`` to bypass
  retrieval and feed a canned context if you just want the LLM-path evidence.

The script backs up and restores any existing ~/Library/Application
Support/chirp/models.toml and kills any pre-existing chirpd before
running so timings are not contaminated. Pass ``--keep-models-toml`` to
leave an existing registry in place.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_REPO = "mlx-community/gemma-4-4b-it-4bit"
DEFAULT_QUESTION = "say hi in five words"
ASK_TIMEOUT_S = 90.0
COLD_FIRST_TOKEN_BUDGET_M1_S = 8.0
COLD_FIRST_TOKEN_BUDGET_M2_S = 5.0
WARM_FIRST_TOKEN_BUDGET_S = 0.5
SOCKET_POLL_INTERVAL_S = 0.05
DAEMON_VACATE_TIMEOUT_S = 10.0

APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "chirp"
MODELS_TOML_PATH = APP_SUPPORT_DIR / "models.toml"
SOCKET_PATH = APP_SUPPORT_DIR / "chirpd.sock"
LOCK_PATH = APP_SUPPORT_DIR / "chirpd.lock"
LOG_FILE = Path.home() / "Library" / "Logs" / "chirp" / "chirpd.log"
DAEMON_SPAWN_WAIT_S = 30.0

BANNER_RE = re.compile(r"chirp\s*›")


def _step(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)


def _ok(message: str) -> None:
    print(f"  PASS: {message}", flush=True)


def _info(message: str) -> None:
    print(f"  {message}", flush=True)


def _fail(message: str) -> str:
    print(f"  FAIL: {message}", flush=True)
    return message


def _alias_for(repo: str) -> str:
    return repo.rsplit("/", 1)[-1].lower().replace("_", "-")


def _hf_cache_dir_for(repo: str) -> Path:
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return base / "hub" / f"models--{repo.replace('/', '--')}"


def _preflight(repo: str) -> str | None:
    if platform.machine() != "arm64":
        return f"requires Apple Silicon (arm64); detected {platform.machine()}"
    cache_dir = _hf_cache_dir_for(repo)
    if not cache_dir.exists():
        return (
            f"model weights not in HF cache at {cache_dir}\n  Run: hf download {repo}"
        )
    return None


def _kill_existing_chirpd() -> int:
    result = subprocess.run(["pgrep", "-f", "/chirpd$"], capture_output=True, text=True)
    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + DAEMON_VACATE_TIMEOUT_S
    while time.monotonic() < deadline:
        still = subprocess.run(
            ["pgrep", "-f", "/chirpd$"], capture_output=True, text=True
        ).stdout.split()
        if not still:
            break
        time.sleep(SOCKET_POLL_INTERVAL_S)
    # Anyone left gets SIGKILL.
    result = subprocess.run(["pgrep", "-f", "/chirpd$"], capture_output=True, text=True)
    stragglers = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    for pid in stragglers:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
    # Lock + socket cleanup so the next spawn can flock cleanly.
    for path in (SOCKET_PATH, LOCK_PATH):
        try:
            path.unlink()
        except FileNotFoundError:
            # Best-effort cleanup: the socket/lock file is already gone.
            pass
        except OSError:
            # Best-effort cleanup: the socket/lock file could not be removed.
            pass
    return len(pids) + len(stragglers)


def _spawn_chirpd() -> subprocess.Popen[bytes]:
    chirpd_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "chirpd"
    if not chirpd_bin.exists():
        found = shutil.which("chirpd")
        if not found:
            raise RuntimeError("chirpd console script not found")
        chirpd_bin = Path(found)
    return subprocess.Popen(
        [str(chirpd_bin)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_for_socket(timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if SOCKET_PATH.is_socket():
            return True
        time.sleep(SOCKET_POLL_INTERVAL_S)
    return False


def _stage_models_toml(repo: str, alias: str) -> Path | None:
    APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup_path: Path | None = None
    if MODELS_TOML_PATH.exists():
        backup_path = MODELS_TOML_PATH.with_suffix(".toml.smoke-3-7-backup")
        shutil.copy2(MODELS_TOML_PATH, backup_path)
        _info(f"backed up existing models.toml → {backup_path}")
    MODELS_TOML_PATH.write_text(
        "schema_version = 1\n"
        f'default_chat = "{alias}"\n'
        "\n"
        f'[models."{alias}"]\n'
        f'hf_repo = "{repo}"\n'
        'role = "chat"\n'
        "options = { temperature = 0.7, max_tokens = 128 }\n"
    )
    return backup_path


def _restore_models_toml(backup_path: Path | None, dropped_existing: bool) -> None:
    if backup_path is not None:
        shutil.move(str(backup_path), MODELS_TOML_PATH)
        return
    if dropped_existing:
        MODELS_TOML_PATH.unlink(missing_ok=True)


@dataclass
class _AskResult:
    exit_code: int
    total_seconds: float
    first_token_seconds: float | None
    streamed_text: str
    full_output: str


def _run_ask(
    question: str,
    *,
    mock_retrieval: bool,
    chirp_bin: Path,
    extra_env: dict[str, str] | None = None,
) -> _AskResult:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if mock_retrieval:
        env["CHIRP_SMOKE_FAKE_CONTEXT"] = "1"

    if mock_retrieval:
        cmd = [
            sys.executable,
            "-c",
            _MOCK_RETRIEVAL_RUNNER,
            question,
        ]
    else:
        cmd = [str(chirp_bin), "ask", "-q", question, "--no-markdown"]

    started_at = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdout is not None

    captured = bytearray()
    first_token_at: float | None = None
    banner_seen = False
    deadline = started_at + ASK_TIMEOUT_S
    while True:
        if time.monotonic() > deadline:
            proc.kill()
            break
        chunk = proc.stdout.read(1)
        if not chunk:
            break
        captured += chunk
        text_so_far = captured.decode("utf-8", errors="replace")
        if not banner_seen and BANNER_RE.search(text_so_far):
            banner_seen = True
            continue
        if banner_seen and first_token_at is None and chunk not in (b"\r", b"\n", b" "):
            first_token_at = time.monotonic() - started_at
    proc.wait(timeout=5)
    total = time.monotonic() - started_at
    output = captured.decode("utf-8", errors="replace")
    streamed = _slice_streamed(output)
    return _AskResult(
        exit_code=proc.returncode,
        total_seconds=total,
        first_token_seconds=first_token_at,
        streamed_text=streamed,
        full_output=output,
    )


def _slice_streamed(output: str) -> str:
    match = BANNER_RE.search(output)
    if match is None:
        return ""
    tail = output[match.end() :]
    sources_idx = tail.find("sources:")
    if sources_idx >= 0:
        tail = tail[:sources_idx]
    return tail.strip()


def _chirp_binary() -> Path:
    candidate = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "chirp"
    if candidate.exists():
        return candidate
    found = shutil.which("chirp")
    if found:
        return Path(found)
    raise RuntimeError(
        "chirp console script not found; install with `uv pip install -e .`"
    )


def _chirpd_pid() -> int | None:
    result = subprocess.run(["pgrep", "-f", "/chirpd$"], capture_output=True, text=True)
    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    return pids[0] if pids else None


def _lsof_network_sockets(pid: int) -> list[str]:
    result = subprocess.run(
        ["lsof", "-a", "-iTCP", "-iUDP", "-P", "-n", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[1:] if lines else []


def _socket_mode() -> str | None:
    if not SOCKET_PATH.exists():
        return None
    result = subprocess.run(
        ["stat", "-f", "%Lp", str(SOCKET_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _legacy_11434_listener_present() -> bool:
    result = subprocess.run(
        ["lsof", "-nP", "-iTCP:11434", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _read_recent_log_lines(question: str) -> list[str]:
    if not LOG_FILE.exists():
        return []
    text = LOG_FILE.read_text(errors="replace").splitlines()
    return [line for line in text[-200:] if "op=chat" in line]


def _log_contains_user_content(question: str, lines: list[str]) -> bool:
    question_tokens = [w for w in question.split() if len(w) >= 4]
    for line in lines:
        for tok in question_tokens:
            if tok.lower() in line.lower():
                return True
    return False


def _budget_for_first_token() -> tuple[float, str]:
    # Treat anything that isn't an explicit M2/M3 chip as M1-class for the budget.
    sysctl = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=False,
    )
    brand = sysctl.stdout.strip().lower()
    if "m2" in brand or "m3" in brand or "m4" in brand:
        return COLD_FIRST_TOKEN_BUDGET_M2_S, brand or "M2-class"
    return COLD_FIRST_TOKEN_BUDGET_M1_S, brand or "M1-class"


_MOCK_RETRIEVAL_RUNNER = """
import sys, os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import notes_chat.cli as cli  # noqa: E402
import notes_chat.retrieval as retrieval  # noqa: E402
import notes_chat.cache as cache  # noqa: E402

class _Settings:
    pass

def _fake_retrieve(config, question, when_filter=None):
    return {
        "success": True,
        "context": "Demo note: the team discussed the Q3 roadmap and budget.",
        "sources": ["note #1 (Smoke fixture)"],
        "retrieved_ids": ["c1"],
    }

cli.get_notes_config = lambda: _Settings()
retrieval.retrieve_context = _fake_retrieve
cache.get_cached_answer = lambda *a, **kw: None
cache.cache_answer = lambda *a, **kw: None

question = sys.argv[1]
sys.argv = ["chirp-smoke", "ask", "-q", question, "--no-markdown"]
try:
    cli.app(standalone_mode=False)
    sys.exit(0)
except SystemExit:
    raise
except Exception as exc:  # noqa: BLE001
    print(f"Query failed: {exc}", file=sys.stderr)
    sys.exit(1)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--mock-retrieval",
        action="store_true",
        help="Bypass chroma retrieval and inject a canned context.",
    )
    parser.add_argument(
        "--keep-models-toml",
        action="store_true",
        help="Don't rewrite the user's existing ~/Library/Application Support/chirp/models.toml.",
    )
    parser.add_argument(
        "--check-idle-unload",
        action="store_true",
        help="After the warm run, wait CHIRP_MODEL_IDLE_TIMEOUT + 5s and confirm the model unloads but the daemon stays up.",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=15,
        help="Seconds to set CHIRP_MODEL_IDLE_TIMEOUT to (only used with --check-idle-unload).",
    )
    parser.add_argument(
        "--warm-budget-s",
        type=float,
        default=None,
        help=(
            "Override the warm-path first-token budget (default: NFR-P1 = 0.5s). "
            "Larger 4B-class chat models on M-series may need a relaxation; "
            "record the override in the PR description as a known performance gap."
        ),
    )
    parser.add_argument(
        "--cold-budget-s",
        type=float,
        default=None,
        help=(
            "Override the cold-path first-token budget (default: NFR-P2 = 5s on "
            "M2-class / 8s on M1)."
        ),
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
    chirp_bin = _chirp_binary()
    _ok(f"chirp binary: {chirp_bin}")

    _step("Ensure no chirpd is running")
    killed = _kill_existing_chirpd()
    if killed:
        _info(f"terminated {killed} pre-existing chirpd process(es)")
    else:
        _ok("no pre-existing chirpd processes")

    _step("Stage models.toml")
    backup_path: Path | None = None
    rewrote_registry = False
    if args.keep_models_toml:
        if not MODELS_TOML_PATH.exists():
            _fail(f"--keep-models-toml set but {MODELS_TOML_PATH} does not exist")
            return 1
        _ok(f"keeping existing registry at {MODELS_TOML_PATH}")
    else:
        backup_path = _stage_models_toml(repo, alias)
        rewrote_registry = True
        _ok(f"wrote registry → {MODELS_TOML_PATH}")

    ask_env: dict[str, str] = {}
    if args.check_idle_unload:
        ask_env["CHIRP_MODEL_IDLE_TIMEOUT"] = str(args.idle_timeout)
        _info(
            f"using CHIRP_MODEL_IDLE_TIMEOUT={args.idle_timeout} for idle-unload check"
        )

    daemon_proc: subprocess.Popen[bytes] | None = None
    try:
        _step("Pre-spawn chirpd (separating spawn from first-token measurement)")
        spawn_env = os.environ.copy()
        spawn_env.update(ask_env)
        chirpd_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "chirpd"
        if not chirpd_bin.exists():
            chirpd_bin = Path(shutil.which("chirpd") or "chirpd")
        spawn_started = time.monotonic()
        daemon_proc = subprocess.Popen(
            [str(chirpd_bin)],
            env=spawn_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if not _wait_for_socket(DAEMON_SPAWN_WAIT_S):
            failures.append(
                _fail(
                    f"chirpd did not bind {SOCKET_PATH} within {DAEMON_SPAWN_WAIT_S}s"
                )
            )
        else:
            spawn_ms = (time.monotonic() - spawn_started) * 1000
            _ok(f"chirpd pid {daemon_proc.pid} bound socket in {spawn_ms:.0f}ms")

        default_cold_budget_s, chip = _budget_for_first_token()
        cold_budget_s = (
            args.cold_budget_s
            if args.cold_budget_s is not None
            else default_cold_budget_s
        )
        cold_budget_label = (
            "model override" if args.cold_budget_s is not None else f"NFR-P2 on {chip}"
        )
        warm_budget_s = (
            args.warm_budget_s
            if args.warm_budget_s is not None
            else WARM_FIRST_TOKEN_BUDGET_S
        )
        warm_budget_label = (
            "model override" if args.warm_budget_s is not None else "NFR-P1"
        )

        _step(
            f"Cold-path: chirp ask (first-token budget ≤ {cold_budget_s:.1f}s, "
            f"{cold_budget_label})"
        )
        cold = _run_ask(
            args.question,
            mock_retrieval=args.mock_retrieval,
            chirp_bin=chirp_bin,
            extra_env=ask_env,
        )
        if cold.exit_code != 0:
            failures.append(
                _fail(
                    f"cold-path chirp ask exited {cold.exit_code}\n"
                    f"    output:\n{_indent(cold.full_output)}"
                )
            )
        else:
            _ok(f"exit 0; total wall-clock {cold.total_seconds:.2f}s")
        if cold.first_token_seconds is None:
            failures.append(_fail("did not observe a streamed token in cold path"))
        else:
            line = (
                f"first-token latency {cold.first_token_seconds * 1000:.0f}ms "
                f"(budget {cold_budget_s * 1000:.0f}ms, {cold_budget_label})"
            )
            if cold.first_token_seconds > cold_budget_s:
                failures.append(_fail(line + " — OVER BUDGET"))
            else:
                _ok(line)
        if cold.streamed_text:
            _info(f"streamed answer: {cold.streamed_text!r}")
        else:
            failures.append(_fail("no streamed answer captured"))

        _step("Daemon stayed up after cold ask")
        pid = _chirpd_pid()
        if pid is None:
            failures.append(_fail("pgrep chirpd returned no PID after cold ask"))
        else:
            _ok(f"chirpd pid {pid}")

        _step(
            f"Warm-path: second chirp ask (first-token budget ≤ "
            f"{warm_budget_s * 1000:.0f}ms, {warm_budget_label})"
        )
        warm = _run_ask(
            args.question,
            mock_retrieval=args.mock_retrieval,
            chirp_bin=chirp_bin,
            extra_env=ask_env,
        )
        if warm.exit_code != 0:
            failures.append(
                _fail(
                    f"warm-path chirp ask exited {warm.exit_code}\n"
                    f"    output:\n{_indent(warm.full_output)}"
                )
            )
        else:
            _ok(f"exit 0; total wall-clock {warm.total_seconds:.2f}s")
        if warm.first_token_seconds is None:
            failures.append(_fail("did not observe a streamed token in warm path"))
        else:
            line = (
                f"first-token latency {warm.first_token_seconds * 1000:.0f}ms "
                f"(budget {warm_budget_s * 1000:.0f}ms, {warm_budget_label})"
            )
            if warm.first_token_seconds > warm_budget_s:
                failures.append(_fail(line + " — OVER BUDGET"))
            else:
                _ok(line)

        _step("AC-14 evidence: SC-8 (no network sockets)")
        chirpd_pid = _chirpd_pid()
        if chirpd_pid is None:
            failures.append(_fail("pgrep chirpd returned no PID for lsof"))
        else:
            rows = _lsof_network_sockets(chirpd_pid)
            if rows:
                failures.append(
                    _fail("lsof reports network sockets:\n    " + "\n    ".join(rows))
                )
            else:
                _ok(f"lsof -iTCP -iUDP for pid {chirpd_pid} returned no rows")

        _step("AC-14 evidence: SC-9 (socket mode 600)")
        mode = _socket_mode()
        if mode is None:
            failures.append(_fail(f"no socket at {SOCKET_PATH}"))
        elif mode != "600":
            failures.append(_fail(f"socket mode is {mode}, expected 600"))
        else:
            _ok(f"{SOCKET_PATH} mode={mode}")

        _step("AC-14 evidence: NFR-S5 (no user content in log)")
        chat_lines = _read_recent_log_lines(args.question)
        if not chat_lines:
            failures.append(_fail("no op=chat lines found in chirpd.log"))
        else:
            _ok(f"found {len(chat_lines)} op=chat line(s) in {LOG_FILE}")
            for line in chat_lines[-3:]:
                _info(line)
            if _log_contains_user_content(args.question, chat_lines):
                failures.append(
                    _fail(
                        "log line appears to contain words from the user's question "
                        "— privacy invariant violated"
                    )
                )
            else:
                _ok("log lines contain no words from the user's question")

        _step("AC-10 step 6: no stray :11434 listener required")
        if _legacy_11434_listener_present():
            _info(
                "lsof shows a listener on :11434 (a leftover from a pre-migration "
                "setup — chirp no longer uses it; harmless)"
            )
        else:
            _info("no :11434 listener detected (expected post-migration)")

        if args.check_idle_unload:
            _step(
                f"AC-10 step 10: idle-unload after {args.idle_timeout}s "
                "(daemon stays up, model unloads)"
            )
            wait = args.idle_timeout + 5
            _info(f"sleeping {wait}s...")
            time.sleep(wait)
            still_pid = _chirpd_pid()
            if still_pid is None:
                failures.append(_fail("chirpd exited during idle window"))
            else:
                _ok(f"chirpd pid {still_pid} still running")
            try:
                from llm.client import LLMClient

                status = LLMClient().model_status_sync()
                loaded = [m["alias"] for m in status.get("models", [])]
                if alias in loaded:
                    failures.append(
                        _fail(f"model still loaded after idle window: {loaded}")
                    )
                else:
                    _ok(f"model unloaded; status.models={loaded}")
            except Exception as exc:  # noqa: BLE001
                failures.append(_fail(f"model.status query failed: {exc}"))

    finally:
        _step("Teardown")
        if _chirpd_pid() is not None:
            killed = _kill_existing_chirpd()
            _info(f"terminated chirpd ({killed} process)")
        _restore_models_toml(backup_path, rewrote_registry)
        _ok("models.toml restored")
        if LOG_FILE.exists():
            _info(f"daemon log: {LOG_FILE}")

    _step("Summary")
    if failures:
        for line in failures:
            print(f"  - {line}", flush=True)
        print(f"\n{len(failures)} check(s) failed.", flush=True)
        return 1
    print("\nAll checks passed.", flush=True)
    return 0


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    sys.exit(main())
