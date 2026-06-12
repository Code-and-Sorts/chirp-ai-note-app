"""Unit tests for the chirp init flow.

The external pieces (homebrew, the chirpd daemon, the model registry, audio
permission probes, subprocess calls) are mocked out so the tests stay fast,
platform-agnostic, and never spawn a real daemon or write a real models.toml.
"""

from io import StringIO

from rich.console import Console

from chirp import init_flow
from config.settings import ChirpSettings
from llm.exceptions import LLMDaemonSpawnFailed, LLMDaemonUnreachable
from llm.registry import Registry, RegistryEntry


def _fake_settings(tmp_path):
    settings = ChirpSettings()
    settings.directories.notes_root = tmp_path / "notes"
    settings.notes_chat.index_dir = tmp_path / "index"
    return settings


def _console() -> Console:
    return Console(file=StringIO(), width=120, force_terminal=False)


def _registry_with_default(default_chat="gemma-4-4b-it-4bit", extra_aliases=()):
    models = {
        default_chat: RegistryEntry(
            hf_repo=f"mlx-community/{default_chat}", role="chat"
        ),
    }
    for alias in extra_aliases:
        models[alias] = RegistryEntry(hf_repo=f"mlx-community/{alias}", role="chat")
    return Registry(schema_version=1, default_chat=default_chat, models=models)


def _empty_registry():
    return Registry(schema_version=1, models={})


def _stub_healthy_daemon(monkeypatch):
    monkeypatch.setattr(
        "llm.client.LLMClient.health_sync",
        lambda self, **kwargs: {"event": "ready", "status": "ok", "version": "0.1.0"},
    )


def _stub_verify_deps(monkeypatch, registry=None, arm64=True):
    """Stub every external probe for a healthy Apple Silicon box."""
    monkeypatch.setattr(
        init_flow.platform, "machine", lambda: "arm64" if arm64 else "x86_64"
    )
    monkeypatch.setattr(init_flow, "_which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(
        init_flow,
        "_brew_installed",
        lambda: init_flow.DependencyStatus("homebrew", True, "/usr/bin/brew"),
    )
    monkeypatch.setattr(
        init_flow,
        "_ffmpeg_installed",
        lambda: init_flow.DependencyStatus("ffmpeg", True, "7.1.1"),
    )
    _stub_healthy_daemon(monkeypatch)
    monkeypatch.setattr(
        "llm.registry.read_registry",
        lambda path=None: (
            registry if registry is not None else _registry_with_default()
        ),
    )


# --- Apple-Silicon gate (AC-1) ----------------------------------------------


def test_run_init_fails_fast_on_intel(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "machine", lambda: "x86_64")

    health_calls = []
    monkeypatch.setattr(
        "llm.client.LLMClient.health_sync",
        lambda self, **kwargs: health_calls.append(True) or {},
    )

    console = _console()
    settings = _fake_settings(tmp_path)

    code = init_flow.run_init(settings, console)

    assert code == 7
    output = console.file.getvalue()
    assert "Apple Silicon" in output
    assert "x86_64" in output
    assert "homebrew" not in output  # no verify table printed
    assert health_calls == []  # daemon never lazy-spawned
    assert not settings.directories.notes_root.exists()  # no filesystem mutation
    assert not settings.notes_chat.index_dir.exists()


def test_run_init_apple_silicon_passes_through(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        init_flow.ChirpSettings,
        "get_config_path",
        lambda: tmp_path / "config.toml",
    )

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console, recheck=True)

    assert code == 0
    output = console.file.getvalue()
    assert "checking what you've already got" in output
    assert "homebrew" in output


# --- verify() rows (AC-2, AC-3, AC-4) ----------------------------------------


def test_verify_includes_daemon_and_registry_rows_no_ollama(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    names = [s.name for s in statuses]
    assert names == [
        "homebrew",
        "ffmpeg",
        "chirpd",
        "default chat model",
        "screen recording permission",
    ]
    assert not any(n == "Ollama" or n.startswith("model:") for n in names)

    chirpd = statuses[2]
    assert chirpd.installed is True
    assert chirpd.detail == "healthy · v0.1.0"

    chat = statuses[3]
    assert chat.installed is True
    assert chat.detail == "default chat: gemma-4-4b-it-4bit"


def test_verify_handles_daemon_spawn_failure(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    def _spawn_failed(self, **kwargs):
        raise LLMDaemonSpawnFailed("spawn timed out")

    monkeypatch.setattr("llm.client.LLMClient.health_sync", _spawn_failed)

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    chirpd = next(s for s in statuses if s.name == "chirpd")
    assert chirpd.installed is False
    assert chirpd.required is True
    assert "chirp daemon logs" in chirpd.detail


def test_verify_handles_daemon_unreachable(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    def _unreachable(self, **kwargs):
        raise LLMDaemonUnreachable("socket refused connection")

    monkeypatch.setattr("llm.client.LLMClient.health_sync", _unreachable)

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    chirpd = next(s for s in statuses if s.name == "chirpd")
    assert chirpd.installed is False
    assert "daemon unreachable" in chirpd.detail
    assert "socket refused connection" in chirpd.detail


def test_verify_handles_empty_registry(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch, registry=_empty_registry())
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    chat = next(s for s in statuses if s.name == "default chat model")
    assert chat.installed is False
    assert chat.required is False
    assert "chirp models add mlx-community/gemma-4-4b-it-4bit" in chat.detail


def test_verify_handles_missing_registry_file(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    def _missing(path=None):
        raise FileNotFoundError("models.toml absent")

    monkeypatch.setattr("llm.registry.read_registry", _missing)

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    chat = next(s for s in statuses if s.name == "default chat model")
    assert chat.installed is False
    assert chat.required is False
    assert "chirp models add" in chat.detail


def test_run_init_prints_models_add_hint_when_registry_empty(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch, registry=_empty_registry())
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        init_flow.ChirpSettings,
        "get_config_path",
        lambda: tmp_path / "config.toml",
    )

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console)

    assert code == 0
    output = console.file.getvalue()
    assert "chirp models add mlx-community/gemma-4-4b-it-4bit" in output
    assert "ollama" not in output.lower()


# --- --switch-model (AC-7) ----------------------------------------------------


def test_switch_model_with_empty_registry_prints_models_add_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        "llm.registry.read_registry", lambda path=None: _empty_registry()
    )

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console, switch_model=True)

    assert code == 0
    output = console.file.getvalue()
    assert "chirp models add mlx-community/gemma-4-4b-it-4bit" in output
    assert "pick" not in output.lower()  # no tag picker shown


def test_switch_model_with_registered_aliases_calls_set_default(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")
    registry = _registry_with_default(
        "gemma-4-4b-it-4bit", extra_aliases=["qwen2.5-7b-instruct-4bit"]
    )
    monkeypatch.setattr("llm.registry.read_registry", lambda path=None: registry)

    written = []
    monkeypatch.setattr(
        "llm.registry.write_registry", lambda reg, path=None: written.append(reg)
    )

    console = _console()
    # Aliases listed sorted: 1=gemma..., 2=qwen... — pick the second.
    monkeypatch.setattr(console, "input", lambda *args, **kwargs: "2")

    code = init_flow.run_init(_fake_settings(tmp_path), console, switch_model=True)

    assert code == 0
    # The real set_default_for_role ran; the written registry proves the flip.
    assert len(written) == 1
    assert written[0].default_chat == "qwen2.5-7b-instruct-4bit"


def test_switch_model_user_abort_returns_one(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        "llm.registry.read_registry", lambda path=None: _registry_with_default()
    )

    console = _console()

    def _interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(console, "input", _interrupt)

    code = init_flow.run_init(_fake_settings(tmp_path), console, switch_model=True)
    assert code == 1


# --- install_missing (AC-5) ---------------------------------------------------


def test_install_missing_surfaces_daemon_failure_without_installing(monkeypatch):
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(init_flow, "_which", lambda cmd: f"/usr/bin/{cmd}")

    run_calls: list[list[str]] = []

    def _fake_run(args, timeout=10.0):
        run_calls.append(list(args))
        return 0, ""

    monkeypatch.setattr(init_flow, "_run", _fake_run)

    statuses = [
        init_flow.DependencyStatus(
            "chirpd",
            False,
            "daemon could not be started — run 'chirp daemon logs' for details",
        )
    ]

    console = _console()
    result = init_flow.install_missing(console, statuses)

    assert result is False
    assert run_calls == []  # nothing brew-installable for the daemon
    output = console.file.getvalue()
    assert "chirp daemon logs" in output


def test_install_missing_dispatches_build_for_missing_binary(tmp_path, monkeypatch):
    import sys as _sys

    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(init_flow, "_which", lambda cmd: f"/usr/bin/{cmd}")

    run_calls: list[list[str]] = []

    def _fake_run(args, timeout=10.0):
        run_calls.append(list(args))
        return 0, ""

    monkeypatch.setattr(init_flow, "_run", _fake_run)

    statuses = [
        init_flow.DependencyStatus(
            name="screen recording permission",
            installed=False,
            detail="capture_audio binary not built — run python -m audio_capture.build",
        )
    ]

    init_flow.install_missing(_console(), statuses)

    assert len(run_calls) == 1, f"expected exactly one dispatch, got: {run_calls}"
    args = run_calls[0]
    assert _sys.executable in args and "-m" in args and "audio_capture.build" in args, (
        f"expected audio_capture.build dispatch, got: {args}"
    )


def test_install_missing_surfaces_denied_permission_without_dispatching(monkeypatch):
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(init_flow, "_which", lambda cmd: f"/usr/bin/{cmd}")

    run_calls: list[list[str]] = []

    def _fake_run(args, timeout=10.0):
        run_calls.append(list(args))
        return 0, ""

    monkeypatch.setattr(init_flow, "_run", _fake_run)

    statuses = [
        init_flow.DependencyStatus(
            name="screen recording permission",
            installed=False,
            detail="denied — open System Settings → Privacy & Security → Screen Recording",
        )
    ]

    console = _console()
    result = init_flow.install_missing(console, statuses)

    assert len(run_calls) == 0, f"expected zero dispatches, got: {run_calls}"
    assert result is False
    output = console.file.getvalue()
    assert "screen recording permission must be granted manually" in output
    assert "init incomplete" in output


# --- run_init orchestration (AC-6) ---------------------------------------------


def test_run_init_recheck_short_circuits(tmp_path, monkeypatch):
    """--recheck returns 0 after the verify table without install/finalize."""
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    finalize_calls = []
    monkeypatch.setattr(
        init_flow, "_finalize_paths", lambda *a, **k: finalize_calls.append(True)
    )

    code = init_flow.run_init(_fake_settings(tmp_path), _console(), recheck=True)

    assert code == 0
    assert finalize_calls == []


def test_run_init_phases_run_in_order(tmp_path, monkeypatch):
    """Smoke: run_init walks gate → verify → install → finalize."""
    calls: list[str] = []

    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        init_flow,
        "verify",
        lambda settings, console: (
            calls.append("verify")
            or [init_flow.DependencyStatus("homebrew", False, "missing")]
        ),
    )
    monkeypatch.setattr(init_flow, "_confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        init_flow,
        "install_missing",
        lambda console, statuses: calls.append("install") or True,
    )
    monkeypatch.setattr(
        init_flow,
        "_finalize_paths",
        lambda *args, **kwargs: calls.append("finalize"),
    )

    code = init_flow.run_init(_fake_settings(tmp_path), _console())
    assert code == 0
    assert calls == ["verify", "install", "finalize"]


def test_run_init_rerun_is_idempotent_and_preserves_user_keys(tmp_path, monkeypatch):
    """AC-11: run twice in a tmp HOME, assert second run preserves a hand-edited key."""
    config_path = tmp_path / "config.toml"

    settings = _fake_settings(tmp_path)
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    monkeypatch.setattr(init_flow.ChirpSettings, "get_config_path", lambda: config_path)

    code = init_flow.run_init(settings, _console())
    assert code == 0
    assert config_path.exists()

    import tomllib

    import tomli_w

    with config_path.open("rb") as fh:
        first = tomllib.load(fh)
    first["user_custom"] = {"theme": "midnight"}
    with config_path.open("wb") as fh:
        tomli_w.dump(first, fh)

    first_mtime = config_path.stat().st_mtime
    import time as _time

    _time.sleep(0.01)

    code = init_flow.run_init(settings, _console())
    assert code == 0
    assert config_path.stat().st_mtime == first_mtime  # existing config untouched

    with config_path.open("rb") as fh:
        second = tomllib.load(fh)
    assert second["user_custom"] == {"theme": "midnight"}


def test_merge_config_backs_up_corrupt_file(tmp_path):
    """A corrupt config.toml is renamed to .bak-<ts> before fresh write."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("this is { not valid toml\n", encoding="utf-8")

    console = _console()
    init_flow._merge_config(config_path, console=console)

    backups = list(tmp_path.glob("config.toml.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "this is { not valid toml\n"

    output = console.file.getvalue()
    assert "could not be parsed" in output
    assert config_path.exists()


# --- screen-recording permission rows (unchanged behavior from story 2.3) -----


def test_verify_includes_screen_recording_permission_last(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(
        init_flow.audio_capture,
        "check_permissions",
        lambda: {"screen_recording": "granted", "microphone": "granted"},
    )
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    assert statuses[-1].name == "screen recording permission"
    assert statuses[-1].installed is True
    assert statuses[-1].detail == "granted"


def test_verify_handles_denied_screen_recording(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(
        init_flow.audio_capture,
        "check_permissions",
        lambda: {"screen_recording": "denied", "microphone": "granted"},
    )
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    perm = next(s for s in statuses if s.name == "screen recording permission")
    assert perm.installed is False
    assert (
        "denied — open System Settings → Privacy & Security → Screen Recording"
        in perm.detail
    )


def test_verify_handles_missing_binary(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)

    def _raise():
        raise FileNotFoundError(
            "capture_audio binary not found. Build it with: python -m audio_capture.build"
        )

    monkeypatch.setattr(init_flow.audio_capture, "check_permissions", _raise)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    perm = next(s for s in statuses if s.name == "screen recording permission")
    assert perm.installed is False
    assert (
        "capture_audio binary not built — run python -m audio_capture.build"
        in perm.detail
    )


def test_verify_screen_recording_skipped_on_non_darwin(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)

    called = []

    def _track():
        called.append(True)
        return {}

    monkeypatch.setattr(init_flow.audio_capture, "check_permissions", _track)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    perm = next(s for s in statuses if s.name == "screen recording permission")
    assert perm.installed is True
    assert perm.required is False
    assert "not applicable on Linux" in perm.detail
    assert called == [], "check_permissions must not be called on non-Darwin"


def test_verify_handles_undetermined_screen_recording(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(
        init_flow.audio_capture,
        "check_permissions",
        lambda: {"screen_recording": "undetermined", "microphone": "granted"},
    )
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    perm = next(s for s in statuses if s.name == "screen recording permission")
    assert perm.installed is True
    assert perm.required is False
    assert perm.detail == "will prompt on first record"


def test_verify_handles_runtime_error_from_check_permissions(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)

    def _raise():
        raise RuntimeError("malformed permission line in helper output: 'garbage'")

    monkeypatch.setattr(init_flow.audio_capture, "check_permissions", _raise)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    perm = next(s for s in statuses if s.name == "screen recording permission")
    assert perm.installed is False
    assert "permission probe failed" in perm.detail
    assert "malformed permission line in helper output: 'garbage'" in perm.detail


def test_verify_guards_against_unexpected_permission_state(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(
        init_flow.audio_capture,
        "check_permissions",
        lambda: {"screen_recording": "BUSTED", "microphone": "granted"},
    )
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    perm = next(s for s in statuses if s.name == "screen recording permission")
    assert perm.installed is False
    assert "unexpected permission state" in perm.detail
    assert "BUSTED" in perm.detail


# --- helper-level coverage ------------------------------------------------------


def test_run_returns_127_on_missing_binary():
    code, out = init_flow._run(["/nonexistent/definitely-not-a-binary"])
    assert code == 127
    assert out


def test_brew_missing_is_required_on_darwin(monkeypatch):
    monkeypatch.setattr(init_flow, "_which", lambda _cmd: None)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")

    status = init_flow._brew_installed()

    assert status.installed is False
    assert status.required is True
    assert "brew.sh" in status.detail


def test_ffmpeg_version_parsed_from_output(monkeypatch):
    monkeypatch.setattr(init_flow, "_which", lambda _cmd: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        init_flow, "_run", lambda args, timeout=10.0: (0, "ffmpeg version 7.1.1 etc\n")
    )

    status = init_flow._ffmpeg_installed()

    assert status.installed is True
    assert status.detail == "7.1.1"


def test_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(init_flow, "_which", lambda _cmd: None)
    assert init_flow._ffmpeg_installed().installed is False


def test_confirm_answers(monkeypatch):
    console = _console()

    monkeypatch.setattr(console, "input", lambda *a, **k: "")
    assert init_flow._confirm(console, "go?") is True

    monkeypatch.setattr(console, "input", lambda *a, **k: "n")
    assert init_flow._confirm(console, "go?") is False

    monkeypatch.setattr(console, "input", lambda *a, **k: "yes")
    assert init_flow._confirm(console, "go?", default=False) is True

    def _interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(console, "input", _interrupt)
    assert init_flow._confirm(console, "go?") is False


def test_install_missing_is_macos_only(monkeypatch):
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    assert init_flow.install_missing(_console(), []) is False


def test_install_missing_requires_brew(monkeypatch):
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(init_flow, "_which", lambda _cmd: None)
    assert init_flow.install_missing(_console(), []) is False


def test_install_missing_brew_installs_ffmpeg(monkeypatch):
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(init_flow, "_which", lambda cmd: f"/usr/bin/{cmd}")

    run_calls: list[list[str]] = []
    monkeypatch.setattr(
        init_flow,
        "_run",
        lambda args, timeout=10.0: run_calls.append(list(args)) or (0, ""),
    )

    statuses = [init_flow.DependencyStatus("ffmpeg", False, "not found")]
    assert init_flow.install_missing(_console(), statuses) is True
    assert run_calls == [["/usr/bin/brew", "install", "ffmpeg"]]


def test_install_missing_surfaces_brew_failure(monkeypatch):
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(init_flow, "_which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(
        init_flow, "_run", lambda args, timeout=10.0: (1, "Error: install exploded")
    )

    statuses = [init_flow.DependencyStatus("ffmpeg", False, "not found")]
    console = _console()
    assert init_flow.install_missing(console, statuses) is False
    assert "install exploded" in console.file.getvalue()


def test_switch_model_blank_input_keeps_current(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        "llm.registry.read_registry", lambda path=None: _registry_with_default()
    )
    written = []
    monkeypatch.setattr(
        "llm.registry.write_registry", lambda reg, path=None: written.append(reg)
    )

    console = _console()
    monkeypatch.setattr(console, "input", lambda *a, **k: "")

    code = init_flow.run_init(_fake_settings(tmp_path), console, switch_model=True)
    assert code == 0
    assert written == []


def test_switch_model_invalid_selection_returns_one(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        "llm.registry.read_registry", lambda path=None: _registry_with_default()
    )

    console = _console()
    monkeypatch.setattr(console, "input", lambda *a, **k: "99")

    code = init_flow.run_init(_fake_settings(tmp_path), console, switch_model=True)
    assert code == 1
    assert "invalid selection" in console.file.getvalue()


def test_run_init_user_declines_install(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        init_flow,
        "verify",
        lambda settings, console: [
            init_flow.DependencyStatus("ffmpeg", False, "not found")
        ],
    )
    monkeypatch.setattr(init_flow, "_confirm", lambda *a, **k: False)

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console)

    assert code == 1
    assert "skipped install" in console.file.getvalue()


def test_finalize_paths_preserves_existing_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[user_custom]\ntheme = "midnight"\n', encoding="utf-8")
    original_mtime = config_path.stat().st_mtime

    settings = _fake_settings(tmp_path)
    monkeypatch.setattr(init_flow.ChirpSettings, "get_config_path", lambda: config_path)
    monkeypatch.setattr(init_flow, "_offer_launch_agent", lambda *a, **k: None)

    init_flow._finalize_paths(settings, _console())

    assert config_path.stat().st_mtime == original_mtime
    assert (settings.notes_chat.index_dir / "chroma").exists()
    assert settings.directories.notes_root.exists()


# --- --recheck Ollama migration plan (story 7.2) ---------------------------------


def _stub_detection(monkeypatch, tmp_path, cli=False, data_dir=False, module=False):
    """Control the three _detect_ollama_install heuristics."""
    monkeypatch.setattr(
        init_flow,
        "_which",
        lambda cmd: (
            ("/opt/homebrew/bin/ollama" if cli else None)
            if cmd == "ollama"
            else f"/usr/bin/{cmd}"
        ),
    )
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    if data_dir:
        (fake_home / ".ollama").mkdir()
    monkeypatch.setattr(init_flow.Path, "home", lambda: fake_home)
    monkeypatch.setattr(init_flow, "_try_import_ollama_module", lambda: module)


def test_recheck_with_no_ollama_detected_prints_no_plan(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    _stub_detection(monkeypatch, tmp_path)

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console, recheck=True)

    assert code == 0
    output = console.file.getvalue()
    assert "checking what you've already got" in output  # verify table printed
    assert "Ollama migration" not in output


def test_recheck_with_ollama_cli_on_path_prints_plan(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    _stub_detection(monkeypatch, tmp_path, cli=True)

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console, recheck=True)

    assert code == 0
    output = console.file.getvalue()
    assert "Ollama migration" in output
    assert f"chirp models add {init_flow.RECOMMENDED_CHAT_REPO}" in output
    assert "[ollama on PATH]            yes" in output
    # The plan prints after the verify table.
    assert output.index("checking what you've already got") < output.index(
        "Ollama migration"
    )


def test_recheck_with_data_dir_only_prints_plan(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    _stub_detection(monkeypatch, tmp_path, data_dir=True)

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console, recheck=True)

    assert code == 0
    output = console.file.getvalue()
    assert "Ollama migration" in output
    assert "[~/.ollama directory]       yes" in output
    assert "[ollama on PATH]            no" in output
    assert "[ollama python module]      no" in output


def test_recheck_with_python_module_importable_prints_plan(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    _stub_detection(monkeypatch, tmp_path, module=True)

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console, recheck=True)

    assert code == 0
    output = console.file.getvalue()
    assert "Ollama migration" in output
    assert "[ollama python module]      yes" in output


def test_recheck_plan_does_not_mutate_ollama_install(tmp_path, monkeypatch):
    from unittest.mock import Mock

    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    _stub_detection(monkeypatch, tmp_path, cli=True, data_dir=True, module=True)
    monkeypatch.setattr(
        init_flow.subprocess,
        "run",
        Mock(side_effect=AssertionError("must not call subprocess during recheck")),
    )

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console, recheck=True)

    assert code == 0
    output = console.file.getvalue()
    assert "Ollama migration" in output
    assert "chirp will NOT do this for you" in output
    # All three heuristics fired and were reported.
    assert output.count("yes") >= 3


def test_full_init_no_recheck_does_not_print_plan(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    _stub_detection(monkeypatch, tmp_path, cli=True, data_dir=True, module=True)
    monkeypatch.setattr(
        init_flow.ChirpSettings,
        "get_config_path",
        lambda: tmp_path / "config.toml",
    )

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console)

    assert code == 0
    assert "Ollama migration" not in console.file.getvalue()


def test_plan_recommends_default_chat_repo(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    _stub_detection(monkeypatch, tmp_path, cli=True)

    console = _console()
    init_flow.run_init(_fake_settings(tmp_path), console, recheck=True)

    output = console.file.getvalue()
    assert "mlx-community/gemma-4-4b-it-4bit" in output
    assert init_flow.SMALLER_CHAT_REPO in output


# --- LaunchAgent offer (story 7.3) -----------------------------------------------


def _launch_agent_env(monkeypatch, tmp_path, installed=False, answer=None):
    """Darwin + mocked launchd + tmp config path; returns the install recorder."""
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")
    monkeypatch.setattr("chirpd.launchd.is_launch_agent_installed", lambda: installed)
    install_calls = []
    monkeypatch.setattr(
        "chirpd.launchd.install_launch_agent",
        lambda **kwargs: install_calls.append(kwargs) or (tmp_path / "plist"),
    )
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(init_flow.ChirpSettings, "get_config_path", lambda: config_path)
    console = _console()
    prompt_calls = []

    def _input(*args, **kwargs):
        prompt_calls.append(True)
        if answer is None:
            raise AssertionError("prompt must not be shown")
        return answer

    monkeypatch.setattr(console, "input", _input)
    return console, install_calls, prompt_calls, config_path


def _persisted_timestamp(config_path):
    import tomllib

    with config_path.open("rb") as fh:
        return tomllib.load(fh).get("init", {}).get("launch_agent_prompted_at")


def test_launch_agent_prompt_default_no_skips_install(tmp_path, monkeypatch):
    console, install_calls, prompt_calls, config_path = _launch_agent_env(
        monkeypatch, tmp_path, answer=""
    )
    settings = _fake_settings(tmp_path)

    init_flow._offer_launch_agent(settings, console)

    assert prompt_calls == [True]
    assert install_calls == []
    assert "launch agent skipped" in console.file.getvalue()
    assert _persisted_timestamp(config_path) is not None
    assert settings.init.launch_agent_prompted_at is not None


def test_launch_agent_prompt_yes_installs(tmp_path, monkeypatch):
    console, install_calls, prompt_calls, config_path = _launch_agent_env(
        monkeypatch, tmp_path, answer="y"
    )
    settings = _fake_settings(tmp_path)

    init_flow._offer_launch_agent(settings, console)

    assert len(install_calls) == 1
    assert "LaunchAgent installed" in console.file.getvalue()
    assert _persisted_timestamp(config_path) is not None


def test_launch_agent_prompt_install_failure_persists_anyway(tmp_path, monkeypatch):
    console, _install_calls, _prompt_calls, config_path = _launch_agent_env(
        monkeypatch, tmp_path, answer="y"
    )

    def _boom(**kwargs):
        raise RuntimeError("launchctl load failed")

    monkeypatch.setattr("chirpd.launchd.install_launch_agent", _boom)
    settings = _fake_settings(tmp_path)

    init_flow._offer_launch_agent(settings, console)

    output = console.file.getvalue()
    assert "LaunchAgent install failed: launchctl load failed" in output
    assert "chirp daemon enable" in output
    assert _persisted_timestamp(config_path) is not None


def test_launch_agent_prompt_suppressed_when_already_asked(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    console, install_calls, prompt_calls, _config_path = _launch_agent_env(
        monkeypatch, tmp_path, answer=None
    )
    settings = _fake_settings(tmp_path)
    settings.init.launch_agent_prompted_at = datetime(2026, 1, 1, tzinfo=UTC)

    init_flow._offer_launch_agent(settings, console)

    assert prompt_calls == []
    assert install_calls == []


def test_launch_agent_prompt_recheck_revisits_when_not_installed(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    console, _install_calls, prompt_calls, _config_path = _launch_agent_env(
        monkeypatch, tmp_path, answer=""
    )
    settings = _fake_settings(tmp_path)
    settings.init.launch_agent_prompted_at = datetime(2026, 1, 1, tzinfo=UTC)

    init_flow._offer_launch_agent(settings, console, force_prompt=True)

    assert prompt_calls == [True]


def test_launch_agent_prompt_skipped_when_agent_already_installed(
    tmp_path, monkeypatch
):
    console, install_calls, prompt_calls, config_path = _launch_agent_env(
        monkeypatch, tmp_path, installed=True, answer=None
    )
    settings = _fake_settings(tmp_path)

    init_flow._offer_launch_agent(settings, console)

    assert prompt_calls == []
    assert install_calls == []
    assert "LaunchAgent" not in console.file.getvalue()  # silent skip
    # User installed via `chirp daemon enable` elsewhere — mark as prompted.
    assert _persisted_timestamp(config_path) is not None


def test_launch_agent_prompt_not_run_on_switch_model(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        "llm.registry.read_registry", lambda path=None: _empty_registry()
    )
    offers = []
    monkeypatch.setattr(
        init_flow, "_offer_launch_agent", lambda *a, **k: offers.append(True)
    )

    code = init_flow.run_init(_fake_settings(tmp_path), _console(), switch_model=True)

    assert code == 0
    assert offers == []


def test_launch_agent_offer_is_noop_off_darwin(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    probe_calls = []
    monkeypatch.setattr(
        "chirpd.launchd.is_launch_agent_installed",
        lambda: probe_calls.append(True) or False,
    )

    init_flow._offer_launch_agent(_fake_settings(tmp_path), _console())

    assert probe_calls == []


def test_recheck_offers_launch_agent_with_force_prompt(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")
    _stub_detection(monkeypatch, tmp_path)

    offers = []
    monkeypatch.setattr(
        init_flow,
        "_offer_launch_agent",
        lambda settings, console, force_prompt=False: offers.append(force_prompt),
    )

    code = init_flow.run_init(_fake_settings(tmp_path), _console(), recheck=True)

    assert code == 0
    assert offers == [True]


def test_verify_surfaces_malformed_registry(tmp_path, monkeypatch):
    from llm.exceptions import LLMMalformedResponse

    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Linux")

    def _corrupt(path=None):
        raise LLMMalformedResponse("models.toml is not valid TOML")

    monkeypatch.setattr("llm.registry.read_registry", _corrupt)

    statuses = init_flow.verify(_fake_settings(tmp_path), _console())

    chat = next(s for s in statuses if s.name == "default chat model")
    assert chat.installed is False
    assert "registry unreadable" in chat.detail
    assert "not valid TOML" in chat.detail
    assert "chirp models add" not in chat.detail  # not the fresh-install hint


def test_switch_model_surfaces_malformed_registry(tmp_path, monkeypatch):
    from llm.exceptions import LLMMalformedResponse

    monkeypatch.setattr(init_flow.platform, "machine", lambda: "arm64")

    def _corrupt(path=None):
        raise LLMMalformedResponse("models.toml is not valid TOML")

    monkeypatch.setattr("llm.registry.read_registry", _corrupt)

    console = _console()
    code = init_flow.run_init(_fake_settings(tmp_path), console, switch_model=True)

    assert code == 1
    output = console.file.getvalue()
    assert "registry unreadable" in output
    assert "chirp models add" not in output


def test_merge_config_replaces_non_table_section_value(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('init = "oops"\n[user_custom]\ntheme = "midnight"\n')

    init_flow._merge_config(config_path, updates={"init": {"flag": True}})

    import tomllib

    with config_path.open("rb") as fh:
        merged = tomllib.load(fh)
    assert merged["init"] == {"flag": True}
    assert merged["user_custom"] == {"theme": "midnight"}
