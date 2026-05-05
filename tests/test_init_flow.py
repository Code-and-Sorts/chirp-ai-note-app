"""Unit tests for the chirp init flow.

The external pieces (homebrew, ollama, audio midi setup, subprocess calls)
are mocked out so the tests stay fast and platform-agnostic.
"""

from io import StringIO

from rich.console import Console

from chirp import init_flow
from config.settings import ChirpSettings


def _fake_settings(tmp_path):
    settings = ChirpSettings()
    settings.directories.notes_root = tmp_path / "notes"
    settings.notes_chat.index_dir = tmp_path / "index"
    return settings


def _console() -> Console:
    return Console(file=StringIO(), width=120, force_terminal=False)


def test_parse_percent_reads_ollama_progress():
    assert init_flow._parse_percent("pulling manifest 12%") == 12.0
    assert init_flow._parse_percent("digest 99.9% done") == 99.9
    assert init_flow._parse_percent("") is None
    assert init_flow._parse_percent("done") is None


def test_model_installed_matches_latest_and_base():
    assert init_flow._model_installed("llama3.1:8b", ["llama3.1:8b"])
    assert init_flow._model_installed("llama3.1:8b", ["llama3.1:8b:latest"])
    assert init_flow._model_installed("nomic-embed-text", ["nomic-embed-text:v1.5"])
    assert not init_flow._model_installed("llama3.1:8b", ["qwen2.5:7b"])


def test_verify_reports_missing_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(init_flow, "_which", lambda _cmd: None)
    monkeypatch.setattr(init_flow, "_ollama_models", lambda: [])

    console = _console()
    settings = _fake_settings(tmp_path)

    statuses = init_flow.verify(settings, console)

    names = [s.name for s in statuses]
    assert "homebrew" in names
    assert "ffmpeg" in names
    assert "Ollama" in names
    missing_required = [s for s in statuses if s.required and not s.installed]
    assert missing_required, "verify should report missing pieces"


def test_run_init_recheck_short_circuits(tmp_path, monkeypatch):
    """--recheck returns 0 without prompting / installing."""
    monkeypatch.setattr(init_flow, "_which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(
        init_flow,
        "_ollama_installed",
        lambda: init_flow.DependencyStatus("Ollama", True, "running · 0.1"),
    )
    monkeypatch.setattr(
        init_flow, "_ollama_models", lambda: ["llama3.1:8b", "nomic-embed-text"]
    )
    monkeypatch.setattr(
        init_flow,
        "_ffmpeg_installed",
        lambda: init_flow.DependencyStatus("ffmpeg", True, "7.1.1"),
    )

    console = _console()
    settings = _fake_settings(tmp_path)

    code = init_flow.run_init(settings, console, recheck=True)
    assert code == 0


def test_pick_models_defaults_on_blank_input(monkeypatch):
    console = _console()
    answers = iter(["", ""])
    monkeypatch.setattr(console, "input", lambda *args, **kwargs: next(answers))

    chat, embed = init_flow.pick_models(console)

    assert chat == "llama3.1:8b"
    assert embed == "nomic-embed-text"


def test_pick_models_honours_numeric_selection(monkeypatch):
    console = _console()
    answers = iter(["2", "3"])
    monkeypatch.setattr(console, "input", lambda *args, **kwargs: next(answers))

    chat, embed = init_flow.pick_models(console)

    assert chat == init_flow.CHAT_MODELS[1].tag
    assert embed == init_flow.EMBEDDING_MODELS[2].tag


def test_run_init_phases_run_in_order(tmp_path, monkeypatch):
    """Smoke: run_init walks verify → install → keep_or_pick → finalize."""
    calls: list[str] = []

    monkeypatch.setattr(
        init_flow,
        "verify",
        lambda settings, console: (
            calls.append("verify")
            or [
                init_flow.DependencyStatus("homebrew", False, "missing"),
            ]
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
        "_ollama_installed",
        lambda: init_flow.DependencyStatus("Ollama", True, "running · 0.1"),
    )
    monkeypatch.setattr(
        init_flow,
        "keep_or_pick",
        lambda console, settings: (
            calls.append("keep_or_pick") or ("llama3.1:8b", "nomic-embed-text", True)
        ),
    )
    monkeypatch.setattr(
        init_flow,
        "pull_and_finalize",
        lambda *args, **kwargs: calls.append("finalize"),
    )

    code = init_flow.run_init(_fake_settings(tmp_path), _console())
    assert code == 0
    assert calls == ["verify", "install", "keep_or_pick", "finalize"]


def test_keep_or_pick_keeps_when_user_accepts_default(tmp_path, monkeypatch):
    settings = _fake_settings(tmp_path)
    settings.models.llm = "llama3.1:8b"
    settings.notes_chat.emb_model = "nomic-embed-text"
    monkeypatch.setattr(
        init_flow, "_ollama_models", lambda: ["llama3.1:8b", "nomic-embed-text"]
    )

    console = _console()
    monkeypatch.setattr(console, "input", lambda *args, **kwargs: "")

    chat, embed, changed = init_flow.keep_or_pick(console, settings)
    assert (chat, embed, changed) == ("llama3.1:8b", "nomic-embed-text", False)


def test_keep_or_pick_runs_pickers_when_user_picks(tmp_path, monkeypatch):
    settings = _fake_settings(tmp_path)
    settings.models.llm = "llama3.1:8b"
    settings.notes_chat.emb_model = "nomic-embed-text"
    monkeypatch.setattr(
        init_flow, "_ollama_models", lambda: ["llama3.1:8b", "nomic-embed-text"]
    )
    monkeypatch.setattr(
        init_flow,
        "pick_models",
        lambda console: ("qwen2.5:7b", "mxbai-embed-large"),
    )

    console = _console()
    monkeypatch.setattr(console, "input", lambda *args, **kwargs: "p")

    chat, embed, changed = init_flow.keep_or_pick(console, settings)
    assert (chat, embed, changed) == ("qwen2.5:7b", "mxbai-embed-large", True)


def test_keep_or_pick_falls_through_when_models_missing(tmp_path, monkeypatch):
    """If current models aren't installed, skip the prompt and run pickers."""
    settings = _fake_settings(tmp_path)
    settings.models.llm = "llama3.1:8b"
    settings.notes_chat.emb_model = "nomic-embed-text"
    monkeypatch.setattr(init_flow, "_ollama_models", lambda: [])  # nothing installed
    monkeypatch.setattr(
        init_flow,
        "pick_models",
        lambda console: ("phi3:mini", "all-minilm"),
    )

    console = _console()
    chat, embed, changed = init_flow.keep_or_pick(console, settings)
    assert (chat, embed, changed) == ("phi3:mini", "all-minilm", True)


def test_pull_and_finalize_merges_user_keys(tmp_path, monkeypatch):
    """A custom key already in config.toml survives a re-run."""
    config_path = tmp_path / "config.toml"
    chroma_dir = tmp_path / "chroma"
    notes_root = tmp_path / "notes"

    import tomli_w

    with config_path.open("wb") as fh:
        tomli_w.dump(
            {
                "models": {"llm": "old-llm"},
                "notes_chat": {"emb_model": "old-embed"},
                "user_custom": {"theme": "midnight"},
            },
            fh,
        )

    settings = _fake_settings(tmp_path)
    settings.directories.notes_root = notes_root
    settings.notes_chat.index_dir = tmp_path

    monkeypatch.setattr(init_flow.ChirpSettings, "get_config_path", lambda: config_path)
    monkeypatch.setattr(init_flow, "_ollama_models", lambda: ["new-llm", "new-embed"])

    console = _console()
    init_flow.pull_and_finalize(
        console, settings, "new-llm", "new-embed", models_changed=True
    )

    import tomllib

    with config_path.open("rb") as fh:
        merged = tomllib.load(fh)
    assert merged["user_custom"] == {"theme": "midnight"}
    assert merged["models"]["llm"] == "new-llm"
    assert merged["notes_chat"]["emb_model"] == "new-embed"
    assert chroma_dir.exists()
    assert notes_root.exists()


def test_pull_and_finalize_skips_config_write_when_models_unchanged(
    tmp_path, monkeypatch
):
    """When the user keeps current models, config.toml isn't rewritten."""
    config_path = tmp_path / "config.toml"
    chroma_dir = tmp_path / "chroma"
    notes_root = tmp_path / "notes"

    import tomli_w

    with config_path.open("wb") as fh:
        tomli_w.dump({"models": {"llm": "kept"}}, fh)

    original_mtime = config_path.stat().st_mtime
    import time as _time

    _time.sleep(0.01)  # ensure mtime can change if we did write

    settings = _fake_settings(tmp_path)
    settings.directories.notes_root = notes_root
    settings.notes_chat.index_dir = tmp_path

    monkeypatch.setattr(init_flow.ChirpSettings, "get_config_path", lambda: config_path)
    monkeypatch.setattr(init_flow, "_ollama_models", lambda: ["kept", "kept-embed"])

    console = _console()
    init_flow.pull_and_finalize(
        console, settings, "kept", "kept-embed", models_changed=False
    )

    assert config_path.stat().st_mtime == original_mtime
    assert chroma_dir.exists()
    assert notes_root.exists()


def _stub_for_clean_box(monkeypatch):
    """Stub all external probes for a fully-installed, ollama-running box."""
    monkeypatch.setattr(init_flow, "_which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(
        init_flow,
        "_ollama_installed",
        lambda: init_flow.DependencyStatus("Ollama", True, "running · 0.1"),
    )
    monkeypatch.setattr(
        init_flow,
        "_ffmpeg_installed",
        lambda: init_flow.DependencyStatus("ffmpeg", True, "7.1.1"),
    )
    monkeypatch.setattr(
        init_flow, "_ollama_models", lambda: ["llama3.1:8b", "nomic-embed-text"]
    )


def test_run_init_rerun_is_idempotent_and_preserves_user_keys(tmp_path, monkeypatch):
    """AC-11: run twice in a tmp HOME, assert second run preserves a hand-edited key."""
    config_path = tmp_path / "config.toml"
    notes_root = tmp_path / "notes"

    settings = _fake_settings(tmp_path)
    settings.directories.notes_root = notes_root
    settings.notes_chat.index_dir = tmp_path
    settings.models.llm = "llama3.1:8b"
    settings.notes_chat.emb_model = "nomic-embed-text"

    monkeypatch.setattr(init_flow.ChirpSettings, "get_config_path", lambda: config_path)
    _stub_for_clean_box(monkeypatch)

    pick_calls = {"count": 0}

    def boom_pick(_console):
        pick_calls["count"] += 1
        return ("llama3.1:8b", "nomic-embed-text")

    monkeypatch.setattr(init_flow, "pick_models", boom_pick)

    console1 = _console()
    monkeypatch.setattr(console1, "input", lambda *args, **kwargs: "")  # K/p → keep

    code = init_flow.run_init(settings, console1)
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

    console2 = _console()
    monkeypatch.setattr(console2, "input", lambda *args, **kwargs: "")  # K/p → keep

    code = init_flow.run_init(settings, console2)
    assert code == 0
    assert pick_calls["count"] == 0  # user kept current models on both runs
    assert config_path.stat().st_mtime == first_mtime  # no rewrite

    with config_path.open("rb") as fh:
        second = tomllib.load(fh)
    assert second["user_custom"] == {"theme": "midnight"}


def test_merge_config_backs_up_corrupt_file(tmp_path):
    """A corrupt config.toml is renamed to .bak-<ts> before fresh write."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("this is { not valid toml\n", encoding="utf-8")

    console = _console()
    init_flow._merge_config(config_path, "new-llm", "new-embed", console=console)

    backups = list(tmp_path.glob("config.toml.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "this is { not valid toml\n"

    output = console.file.getvalue()
    assert "could not be parsed" in output

    import tomllib

    with config_path.open("rb") as fh:
        new = tomllib.load(fh)
    assert new["models"]["llm"] == "new-llm"
    assert new["notes_chat"]["emb_model"] == "new-embed"


def _stub_verify_deps(monkeypatch):
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
    monkeypatch.setattr(
        init_flow,
        "_ollama_installed",
        lambda: init_flow.DependencyStatus("Ollama", True, "running · 0.1"),
    )
    monkeypatch.setattr(
        init_flow, "_ollama_models", lambda: ["llama3.1:8b", "nomic-embed-text"]
    )


def test_verify_includes_screen_recording_permission_last(tmp_path, monkeypatch):
    _stub_verify_deps(monkeypatch)
    monkeypatch.setattr(
        init_flow.audio_capture,
        "check_permissions",
        lambda: {"screen_recording": "granted", "microphone": "granted"},
    )
    monkeypatch.setattr(init_flow.platform, "system", lambda: "Darwin")

    settings = _fake_settings(tmp_path)
    statuses = init_flow.verify(settings, _console())

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

    settings = _fake_settings(tmp_path)
    statuses = init_flow.verify(settings, _console())

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

    settings = _fake_settings(tmp_path)
    statuses = init_flow.verify(settings, _console())

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

    settings = _fake_settings(tmp_path)
    statuses = init_flow.verify(settings, _console())

    perm = next(s for s in statuses if s.name == "screen recording permission")
    assert perm.installed is True
    assert perm.required is False
    assert "not applicable on Linux" in perm.detail
    assert called == [], "check_permissions must not be called on non-Darwin"


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
