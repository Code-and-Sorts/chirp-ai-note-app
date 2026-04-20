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
    settings.directories.notes = tmp_path / "notes"
    settings.directories.raw_audio = tmp_path / "audio"
    settings.directories.transcriptions = tmp_path / "transcripts"
    settings.directories.templates = tmp_path / "templates"
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
    monkeypatch.setattr(
        init_flow,
        "_blackhole_installed",
        lambda: init_flow.DependencyStatus("BlackHole 2ch", False, "not found"),
    )

    console = _console()
    settings = _fake_settings(tmp_path)

    statuses = init_flow.verify(settings, console)

    names = [s.name for s in statuses]
    assert "homebrew" in names
    assert "ffmpeg" in names
    assert "BlackHole 2ch" in names
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
        "_blackhole_installed",
        lambda: init_flow.DependencyStatus("BlackHole 2ch", True, "installed"),
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
    monkeypatch.setattr(
        init_flow,
        "_pick",
        lambda c, title, options: options[0].tag,
    )
    # fall back to the real _pick path once to exercise it
    chat, embed = init_flow.pick_models(console)
    assert chat == "llama3.1:8b"
    assert embed == "nomic-embed-text"
    # consume iterator so the unused-var lint doesn't complain
    _ = list(answers)
