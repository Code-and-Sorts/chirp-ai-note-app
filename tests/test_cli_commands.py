from pathlib import Path
from unittest.mock import patch

from config.settings import ChirpSettings


def _make_settings(tmp_path: Path) -> ChirpSettings:
    settings = ChirpSettings()
    settings.directories.notes = tmp_path
    return settings


class TestListNotes:
    def test_list_notes_empty_directory(self, tmp_path, monkeypatch, capsys):
        from chirp.cli import list_notes

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        list_notes()

        output = capsys.readouterr().out
        assert "No notes found" in output

    def test_list_notes_shows_title_from_header(self, tmp_path, monkeypatch, capsys):
        from chirp.cli import list_notes

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        note = tmp_path / "meetings_2025_01_15.md"
        note.write_text("# Team Standup\n\nSome meeting content here.\n")

        list_notes()

        output = capsys.readouterr().out
        assert "Team Standup" in output
        assert "meetings_2025_01_15.md" in output

    def test_list_notes_shows_stem_when_no_header(self, tmp_path, monkeypatch, capsys):
        from chirp.cli import list_notes

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        note = tmp_path / "my_note.md"
        note.write_text("No heading here, just content.\n")

        list_notes()

        output = capsys.readouterr().out
        assert "my_note" in output

    def test_list_notes_shows_total_count(self, tmp_path, monkeypatch, capsys):
        from chirp.cli import list_notes

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        (tmp_path / "note1.md").write_text("# First\n")
        (tmp_path / "note2.md").write_text("# Second\n")

        list_notes()

        output = capsys.readouterr().out
        assert "2 note(s)" in output


class TestVersion:
    def test_version_installed(self, capsys):
        from chirp.cli import version

        with patch("importlib.metadata.version", return_value="1.2.3"):
            version()

        output = capsys.readouterr().out
        assert "1.2.3" in output

    def test_version_not_installed(self, capsys):
        from importlib.metadata import PackageNotFoundError

        from chirp.cli import version

        with patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("chirp-notes-ai"),
        ):
            version()

        output = capsys.readouterr().out
        assert "dev" in output


class TestTranscribeModelOverride:
    def test_model_override_passed_to_batch_processor(self, tmp_path, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        settings = _make_settings(tmp_path)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "test.wav").write_bytes(b"\x00" * 100)
        settings.directories.raw_audio = audio_dir

        captured_args = {}

        class FakeBatchProcessor:
            def __init__(self, s, model_override=None):
                captured_args["settings"] = s
                captured_args["model_override"] = model_override

            def process_files(
                self,
                files,
                force=False,
                progress_callback=None,
                on_segment=None,
            ):
                return [{"success": True}]

        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        monkeypatch.setitem(sys.modules, "faster_whisper", MagicMock())
        monkeypatch.setattr(
            "transcriber.batch_processor.BatchProcessor",
            FakeBatchProcessor,
        )

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--model", "small"])

        assert result.exit_code == 0
        assert captured_args.get("model_override") == "small"

    def test_transcribe_without_model_override(self, tmp_path, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        settings = _make_settings(tmp_path)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "test.wav").write_bytes(b"\x00" * 100)
        settings.directories.raw_audio = audio_dir

        captured_args = {}

        class FakeBatchProcessor:
            def __init__(self, s, model_override=None):
                captured_args["model_override"] = model_override

            def process_files(
                self,
                files,
                force=False,
                progress_callback=None,
                on_segment=None,
            ):
                return [{"success": True}]

        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        monkeypatch.setitem(sys.modules, "faster_whisper", MagicMock())
        monkeypatch.setattr(
            "transcriber.batch_processor.BatchProcessor",
            FakeBatchProcessor,
        )

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe"])

        assert result.exit_code == 0
        assert captured_args.get("model_override") is None
