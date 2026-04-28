from pathlib import Path

from config.settings import ChirpSettings


def _make_settings(tmp_path: Path) -> ChirpSettings:
    settings = ChirpSettings()
    settings.directories.notes_root = tmp_path
    return settings


def _write_note(tmp_path: Path, slug: str, title: str, body: str) -> Path:
    note_dir = tmp_path / slug
    note_dir.mkdir()
    notes_path = note_dir / "notes.md"
    notes_path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")

    import tomli_w

    with (note_dir / "meta.toml").open("wb") as fh:
        tomli_w.dump(
            {
                "title": title,
                "date": "2026-04-20T09:00:00",
                "tags": [],
            },
            fh,
        )
    return notes_path


class TestListNotes:
    def test_list_notes_empty_directory(self, tmp_path, monkeypatch, capsys):
        from chirp.cli import notes_list

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        notes_list()

        output = capsys.readouterr().out
        assert "No notes found" in output

    def test_list_notes_shows_title_from_header(self, tmp_path, monkeypatch, capsys):
        from chirp.cli import notes_list

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        _write_note(tmp_path, "team-standup-2026-01-15", "Team Standup", "Content")

        notes_list()

        output = capsys.readouterr().out
        assert "Team Standup" in output

    def test_list_notes_shows_total_count(self, tmp_path, monkeypatch, capsys):
        from chirp.cli import notes_list

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        _write_note(tmp_path, "first", "First", "Body")
        _write_note(tmp_path, "second", "Second", "Body")

        notes_list()

        output = capsys.readouterr().out
        assert "2 total" in output


class TestTranscribeModelOverride:
    def _seed_note_with_audio(self, tmp_path: Path) -> None:
        note_dir = tmp_path / "sample-2026-04-20"
        note_dir.mkdir()
        (note_dir / "audio.wav").write_bytes(b"\x00" * 100)

        import tomli_w

        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {
                    "title": "Sample",
                    "date": "2026-04-20T09:00:00",
                    "tags": [],
                },
                fh,
            )

    def test_model_override_passed_to_batch_processor(self, tmp_path, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        settings = _make_settings(tmp_path)

        self._seed_note_with_audio(tmp_path)

        captured_args = {}

        class FakeBatchProcessor:
            def __init__(self, s, model_override=None):
                captured_args["settings"] = s
                captured_args["model_override"] = model_override

            def process_records(
                self,
                records,
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

        self._seed_note_with_audio(tmp_path)

        captured_args = {}

        class FakeBatchProcessor:
            def __init__(self, s, model_override=None):
                captured_args["model_override"] = model_override

            def process_records(
                self,
                records,
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


class TestTranscribeRegen:
    def _seed_record_with_transcript(self, tmp_path: Path, slug: str) -> None:
        note_dir = tmp_path / slug
        note_dir.mkdir()
        (note_dir / "audio.wav").write_bytes(b"\x00" * 100)
        (note_dir / "transcript.txt").write_text("hello world", encoding="utf-8")

        import tomli_w

        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {
                    "title": slug,
                    "date": "2026-04-20T09:00:00",
                    "tags": [],
                },
                fh,
            )

    def test_regen_calls_note_generator_with_force_true(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        self._seed_record_with_transcript(tmp_path, "first-2026-04-20")
        self._seed_record_with_transcript(tmp_path, "second-2026-04-20")

        captured = {}

        class FakeNoteGenerator:
            def __init__(self, s):
                captured["settings"] = s

            def generate_for_records(self, records, force=False):
                captured["records"] = records
                captured["force"] = force
                return {
                    "success": True,
                    "results": [{"success": True, "slug": r.slug} for r in records],
                }

        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        monkeypatch.setattr("notes.note_generator.NoteGenerator", FakeNoteGenerator)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--regen"])

        assert result.exit_code == 0
        assert captured.get("force") is True
        assert len(captured.get("records", [])) == 2
        assert "Regenerated notes for 2/2" in result.stdout

    def test_regen_with_note_id_rejected(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--regen", "1"])

        assert result.exit_code == 2
        assert "do not pass an index" in result.stdout

    def test_regen_with_force_rejected(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--regen", "--force"])

        assert result.exit_code == 2
        assert "mutually exclusive" in result.stdout

    def test_regen_with_no_notes_rejected(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--regen", "--no-notes"])

        assert result.exit_code == 2
        assert "mutually exclusive" in result.stdout

    def test_regen_when_no_transcripts(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--regen"])

        assert result.exit_code == 0
        assert "No transcripts found" in result.stdout
