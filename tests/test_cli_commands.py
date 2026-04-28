from pathlib import Path

from config.settings import ChirpSettings


def _make_settings(tmp_path: Path) -> ChirpSettings:
    settings = ChirpSettings()
    settings.directories.notes_root = tmp_path
    return settings


def _write_note(
    tmp_path: Path,
    slug: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
) -> Path:
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
                "tags": tags or [],
            },
            fh,
        )
    return notes_path


class TestListNotes:
    def _runner(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import chirp.cli

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        return CliRunner(), chirp.cli.app

    def test_list_notes_empty_directory(self, tmp_path, monkeypatch):
        runner, app = self._runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes"])
        assert result.exit_code == 0
        assert "No notes found" in result.stdout

    def test_list_notes_shows_title_from_header(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "team-standup-2026-01-15", "Team Standup", "Content")
        runner, app = self._runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes"])
        assert result.exit_code == 0
        assert "Team Standup" in result.stdout

    def test_list_notes_shows_total_count(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "first-2026-01-15", "First", "Body")
        _write_note(tmp_path, "second-2026-01-16", "Second", "Body")
        runner, app = self._runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes"])
        assert result.exit_code == 0
        assert "2 total" in result.stdout

    def test_list_notes_shows_subcommand_hints(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "team-standup-2026-01-15", "Team Standup", "Content")
        runner, app = self._runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes"])
        assert result.exit_code == 0
        assert "chirp notes view <id>" in result.stdout
        assert "chirp notes edit <id>" in result.stdout
        assert "chirp notes delete <id>" in result.stdout
        assert "chirp notes --tag meeting" in result.stdout


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


class TestNotesResolveAndTagFilter:
    def _runner(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import chirp.cli

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        return CliRunner(), chirp.cli.app

    def test_tag_filter_and_semantics(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x", tags=["meeting"])
        _write_note(tmp_path, "b-2026-04-20", "B", "x", tags=["meeting", "pricing"])
        _write_note(tmp_path, "c-2026-04-20", "C", "x", tags=["pricing"])
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "--tag", "meeting"])
        assert result.exit_code == 0
        assert "2 of 3" in result.stdout
        assert "tag: meeting" in result.stdout

        result = runner.invoke(app, ["notes", "--tag", "meeting,pricing"])
        assert result.exit_code == 0
        assert "1 of 3" in result.stdout

        result = runner.invoke(app, ["notes", "--tag", "nope"])
        assert result.exit_code == 0
        assert "No notes matching tag 'nope'" in result.stdout

    def test_tag_with_subcommand_rejected(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "--tag", "meeting", "view", "a"])
        assert result.exit_code == 2
        assert "--tag is only valid when listing notes" in result.stdout

    def test_unknown_id(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "view", "missing"])
        assert result.exit_code == 1
        assert "no note matching 'missing'" in result.stdout

    def test_ambiguous_prefix(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "standup-monday-2026-04-20", "M", "x")
        _write_note(tmp_path, "standup-tuesday-2026-04-21", "T", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "view", "standup"])
        assert result.exit_code == 1
        assert "matches 2 notes" in result.stdout

    def test_resolve_full_id(self, tmp_path, monkeypatch):
        from chirp.cli import _resolve_note
        from utils.file_utils import list_notes

        _write_note(tmp_path, "alpha-2026-04-20", "Alpha", "x")
        _write_note(tmp_path, "beta-2026-04-21", "Beta", "x")
        records = [r for r in list_notes(tmp_path) if r.notes is not None]

        record = _resolve_note(records, "alpha-2026-04-20")
        assert record.slug == "alpha-2026-04-20"

    def test_resolve_unique_prefix(self, tmp_path, monkeypatch):
        from chirp.cli import _resolve_note
        from utils.file_utils import list_notes

        _write_note(tmp_path, "alpha-2026-04-20", "Alpha", "x")
        _write_note(tmp_path, "beta-2026-04-21", "Beta", "x")
        records = [r for r in list_notes(tmp_path) if r.notes is not None]

        record = _resolve_note(records, "alpha")
        assert record.slug == "alpha-2026-04-20"


class TestNotesDelete:
    def _runner(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import chirp.cli

        settings = _make_settings(tmp_path)
        settings.notes_chat.auto_index = False
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        return CliRunner(), chirp.cli.app

    def test_delete_with_yes_flag(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "doomed-2026-04-20", "Doomed", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        captured = {}
        monkeypatch.setattr(
            "chirp.cli._drop_from_index",
            lambda settings, path: captured.setdefault("dropped", str(path)),
        )

        result = runner.invoke(app, ["notes", "delete", "doomed", "--yes"])
        assert result.exit_code == 0
        assert "Deleted" in result.stdout
        assert not (tmp_path / "doomed-2026-04-20").exists()
        assert "doomed-2026-04-20/notes.md" in captured.get("dropped", "")

    def test_delete_aborts_on_no(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "safe-2026-04-20", "Safe", "x")
        runner, app = self._runner(tmp_path, monkeypatch)
        monkeypatch.setattr("chirp.cli._drop_from_index", lambda *args: None)

        result = runner.invoke(app, ["notes", "delete", "safe"], input="n\n")
        assert result.exit_code == 0
        assert "Deletion cancelled" in result.stdout
        assert (tmp_path / "safe-2026-04-20").exists()


class TestNotesReviewPatches:
    def _runner(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import chirp.cli

        settings = _make_settings(tmp_path)
        settings.notes_chat.auto_index = False
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        return CliRunner(), chirp.cli.app

    def test_empty_tags_render_as_em_dash(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "untagged-2026-04-20", "Untagged", "x", tags=[])
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes"])
        assert result.exit_code == 0
        assert "—" in result.stdout

    def test_empty_note_id_reports_not_found(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "alpha-2026-04-20", "Alpha", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "view", ""])
        assert result.exit_code == 1
        assert "no note matching" in result.stdout
        assert "matches" not in result.stdout

    def test_delete_handles_rmtree_error(self, tmp_path, monkeypatch):
        import shutil

        _write_note(tmp_path, "doomed-2026-04-20", "Doomed", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        def boom(path, ignore_errors=False):
            raise PermissionError("simulated permission denied")

        monkeypatch.setattr(shutil, "rmtree", boom)

        result = runner.invoke(app, ["notes", "delete", "doomed", "--yes"])
        assert result.exit_code == 1
        # Rich wraps long lines so flatten whitespace before asserting.
        flat = " ".join(result.stdout.split())
        assert "failed to delete" in flat
        assert "simulated permission denied" in flat
