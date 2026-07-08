from pathlib import Path

import pytest

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
        # Diagnostics go to stderr, data to stdout (AC-1).
        assert "No notes found" in result.stderr

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
        assert "2 total" in result.stderr

    def test_list_notes_shows_subcommand_hints(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "team-standup-2026-01-15", "Team Standup", "Content")
        runner, app = self._runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes"])
        assert result.exit_code == 0
        assert "chirp notes view <id>" in result.stderr
        assert "chirp notes edit <id>" in result.stderr
        assert "chirp notes delete <id>" in result.stderr
        assert "chirp notes --tag meeting" in result.stderr
        assert "chirp notes view <id>" not in result.stdout


class TestTranscribeCli:
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

    def _fake_processor(self, captured: dict):
        class FakeBatchProcessor:
            def __init__(self, s, model_override=None):
                captured["settings"] = s
                captured["model_override"] = model_override

            def run_queue(self, n=None, force=False, console=None):
                captured["n"] = n
                captured["force"] = force
                return {"ok": 1, "failed": 0, "total": 1}

        return FakeBatchProcessor

    def test_model_override_passed_to_batch_processor(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        self._seed_note_with_audio(tmp_path)
        captured: dict = {}

        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        monkeypatch.setattr(
            "transcriber.batch_processor.BatchProcessor",
            self._fake_processor(captured),
        )

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--model", "small"])

        assert result.exit_code == 0
        assert captured["model_override"] == "small"
        assert captured["n"] is None
        assert captured["force"] is False

    def test_n_and_force_passed_through(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        self._seed_note_with_audio(tmp_path)
        captured: dict = {}

        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        monkeypatch.setattr(
            "transcriber.batch_processor.BatchProcessor",
            self._fake_processor(captured),
        )

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "2", "--force"])

        assert result.exit_code == 0
        assert captured["n"] == 2
        assert captured["force"] is True

    def test_n_must_be_positive(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "0"])
        assert result.exit_code == 2
        assert "must be a positive integer" in result.stderr

    def test_no_stream_flag_removed(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--no-stream"])
        assert result.exit_code != 0
        # result.output is the combined stream (Click usage errors hit stderr).
        assert "no such option" in result.output.lower() or result.exit_code == 2


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

            def generate_for_records(
                self, records, force=False, template_override=None
            ):
                captured["records"] = records
                captured["force"] = force
                captured["template_override"] = template_override
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
        assert "Regenerated notes for 2/2" in result.stderr

    def test_regen_with_n_rejected(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--regen", "1"])

        assert result.exit_code == 2
        assert "do not pass N" in result.stderr

    def test_regen_with_force_rejected(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--regen", "--force"])

        assert result.exit_code == 2
        assert "mutually exclusive" in result.stderr

    def test_regen_when_no_transcripts(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["transcribe", "--regen"])

        assert result.exit_code == 0
        assert "No transcripts found" in result.stderr


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
        assert "2 of 3" in result.stderr
        assert "tag: meeting" in result.stderr

        result = runner.invoke(app, ["notes", "--tag", "meeting,pricing"])
        assert result.exit_code == 0
        assert "1 of 3" in result.stderr

        result = runner.invoke(app, ["notes", "--tag", "nope"])
        assert result.exit_code == 0
        assert "No notes matching tag 'nope'" in result.stderr

    def test_tag_with_subcommand_rejected(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "--tag", "meeting", "view", "a"])
        assert result.exit_code == 2
        assert "--tag is only valid when listing notes" in result.stderr

    def test_unknown_id(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "view", "missing"])
        assert result.exit_code == 1
        assert "no note matching 'missing'" in result.stderr

    def test_ambiguous_prefix(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "standup-monday-2026-04-20", "M", "x")
        _write_note(tmp_path, "standup-tuesday-2026-04-21", "T", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "view", "standup"])
        assert result.exit_code == 1
        assert "matches 2 notes" in result.stderr

    def test_resolve_full_id(self, tmp_path, monkeypatch):
        import chirp.cli
        from utils.file_utils import list_notes

        _write_note(tmp_path, "alpha-2026-04-20", "Alpha", "x")
        _write_note(tmp_path, "beta-2026-04-21", "Beta", "x")
        records = [r for r in list_notes(tmp_path) if r.notes is not None]

        record = chirp.cli._resolve_note(records, "alpha-2026-04-20")
        assert record.slug == "alpha-2026-04-20"

    def test_resolve_unique_prefix(self, tmp_path, monkeypatch):
        import chirp.cli
        from utils.file_utils import list_notes

        _write_note(tmp_path, "alpha-2026-04-20", "Alpha", "x")
        _write_note(tmp_path, "beta-2026-04-21", "Beta", "x")
        records = [r for r in list_notes(tmp_path) if r.notes is not None]

        record = chirp.cli._resolve_note(records, "alpha")
        assert record.slug == "alpha-2026-04-20"

    def test_resolve_integer_id_uses_newest_first(self, tmp_path, monkeypatch):
        import chirp.cli
        from utils.file_utils import list_notes

        # `list_notes` sorts oldest-first; the newest-first index 1 should
        # be the most recent note.
        _write_note(tmp_path, "older", "Older", "x")
        # Force a later created_at via meta.toml date so ordering is
        # deterministic regardless of mtime jitter.
        import tomli_w

        with (tmp_path / "older" / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {
                    "title": "Older",
                    "date": "2026-04-20T09:00:00",
                    "tags": [],
                },
                fh,
            )
        _write_note(tmp_path, "newer", "Newer", "x")
        with (tmp_path / "newer" / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {
                    "title": "Newer",
                    "date": "2026-04-21T09:00:00",
                    "tags": [],
                },
                fh,
            )
        records = [r for r in list_notes(tmp_path) if r.notes is not None]

        assert chirp.cli._resolve_note(records, "1").slug == "newer"
        assert chirp.cli._resolve_note(records, "2").slug == "older"

        with pytest.raises(chirp.cli.NoteNotFound):
            chirp.cli._resolve_note(records, "99")


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
        assert "deleted" in result.stderr.lower()
        assert not (tmp_path / "doomed-2026-04-20").exists()
        assert "doomed-2026-04-20/notes.md" in captured.get("dropped", "")

    def test_delete_aborts_on_no(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "safe-2026-04-20", "Safe", "x")
        runner, app = self._runner(tmp_path, monkeypatch)
        monkeypatch.setattr("chirp.cli._drop_from_index", lambda *args: None)

        result = runner.invoke(app, ["notes", "delete", "safe"], input="n\n")
        assert result.exit_code == 0
        assert "deletion cancelled" in result.stderr.lower()
        assert (tmp_path / "safe-2026-04-20").exists()


class TestDaemonRegistration:
    """Story 5.6: the daemon subapp is wired in hidden and dispatches end-to-end.

    These exercise the *top-level* ``chirp`` app, complementing
    ``tests/llm/test_cli_daemon*.py`` which drive the subapp directly.
    """

    _SUBCOMMANDS = ("status", "start", "stop", "restart", "enable", "disable", "logs")

    def test_daemon_subapp_is_hidden_in_top_help(self):
        from typer.testing import CliRunner

        import chirp.cli

        result = CliRunner().invoke(chirp.cli.app, ["--help"])
        assert result.exit_code == 0
        assert "daemon" not in result.stdout

    def test_visible_commands_unchanged(self):
        from typer.testing import CliRunner

        import chirp.cli

        result = CliRunner().invoke(chirp.cli.app, ["--help"])
        assert result.exit_code == 0
        for command in chirp.cli.VISIBLE_COMMAND_ORDER:
            assert command in result.stdout
        for hidden in ("daemon", "config", "devices", "index"):
            assert hidden not in result.stdout

    def test_daemon_subapp_help_lists_all_seven_subcommands(self):
        from typer.testing import CliRunner

        import chirp.cli

        result = CliRunner().invoke(chirp.cli.app, ["daemon", "--help"])
        assert result.exit_code == 0
        for subcommand in self._SUBCOMMANDS:
            assert subcommand in result.stdout

    def test_chirp_daemon_status_dispatch(self, monkeypatch):
        from unittest.mock import MagicMock

        from typer.testing import CliRunner

        import chirp.cli
        from llm.cli import daemon as daemon_module

        client = MagicMock()
        client.daemon_lazy_spawned = False
        client.daemon_respawned = False
        client.health_sync.return_value = {
            "status": "ok",
            "uptime_seconds": 12.0,
            "version": "0.7.0",
        }
        client.model_status_sync.return_value = {
            "pid": 4242,
            "rss_bytes": 1024,
            "last_request_at": None,
            "models": [],
        }
        monkeypatch.setattr(daemon_module, "configure_logging", MagicMock())
        monkeypatch.setattr(daemon_module, "LLMClient", MagicMock(return_value=client))

        result = CliRunner().invoke(chirp.cli.app, ["daemon", "status"])
        assert result.exit_code == 0
        assert '"running": true' in result.stdout

    def test_chirp_daemon_logs_dispatch(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from typer.testing import CliRunner

        import chirp.cli
        from llm.cli import daemon as daemon_module

        log_file = tmp_path / "chirpd.log"
        log_file.write_text("first line\nsecond line\n", encoding="utf-8")
        monkeypatch.setattr(daemon_module, "configure_logging", MagicMock())
        monkeypatch.setattr(daemon_module, "resolve_log_path", lambda *a, **k: log_file)

        result = CliRunner().invoke(chirp.cli.app, ["daemon", "logs"])
        assert result.exit_code == 0
        assert "first line" in result.stdout
        assert "second line" in result.stdout


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
        assert "no note matching" in result.stderr
        assert "matches" not in result.stderr

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
        flat = " ".join(result.stderr.split())
        assert "failed to delete" in flat
        assert "simulated permission denied" in flat


class TestDevicesCommand:
    """`chirp devices` lists audio devices via sounddevice."""

    def test_devices_lists_without_error(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from typer.testing import CliRunner

        import chirp.cli

        monkeypatch.setattr("chirp.cli.get_settings", lambda: _make_settings(tmp_path))
        monkeypatch.setattr("chirp.cli.platform.system", lambda: "Linux")

        devices = [
            {
                "index": 0,
                "name": "Built-in Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            }
        ]

        def _query(*args, **kwargs):
            return devices[0] if kwargs.get("kind") else devices

        with patch("recorder.device_manager.sd.query_devices", side_effect=_query):
            result = CliRunner().invoke(chirp.cli.app, ["devices"])

        assert result.exit_code == 0
        assert "Built-in Microphone" in result.stdout


class TestTranscribeSurfacesModelLoadError:
    """AC-2: offline transcribe surfaces the typed Whisper load error cleanly."""

    def test_transcribe_reports_model_load_error(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import chirp.cli
        from chirp.exceptions import WhisperModelLoadError

        TestTranscribeCli()._seed_note_with_audio(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: _make_settings(tmp_path))

        class _BoomProcessor:
            def __init__(self, settings, model_override=None):
                raise WhisperModelLoadError(
                    "Could not download or load the Whisper model 'small'. "
                    "Check your network connection and free disk space."
                )

        monkeypatch.setattr(
            "transcriber.batch_processor.BatchProcessor", _BoomProcessor
        )

        result = CliRunner().invoke(chirp.cli.app, ["transcribe"])

        assert result.exit_code == 1
        flat = " ".join(result.stderr.split())
        assert "Could not download or load the Whisper model" in flat


class TestTranscribeRegenTemplateAndNoteFilter:
    def _seed(self, tmp_path: Path, slug: str, tags: list[str] | None = None) -> None:
        note_dir = tmp_path / slug
        note_dir.mkdir()
        (note_dir / "transcript.txt").write_text("hello world", encoding="utf-8")
        (note_dir / "notes.md").write_text(f"# {slug}\n", encoding="utf-8")

        import tomli_w

        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {"title": slug, "date": "2026-04-20T09:00:00", "tags": tags or []},
                fh,
            )

    def _fake_generator(self, captured: dict):
        class FakeNoteGenerator:
            def __init__(self, s):
                captured["settings"] = s

            def generate_for_records(
                self, records, force=False, template_override=None
            ):
                captured["records"] = records
                captured["template_override"] = template_override
                return {
                    "success": True,
                    "results": [{"success": True, "slug": r.slug} for r in records],
                }

        return FakeNoteGenerator

    def _invoke(self, tmp_path, monkeypatch, args, captured=None):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        if captured is not None:
            monkeypatch.setattr(
                "notes.note_generator.NoteGenerator", self._fake_generator(captured)
            )

        from typer.testing import CliRunner

        import chirp.cli

        return CliRunner().invoke(chirp.cli.app, args)

    def test_note_or_template_without_regen_rejected(self, tmp_path, monkeypatch):
        result = self._invoke(
            tmp_path, monkeypatch, ["transcribe", "--template", "standup"]
        )
        assert result.exit_code == 2
        assert "require --regen" in result.stderr

        result = self._invoke(tmp_path, monkeypatch, ["transcribe", "--note", "1"])
        assert result.exit_code == 2
        assert "require --regen" in result.stderr

    def test_regen_unknown_template_rejected_with_names(self, tmp_path, monkeypatch):
        self._seed(tmp_path, "a-2026-04-20")
        result = self._invoke(
            tmp_path, monkeypatch, ["transcribe", "--regen", "--template", "nope"]
        )
        assert result.exit_code == 2
        assert "unknown template 'nope'" in result.stderr
        assert "meeting" in result.stderr

    def test_regen_note_filter_selects_single_record(self, tmp_path, monkeypatch):
        self._seed(tmp_path, "a-2026-04-20")
        self._seed(tmp_path, "b-2026-04-21")
        captured: dict = {}

        result = self._invoke(
            tmp_path,
            monkeypatch,
            ["transcribe", "--regen", "--note", "a-2026-04-20"],
            captured,
        )

        assert result.exit_code == 0
        assert [r.slug for r in captured["records"]] == ["a-2026-04-20"]
        assert captured["template_override"] is None

    def test_regen_template_with_note_persists_to_meta(self, tmp_path, monkeypatch):
        import tomllib

        self._seed(tmp_path, "a-2026-04-20")
        self._seed(tmp_path, "b-2026-04-21")
        captured: dict = {}

        result = self._invoke(
            tmp_path,
            monkeypatch,
            [
                "transcribe",
                "--regen",
                "--note",
                "a-2026-04-20",
                "--template",
                "standup",
            ],
            captured,
        )

        assert result.exit_code == 0
        assert captured["template_override"] == "standup"
        with (tmp_path / "a-2026-04-20" / "meta.toml").open("rb") as fh:
            meta = tomllib.load(fh)
        assert meta["template"] == "standup"
        with (tmp_path / "b-2026-04-21" / "meta.toml").open("rb") as fh:
            other = tomllib.load(fh)
        assert "template" not in other

    def test_regen_unknown_note_id_fails(self, tmp_path, monkeypatch):
        self._seed(tmp_path, "a-2026-04-20")
        result = self._invoke(
            tmp_path, monkeypatch, ["transcribe", "--regen", "--note", "ghost"]
        )
        assert result.exit_code == 1
        assert "no note matching" in result.stderr


class TestNotesTag:
    def _runner(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import chirp.cli

        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        return CliRunner(), chirp.cli.app

    def _read_tags(self, tmp_path: Path, slug: str) -> list[str]:
        import tomllib

        with (tmp_path / slug / "meta.toml").open("rb") as fh:
            return tomllib.load(fh)["tags"]

    def test_add_remove_round_trip(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x", tags=["meeting"])
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(
            app, ["notes", "tag", "a-2026-04-20", "--add", "standup", "--add", "dsu"]
        )
        assert result.exit_code == 0
        assert self._read_tags(tmp_path, "a-2026-04-20") == [
            "meeting",
            "standup",
            "dsu",
        ]

        result = runner.invoke(
            app, ["notes", "tag", "a-2026-04-20", "--remove", "meeting"]
        )
        assert result.exit_code == 0
        assert self._read_tags(tmp_path, "a-2026-04-20") == ["standup", "dsu"]

    def test_clear_then_add(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x", tags=["old", "stale"])
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(
            app, ["notes", "tag", "a-2026-04-20", "--clear", "--add", "fresh"]
        )
        assert result.exit_code == 0
        assert self._read_tags(tmp_path, "a-2026-04-20") == ["fresh"]

    def test_numeric_id_matches_notes_table(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "older-2026-04-19", "Older", "x")
        _write_note(tmp_path, "newer-2026-04-20", "Newer", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "tag", "1", "--add", "top"])
        assert result.exit_code == 0
        assert self._read_tags(tmp_path, "newer-2026-04-20") == ["top"]

    def test_tags_untranscribed_note_by_slug(self, tmp_path, monkeypatch):
        note_dir = tmp_path / "fresh-recording-2026-07-08"
        note_dir.mkdir()
        (note_dir / "audio.wav").write_bytes(b"\x00" * 10)

        import tomli_w

        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {"title": "Fresh", "date": "2026-07-08T09:00:00", "tags": []}, fh
            )

        runner, app = self._runner(tmp_path, monkeypatch)
        result = runner.invoke(
            app, ["notes", "tag", "fresh-recording", "--add", "standup"]
        )
        assert result.exit_code == 0
        assert self._read_tags(tmp_path, "fresh-recording-2026-07-08") == ["standup"]

    def test_requires_a_flag(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x")
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "tag", "a-2026-04-20"])
        assert result.exit_code == 2
        assert "nothing to do" in result.stderr

    def test_duplicate_add_is_ignored(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "a-2026-04-20", "A", "x", tags=["work"])
        runner, app = self._runner(tmp_path, monkeypatch)

        result = runner.invoke(app, ["notes", "tag", "a-2026-04-20", "--add", "work"])
        assert result.exit_code == 0
        assert self._read_tags(tmp_path, "a-2026-04-20") == ["work"]


class TestRecordTemplateFlag:
    def test_unknown_template_rejected_before_recording(self, tmp_path, monkeypatch):
        settings = _make_settings(tmp_path)
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)

        from typer.testing import CliRunner

        import chirp.cli

        runner = CliRunner()
        result = runner.invoke(chirp.cli.app, ["record", "--template", "nope"])

        assert result.exit_code == 2
        assert "unknown template 'nope'" in result.stderr
        assert "standup" in result.stderr
