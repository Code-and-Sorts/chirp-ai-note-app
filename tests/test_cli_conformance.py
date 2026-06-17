"""Story 8.5 — CLI/TUI best-practices conformance tests.

Covers the stdout/stderr split, ``--json`` schemas, ``--version``, prompt
escape, NO_COLOR, exit-code constants, editor help, completion, AppleScript
escaping, the ``ask`` dual-input warning, and the extractable terminal-restore
seam. TTY/termios signal paths are verified manually (repo memory: no unit tests
for OS-touching TTY code), but the restore seam is unit-tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomli_w
from typer.testing import CliRunner

from config.settings import ChirpSettings


def _make_settings(tmp_path: Path) -> ChirpSettings:
    settings = ChirpSettings()
    settings.directories.notes_root = tmp_path
    settings.notes_chat.auto_index = False
    return settings


def _write_note(
    tmp_path: Path,
    slug: str,
    title: str,
    body: str = "body",
    tags: list[str] | None = None,
) -> Path:
    note_dir = tmp_path / slug
    note_dir.mkdir()
    notes_path = note_dir / "notes.md"
    notes_path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    with (note_dir / "meta.toml").open("wb") as fh:
        tomli_w.dump(
            {"title": title, "date": "2026-04-20T09:00:00", "tags": tags or []}, fh
        )
    return notes_path


def _runner(tmp_path, monkeypatch):
    import chirp.cli

    monkeypatch.setattr("chirp.cli.get_settings", lambda: _make_settings(tmp_path))
    return CliRunner(), chirp.cli.app


def typer_main_command(app):
    import typer.main

    return typer.main.get_command(app)


# ---------------------------------------------------------------------------
# AC-3 — --version / -V and about --plain
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_flag_prints_one_line_and_exits_0(self):
        import chirp.cli
        from chirp.about import _installed_version

        result = CliRunner().invoke(chirp.cli.app, ["--version"])
        assert result.exit_code == 0
        line = result.stdout.strip()
        assert line == f"chirp {_installed_version()}"

    def test_short_version_flag_is_identical(self):
        import chirp.cli

        long_ = CliRunner().invoke(chirp.cli.app, ["--version"])
        short = CliRunner().invoke(chirp.cli.app, ["-V"])
        assert short.exit_code == 0
        assert short.stdout.strip() == long_.stdout.strip()

    def test_help_lists_version(self):
        import chirp.cli

        result = CliRunner().invoke(chirp.cli.app, ["--help"])
        assert "--version" in result.stdout

    def test_about_plain_renders_static_panel(self, tmp_path, monkeypatch):
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["about", "--plain"])
        assert result.exit_code == 0
        # The static panel carries the credits line; no spinner frames.
        assert "Colby Timm" in result.stdout
        assert "⠋" not in result.stdout
        assert "⠙" not in result.stdout


# ---------------------------------------------------------------------------
# AC-1 — stdout/stderr split (search)
# ---------------------------------------------------------------------------


class TestSearchStreamSplit:
    def _seed(self, tmp_path):
        note_dir = tmp_path / "pricing-call-2026-04-20"
        note_dir.mkdir()
        (note_dir / "transcript.txt").write_text(
            "we discussed pricing at length", encoding="utf-8"
        )
        (note_dir / "notes.md").write_text(
            "# Pricing Call\n\npricing was the topic\n", encoding="utf-8"
        )
        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump(
                {
                    "title": "Pricing Call",
                    "date": "2026-04-20T09:00:00",
                    "tags": [],
                },
                fh,
            )

    def test_table_to_stdout_header_to_stderr(self, tmp_path, monkeypatch):
        self._seed(tmp_path)
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["search", "pricing"])
        assert result.exit_code == 0
        # Excerpt/table data on stdout; preamble + footer hints on stderr.
        assert "Pricing Call" in result.stdout
        assert "searching 1 note for" in result.stderr
        assert "chirp ask" in result.stderr
        assert "searching 1 note for" not in result.stdout


# ---------------------------------------------------------------------------
# AC-2 — --json for notes / ask / search (raw, no Rich chrome)
# ---------------------------------------------------------------------------


class TestNotesJson:
    def test_notes_json_schema_and_no_box(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "alpha-2026-04-20", "Alpha", tags=["x"])
        _write_note(tmp_path, "beta-2026-04-21", "Beta")
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        assert len(payload) == 2
        for row in payload:
            assert set(row) == {"id", "slug", "title", "date", "tags", "notes_path"}
        # newest-first id == 1 is the most recent note (beta).
        assert payload[0]["id"] == 1
        assert payload[0]["slug"] == "beta-2026-04-21"
        # No Rich box-drawing characters in the JSON stream.
        assert "─" not in result.stdout
        assert "│" not in result.stdout

    def test_notes_json_empty_is_empty_array(self, tmp_path, monkeypatch):
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.stdout) == []


class TestAskJson:
    def _patch_one_shot(self, monkeypatch):
        import notes_chat.cli as nc_cli

        monkeypatch.setattr(nc_cli, "get_notes_config", MagicMock)

        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": True,
                "context": "ctx",
                "sources": ["note #1"],
                "retrieved_ids": ["c1"],
            }

        def fake_generate(config, question, context):
            return {"success": True, "answer": "the decision was X"}

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        monkeypatch.setattr("notes_chat.prompting.generate_answer", fake_generate)
        monkeypatch.setattr("notes_chat.cache.get_cached_answer", lambda *a: None)
        monkeypatch.setattr("notes_chat.cache.cache_answer", lambda *a: None)

    def test_ask_json_object_schema(self, monkeypatch):
        import chirp.cli

        self._patch_one_shot(monkeypatch)
        result = CliRunner().invoke(
            chirp.cli.app, ["ask", "--json", "what did we decide?"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert set(payload) == {"question", "answer", "sources"}
        assert payload["question"] == "what did we decide?"
        assert payload["answer"] == "the decision was X"
        assert payload["sources"] == ["note #1"]

    def test_ask_json_never_instantiates_markdown(self, monkeypatch):
        import chirp.cli

        self._patch_one_shot(monkeypatch)
        with patch("rich.markdown.Markdown") as md:
            result = CliRunner().invoke(chirp.cli.app, ["ask", "--json", "q?"])
        assert result.exit_code == 0
        md.assert_not_called()


class TestSearchJsonRaw:
    def _seed(self, tmp_path):
        note_dir = tmp_path / "pricing-2026-04-20"
        note_dir.mkdir()
        (note_dir / "notes.md").write_text("# P\n\npricing\n", encoding="utf-8")
        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump({"title": "P", "date": "2026-04-20T09:00:00", "tags": []}, fh)

    def test_search_json_is_raw_parseable(self, tmp_path, monkeypatch):
        self._seed(tmp_path)
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["search", "--json", "pricing"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["query"] == "pricing"
        assert isinstance(payload["matches"], list)
        # Nothing on stderr — JSON path emits no chrome.
        assert result.stderr == ""


# ---------------------------------------------------------------------------
# AC-4 — uniform prompt escape in record
# ---------------------------------------------------------------------------


class TestPromptEscape:
    @pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
    def test_prompt_timeframe_aborts(self, exc, monkeypatch):
        import typer

        import chirp.cli

        monkeypatch.setattr(chirp.cli.console, "input", MagicMock(side_effect=exc))
        with pytest.raises(typer.Exit) as caught:
            chirp.cli._prompt_timeframe()
        assert caught.value.exit_code == chirp.cli.exit_codes.RUNTIME_ERROR

    @pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
    def test_prompt_tags_aborts(self, exc, monkeypatch):
        import typer

        import chirp.cli

        monkeypatch.setattr(chirp.cli.console, "input", MagicMock(side_effect=exc))
        with pytest.raises(typer.Exit) as caught:
            chirp.cli._prompt_tags()
        assert caught.value.exit_code == chirp.cli.exit_codes.RUNTIME_ERROR

    def test_prompt_timeframe_empty_enter_skips(self, monkeypatch):
        import chirp.cli

        monkeypatch.setattr(chirp.cli.console, "input", MagicMock(return_value="  "))
        assert chirp.cli._prompt_timeframe() is None

    def test_prompt_tags_empty_enter_skips(self, monkeypatch):
        import chirp.cli

        monkeypatch.setattr(chirp.cli.console, "input", MagicMock(return_value=""))
        assert chirp.cli._prompt_tags() == []


# ---------------------------------------------------------------------------
# AC-5 — NO_COLOR and --no-color
# ---------------------------------------------------------------------------


class TestNoColor:
    def test_no_color_env_strips_ansi(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "alpha-2026-04-20", "Alpha")
        monkeypatch.setenv("NO_COLOR", "1")
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes"])
        assert result.exit_code == 0
        assert "\x1b[" not in result.stdout
        assert "\x1b[" not in result.stderr

    def test_no_color_flag_strips_ansi(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "alpha-2026-04-20", "Alpha")
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["--no-color", "notes"])
        assert result.exit_code == 0
        assert "\x1b[" not in result.stdout
        assert "\x1b[" not in result.stderr

    def test_editor_render_cache_honors_no_color(self, monkeypatch):
        from notes.note_editor import ManualNoteEditor

        monkeypatch.setenv("NO_COLOR", "1")
        editor = ManualNoteEditor("T", "# Heading\n\n**bold** text\n")
        editor._ensure_view_cache(80)
        # Width measurement still works (display lines built), but no styled
        # segments carry color when NO_COLOR is in effect.
        assert editor._display_lines
        for display in editor._display_lines:
            for segment in display.segments:
                if segment.style is not None:
                    assert segment.style.color is None


# ---------------------------------------------------------------------------
# AC-6 — centralized exit-code constants
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_constants_match_documented_table(self):
        from chirp import exit_codes

        assert exit_codes.SUCCESS == 0
        assert exit_codes.RUNTIME_ERROR == 1
        assert exit_codes.USAGE_ERROR == 2
        assert exit_codes.DAEMON_UNREACHABLE == 3
        assert exit_codes.MODEL_LOAD_FAILED == 4
        assert exit_codes.MODEL_NOT_FOUND == 5
        assert exit_codes.NOT_APPLE_SILICON == 7

    def test_init_flow_uses_central_not_apple_silicon(self):
        from chirp import exit_codes, init_flow

        assert init_flow.EXIT_NOT_APPLE_SILICON == exit_codes.NOT_APPLE_SILICON

    def test_search_usage_error_is_2(self, tmp_path, monkeypatch):
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["search", "   "])
        assert result.exit_code == 2

    def test_notes_view_missing_is_1(self, tmp_path, monkeypatch):
        _write_note(tmp_path, "alpha-2026-04-20", "Alpha")
        runner, app = _runner(tmp_path, monkeypatch)
        result = runner.invoke(app, ["notes", "view", "nope"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# AC-7 — extractable terminal-restore seam
# ---------------------------------------------------------------------------


class TestTerminalRestore:
    def test_restore_terminal_calls_tcsetattr_and_show_cursor(self, monkeypatch):
        import chirp.cli

        calls = {"tcsetattr": False, "cursor": None}

        fake_termios = MagicMock()
        fake_termios.TCSADRAIN = 1
        fake_termios.error = Exception

        def fake_tcsetattr(fd, when, settings):
            calls["tcsetattr"] = True

        fake_termios.tcsetattr = fake_tcsetattr
        monkeypatch.setitem(__import__("sys").modules, "termios", fake_termios)
        monkeypatch.setattr(
            chirp.cli.console,
            "show_cursor",
            lambda visible: calls.__setitem__("cursor", visible),
        )

        chirp.cli._restore_terminal(0, object())
        assert calls["tcsetattr"] is True
        assert calls["cursor"] is True

    def test_restore_terminal_noop_without_settings(self, monkeypatch):
        import chirp.cli

        cursor = {"shown": None}
        monkeypatch.setattr(
            chirp.cli.console,
            "show_cursor",
            lambda visible: cursor.__setitem__("shown", visible),
        )
        chirp.cli._restore_terminal(None, None)
        # Still shows the cursor even when there were no termios settings.
        assert cursor["shown"] is True


# ---------------------------------------------------------------------------
# AC-8 — note editor help / keybinding hint
# ---------------------------------------------------------------------------


class TestEditorHelp:
    def test_keybinding_hint_constant_present(self):
        from notes.note_editor import KEYBINDING_HINT

        assert ":wq save" in KEYBINDING_HINT
        assert ":help" in KEYBINDING_HINT

    def test_question_mark_opens_help_in_view(self):
        from notes.note_editor import ManualNoteEditor

        editor = ManualNoteEditor("T", "body\n")
        editor.mode = "view"
        editor._handle_view_mode("?")
        assert editor.show_help is True

    def test_help_command_opens_overlay(self):
        from notes.note_editor import ManualNoteEditor

        editor = ManualNoteEditor("T", "body\n")
        editor.command_buffer = ":help"
        result = editor._execute_command()
        assert result is False
        assert editor.show_help is True

    def test_unknown_command_points_at_help(self):
        from notes.note_editor import ManualNoteEditor

        editor = ManualNoteEditor("T", "body\n")
        editor.command_buffer = ":frobnicate"
        editor._execute_command()
        assert ":help" in editor.message


class TestExternalEditorHatch:
    def test_external_editor_command_prefers_visual(self, monkeypatch):
        import chirp.cli

        monkeypatch.setenv("VISUAL", "code --wait")
        monkeypatch.setenv("EDITOR", "nano")
        assert chirp.cli._external_editor_command() == "code --wait"

    def test_external_editor_command_falls_back_to_editor(self, monkeypatch):
        import chirp.cli

        monkeypatch.delenv("VISUAL", raising=False)
        monkeypatch.setenv("EDITOR", "nano")
        assert chirp.cli._external_editor_command() == "nano"

    def test_external_editor_command_none_when_unset(self, monkeypatch):
        import chirp.cli

        monkeypatch.delenv("VISUAL", raising=False)
        monkeypatch.delenv("EDITOR", raising=False)
        assert chirp.cli._external_editor_command() is None

    def test_edit_in_external_editor_passes_notes_path(self, tmp_path, monkeypatch):
        import chirp.cli

        notes_path = tmp_path / "notes.md"
        notes_path.write_text("# x\n", encoding="utf-8")
        captured = {}

        def fake_run(argv, check=False):
            captured["argv"] = argv
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        ok = chirp.cli._edit_in_external_editor(notes_path, "nano")
        assert ok is True
        assert captured["argv"] == ["nano", str(notes_path)]

    def test_edit_in_external_editor_handles_launch_failure(
        self, tmp_path, monkeypatch
    ):
        import chirp.cli

        notes_path = tmp_path / "notes.md"
        notes_path.write_text("# x\n", encoding="utf-8")
        monkeypatch.setattr(
            "subprocess.run", MagicMock(side_effect=OSError("no such editor"))
        )
        assert chirp.cli._edit_in_external_editor(notes_path, "ghost-editor") is False


# ---------------------------------------------------------------------------
# AC-9 — shell completion enabled (alias completer reachable)
# ---------------------------------------------------------------------------


class TestCompletion:
    def test_completion_meta_commands_hidden_but_present(self):
        import chirp.cli

        result = CliRunner().invoke(chirp.cli.app, ["--help"])
        assert result.exit_code == 0
        # Hidden from the 7-command surface...
        assert "--install-completion" not in result.stdout
        assert "--show-completion" not in result.stdout

    def test_completion_enabled_meta_options_exist_but_hidden(self):
        import click

        import chirp.cli

        command = typer_main_command(chirp.cli.app)
        ctx = click.Context(command)
        params = command.get_params(ctx)
        meta = {
            opt
            for param in params
            for opt in getattr(param, "opts", [])
            if opt in {"--install-completion", "--show-completion"}
        }
        # Completion is enabled, so both meta-options are present on the app...
        assert meta == {"--install-completion", "--show-completion"}
        # ...but every one of them is hidden (kept off the 7-command surface).
        for param in params:
            if set(getattr(param, "opts", [])) & meta:
                assert param.hidden is True

    def test_show_completion_is_not_an_unknown_option(self):
        import chirp.cli

        # add_completion=True means --show-completion is a recognized option
        # (the generated completer is reachable), not an unknown-option error.
        result = CliRunner().invoke(chirp.cli.app, ["--show-completion"])
        assert "No such option" not in result.output


# ---------------------------------------------------------------------------
# AC-10 — AppleScript escaping
# ---------------------------------------------------------------------------


class TestAppleScriptEscaping:
    def test_escape_helper_escapes_quotes_and_backslashes(self):
        from utils.popup_manager import _escape_applescript

        assert _escape_applescript('say "hi"') == 'say \\"hi\\"'
        assert _escape_applescript("a\\b") == "a\\\\b"

    def test_notification_script_has_escaped_args(self):
        from utils.popup_manager import PopupManager

        with patch("utils.popup_manager.platform.system", return_value="Darwin"):
            popup = PopupManager()
        with patch("utils.popup_manager.subprocess.run") as run:
            run.return_value = MagicMock()
            popup._show_macos_notification('Ti"tle', 'mes"sage\\x')
            script = run.call_args[0][0][2]
        # The raw unescaped quote must never appear bare in the literal.
        assert '\\"' in script
        assert 'mes\\"sage' in script

    def test_dialog_script_has_escaped_args(self):
        from utils.popup_manager import PopupManager

        with patch("utils.popup_manager.platform.system", return_value="Darwin"):
            popup = PopupManager()
        mock_result = MagicMock()
        mock_result.stdout = "button returned:Yes"
        with patch(
            "utils.popup_manager.subprocess.run", return_value=mock_result
        ) as run:
            popup.ask_yes_no('Ti"tle', 'ques"tion')
            script = run.call_args[0][0][2]
        assert 'ques\\"tion' in script


# ---------------------------------------------------------------------------
# AC-11 — ask dual-input conflict warning
# ---------------------------------------------------------------------------


class TestAskDualInput:
    def _patch(self, monkeypatch):
        import notes_chat.cli as nc_cli

        monkeypatch.setattr(nc_cli, "get_notes_config", MagicMock)

        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": True,
                "context": "ctx",
                "sources": [],
                "retrieved_ids": ["c1"],
            }

        def fake_generate(config, question, context):
            return {"success": True, "answer": f"answered: {question}"}

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        monkeypatch.setattr("notes_chat.prompting.generate_answer", fake_generate)
        monkeypatch.setattr("notes_chat.cache.get_cached_answer", lambda *a: None)
        monkeypatch.setattr("notes_chat.cache.cache_answer", lambda *a: None)

    def test_both_inputs_warns_and_uses_positional(self, monkeypatch):
        import chirp.cli

        self._patch(monkeypatch)
        result = CliRunner().invoke(
            chirp.cli.app,
            ["ask", "--json", "positional-q", "--question", "option-q"],
        )
        assert result.exit_code == 0
        # The warning lands on stderr; the positional wins.
        assert "using the positional" in result.stderr
        payload = json.loads(result.stdout)
        assert payload["question"] == "positional-q"

    def test_option_alone_still_works(self, monkeypatch):
        import chirp.cli

        self._patch(monkeypatch)
        result = CliRunner().invoke(
            chirp.cli.app, ["ask", "--json", "--question", "option-q"]
        )
        assert result.exit_code == 0
        assert "using the positional" not in result.stderr
        assert json.loads(result.stdout)["question"] == "option-q"
