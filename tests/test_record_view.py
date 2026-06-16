"""Unit tests for the chirp record polish (story 1.6)."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from chirp.cli import (
    _parse_tag_input,
    _RecordViewState,
    _render_record_view,
    _render_waveform_box,
)


class TestParseTagInput:
    def test_empty_input(self):
        assert _parse_tag_input("") == []

    def test_single_tag(self):
        assert _parse_tag_input("meeting") == ["meeting"]

    def test_two_tags(self):
        assert _parse_tag_input("a, b") == ["a", "b"]

    def test_strips_whitespace_and_drops_empties(self):
        assert _parse_tag_input("  a ,b , c , ") == ["a", "b", "c"]


def _render(state: _RecordViewState) -> str:
    buf = StringIO()
    Console(file=buf, force_terminal=False, width=80).print(_render_record_view(state))
    return buf.getvalue()


class TestRecordView:
    def test_default_state_renders_listening(self):
        state = _RecordViewState(
            title="Standup",
            cap_minutes=5,
            mic_name="Built-in Mic",
        )
        out = _render(state)
        assert "[Standup]" in out
        assert "● REC" in out
        assert "00:00" in out
        assert "/ 05:00" in out
        assert "Built-in Mic" in out
        assert "listening" in out
        assert "[space] pause" in out
        assert "[q / ^C] stop & save" in out
        assert "[x] discard" in out

    def test_paused_state_flips_status(self):
        state = _RecordViewState(
            title="Standup", cap_minutes=None, mic_name="x", paused=True
        )
        out = _render(state)
        assert "paused" in out
        assert "listening" not in out

    def test_stopped_by_cap_state(self):
        state = _RecordViewState(
            title="Standup",
            cap_minutes=5,
            mic_name="x",
            stopped_by_cap=True,
        )
        out = _render(state)
        assert "stopped" in out

    def test_no_title_omits_title_line(self):
        state = _RecordViewState(title=None, cap_minutes=None, mic_name="x")
        out = _render(state)
        assert "[" not in out.split("\n")[0]

    def test_waveform_box_chrome_present(self):
        # Chrome border characters from the Panel renderer.
        from collections import deque

        from chirp.cli import WAVEFORM_WIDTH

        levels = deque([0.5] * WAVEFORM_WIDTH, maxlen=WAVEFORM_WIDTH)
        buf = StringIO()
        Console(file=buf, force_terminal=False, width=80).print(
            _render_waveform_box(levels)
        )
        out = buf.getvalue()
        assert "waveform" in out

    def test_waveform_glyphs_reflect_per_slot_levels(self):
        from collections import deque

        from chirp.cli import WAVEFORM_GLYPHS, WAVEFORM_WIDTH

        # Half of the buffer silent, the other half loud — verify both
        # extremes appear in the output.
        levels = deque(
            [0.0] * (WAVEFORM_WIDTH // 2) + [0.95] * (WAVEFORM_WIDTH // 2),
            maxlen=WAVEFORM_WIDTH,
        )
        buf = StringIO()
        Console(file=buf, force_terminal=False, width=80).print(
            _render_waveform_box(levels)
        )
        out = buf.getvalue()
        assert "▁" in out
        assert WAVEFORM_GLYPHS[-1] in out


class TestRecordTagFlag:
    """Drive `chirp record --tag` end-to-end via CliRunner."""

    def test_repeated_tag_flags_reach_recorder(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from chirp.cli import app
        from config.settings import ChirpSettings

        captured: dict = {}

        class FakeRecorder:
            def __init__(self, settings):
                self.note_dir = tmp_path / "fake-2026-04-27"
                self.note_dir.mkdir()
                self.start_time = None
                self._paused = False

            @property
            def is_paused(self) -> bool:
                return bool(self._paused)

            def pause(self) -> None:
                self._paused = True

            def resume(self) -> None:
                self._paused = False

            def stop_recording(self) -> None:
                pass

            def start_recording(
                self,
                duration_minutes=None,
                title=None,
                level_callback=None,
                tags=None,
            ) -> str:
                captured["tags"] = list(tags or [])
                captured["title"] = title
                captured["duration_minutes"] = duration_minutes
                return str(self.note_dir / "audio.wav")

        class FakeDeviceManager:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get_default_input_device(self):
                return 0

            def list_devices(self):
                return [{"index": 0, "name": "Mic"}]

        settings = ChirpSettings()
        settings.directories.notes_root = tmp_path
        monkeypatch.setattr("chirp.cli.get_settings", lambda: settings)
        monkeypatch.setattr("recorder.audio_recorder.AudioRecorder", FakeRecorder)
        monkeypatch.setattr("recorder.device_manager.DeviceManager", FakeDeviceManager)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "record",
                "--title",
                "T",
                "--duration",
                "1",
                "--tag",
                "a",
                "--tag",
                "b",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert captured["tags"] == ["a", "b"]
        assert captured["title"] == "T"
