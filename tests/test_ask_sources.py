"""Story 1.7 — sources formatter (`note #N (mm:ss)`)."""

from pathlib import Path

import tomli_w

from notes_chat.retrieval import _build_note_index, format_sources


def _seed_note(tmp_path: Path, slug: str, date: str) -> Path:
    note_dir = tmp_path / slug
    note_dir.mkdir()
    (note_dir / "notes.md").write_text("# x\n", encoding="utf-8")
    with (note_dir / "meta.toml").open("wb") as fh:
        tomli_w.dump({"title": slug, "date": date, "tags": []}, fh)
    return note_dir


def _settings(tmp_path: Path):
    from config.settings import ChirpSettings

    settings = ChirpSettings()
    settings.directories.notes_root = tmp_path
    return settings


def _chunk(slug: str, **metadata):
    return (
        f"chunk-{slug}",
        "# header\nbody\n",
        {
            "metadata": {"path": f"/abs/{slug}/notes.md", **metadata},
        },
    )


class TestFormatSources:
    def test_indexes_match_chirp_notes_newest_first(self, tmp_path):
        # Two notes; newest by date should be #1.
        _seed_note(tmp_path, "older-2026-04-20", "2026-04-20T09:00:00")
        _seed_note(tmp_path, "newer-2026-04-21", "2026-04-21T09:00:00")
        note_index = _build_note_index(_settings(tmp_path))

        chunks = [_chunk("newer-2026-04-21")]
        sources = format_sources(chunks, note_index)
        assert sources == ["note #1"]

        chunks = [_chunk("older-2026-04-20")]
        sources = format_sources(chunks, note_index)
        assert sources == ["note #2"]

    def test_omits_timestamp_when_metadata_lacks_one(self, tmp_path):
        _seed_note(tmp_path, "demo-2026-04-20", "2026-04-20T09:00:00")
        note_index = _build_note_index(_settings(tmp_path))

        sources = format_sources([_chunk("demo-2026-04-20")], note_index)
        assert sources == ["note #1"]

    def test_appends_mm_ss_when_chunk_has_start_ms(self, tmp_path):
        _seed_note(tmp_path, "demo-2026-04-20", "2026-04-20T09:00:00")
        note_index = _build_note_index(_settings(tmp_path))

        chunks = [_chunk("demo-2026-04-20", start_ms=760000)]
        sources = format_sources(chunks, note_index)
        assert sources == ["note #1 (12:40)"]

    def test_collapses_chunks_from_same_note_keeps_earliest_timestamp(self, tmp_path):
        _seed_note(tmp_path, "demo-2026-04-20", "2026-04-20T09:00:00")
        note_index = _build_note_index(_settings(tmp_path))

        chunks = [
            _chunk("demo-2026-04-20", start_ms=120000),  # 02:00
            _chunk("demo-2026-04-20", start_ms=30000),  # 00:30 (earlier)
            _chunk("demo-2026-04-20", start_ms=600000),  # 10:00
        ]
        sources = format_sources(chunks, note_index)
        assert sources == ["note #1 (00:30)"]

    def test_falls_back_to_slug_when_index_missing(self):
        chunks = [_chunk("orphan-slug")]
        sources = format_sources(chunks, note_index={})
        assert sources == ["orphan-slug"]


class TestInteractiveBanner:
    def test_banner_includes_note_count_and_model(self, tmp_path, monkeypatch):
        from io import StringIO

        from rich.console import Console as RichConsole

        from notes_chat.interactive import InteractiveChatSession

        _seed_note(tmp_path, "first-2026-04-20", "2026-04-20T09:00:00")
        _seed_note(tmp_path, "second-2026-04-21", "2026-04-21T09:00:00")

        config = _settings(tmp_path)
        config.models.llm = "llama3.1:8b"

        captured = StringIO()
        rich_console = RichConsole(file=captured, force_terminal=False, width=120)
        monkeypatch.setattr("notes_chat.interactive.console", rich_console)

        session = InteractiveChatSession(config)

        # Don't actually start the prompt loop — just exercise the banner.
        # `start` blocks on input, so call the helpers directly.
        rich_console.print(
            f"Chirp · chat over {session._count_notes()} notes · "
            f"{config.models.llm} (local)"
        )
        rich_console.print("type your question, or /help · ctrl+d to exit")

        out = captured.getvalue()
        assert "chat over 2 notes" in out
        assert "llama3.1:8b (local)" in out
        assert "type your question, or /help · ctrl+d to exit" in out


class TestAskMarkdownToggle:
    def _stub_pipeline(self, monkeypatch, answer: str = "**bold** and _italic_"):
        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": True,
                "context": "ctx",
                "sources": ["note #1"],
                "retrieved_ids": ["c1"],
            }

        def fake_generate(config, question, context):
            return {"success": True, "answer": answer}

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        monkeypatch.setattr("notes_chat.prompting.generate_answer", fake_generate)
        monkeypatch.setattr("notes_chat.cache.get_cached_answer", lambda *args: None)
        monkeypatch.setattr("notes_chat.cache.cache_answer", lambda *args: None)

    def test_markdown_default_renders_as_markdown(self, monkeypatch, tmp_path):
        from typer.testing import CliRunner

        from notes_chat.cli import app

        rendered = {"calls": 0}

        class FakeMarkdown:
            def __init__(self, text):
                rendered["calls"] += 1
                rendered["text"] = text

            def __rich_console__(self, console, options):
                yield rendered["text"]

        self._stub_pipeline(monkeypatch, answer="**hi**")
        monkeypatch.setattr("rich.markdown.Markdown", FakeMarkdown)

        result = CliRunner().invoke(app, ["ask", "-q", "what?"])
        assert result.exit_code == 0
        assert rendered["calls"] == 1
        assert rendered["text"] == "**hi**"

    def test_no_markdown_prints_answer_verbatim(self, monkeypatch):
        from typer.testing import CliRunner

        from notes_chat.cli import app

        boom = {"called": False}

        class BoomMarkdown:
            def __init__(self, *args, **kwargs):
                boom["called"] = True

        self._stub_pipeline(monkeypatch, answer="**raw markdown**")
        monkeypatch.setattr("rich.markdown.Markdown", BoomMarkdown)

        result = CliRunner().invoke(app, ["ask", "-q", "what?", "--no-markdown"])
        assert result.exit_code == 0
        assert boom["called"] is False
        assert "**raw markdown**" in result.stdout

    def test_sources_line_uses_ac3_format(self, monkeypatch):
        from typer.testing import CliRunner

        from notes_chat.cli import app

        self._stub_pipeline(monkeypatch, answer="answer")
        result = CliRunner().invoke(app, ["ask", "-q", "what?", "--no-markdown"])
        assert result.exit_code == 0
        assert "sources: note #1" in result.stdout
        assert "📚" not in result.stdout

    def test_no_sources_flag_omits_footer(self, monkeypatch):
        from typer.testing import CliRunner

        from notes_chat.cli import app

        self._stub_pipeline(monkeypatch, answer="answer")
        result = CliRunner().invoke(
            app, ["ask", "-q", "what?", "--no-markdown", "--no-sources"]
        )
        assert result.exit_code == 0
        assert "sources:" not in result.stdout
