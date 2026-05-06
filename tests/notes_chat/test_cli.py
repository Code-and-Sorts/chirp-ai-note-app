"""Unit tests for notes_chat/cli.py covering index and ask commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from notes_chat.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_config():
    cfg = MagicMock()
    return cfg


def _patch_config(monkeypatch):
    cfg = _fake_config()
    monkeypatch.setattr("notes_chat.cli.get_notes_config", lambda: cfg)
    return cfg


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


class TestIndexCommand:
    def _patch_build_index(self, monkeypatch, result: dict):
        monkeypatch.setattr("notes_chat.cli.get_notes_config", lambda: _fake_config())

        def fake_build(config, force=False, progress_callback=None):
            if progress_callback is not None:
                progress_callback()
            return result

        monkeypatch.setattr("notes_chat.index.build_index", fake_build)
        with patch("notes_chat.cli.__import__"):
            pass
        return fake_build

    def test_index_success_exits_0(self, monkeypatch):
        _patch_config(monkeypatch)
        fake_build = MagicMock(return_value={"success": True, "files_processed": 5})
        with patch("notes_chat.cli.get_notes_config", return_value=_fake_config()):
            with patch("notes_chat.index.build_index", fake_build):
                with patch.dict(
                    "sys.modules",
                    {"notes_chat.index": MagicMock(build_index=fake_build)},
                ):
                    result = runner.invoke(app, ["index"])
        assert result.exit_code == 0
        assert "5 files processed" in result.output

    def test_index_success_with_force_flag(self, monkeypatch):
        _patch_config(monkeypatch)
        fake_build = MagicMock(return_value={"success": True, "files_processed": 3})
        with patch("notes_chat.cli.get_notes_config", return_value=_fake_config()):
            with patch.dict(
                "sys.modules",
                {"notes_chat.index": MagicMock(build_index=fake_build)},
            ):
                result = runner.invoke(app, ["index", "--force"])
        assert result.exit_code == 0
        assert "--force specified" in result.output

    def test_index_build_failure_exits_1(self, monkeypatch):
        _patch_config(monkeypatch)
        fake_build = MagicMock(return_value={"success": False, "error": "disk full"})
        with patch("notes_chat.cli.get_notes_config", return_value=_fake_config()):
            with patch.dict(
                "sys.modules",
                {"notes_chat.index": MagicMock(build_index=fake_build)},
            ):
                result = runner.invoke(app, ["index"])
        assert result.exit_code == 1
        assert "disk full" in result.output

    def test_index_exception_exits_1(self, monkeypatch):
        _patch_config(monkeypatch)
        fake_build = MagicMock(side_effect=RuntimeError("boom"))
        with patch("notes_chat.cli.get_notes_config", return_value=_fake_config()):
            with patch.dict(
                "sys.modules",
                {"notes_chat.index": MagicMock(build_index=fake_build)},
            ):
                result = runner.invoke(app, ["index"])
        assert result.exit_code == 1
        assert "boom" in result.output

    def test_index_progress_callback_increments(self, monkeypatch):
        """Progress callback is invoked and the file count increases."""
        _patch_config(monkeypatch)
        call_count = {"n": 0}

        def fake_build(config, force=False, progress_callback=None):
            if progress_callback:
                progress_callback()
                progress_callback()
                call_count["n"] = 2
            return {"success": True, "files_processed": 2}

        with patch("notes_chat.cli.get_notes_config", return_value=_fake_config()):
            with patch.dict(
                "sys.modules",
                {"notes_chat.index": MagicMock(build_index=fake_build)},
            ):
                result = runner.invoke(app, ["index"])
        assert result.exit_code == 0
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# ask command — interactive (no question)
# ---------------------------------------------------------------------------


class TestAskInteractiveMode:
    def test_no_question_starts_interactive_session(self, monkeypatch):
        _patch_config(monkeypatch)
        started = {"called": False}

        class FakeSession:
            def __init__(self, config, markdown=True):
                pass

            def start(self):
                started["called"] = True

        monkeypatch.setattr(
            "notes_chat.interactive.InteractiveChatSession", FakeSession
        )
        result = runner.invoke(app, ["ask"])
        assert result.exit_code == 0
        assert started["called"]

    def test_question_option_used_when_positional_absent(self, monkeypatch):
        _patch_config(monkeypatch)

        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": True,
                "context": "ctx",
                "sources": [],
                "retrieved_ids": ["c1"],
            }

        def fake_generate(config, question, context):
            return {"success": True, "answer": "from option"}

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        monkeypatch.setattr("notes_chat.prompting.generate_answer", fake_generate)
        monkeypatch.setattr("notes_chat.cache.get_cached_answer", lambda *a: None)
        monkeypatch.setattr("notes_chat.cache.cache_answer", lambda *a: None)

        result = runner.invoke(app, ["ask", "--question", "what happened?"])
        assert result.exit_code == 0
        assert "from option" in result.output


# ---------------------------------------------------------------------------
# ask command — retrieval failures
# ---------------------------------------------------------------------------


class TestAskRetrievalFailure:
    def _setup(self, monkeypatch):
        _patch_config(monkeypatch)

    def test_no_documents_found_prints_message(self, monkeypatch):
        self._setup(monkeypatch)

        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": False,
                "error": "no documents found for your query",
                "suggestion": "try broader terms",
            }

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        result = runner.invoke(app, ["ask", "no docs?"])
        # typer.Exit(2) is caught by the outer except Exception handler → exit code 1
        assert result.exit_code == 1
        assert "No relevant documents" in result.output

    def test_no_documents_found_without_suggestion(self, monkeypatch):
        self._setup(monkeypatch)

        def fake_retrieve(config, question, when_filter=None):
            return {"success": False, "error": "no documents found"}

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        result = runner.invoke(app, ["ask", "empty?"])
        assert result.exit_code == 1

    def test_generic_retrieval_error_exits_1(self, monkeypatch):
        self._setup(monkeypatch)

        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": False,
                "error": "index corrupted",
                "suggestion": "rebuild index",
            }

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        result = runner.invoke(app, ["ask", "something?"])
        assert result.exit_code == 1
        assert "index corrupted" in result.output
        assert "rebuild index" in result.output

    def test_generic_retrieval_error_without_suggestion(self, monkeypatch):
        self._setup(monkeypatch)

        def fake_retrieve(config, question, when_filter=None):
            return {"success": False, "error": "index corrupted"}

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        result = runner.invoke(app, ["ask", "something?"])
        assert result.exit_code == 1
        assert "index corrupted" in result.output


# ---------------------------------------------------------------------------
# ask command — dry_run
# ---------------------------------------------------------------------------


class TestAskDryRun:
    def _setup_retrieval(self, monkeypatch, context_len=2000):
        _patch_config(monkeypatch)
        context = "x" * context_len

        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": True,
                "context": context,
                "sources": ["note #1"],
                "retrieved_ids": ["c1", "c2"],
            }

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)

    def test_dry_run_prints_context_info_and_exits_0(self, monkeypatch):
        self._setup_retrieval(monkeypatch, context_len=500)
        result = runner.invoke(app, ["ask", "--dry-run", "what?"])
        assert result.exit_code == 0
        assert "dry run" in result.output
        assert "Context length: 500" in result.output
        assert "Retrieved chunks: 2" in result.output

    def test_dry_run_truncates_long_context(self, monkeypatch):
        self._setup_retrieval(monkeypatch, context_len=2000)
        result = runner.invoke(app, ["ask", "--dry-run", "what?"])
        assert result.exit_code == 0
        assert "..." in result.output

    def test_dry_run_skips_llm_call(self, monkeypatch):
        self._setup_retrieval(monkeypatch)
        generate_called = {"v": False}

        def fake_generate(*args, **kwargs):
            generate_called["v"] = True
            return {"success": True, "answer": "should not appear"}

        monkeypatch.setattr("notes_chat.prompting.generate_answer", fake_generate)
        runner.invoke(app, ["ask", "--dry-run", "what?"])
        assert not generate_called["v"]


# ---------------------------------------------------------------------------
# ask command — cached answer
# ---------------------------------------------------------------------------


class TestAskCachedAnswer:
    def test_uses_cached_answer_when_available(self, monkeypatch):
        _patch_config(monkeypatch)

        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": True,
                "context": "ctx",
                "sources": ["note #1"],
                "retrieved_ids": ["c1"],
            }

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        monkeypatch.setattr(
            "notes_chat.cache.get_cached_answer", lambda *a: "cached result"
        )
        monkeypatch.setattr("notes_chat.cache.cache_answer", lambda *a: None)

        generate_called = {"v": False}

        def boom(*args, **kwargs):
            generate_called["v"] = True
            return {"success": True, "answer": "fresh"}

        monkeypatch.setattr("notes_chat.prompting.generate_answer", boom)

        result = runner.invoke(app, ["ask", "--no-markdown", "what?"])
        assert result.exit_code == 0
        assert "cached result" in result.output
        assert "using cached answer" in result.output
        assert not generate_called["v"]


# ---------------------------------------------------------------------------
# ask command — generation failure
# ---------------------------------------------------------------------------


class TestAskGenerationFailure:
    def test_generation_failure_exits_1(self, monkeypatch):
        _patch_config(monkeypatch)

        def fake_retrieve(config, question, when_filter=None):
            return {
                "success": True,
                "context": "ctx",
                "sources": [],
                "retrieved_ids": ["c1"],
            }

        def fake_generate(config, question, context):
            return {"success": False, "error": "LLM timeout"}

        monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
        monkeypatch.setattr("notes_chat.prompting.generate_answer", fake_generate)
        monkeypatch.setattr("notes_chat.cache.get_cached_answer", lambda *a: None)

        result = runner.invoke(app, ["ask", "something?"])
        assert result.exit_code == 1
        assert "LLM timeout" in result.output


# ---------------------------------------------------------------------------
# ask command — outer exception handler
# ---------------------------------------------------------------------------


class TestAskOuterException:
    def test_unexpected_exception_exits_1_with_message(self, monkeypatch):
        _patch_config(monkeypatch)
        monkeypatch.setattr(
            "notes_chat.retrieval.retrieve_context",
            MagicMock(side_effect=RuntimeError("unexpected crash")),
        )
        result = runner.invoke(app, ["ask", "something?"])
        assert result.exit_code == 1
        assert "unexpected crash" in result.output
