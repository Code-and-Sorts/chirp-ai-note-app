from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner


def _build_settings(tmp_path: Path) -> SimpleNamespace:
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    return SimpleNamespace(
        directories=SimpleNamespace(notes_root=notes_root),
        notes_chat=SimpleNamespace(index_dir=tmp_path / "index"),
    )


def _seed_note(
    settings: SimpleNamespace,
    slug: str,
    created_at: datetime,
    title: str,
    transcript: str | None = None,
    notes_md: str | None = None,
) -> Path:
    note_dir = settings.directories.notes_root / slug
    note_dir.mkdir()
    if transcript is not None:
        (note_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    if notes_md is not None:
        (note_dir / "notes.md").write_text(notes_md, encoding="utf-8")
    meta_lines = [f'title = "{title}"', f'date = "{created_at.isoformat()}"']
    (note_dir / "meta.toml").write_text("\n".join(meta_lines), encoding="utf-8")
    return Path(note_dir)


@pytest.fixture
def seeded_settings(tmp_path: Path) -> SimpleNamespace:
    settings = _build_settings(tmp_path)
    now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    _seed_note(
        settings,
        slug="call-w-jamie-recent",
        created_at=now,
        title="call w/ jamie",
        transcript=(
            "drop the free tier, raise pricing from $8 to $12\n"
            "team pricing stays per-seat for now\n"
        ),
        notes_md=(
            "tl;dr: pricing overhaul. drop free, raise pro.\n"
            "next: review with finance.\n"
            "draft pricing page\n"
        ),
    )
    _seed_note(
        settings,
        slug="walk-thought-pricing-recent",
        created_at=now - timedelta(days=4),
        title="walk thought: pricing",
        transcript="thinking about tier structure and pricing psychology\n",
        notes_md="anchor pricing matters more than the number itself\n",
    )
    _seed_note(
        settings,
        slug="old-note-ancient",
        created_at=now - timedelta(days=120),
        title="old note",
        transcript="we discussed the pricing roadmap last quarter\n",
        notes_md="historical pricing review\n",
    )
    return settings


def test_run_search_finds_hits_across_files(seeded_settings):
    from notes_chat.search_keyword import SearchOptions, run_search

    result = run_search(seeded_settings, SearchOptions(query="pricing"))

    assert result["total_notes_scanned"] == 3
    assert len(result["matches"]) == 3
    assert result["matches"][0]["title"] == "call w/ jamie"
    assert result["matches"][0]["hits"] >= 4

    sources = {e["source"] for m in result["matches"] for e in m["excerpts"]}
    assert sources == {"transcript", "notes"}


def test_since_filter_excludes_old_notes(seeded_settings):
    from notes_chat.search_keyword import SearchOptions, run_search

    result = run_search(
        seeded_settings,
        SearchOptions(query="pricing", since_minutes=30 * 24 * 60),
    )
    titles = [m["title"] for m in result["matches"]]
    assert "old note" not in titles
    assert "call w/ jamie" in titles


def test_regex_query(seeded_settings):
    from notes_chat.search_keyword import SearchOptions, run_search

    result = run_search(
        seeded_settings,
        SearchOptions(query=r"pric\w+", regex=True),
    )
    assert any(m["hits"] for m in result["matches"])


def test_no_hits_returns_empty_matches(seeded_settings):
    from notes_chat.search_keyword import SearchOptions, run_search

    result = run_search(seeded_settings, SearchOptions(query="acquisition"))
    assert result["matches"] == []


def test_excerpt_window_truncates_long_lines(tmp_path):
    from notes_chat.search_keyword import _window_excerpt

    long_line = "x" * 200 + " pricing " + "y" * 200
    match_start = long_line.index("pricing")
    out = _window_excerpt(long_line, match_start, match_start + len("pricing"))
    assert "pricing" in out
    assert len(out) <= 122
    assert out.startswith("…")
    assert out.endswith("…")


def test_excerpt_short_line_unchanged(tmp_path):
    from notes_chat.search_keyword import _window_excerpt

    line = "short pricing line"
    match_start = line.index("pricing")
    assert _window_excerpt(line, match_start, match_start + len("pricing")) == line


def test_humanize_duration():
    from notes_chat.search_keyword import _humanize_duration

    assert _humanize_duration(30 * 24 * 60) == "30 days"
    assert _humanize_duration(2 * 7 * 24 * 60) == "2 weeks"
    assert _humanize_duration(48 * 60) == "2 days"
    assert _humanize_duration(60) == "1 hour"


def test_levenshtein_close_keywords_filters_query_tokens():
    from notes_chat.search_keyword import _is_close

    assert _is_close("growth", "growth") is True
    assert _is_close("priced", "pricing") is True
    assert _is_close("pricing", "pricin") is True
    assert _is_close("xylophone", "pricing") is False


def test_literal_query_matches_same_tokens_as_bm25(tmp_path):
    """`chirp search` matches the tokens BM25 would score for a hyphenated ID.

    "JIRA-1234" tokenizes to ["jira", "1234"], so the literal search finds a note
    that contains the ID even though it never appears as one un-split substring.
    """
    from notes_chat.bm25 import _tokenize_document
    from notes_chat.search_keyword import SearchOptions, run_search

    settings = _build_settings(tmp_path)
    now = datetime.now()
    _seed_note(
        settings,
        slug="jira-note-recent",
        created_at=now,
        title="jira note",
        transcript="JIRA-1234 must ship before the cutoff\n",
        notes_md="owner follows up on the ticket\n",
    )

    assert _tokenize_document("JIRA-1234") == ["jira", "1234"]
    result = run_search(settings, SearchOptions(query="JIRA-1234"))
    assert result["matches"]
    assert result["matches"][0]["hits"] >= 1


def test_literal_query_uses_whole_word_matching(tmp_path):
    """A literal search agrees with BM25's token equality, not substrings.

    BM25 scores the token "tier" distinctly from "tiers", so searching "tier"
    must not match a note whose only near-form is "tiers" — matching what
    `chirp ask` would retrieve.
    """
    from notes_chat.search_keyword import SearchOptions, run_search

    settings = _build_settings(tmp_path)
    now = datetime.now()
    _seed_note(
        settings,
        slug="tiers-note-recent",
        created_at=now,
        title="tiers note",
        transcript="we compared the pricing tiers carefully\n",
        notes_md="no standalone keyword here\n",
    )

    assert run_search(settings, SearchOptions(query="tier"))["matches"] == []
    assert run_search(settings, SearchOptions(query="tiers"))["matches"]


def test_cjk_query_matches_bm25_tokens_not_substrings(tmp_path):
    """Token matching is Unicode-safe and tracks BM25 across scripts.

    `_tokenize_document` splits on whitespace/punctuation, so "日本語" is a token
    only when whitespace-separated. A literal search must match exactly that —
    not a substring inside a longer CJK run, which `\\b` would also reject but
    only by accident of ASCII assumptions.
    """
    from notes_chat.bm25 import _tokenize_document
    from notes_chat.search_keyword import SearchOptions, run_search

    assert _tokenize_document("読む 日本語 です") == ["読む", "日本語", "です"]
    assert _tokenize_document("読む日本語です") == ["読む日本語です"]

    settings = _build_settings(tmp_path)
    now = datetime.now()
    _seed_note(
        settings,
        slug="cjk-spaced-recent",
        created_at=now,
        title="cjk spaced",
        transcript="読む 日本語 です\n",
        notes_md="meeting\n",
    )
    _seed_note(
        settings,
        slug="cjk-joined-recent",
        created_at=now,
        title="cjk joined",
        transcript="読む日本語です\n",
        notes_md="meeting\n",
    )

    titles = {
        m["title"]
        for m in run_search(settings, SearchOptions(query="日本語"))["matches"]
    }
    assert titles == {"cjk spaced"}


def test_underscore_token_matches_whole_token_only(tmp_path):
    """`user_name` is one BM25 token (underscore is a word char), so a search for
    `name` must not match it — exactly as `chirp ask` would score it."""
    from notes_chat.bm25 import _tokenize_document
    from notes_chat.search_keyword import SearchOptions, run_search

    assert _tokenize_document("user_name") == ["user_name"]

    settings = _build_settings(tmp_path)
    _seed_note(
        settings,
        slug="snake-case-recent",
        created_at=datetime.now(),
        title="snake case",
        transcript="the user_name field is required\n",
        notes_md="meeting\n",
    )

    assert run_search(settings, SearchOptions(query="name"))["matches"] == []
    assert run_search(settings, SearchOptions(query="user_name"))["matches"]


def test_multi_word_literal_query_matches_any_token(tmp_path):
    """A multi-word literal query matches notes containing ANY of its tokens
    (BM25's OR semantics), not only the adjacent phrase."""
    from notes_chat.search_keyword import SearchOptions, run_search

    settings = _build_settings(tmp_path)
    _seed_note(
        settings,
        slug="pricing-only-recent",
        created_at=datetime.now(),
        title="pricing only",
        transcript="the pricing review is tomorrow\n",
        notes_md="meeting\n",
    )

    result = run_search(settings, SearchOptions(query="pricing roadmap"))
    assert {m["title"] for m in result["matches"]} == {"pricing only"}


def _invoke_search(args: list[str], settings):
    import importlib

    import chirp.cli as cli_module

    importlib.reload(cli_module)
    runner = CliRunner()
    with patch.object(cli_module, "get_settings", return_value=settings):
        return runner.invoke(cli_module.app, ["search", *args])


def test_cli_empty_query_exits_2(tmp_path):
    settings = _build_settings(tmp_path)
    result = _invoke_search(["   "], settings)
    assert result.exit_code == 2
    assert "search query is required" in result.output


def test_cli_invalid_since(seeded_settings):
    result = _invoke_search(["pricing", "--since", "5m"], seeded_settings)
    assert result.exit_code == 2
    assert "invalid --since" in result.output


def test_cli_invalid_regex(seeded_settings):
    result = _invoke_search(["[unclosed", "--regex"], seeded_settings)
    assert result.exit_code == 2
    assert "invalid regex" in result.output


def test_cli_json_output(seeded_settings):
    result = _invoke_search(["pricing", "--json"], seeded_settings)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["query"] == "pricing"
    assert payload["since"] is None
    assert payload["regex"] is False
    assert isinstance(payload["matches"], list)
    assert len(payload["matches"]) == 3
    first_excerpt = payload["matches"][0]["excerpts"][0]
    assert set(first_excerpt) == {"source", "line", "text"}


def test_cli_renders_results(seeded_settings):
    result = _invoke_search(["pricing"], seeded_settings)
    assert result.exit_code == 0
    assert "searching 3 notes for" in result.output
    assert "call w/ jamie" in result.output
    assert "transcript" in result.output
    assert "notes.md:" in result.output


def test_cli_no_hits_renders_empty_block(seeded_settings):
    result = _invoke_search(["acquisition strategy"], seeded_settings)
    assert result.exit_code == 0
    assert "no exact matches." in result.output
    assert 'chirp ask "acquisition strategy"' in result.output


def test_cli_since_renders_scope_line(seeded_settings):
    result = _invoke_search(["pricing", "--since", "30d"], seeded_settings)
    assert result.exit_code == 0
    assert "scope: last 30 days" in result.output
    assert "case-insensitive" not in result.output
