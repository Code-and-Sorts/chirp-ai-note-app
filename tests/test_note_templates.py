import pytest

from notes.note_templates import (
    DEFAULT_TEMPLATE_NAME,
    ROOT_TAG,
    TITLE_TAG,
    NoteTemplate,
    TemplateError,
    TemplateLoader,
    build_system_prompt,
    parse_template,
)

STANDUP = """---
description: "Daily standup"
tags:
- dsu
- standup
---
## {title}

**Time:** {time}
**Duration:** {duration}

### Yesterday

{yesterday}

### Today

{today}

### Blockers

{blockers}

---
"""


class TestParseTemplate:
    def test_parses_frontmatter_and_sections(self):
        template = parse_template("standup", STANDUP)

        assert template.description == "Daily standup"
        assert template.tags == ("dsu", "standup")
        assert [s.key for s in template.sections] == ["yesterday", "today", "blockers"]
        assert [s.tag for s in template.sections] == ["YESTERDAY", "TODAY", "BLOCKERS"]
        assert [s.heading for s in template.sections] == [
            "Yesterday",
            "Today",
            "Blockers",
        ]
        assert all(s.kind == "list" for s in template.sections)

    def test_flow_style_tags(self):
        template = parse_template(
            "t", "---\ntags: [dsu, standup]\n---\n### Items\n\n{items}\n"
        )
        assert template.tags == ("dsu", "standup")

    def test_prose_override_and_conventions(self):
        content = (
            "---\nprose:\n- overview\n---\n"
            "### Overview\n\n{overview}\n\n"
            "### Summary\n\n{summary}\n\n"
            "### Points\n\n{points}\n"
        )
        template = parse_template("t", content)
        kinds = {s.key: s.kind for s in template.sections}
        assert kinds == {"overview": "prose", "summary": "prose", "points": "list"}

    def test_action_items_kind(self):
        template = parse_template("t", "### Action Items\n\n{action_items}\n")
        assert template.sections[0].kind == "action_list"

    def test_reserved_placeholders_are_not_sections(self):
        template = parse_template(
            "t", "## {title}\n{time} {duration}\n### Items\n\n{items}\n"
        )
        assert [s.key for s in template.sections] == ["items"]

    def test_heading_fallback_is_titleized_key(self):
        template = parse_template("t", "{open_questions}\n")
        assert template.sections[0].heading == "Open Questions"

    def test_no_sections_raises(self):
        with pytest.raises(TemplateError, match="no sections"):
            parse_template("t", "## {title}\n\nJust prose.\n")

    def test_unclosed_frontmatter_raises(self):
        with pytest.raises(TemplateError, match="never closed"):
            parse_template("t", "---\ntags:\n- a\n### X\n\n{x}\n")

    def test_bad_frontmatter_line_raises(self):
        with pytest.raises(TemplateError, match="expected 'key: value'"):
            parse_template("t", "---\nnot yaml at all\n---\n### X\n\n{x}\n")

    def test_list_item_without_key_raises(self):
        with pytest.raises(TemplateError, match="without a key"):
            parse_template("t", "---\n- dsu\n---\n### X\n\n{x}\n")

    def test_template_without_frontmatter(self):
        template = parse_template("t", "### Items\n\n{items}\n")
        assert template.tags == ()
        assert template.description == ""

    def test_list_keys_and_summary_key(self):
        template = parse_template(
            "t",
            "### Summary\n\n{summary}\n\n### Items\n\n{items}\n\n"
            "### Action Items\n\n{action_items}\n",
        )
        assert template.list_keys == ("items", "action_items")
        assert template.summary_key == "summary"


class TestBuildSystemPrompt:
    def test_contract_lists_sections_in_order(self):
        template = parse_template("standup", STANDUP)
        prompt = build_system_prompt(template)

        assert f"<{ROOT_TAG}> ... </{ROOT_TAG}>" in prompt
        assert f"1) <{TITLE_TAG}>...</{TITLE_TAG}>" in prompt
        assert "2) <YESTERDAY> <ITEM>...</ITEM> ... </YESTERDAY>" in prompt
        assert "4) <BLOCKERS> <ITEM>...</ITEM> ... </BLOCKERS>" in prompt
        assert '"Yesterday"' in prompt

    def test_action_list_and_prose_shapes(self):
        template = parse_template(
            "t",
            "### Summary\n\n{summary}\n\n### Action Items\n\n{action_items}\n",
        )
        prompt = build_system_prompt(template)
        assert "<SUMMARY>...</SUMMARY>" in prompt
        assert '<ACTION_ITEMS> <ITEM task="..." owner="..." deadline="..."/>' in prompt
        assert '• If owner missing: "Unassigned"' in prompt


@pytest.fixture
def loader(tmp_path):
    return TemplateLoader(user_dir=tmp_path / "templates")


class TestTemplateLoader:
    def test_available_includes_builtins(self, loader):
        assert {"meeting", "standup", "one-on-one", "brainstorm"} <= set(
            loader.available()
        )

    def test_builtins_ship_without_tags(self, loader):
        for name in ("meeting", "standup", "one-on-one", "brainstorm"):
            assert loader.load(name).tags == ()

    def test_scaffold_writes_builtins_once(self, loader):
        written = loader.scaffold()
        assert {path.stem for path in written} >= {"meeting", "standup"}

        standup = loader.user_dir / "standup.md"
        standup.write_text("---\ntags: [dsu]\n---\n### X\n\n{x}\n", encoding="utf-8")
        assert loader.scaffold() == []
        assert "dsu" in standup.read_text(encoding="utf-8")

    def test_user_file_shadows_builtin(self, loader):
        loader.user_dir.mkdir(parents=True)
        (loader.user_dir / "standup.md").write_text(
            "---\ntags: [dsu]\n---\n### Wins\n\n{wins}\n", encoding="utf-8"
        )
        template = loader.load("standup")
        assert template.tags == ("dsu",)
        assert [s.key for s in template.sections] == ["wins"]

    def test_broken_user_file_falls_back_to_builtin(self, loader):
        loader.user_dir.mkdir(parents=True)
        (loader.user_dir / "standup.md").write_text(
            "---\nbroken frontmatter\n---\n", encoding="utf-8"
        )
        template = loader.load("standup")
        assert [s.key for s in template.sections][0] == "yesterday"

    def test_broken_user_file_without_builtin_falls_back_to_default(self, loader):
        loader.user_dir.mkdir(parents=True)
        (loader.user_dir / "retro.md").write_text("---\nbad\n---\n", encoding="utf-8")
        template = loader.load("retro")
        assert template.name == DEFAULT_TEMPLATE_NAME

    def test_unknown_name_raises_with_available(self, loader):
        with pytest.raises(TemplateError, match="unknown template 'nope'.*meeting"):
            loader.load("nope")


def _write(loader: TemplateLoader, name: str, tags: list[str]) -> None:
    loader.user_dir.mkdir(parents=True, exist_ok=True)
    tag_lines = "".join(f"- {tag}\n" for tag in tags)
    (loader.user_dir / f"{name}.md").write_text(
        f"---\ntags:\n{tag_lines}---\n### Items\n\n{{items}}\n", encoding="utf-8"
    )


class TestSelection:
    def test_any_of_tag_match(self, loader):
        _write(loader, "standup", ["dsu", "standup"])
        assert loader.match_by_tags(["dsu"]).name == "standup"
        assert loader.match_by_tags(["standup", "work"]).name == "standup"

    def test_no_match_returns_none(self, loader):
        _write(loader, "standup", ["dsu"])
        assert loader.match_by_tags(["meeting-tag"]) is None
        assert loader.match_by_tags([]) is None

    def test_most_shared_tags_wins(self, loader):
        _write(loader, "standup", ["daily"])
        _write(loader, "team-sync", ["daily", "sync"])
        assert loader.match_by_tags(["daily", "sync"]).name == "team-sync"

    def test_tie_breaks_by_name(self, loader):
        _write(loader, "zebra", ["shared"])
        _write(loader, "alpha", ["shared"])
        assert loader.match_by_tags(["shared"]).name == "alpha"

    def test_resolve_precedence(self, loader):
        _write(loader, "standup", ["dsu"])

        assert loader.resolve(None, [], None).name == DEFAULT_TEMPLATE_NAME
        assert loader.resolve(None, ["dsu"], None).name == "standup"
        assert loader.resolve("brainstorm", ["dsu"], None).name == "brainstorm"
        assert loader.resolve("brainstorm", ["dsu"], "one-on-one").name == "one-on-one"

    def test_resolve_unknown_meta_falls_through(self, loader):
        _write(loader, "standup", ["dsu"])
        assert loader.resolve("ghost", ["dsu"], None).name == "standup"
        assert loader.resolve("ghost", [], None).name == DEFAULT_TEMPLATE_NAME


def test_note_template_is_immutable():
    template = parse_template("t", "### Items\n\n{items}\n")
    assert isinstance(template, NoteTemplate)
    with pytest.raises(AttributeError):
        template.name = "other"
