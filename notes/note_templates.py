"""Note templates: user-editable markdown files that shape generated notes.

A template is a markdown file with YAML frontmatter::

    ---
    description: "Daily standup"
    tags:
    - dsu
    - standup
    ---
    ## {title}

    ### Yesterday

    {yesterday}

Frontmatter follows the Jekyll/Hugo convention — a ``---``-fenced block at
the top of the file parsed as YAML (``yaml.safe_load``). Sections are derived
from the body: each ``{placeholder}`` (except the built-ins ``{title}``,
``{time}``, ``{duration}``) becomes a section whose XML tag is the
upper-snake placeholder and whose heading is the nearest preceding markdown
heading. The frontmatter ``tags`` list links the template to note tags;
``prose`` names placeholders extracted as prose instead of bullet lists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml

from utils.file_utils import ensure_private_directory

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_NAME = "meeting"

BUILTIN_PROSE_KEYS = frozenset({"executive_summary", "summary"})
ACTION_LIST_KEY = "action_items"
RESERVED_KEYS = frozenset({"title", "time", "duration"})

_PLACEHOLDER_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_FM_BOUNDARY = re.compile(r"^-{3,}\s*$", re.MULTILINE)


class TemplateError(Exception):
    """A template file is missing or cannot be parsed."""


@dataclass(frozen=True)
class Section:
    key: str
    tag: str
    heading: str
    kind: Literal["prose", "list", "action_list"]


@dataclass(frozen=True)
class NoteTemplate:
    name: str
    description: str
    tags: tuple[str, ...]
    sections: tuple[Section, ...]
    body: str

    @property
    def list_keys(self) -> tuple[str, ...]:
        return tuple(
            section.key
            for section in self.sections
            if section.kind in ("list", "action_list")
        )

    @property
    def summary_key(self) -> str | None:
        for section in self.sections:
            if section.kind == "prose":
                return section.key
        return None

    def section_for_tag(self, tag: str) -> Section | None:
        for section in self.sections:
            if section.tag == tag:
                return section
        return None


def parse_template(name: str, content: str) -> NoteTemplate:
    meta, body = _split_front_matter(content)

    prose_keys = set(BUILTIN_PROSE_KEYS) | set(_string_list(meta.get("prose")))
    sections = _derive_sections(body, prose_keys)
    if not sections:
        raise TemplateError(
            "template body declares no sections — add at least one "
            "{placeholder} besides {title}/{time}/{duration}"
        )

    description = meta.get("description")
    return NoteTemplate(
        name=name,
        description=description if isinstance(description, str) else "",
        tags=tuple(_string_list(meta.get("tags"))),
        sections=sections,
        body=body.strip("\n"),
    )


def _split_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Split a ``---``-fenced YAML frontmatter block from the body.

    Same convention as Jekyll/Hugo/python-frontmatter: the block must open
    the file and is parsed with ``yaml.safe_load``.
    """
    stripped = content.lstrip("﻿\n")
    if not _FM_BOUNDARY.match(stripped):
        return {}, stripped
    pieces = _FM_BOUNDARY.split(stripped, 2)
    if len(pieces) < 3:
        raise TemplateError("frontmatter opened with --- but never closed")
    _, frontmatter, body = pieces
    try:
        meta = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise TemplateError(f"invalid YAML frontmatter: {exc}")
    if meta is None:
        return {}, body
    if not isinstance(meta, dict):
        raise TemplateError("frontmatter must be a YAML mapping of key: value")
    return meta, body


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if item is not None and str(item)]


def _derive_sections(body: str, prose_keys: set[str]) -> tuple[Section, ...]:
    sections: list[Section] = []
    seen: set[str] = set()
    heading: str | None = None
    for line in body.splitlines():
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            heading = _PLACEHOLDER_RE.sub("", heading_match.group(1)).strip()
        for match in _PLACEHOLDER_RE.finditer(line):
            key = match.group(1)
            if key in RESERVED_KEYS or key in seen:
                continue
            if key.upper() in _STRUCTURAL_TAGS:
                raise TemplateError(
                    f"section placeholder {{{key}}} collides with the XML "
                    "contract — rename it (e.g. {" + key + "_section})"
                )
            seen.add(key)
            kind: Literal["prose", "list", "action_list"]
            if key == ACTION_LIST_KEY:
                kind = "action_list"
            elif key in prose_keys:
                kind = "prose"
            else:
                kind = "list"
            sections.append(
                Section(
                    key=key,
                    tag=key.upper(),
                    heading=heading or key.replace("_", " ").title(),
                    kind=kind,
                )
            )
    return tuple(sections)


_PROMPT_PREAMBLE = """You are Chirp, the user's note-taking co-pilot.
Your sole purpose is to transform raw transcripts into structured notes.
Always produce notes using the canonical tags below. Never invent content.

<core_principles>
- Output only what is explicitly stated in the transcript.
- **NEVER** infer, guess, or fabricate tasks, owners, deadlines, or decisions.
- Always prioritize the latest statement if contradictions appear.
- If transcript contains no actionable content, output:
  "Transcript contained no actionable content. No notes available."
- Maintain professional, neutral, concise tone.
</core_principles>"""

_PROMPT_RULES = """<handling_rules>
- Do not include small talk or filler unless relevant to work.
- If technical/code discussed:
  • Render in fenced code blocks
  • Add concise explanatory bullets below
- If transcript mentions topics but no detail:
  • Record: "Limited info available on [topic]"
- Do not use pronouns; use names/roles if stated.
</handling_rules>

<consistency_requirements>
- Always output all sections, even if "None".
- Keep responses deterministic and repeatable.
- Maintain Markdown-safe formatting inside tags where applicable.
- Keep bullets crisp, no prose paragraphs except in technical explanations.
</consistency_requirements>"""

ROOT_TAG = "NOTES"
TITLE_TAG = "TITLE"
_STRUCTURAL_TAGS = frozenset({ROOT_TAG, TITLE_TAG, "ITEM"})


def build_system_prompt(template: NoteTemplate) -> str:
    contract_lines = [
        "<output_contract>",
        "- Emit a SINGLE well-formed UTF-8 XML document with this exact structure:",
        '  - XML declaration: <?xml version="1.0" encoding="UTF-8"?>',
        f"  - Root element: <{ROOT_TAG}> ... </{ROOT_TAG}>",
        "  - Inside the root, ALWAYS include these child tags in this order:",
        f"    1) <{TITLE_TAG}>...</{TITLE_TAG}>",
    ]
    for index, section in enumerate(template.sections, start=2):
        contract_lines.append(f"    {index}) {_contract_shape(section)}")
    contract_lines.extend(
        [
            '- If a section has no content, include the tag with a single text node "None"',
            f"  (e.g., <{template.sections[0].tag}>None</{template.sections[0].tag}>).",
            "- Do NOT output markdown fences or prose. XML ONLY.",
            "- Escape XML special characters (&, <, >) in text nodes.",
            "- For code snippets or multi-line technical blocks, wrap content in <![CDATA[ ... ]]> within the relevant ITEM.",
            "</output_contract>",
        ]
    )

    definition_blocks = [
        f"<{TITLE_TAG}>\n• Short headline for the note (≤6 words)\n</{TITLE_TAG}>"
    ]
    definition_blocks.extend(
        f"<{section.tag}>\n{_section_guidance(section)}\n</{section.tag}>"
        for section in template.sections
    )
    definitions = "\n\n".join(definition_blocks)

    return (
        f"{_PROMPT_PREAMBLE}\n\n"
        + "\n".join(contract_lines)
        + f"\n\n<tag_definitions>\n\n{definitions}\n\n</tag_definitions>\n\n"
        + _PROMPT_RULES
    )


def _contract_shape(section: Section) -> str:
    if section.kind == "prose":
        return f"<{section.tag}>...</{section.tag}>"
    if section.kind == "action_list":
        return (
            f'<{section.tag}> <ITEM task="..." owner="..." deadline="..."/> '
            f"... </{section.tag}>"
        )
    return f"<{section.tag}> <ITEM>...</ITEM> ... </{section.tag}>"


def _section_guidance(section: Section) -> str:
    if section.kind == "prose":
        return (
            f'• Concise prose for the "{section.heading}" section (2-4 sentences)\n'
            "• Only content explicitly stated in the transcript\n"
            '• If the transcript has nothing for this section, output "None"'
        )
    if section.kind == "action_list":
        return (
            "• Format: [Task] — [Owner if stated] — [Deadline if stated]\n"
            '• If owner missing: "Unassigned"\n'
            "• If deadline missing: leave blank\n"
            "• Always use the latest statement if contradictions occur"
        )
    return (
        f'• Short bullet items for the "{section.heading}" section, one <ITEM> each '
        "(≤15 words)\n"
        "• Only content explicitly stated in the transcript\n"
        '• If none stated, output "None"'
    )


def user_templates_dir() -> Path:
    from config.settings import default_chirp_home

    return default_chirp_home() / "templates"


def _builtin_files() -> dict[str, str]:
    root = resources.files("notes") / "templates"
    builtins: dict[str, str] = {}
    for entry in root.iterdir():
        if entry.name.endswith(".md"):
            builtins[entry.name[: -len(".md")]] = entry.read_text(encoding="utf-8")
    return builtins


class TemplateLoader:
    def __init__(self, user_dir: Path | None = None) -> None:
        self.user_dir = user_dir if user_dir is not None else user_templates_dir()
        self._builtins = _builtin_files()

    def available(self) -> list[str]:
        names = set(self._builtins)
        if self.user_dir.is_dir():
            names.update(path.stem for path in self.user_dir.glob("*.md"))
        return sorted(names)

    def scaffold(self) -> list[Path]:
        """Copy missing built-in templates into the user dir; never overwrite."""
        ensure_private_directory(self.user_dir)
        written: list[Path] = []
        for name, content in sorted(self._builtins.items()):
            target = self.user_dir / f"{name}.md"
            if target.exists():
                continue
            target.write_text(content, encoding="utf-8")
            written.append(target)
        return written

    def load(self, name: str) -> NoteTemplate:
        """Load ``name``, preferring the user's file over the built-in.

        A broken user file falls back to the built-in of the same name, then to
        the default template, so note generation never hard-fails on a bad
        template edit. An unknown name raises ``TemplateError`` listing the
        available templates.
        """
        user_path = self.user_dir / f"{name}.md"
        if user_path.is_file():
            try:
                return parse_template(name, user_path.read_text(encoding="utf-8"))
            except (TemplateError, OSError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Template %s is broken (%s); using fallback", user_path, exc
                )
        if name in self._builtins:
            return parse_template(name, self._builtins[name])
        if user_path.is_file():
            return self.load_default()
        raise TemplateError(
            f"unknown template '{name}' — available: {', '.join(self.available())}"
        )

    def load_default(self) -> NoteTemplate:
        return self.load(DEFAULT_TEMPLATE_NAME)

    def match_by_tags(self, note_tags: list[str]) -> NoteTemplate | None:
        """The template sharing the most tags with the note; ties break by name."""
        if not note_tags:
            return None
        tag_set = set(note_tags)
        best: NoteTemplate | None = None
        best_overlap = 0
        for name in self.available():
            try:
                template = self.load(name)
            except TemplateError:
                continue
            overlap = len(tag_set & set(template.tags))
            if overlap > best_overlap:
                best = template
                best_overlap = overlap
        return best

    def resolve(
        self,
        meta_template: str | None,
        note_tags: list[str],
        override: str | None = None,
    ) -> NoteTemplate:
        """Pick a note's template: override > meta.toml > tag match > default."""
        for requested in (override, meta_template):
            if not requested:
                continue
            try:
                return self.load(requested)
            except TemplateError as exc:
                logger.warning("%s; using fallback", exc)
        matched = self.match_by_tags(note_tags)
        if matched is not None:
            return matched
        return self.load_default()
