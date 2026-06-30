import logging
import tomllib
import wave
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomli_w
from rich.console import Console

from config.settings import ChirpSettings
from llm.client import LLMClient
from llm.registry import resolved_chat_model
from notes.constants import DEFAULT_MEETING_NAME
from notes.template_engine import TemplateEngine
from utils.file_utils import META_FILENAME, NOTES_FILENAME, NoteRecord, list_notes
from utils.popup_manager import PopupManager

logger = logging.getLogger(__name__)

# Bounds the prompt INPUT, not just num_predict (the OUTPUT): an unbounded
# transcript was silently truncated mid-generation, so window it and warn.
MAX_TRANSCRIPT_CHARS = 24000

# Below this, a transcript has nothing to summarize (a silent/near-empty clip);
# generating notes would only yield junk that later poisons `ask`. Callers treat
# this as a benign skip, not a failure.
MIN_TRANSCRIPT_CHARS = 50

MAX_PARSE_ATTEMPTS = 2

SYSTEM_PROMPT = """You are Chirp, the user's meeting note co-pilot.
Your sole purpose is to transform raw meeting transcripts into structured meeting notes.
Always produce notes in a consistent format. Never invent content.
Always produce notes using the canonical tags below. Never invent content.

<core_principles>
- Output only what is explicitly stated in the transcript.
- **NEVER** infer, guess, or fabricate tasks, owners, deadlines, or decisions.
- Always prioritize the latest statement if contradictions appear.
  • Example of prioritize latest statement:

    ```text
    Transcript:
      • "We'll launch feature X on June 1."
      • "Actually, make that June 15."

    Notes:
      • Decisions: Launch feature X on June 15.
    ```

- If transcript contains no actionable content, output:
  "Transcript contained no actionable content. No notes available."
- Maintain professional, neutral, concise tone.
</core_principles>

<output_contract>
- Emit a SINGLE well-formed UTF-8 XML document with this exact structure:
  - XML declaration: <?xml version="1.0" encoding="UTF-8"?>
  - Root element: <MEETING_NOTES> ... </MEETING_NOTES>
  - Inside the root, ALWAYS include these child tags in this order:
    1) <MEETING_TITLE>...</MEETING_TITLE>
    2) <EXECUTIVE_SUMMARY>...</EXECUTIVE_SUMMARY>
    3) <AGENDA> <ITEM>...</ITEM> ... </AGENDA>
    4) <ACTION_ITEMS> <ITEM task="..." owner="..." deadline="..."/> ... </ACTION_ITEMS>
    5) <NEXT_STEPS> <ITEM>...</ITEM> ... </NEXT_STEPS>
    6) <DECISIONS> <ITEM>...</ITEM> ... </DECISIONS>
    7) <OPEN_QUESTIONS> <ITEM>...</ITEM> ... </OPEN_QUESTIONS>
    8) <DISCUSSION_HIGHLIGHTS> <ITEM>...</ITEM> ... </DISCUSSION_HIGHLIGHTS>

- If a section has no content, include the tag with a single text node "None"
  (e.g., <AGENDA>None</AGENDA>), except:
  - For ACTION_ITEMS when empty, emit <ACTION_ITEMS>None</ACTION_ITEMS> (no ITEM children).
- Do NOT output markdown fences or prose. XML ONLY.
- Escape XML special characters (&, <, >) in text nodes.
- For code snippets or multi-line technical blocks, wrap content in <![CDATA[ ... ]]> within the relevant ITEM.
</output_contract>

<tag_definitions>

<MEETING_TITLE>
• Short headline (≤6 words, e.g., "Project Alpha Sync")
</MEETING_TITLE>

<EXECUTIVE_SUMMARY>
• 2-4 sentences maximum
• High-level overview of meeting purpose, key outcomes, and tone
• No action details — keep those in ACTION_ITEMS / NEXT_STEPS
• If transcript lacks substance, output "None"
</EXECUTIVE_SUMMARY>

<AGENDA>
• Agenda items mentioned by participants
• If none stated, output "None"
</AGENDA>

<ACTION_ITEMS>
• Format: [Task] — [Owner if stated] — [Deadline if stated]
• If owner missing: "Unassigned"
• If deadline missing: leave blank
• Always use the latest statement if contradictions occur
</ACTION_ITEMS>

<NEXT_STEPS>
• Broader follow-ups not tied to an individual
• Team-level actions or reminders
• Future agenda items
• If none present, output "None"
</NEXT_STEPS>

<DECISIONS>
• Record only explicit, final conclusions
• If later statements revise earlier ones, keep the most recent
• If unresolved conflict remains: "Unresolved: conflicting statements on [topic]"
</DECISIONS>

<OPEN_QUESTIONS>
• Capture unresolved questions or risks explicitly raised
• If none present, output "None"
</OPEN_QUESTIONS>

<DISCUSSION_HIGHLIGHTS>
• Short bullets (≤15 words each)
• Optional sub-bullets for context (≤20 words)
• Do not include small talk
• Prioritize final statements when contradictions occur
</DISCUSSION_HIGHLIGHTS>

</tag_definitions>

<handling_rules>
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
- Use Markdown formatting for clarity.
- Keep responses deterministic and repeatable.
- Maintain Markdown-safe formatting inside tags where applicable.
- Keep bullets crisp, no prose paragraphs except in technical explanations.
</consistency_requirements>
"""


class NoteGenerator:
    def __init__(
        self,
        settings: ChirpSettings,
        console: Console | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.template_engine = TemplateEngine(settings)
        self.popup_manager = PopupManager()
        self.console = console if console is not None else Console()
        self._llm_client = llm_client

    def generate_for_records(
        self,
        records: list[NoteRecord],
        force: bool = False,
    ) -> dict[str, Any]:
        results = []
        for record in records:
            result = self._generate_for_record(record, force)
            results.append(result)

        newly_generated = [r for r in results if r["success"] and "message" not in r]
        successful = [r for r in results if r["success"]]
        if newly_generated:
            latest = newly_generated[-1]
            self.popup_manager.show_notes_generated(latest["filename"])
            return {**latest, "results": results}
        if successful:
            return {**successful[-1], "results": results}

        # Distinguish a benign skip (nothing to summarize) from a real failure so
        # callers can surface it as a skip rather than a hard error. Only when
        # *every* record was skipped — a single genuine failure makes the batch a
        # failure.
        skipped = [r for r in results if r.get("skipped")]
        if results and len(skipped) == len(results):
            return {
                "success": False,
                "skipped": True,
                "error": skipped[-1]["error"],
                "results": results,
            }

        last_error = next(
            (r["error"] for r in reversed(results) if r.get("error")),
            "Failed to generate notes for any record",
        )
        return {
            "success": False,
            "error": last_error,
            "results": results,
        }

    def generate_from_notes_root(
        self,
        notes_root: Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        resolved_root = notes_root or self.settings.directories.notes_root
        records = list_notes(resolved_root)
        candidates = [
            record
            for record in records
            if record.transcript is not None and (force or record.notes is None)
        ]
        if not candidates:
            return {
                "success": False,
                "error": "No transcripts ready for notes generation",
                "results": [],
            }
        return self.generate_for_records(candidates, force=force)

    def _generate_for_record(self, record: NoteRecord, force: bool) -> dict[str, Any]:
        notes_path = record.dir / NOTES_FILENAME

        if notes_path.exists() and not force:
            return {
                "success": True,
                "slug": record.slug,
                "filename": notes_path.name,
                "path": str(notes_path),
                "message": "Notes already exist (use --force to regenerate)",
            }

        if record.transcript is None:
            return {
                "success": False,
                "slug": record.slug,
                "error": "No transcript available for this record",
            }

        transcript_text = record.transcript.read_text(encoding="utf-8").strip()
        if len(transcript_text) < MIN_TRANSCRIPT_CHARS:
            return {
                "success": False,
                "skipped": True,
                "slug": record.slug,
                "error": (
                    f"Insufficient transcript content "
                    f"(< {MIN_TRANSCRIPT_CHARS} characters)"
                ),
            }

        provided_title = record.title
        try:
            structured_notes = self._generate_structured_notes(
                transcript_text, provided_title
            )
        except Exception as exc:  # noqa: BLE001 - surface the real daemon/LLM failure
            logger.warning("Note generation failed for %s: %s", record.slug, exc)
            return {
                "success": False,
                "slug": record.slug,
                "error": f"note generation failed — {exc}",
            }
        if not structured_notes:
            # Never write or auto-index unparsable output — a junk note would
            # later poison `ask` (AC-12).
            self.console.print(
                f"[red]Could not generate structured notes for {record.slug}; "
                "skipping (not written, not indexed).[/red]"
            )
            return {
                "success": False,
                "degraded": True,
                "slug": record.slug,
                "error": "Could not parse structured notes from the model output",
            }

        meeting_title = (
            provided_title
            if provided_title
            else structured_notes.get("meeting_title", DEFAULT_MEETING_NAME)
        )
        meeting_notes = {
            "meeting_title": meeting_title,
            "executive_summary": structured_notes.get(
                "executive_summary", "No summary available"
            ),
            "agenda": structured_notes.get("agenda", []),
            "action_items": structured_notes.get("action_items", []),
            "next_steps": structured_notes.get("next_steps", []),
            "decisions": structured_notes.get("decisions", []),
            "open_questions": structured_notes.get("open_questions", []),
            "discussion_highlights": structured_notes.get("discussion_highlights", []),
            "metadata": {
                "date": record.created_at.isoformat(),
                "duration_s": self._resolve_duration_seconds(record),
            },
        }

        body = self.template_engine.render_meeting_section(meeting_notes)
        content = self._format_generated_note(body, record.created_at)

        notes_path.write_text(content, encoding="utf-8")
        self._update_meta(record.dir)
        self._auto_index_note(notes_path)

        return {
            "success": True,
            "slug": record.slug,
            "filename": notes_path.name,
            "path": str(notes_path),
        }

    def _resolve_duration_seconds(self, record: NoteRecord) -> float:
        """Best-effort clip length for the note's Duration field.

        Prefers ``duration_s`` from meta.toml (written by the recorder and the
        transcribe save stage), falls back to the legacy ``duration`` key, then
        to the WAV header — so a real duration shows even for imported audio
        whose meta predates the field. Returns 0.0 when unknown (renders as
        "Unknown"). Previously the generator passed no duration at all, so every
        generated note read "Duration: Unknown" despite meta carrying it.
        """
        meta_path = record.dir / META_FILENAME
        if meta_path.exists():
            try:
                with meta_path.open("rb") as fh:
                    meta = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                meta = {}
            for key in ("duration_s", "duration"):
                value = meta.get(key)
                if value is None:
                    continue
                try:
                    seconds = float(value)
                except (TypeError, ValueError):
                    continue
                if seconds > 0:
                    return seconds
        audio = record.audio
        if audio is not None and audio.exists():
            try:
                with wave.open(str(audio), "rb") as handle:
                    frames = handle.getnframes()
                    rate = handle.getframerate()
                if rate > 0:
                    return frames / float(rate)
            except (OSError, wave.Error) as exc:
                logger.debug("Could not read duration from %s: %s", audio, exc)
        return 0.0

    def _update_meta(self, note_dir: Path) -> None:
        meta_path = note_dir / META_FILENAME
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with meta_path.open("rb") as fh:
                    meta = dict(tomllib.load(fh))
            except (OSError, tomllib.TOMLDecodeError):
                meta = {}

        meta["whisper_model"] = self.settings.models.whisper
        meta["llm_model"] = resolved_chat_model(self.settings.models.llm)
        meta["indexed_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("wb") as fh:
            tomli_w.dump(meta, fh)

    def _format_generated_note(self, body: str, note_date: datetime) -> str:
        metadata = {
            "chirp_source": "generated",
            "readonly": True,
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "note_date": note_date.date().isoformat(),
        }
        cleaned_body = self._strip_front_matter(body)
        return self._apply_front_matter(cleaned_body, metadata)

    def _apply_front_matter(self, body: str, metadata: dict[str, Any]) -> str:
        header_lines = ["---"]
        for key, value in metadata.items():
            header_lines.append(f"{key}: {self._format_front_matter_value(value)}")
        header_lines.append("---")

        header = "\n".join(header_lines)
        stripped_body = body.lstrip("\n")

        content = f"{header}\n\n{stripped_body}" if stripped_body else f"{header}\n"

        if not content.endswith("\n"):
            content += "\n"

        return content

    def _strip_front_matter(self, content: str) -> str:
        stripped = content.lstrip()
        if not stripped.startswith("---"):
            return content.lstrip("\n")

        remainder = stripped[3:]
        closing_index = remainder.find("\n---")
        if closing_index == -1:
            return content.lstrip("\n")

        after = remainder[closing_index + len("\n---") :]
        return after.lstrip("\n")

    def _format_front_matter_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _window_transcript(self, transcript_text: str) -> str:
        """Cap the transcript to ``MAX_TRANSCRIPT_CHARS``, warning on truncation."""
        if len(transcript_text) <= MAX_TRANSCRIPT_CHARS:
            return transcript_text
        self.console.print(
            f"[yellow]Transcript is {len(transcript_text)} chars; truncating to "
            f"{MAX_TRANSCRIPT_CHARS} for note generation.[/yellow]"
        )
        return transcript_text[:MAX_TRANSCRIPT_CHARS]

    def _generate_structured_notes(
        self, transcript_text: str, provided_title: str | None = None
    ) -> dict[str, Any] | None:
        title_instruction = ""
        if provided_title:
            title_instruction = (
                f"\n\nIMPORTANT: Use this exact meeting title in the "
                f"MEETING_TITLE tag: {provided_title}"
            )

        windowed_transcript = self._window_transcript(transcript_text)
        prompt = f"""{SYSTEM_PROMPT}

{title_instruction}

Transcript:
{windowed_transcript}

Return ONLY the XML document, no additional text before or after."""

        # Returns None on persistent parse failure so the caller skips writing
        # and indexing a junk note (AC-12).
        for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
            response = self._call_llm(prompt)

            parsed = self._parse_xml_response(response)
            if parsed is not None:
                return parsed

            if attempt < MAX_PARSE_ATTEMPTS:
                self.console.print(
                    "[yellow]Could not parse structured notes; retrying once…[/yellow]"
                )

        logger.debug(
            "Structured-note parsing failed after %d attempts", MAX_PARSE_ATTEMPTS
        )
        return None

    def _call_llm(self, prompt: str) -> str:
        # Single user message so the chat template wraps SYSTEM_PROMPT + prompt
        # verbatim — preserving the prompt shape the regression baseline was
        # captured against.
        messages = [{"role": "user", "content": prompt}]
        options = {"max_tokens": self.settings.models.num_predict}
        # Reuse one client across records in a batch — LLMClient() resolves the
        # socket path on construction, so per-record instantiation is wasteful.
        if self._llm_client is None:
            self._llm_client = LLMClient()
        client = self._llm_client

        deltas: list[str] = []
        chunk_count = 0

        for token in client.chat_stream_sync(
            messages, model="default", options=options
        ):
            if token:
                deltas.append(token)
                chunk_count += 1
                if chunk_count % 20 == 0:
                    self.console.print(
                        f"[dim]  ↳ Generating... ({chunk_count} chunks)[/dim]",
                        end="\r",
                    )

        if chunk_count > 0:
            self.console.print(f"[dim]  ↳ Generated {chunk_count} chunks[/dim]       ")

        return "".join(deltas).strip()

    def _parse_xml_response(self, response: str) -> dict[str, Any] | None:
        try:
            xml_start = response.find("<?xml")
            if xml_start == -1:
                xml_start = response.find("<MEETING_NOTES>")

            if xml_start == -1:
                # No XML: signal parse failure so the caller retries/degrades
                # rather than persisting a junk note (AC-12).
                return None

            xml_content = response[xml_start:]
            xml_end = xml_content.find("</MEETING_NOTES>")
            if xml_end != -1:
                xml_content = xml_content[: xml_end + len("</MEETING_NOTES>")]

            root = ET.fromstring(xml_content)

            def get_text(element_name: str) -> str:
                elem = root.find(element_name)
                if (
                    elem is not None
                    and elem.text
                    and elem.text.strip()
                    and elem.text.strip().lower() != "none"
                ):
                    return elem.text.strip()
                return ""

            def get_items(element_name: str) -> list[str]:
                parent = root.find(element_name)
                if parent is None:
                    return []
                if parent.text and parent.text.strip().lower() == "none":
                    return []

                items = []
                for item in parent.findall("ITEM"):
                    if element_name == "ACTION_ITEMS":
                        task = item.get("task", "").strip()
                        owner = item.get("owner", "").strip()
                        deadline = item.get("deadline", "").strip()
                        parts = []
                        if task:
                            parts.append(task)
                        if owner:
                            parts.append(f"Owner: {owner}")
                        if deadline:
                            parts.append(f"Deadline: {deadline}")
                        if parts:
                            items.append(" — ".join(parts))
                    else:
                        text = item.text.strip() if item.text else ""
                        if text:
                            items.append(text)

                return items

            return {
                "meeting_title": get_text("MEETING_TITLE") or DEFAULT_MEETING_NAME,
                "executive_summary": get_text("EXECUTIVE_SUMMARY")
                or "No summary available",
                "agenda": get_items("AGENDA"),
                "action_items": get_items("ACTION_ITEMS"),
                "next_steps": get_items("NEXT_STEPS"),
                "decisions": get_items("DECISIONS"),
                "open_questions": get_items("OPEN_QUESTIONS"),
                "discussion_highlights": get_items("DISCUSSION_HIGHLIGHTS"),
            }

        except ET.ParseError:
            # Malformed XML: let the caller retry/degrade, never write it (AC-12).
            return None
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def _auto_index_note(self, notes_path: Path):
        if not self.settings.notes_chat.auto_index:
            return

        from chirp.exceptions import EmbedModelChanged

        try:
            from notes_chat.index import IndexManager

            index_manager = IndexManager(self.settings)
            indexed = index_manager.add_note(
                notes_path,
                guard_embed_fingerprint=True,
                incremental_bm25=True,
            )

            if indexed:
                self.console.print(
                    f"[dim green]✓ Auto-indexed {notes_path.name}[/dim green]"
                )
        except EmbedModelChanged as exc:
            logger.debug("Auto-indexing skipped for %s: %s", notes_path.name, exc)
            self.console.print(
                f"[yellow]Auto-indexing skipped for {notes_path.name}: {exc}[/yellow]"
            )
        except Exception as exc:  # noqa: BLE001 - defensive auto-index; IndexManager can raise many types
            logger.debug("Auto-indexing failed for %s: %s", notes_path.name, exc)
            self.console.print(
                f"[dim yellow]Auto-indexing failed for {notes_path.name}: {exc}[/dim yellow]"
            )
