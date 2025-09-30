import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from rich.console import Console

from config.settings import ChirpSettings
from notes.constants import DEFAULT_MEETING_NAME
from notes.daily_aggregator import DailyAggregator
from notes.template_engine import TemplateEngine
from transcriber.compression import JSONCompressor
from utils.popup_manager import PopupManager
from utils.time_utils import get_daily_note_filename

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
    def __init__(self, settings: ChirpSettings):
        self.settings = settings
        self.template_engine = TemplateEngine(settings)
        self.daily_aggregator = DailyAggregator(settings)
        self.compressor = JSONCompressor()
        self.popup_manager = PopupManager()
        self.console = Console()

    def generate_daily_notes(
        self,
        transcription_files: list[Path],
        force: bool = False,
        filename_override: Optional[str] = None,
    ) -> dict[str, Any]:
        try:
            daily_groups = self.daily_aggregator.group_transcriptions_by_day(
                transcription_files
            )

            results = []

            for date, files in daily_groups.items():
                result = self._generate_notes_for_day(
                    date, files, force, filename_override
                )
                results.append(result)

            successful_results = [r for r in results if r["success"]]

            if successful_results:
                latest_result = max(successful_results, key=lambda x: x["date"])
                self.popup_manager.show_notes_generated(latest_result["filename"])
                return latest_result
            else:
                return {
                    "success": False,
                    "error": "Failed to generate notes for any day",
                    "results": results,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_notes_for_day(
        self,
        date: datetime,
        transcription_files: list[Path],
        force: bool,
        filename_override: Optional[str] = None,
    ) -> dict[str, Any]:
        if filename_override:
            notes_filename = (
                filename_override
                if filename_override.endswith(".md")
                else f"{filename_override}.md"
            )
        else:
            notes_filename = get_daily_note_filename(date)
        notes_path = self.settings.directories.notes / notes_filename

        if notes_path.exists() and not force:
            return {
                "success": True,
                "filename": notes_filename,
                "path": str(notes_path),
                "date": date.isoformat(),
                "message": "Notes already exist (use --force to regenerate)",
            }

        try:
            meeting_sections = []
            total_duration = 0.0
            skipped_files = []

            for transcription_file in transcription_files:
                transcription_data = self.compressor.decompress_json(transcription_file)
                metadata = transcription_data.get("metadata", {})
                recording_label = (
                    metadata.get("recording_id")
                    or metadata.get("meeting_name")
                    or transcription_file.parent.name
                    or transcription_file.stem
                )

                if not transcription_data.get("success", False):
                    skipped_files.append((recording_label, "Failed transcription"))
                    continue

                meeting_notes = self._generate_meeting_notes(transcription_data)
                if meeting_notes:
                    meeting_section = self.template_engine.render_meeting_section(
                        meeting_notes
                    )
                    meeting_sections.append(meeting_section)

                    duration = metadata.get(
                        "recording_length_seconds", metadata.get("duration", 0)
                    )
                    try:
                        total_duration += float(duration)
                    except (TypeError, ValueError):
                        pass
                else:
                    transcript_text = transcription_data.get("full_text", "").strip()
                    reason = (
                        "Insufficient content (< 50 characters)"
                        if len(transcript_text) < 50
                        else "Failed to generate notes"
                    )
                    skipped_files.append((recording_label, reason))

            if skipped_files:
                self.console.print(
                    f"[yellow]⚠️  Skipped {len(skipped_files)} transcription(s):[/yellow]"
                )
                for filename, reason in skipped_files:
                    self.console.print(f"[dim]   • {filename}: {reason}[/dim]")

            if not meeting_sections:
                return {
                    "success": False,
                    "error": "No valid transcriptions found for this day",
                    "date": date.isoformat(),
                }

            daily_notes = self.template_engine.render_daily_notes(
                date, meeting_sections, len(meeting_sections), total_duration
            )

            note_content = self._format_generated_note(daily_notes, date)

            self.settings.directories.notes.mkdir(parents=True, exist_ok=True)

            with open(notes_path, "w", encoding="utf-8") as f:
                f.write(note_content)

            self._auto_index_note(notes_path)

            return {
                "success": True,
                "filename": notes_filename,
                "path": str(notes_path),
                "date": date.isoformat(),
                "meeting_count": len(meeting_sections),
                "total_duration": total_duration,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "date": date.isoformat()}

    def _generate_meeting_notes(
        self, transcription_data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        transcript_text = transcription_data.get("full_text", "").strip()

        if not transcript_text or len(transcript_text) < 50:
            return None

        try:
            metadata = transcription_data.get("metadata", {})
            provided_title = metadata.get("title")

            structured_notes = self._generate_structured_notes(
                transcript_text, provided_title
            )

            if not structured_notes:
                return None

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
                "discussion_highlights": structured_notes.get(
                    "discussion_highlights", []
                ),
                "metadata": transcription_data.get("metadata", {}),
            }

            return meeting_notes

        except Exception:
            return None

    def _format_generated_note(self, body: str, note_date: datetime) -> str:
        metadata = {
            "chirp_source": "generated",
            "readonly": True,
            "generated_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
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

        if stripped_body:
            content = f"{header}\n\n{stripped_body}"
        else:
            content = f"{header}\n"

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

    def _generate_structured_notes(
        self, transcript_text: str, provided_title: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        title_instruction = ""
        if provided_title:
            title_instruction = f"\n\nIMPORTANT: Use this exact meeting title in the MEETING_TITLE tag: {provided_title}"

        prompt = f"""{SYSTEM_PROMPT}

{title_instruction}

Transcript:
{transcript_text}

Return ONLY the XML document, no additional text before or after."""

        try:
            response = self._call_ollama(prompt)
            return self._parse_xml_response(response)

        except Exception:
            return None

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.settings.models.ollama_url}/api/generate"
        num_predict = self.settings.models.num_predict

        payload = {
            "model": self.settings.models.llm,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": num_predict},
        }

        import json

        full_response = []
        chunk_count = 0

        with requests.post(url, json=payload, timeout=300, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    self.console.print(
                        "[yellow]  ↳ Warning: skipped malformed stream chunk[/yellow]"
                    )
                    continue

                token = chunk.get("response", "")
                if token:
                    full_response.append(token)
                    chunk_count += 1

                    if chunk_count % 20 == 0:
                        self.console.print(
                            f"[dim]  ↳ Generating... ({chunk_count} chunks)[/dim]",
                            end="\r",
                        )

                if chunk.get("done", False):
                    break

        if chunk_count > 0:
            self.console.print(f"[dim]  ↳ Generated {chunk_count} chunks[/dim]       ")

        response_text = "".join(full_response).strip()
        return response_text

    def _parse_xml_response(self, response: str) -> Optional[dict[str, Any]]:
        try:
            xml_start = response.find("<?xml")
            if xml_start == -1:
                xml_start = response.find("<MEETING_NOTES>")

            if xml_start == -1:
                return self._parse_fallback_response(response)

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
            return self._parse_fallback_response(response)
        except Exception:
            return None

    def _parse_xml_response(self, response: str) -> Optional[dict[str, Any]]:
        try:
            xml_start = response.find("<?xml")
            if xml_start == -1:
                xml_start = response.find("<MEETING_NOTES>")

            if xml_start == -1:
                return self._parse_fallback_response(response)

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
                "meeting_title": get_text("MEETING_TITLE") or "Meeting Notes",
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
            return self._parse_fallback_response(response)
        except Exception:
            return None

    def _parse_fallback_response(self, response: str) -> dict[str, Any]:
        return {
            "meeting_title": DEFAULT_MEETING_NAME,
            "executive_summary": response[:200] + "..."
            if len(response) > 200
            else response,
            "agenda": [],
            "action_items": [],
            "next_steps": [],
            "decisions": [],
            "open_questions": [],
            "discussion_highlights": [
                "Unable to parse structured notes from AI response"
            ],
        }

    def _auto_index_note(self, notes_path: Path):
        if not self.settings.notes_chat.auto_index:
            return

        try:
            from notes_chat.index import IndexManager

            index_manager = IndexManager(self.settings)
            success = index_manager._add_to_index(notes_path)

            if success:
                manifest = index_manager._load_manifest()
                current_files = index_manager._scan_notes_files()

                file_path = str(notes_path)
                if file_path in current_files:
                    manifest[file_path] = current_files[file_path]
                    index_manager._save_manifest(manifest)

                index_manager._rebuild_bm25()

                self.console.print(
                    f"[dim green]✓ Auto-indexed {notes_path.name}[/dim green]"
                )

        except Exception as e:
            self.console.print(
                f"[dim yellow]⚠️ Auto-indexing failed for {notes_path.name}: {e}[/dim yellow]"
            )
