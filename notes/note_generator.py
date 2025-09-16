import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from rich.console import Console

from config.settings import ChirpSettings
from notes.daily_aggregator import DailyAggregator
from notes.template_engine import TemplateEngine
from transcriber.compression import JSONCompressor
from utils.popup_manager import PopupManager
from utils.time_utils import get_daily_note_filename


class NoteGenerator:
    def __init__(self, settings: ChirpSettings):
        self.settings = settings
        self.template_engine = TemplateEngine(settings)
        self.daily_aggregator = DailyAggregator(settings)
        self.compressor = JSONCompressor()
        self.popup_manager = PopupManager()
        self.console = Console()

    def generate_daily_notes(
        self, transcription_files: list[Path], force: bool = False
    ) -> dict[str, Any]:
        try:
            daily_groups = self.daily_aggregator.group_transcriptions_by_day(
                transcription_files
            )

            results = []

            for date, files in daily_groups.items():
                result = self._generate_notes_for_day(date, files, force)
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
        self, date: datetime, transcription_files: list[Path], force: bool
    ) -> dict[str, Any]:
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

                if not transcription_data.get("success", False):
                    skipped_files.append(
                        (transcription_file.name, "Failed transcription")
                    )
                    continue

                meeting_notes = self._generate_meeting_notes(transcription_data)
                if meeting_notes:
                    meeting_section = self.template_engine.render_meeting_section(
                        meeting_notes
                    )
                    meeting_sections.append(meeting_section)

                    duration = transcription_data.get("metadata", {}).get("duration", 0)
                    total_duration += duration
                else:
                    transcript_text = transcription_data.get("full_text", "").strip()
                    reason = (
                        "Insufficient content (< 50 characters)"
                        if len(transcript_text) < 50
                        else "Failed to generate notes"
                    )
                    skipped_files.append((transcription_file.name, reason))

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

            self.settings.directories.notes.mkdir(parents=True, exist_ok=True)

            with open(notes_path, "w", encoding="utf-8") as f:
                f.write(daily_notes)

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
            meeting_title = self._generate_meeting_title(transcript_text)
            structured_notes = self._generate_structured_notes(transcript_text)

            if not structured_notes:
                return None

            meeting_notes = {
                "meeting_title": meeting_title,
                "participants": structured_notes.get("participants", "Not specified"),
                "executive_summary": structured_notes.get(
                    "executive_summary", "No summary available"
                ),
                "key_points": structured_notes.get("key_points", []),
                "decisions": structured_notes.get("decisions", []),
                "action_items": structured_notes.get("action_items", []),
                "next_steps": structured_notes.get("next_steps", []),
                "metadata": transcription_data.get("metadata", {}),
            }

            return meeting_notes

        except Exception:
            return None

    def _generate_meeting_title(self, transcript_text: str) -> str:
        prompt = f"""Please analyze this meeting transcript and generate a concise, descriptive title that captures the main topic or purpose of the meeting.

Transcript excerpt (first 500 characters):
{transcript_text[:500]}

Generate only a title, no additional text. Keep it under 60 characters."""

        try:
            response = self._call_ollama(prompt)
            title = response.strip().strip('"').strip("'")

            if len(title) > 60:
                title = title[:57] + "..."

            return title if title else "Meeting Notes"

        except Exception:
            return "Meeting Notes"

    def _generate_structured_notes(
        self, transcript_text: str
    ) -> Optional[dict[str, Any]]:
        prompt = f"""Please analyze this meeting transcript and extract structured information. Return a JSON object with the following fields:

- participants: String with participant names/roles if mentioned, or "Not specified"
- executive_summary: 2-3 sentence summary of the meeting
- key_points: Array of main discussion points (3-5 items)
- decisions: Array of decisions made (if any)
- action_items: Array of action items or tasks assigned (if any)
- next_steps: Array of next steps or follow-up items (if any)

Transcript:
{transcript_text}

Return only valid JSON, no additional text or formatting."""

        try:
            response = self._call_ollama(prompt)

            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                return dict(data) if isinstance(data, dict) else None
            else:
                return self._parse_fallback_response(response)

        except Exception:
            return None

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.settings.models.ollama_url}/api/generate"

        payload = {
            "model": self.settings.models.llm,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 500},
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "")
        return str(response_text).strip() if response_text else ""

    def _parse_fallback_response(self, response: str) -> dict[str, Any]:
        return {
            "participants": "Not specified",
            "executive_summary": response[:200] + "..."
            if len(response) > 200
            else response,
            "key_points": ["Unable to parse structured notes from AI response"],
            "decisions": [],
            "action_items": [],
            "next_steps": [],
        }

    def test_ollama_connection(self) -> dict[str, Any]:
        try:
            url = f"{self.settings.models.ollama_url}/api/version"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                version_info = response.json()

                models_url = f"{self.settings.models.ollama_url}/api/tags"
                models_response = requests.get(models_url, timeout=5)

                available_models = []
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    available_models = [
                        model["name"] for model in models_data.get("models", [])
                    ]

                return {
                    "connected": True,
                    "version": version_info.get("version", "Unknown"),
                    "url": self.settings.models.ollama_url,
                    "configured_model": self.settings.models.llm,
                    "available_models": available_models,
                    "model_available": self.settings.models.llm in available_models,
                }
            else:
                return {
                    "connected": False,
                    "error": f"HTTP {response.status_code}",
                    "url": self.settings.models.ollama_url,
                }

        except requests.exceptions.ConnectionError:
            return {
                "connected": False,
                "error": "Connection refused - is Ollama running?",
                "url": self.settings.models.ollama_url,
            }

        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "url": self.settings.models.ollama_url,
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
