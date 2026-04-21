from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import ChirpSettings
from transcriber.whisper_transcriber import WhisperTranscriber
from utils.file_utils import TRANSCRIPT_FILENAME, NoteRecord, list_notes
from utils.popup_manager import PopupManager


class BatchProcessor:
    def __init__(self, settings: ChirpSettings, model_override: str | None = None):
        if model_override:
            settings = settings.model_copy(
                update={
                    "models": settings.models.model_copy(
                        update={"whisper": model_override}
                    )
                }
            )
        self.settings = settings
        self.transcriber = WhisperTranscriber(settings)
        self.popup_manager = PopupManager()
        self._lock = threading.Lock()

    def process_records(
        self,
        records: list[NoteRecord],
        force: bool = False,
        progress_callback: Callable | None = None,
        on_segment: Callable | None = None,
        max_workers: int = 1,
    ) -> list[dict[str, Any]]:
        records_to_process = self._filter_records(records, force)

        if not records_to_process:
            return []

        if max_workers == 1:
            results = self._process_sequentially(
                records_to_process, progress_callback, on_segment
            )
        else:
            results = self._process_concurrently(
                records_to_process, progress_callback, max_workers
            )

        success_count = sum(1 for r in results if r["success"])
        if success_count > 0:
            self.popup_manager.show_transcription_complete(success_count)

        return results

    def _process_sequentially(
        self,
        records: list[NoteRecord],
        progress_callback: Callable | None,
        on_segment: Callable | None,
    ) -> list[dict[str, Any]]:
        results = []
        for record in records:
            results.append(self._process_record_safely(record, on_segment))
            if progress_callback:
                progress_callback()
        return results

    def _process_concurrently(
        self,
        records: list[NoteRecord],
        progress_callback: Callable | None,
        max_workers: int,
    ) -> list[dict[str, Any]]:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record = {
                executor.submit(self._process_record_safely, record, None): record
                for record in records
            }
            for future in as_completed(future_to_record):
                results.append(future.result())
                if progress_callback:
                    progress_callback()
        return results

    def _process_record_safely(
        self,
        record: NoteRecord,
        on_segment: Callable | None,
    ) -> dict[str, Any]:
        try:
            return self._process_record(record, on_segment)
        except Exception as exc:
            return {
                "success": False,
                "slug": record.slug,
                "filename": record.audio.name if record.audio else record.slug,
                "error": str(exc),
                "transcribed_at": datetime.now().isoformat(),
            }

    def _process_record(
        self,
        record: NoteRecord,
        on_segment: Callable | None,
    ) -> dict[str, Any]:
        if record.audio is None:
            return {
                "success": False,
                "slug": record.slug,
                "filename": record.slug,
                "error": "No audio file",
                "transcribed_at": datetime.now().isoformat(),
            }

        transcription_result = self.transcriber.transcribe_file(
            record.audio, on_segment=on_segment
        )

        if transcription_result.get("success"):
            transcript_path = record.dir / TRANSCRIPT_FILENAME
            transcript_path.write_text(
                transcription_result.get("full_text", ""),
                encoding="utf-8",
            )
            transcription_result["transcript_path"] = str(transcript_path)

        transcription_result["slug"] = record.slug
        return transcription_result

    def _filter_records(
        self, records: list[NoteRecord], force: bool
    ) -> list[NoteRecord]:
        if force:
            return [record for record in records if record.audio is not None]
        return [
            record
            for record in records
            if record.audio is not None and record.transcript is None
        ]

    def process_directory(
        self,
        directory: Path,
        force: bool = False,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        records = list_notes(directory)
        candidates = [record for record in records if record.audio is not None]

        if not candidates:
            return {
                "success": True,
                "processed_count": 0,
                "total_count": 0,
                "results": [],
                "message": f"No audio files found in {directory}",
            }

        results = self.process_records(candidates, force, progress_callback)
        success_count = sum(1 for r in results if r["success"])

        return {
            "success": True,
            "processed_count": success_count,
            "total_count": len(candidates),
            "results": results,
            "message": f"Processed {success_count}/{len(candidates)} notes successfully",
        }

    def get_processing_stats(self) -> dict[str, Any]:
        notes_root = self.settings.directories.notes_root
        records = list_notes(notes_root)
        transcripts = [record.transcript for record in records if record.transcript]
        total_size = sum(transcript.stat().st_size for transcript in transcripts)

        return {
            "total_transcriptions": len(transcripts),
            "total_size_mb": total_size / (1024 * 1024),
            "notes_root": str(notes_root),
            "model_info": self.transcriber.get_model_info(),
        }
