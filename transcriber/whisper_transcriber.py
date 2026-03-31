import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from faster_whisper import WhisperModel

from config.settings import ChirpSettings
from notes.constants import DEFAULT_MEETING_NAME
from utils.file_utils import get_file_size_mb
from utils.time_utils import derive_recording_id, parse_timestamp_from_filename


class WhisperTranscriber:
    def __init__(self, settings: ChirpSettings):
        self.settings = settings
        self.model = None
        self._load_model()

    def _load_model(self):
        device = self._get_optimal_device()
        compute_type = self._get_compute_type()

        self.model = WhisperModel(
            self.settings.models.whisper,
            device=device,
            compute_type=compute_type,
            cpu_threads=self._get_cpu_threads(),
        )

    def _get_optimal_device(self) -> str:
        system = platform.system()

        if system == "Darwin":
            return "cpu"
        else:
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"

    def _get_compute_type(self) -> str:
        device = self._get_optimal_device()

        if device == "cpu":
            return "int8"
        else:
            return "float16"

    def _get_cpu_threads(self) -> int:
        import os

        cpu_count = os.cpu_count()
        if cpu_count and cpu_count >= 8:
            return max(4, cpu_count - 2)
        return max(1, cpu_count // 2) if cpu_count else 1

    def transcribe_file(
        self,
        audio_file_path: Path,
        fast_mode: bool = False,
        language: str | None = None,
        on_segment: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        if not self.model:
            raise RuntimeError("Whisper model not loaded")

        start_time = datetime.now()
        audio_metadata = self._read_audio_metadata(audio_file_path)
        recording_datetime = self._get_recording_datetime(
            audio_file_path, audio_metadata
        )
        recording_id = derive_recording_id(audio_file_path)
        meeting_name = self._get_meeting_name(audio_metadata)
        device = self._get_optimal_device()
        compute_type = self._get_compute_type()

        if fast_mode:
            beam_size = 1
            best_of = 1
            word_timestamps = False
            hallucination_silence_threshold = None
        else:
            beam_size = 5
            best_of = 5
            word_timestamps = True
            hallucination_silence_threshold = 2.0

        try:
            vad_enabled = self.settings.audio.vad_enabled
            vad_params = (
                self.settings.audio.vad_parameters.model_dump() if vad_enabled else None
            )

            segments, info = self.model.transcribe(
                str(audio_file_path),
                beam_size=beam_size,
                best_of=best_of,
                language=language,
                condition_on_previous_text=False,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                initial_prompt=None,
                vad_filter=vad_enabled,
                vad_parameters=vad_params,
                word_timestamps=word_timestamps,
                hallucination_silence_threshold=hallucination_silence_threshold,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            )

            transcript_segments: list[dict[str, Any]] = []
            transcript_text_parts: list[str] = []

            for segment in segments:
                segment_text = segment.text.strip()
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment_text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                }
                if word_timestamps and hasattr(segment, "words") and segment.words:
                    segment_data["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ]
                transcript_segments.append(segment_data)
                if segment_text:
                    transcript_text_parts.append(segment_text)

                if on_segment and segment_text:
                    try:
                        on_segment(segment_data)
                    except Exception:
                        pass

            full_text = " ".join(transcript_text_parts).strip()
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            transcribed_at = end_time.isoformat()
            segment_count = len(transcript_segments)
            word_count = len(full_text.split()) if full_text else 0
            character_count = len(full_text)
            recording_completed_at = (
                recording_datetime + timedelta(seconds=info.duration)
                if info.duration
                else None
            )

            metadata = {
                "schema_version": 2,
                "recording_id": recording_id,
                "recording_filename": audio_file_path.name,
                "recording_path": str(audio_file_path.resolve()),
                "meeting_name": meeting_name,
                "recording_datetime": recording_datetime.isoformat()
                if recording_datetime
                else None,
                "recording_completed_at": recording_completed_at.isoformat()
                if recording_completed_at
                else None,
                "recorded_at": recording_datetime.isoformat()
                if recording_datetime
                else None,
                "recording_length_seconds": info.duration,
                "duration": info.duration,
                "language": info.language,
                "language_probability": info.language_probability,
                "transcribed_at": transcribed_at,
                "transcription_time": processing_time,
                "transcription_time_seconds": processing_time,
                "model": self.settings.models.whisper,
                "device": device,
                "compute_type": compute_type,
                "file_size_mb": get_file_size_mb(audio_file_path),
                "segment_count": segment_count,
                "word_count": word_count,
                "character_count": character_count,
                "title": audio_metadata.get("title") if audio_metadata else None,
                "audio_metadata": audio_metadata or {},
            }

            return {
                "success": True,
                "filename": audio_file_path.name,
                "full_text": full_text,
                "segments": transcript_segments,
                "metadata": metadata,
                "error": None,
            }

        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            error_metadata = {
                "schema_version": 2,
                "recording_id": recording_id,
                "recording_filename": audio_file_path.name,
                "recording_path": str(audio_file_path.resolve()),
                "meeting_name": meeting_name,
                "recording_datetime": recording_datetime.isoformat()
                if recording_datetime
                else None,
                "transcription_time": processing_time,
                "transcription_time_seconds": processing_time,
                "transcribed_at": end_time.isoformat(),
                "title": audio_metadata.get("title") if audio_metadata else None,
            }

            return {
                "success": False,
                "filename": audio_file_path.name,
                "full_text": "",
                "segments": [],
                "metadata": error_metadata,
                "error": str(e),
            }

    def transcribe_batch(self, audio_files: list[Path]) -> list[dict[str, Any]]:
        results = []

        for audio_file in audio_files:
            try:
                result = self.transcribe_file(audio_file)
                results.append(result)
            except Exception as e:
                audio_metadata = self._read_audio_metadata(audio_file)
                recording_datetime = self._get_recording_datetime(
                    audio_file, audio_metadata
                )
                results.append(
                    {
                        "success": False,
                        "filename": audio_file.name,
                        "full_text": "",
                        "segments": [],
                        "metadata": {
                            "schema_version": 2,
                            "recording_id": derive_recording_id(audio_file),
                            "recording_filename": audio_file.name,
                            "recording_path": str(audio_file.resolve()),
                            "meeting_name": self._get_meeting_name(audio_metadata),
                            "recording_datetime": recording_datetime.isoformat()
                            if recording_datetime
                            else None,
                            "title": audio_metadata.get("title")
                            if audio_metadata
                            else None,
                            "transcribed_at": datetime.now().isoformat(),
                        },
                        "error": str(e),
                    }
                )

        return results

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.settings.models.whisper,
            "device": self._get_optimal_device(),
            "compute_type": self._get_compute_type(),
            "cpu_threads": self._get_cpu_threads(),
            "loaded": self.model is not None,
        }

    def _read_audio_metadata(self, audio_file_path: Path) -> Optional[dict]:
        import json

        metadata_file = audio_file_path.with_suffix(f"{audio_file_path.suffix}.meta")

        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    data = json.load(f)
                    return dict(data) if isinstance(data, dict) else None
            except Exception:
                pass

        return None

    def _get_recording_datetime(
        self, audio_file_path: Path, audio_metadata: Optional[dict]
    ) -> datetime:
        if audio_metadata:
            recorded_at = audio_metadata.get("recorded_at")
            if isinstance(recorded_at, str) and recorded_at.strip():
                cleaned = recorded_at.strip().replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(cleaned)
                except ValueError:
                    pass

        parsed = parse_timestamp_from_filename(audio_file_path.name)
        if parsed:
            return parsed

        try:
            return datetime.fromtimestamp(audio_file_path.stat().st_mtime)
        except (OSError, ValueError):
            return datetime.now()

    def _get_meeting_name(self, audio_metadata: Optional[dict]) -> str:
        if audio_metadata:
            title = audio_metadata.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
        return DEFAULT_MEETING_NAME

    def __del__(self):
        if hasattr(self, "model") and self.model:
            del self.model
