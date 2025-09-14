import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from config.settings import ChirpSettings
from utils.file_utils import get_file_size_mb
from utils.time_utils import parse_timestamp_from_filename


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
        processor = platform.processor()

        if system == "Darwin" and (
            "M1" in processor or "M2" in processor or "M3" in processor
        ):
            return "cpu"  # faster-whisper doesn't support Metal yet, but CPU is optimized for Apple Silicon
        elif system == "Darwin":
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
            if platform.system() == "Darwin" and (
                "M1" in platform.processor()
                or "M2" in platform.processor()
                or "M3" in platform.processor()
            ):
                return "int8"  # Optimized for Apple Silicon
            else:
                return "int8"
        else:
            return "float16"

    def _get_cpu_threads(self) -> int:
        import os

        cpu_count = os.cpu_count()
        return max(1, cpu_count // 2) if cpu_count else 1

    def transcribe_file(self, audio_file_path: Path) -> dict[str, Any]:
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        if not self.model:
            raise RuntimeError("Whisper model not loaded")

        start_time = datetime.now()

        try:
            segments, info = self.model.transcribe(
                str(audio_file_path),
                beam_size=5,
                language=None,  # Auto-detect
                condition_on_previous_text=False,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                initial_prompt=None,
            )

            transcript_segments = []
            full_text = ""

            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                }
                transcript_segments.append(segment_data)
                full_text += segment.text.strip() + " "

            processing_time = (datetime.now() - start_time).total_seconds()

            recording_timestamp = parse_timestamp_from_filename(audio_file_path.name)

            result = {
                "success": True,
                "filename": audio_file_path.name,
                "full_text": full_text.strip(),
                "segments": transcript_segments,
                "metadata": {
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "transcription_time": processing_time,
                    "model": self.settings.models.whisper,
                    "device": self._get_optimal_device(),
                    "compute_type": self._get_compute_type(),
                    "file_size_mb": get_file_size_mb(audio_file_path),
                    "recorded_at": recording_timestamp.isoformat()
                    if recording_timestamp
                    else None,
                    "transcribed_at": datetime.now().isoformat(),
                },
                "error": None,
            }

            return result

        except Exception as e:
            return {
                "success": False,
                "filename": audio_file_path.name,
                "full_text": "",
                "segments": [],
                "metadata": {
                    "transcription_time": (datetime.now() - start_time).total_seconds(),
                    "transcribed_at": datetime.now().isoformat(),
                },
                "error": str(e),
            }

    def transcribe_batch(self, audio_files: list[Path]) -> list[dict[str, Any]]:
        results = []

        for audio_file in audio_files:
            try:
                result = self.transcribe_file(audio_file)
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "filename": audio_file.name,
                        "full_text": "",
                        "segments": [],
                        "metadata": {"transcribed_at": datetime.now().isoformat()},
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

    def __del__(self):
        if hasattr(self, "model") and self.model:
            del self.model
