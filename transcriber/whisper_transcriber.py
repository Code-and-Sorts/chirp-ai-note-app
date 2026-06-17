import logging
import platform
import tomllib
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chirp.exceptions import WhisperModelLoadError
from config.settings import ChirpSettings
from notes.constants import DEFAULT_MEETING_NAME
from utils.file_utils import META_FILENAME, get_file_size_mb
from utils.time_utils import derive_recording_id, parse_timestamp_from_filename

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_MLX_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _resolve_model_repo(name: str) -> str:
    if "/" in name:
        return name
    return _MLX_MODEL_REPOS.get(name, name)


def _import_mlx_whisper():
    # Imported lazily: mlx-whisper only installs on macOS arm64, so a top-level
    # import would break imports/tests on every other platform.
    import mlx_whisper
    import mlx_whisper.audio
    import mlx_whisper.load_models

    return mlx_whisper


class WhisperTranscriber:
    def __init__(self, settings: ChirpSettings):
        self.settings = settings
        self.model = None
        self._vad_model = None
        self.model_repo = _resolve_model_repo(settings.models.whisper)
        self._load_model()

    def _load_model(self):
        mlx_whisper = _import_mlx_whisper()
        try:
            # Load eagerly so an invalid model name fails fast here rather than
            # on the first transcribe() call.
            self.model = mlx_whisper.load_models.load_model(self.model_repo)
        except Exception as exc:
            raise WhisperModelLoadError(
                f"Could not download or load the Whisper model {self.model_repo!r}. "
                "Check your network connection and free disk space, then retry. "
                "If the problem persists, set a valid model name in your config "
                f"(models.whisper). Underlying error: {exc}"
            ) from exc

    def _get_optimal_device(self) -> str:
        return "mlx" if platform.system() == "Darwin" else "cpu"

    def _get_compute_type(self) -> str:
        return "float16"

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

        mlx_whisper = _import_mlx_whisper()
        start_time = datetime.now()
        audio_metadata = self._read_audio_metadata(audio_file_path)
        recording_datetime = self._get_recording_datetime(
            audio_file_path, audio_metadata
        )
        recording_id = derive_recording_id(audio_file_path, recording_datetime)
        meeting_name = self._get_meeting_name(audio_metadata)
        device = self._get_optimal_device()
        compute_type = self._get_compute_type()

        # mlx-whisper has no beam search (it raises NotImplementedError), so the
        # fast/normal split is driven by word timestamps and hallucination
        # filtering rather than beam_size.
        if fast_mode:
            word_timestamps = False
            hallucination_silence_threshold = None
        else:
            word_timestamps = True
            hallucination_silence_threshold = 2.0

        try:
            audio = mlx_whisper.audio.load_audio(str(audio_file_path))
            duration = len(audio) / SAMPLE_RATE

            # mlx-whisper has no built-in VAD. We feed silero-detected speech
            # regions via clip_timestamps so silence is skipped while segment
            # timestamps stay in the original recording's timeline.
            clip_timestamps = self._speech_clip_timestamps(audio)

            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_repo,
                language=language,
                temperature=0.0,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                initial_prompt=None,
                word_timestamps=word_timestamps,
                hallucination_silence_threshold=hallucination_silence_threshold,
                clip_timestamps=clip_timestamps,
            )

            transcript_segments: list[dict[str, Any]] = []
            transcript_text_parts: list[str] = []

            for segment in result.get("segments", []):
                segment_text = (segment.get("text") or "").strip()
                segment_data = {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment_text,
                    "avg_logprob": segment.get("avg_logprob"),
                    "no_speech_prob": segment.get("no_speech_prob"),
                }
                if word_timestamps and segment.get("words"):
                    segment_data["words"] = [
                        {
                            "word": word.get("word"),
                            "start": word.get("start"),
                            "end": word.get("end"),
                            "probability": word.get("probability"),
                        }
                        for word in segment["words"]
                    ]
                transcript_segments.append(segment_data)
                if segment_text:
                    transcript_text_parts.append(segment_text)

                if on_segment and segment_text:
                    try:
                        on_segment(segment_data)
                    except Exception as exc:  # noqa: BLE001 - arbitrary user callback
                        logger.warning(
                            "on_segment callback failed: %s", exc, exc_info=True
                        )

            full_text = " ".join(transcript_text_parts).strip()
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            transcribed_at = end_time.isoformat()
            segment_count = len(transcript_segments)
            word_count = len(full_text.split()) if full_text else 0
            character_count = len(full_text)
            recording_completed_at = (
                recording_datetime + timedelta(seconds=duration) if duration else None
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
                "recording_length_seconds": duration,
                "duration": duration,
                "language": result.get("language"),
                "language_probability": None,
                "transcribed_at": transcribed_at,
                "transcription_time": processing_time,
                "transcription_time_seconds": processing_time,
                "model": self.settings.models.whisper,
                "model_repo": self.model_repo,
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

        except Exception as e:  # noqa: BLE001 - transcription can fail in many ways; return structured error
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

    def _speech_clip_timestamps(self, audio) -> str | list[float]:
        """Build mlx-whisper clip_timestamps from a silero VAD pass.

        Returns the sentinel ``"0"`` (whole-file) when VAD is disabled or no
        speech is detected; otherwise a flat ``[start, end, start, end, ...]``
        list of speech regions in seconds.
        """
        if not self.settings.audio.vad_enabled:
            return "0"

        timestamps = self._detect_speech(audio)
        if not timestamps:
            return "0"

        clips: list[float] = []
        for region in timestamps:
            clips.append(region["start"] / SAMPLE_RATE)
            clips.append(region["end"] / SAMPLE_RATE)
        return clips

    def _detect_speech(self, audio) -> list[dict[str, int]]:
        import torch
        from silero_vad import get_speech_timestamps

        model = self._load_vad_model()
        params = self.settings.audio.vad_parameters.model_dump()
        audio_tensor = torch.from_numpy(audio)
        timestamps: list[dict[str, int]] = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=SAMPLE_RATE,
            **params,
        )
        return timestamps

    def _load_vad_model(self):
        if self._vad_model is None:
            from silero_vad import load_silero_vad

            self._vad_model = load_silero_vad()
        return self._vad_model

    def transcribe_batch(self, audio_files: list[Path]) -> list[dict[str, Any]]:
        results = []

        for audio_file in audio_files:
            try:
                result = self.transcribe_file(audio_file)
                results.append(result)
            except Exception as e:  # noqa: BLE001 - batch transcription; return error per file
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
                            "recording_id": derive_recording_id(
                                audio_file, recording_datetime
                            ),
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
            "model_repo": self.model_repo,
            "device": self._get_optimal_device(),
            "compute_type": self._get_compute_type(),
            "loaded": self.model is not None,
        }

    def _read_audio_metadata(self, audio_file_path: Path) -> dict | None:
        meta_path = audio_file_path.parent / META_FILENAME

        if meta_path.exists():
            try:
                with meta_path.open("rb") as fh:
                    data = tomllib.load(fh)
                    return dict(data) if isinstance(data, dict) else None
            except (OSError, tomllib.TOMLDecodeError):
                pass

        return None

    def _get_recording_datetime(
        self, audio_file_path: Path, audio_metadata: dict | None
    ) -> datetime:
        if audio_metadata:
            date_value = audio_metadata.get("date") or audio_metadata.get("recorded_at")
            if isinstance(date_value, datetime):
                return date_value
            if isinstance(date_value, str) and date_value.strip():
                cleaned = date_value.strip().replace("Z", "+00:00")
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

    def _get_meeting_name(self, audio_metadata: dict | None) -> str:
        if audio_metadata:
            title = audio_metadata.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
        return DEFAULT_MEETING_NAME

    def close(self) -> None:
        """Drop references to the loaded models. Idempotent.

        mlx-whisper keeps its own module-level LRU cache of the converted model;
        clearing our reference is the contract this class can honor.
        """
        self.model = None
        self._vad_model = None

    def __del__(self):
        self.close()
