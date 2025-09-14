import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from config.settings import ChirpSettings
from transcriber.compression import JSONCompressor
from transcriber.whisper_transcriber import WhisperTranscriber
from utils.file_utils import get_audio_files
from utils.popup_manager import PopupManager


class BatchProcessor:
    def __init__(self, settings: ChirpSettings):
        self.settings = settings
        self.transcriber = WhisperTranscriber(settings)
        self.compressor = JSONCompressor()
        self.popup_manager = PopupManager()
        self._lock = threading.Lock()

    def process_files(
        self,
        audio_files: list[Path],
        force: bool = False,
        progress_callback: Optional[Callable] = None,
        max_workers: int = 1,  # Keep at 1 for Whisper to avoid memory issues
    ) -> list[dict[str, Any]]:
        files_to_process = self._filter_files_to_process(audio_files, force)

        if not files_to_process:
            return []

        self.settings.directories.transcriptions.mkdir(parents=True, exist_ok=True)

        results = []

        if max_workers == 1:
            results = self._process_sequentially(files_to_process, progress_callback)
        else:
            results = self._process_concurrently(
                files_to_process, progress_callback, max_workers
            )

        success_count = sum(1 for r in results if r["success"])

        if success_count > 0:
            self.popup_manager.show_transcription_complete(success_count)

        return results

    def _process_sequentially(
        self, files_to_process: list[Path], progress_callback: Optional[Callable] = None
    ) -> list[dict[str, Any]]:
        results = []

        for audio_file in files_to_process:
            try:
                result = self._process_single_file(audio_file)
                results.append(result)

                if progress_callback:
                    progress_callback()

            except Exception as e:
                error_result = {
                    "success": False,
                    "filename": audio_file.name,
                    "error": str(e),
                    "transcribed_at": datetime.now().isoformat(),
                }
                results.append(error_result)

                if progress_callback:
                    progress_callback()

        return results

    def _process_concurrently(
        self,
        files_to_process: list[Path],
        progress_callback: Optional[Callable] = None,
        max_workers: int = 2,
    ) -> list[dict[str, Any]]:
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, audio_file): audio_file
                for audio_file in files_to_process
            }

            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_result = {
                        "success": False,
                        "filename": audio_file.name,
                        "error": str(e),
                        "transcribed_at": datetime.now().isoformat(),
                    }
                    results.append(error_result)

                if progress_callback:
                    progress_callback()

        return results

    def _process_single_file(self, audio_file: Path) -> dict[str, Any]:
        output_path = self._get_output_path(audio_file)

        transcription_result = self.transcriber.transcribe_file(audio_file)

        if transcription_result["success"]:
            if self.compressor.compress_json(transcription_result, output_path):
                transcription_result["output_path"] = str(output_path)
                transcription_result["compressed"] = True
            else:
                transcription_result["compressed"] = False
                transcription_result["compression_error"] = (
                    "Failed to compress transcription"
                )

        return transcription_result

    def _filter_files_to_process(
        self, audio_files: list[Path], force: bool
    ) -> list[Path]:
        if force:
            return audio_files

        files_to_process = []

        for audio_file in audio_files:
            output_path = self._get_output_path(audio_file)
            if not output_path.exists():
                files_to_process.append(audio_file)

        return files_to_process

    def _get_output_path(self, audio_file: Path) -> Path:
        base_name = audio_file.stem
        return self.settings.directories.transcriptions / f"{base_name}.json.gz"

    def get_transcription_data(self, audio_file_path: Path) -> Optional[dict[str, Any]]:
        output_path = self._get_output_path(audio_file_path)

        if not output_path.exists():
            return None

        try:
            return self.compressor.decompress_json(output_path)
        except Exception:
            return None

    def process_directory(
        self,
        directory: Path,
        force: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        audio_files = get_audio_files(directory)

        if not audio_files:
            return {
                "success": True,
                "processed_count": 0,
                "total_count": 0,
                "results": [],
                "message": f"No audio files found in {directory}",
            }

        results = self.process_files(audio_files, force, progress_callback)

        success_count = sum(1 for r in results if r["success"])

        return {
            "success": True,
            "processed_count": success_count,
            "total_count": len(audio_files),
            "results": results,
            "message": f"Processed {success_count}/{len(audio_files)} files successfully",
        }

    def get_processing_stats(self) -> dict[str, Any]:
        transcription_files = list(
            self.settings.directories.transcriptions.glob("*.json.gz")
        )

        total_files = len(transcription_files)
        total_size = sum(f.stat().st_size for f in transcription_files)

        return {
            "total_transcriptions": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "transcription_directory": str(self.settings.directories.transcriptions),
            "model_info": self.transcriber.get_model_info(),
        }
