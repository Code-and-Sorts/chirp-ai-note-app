import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_audio_filename(title: Optional[str] = None, extension: str = "wav") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if title:
        sanitized_title = sanitize_filename(title)
        return f"{timestamp}_{sanitized_title}.{extension}"
    return f"{timestamp}.{extension}"


def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join(c for c in filename if c not in invalid_chars)
    return sanitized.strip()[:50]


def get_audio_files(directory: Path) -> list[Path]:
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".aac"}
    audio_files: list[Path] = []

    if not directory.exists():
        return audio_files

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)

    return sorted(audio_files, key=lambda x: x.stat().st_mtime)


def get_transcription_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    transcription_paths: dict[Path, float] = {}

    for file_path in directory.rglob("*.json.gz"):
        if file_path.is_file() and file_path.name != "metadata.json":
            try:
                transcription_paths[file_path] = file_path.stat().st_mtime
            except (OSError, ValueError):
                continue

    return sorted(
        transcription_paths.keys(), key=lambda path: transcription_paths[path]
    )


def get_notes_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in {".md", ".txt"}
        ],
        key=lambda x: x.stat().st_mtime,
    )


def move_file(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    except Exception:
        return False


def copy_file(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return True
    except Exception:
        return False


def get_file_size_mb(file_path: Path) -> float:
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def ensure_directory(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def clean_old_files(directory: Path, days_old: int = 30) -> int:
    if not directory.exists():
        return 0

    cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    deleted_count = 0

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception:
                pass

    return deleted_count
