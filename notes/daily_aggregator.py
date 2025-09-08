from collections import defaultdict
from datetime import datetime
from pathlib import Path

from config.settings import ChirpSettings
from transcriber.compression import JSONCompressor
from utils.time_utils import is_same_day, parse_timestamp_from_filename


class DailyAggregator:
    def __init__(self, settings: ChirpSettings):
        self.settings = settings
        self.compressor = JSONCompressor()

    def group_transcriptions_by_day(
        self, transcription_files: list[Path]
    ) -> dict[datetime, list[Path]]:
        daily_groups = defaultdict(list)

        for transcription_file in transcription_files:
            meeting_date = self._extract_meeting_date(transcription_file)
            if meeting_date:
                daily_groups[
                    meeting_date.replace(hour=0, minute=0, second=0, microsecond=0)
                ].append(transcription_file)

        return dict(daily_groups)

    def _extract_meeting_date(self, transcription_file: Path) -> datetime:
        audio_filename = transcription_file.stem.replace(".json", "")

        timestamp_from_filename = parse_timestamp_from_filename(audio_filename)
        if timestamp_from_filename:
            return timestamp_from_filename

        try:
            transcription_data = self.compressor.decompress_json(transcription_file)
            recorded_at = transcription_data.get("metadata", {}).get("recorded_at")

            if recorded_at:
                try:
                    return datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
                except ValueError:
                    pass
        except Exception:
            pass

        file_mtime = datetime.fromtimestamp(transcription_file.stat().st_mtime)
        return file_mtime

    def get_transcriptions_for_date(
        self, target_date: datetime, transcription_files: list[Path]
    ) -> list[Path]:
        matching_files = []

        for transcription_file in transcription_files:
            meeting_date = self._extract_meeting_date(transcription_file)
            if meeting_date and is_same_day(meeting_date, target_date):
                matching_files.append(transcription_file)

        return sorted(matching_files, key=lambda x: self._extract_meeting_date(x))

    def get_daily_statistics(self, transcription_files: list[Path]) -> dict[str, any]:
        daily_groups = self.group_transcriptions_by_day(transcription_files)

        stats = {}
        total_meetings = 0
        total_duration = 0.0

        for date, files in daily_groups.items():
            day_duration = 0.0
            day_meetings = len(files)

            for transcription_file in files:
                try:
                    transcription_data = self.compressor.decompress_json(
                        transcription_file
                    )
                    duration = transcription_data.get("metadata", {}).get("duration", 0)
                    day_duration += duration
                except Exception:
                    pass

            stats[date.strftime("%Y-%m-%d")] = {
                "date": date,
                "meeting_count": day_meetings,
                "total_duration": day_duration,
                "files": [f.name for f in files],
            }

            total_meetings += day_meetings
            total_duration += day_duration

        return {
            "daily_stats": stats,
            "summary": {
                "total_days": len(daily_groups),
                "total_meetings": total_meetings,
                "total_duration": total_duration,
                "average_meetings_per_day": total_meetings / len(daily_groups)
                if daily_groups
                else 0,
            },
        }

    def find_recent_meetings(
        self, transcription_files: list[Path], days: int = 7
    ) -> list[Path]:
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)

        recent_files = []

        for transcription_file in transcription_files:
            meeting_date = self._extract_meeting_date(transcription_file)
            if meeting_date and meeting_date >= cutoff_date:
                recent_files.append(transcription_file)

        return sorted(
            recent_files, key=lambda x: self._extract_meeting_date(x), reverse=True
        )
