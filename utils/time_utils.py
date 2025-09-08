from datetime import datetime
from typing import Optional


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_date_string() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_time_string() -> str:
    return datetime.now().strftime("%H:%M:%S")


def get_filename_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    try:
        timestamp_part = filename.split("_")[0] + "_" + filename.split("_")[1]
        if len(timestamp_part) == 15:  # YYYYMMDD_HHMMSS
            return datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
    except (IndexError, ValueError):
        pass
    return None


def get_recording_duration(start_time: datetime) -> float:
    return (datetime.now() - start_time).total_seconds()


def should_warn_user(
    start_time: datetime,
    warning_minutes: int,
    last_warning: Optional[datetime] = None,
    warning_interval: int = 15,
) -> bool:
    elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60

    if elapsed_minutes < warning_minutes:
        return False

    if last_warning is None:
        return True

    minutes_since_warning = (datetime.now() - last_warning).total_seconds() / 60
    return minutes_since_warning >= warning_interval


def format_meeting_time(timestamp: datetime) -> str:
    return timestamp.strftime("%I:%M %p").lstrip("0")


def get_daily_note_filename(date: Optional[datetime] = None) -> str:
    if date is None:
        date = datetime.now()
    return f"meetings_{date.strftime('%Y_%m_%d')}.md"


def is_same_day(dt1: datetime, dt2: datetime) -> bool:
    return dt1.date() == dt2.date()
