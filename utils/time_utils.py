from datetime import datetime
from pathlib import Path
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


def derive_recording_id(
    audio_file: Path, recorded_at: Optional[datetime] = None
) -> str:
    timestamp = parse_timestamp_from_filename(audio_file.name)

    if timestamp is None and recorded_at is not None:
        timestamp = recorded_at

    if timestamp is None:
        try:
            timestamp = datetime.fromtimestamp(audio_file.stat().st_mtime)
        except (OSError, ValueError):
            timestamp = datetime.now()

    return timestamp.strftime("%Y%m%d_%H%M%S")


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


def parse_timeframe(text: str) -> Optional[int]:
    """Parse the design's ``30s`` / ``5m`` / ``1h`` timeframe input into minutes.

    Returns ``None`` for empty input (user pressed ⏎ to skip). Raises
    ``ValueError`` for unrecognized formats or non-positive durations.
    Rounds up so a 30s request still gets at least 1 minute and a 2.5m
    request gets 3, never less than what the user asked for.
    """
    import math

    text = (text or "").strip().lower()
    if not text:
        return None

    suffix_minutes = {"s": 1 / 60, "m": 1, "h": 60}
    if text[-1] in suffix_minutes:
        amount_str, unit = text[:-1], text[-1]
    else:
        amount_str, unit = text, "m"

    try:
        amount = float(amount_str)
    except ValueError as exc:
        raise ValueError(f"unrecognized timeframe: {text!r}") from exc

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"timeframe must be positive: {text!r}")

    minutes = amount * suffix_minutes[unit]
    return max(1, math.ceil(minutes))
