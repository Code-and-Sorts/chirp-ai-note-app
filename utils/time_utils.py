from datetime import datetime
from pathlib import Path


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


def parse_timestamp_from_filename(filename: str) -> datetime | None:
    try:
        timestamp_part = filename.split("_")[0] + "_" + filename.split("_")[1]
        if len(timestamp_part) == 15:  # YYYYMMDD_HHMMSS
            return datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
    except (IndexError, ValueError):
        pass
    return None


def derive_recording_id(audio_file: Path, recorded_at: datetime | None = None) -> str:
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
    last_warning: datetime | None = None,
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


def get_daily_note_filename(date: datetime | None = None) -> str:
    if date is None:
        date = datetime.now()
    return f"meetings_{date.strftime('%Y_%m_%d')}.md"


def is_same_day(dt1: datetime, dt2: datetime) -> bool:
    return dt1.date() == dt2.date()


SUFFIX_MINUTES: dict[str, float] = {
    "s": 1 / 60,
    "m": 1,
    "h": 60,
    "d": 24 * 60,
    "w": 7 * 24 * 60,
}


def parse_timeframe(text: str) -> int | None:
    """Parse the design's ``30s`` / ``5m`` / ``1h`` / ``2d`` / ``1w`` timeframe input into minutes.

    Returns ``None`` for empty input (user pressed ⏎ to skip). Raises
    ``ValueError`` for unrecognized formats or non-positive durations.
    Rounds up so a 30s request still gets at least 1 minute and a 2.5m
    request gets 3, never less than what the user asked for.
    """
    import math

    text = (text or "").strip().lower()
    if not text:
        return None

    if text[-1] in SUFFIX_MINUTES:
        amount_str, unit = text[:-1], text[-1]
    else:
        amount_str, unit = text, "m"

    try:
        amount = float(amount_str)
    except ValueError as exc:
        raise ValueError(f"unrecognized timeframe: {text!r}") from exc

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"timeframe must be positive: {text!r}")

    minutes = amount * SUFFIX_MINUTES[unit]
    return max(1, math.ceil(minutes))


def parse_since(text: str) -> int:
    """Parse ``--since`` arguments (``30d`` / ``2w`` / ``48h``) into minutes.

    Stricter than ``parse_timeframe``: empty input is rejected, and durations
    shorter than one hour are rejected too. ``5m`` / ``30s`` are not useful
    recency filters and would surprise users who expected ``m`` to mean months.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("--since requires a value (e.g. 30d, 2w, 48h)")

    minutes = parse_timeframe(cleaned)
    assert minutes is not None
    if minutes < 60:
        raise ValueError(
            f"--since must be at least 1h (got {cleaned!r}); use 1h, 2d, 1w, etc."
        )
    return minutes
