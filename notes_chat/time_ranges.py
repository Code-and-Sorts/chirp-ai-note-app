import re
from datetime import datetime, timedelta
from typing import Optional

from dateutil import tz

from notes_chat.types import TimeRange

WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def parse_time_range(query: str, when_arg: Optional[str] = None) -> Optional[TimeRange]:
    now = datetime.now(tz.tzlocal())

    if when_arg:
        return _parse_when_arg(when_arg, now)

    return _parse_from_query(query, now)


def _parse_when_arg(when_arg: str, now: datetime) -> Optional[TimeRange]:
    when_arg = when_arg.lower().strip()

    if when_arg.startswith("on:"):
        date_str = when_arg[3:]
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
            return TimeRange(
                start=date.replace(hour=0, minute=0, second=0, microsecond=0),
                end_exclusive=date.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                ),
            )
        except ValueError:
            return None

    if ":" in when_arg and len(when_arg.split(":")) == 2:
        start_str, end_str = when_arg.split(":")
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
            end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
            return TimeRange(
                start=start.replace(hour=0, minute=0, second=0, microsecond=0),
                end_exclusive=end.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                ),
            )
        except ValueError:
            return None

    return _parse_keyword_range(when_arg, now)


def _parse_from_query(query: str, now: datetime) -> Optional[TimeRange]:
    query_lower = query.lower()

    patterns_and_keywords = [
        (r"\byesterday\b", "yesterday"),
        (r"\blast (" + "|".join(WEEKDAY_NAMES.keys()) + r")\b", None),
        (r"\blast week\b", "last week"),
        (r"\bthis week\b", "this week"),
        (r"\blast month\b", "last month"),
    ]

    for pattern, fixed_keyword in patterns_and_keywords:
        match = re.search(pattern, query_lower)
        if match:
            keyword = fixed_keyword or match.group().strip()
            return _parse_keyword_range(keyword, now)

    return None


def _parse_keyword_range(keyword: str, now: datetime) -> Optional[TimeRange]:
    keyword = keyword.lower().strip()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if keyword == "yesterday":
        start = today - timedelta(days=1)
        return TimeRange(start=start, end_exclusive=today)

    elif keyword == "last week":
        days_since_monday = now.weekday()
        last_monday = today - timedelta(days=days_since_monday + 7)
        this_monday = today - timedelta(days=days_since_monday)
        return TimeRange(start=last_monday, end_exclusive=this_monday)

    elif keyword == "this week":
        days_since_monday = now.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        next_monday = this_monday + timedelta(days=7)
        return TimeRange(start=this_monday, end_exclusive=next_monday)

    elif keyword == "last month":
        if now.month == 1:
            last_month_start = now.replace(
                year=now.year - 1,
                month=12,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            this_month_start = now.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
        else:
            last_month_start = now.replace(
                month=now.month - 1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            this_month_start = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
        return TimeRange(start=last_month_start, end_exclusive=this_month_start)

    elif keyword.startswith("last ") and len(keyword.split()) == 2:
        _, day_name = keyword.split()
        if day_name in WEEKDAY_NAMES:
            target_weekday = WEEKDAY_NAMES[day_name]
            current_weekday = now.weekday()

            days_back = (current_weekday - target_weekday) % 7
            if days_back == 0:
                days_back = 7

            target_day = today - timedelta(days=days_back)
            return TimeRange(
                start=target_day, end_exclusive=target_day + timedelta(days=1)
            )

    return None
