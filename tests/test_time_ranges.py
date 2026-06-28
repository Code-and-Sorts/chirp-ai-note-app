from datetime import datetime

from dateutil import tz

from notes_chat.time_ranges import parse_time_range


def _at(year: int, month: int, day: int, hour: int = 14, minute: int = 30) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=tz.tzlocal())


class TestParseTimeRange:
    def test_yesterday(self):
        result = parse_time_range("what happened yesterday?", now=_at(2025, 1, 15))
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 14).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 15).date()

    def test_last_tuesday(self):
        result = parse_time_range("meetings last tuesday", now=_at(2025, 1, 15))
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 14).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 15).date()

    def test_last_friday(self):
        result = parse_time_range(
            "what did we decide last friday", now=_at(2025, 1, 15)
        )
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 10).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 11).date()

    def test_last_week(self):
        result = parse_time_range("action items from last week", now=_at(2025, 1, 15))
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 6).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 13).date()

    def test_this_week(self):
        result = parse_time_range("meetings this week", now=_at(2025, 1, 15))
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 13).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 20).date()

    def test_last_month(self):
        result = parse_time_range("decisions from last month", now=_at(2025, 1, 15))
        assert result is not None
        assert result.start.date() == datetime(2024, 12, 1).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 1).date()

    def test_when_arg_on_date(self):
        result = parse_time_range("what happened", when_arg="on:2025-01-10")
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 10).date()

    def test_when_arg_date_range(self):
        result = parse_time_range("meetings", when_arg="2025-01-10:2025-01-15")
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 10).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 15).date()

    def test_when_arg_yesterday(self):
        result = parse_time_range(
            "meetings", when_arg="yesterday", now=_at(2025, 1, 15)
        )
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 14).date()

    def test_no_match(self):
        result = parse_time_range("general question about meetings")
        assert result is None

    def test_invalid_date_format(self):
        result = parse_time_range("meetings", when_arg="on:invalid-date")
        assert result is None

    def test_invalid_range_format(self):
        result = parse_time_range("meetings", when_arg="2025-01-10:invalid")
        assert result is None

    def test_last_month_january(self):
        result = parse_time_range("last month meetings", now=_at(2025, 1, 1))
        assert result is not None
        assert result.start.date() == datetime(2024, 12, 1).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 1).date()

    def test_last_monday_when_today_is_monday(self):
        result = parse_time_range("last monday meetings", now=_at(2025, 1, 13))
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 6).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 7).date()

    def test_all_weekdays(self):
        weekdays = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        for day in weekdays:
            result = parse_time_range(f"meetings last {day}", now=_at(2025, 1, 15))
            assert result is not None, f"Failed for {day}"
