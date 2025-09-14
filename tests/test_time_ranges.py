from datetime import datetime

from freezegun import freeze_time

from notes_chat.time_ranges import parse_time_range


class TestParseTimeRange:
    @freeze_time("2025-01-15 14:30:00")
    def test_yesterday(self):
        result = parse_time_range("what happened yesterday?")
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 14).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 15).date()

    @freeze_time("2025-01-15 14:30:00")
    def test_last_tuesday(self):
        result = parse_time_range("meetings last tuesday")
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 14).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 15).date()

    @freeze_time("2025-01-15 14:30:00")
    def test_last_friday(self):
        result = parse_time_range("what did we decide last friday")
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 10).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 11).date()

    @freeze_time("2025-01-15 14:30:00")
    def test_last_week(self):
        result = parse_time_range("action items from last week")
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 6).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 13).date()

    @freeze_time("2025-01-15 14:30:00")
    def test_this_week(self):
        result = parse_time_range("meetings this week")
        assert result is not None
        assert result.start.date() == datetime(2025, 1, 13).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 20).date()

    @freeze_time("2025-01-15 14:30:00")
    def test_last_month(self):
        result = parse_time_range("decisions from last month")
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
        with freeze_time("2025-01-15 14:30:00"):
            result = parse_time_range("meetings", when_arg="yesterday")
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

    @freeze_time("2025-01-01 14:30:00")
    def test_last_month_january(self):
        result = parse_time_range("last month meetings")
        assert result is not None
        assert result.start.date() == datetime(2024, 12, 1).date()
        assert result.end_exclusive.date() == datetime(2025, 1, 1).date()

    @freeze_time("2025-01-13 14:30:00")
    def test_last_monday_when_today_is_monday(self):
        result = parse_time_range("last monday meetings")
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
            result = parse_time_range(f"meetings last {day}")
            assert result is not None, f"Failed for {day}"
