import pytest

from utils.time_utils import parse_since, parse_timeframe


def test_empty_returns_none():
    assert parse_timeframe("") is None
    assert parse_timeframe("   ") is None


def test_minutes_are_default_unit():
    assert parse_timeframe("10") == 10
    assert parse_timeframe("5m") == 5


def test_seconds_round_up_to_one_minute():
    assert parse_timeframe("30s") == 1
    assert parse_timeframe("90s") == 2


def test_hours():
    assert parse_timeframe("1h") == 60
    assert parse_timeframe("2h") == 120


def test_handles_uppercase_and_whitespace():
    assert parse_timeframe("  10M  ") == 10


def test_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timeframe("soon")


def test_fractional_rounds_up():
    assert parse_timeframe("2.5m") == 3
    assert parse_timeframe("0.6h") == 36
    assert parse_timeframe("45s") == 1


def test_rejects_zero_and_negative():
    for bad in ["0m", "0", "0h", "-1m", "-30s"]:
        with pytest.raises(ValueError):
            parse_timeframe(bad)


def test_default_unit_tolerates_whitespace():
    assert parse_timeframe("  7  ") == 7


def test_days_and_weeks():
    assert parse_timeframe("1d") == 24 * 60
    assert parse_timeframe("30d") == 30 * 24 * 60
    assert parse_timeframe("2w") == 14 * 24 * 60


class TestParseSince:
    def test_accepts_hours_days_weeks(self):
        assert parse_since("1h") == 60
        assert parse_since("48h") == 48 * 60
        assert parse_since("30d") == 30 * 24 * 60
        assert parse_since("2w") == 14 * 24 * 60

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="--since requires a value"):
            parse_since("")
        with pytest.raises(ValueError, match="--since requires a value"):
            parse_since("   ")

    def test_rejects_sub_hour_units(self):
        for bad in ["5m", "30s", "59m"]:
            with pytest.raises(ValueError, match="at least 1h"):
                parse_since(bad)

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_since("soon")

    def test_rejects_zero_and_negative(self):
        for bad in ["0d", "0h", "-1d"]:
            with pytest.raises(ValueError):
                parse_since(bad)
