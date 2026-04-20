import pytest

from utils.time_utils import parse_timeframe


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
