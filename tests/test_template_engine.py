from datetime import datetime
from unittest.mock import Mock

import pytest

from notes.template_engine import (
    TemplateEngine,
    _extract_meeting_time,
    _format_list_items,
    _substitute_variables,
)


@pytest.fixture
def settings():
    return Mock()


@pytest.fixture
def engine(settings):
    return TemplateEngine(settings)


class TestSubstituteVariables:
    def test_replaces_single_variable(self):
        result = _substitute_variables("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_replaces_multiple_variables(self):
        result = _substitute_variables("{a} and {b}", {"a": "foo", "b": "bar"})
        assert result == "foo and bar"

    def test_converts_non_string_value_to_string(self):
        result = _substitute_variables("count: {n}", {"n": 42})
        assert result == "count: 42"

    def test_returns_template_unchanged_when_no_variables(self):
        result = _substitute_variables("no placeholders", {})
        assert result == "no placeholders"


class TestFormatListItems:
    def test_returns_none_for_empty_list(self):
        assert _format_list_items([]) == "None"

    def test_single_item(self):
        assert _format_list_items(["only item"]) == "- only item"

    def test_multiple_items(self):
        result = _format_list_items(["alpha", "beta", "gamma"])
        assert result == "- alpha\n- beta\n- gamma"


class TestExtractMeetingTime:
    def test_returns_formatted_time_for_datetime_in_date_key(self):
        dt = datetime(2026, 4, 20, 9, 30, 0)
        result = _extract_meeting_time({"date": dt})
        assert "9:30" in result or "09:30" in result

    def test_returns_formatted_time_for_datetime_in_recorded_at_key(self):
        dt = datetime(2026, 4, 20, 14, 0, 0)
        result = _extract_meeting_time({"recorded_at": dt})
        assert "2:00" in result or "14:00" in result

    def test_parses_iso_string_date(self):
        result = _extract_meeting_time({"date": "2026-04-20T09:30:00"})
        assert result != "Unknown time"

    def test_parses_iso_string_with_z_suffix(self):
        result = _extract_meeting_time({"date": "2026-04-20T09:30:00Z"})
        assert result != "Unknown time"

    def test_returns_unknown_time_for_invalid_string(self):
        result = _extract_meeting_time({"date": "not-a-date"})
        assert result == "Unknown time"

    def test_returns_unknown_time_when_no_date_keys(self):
        result = _extract_meeting_time({})
        assert result == "Unknown time"

    def test_returns_unknown_time_for_none_value(self):
        result = _extract_meeting_time({"date": None, "recorded_at": None})
        assert result == "Unknown time"

    def test_date_key_takes_precedence_over_recorded_at(self):
        dt_date = datetime(2026, 1, 1, 8, 0, 0)
        dt_recorded = datetime(2026, 6, 15, 18, 0, 0)
        result_with_date = _extract_meeting_time(
            {"date": dt_date, "recorded_at": dt_recorded}
        )
        result_only_recorded = _extract_meeting_time({"recorded_at": dt_recorded})
        assert result_with_date != result_only_recorded


class TestTemplateEngineInit:
    def test_stores_settings(self, settings):
        engine = TemplateEngine(settings)
        assert engine.settings is settings


class TestRenderDailyNotes:
    def test_includes_formatted_date(self, engine):
        result = engine.render_daily_notes(
            date=datetime(2026, 4, 20),
            meeting_sections=["## Meeting A"],
            meeting_count=1,
            total_duration=3600.0,
        )
        assert "April 20, 2026" in result

    def test_joins_multiple_sections(self, engine):
        result = engine.render_daily_notes(
            date=datetime(2026, 4, 20),
            meeting_sections=["## Section A", "## Section B"],
            meeting_count=2,
            total_duration=120.0,
        )
        assert "## Section A" in result
        assert "## Section B" in result

    def test_includes_meeting_count(self, engine):
        result = engine.render_daily_notes(
            date=datetime(2026, 4, 20),
            meeting_sections=[],
            meeting_count=3,
            total_duration=0.0,
        )
        assert "3" in result

    def test_includes_total_duration(self, engine):
        result = engine.render_daily_notes(
            date=datetime(2026, 4, 20),
            meeting_sections=[],
            meeting_count=0,
            total_duration=90.0,
        )
        assert "1m 30s" in result


class TestRenderMeetingSection:
    def _full_meeting_data(self, **overrides) -> dict:
        base = {
            "meeting_title": "Sprint Review",
            "executive_summary": "Reviewed sprint progress.",
            "agenda": ["Item A", "Item B"],
            "discussion_highlights": ["Highlight 1"],
            "action_items": ["Action 1"],
            "decisions": ["Decision 1"],
            "open_questions": ["Question 1"],
            "next_steps": ["Step 1"],
            "metadata": {"date": datetime(2026, 4, 20, 10, 0, 0), "duration": 3600.0},
        }
        base.update(overrides)
        return base

    def test_includes_meeting_title(self, engine):
        result = engine.render_meeting_section(self._full_meeting_data())
        assert "Sprint Review" in result

    def test_includes_executive_summary(self, engine):
        result = engine.render_meeting_section(self._full_meeting_data())
        assert "Reviewed sprint progress." in result

    def test_formats_list_fields(self, engine):
        result = engine.render_meeting_section(self._full_meeting_data())
        assert "- Item A" in result
        assert "- Item B" in result
        assert "- Action 1" in result
        assert "- Decision 1" in result

    def test_uses_unknown_duration_when_zero(self, engine):
        data = self._full_meeting_data()
        data["metadata"] = {"duration": 0}
        result = engine.render_meeting_section(data)
        assert "Unknown" in result

    def test_uses_untitled_meeting_when_title_missing(self, engine):
        data = self._full_meeting_data()
        del data["meeting_title"]
        result = engine.render_meeting_section(data)
        assert "Untitled Meeting" in result

    def test_uses_no_summary_when_summary_missing(self, engine):
        data = self._full_meeting_data()
        del data["executive_summary"]
        result = engine.render_meeting_section(data)
        assert "No summary available" in result

    def test_empty_lists_render_as_none(self, engine):
        data = self._full_meeting_data()
        data["agenda"] = []
        result = engine.render_meeting_section(data)
        assert "None" in result

    def test_missing_metadata_uses_unknown_time(self, engine):
        data = self._full_meeting_data()
        data["metadata"] = {}
        result = engine.render_meeting_section(data)
        assert "Unknown time" in result

    def test_formats_duration_from_metadata(self, engine):
        result = engine.render_meeting_section(self._full_meeting_data())
        assert "1h 0m" in result
