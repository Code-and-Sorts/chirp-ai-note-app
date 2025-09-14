from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from notes.daily_aggregator import DailyAggregator


class TestDailyAggregator:
    @pytest.fixture
    def mock_settings(self):
        return Mock()

    @pytest.fixture
    def aggregator(self, mock_settings):
        with patch("notes.daily_aggregator.JSONCompressor"):
            return DailyAggregator(mock_settings)

    def test_initialization(self, mock_settings):
        with patch("notes.daily_aggregator.JSONCompressor"):
            aggregator = DailyAggregator(mock_settings)
            assert aggregator.settings == mock_settings

    def test_group_transcriptions_by_day(self, aggregator):
        # Mock transcription files with different dates
        files = [Mock(), Mock(), Mock()]

        with patch.object(aggregator, "_extract_meeting_date") as mock_extract:
            # Return different dates for grouping
            mock_extract.side_effect = [
                datetime(2023, 12, 1, 10, 0, 0),
                datetime(2023, 12, 1, 14, 0, 0),  # Same day as first
                datetime(2023, 12, 2, 9, 0, 0),  # Different day
            ]

            result = aggregator.group_transcriptions_by_day(files)

            assert len(result) == 2  # Two different days
            # First day should have 2 files
            dec_1 = datetime(2023, 12, 1, 0, 0, 0)
            assert len(result[dec_1]) == 2
