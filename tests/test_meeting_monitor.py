from __future__ import annotations

import threading
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from config.settings import MonitoringConfig
from recorder.meeting_monitor import MeetingMonitor


def _monitor(
    start_offset_minutes: float = 0.0,
    warning_minutes: int = 60,
    should_stop: bool = False,
    on_warning=None,
) -> MeetingMonitor:
    return MeetingMonitor(
        config=MonitoringConfig(warning_minutes=warning_minutes),
        start_time=datetime.now() - timedelta(minutes=start_offset_minutes),
        warning_callback=on_warning or (lambda _minutes: None),
        should_stop_callback=lambda: should_stop,
    )


class TestElapsedTime:
    def test_elapsed_minutes_reflect_start_time(self):
        monitor = _monitor(start_offset_minutes=42)
        assert monitor.get_elapsed_minutes() == 42
        assert monitor.get_elapsed_time() == pytest.approx(42 * 60, abs=5)

    def test_over_warning_threshold_boundary(self):
        assert _monitor(61, warning_minutes=60).is_over_warning_threshold()
        assert _monitor(60, warning_minutes=60).is_over_warning_threshold()
        assert not _monitor(30, warning_minutes=60).is_over_warning_threshold()


class TestLifecycle:
    def test_start_is_idempotent(self):
        monitor = _monitor()
        monitor.start()
        try:
            first_thread = monitor.monitor_thread
            monitor.start()
            assert monitor.monitor_thread is first_thread
        finally:
            monitor.stop()

    def test_stop_wakes_and_joins_the_loop(self):
        monitor = _monitor()
        monitor.start()
        assert monitor.monitor_thread.is_alive()
        monitor.stop()
        assert monitor.is_monitoring is False
        assert not monitor.monitor_thread.is_alive()

    def test_stop_without_start_is_safe(self):
        _monitor().stop()


class TestMonitorLoop:
    def test_fires_warning_when_over_threshold(self):
        warned = threading.Event()
        warned_minutes: list[int] = []

        def on_warning(minutes: int) -> None:
            warned_minutes.append(minutes)
            warned.set()

        monitor = _monitor(
            start_offset_minutes=90, warning_minutes=60, on_warning=on_warning
        )
        monitor.start()
        try:
            assert warned.wait(timeout=5.0)
        finally:
            monitor.stop()
        assert warned_minutes[0] >= 90
        assert monitor.last_warning is not None

    def test_should_stop_shows_popup_and_exits_loop(self):
        with patch("utils.popup_manager.PopupManager") as popup_cls:
            monitor = _monitor(should_stop=True)
            monitor.start()
            monitor.monitor_thread.join(timeout=5.0)
            try:
                assert not monitor.monitor_thread.is_alive()
            finally:
                monitor.stop()
        message = popup_cls.return_value.show_error.call_args.args[0]
        assert "stopped automatically" in message
        assert str(monitor.config.max_recording_hours) in message

    def test_loop_survives_warning_callback_errors(self):
        calls = threading.Event()

        def exploding_warning(_minutes: int) -> None:
            calls.set()
            raise RuntimeError("boom")

        monitor = _monitor(
            start_offset_minutes=90, warning_minutes=60, on_warning=exploding_warning
        )
        monitor.start()
        try:
            assert calls.wait(timeout=5.0)
            assert monitor.monitor_thread.is_alive()
        finally:
            monitor.stop()
