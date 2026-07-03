from __future__ import annotations

import queue
import threading
import time

import pytest
from rich.console import Console

from recorder.live_dashboard import LiveDashboard
from recorder.live_types import DashboardEvent, TranscriptSegment


@pytest.fixture
def dashboard(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    return LiveDashboard(
        console=Console(width=100, height=40, force_terminal=False),
        event_queue=queue.Queue(),
        stop_event=threading.Event(),
        start_time=time.monotonic(),
    )


def _segment(text: str, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(
        text=text, start=start, end=start + 1.0, words=len(text.split())
    )


class TestHandleEvent:
    def test_transcript_event_appends_segments_and_updates_totals(self, dashboard):
        segments = [_segment("hello world"), _segment("second line", start=2.0)]
        dashboard._handle_event(
            DashboardEvent(
                type="transcript",
                payload={"segments": segments, "language": "en", "total_words": 4},
            )
        )
        assert dashboard._transcripts == segments
        assert dashboard._language == "en"
        assert dashboard._total_words == 4

    def test_transcript_event_keeps_prior_language_when_absent(self, dashboard):
        dashboard._language = "en"
        dashboard._handle_event(
            DashboardEvent(type="transcript", payload={"segments": []})
        )
        assert dashboard._language == "en"

    def test_transcript_buffer_is_capped(self, dashboard):
        dashboard._max_transcripts = 5
        dashboard._transcripts = [_segment(f"old {i}") for i in range(5)]
        dashboard._handle_event(
            DashboardEvent(
                type="transcript",
                payload={"segments": [_segment("new 1"), _segment("new 2")]},
            )
        )
        assert len(dashboard._transcripts) == 5
        assert dashboard._transcripts[-1].text == "new 2"
        assert dashboard._transcripts[0].text == "old 2"

    def test_level_event_updates_latest_level(self, dashboard):
        dashboard._handle_event(DashboardEvent(type="level", payload={"value": 0.7}))
        assert dashboard._latest_level == pytest.approx(0.7)

    def test_dropped_event_records_dropped_chunks(self, dashboard):
        dashboard._handle_event(
            DashboardEvent(type="dropped", payload={"dropped_chunks": 3})
        )
        assert dashboard._dropped_chunks == 3

    def test_vad_status_event_updates_speech_state(self, dashboard):
        dashboard._handle_event(
            DashboardEvent(
                type="vad_status",
                payload={
                    "frames": 10,
                    "speech_frames": 4,
                    "triggered": True,
                    "chunks_emitted": 2,
                },
            )
        )
        assert dashboard._vad_triggered is True
        assert dashboard._vad_frames == 10
        assert dashboard._vad_speech_frames == 4
        assert dashboard._vad_chunks_emitted == 2

    def test_unknown_event_type_is_ignored(self, dashboard):
        dashboard._handle_event(DashboardEvent(type="nonsense", payload={"x": 1}))
        assert dashboard._transcripts == []


class TestScrolling:
    def _fill(self, dashboard, count: int):
        dashboard._transcripts = [
            _segment(f"line {i}", start=float(i)) for i in range(count)
        ]

    def test_scroll_up_disables_auto_scroll(self, dashboard):
        self._fill(dashboard, 200)
        dashboard._handle_scroll_up()
        assert dashboard._auto_scroll is False
        assert dashboard._scroll_offset == 5

    def test_scroll_up_clamps_to_history_size(self, dashboard):
        self._fill(dashboard, 3)
        dashboard._handle_scroll_up()
        assert dashboard._scroll_offset == 0

    def test_scroll_down_to_bottom_reenables_auto_scroll(self, dashboard):
        self._fill(dashboard, 200)
        dashboard._handle_scroll_up()
        dashboard._handle_scroll_down()
        assert dashboard._scroll_offset == 0
        assert dashboard._auto_scroll is True

    def test_page_up_and_page_down_are_symmetric(self, dashboard):
        self._fill(dashboard, 500)
        dashboard._handle_page_up()
        offset_after_page_up = dashboard._scroll_offset
        assert offset_after_page_up > 0
        assert dashboard._auto_scroll is False
        dashboard._handle_page_down()
        assert dashboard._scroll_offset == 0
        assert dashboard._auto_scroll is True

    def test_scroll_to_bottom_resets_state(self, dashboard):
        self._fill(dashboard, 200)
        dashboard._handle_page_up()
        dashboard._handle_scroll_to_bottom()
        assert dashboard._scroll_offset == 0
        assert dashboard._auto_scroll is True


class TestRendering:
    def test_render_transcript_empty_shows_waiting(self, dashboard):
        panel = dashboard._render_transcript()
        assert "Waiting for speech" in str(panel.renderable)

    def test_render_transcript_scrolled_title_shows_window(self, dashboard):
        dashboard._transcripts = [_segment(f"line {i}") for i in range(100)]
        dashboard._auto_scroll = False
        dashboard._scroll_offset = 10
        panel = dashboard._render_transcript()
        assert "showing" in str(panel.title)

    def test_render_status_includes_dropped_row_only_when_dropping(self, dashboard):
        console = Console(width=60, force_terminal=False)
        with console.capture() as capture:
            console.print(dashboard._render_status())
        assert "Dropped" not in capture.get()

        dashboard._dropped_chunks = 4
        with console.capture() as capture:
            console.print(dashboard._render_status())
        assert "Dropped" in capture.get()

    def test_footer_swaps_instructions_when_scrolled(self, dashboard):
        console = Console(width=80, force_terminal=False)
        with console.capture() as capture:
            console.print(dashboard._render_footer())
        assert "to scroll" in capture.get()

        dashboard._auto_scroll = False
        with console.capture() as capture:
            console.print(dashboard._render_footer())
        assert "resume auto-scroll" in capture.get()


class TestHelpers:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "00:00"),
            (59, "00:59"),
            (61, "01:01"),
            (3599, "59:59"),
            (3661, "01:01:01"),
            (-5, "00:00"),
        ],
    )
    def test_format_elapsed(self, seconds, expected):
        assert LiveDashboard._format_elapsed(seconds) == expected

    def test_sanitize_text_strips_escape_and_unprintable(self):
        assert (
            LiveDashboard._sanitize_text("a\x1b[31mred\x07 b\tc\n") == "a[31mred b\tc\n"
        )

    @pytest.mark.parametrize(
        ("level", "filled"),
        [(0.0, 0), (0.5, 5), (1.0, 10), (2.0, 10)],
    )
    def test_render_level_bar(self, level, filled):
        bar = LiveDashboard._render_level_bar(level)
        assert bar.count("█") == filled
        assert len(bar) == 10

    def test_estimate_visible_lines_floor(self, dashboard):
        dashboard.console = Console(width=80, height=10, force_terminal=False)
        assert dashboard._estimate_visible_lines() == 10


class TestRunLoop:
    def test_run_drains_events_and_stops_on_stop_event(self, monkeypatch):
        events: queue.Queue[DashboardEvent] = queue.Queue()
        stop = threading.Event()
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        dashboard = LiveDashboard(
            console=Console(width=80, height=24, force_terminal=False),
            event_queue=events,
            stop_event=stop,
            start_time=time.monotonic(),
        )
        events.put(
            DashboardEvent(
                type="transcript",
                payload={"segments": [_segment("hi")], "total_words": 1},
            )
        )

        runner = threading.Thread(target=dashboard.run)
        runner.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not dashboard._transcripts:
                time.sleep(0.02)
        finally:
            stop.set()
            runner.join(timeout=5.0)

        assert not runner.is_alive()
        assert dashboard._total_words == 1
