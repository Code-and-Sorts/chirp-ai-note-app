import time
from unittest.mock import Mock, patch

from config.settings import ChirpSettings
from llm.exceptions import LLMDaemonUnreachable
from notes_chat.interactive import InteractiveChatSession


def _make_session():
    return InteractiveChatSession(Mock(spec=ChirpSettings), markdown=False)


class _FakeClient:
    def __init__(self, recorder, raises=None):
        self._recorder = recorder
        self._raises = raises

    def cancel_sync(self, target_id):
        self._recorder.append(target_id)
        if self._raises is not None:
            raise self._raises
        return {"event": "done"}


def test_handle_question_cancels_inflight_on_keyboard_interrupt():
    """A mid-stream Ctrl-C cancels the in-flight request and returns fast."""
    req_id = "r-0123456789ab"

    def fake_stream(config, question):
        yield {"type": "request_started", "req_id": req_id}
        yield {"type": "token", "content": "partial answer "}
        raise KeyboardInterrupt

    cancel_calls: list[str] = []
    session = _make_session()

    with (
        patch("notes_chat.interactive.enhanced_search_and_answer_stream", fake_stream),
        patch(
            "notes_chat.interactive.LLMClient",
            lambda *a, **k: _FakeClient(cancel_calls),
        ),
    ):
        start = time.perf_counter()
        session.handle_question("tell me everything in detail")
        elapsed = time.perf_counter() - start

    assert cancel_calls == [req_id]
    # This is a hang guard, not a precise latency check: with a fake cancel_sync
    # the handler returns in microseconds, so a strict wall-clock bound would
    # only flake under CI load. The real NFR-P4 ≤200 ms is verified by the live
    # smoke and owned daemon-side; here we only assert the handler returns
    # promptly rather than blocking. (2 s also stays under the 3 s spawn timeout,
    # so a regression to a spawning cancel would still trip the daemon-down test.)
    assert elapsed < 2.0
    assert session._inflight_req_id is None


def test_handle_question_cancel_when_daemon_unreachable_returns_cleanly():
    """A Ctrl-C when the daemon is unreachable still returns to the prompt.

    cancel_sync raising LLMDaemonUnreachable (what a down daemon yields, fast,
    because cancel never spawns) must be swallowed by the handler — not crash
    the session.
    """
    req_id = "r-0123456789ab"

    def fake_stream(config, question):
        yield {"type": "request_started", "req_id": req_id}
        yield {"type": "token", "content": "partial "}
        raise KeyboardInterrupt

    cancel_calls: list[str] = []
    session = _make_session()

    with (
        patch("notes_chat.interactive.enhanced_search_and_answer_stream", fake_stream),
        patch(
            "notes_chat.interactive.LLMClient",
            lambda *a, **k: _FakeClient(
                cancel_calls, raises=LLMDaemonUnreachable("daemon down")
            ),
        ),
    ):
        start = time.perf_counter()
        session.handle_question("tell me everything")  # must not raise
        elapsed = time.perf_counter() - start

    assert cancel_calls == [req_id]
    # Hang guard (see note above). Crucially 2 s is below the 3 s daemon spawn
    # timeout, so a regression where cancel() spawns instead of failing fast
    # (NFR-P4) would push this over the bound and fail.
    assert elapsed < 2.0
    assert session._inflight_req_id is None


def test_handle_question_does_not_cancel_on_normal_completion():
    """A stream that completes normally never issues a cancel."""

    def fake_stream(config, question):
        yield {"type": "request_started", "req_id": "r-aaaaaaaaaaaa"}
        yield {"type": "token", "content": "hello"}
        yield {"type": "complete", "answer": "hello"}

    cancel_calls: list[str] = []
    session = _make_session()

    with (
        patch("notes_chat.interactive.enhanced_search_and_answer_stream", fake_stream),
        patch(
            "notes_chat.interactive.LLMClient",
            lambda *a, **k: _FakeClient(cancel_calls),
        ),
    ):
        session.handle_question("hi")

    assert cancel_calls == []
    assert session._inflight_req_id is None
