import time
from unittest.mock import Mock, patch

from config.settings import ChirpSettings
from notes_chat.interactive import InteractiveChatSession


def _make_session():
    return InteractiveChatSession(Mock(spec=ChirpSettings), markdown=False)


class _FakeClient:
    def __init__(self, recorder):
        self._recorder = recorder

    def cancel_sync(self, target_id):
        self._recorder.append(target_id)
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
    # NFR-P4: the handler returns control to the prompt within 200 ms.
    assert elapsed < 0.2
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
