"""End-to-end socket tests for the chat / cancel dispatcher ops."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio

from chirpd.backend import FakeBackend
from chirpd.dispatcher import Dispatcher
from chirpd.server import serve
from chirpd.state import DaemonState
from llm import error_codes
from llm.protocol import (
    EVENT_DELTA,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_LOADING,
    OP_CANCEL,
    OP_CHAT,
    OP_HELLO,
    PROTOCOL_VERSION,
    new_request_id,
    package_version,
)
from llm.registry import Registry, RegistryEntry


@pytest.fixture
def chat_socket_path() -> Iterator[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="cdchat-", dir="/tmp"))
    path = tmp_dir / "s"
    yield path
    try:
        if path.exists():
            path.unlink()
    except OSError:
        # Best-effort cleanup: the temp file may already be removed.
        pass
    try:
        tmp_dir.rmdir()
    except OSError:
        # Best-effort cleanup: the temp directory may be missing or non-empty.
        pass


@pytest.fixture
def chat_registry() -> Registry:
    return Registry(
        schema_version=1,
        default_chat="gemma",
        default_embed="nomic",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
            "phi": RegistryEntry(hf_repo="mlx-community/phi-4-mini-4bit", role="chat"),
            "nomic": RegistryEntry(hf_repo="mlx-community/nomic-embed", role="embed"),
        },
    )


@pytest_asyncio.fixture
async def chat_server(
    chat_socket_path: Path,
    chat_registry: Registry,
) -> AsyncIterator[tuple[FakeBackend, DaemonState, Dispatcher]]:
    backend = FakeBackend(chat_tokens=["hello", " ", "world"])
    state = DaemonState(backend=backend, registry=chat_registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(chat_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not chat_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        yield backend, state, dispatcher
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            # Best-effort teardown: the task is already being cancelled.
            pass


async def _do_hello(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> dict:
    envelope = {
        "id": new_request_id(),
        "op": OP_HELLO,
        "client_version": package_version(),
        "protocol_version": PROTOCOL_VERSION,
    }
    writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    parsed: dict = json.loads(line.decode("utf-8"))
    return parsed


async def _send(writer: asyncio.StreamWriter, envelope: dict) -> None:
    writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
    await writer.drain()


async def _read_events_until_done(
    reader: asyncio.StreamReader, timeout: float = 5.0
) -> list[dict]:
    events: list[dict] = []
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            break
        event = json.loads(line.decode("utf-8"))
        events.append(event)
        if event.get("event") in (EVENT_DONE, EVENT_ERROR):
            break
    return events


async def _connect(
    socket_path: Path,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    hello = await _do_hello(reader, writer)
    assert hello["event"] == "ready"
    return reader, writer


async def _close(writer: asyncio.StreamWriter) -> None:
    if not writer.is_closing():
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError, OSError):
            # Best-effort teardown: the writer/peer may already be gone.
            pass


def _chat_envelope(
    request_id: str | None = None,
    model: str = "gemma",
    messages: list[dict] | None = None,
    options: dict | None = None,
    keep_alive: int | None = None,
) -> dict:
    return {
        "id": request_id or new_request_id(),
        "op": OP_CHAT,
        "model": model,
        "messages": messages or [{"role": "user", "content": "hi"}],
        "options": options or {},
        "keep_alive": keep_alive,
    }


async def test_chat_streams_deltas_then_done(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = ["a", "b", "c"]
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    deltas = [e for e in events if e["event"] == EVENT_DELTA]
    assert [e["text"] for e in deltas] == ["a", "b", "c"]
    assert events[-1]["event"] == EVENT_DONE
    usage = events[-1]["usage"]
    assert usage["completion_tokens"] == 3
    assert usage["prompt_tokens"] > 0
    assert isinstance(usage["ms"], int)


async def test_chat_applies_chat_template_before_generation(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(
            writer,
            _chat_envelope(
                messages=[
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "ping"},
                ]
            ),
        )
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert backend.last_prompt is not None
    assert "system" in backend.last_prompt
    assert "be brief" in backend.last_prompt
    assert "ping" in backend.last_prompt
    assert backend.last_prompt.endswith("<assistant>")


async def test_chat_lazy_loads_model_on_first_request(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    _, state, _ = chat_server
    assert state.get("gemma") is None
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert state.get("gemma") is not None


async def test_chat_emits_loading_when_load_triggered(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[0]["event"] == EVENT_LOADING
    assert events[0]["model"] == "gemma"


async def test_chat_does_not_emit_loading_when_model_already_resident(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    _, state, _ = chat_server
    await state.load("gemma", "chat")
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    event_names = [e["event"] for e in events]
    assert EVENT_LOADING not in event_names


async def test_chat_serializes_concurrent_requests_against_same_model(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = ["x", "y", "z"]
    backend.generation_delay_s = 0.02

    async def _drive() -> list[str]:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(writer, _chat_envelope())
            order: list[str] = []
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not line:
                    break
                event = json.loads(line.decode("utf-8"))
                if event["event"] == EVENT_DELTA:
                    order.append(event["text"])
                if event["event"] in (EVENT_DONE, EVENT_ERROR):
                    break
            return order
        finally:
            await _close(writer)

    a, b = await asyncio.gather(_drive(), _drive())
    assert a == ["x", "y", "z"]
    assert b == ["x", "y", "z"]


async def test_chat_parallel_with_embed_against_different_models(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = ["one", "two", "three"]
    backend.generation_delay_s = 0.02

    async def _do_chat() -> list[dict]:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(writer, _chat_envelope())
            return await _read_events_until_done(reader)
        finally:
            await _close(writer)

    async def _do_embed() -> list[dict]:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(
                writer,
                {
                    "id": new_request_id(),
                    "op": "embed",
                    "model": "nomic",
                    "inputs": ["hello"],
                },
            )
            return await _read_events_until_done(reader)
        finally:
            await _close(writer)

    chat_events, embed_events = await asyncio.gather(_do_chat(), _do_embed())
    assert chat_events[-1]["event"] == EVENT_DONE
    assert embed_events[-1]["event"] == EVENT_DONE
    assert "vectors" in embed_events[-1]


async def test_cancel_halts_generation_within_200ms(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = [f"t{i}" for i in range(100)]
    backend.generation_delay_s = 0.05

    chat_request_id = new_request_id()

    async def _drive_chat() -> tuple[list[dict], float]:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(
                writer,
                _chat_envelope(request_id=chat_request_id),
            )
            events: list[dict] = []
            cancel_observed_at = 0.0
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not line:
                    break
                event = json.loads(line.decode("utf-8"))
                events.append(event)
                if event["event"] == EVENT_ERROR or event["event"] == EVENT_DONE:
                    cancel_observed_at = time.monotonic()
                    break
            return events, cancel_observed_at
        finally:
            await _close(writer)

    async def _drive_cancel() -> float:
        await asyncio.sleep(0.2)
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(
                writer,
                {
                    "id": new_request_id(),
                    "op": OP_CANCEL,
                    "target_id": chat_request_id,
                },
            )
            sent_at = time.monotonic()
            await _read_events_until_done(reader)
            return sent_at
        finally:
            await _close(writer)

    (events, cancel_observed_at), cancel_sent_at = await asyncio.gather(
        _drive_chat(), _drive_cancel()
    )
    elapsed_ms = (cancel_observed_at - cancel_sent_at) * 1000.0
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.MODEL_CANCELLED
    assert elapsed_ms <= 250.0, f"cancel took {elapsed_ms:.1f}ms"


async def test_cancel_unknown_target_emits_error(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(
            writer,
            {
                "id": new_request_id(),
                "op": OP_CANCEL,
                "target_id": "r-ffffffffffff",
            },
        )
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.REQUEST_NOT_FOUND


async def test_chat_idle_unload_scheduled_after_completion(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    _, state, _ = chat_server
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    loaded = state.get("gemma")
    assert loaded is not None
    assert loaded.idle_unload_task is not None
    assert not loaded.idle_unload_task.done()


async def test_chat_with_keep_alive_minus_one_pins_model(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    _, state, _ = chat_server
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope(keep_alive=-1))
        await _read_events_until_done(reader)
    finally:
        await _close(writer)
    loaded = state.get("gemma")
    assert loaded is not None
    assert loaded.idle_unload_task is None


async def test_chat_propagates_generation_failure(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, _, _ = chat_server
    backend.chat_tokens = ["one", "two", "three", "four"]
    backend.stream_raises = RuntimeError("mlx blew up")
    backend.stream_raises_after = 2
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(writer, _chat_envelope())
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    deltas = [e for e in events if e["event"] == EVENT_DELTA]
    assert [e["text"] for e in deltas] == ["one", "two"]
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.MODEL_GENERATION_FAILED


async def test_connection_close_during_chat_triggers_cancel(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, state, _ = chat_server
    backend.chat_tokens = [f"t{i}" for i in range(50)]
    backend.generation_delay_s = 0.02

    request_id = new_request_id()
    reader, writer = await _connect(chat_socket_path)
    await _send(writer, _chat_envelope(request_id=request_id))
    delta_count = 0
    while delta_count < 2:
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        event = json.loads(line)
        if event["event"] == EVENT_DELTA:
            delta_count += 1
    await _close(writer)

    deadline = asyncio.get_running_loop().time() + 2.0
    while asyncio.get_running_loop().time() < deadline:
        if state.get_cancellation(request_id) is None:
            break
        await asyncio.sleep(0.02)
    assert state.get_cancellation(request_id) is None


async def test_chat_missing_messages_field_emits_protocol_malformed(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(
            writer,
            {
                "id": new_request_id(),
                "op": OP_CHAT,
                "model": "gemma",
            },
        )
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.PROTOCOL_MALFORMED


async def test_cancel_missing_target_id_emits_protocol_malformed(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    reader, writer = await _connect(chat_socket_path)
    try:
        await _send(
            writer,
            {"id": new_request_id(), "op": OP_CANCEL},
        )
        events = await _read_events_until_done(reader)
    finally:
        await _close(writer)
    assert events[-1]["event"] == EVENT_ERROR
    assert events[-1]["code"] == error_codes.PROTOCOL_MALFORMED


# --- AC-4: duplicate in-flight request id is rejected, never cross-cancels ---


async def test_duplicate_in_flight_request_id_is_rejected(
    chat_server: tuple[FakeBackend, DaemonState, Dispatcher],
    chat_socket_path: Path,
) -> None:
    backend, state, _ = chat_server
    backend.chat_tokens = [f"t{i}" for i in range(100)]
    backend.generation_delay_s = 0.05

    shared_id = "r-aaaaaaaaaaaa"

    # Connection A: register a slow in-flight chat under shared_id.
    reader_a, writer_a = await _connect(chat_socket_path)
    await _send(writer_a, _chat_envelope(request_id=shared_id))
    # Wait until A's request is actually in flight (cancellation registered).
    deadline = asyncio.get_running_loop().time() + 2.0
    while state.get_cancellation(shared_id) is None:
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("first request never registered")
        await asyncio.sleep(0.01)
    original_event = state.get_cancellation(shared_id)

    # Connection B: a second chat with the SAME id is rejected with a conflict.
    reader_b, writer_b = await _connect(chat_socket_path)
    try:
        await _send(writer_b, _chat_envelope(request_id=shared_id))
        b_events = await _read_events_until_done(reader_b)
    finally:
        await _close(writer_b)
    assert b_events[-1]["event"] == EVENT_ERROR
    assert b_events[-1]["code"] == error_codes.PROTOCOL_REQUEST_CONFLICT

    # The duplicate's rejection (and its finally) must NOT evict A's live event.
    assert state.get_cancellation(shared_id) is original_event

    # A cancel for shared_id still cancels the ORIGINAL request, not the dup.
    reader_c, writer_c = await _connect(chat_socket_path)
    try:
        await _send(
            writer_c,
            {"id": new_request_id(), "op": OP_CANCEL, "target_id": shared_id},
        )
        await _read_events_until_done(reader_c)
    finally:
        await _close(writer_c)

    a_events = await _read_events_until_done(reader_a)
    await _close(writer_a)
    assert a_events[-1]["event"] == EVENT_ERROR
    assert a_events[-1]["code"] == error_codes.MODEL_CANCELLED


# --- AC-7: a cancel after a fully-streamed answer is graceful, not an error ---


class _LateCancelBackend(FakeBackend):
    """Reproduces AC-7 finding #5: a cancel races in as the LAST token streams.

    After the final token, it sets ``should_stop`` itself (simulating a cancel
    that lands during the final delta write) and then runs the generator to
    natural exhaustion, marking ``usage_out["completed"]``. So at loop exit
    ``should_stop`` IS set even though the full answer streamed — the exact
    ambiguity the fix resolves: a naive ``should_stop.is_set()`` check would emit
    a spurious MODEL_CANCELLED tail, while the ``completed`` signal keeps it
    graceful (done). The cancel is real, so the test is not a tautology.
    """

    def __init__(self, tokens: list[str]) -> None:
        super().__init__(chat_tokens=tokens)
        self.cancel_fired = False

    async def stream_generate(self, handle, messages, options, should_stop, usage_out):
        self.last_messages = list(messages)
        self.last_options = dict(options)
        self.last_prompt = "<assistant>"
        usage_out["prompt_tokens"] = 1
        for index, token in enumerate(self.chat_tokens):
            yield token
            if index == len(self.chat_tokens) - 1:
                # The cancel lands AFTER the last token was delivered but BEFORE
                # the loop exits — should_stop is set at classification time.
                self.cancel_fired = True
                should_stop.set()
        # Natural exhaustion: the full answer streamed despite the late cancel.
        usage_out["completed"] = 1


async def test_late_cancel_after_full_stream_is_graceful(
    chat_socket_path: Path,
    chat_registry: Registry,
) -> None:
    backend = _LateCancelBackend(["all", " ", "done"])
    state = DaemonState(backend=backend, registry=chat_registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(chat_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not chat_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)

    request_id = new_request_id()
    try:
        reader, writer = await _connect(chat_socket_path)
        try:
            await _send(writer, _chat_envelope(request_id=request_id))
            events = await _read_events_until_done(reader)
        finally:
            await _close(writer)
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            # Best-effort teardown: the task is already being cancelled.
            pass

    # The late cancel really fired (the test isn't a tautology)...
    assert backend.cancel_fired
    # ...yet the fully-streamed answer ends in done, NOT a MODEL_CANCELLED tail.
    deltas = [e["text"] for e in events if e["event"] == EVENT_DELTA]
    assert deltas == ["all", " ", "done"]
    assert events[-1]["event"] == EVENT_DONE
    assert not any(e["event"] == EVENT_ERROR for e in events)


async def test_idle_unload_race_keeps_model_resident_after_long_generation() -> None:
    """AC-6 end-to-end: a generation longer than idle_timeout_s leaves the model."""
    socket_dir = Path(tempfile.mkdtemp(prefix="cd6-", dir="/tmp"))
    socket_path = socket_dir / "s"
    registry = Registry(
        schema_version=1,
        default_chat="gemma",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            )
        },
    )
    backend = FakeBackend(chat_tokens=["a", "b", "c"], generation_delay_s=0.05)
    # Idle timer (0.1s) is shorter than the generation (3 × 0.05 = 0.15s).
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=0.1)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(socket_path, dispatcher))
    try:
        deadline = asyncio.get_running_loop().time() + 2.0
        while not socket_path.exists():
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError("socket never appeared")
            await asyncio.sleep(0.02)

        reader, writer = await _connect(socket_path)
        try:
            await _send(writer, _chat_envelope())
            events = await _read_events_until_done(reader)
        finally:
            await _close(writer)
        assert events[-1]["event"] == EVENT_DONE

        # The model survived a generation that outlived the idle timer.
        loaded = state.get("gemma")
        assert loaded is not None, "long generation must not evict its own model"

        # A follow-up request does not re-trigger a loading event (still resident).
        reader2, writer2 = await _connect(socket_path)
        try:
            await _send(writer2, _chat_envelope())
            events2 = await _read_events_until_done(reader2)
        finally:
            await _close(writer2)
        assert EVENT_LOADING not in [e["event"] for e in events2]
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            # Best-effort teardown: the task is already being cancelled.
            pass
        try:
            if socket_path.exists():
                socket_path.unlink()
            socket_dir.rmdir()
        except OSError:
            # Best-effort cleanup: the temp directory may be missing or non-empty.
            pass
