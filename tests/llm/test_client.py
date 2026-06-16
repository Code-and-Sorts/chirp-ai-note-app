"""Tests for :mod:`llm.client`."""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from llm import client as client_module
from llm.client import (
    DEFAULT_SOCKET_PATH,
    DEFAULT_SPAWN_TIMEOUT_S,
    VERSION_MISMATCH_RESPAWN_WAIT_S,
    LLMClient,
    resolve_socket_path,
)
from llm.error_codes import (
    MODEL_CANCELLED,
    MODEL_CAPACITY_EXCEEDED,
    MODEL_GENERATION_FAILED,
    MODEL_LOAD_FAILED,
    MODEL_NOT_FOUND,
    PROTOCOL_MALFORMED,
    PROTOCOL_REQUEST_CONFLICT,
    PROTOCOL_VERSION_MISMATCH,
    REQUEST_NOT_FOUND,
)
from llm.exceptions import (
    CODE_TO_EXCEPTION,
    LLMCancelled,
    LLMConnectionLost,
    LLMDaemonSpawnFailed,
    LLMDaemonUnreachable,
    LLMError,
    LLMGenerationFailed,
    LLMInferenceTimeout,
    LLMMalformedResponse,
    LLMModelCapacityExceeded,
    LLMModelLoadFailed,
    LLMModelNotFound,
    LLMProtocolError,
    LLMRequestConflict,
    LLMRequestNotFound,
    LLMVersionMismatch,
)
from llm.protocol import (
    EVENT_DELTA,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_READY,
    EVENT_VERSION_MISMATCH,
    OP_HELLO,
    package_version,
)
from tests.llm.conftest import (
    FakeDaemonFactory,
    make_ready_then_done_handler,
    make_version_mismatch_then_exit_handler,
    read_request,
    write_event,
)

CLIENT_VERSION_MATCH = package_version()
VERSION_MISMATCH_BUDGET_S = VERSION_MISMATCH_RESPAWN_WAIT_S + DEFAULT_SPAWN_TIMEOUT_S


def test_resolve_socket_path_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHIRP_DAEMON_SOCKET", "/tmp/llmc-env.sock")
    assert resolve_socket_path() == Path("/tmp/llmc-env.sock")


def test_resolve_socket_path_config_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHIRP_DAEMON_SOCKET", raising=False)

    class _StubLLM:
        daemon_socket = Path("/tmp/llmc-config.sock")

    class _StubSettings:
        llm = _StubLLM()

    monkeypatch.setattr(client_module, "get_settings", lambda: _StubSettings())
    assert resolve_socket_path() == Path("/tmp/llmc-config.sock")


def test_resolve_socket_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHIRP_DAEMON_SOCKET", raising=False)

    class _StubLLM:
        daemon_socket = None

    class _StubSettings:
        llm = _StubLLM()

    monkeypatch.setattr(client_module, "get_settings", lambda: _StubSettings())
    assert resolve_socket_path() == DEFAULT_SOCKET_PATH


def test_resolve_socket_path_swallows_settings_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHIRP_DAEMON_SOCKET", raising=False)

    def _boom() -> Any:
        raise RuntimeError("broken settings")

    monkeypatch.setattr(client_module, "get_settings", _boom)
    assert resolve_socket_path() == DEFAULT_SOCKET_PATH


async def test_health_against_in_process_daemon(
    in_process_daemon: asyncio.Task[None],
    temp_socket_path: Path,
) -> None:
    llm_client = LLMClient(socket_path=temp_socket_path)
    payload = await llm_client.health()
    assert payload["status"] == "ok"
    assert "version" in payload


def test_health_sync_against_in_process_daemon(
    temp_socket_path: Path,
) -> None:
    async def _drive() -> dict[str, Any]:
        from chirpd.dispatcher import Dispatcher
        from chirpd.server import serve

        dispatcher = Dispatcher()
        task = asyncio.create_task(serve(temp_socket_path, dispatcher))
        deadline = asyncio.get_running_loop().time() + 2.0
        while not temp_socket_path.exists():
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError("socket never appeared")
            await asyncio.sleep(0.02)
        try:
            # health_sync can't run inside this loop; offload to a thread.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, LLMClient(socket_path=temp_socket_path).health_sync
            )
        finally:
            if not task.done():
                task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, TimeoutError):
                pass

    payload = asyncio.run(_drive())
    assert payload["status"] == "ok"


async def test_model_list_spawn_if_absent_false_raises_without_spawning(
    temp_socket_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_popen(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("spawn_if_absent=False must not spawn a daemon")

    monkeypatch.setattr(subprocess, "Popen", _fail_popen)

    llm_client = LLMClient(socket_path=temp_socket_path)
    with pytest.raises(LLMDaemonUnreachable):
        await llm_client.model_list(spawn_if_absent=False)


async def test_lazy_spawn_when_socket_missing(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_calls: list[list[str]] = []
    spawn_envs: list[dict[str, str]] = []

    class _FakeProc:
        def poll(self) -> int | None:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    async def _delayed_start() -> None:
        await asyncio.sleep(0.1)
        await fake_daemon_factory(make_ready_then_done_handler(CLIENT_VERSION_MATCH))

    delayed_task = asyncio.create_task(_delayed_start())

    def _fake_popen(args: list[str], **kwargs: Any) -> _FakeProc:
        spawn_calls.append(args)
        spawn_envs.append(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    llm_client = LLMClient(socket_path=temp_socket_path, spawn_timeout_s=2.0)
    payload = await llm_client.health()
    await delayed_task
    assert payload["status"] == "ok"
    assert spawn_calls
    assert spawn_calls[0][0] == "chirpd"
    # The child chirpd must bind the socket the client is polling — otherwise
    # a custom socket_path would deadlock against the default-bound daemon.
    assert spawn_envs[0].get("CHIRP_DAEMON_SOCKET") == str(temp_socket_path)


async def test_lazy_spawn_timeout_raises_daemon_spawn_failed(
    temp_socket_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IdleProc:
        def __init__(self) -> None:
            self._dead = False
            self.terminate_calls = 0
            self.wait_calls = 0

        def poll(self) -> int | None:
            return None if not self._dead else 0

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            self._dead = True
            return 0

        def terminate(self) -> None:
            self.terminate_calls += 1
            self._dead = True

        def kill(self) -> None:
            self._dead = True

    spawned: list[_IdleProc] = []

    def _fake_popen(args: list[str], **kwargs: Any) -> _IdleProc:
        proc = _IdleProc()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    llm_client = LLMClient(socket_path=temp_socket_path, spawn_timeout_s=0.3)
    with pytest.raises(LLMDaemonSpawnFailed):
        await llm_client.health()
    assert spawned, "expected at least one Popen call"
    assert all(proc.terminate_calls >= 1 for proc in spawned), (
        "_reap_failed_spawn should have terminated the stuck Popen"
    )


async def test_spawn_chirpd_executable_not_found(
    temp_socket_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(args: list[str], **kwargs: Any) -> Any:
        raise FileNotFoundError("chirpd")

    monkeypatch.setattr(subprocess, "Popen", _missing)
    llm_client = LLMClient(socket_path=temp_socket_path, spawn_timeout_s=0.1)
    with pytest.raises(LLMDaemonSpawnFailed):
        await llm_client.health()


async def test_spawn_oserror_raises_daemon_spawn_failed(
    temp_socket_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken(args: list[str], **kwargs: Any) -> Any:
        raise OSError("too many open files")

    monkeypatch.setattr(subprocess, "Popen", _broken)
    llm_client = LLMClient(socket_path=temp_socket_path, spawn_timeout_s=0.1)
    with pytest.raises(LLMDaemonSpawnFailed):
        await llm_client.health()


async def test_version_mismatch_triggers_respawn_and_retry(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_holder: list[Any] = []

    async def _close_first_listener() -> None:
        if first_holder:
            first_holder[0].close_listener()

    first = await fake_daemon_factory(
        make_version_mismatch_then_exit_handler(
            "9.9.9-bogus", on_mismatch=_close_first_listener
        )
    )
    first_holder.append(first)

    async def _respawn() -> None:
        await fake_daemon_factory(make_ready_then_done_handler(CLIENT_VERSION_MATCH))

    respawn_task: asyncio.Task[None] | None = None

    class _FakeProc:
        def poll(self) -> int | None:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    def _fake_popen(args: list[str], **kwargs: Any) -> _FakeProc:
        nonlocal respawn_task
        if respawn_task is None:
            respawn_task = asyncio.create_task(_respawn())
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    llm_client = LLMClient(socket_path=temp_socket_path, spawn_timeout_s=2.0)
    started = time.monotonic()
    payload = await llm_client.health()
    elapsed = time.monotonic() - started

    if respawn_task is not None:
        await respawn_task
    assert payload["status"] == "ok"
    assert elapsed <= VERSION_MISMATCH_BUDGET_S


async def test_version_mismatch_retry_disabled_raises(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    await fake_daemon_factory(make_version_mismatch_then_exit_handler("9.9.9-bogus"))
    llm_client = LLMClient(
        socket_path=temp_socket_path, retry_on_version_mismatch=False
    )
    with pytest.raises(LLMVersionMismatch):
        await llm_client.health()


_ERROR_CODE_CASES: list[tuple[str, type[Exception]]] = [
    (PROTOCOL_VERSION_MISMATCH, LLMVersionMismatch),
    (PROTOCOL_MALFORMED, LLMMalformedResponse),
    (PROTOCOL_REQUEST_CONFLICT, LLMRequestConflict),
    (REQUEST_NOT_FOUND, LLMRequestNotFound),
    (MODEL_NOT_FOUND, LLMModelNotFound),
    (MODEL_LOAD_FAILED, LLMModelLoadFailed),
    (MODEL_GENERATION_FAILED, LLMGenerationFailed),
    (MODEL_CANCELLED, LLMCancelled),
    (MODEL_CAPACITY_EXCEEDED, LLMModelCapacityExceeded),
]


@pytest.mark.parametrize(("code", "expected"), _ERROR_CODE_CASES)
async def test_error_event_maps_to_typed_exception(
    code: str,
    expected: type[Exception],
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        assert hello["op"] == OP_HELLO
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        request = await read_request(reader)
        await write_event(
            writer,
            {
                "id": request.get("id"),
                "event": EVENT_ERROR,
                "code": code,
                "message": "wire error from fake daemon",
                "details": {"extra": "yes"},
            },
        )

    await fake_daemon_factory(_handler)
    llm_client = LLMClient(socket_path=temp_socket_path)
    with pytest.raises(expected) as exc_info:
        await llm_client.health()
    assert exc_info.value.message == "wire error from fake daemon"
    assert exc_info.value.details == {"extra": "yes"}


def test_code_to_exception_table_covered_by_parametrize() -> None:
    covered = {code for code, _ in _ERROR_CODE_CASES}
    assert covered == set(CODE_TO_EXCEPTION.keys())


async def test_unknown_error_code_preserves_daemon_message(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    """AC-8: an unknown code (likely a version skew) keeps the daemon's message."""
    distinctive = "a future daemon refused for a reason this client predates"

    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        request = await read_request(reader)
        await write_event(
            writer,
            {
                "id": request.get("id"),
                "event": EVENT_ERROR,
                "code": "SOME_FUTURE_CODE",
                "message": distinctive,
                "details": {},
            },
        )

    await fake_daemon_factory(_handler)
    llm_client = LLMClient(socket_path=temp_socket_path)
    with pytest.raises(LLMProtocolError) as exc_info:
        await llm_client.health()
    # The daemon's real message must survive — not degrade to "unknown code".
    assert distinctive in str(exc_info.value)
    assert "SOME_FUTURE_CODE" in str(exc_info.value)
    assert not isinstance(exc_info.value, LLMMalformedResponse)


async def test_connection_lost_raises_typed_exception(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        await read_request(reader)
        writer.close()

    await fake_daemon_factory(_handler)
    llm_client = LLMClient(socket_path=temp_socket_path)
    with pytest.raises(LLMConnectionLost):
        await llm_client.health()


# --- AC-1: per-inter-event inference read timeout ---


async def test_inference_timeout_when_daemon_stalls_after_hello(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    """A daemon that completes hello then never emits a delta must not hang."""
    forever = asyncio.Event()  # never set — the daemon stalls

    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        await read_request(reader)
        await forever.wait()  # accept the request, then stall forever

    await fake_daemon_factory(_handler)
    # Pin both budgets small: the first read after a request uses the
    # first-event budget, so both must be tiny for the test to be fast.
    llm_client = LLMClient(
        socket_path=temp_socket_path,
        inference_timeout_s=0.2,
        first_event_timeout_s=0.2,
    )
    start = time.monotonic()
    with pytest.raises(LLMInferenceTimeout) as exc_info:
        await llm_client.health()
    elapsed = time.monotonic() - start
    # Fires within the budget (+ generous slack), not an unbounded hang.
    assert elapsed < 5.0
    assert exc_info.value.details["timeout_seconds"] == 0.2
    assert "chirp daemon" in str(exc_info.value)
    forever.set()


def test_first_event_budget_is_proportional_to_per_event_budget() -> None:
    """M2: the first-event budget scales with the per-event budget (×2 default)."""
    assert client_module._first_event_budget(60.0) == 120.0
    assert client_module._first_event_budget(0.2) == pytest.approx(0.4)
    # The multiplier is derived from the default pair so they stay consistent.
    assert client_module._FIRST_EVENT_BUDGET_MULTIPLIER == 2.0


async def test_first_event_timeout_honors_env_override(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M2: CHIRP_INFERENCE_TIMEOUT bounds the FIRST read too (no fixed 60s floor)."""
    forever = asyncio.Event()

    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        await read_request(reader)
        await forever.wait()  # stall on the FIRST event after the request

    await fake_daemon_factory(_handler)
    # Drive purely via the env override — NO constructor budget args. With the
    # old fixed +60s headroom this first read would block ~60s; with the
    # proportional multiplier it fires at ~0.2 × 2 = 0.4s.
    monkeypatch.setenv("CHIRP_INFERENCE_TIMEOUT", "0.2")
    llm_client = LLMClient(socket_path=temp_socket_path)
    assert llm_client._first_event_timeout_s == pytest.approx(0.4)
    start = time.monotonic()
    with pytest.raises(LLMInferenceTimeout):
        await llm_client.health()
    elapsed = time.monotonic() - start
    # Fires fast (env-honored), nowhere near a 60s fixed floor.
    assert elapsed < 5.0
    forever.set()


async def test_healthy_slow_but_steady_stream_does_not_trip_timeout(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    """Negative test: a delta within the per-event budget renews it — no false trip."""
    tokens = ["a", "b", "c", "d", "e"]

    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        request = await read_request(reader)
        req_id = request.get("id")
        # Each delta lands at 0.05s — well within the 0.3s per-event budget, so a
        # total stream (0.25s) longer than the budget still never trips.
        for token in tokens:
            await asyncio.sleep(0.05)
            await write_event(
                writer, {"id": req_id, "event": EVENT_DELTA, "text": token}
            )
        await write_event(
            writer,
            {"id": req_id, "event": EVENT_DONE, "usage": {"completion_tokens": 5}},
        )

    await fake_daemon_factory(_handler)
    llm_client = LLMClient(socket_path=temp_socket_path, inference_timeout_s=0.3)
    stream = llm_client.chat_stream(
        messages=[{"role": "user", "content": "hi"}], model="gemma"
    )
    received = [t async for t in stream]
    assert received == tokens


async def test_malformed_response_raises_typed_exception(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        await read_request(reader)
        writer.write(b"not json at all\n")
        await writer.drain()

    await fake_daemon_factory(_handler)
    llm_client = LLMClient(socket_path=temp_socket_path)
    with pytest.raises(LLMMalformedResponse):
        await llm_client.health()


async def test_malformed_handshake_event_raises(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await read_request(reader)
        await write_event(writer, {"id": None, "event": "weird-event"})

    await fake_daemon_factory(_handler)
    llm_client = LLMClient(socket_path=temp_socket_path)
    with pytest.raises(LLMMalformedResponse):
        await llm_client.health()


def test_health_sync_invokes_async_health(
    temp_socket_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stub_health(
        self: LLMClient, *, spawn_if_absent: bool = True
    ) -> dict[str, Any]:
        return {"status": "ok", "uptime_seconds": 0.0, "version": "test"}

    monkeypatch.setattr(LLMClient, "health", _stub_health)
    llm_client = LLMClient(socket_path=temp_socket_path)
    payload = llm_client.health_sync()
    assert payload["status"] == "ok"


def test_default_constants_match_spec() -> None:
    assert DEFAULT_SPAWN_TIMEOUT_S == 3.0
    assert client_module.VERSION_MISMATCH_RESPAWN_WAIT_S == 1.0


async def test_chat_stream_yields_tokens(temp_socket_path: Path) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.registry import Registry, RegistryEntry

    registry = Registry(
        schema_version=1,
        default_chat="gemma",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
        },
    )
    backend = FakeBackend(chat_tokens=["hello", " world"])
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not temp_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        llm_client = LLMClient(socket_path=temp_socket_path)
        stream = llm_client.chat_stream(
            messages=[{"role": "user", "content": "hi"}], model="gemma"
        )
        tokens = [t async for t in stream]
        assert tokens == ["hello", " world"]
        assert stream.usage is not None
        assert stream.usage["completion_tokens"] == 2
        assert stream.usage["prompt_tokens"] > 0
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


async def test_chat_non_streaming_returns_concatenated_string(
    temp_socket_path: Path,
) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.registry import Registry, RegistryEntry

    registry = Registry(
        schema_version=1,
        default_chat="gemma",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
        },
    )
    backend = FakeBackend(chat_tokens=["hello", " world"])
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not temp_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        llm_client = LLMClient(socket_path=temp_socket_path)
        text = await llm_client.chat(
            messages=[{"role": "user", "content": "hi"}], model="gemma"
        )
        assert text == "hello world"
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


async def test_embed_batched_round_trip(temp_socket_path: Path) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.registry import Registry, RegistryEntry

    registry = Registry(
        schema_version=1,
        default_embed="nomic",
        models={
            "nomic": RegistryEntry(hf_repo="mlx-community/nomic-embed", role="embed"),
        },
    )
    backend = FakeBackend(embed_dim=3)
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not temp_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        llm_client = LLMClient(socket_path=temp_socket_path)
        vectors = await llm_client.embed(["a", "bb", "ccc"], model="nomic")
        assert len(vectors) == 3
        assert all(len(v) == 3 for v in vectors)
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


async def test_cancel_in_flight_chat_raises_llm_cancelled(
    temp_socket_path: Path,
) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.protocol import new_request_id
    from llm.registry import Registry, RegistryEntry

    registry = Registry(
        schema_version=1,
        default_chat="gemma",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
        },
    )
    backend = FakeBackend(
        chat_tokens=[f"t{i}" for i in range(50)], generation_delay_s=0.03
    )
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not temp_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        chat_client = LLMClient(socket_path=temp_socket_path)
        cancel_client = LLMClient(socket_path=temp_socket_path)
        chat_request_id = new_request_id()

        async def _drain() -> list[str]:
            tokens: list[str] = []
            async for token in chat_client.chat_stream(
                messages=[{"role": "user", "content": "go"}],
                model="gemma",
                request_id=chat_request_id,
            ):
                tokens.append(token)
            return tokens

        chat_task = asyncio.create_task(_drain())
        await asyncio.sleep(0.15)
        await cancel_client.cancel(chat_request_id)
        with pytest.raises(LLMCancelled):
            await chat_task
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


async def test_chat_propagates_model_generation_failed(
    temp_socket_path: Path,
) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.registry import Registry, RegistryEntry

    registry = Registry(
        schema_version=1,
        default_chat="gemma",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
        },
    )
    backend = FakeBackend(stream_raises=RuntimeError("boom"))
    state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not temp_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)
    try:
        llm_client = LLMClient(socket_path=temp_socket_path)
        with pytest.raises(LLMGenerationFailed):
            await llm_client.chat(
                messages=[{"role": "user", "content": "hi"}], model="gemma"
            )
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


def test_chat_stream_sync_returns_generator(temp_socket_path: Path) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.registry import Registry, RegistryEntry

    async def _drive() -> list[str]:
        registry = Registry(
            schema_version=1,
            default_chat="gemma",
            models={
                "gemma": RegistryEntry(
                    hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
                ),
            },
        )
        backend = FakeBackend(chat_tokens=["foo", "bar", "baz"])
        state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)
        dispatcher = Dispatcher(state=state)
        task = asyncio.create_task(serve(temp_socket_path, dispatcher))
        deadline = asyncio.get_running_loop().time() + 2.0
        while not temp_socket_path.exists():
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError("socket never appeared")
            await asyncio.sleep(0.02)
        try:
            loop = asyncio.get_running_loop()

            def _collect() -> list[str]:
                client = LLMClient(socket_path=temp_socket_path)
                gen = client.chat_stream_sync(
                    messages=[{"role": "user", "content": "hi"}], model="gemma"
                )
                # Verify lazy iteration: first call returns a token, doesn't drain.
                first = next(gen)
                rest = list(gen)
                return [first, *rest]

            return await loop.run_in_executor(None, _collect)
        finally:
            if not task.done():
                task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, TimeoutError):
                pass

    tokens = asyncio.run(_drive())
    assert tokens == ["foo", "bar", "baz"]


def test_chat_stream_sync_in_running_loop_raises(temp_socket_path: Path) -> None:
    async def _trigger() -> None:
        client = LLMClient(socket_path=temp_socket_path)
        client.chat_stream_sync(messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(LLMError, match="event loop"):
        asyncio.run(_trigger())


async def test_mid_stream_version_mismatch_triggers_retry(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_holder: list[Any] = []
    first_request_id: list[str] = []
    second_request_id: list[str] = []

    async def _mid_stream_mismatch_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        request = await read_request(reader)
        first_request_id.append(request["id"])
        await write_event(
            writer,
            {
                "id": request.get("id"),
                "event": EVENT_VERSION_MISMATCH,
                "daemon_version": "9.9.9-bogus",
            },
        )
        if first_holder:
            first_holder[0].close_listener()

    first = await fake_daemon_factory(_mid_stream_mismatch_handler)
    first_holder.append(first)

    async def _capture_replay_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        request = await read_request(reader)
        second_request_id.append(request["id"])
        await write_event(
            writer,
            {
                "id": request.get("id"),
                "event": EVENT_READY,
                "status": "ok",
                "uptime_seconds": 1.0,
                "version": CLIENT_VERSION_MATCH,
            },
        )
        await write_event(writer, {"id": request.get("id"), "event": EVENT_DONE})

    async def _respawn() -> None:
        await fake_daemon_factory(_capture_replay_handler)

    respawn_task: asyncio.Task[None] | None = None

    class _FakeProc:
        def poll(self) -> int | None:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    def _fake_popen(args: list[str], **kwargs: Any) -> _FakeProc:
        nonlocal respawn_task
        if respawn_task is None:
            respawn_task = asyncio.create_task(_respawn())
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    llm_client = LLMClient(socket_path=temp_socket_path, spawn_timeout_s=2.0)
    payload = await llm_client.health()
    if respawn_task is not None:
        await respawn_task
    assert payload["status"] == "ok"
    assert first_request_id, "first daemon should have observed the health request"
    assert second_request_id, "second daemon should have observed the health request"
    assert first_request_id[0] == second_request_id[0], (
        "client must replay the original request envelope id verbatim"
    )


async def test_mid_stream_version_mismatch_with_retry_disabled_raises(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        request = await read_request(reader)
        await write_event(
            writer,
            {
                "id": request.get("id"),
                "event": EVENT_VERSION_MISMATCH,
                "daemon_version": "9.9.9-bogus",
            },
        )

    await fake_daemon_factory(_handler)
    llm_client = LLMClient(
        socket_path=temp_socket_path, retry_on_version_mismatch=False
    )
    with pytest.raises(LLMVersionMismatch):
        await llm_client.health()


def test_reap_failed_spawn_already_dead_returns() -> None:
    class _DeadProc:
        def poll(self) -> int | None:
            return 0

    client_module._reap_failed_spawn(_DeadProc())  # type: ignore[arg-type]


def test_reap_failed_spawn_terminate_path() -> None:
    calls: list[str] = []

    class _LiveProc:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            calls.append("terminate")

        def wait(self, timeout: float | None = None) -> int:
            calls.append("wait")
            return 0

        def kill(self) -> None:
            calls.append("kill")

    client_module._reap_failed_spawn(_LiveProc())  # type: ignore[arg-type]
    assert calls == ["terminate", "wait"]


def test_reap_failed_spawn_kill_after_timeout() -> None:
    calls: list[str] = []

    class _StuckProc:
        def __init__(self) -> None:
            self._wait_calls = 0

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            calls.append("terminate")

        def wait(self, timeout: float | None = None) -> int:
            self._wait_calls += 1
            if self._wait_calls == 1:
                calls.append("wait1")
                raise subprocess.TimeoutExpired("chirpd", timeout or 1.0)
            calls.append("wait2")
            return 0

        def kill(self) -> None:
            calls.append("kill")

    client_module._reap_failed_spawn(_StuckProc())  # type: ignore[arg-type]
    assert calls == ["terminate", "wait1", "kill", "wait2"]


def test_reap_failed_spawn_swallows_os_error() -> None:
    class _BrokenProc:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            raise OSError("no such process")

    client_module._reap_failed_spawn(_BrokenProc())  # type: ignore[arg-type]


async def test_close_writer_handles_already_closed_writer(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    await fake_daemon_factory(make_ready_then_done_handler(CLIENT_VERSION_MATCH))
    _, writer = await asyncio.open_unix_connection(str(temp_socket_path))
    writer.close()
    await writer.wait_closed()
    await client_module._close_writer(writer)


async def test_health_without_ready_raises_malformed(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    async def _handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = await read_request(reader)
        await write_event(
            writer,
            {
                "id": hello.get("id"),
                "event": EVENT_READY,
                "daemon_version": CLIENT_VERSION_MATCH,
            },
        )
        request = await read_request(reader)
        await write_event(writer, {"id": request.get("id"), "event": EVENT_DONE})

    await fake_daemon_factory(_handler)
    llm_client = LLMClient(socket_path=temp_socket_path)
    with pytest.raises(LLMMalformedResponse):
        await llm_client.health()


async def test_request_yields_events_until_done(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    await fake_daemon_factory(make_ready_then_done_handler(CLIENT_VERSION_MATCH))
    llm_client = LLMClient(socket_path=temp_socket_path)
    events: list[dict[str, Any]] = []
    async for event in llm_client._request({"id": "r-aaaaaaaaaaaa", "op": "health"}):
        events.append(event)
    assert [e["event"] for e in events] == ["ready", "done"]


def test_health_sync_in_running_loop_raises(temp_socket_path: Path) -> None:
    async def _trigger() -> None:
        LLMClient(socket_path=temp_socket_path).health_sync()

    with pytest.raises(LLMError, match="event loop"):
        asyncio.run(_trigger())


async def test_wait_for_socket_gone_timeout_surfaces_unreachable(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
) -> None:
    await fake_daemon_factory(make_version_mismatch_then_exit_handler("9.9.9-bogus"))
    llm_client = LLMClient(socket_path=temp_socket_path)
    with pytest.raises(LLMDaemonUnreachable):
        await llm_client._wait_for_socket_gone(0.05)


async def test_no_orphan_popens_across_failing_requests(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-10 leak audit: every failed spawn is reaped (wait was called)."""

    class _IdleProc:
        def __init__(self) -> None:
            self._dead = False
            self.wait_calls = 0
            self.terminate_calls = 0

        def poll(self) -> int | None:
            return None if not self._dead else 0

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            self._dead = True
            return 0

        def terminate(self) -> None:
            self.terminate_calls += 1
            self._dead = True

        def kill(self) -> None:
            self._dead = True

    spawned: list[_IdleProc] = []

    def _fake_popen(args: list[str], **kwargs: Any) -> _IdleProc:
        proc = _IdleProc()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    # 5 failing health requests; socket never exists so each spawn must time out
    # and the client must reap each Popen.
    failing_client = LLMClient(socket_path=temp_socket_path, spawn_timeout_s=0.1)
    for _ in range(5):
        with pytest.raises(LLMDaemonSpawnFailed):
            await failing_client.health()

    assert len(spawned) == 5, "expected one Popen per failing request"
    assert all(proc.wait_calls >= 1 for proc in spawned), (
        "every Popen should be wait()-reaped to avoid orphan processes"
    )
    assert all(proc.terminate_calls >= 1 for proc in spawned), (
        "every Popen should be terminate()-d on spawn failure"
    )


async def test_cancellation_during_hello_closes_writer(
    temp_socket_path: Path,
    fake_daemon_factory: FakeDaemonFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nit-3: BaseException path in _connect_with_handshake closes the writer."""
    seen_close = asyncio.Event()

    async def _slow_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await read_request(reader)
        try:
            await asyncio.sleep(5.0)
        finally:
            seen_close.set()

    await fake_daemon_factory(_slow_handler)

    llm_client = LLMClient(socket_path=temp_socket_path)

    async def _slow_hello(self: LLMClient, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise asyncio.CancelledError()

    monkeypatch.setattr(LLMClient, "_do_hello", _slow_hello)

    with pytest.raises(asyncio.CancelledError):
        await llm_client.health()


async def test_model_list_against_in_process_daemon(
    temp_socket_path: Path,
) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.registry import Registry, RegistryEntry

    registry = Registry(
        schema_version=1,
        default_chat="gemma",
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
            "nomic": RegistryEntry(hf_repo="mlx-community/nomic-embed", role="embed"),
        },
    )
    state = DaemonState(backend=FakeBackend(), registry=registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not temp_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)

    try:
        llm_client = LLMClient(socket_path=temp_socket_path)
        models = await llm_client.model_list()
        aliases = {m["alias"] for m in models}
        assert {"gemma", "nomic"}.issubset(aliases)
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


async def test_model_load_unload_round_trip(temp_socket_path: Path) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.registry import Registry, RegistryEntry

    registry = Registry(
        schema_version=1,
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
        },
    )
    state = DaemonState(backend=FakeBackend(), registry=registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not temp_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)

    try:
        load_client = LLMClient(socket_path=temp_socket_path)
        ready = await load_client.model_load("gemma", "chat")
        assert ready["model"] == "gemma"
        assert state.get("gemma") is not None

        unload_client = LLMClient(socket_path=temp_socket_path)
        await unload_client.model_unload("gemma")
        assert state.get("gemma") is None
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


async def test_model_status_returns_rich_dict(temp_socket_path: Path) -> None:
    from chirpd.backend import FakeBackend
    from chirpd.dispatcher import Dispatcher
    from chirpd.server import serve
    from chirpd.state import DaemonState
    from llm.registry import Registry, RegistryEntry

    registry = Registry(
        schema_version=1,
        models={
            "gemma": RegistryEntry(
                hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
            ),
        },
    )
    state = DaemonState(backend=FakeBackend(), registry=registry, idle_timeout_s=60.0)
    dispatcher = Dispatcher(state=state)
    task = asyncio.create_task(serve(temp_socket_path, dispatcher))
    deadline = asyncio.get_running_loop().time() + 2.0
    while not temp_socket_path.exists():
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("socket never appeared")
        await asyncio.sleep(0.02)

    try:
        load_client = LLMClient(socket_path=temp_socket_path)
        await load_client.model_load("gemma", "chat")

        status_client = LLMClient(socket_path=temp_socket_path)
        payload = await status_client.model_status()
        assert "pid" in payload
        assert "uptime_seconds" in payload
        assert "daemon_version" in payload
        assert "rss_bytes" in payload
        assert isinstance(payload["models"], list)
    finally:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            pass


def test_llm_config_daemon_socket_roundtrip_none(tmp_path: Path) -> None:
    from config.settings import ChirpSettings, LLMSettings

    settings = ChirpSettings(llm=LLMSettings(daemon_socket=None))
    cfg_path = tmp_path / "config.toml"
    settings.save_to_file(cfg_path)
    loaded = ChirpSettings.load_from_file(cfg_path)
    assert loaded.llm.daemon_socket is None


def test_llm_config_daemon_socket_roundtrip_path(tmp_path: Path) -> None:
    from config.settings import ChirpSettings, LLMSettings

    sock_path = tmp_path / "chirpd.sock"
    settings = ChirpSettings(llm=LLMSettings(daemon_socket=sock_path))
    cfg_path = tmp_path / "config.toml"
    settings.save_to_file(cfg_path)
    loaded = ChirpSettings.load_from_file(cfg_path)
    assert loaded.llm.daemon_socket == sock_path
