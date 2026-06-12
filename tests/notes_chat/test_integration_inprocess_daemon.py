"""End-to-end guardrail over the real wire path (story 6.5, AC-7).

Architecture §Testing Patterns: integration tests start the daemon in-process
against a temp socket and connect with ``LLMClient(socket_path=...)``. This is
the one designated test exercising client → socket → dispatcher →
``FakeBackend`` → streamed ``str`` tokens back; the daemon's own protocol
tests live under ``tests/chirpd/`` (EPIC-CHIRPD-CORE).
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from chirpd.backend import FakeBackend
from chirpd.dispatcher import Dispatcher
from chirpd.server import serve
from chirpd.state import DaemonState
from llm.client import LLMClient
from llm.registry import Registry, RegistryEntry


class _DaemonHarness:
    """Run chirpd's serve loop in a background thread with its own event loop."""

    def __init__(self, socket_path: Path, backend: FakeBackend) -> None:
        self.socket_path = socket_path
        registry = Registry(
            schema_version=1,
            default_chat="gemma",
            models={
                "gemma": RegistryEntry(
                    hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
                ),
            },
        )
        state = DaemonState(backend=backend, registry=registry, idle_timeout_s=60.0)
        self.dispatcher = Dispatcher(state=state)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._serve_task = loop.create_task(
                serve(self.socket_path, self.dispatcher)
            )
            self._ready.set()
            try:
                loop.run_until_complete(self._serve_task)
            except asyncio.CancelledError:
                # Expected teardown path: stop() cancels the serve task.
                pass
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while not self.socket_path.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("daemon socket never appeared")
            time.sleep(0.02)

    def stop(self) -> None:
        if self._loop is None or self._serve_task is None:
            return
        loop = self._loop
        task = self._serve_task

        def _cancel() -> None:
            if not task.done():
                task.cancel()

        loop.call_soon_threadsafe(_cancel)
        if self._thread is not None:
            self._thread.join(timeout=2.0)


@pytest.fixture
def socket_path() -> Iterator[Path]:
    # dir="/tmp" keeps the socket path under macOS's 104-char sun_path limit
    # (pytest's tmp_path can exceed it); TemporaryDirectory owns the cleanup.
    with tempfile.TemporaryDirectory(prefix="inproc-", dir="/tmp") as tmp:
        yield Path(tmp) / "s"


@pytest.mark.integration
def test_chat_stream_sync_streams_tokens_through_inprocess_daemon(socket_path: Path):
    harness = _DaemonHarness(socket_path, FakeBackend(chat_tokens=["hello", " world"]))
    harness.start()
    try:
        client = LLMClient(socket_path=socket_path)
        tokens = list(
            client.chat_stream_sync(
                [{"role": "user", "content": "greet me"}], model="default"
            )
        )
    finally:
        harness.stop()

    assert all(isinstance(token, str) for token in tokens)
    assert "".join(tokens) == "hello world"
