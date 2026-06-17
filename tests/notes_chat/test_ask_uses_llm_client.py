"""Integration test: chirp ask routes generation through the chirpd daemon.

Spawns the real ``chirpd.server.serve`` in a worker thread (its own event loop)
with a ``FakeBackend`` that yields scripted tokens. The retrieval side is
mocked because moving the embed/chroma stack onto the daemon is owned by
EPIC-INTEGRATION-CUTOVER. CliRunner drives ``chirp ask`` against the running
daemon and asserts the scripted tokens reach stdout — the closest automated
proxy for the manual smoke test in story 3.7 AC-10.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chirpd.backend import FakeBackend
from chirpd.dispatcher import Dispatcher
from chirpd.server import serve
from chirpd.state import DaemonState
from llm.registry import Registry, RegistryEntry
from notes_chat.cli import app

runner = CliRunner()


class _DaemonHarness:
    """Run chirpd's serve loop in a background thread with its own event loop."""

    def __init__(self, socket_path: Path, backend: FakeBackend) -> None:
        self.socket_path = socket_path
        self.backend = backend
        registry = Registry(
            schema_version=1,
            default_chat="gemma",
            models={
                "gemma": RegistryEntry(
                    hf_repo="mlx-community/gemma-4-4b-it-4bit", role="chat"
                ),
            },
        )
        self.state = DaemonState(
            backend=backend, registry=registry, idle_timeout_s=60.0
        )
        self.dispatcher = Dispatcher(state=self.state)
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
                # best-effort teardown
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
    tmp = Path(tempfile.mkdtemp(prefix="ask-int-", dir="/tmp"))
    path = tmp / "s"
    yield path
    try:
        if path.exists():
            path.unlink()
    except OSError:
        # best-effort cleanup
        pass
    try:
        tmp.rmdir()
    except OSError:
        # best-effort cleanup
        pass


@pytest.fixture
def daemon(socket_path: Path) -> Iterator[_DaemonHarness]:
    backend = FakeBackend(chat_tokens=["hello", " from", " the", " daemon"])
    harness = _DaemonHarness(socket_path, backend)
    harness.start()
    try:
        yield harness
    finally:
        harness.stop()


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, socket_path: Path) -> None:
    monkeypatch.setenv("CHIRP_DAEMON_SOCKET", str(socket_path))
    monkeypatch.setattr("notes_chat.cli.get_notes_config", object)

    def fake_retrieve(config, question, when_filter=None):
        return {
            "success": True,
            "context": "note body about the budget",
            "sources": ["note #1 (Demo)"],
            "retrieved_ids": ["c1"],
        }

    monkeypatch.setattr("notes_chat.retrieval.retrieve_context", fake_retrieve)
    monkeypatch.setattr("notes_chat.cache.get_cached_answer", lambda *a: None)
    monkeypatch.setattr("notes_chat.cache.cache_answer", lambda *a: None)


def test_ask_no_markdown_streams_tokens_through_daemon(
    daemon: _DaemonHarness,
    socket_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pipeline(monkeypatch, socket_path)
    result = runner.invoke(app, ["ask", "-q", "what?", "--no-markdown"])
    assert result.exit_code == 0, result.output
    assert "hello from the daemon" in result.output
    assert "sources: note #1" in result.output
    assert daemon.backend.last_prompt is not None
    assert "note body about the budget" in daemon.backend.last_prompt


def test_ask_markdown_path_routes_chat_through_daemon(
    daemon: _DaemonHarness,
    socket_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pipeline(monkeypatch, socket_path)
    result = runner.invoke(app, ["ask", "-q", "what?"])
    assert result.exit_code == 0, result.output
    assert "hello from the daemon" in result.output
    assert daemon.backend.last_prompt is not None
