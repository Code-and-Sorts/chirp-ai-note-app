"""Shared test fixtures and configuration for the chirp test suite."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from llm.exceptions import LLMError


@pytest.fixture(autouse=True)
def _isolated_user_templates_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the note-template user dir at a per-test temp dir.

    Without this, any test constructing a default TemplateLoader() (directly
    or via NoteGenerator/_validated_template_or_exit) reads the developer's
    real ~/.chirp/templates/, so local template edits could flip test results.
    """
    path = tmp_path_factory.mktemp("user-templates")
    monkeypatch.setattr("notes.note_templates.user_templates_dir", lambda: path)


@pytest.fixture(autouse=True)
def _force_darwin_platform(request: pytest.FixtureRequest) -> Iterator[None]:
    """Patch sys.platform and platform.mac_ver so macOS version checks pass.

    Tests decorated with @pytest.mark.real_platform opt out and receive
    the host's real platform values.
    """
    if "real_platform" in request.keywords:
        yield
        return
    with (
        mock.patch.object(sys, "platform", "darwin"),
        mock.patch(
            "audio_capture.platform.mac_ver",
            return_value=("13.0.0", ("", "", ""), ""),
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Shared fakes for the `llm.client.LLMClient` surface (story 6.5).
#
# Production code injects a client through `NoteGenerator(..., llm_client=)`,
# `_get_query_embedding(..., client=)`, etc., or constructs `LLMClient()`
# lazily (patch the `LLMClient` symbol imported into the module under test).
# These factories model the real call shapes: `chat_stream_sync` yields plain
# `str` tokens, `chat_sync` returns a `str`, `embed_sync` returns one vector
# per input. Each returned callable records its calls on a `.calls` list so
# tests can assert the exact call shape.
# ---------------------------------------------------------------------------


class _FakeChatStreamSync:
    """``LLMClient.chat_stream_sync`` stand-in.

    Yields each token (a plain ``str``) and, like the real client, returns its
    iterator immediately — ``error`` (if given) is raised *during* iteration,
    after all scripted tokens, mirroring a daemon that dies mid-stream. Script
    ``tokens=[]`` with an ``error`` for a stream that fails before producing
    anything.
    """

    def __init__(self, tokens: list[str], error: Exception | None = None) -> None:
        self._tokens = tokens
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]],
        model: str = "default",
        options: dict[str, Any] | None = None,
        keep_alive: int | None = None,
        request_id: str | None = None,
    ) -> Iterator[str]:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "options": options,
                "keep_alive": keep_alive,
                "request_id": request_id,
            }
        )

        def _stream() -> Iterator[str]:
            yield from self._tokens
            if self._error is not None:
                raise self._error

        return _stream()


class _FakeChatSync:
    """``LLMClient.chat_sync`` stand-in returning scripted text."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]],
        model: str = "default",
        options: dict[str, Any] | None = None,
        keep_alive: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "options": options,
                "keep_alive": keep_alive,
            }
        )
        return self._text


class _FakeEmbedSync:
    """``LLMClient.embed_sync`` stand-in returning scripted vectors.

    Asserts one scripted vector per input (`embed_sync`'s contract from story
    6.3) so a fixture cannot silently script a different batch size than
    production actually sends.
    """

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[dict[str, Any]] = []

    def __call__(self, inputs: list[str], model: str = "default") -> list[list[float]]:
        self.calls.append({"inputs": list(inputs), "model": model})
        assert len(self._vectors) == len(inputs), (
            f"fixture scripted {len(self._vectors)} vectors but production sent "
            f"{len(inputs)} inputs"
        )
        return self._vectors


class _RaiseLLMError:
    """Client-method stand-in raising a typed ``LLMError`` when invoked."""

    def __init__(self, error_cls: type[LLMError], message: str = "test error") -> None:
        self._error_cls = error_cls
        self._message = message
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"args": args, "kwargs": kwargs})
        raise self._error_cls(self._message)


@pytest.fixture
def fake_chat_tokens() -> Callable[..., _FakeChatStreamSync]:
    """Factory fixture for ``chat_stream_sync`` stand-ins (see _FakeChatStreamSync)."""
    return _FakeChatStreamSync


@pytest.fixture
def fake_chat_text() -> Callable[[str], _FakeChatSync]:
    """Factory fixture for ``chat_sync`` stand-ins returning the given text."""
    return _FakeChatSync


@pytest.fixture
def fake_embed() -> Callable[[list[list[float]]], _FakeEmbedSync]:
    """Factory fixture for ``embed_sync`` stand-ins returning the given vectors."""
    return _FakeEmbedSync


@pytest.fixture
def raise_llm_error() -> Callable[..., _RaiseLLMError]:
    """Factory fixture for client methods that raise a typed ``LLMError``."""
    return _RaiseLLMError


@pytest.fixture
def fake_llm_client() -> Callable[..., SimpleNamespace]:
    """Assemble a fake ``LLMClient`` exposing exactly the given methods.

    Unlike a ``MagicMock``, any method production calls that wasn't scripted
    fails loudly with ``AttributeError``. Suitable for injection through
    ``llm_client=`` / ``client=`` params or for patching the ``LLMClient``
    symbol imported into the module under test, e.g.
    ``monkeypatch.setattr("notes.note_generator.LLMClient", lambda *a, **k: fake)``.
    """

    def _factory(**methods: Callable[..., Any]) -> SimpleNamespace:
        return SimpleNamespace(**methods)

    return _factory
