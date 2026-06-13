"""Embedding-adapter tests for the chirpd cutover (story 6.3).

Both embed call sites — query-time (`retrieval._get_query_embedding`) and
indexer-time (`index.IndexManager._get_embeddings`) — now route through
`llm.client.embed_sync`. These tests pin the call shape, the empty-input
short-circuits, the `LLMError -> None` best-effort mapping, and (critically) the
input→output ordering that a future batched-embed optimization must not break.
The client stand-ins are the shared fixtures from `tests/conftest.py` (story
6.5), injected through the `client=` / `llm_client=` params.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from config.settings import ChirpSettings
from llm.exceptions import LLMModelError, LLMProtocolError, LLMTransportError
from notes_chat.index import IndexManager
from notes_chat.retrieval import _get_query_embedding

_DUMMY_CONFIG = SimpleNamespace()  # _get_query_embedding no longer reads config


# --- query-time: retrieval._get_query_embedding (AC-2, AC-5, AC-10) ---------


def test_query_embedding_returns_first_vector_and_call_shape(
    fake_llm_client, fake_embed
):
    client = fake_llm_client(embed_sync=fake_embed([[0.1, 0.2, 0.3]]))

    result = _get_query_embedding(_DUMMY_CONFIG, "what did we decide?", client=client)

    assert result == [0.1, 0.2, 0.3]
    assert client.embed_sync.calls == [
        {"inputs": ["what did we decide?"], "model": "default"}
    ]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_query_embedding_short_circuits_empty_input(query, fake_llm_client, fake_embed):
    client = fake_llm_client(embed_sync=fake_embed([[0.9]]))

    result = _get_query_embedding(_DUMMY_CONFIG, query, client=client)

    assert result is None
    assert client.embed_sync.calls == []  # client must NOT be called on empty input


def test_query_embedding_returns_none_on_empty_vectors(fake_llm_client):
    # A daemon that completes without vectors violates embed_sync's one-vector-
    # per-input contract, so this defensive branch is scripted with a bare
    # callable rather than `fake_embed` (whose length-parity assert forbids
    # scripting that dishonest shape).
    client = fake_llm_client(embed_sync=lambda inputs, model="default": [])

    assert _get_query_embedding(_DUMMY_CONFIG, "q", client=client) is None


@pytest.mark.parametrize("error", [LLMTransportError, LLMProtocolError, LLMModelError])
def test_query_embedding_returns_none_on_llm_error(
    error, fake_llm_client, raise_llm_error
):
    client = fake_llm_client(embed_sync=raise_llm_error(error, "daemon trouble"))

    assert _get_query_embedding(_DUMMY_CONFIG, "q", client=client) is None
    assert len(client.embed_sync.calls) == 1


# --- indexer-time: IndexManager._get_embeddings (AC-3, AC-4, AC-10, AC-11) --


def _make_index_manager(tmp_path, client):
    config = ChirpSettings()
    config.directories.notes_root = tmp_path
    config.notes_chat.index_dir = tmp_path / ".notes_index"
    return IndexManager(config, llm_client=client)


def test_indexer_embeddings_preserve_input_order(tmp_path, fake_llm_client, fake_embed):
    # AC-11: a synthetic 3-element batch must map 1:1 to inputs, in order.
    client = fake_llm_client(embed_sync=fake_embed([[1.0], [2.0], [3.0]]))
    manager = _make_index_manager(tmp_path, client)

    result = manager._get_embeddings(["a", "b", "c"])

    assert result == [[1.0], [2.0], [3.0]]
    assert client.embed_sync.calls == [{"inputs": ["a", "b", "c"], "model": "default"}]


def test_indexer_embeddings_empty_input_skips_client(
    tmp_path, fake_llm_client, fake_embed
):
    client = fake_llm_client(embed_sync=fake_embed([[1.0]]))
    manager = _make_index_manager(tmp_path, client)

    assert manager._get_embeddings([]) == []
    assert client.embed_sync.calls == []


def test_indexer_embeddings_returns_none_on_llm_error(
    tmp_path, fake_llm_client, raise_llm_error
):
    client = fake_llm_client(
        embed_sync=raise_llm_error(LLMTransportError, "embed daemon down")
    )
    manager = _make_index_manager(tmp_path, client)

    assert manager._get_embeddings(["a", "b"]) is None
