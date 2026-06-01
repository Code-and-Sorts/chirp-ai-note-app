"""Opt-in integration test for ``MLXBackend`` embed-model load + pooled inference.

This exercises the real ``mlx_embeddings`` loader and ``generate`` path against a
small real embedding model. It is marked ``@slow @integration`` and SKIPS unless
the model is already present in the local Hugging Face cache, so it never triggers
a network download or runs in the default unit suite.

Run it explicitly (requires Apple Silicon + the model cached):

    # one-time: pull the small (~130MB) model into the HF cache
    uv run python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('mlx-community/bge-small-en-v1.5-bf16')"

    # then run this test
    uv run pytest tests/chirpd/test_backend_embed_integration.py \
        -m "slow and integration" -q
"""

from __future__ import annotations

import math

import pytest

from chirpd.backend import MLXBackend

EMBED_REPO = "mlx-community/bge-small-en-v1.5-bf16"


def _model_is_cached(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    try:
        snapshot_download(repo, local_files_only=True)
    except Exception:  # noqa: BLE001 — any miss (not cached / no hub) → skip
        return False
    return True


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


@pytest.mark.skipif(
    not _model_is_cached(EMBED_REPO),
    reason=f"{EMBED_REPO} not in local HF cache; run snapshot_download first",
)
async def test_embed_returns_pooled_order_preserving_vectors() -> None:
    backend = MLXBackend()
    handle = await backend.load(EMBED_REPO, "embed")
    try:
        # A repeated input ("the cat sat") lets us assert order + determinism;
        # the near-duplicate vs unrelated pair sanity-checks real pooling.
        inputs = [
            "the cat sat on the mat",
            "machine learning models run on hardware",
            "the cat sat on the mat",
            "a kitten rested on the rug",
        ]
        vectors = await backend.embed(handle, inputs)
    finally:
        await backend.unload(handle)

    assert len(vectors) == len(inputs)

    dimensionality = len(vectors[0])
    assert dimensionality > 0
    assert all(len(vector) == dimensionality for vector in vectors)

    # Order preserved + deterministic: the repeated input yields an identical vector.
    assert vectors[0] == vectors[2]

    cat = vectors[0]
    unrelated = vectors[1]
    near_duplicate = vectors[3]

    similarity_near = _cosine(cat, near_duplicate)
    similarity_unrelated = _cosine(cat, unrelated)
    assert similarity_near > similarity_unrelated


async def test_embed_empty_inputs_returns_empty_without_loading() -> None:
    backend = MLXBackend()
    # Empty input must short-circuit before touching the model, so a handle with
    # no real model/processor is fine here.
    result = await backend.embed({"repo": EMBED_REPO}, [])
    assert result == []
