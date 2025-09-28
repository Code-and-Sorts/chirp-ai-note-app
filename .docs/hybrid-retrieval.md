# Hybrid Retrieval: Embeddings + BM25

This note explains why Chirp uses hybrid retrieval (semantic embeddings + BM25 lexical search), how it works, and when you might tune or change it.

## TL;DR

- Keep both: embeddings catch semantic matches; BM25 catches exact terms (IDs, names, phrases).
- Merge the results, dedupe by `(path, content_hash)`, then build a context under a fixed character budget.

## How it works

See the retrieval diagram in the Architecture doc: [Architecture → Retrieval (ask)](./architecture.md#retrieval-ask).

- Embeddings: query and chunks are embedded with the same model; Chroma returns top-k by cosine similarity.
- BM25: ranks chunks by lexical overlap; strong for exact tokens, IDs, acronyms, and phrases.
- Merge + Dedupe: combine lists and deduplicate using `(path, content_hash)`.
- Context: allocate text across top chunks within `ctx_char_budget`, then prompt the LLM and attach sources.

## Why hybrid?

- Short or specific queries: BM25 shines on IDs (e.g., `jira-123`), names, codes, dates, and quoted phrases.
- Paraphrased or fuzzy queries: embeddings retrieve semantically related content even if words differ.
- Local and fast: both run locally (Ollama + Chroma + BM25 corpus) with low overhead.

## When to change it

- Stronger embeddings + re-ranker: if you add a cross-encoder/LLM re-ranking step, embeddings-only can be competitive.
- Domain-specific tokens: increase BM25 weight if queries often include IDs or exact terms.
- Minimal needs: if queries are always natural language, embeddings-only may be sufficient.

## Tuning tips

- k-values: start with `k=10` for both; keep totals small to avoid noisy merges.
- Fusion: simple score normalization or Reciprocal Rank Fusion (RRF) keeps logic robust.
- Routing: detect quotes, ALL-CAPS acronyms, many digits → boost BM25’s influence.
- Budgeting: round-robin or interleaving across sources when constructing the context.

## Failure modes and guardrails

- Empty or low-similarity vectors: backfill from BM25.
- Duplicate content across notes: dedupe with `(path, content_hash)`.
- Very long chunks: rely on chunking and overlaps to keep embeddings effective (see [Chunking](./chunking.md)).
- Index freshness: when `notes_chat.auto_index` is enabled, manual note saves trigger an index update.

## References

- Embeddings backend: [Embeddings](./embeddings.md)
- Architecture overview: [Architecture](./architecture.md)
