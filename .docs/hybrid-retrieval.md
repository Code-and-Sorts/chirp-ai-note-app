# Retrieval: Lexical-first, hybrid when enabled

This note explains how Chirp retrieves context: BM25 lexical search by default, with semantic embeddings added on top when you opt in. It covers how the two combine, how queries are routed, and when to tune or change it.

## TL;DR

- **Lexical-first.** BM25 is the default and only required retriever; it catches exact terms (IDs, names, phrases) and needs no embedding model.
- **Semantic is opt-in** (`notes_chat.semantic_enabled`, default `False`; flip it with `chirp config --semantic`). When on, embeddings also catch paraphrase/conceptual matches and are fused with BM25.
- Merge the results, dedupe by `(path, content_hash)`, then build a context under a fixed character budget.

## How it works

See the retrieval diagram in the Architecture doc: [Architecture → Retrieval (ask)](./architecture.md#retrieval-ask).

- BM25 (always): ranks chunks by lexical overlap; strong for exact tokens, IDs, acronyms, and phrases. With `semantic_enabled=False` this is the entire retrieval path — no Chroma query, no query embedding.
- Embeddings (when `semantic_enabled=True`): query and chunks are embedded with the same model; Chroma returns top-k by cosine similarity.
- Query routing (`_route_query` in `notes_chat/retrieval.py`): when semantic is on, the query's shape picks per-source fusion weights `(bm25, chroma)`. Literal signals — quoted spans, id-shaped tokens (`jira-123`), ALL-CAPS acronyms, or very short queries — favour BM25 (`LITERAL_WEIGHTS = (1.0, 0.3)`); anything else is treated as conceptual (`CONCEPTUAL_WEIGHTS = (1.0, 1.0)`). With semantic off the weights are fixed `LEXICAL_ONLY = (1.0, 0.0)`.
- Merge + Dedupe: combine lists and deduplicate using `(path, content_hash)`, fusing ranks with weighted Reciprocal Rank Fusion (`RRF_K = 60`).
- Context: allocate text across top chunks within `ctx_char_budget`, then prompt the LLM and attach sources.

## Why lexical-first?

- No setup cost: BM25 needs no embedding model, so search works the moment you have notes — nothing to download, no daemon embed calls.
- Short or specific queries: BM25 shines on IDs (e.g., `jira-123`), names, codes, dates, and quoted phrases, which is where many note searches land.
- Opt into semantic when paraphrase matters: enabling embeddings (`chirp config --semantic`) retrieves semantically related content even when the words differ, fused with BM25 rather than replacing it.

## When to enable or change it

- Enable semantic search: run `chirp config --semantic` if queries are often natural-language paraphrases rather than exact terms. It registers the embed model, verifies the daemon can load it, and rebuilds the index.
- Stay lexical-only: if queries are usually exact terms/IDs, the default needs no embedding model at all; `chirp config --no-semantic --purge` returns to it and reclaims the vector store.
- Domain-specific tokens: increase BM25 weight (see `_route_query`) if queries often include IDs or exact terms.

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
