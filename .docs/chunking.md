# Chunking Strategy

This document defines the chunking strategy used for indexing notes into the RAG pipeline.

- Code reference: `notes_chat/index.py`
- Config knobs: `notes_chat.chunk_size`, `notes_chat.overlap` in `config/config.yaml` / `config/settings.py`
- Defaults: `chunk_size: 1000` characters, `overlap: 200` characters

Chunks feed the BM25 lexical index always, and the embedding/Chroma index only when semantic search is enabled (`notes_chat.semantic_enabled`; see [Retrieval](./hybrid-retrieval.md)). Chunking itself is identical in both modes.

## Goals

- Preserve semantic boundaries by splitting around second-level headings first (`##` in Markdown)
- Keep chunks under a target character budget for efficient lexical indexing and (when enabled) embedding
- Add overlap to reduce information loss at boundaries

## Inputs & Outputs

- Input: Markdown note text (from `notes-out/*.md`) and extracted metadata
- Output: List of chunks with fields:
  - `id`: `<file_stem>_<section_index>` or `<file_stem>_<section_index>_<chunk_index>`
  - `content`: the chunk text
  - `meta`: `title`, `date`, `participants`, `duration`, etc.
  - `content_hash`: stable hash of normalized content for de-duplication

## Algorithm

1. Section-aware split

   - Split the document on `\n##`, effectively chunking by second-level headings while retaining the heading text in the section.
   - Skip empty sections and very short ones: sections with `< 50` characters are ignored.
   - Note: Sections are defined as meetings and/or single transcripts.

2. Size check per section

   - If `len(section) <= chunk_size`, emit the whole section as a single chunk.
   - Else, split the large section with overlapping windows (see below).

3. Overlapping windows for large sections

   - Convert character budgets to approximate word windows using a 6 characters-per-word heuristic:
     - `chunk_words = chunk_size // 6`
     - `overlap_words = overlap // 6`
   - Slide a window across the section’s words:
     - `start = 0`
     - `end = min(start + chunk_words, total_words)`
     - Emit `" ".join(words[start:end])`
     - Set `start = end - overlap_words` (floors at 0) and repeat until the end of the section.

4. Metadata and IDs

- Each chunk gets a deterministic `id` and carries `meta` plus `content_hash` (used for de-duplication).

## Defaults & Tuning

- Defaults from `config/config.yaml`:
  - `notes_chat.chunk_size: 1000` (approx. `~166` words per chunk)
  - `notes_chat.overlap: 200` (approx. `~33` words overlap)
- Increase `chunk_size` if your sections are dense and short, or you want fewer, larger chunks.
- Increase `overlap` if you see boundary-loss in answers; decrease for faster indexing/search.

## Edge Cases & Notes

- No headings: the entire file acts as a single section and will be either a single chunk or split by word windows.
- Very short files/sections (`< 50` chars) are ignored to avoid noisy chunks.
- Non-ASCII/whitespace: tokenization uses `str.split()` (whitespace); very long tokens (e.g., URLs) may push beyond targets.
- IDs and signatures (`content_hash`) help merge/dedupe across hybrid retrieval (Chroma + BM25).

## Rationale

- Section-first splitting aligns chunks with human-authored structure.
- Overlap preserves context across chunk boundaries, improving recall in semantic search.
- Word-based windows derived from character budgets keep behavior stable while allowing intuitive char-sized tuning.

See also: [Architecture](./architecture.md)
