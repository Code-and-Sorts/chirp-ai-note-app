import hashlib
import json
import logging
import re
import tomllib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from rich.console import Console

from chirp.exceptions import EmbedModelChanged
from config.settings import ChirpSettings
from llm.client import LLMClient
from llm.exceptions import LLMError
from notes_chat.types import Chunk, NoteMeta
from utils.file_utils import META_FILENAME, _resolve_created_at, atomic_write_json

logger = logging.getLogger(__name__)

console = Console()

COLLECTION_NAME = "notes"
TEMP_COLLECTION_NAME = "notes_rebuild"
EMBED_ALIAS_KEY = "embed_alias"
EMBED_DIM_KEY = "embed_dim"


class IndexManager:
    def __init__(self, config: ChirpSettings, llm_client: LLMClient | None = None):
        self.config = config
        self.settings = config.notes_chat
        self.notes_root = config.directories.notes_root
        self._llm_client = llm_client

        self._chroma_client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._temp_collection = None

        self.manifest_file = self.settings.index_dir / "manifest.json"
        self.bm25_file = self.settings.index_dir / "bm25.json"

    @property
    def chroma_client(self) -> chromadb.ClientAPI:
        """Lazily open the Chroma client (and so the ``chroma/`` dir) on first use.

        A lexical-only install (``semantic_enabled=False``) never touches the
        vector half, so this stays unopened and no ``chroma/`` directory is
        created. Only the gated semantic paths reach it.
        """
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.settings.index_dir / "chroma"),
                settings=ChromaSettings(allow_reset=True),
            )
        return self._chroma_client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @collection.setter
    def collection(self, value: chromadb.Collection) -> None:
        self._collection = value

    def build_index(
        self,
        force: bool = False,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Build or update the search index."""
        if force:
            return self._force_rebuild(progress_callback)

        try:
            self.config.ensure_directories_exist()
            if self.settings.semantic_enabled:
                self._ensure_embed_fingerprint()

            manifest = self._load_manifest()
            current_files = self._scan_notes_files()

            added_files = []
            modified_files = []
            removed_files: list[str] = []

            for file_path, file_sig in current_files.items():
                if file_path not in manifest:
                    added_files.append(file_path)
                elif manifest[file_path] != file_sig:
                    modified_files.append(file_path)

            removed_files.extend(
                file_path for file_path in manifest if file_path not in current_files
            )

            total_changes = len(added_files) + len(modified_files) + len(removed_files)
            if total_changes == 0:
                return {
                    "success": True,
                    "message": "Index is up to date",
                    "files_processed": 0,
                }

            processed_count = 0
            failed_files: list[str] = []

            for file_path in removed_files:
                if self.settings.semantic_enabled:
                    self._remove_from_index(file_path)
                manifest.pop(file_path, None)
                processed_count += 1
                if progress_callback:
                    progress_callback()

            for file_path in added_files + modified_files:
                if self._add_to_index(Path(file_path)):
                    # Record only files that indexed, so a failed embed is
                    # retried next run rather than marked done forever.
                    manifest[file_path] = current_files[file_path]
                    processed_count += 1
                    if progress_callback:
                        progress_callback()
                else:
                    manifest.pop(file_path, None)
                    failed_files.append(file_path)

            self._save_manifest(manifest)
            self._rebuild_bm25()
            if self.settings.semantic_enabled:
                self._stamp_fingerprint_if_missing()
            self._report_failures(failed_files)

            return {
                "success": True,
                "files_processed": processed_count,
                "added": len(added_files),
                "modified": len(modified_files),
                "removed": len(removed_files),
                "failed": len(failed_files),
            }

        except Exception as e:  # noqa: BLE001 - build_index orchestrates many subsystems
            logger.debug("Index build failed: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    def add_note(
        self,
        notes_path: Path,
        *,
        guard_embed_fingerprint: bool = False,
        incremental_bm25: bool = False,
    ) -> bool:
        """Index a single note and persist its manifest + BM25 entries.

        Returns ``True`` when the note was embedded and recorded. With
        ``guard_embed_fingerprint`` an embed-model change raises
        :class:`EmbedModelChanged` before mismatched vectors are appended (the
        note-generation back door). ``incremental_bm25`` appends only this
        note's chunks instead of rebuilding the whole BM25 index, keeping a
        burst of saves O(note) per save.
        """
        if guard_embed_fingerprint and self.settings.semantic_enabled:
            self._ensure_embed_fingerprint()
        if not self._add_to_index(notes_path):
            return False

        manifest = self._load_manifest()
        current_files = self._scan_notes_files()
        file_path = str(notes_path)
        if file_path in current_files:
            manifest[file_path] = current_files[file_path]
            self._save_manifest(manifest)

        if incremental_bm25:
            self.append_bm25_for_file(file_path)
            if self.settings.semantic_enabled:
                self._stamp_fingerprint_if_missing()
        else:
            self._rebuild_bm25()
        return True

    def remove_note(self, notes_path: str | Path) -> None:
        """Drop a single note from the vector index, manifest, and BM25."""
        file_path = str(notes_path)
        if self.settings.semantic_enabled:
            self._remove_from_index(file_path)
        manifest = self._load_manifest()
        manifest.pop(file_path, None)
        self._save_manifest(manifest)
        self._rebuild_bm25()

    def _force_rebuild(
        self, progress_callback: Callable | None = None
    ) -> dict[str, Any]:
        """Rebuild the whole index atomically: the prior index survives a crash.

        The new vectors are written into a temp collection and the new
        manifest/bm25 into temp files; only after the add loop completes are the
        old artifacts swapped out. An interruption (Ctrl-C, embed failure mid-run)
        therefore leaves the previous collection, manifest, and bm25.json intact
        rather than the empty/partial state the up-front delete used to produce.
        """
        try:
            self.config.ensure_directories_exist()

            current_files = self._scan_notes_files()

            if self.settings.semantic_enabled:
                temp_collection = self._reset_temp_collection()
                active_collection = self.collection
                self.collection = temp_collection
                try:
                    new_manifest, failed_files, processed_count = self._index_files(
                        current_files, progress_callback
                    )
                finally:
                    self.collection = active_collection
                self._promote_temp_collection()
                self._write_fingerprint_metadata(self.collection)
            else:
                new_manifest, failed_files, processed_count = self._index_files(
                    current_files, progress_callback
                )

            self._save_manifest(new_manifest)
            self._rebuild_bm25()
            self._report_failures(failed_files)

            return {
                "success": True,
                "files_processed": processed_count,
                "added": len(new_manifest),
                "modified": 0,
                "removed": 0,
                "failed": len(failed_files),
            }

        except Exception as e:  # noqa: BLE001 - force rebuild orchestrates many subsystems
            logger.debug("Index force-rebuild failed: %s", e, exc_info=True)
            if self.settings.semantic_enabled:
                self._discard_temp_collection()
            return {"success": False, "error": str(e)}

    def _index_files(
        self,
        current_files: dict[str, dict[str, Any]],
        progress_callback: Callable | None,
    ) -> tuple[dict[str, Any], list[str], int]:
        """Add each note file to the index, returning (manifest, failed, count)."""
        new_manifest: dict[str, Any] = {}
        failed_files: list[str] = []
        processed_count = 0
        for file_path in current_files:
            if self._add_to_index(Path(file_path)):
                new_manifest[file_path] = current_files[file_path]
                processed_count += 1
                if progress_callback:
                    progress_callback()
            else:
                failed_files.append(file_path)
        return new_manifest, failed_files, processed_count

    def _report_failures(self, failed_files: list[str]) -> None:
        if not failed_files:
            return
        names = ", ".join(Path(path).parent.name for path in failed_files)
        console.print(
            f"[yellow]{len(failed_files)} note(s) failed to index and will be "
            f"retried next run: {names}[/yellow]"
        )

    def _reset_temp_collection(self):
        """Create a fresh temp collection the force rebuild writes into."""
        try:
            self.chroma_client.delete_collection(TEMP_COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001 - chromadb raises various internal exceptions
            logger.debug("Could not delete temp chroma collection: %s", exc)

        self._temp_collection = self.chroma_client.get_or_create_collection(
            name=TEMP_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        return self._temp_collection

    def _promote_temp_collection(self):
        """Swap the freshly-built temp collection in for the live one.

        The old ``notes`` collection is only touched here, after the rebuild's
        add loop has finished — so an interruption before this point leaves the
        previous index fully intact. The swap deletes the old collection and
        renames the temp into its place (chromadb 1.5.6 ``modify(name=...)``),
        which avoids re-adding every vector. The non-atomic window is two
        metadata ops (delete-old → rename-temp); chromadb has no transactional
        rename, so a crash *between* them would leave no ``notes`` collection,
        but the manifest/bm25 are still rewritten only after this returns, and
        the next build recreates the empty collection.
        """
        if self._temp_collection is None:
            return

        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001 - chromadb raises various internal exceptions
            logger.debug("Could not delete live chroma collection: %s", exc)

        self._temp_collection.modify(name=COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._temp_collection = None

    def _discard_temp_collection(self):
        try:
            self.chroma_client.delete_collection(TEMP_COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001 - chromadb raises various internal exceptions
            logger.debug("Could not delete temp chroma collection: %s", exc)
        self._temp_collection = None

    def _resolved_embed_alias(self) -> str:
        """Resolve the active embed model alias from the registry (best effort).

        Used as the human-readable half of the collection's embed fingerprint.
        Falls back to ``"unknown"`` so fingerprinting never blocks indexing when
        the registry is absent (the dimension half still catches real changes).
        """
        try:
            from llm.registry import read_registry

            return read_registry().default_embed or "unknown"
        except Exception as exc:  # noqa: BLE001 - registry IO/parse: fingerprint is best-effort
            logger.debug("Could not resolve embed alias: %s", exc)
            return "unknown"

    def _stored_fingerprint(self) -> tuple[str | None, int | None]:
        metadata = self.collection.metadata or {}
        alias = metadata.get(EMBED_ALIAS_KEY)
        dim = metadata.get(EMBED_DIM_KEY)
        return (alias, int(dim) if dim is not None else None)

    def _current_vector_dim(self) -> int | None:
        try:
            sample = self.collection.get(include=["embeddings"], limit=1)
        except Exception as exc:  # noqa: BLE001 - chromadb raises various internal exceptions
            logger.debug("Could not sample vector dimension: %s", exc)
            return None
        embeddings = sample.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            return len(embeddings[0])
        return None

    def _write_fingerprint_metadata(self, collection) -> None:
        # collection.modify rejects the hnsw:space key (distance function is
        # fixed at creation), so omit it; the cosine space still governs retrieval.
        metadata: dict[str, Any] = {EMBED_ALIAS_KEY: self._resolved_embed_alias()}
        dim = None
        try:
            sample = collection.get(include=["embeddings"], limit=1)
            embeddings = sample.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                dim = len(embeddings[0])
        except Exception as exc:  # noqa: BLE001 - chromadb raises various internal exceptions
            logger.debug("Could not read vector dimension for fingerprint: %s", exc)
        if dim is not None:
            metadata[EMBED_DIM_KEY] = dim
        try:
            collection.modify(metadata=metadata)
        except Exception as exc:  # noqa: BLE001 - chromadb raises various internal exceptions
            logger.debug("Could not write embed fingerprint: %s", exc)

    def _stamp_fingerprint_if_missing(self) -> None:
        """Write the embed fingerprint after the first incremental build.

        A force rebuild stamps it explicitly; an index first populated by an
        incremental build (or auto-index) would otherwise carry no fingerprint,
        so a later model change couldn't be detected. Only stamps when absent and
        the collection actually has vectors.
        """
        stored_alias, stored_dim = self._stored_fingerprint()
        if stored_alias is not None or stored_dim is not None:
            return
        if self._current_vector_dim() is None:
            return
        self._write_fingerprint_metadata(self.collection)

    def _ensure_embed_fingerprint(self) -> None:
        """Guard incremental builds against an embed-model/dimension change.

        ``.docs/embeddings.md`` invites swapping the embed model, but new-dim
        vectors silently mismatch the old ones. If the active embed alias differs
        from the fingerprint stored on the collection, raise an actionable error
        telling the user to rebuild — rather than appending mismatched vectors.
        """
        stored_alias, stored_dim = self._stored_fingerprint()
        if stored_alias is None and stored_dim is None:
            return

        current_alias = self._resolved_embed_alias()
        if (
            stored_alias not in (None, "unknown")
            and current_alias != "unknown"
            and current_alias != stored_alias
        ):
            raise EmbedModelChanged(
                f"embed model changed ({stored_alias!r} → {current_alias!r}); "
                "run `chirp index --force` to rebuild the index"
            )

    def _scan_notes_files(self) -> dict[str, dict[str, Any]]:
        """Scan per-note directories and return file signatures."""
        files: dict[str, dict[str, Any]] = {}
        if not self.notes_root.exists():
            return files

        for note_file in self.notes_root.glob("*/notes.md"):
            stat = note_file.stat()
            files[str(note_file)] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "path": str(note_file),
            }

        return files

    def _load_manifest(self) -> dict[str, Any]:
        """Load the index manifest."""
        if not self.manifest_file.exists():
            return {}

        try:
            with self.manifest_file.open() as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError):
            return {}

    def _save_manifest(self, manifest: dict[str, Any]):
        """Save the index manifest."""
        # A lexical-only add_note never opens Chroma (which used to create
        # index_dir as a side effect), so ensure the directory exists here.
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.manifest_file, manifest)

    def _chunks_for_file(self, file_path: Path) -> list[Chunk]:
        """Read a note file and split it into chunks (no embedding / Chroma touch).

        Shared by the embed path and the file-sourced BM25 build so both derive
        identical chunk ids from the same content.
        """
        content = file_path.read_text(encoding="utf-8")
        meta = self._extract_metadata(file_path, content)
        if not meta:
            return []
        return self._chunk_content(file_path, content, meta)

    def _add_to_index(self, file_path: Path) -> bool:
        """Add a notes file to the index.

        With ``semantic_enabled=False`` the lexical store is the only index, so
        this just validates the file chunks and returns success without loading
        the embed model or touching Chroma; ``_rebuild_bm25`` / the BM25 append
        path persist the lexical entries.
        """
        try:
            chunks = self._chunks_for_file(file_path)

            if not chunks:
                return False

            if not self.settings.semantic_enabled:
                return True

            self._remove_from_index(str(file_path))

            embeddings = self._get_embeddings([chunk.content for chunk in chunks])

            if not embeddings:
                console.print(
                    f"[yellow]Failed to get embeddings for {file_path.name}[/yellow]"
                )
                return False

            self.collection.add(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                embeddings=embeddings,  # type: ignore[arg-type]
                metadatas=[self._chunk_to_metadata(chunk) for chunk in chunks],
            )

            return True

        except Exception as e:  # noqa: BLE001 - add_to_index: chromadb, embeddings, or IO
            logger.debug("Failed to index %s: %s", file_path.name, e)
            console.print(f"[red]Failed to index {file_path.name}: {e}[/red]")
            return False

    def _remove_from_index(self, file_path: str):
        """Remove a file from the index."""
        try:
            results = self.collection.get(where={"path": file_path})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
        except Exception as e:  # noqa: BLE001 - chromadb raises various internal exceptions
            logger.debug("Failed to remove %s: %s", Path(file_path).name, e)
            console.print(
                f"[yellow]Failed to remove {Path(file_path).name}: {e}[/yellow]"
            )

    def _extract_metadata(self, file_path: Path, content: str) -> NoteMeta | None:
        """Extract metadata from a notes file."""
        try:
            title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem

            date = self._resolve_meeting_date(file_path.parent)

            participants = []
            participant_matches = re.findall(
                r"\*\*Participants:\*\* (.+)$", content, re.MULTILINE
            )
            for match in participant_matches:
                participants.extend([p.strip() for p in match.split(",")])

            duration_match = re.search(r"\*\*Duration:\*\* (\d+)", content)
            duration = int(duration_match.group(1)) if duration_match else 0

            stat = file_path.stat()

            return NoteMeta(
                path=file_path,
                title=title,
                date=date,
                participants=participants,
                duration=duration,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )

        except (OSError, ValueError, AttributeError, re.error) as e:
            console.print(
                f"[yellow]Failed to extract metadata from {file_path.name}: {e}[/yellow]"
            )
            return None

    def _resolve_meeting_date(self, note_dir: Path) -> datetime:
        """Return the canonical (naive-local) meeting date for a note directory.

        Reuses ``utils.file_utils._resolve_created_at`` — the single source of
        truth that prefers ``meta.toml``'s ``date`` and falls back to the note
        directory's mtime. The note file's own mtime is deliberately NOT used:
        regenerating ``notes.md`` must not move a meeting's date to "today".
        """
        meta: dict[str, Any] = {}
        meta_path = note_dir / META_FILENAME
        if meta_path.exists():
            try:
                with meta_path.open("rb") as fh:
                    meta = dict(tomllib.load(fh))
            except (OSError, tomllib.TOMLDecodeError):
                meta = {}
        return _resolve_created_at(meta, note_dir)

    def _chunk_content(
        self, file_path: Path, content: str, meta: NoteMeta
    ) -> list[Chunk]:
        """Split content into chunks for indexing."""
        chunks = []

        sections = re.split(r"\n## ", content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            if i > 0:
                section = "## " + section

            if len(section) < 50:
                continue

            if len(section) <= self.settings.chunk_size:
                # Prefix by slug (parent dir), not file_path.stem: the stem is
                # always "notes", so stem-prefixed ids collide across notes and
                # Chroma's add silently keeps only one note's chunks per id.
                chunk_id = f"{file_path.parent.name}_{i:03d}"
                norm = re.sub(r"\s+", " ", section.strip()).lower()
                content_hash = hashlib.sha256(norm.encode()).hexdigest()[:16]

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        path=file_path,
                        content=section,
                        meta=meta,
                        content_hash=content_hash,
                    )
                )
            else:
                for j, chunk_text in enumerate(self._split_large_section(section)):
                    chunk_id = f"{file_path.parent.name}_{i:03d}_{j:03d}"
                    norm = re.sub(r"\s+", " ", chunk_text.strip()).lower()
                    content_hash = hashlib.sha256(norm.encode()).hexdigest()[:16]

                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            path=file_path,
                            content=chunk_text,
                            meta=meta,
                            content_hash=content_hash,
                        )
                    )

        return chunks

    def _split_large_section(self, section: str) -> list[str]:
        """Split large sections into smaller chunks with overlap.

        Uses a simple words-based sliding window derived from character budgets via a
        6-chars-per-word heuristic:
        - chunk_words = chunk_size // 6
        - overlap_words = overlap // 6

        This keeps tuning intuitive in characters while producing reasonably sized
        word chunks for embedding. Overlap reduces boundary information loss.
        """
        words = section.split()
        chunks = []

        chunk_words = self.settings.chunk_size // 6
        overlap_words = self.settings.overlap // 6

        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(chunk_text)

            if end >= len(words):
                break

            start = end - overlap_words

        return chunks

    def _get_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """Embed chunk texts via chirpd, preserving input order. None on failure."""
        if not texts:
            return []
        # Reuse one client across calls — LLMClient() resolves the socket path on
        # construction. embed is batched-by-design, so one call covers all chunks.
        if self._llm_client is None:
            self._llm_client = LLMClient()
        try:
            return self._llm_client.embed_sync(inputs=texts, model="default")
        except LLMError as e:
            console.print(f"[red]Failed to get embeddings: {e}[/red]")
            return None

    def _chunk_to_metadata(self, chunk: Chunk) -> dict[str, Any]:
        """Convert chunk to metadata dict for Chroma."""
        return {
            "path": str(chunk.path),
            "title": chunk.meta.title,
            "date": chunk.meta.date.isoformat(),
            "participants": json.dumps(chunk.meta.participants),
            "duration": chunk.meta.duration,
            "content_hash": chunk.content_hash,
        }

    def _rebuild_bm25(self):
        """Rebuild the BM25 lexical store from the note files on disk.

        Sourced from the files (not Chroma) so the lexical index is fully
        self-sufficient — it exists and answers queries even when the vector
        half is disabled and no ``chroma/`` directory has been created.
        """
        try:
            from notes_chat.bm25 import rebuild_bm25_index

            doc_ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for file_path in self._scan_notes_files():
                for chunk in self._chunks_for_file(Path(file_path)):
                    doc_ids.append(chunk.id)
                    documents.append(chunk.content)
                    metadatas.append(self._chunk_to_metadata(chunk))

            rebuild_bm25_index(doc_ids, documents, metadatas, self.bm25_file)
        except Exception as e:  # noqa: BLE001 - IO / parse; many failure modes
            logger.debug("Failed to rebuild BM25 index: %s", e)
            console.print(f"[yellow]Failed to rebuild BM25 index: {e}[/yellow]")

    def append_bm25_for_file(self, file_path: str) -> None:
        """Incrementally merge one note's chunks into the BM25 lexicon.

        Used by the auto-index-on-save path so a single save doesn't trigger a
        full-corpus re-tokenize (a burst of N saves would otherwise cost N full
        rebuilds). Chunks come from the file itself, so this works with the
        vector half disabled.
        """
        try:
            from notes_chat.bm25 import append_bm25_index

            chunks = self._chunks_for_file(Path(file_path))
            doc_ids = [chunk.id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [self._chunk_to_metadata(chunk) for chunk in chunks]
            if doc_ids:
                # Purge {slug}_NNN ghost ids for this note that vanished when it
                # was re-chunked into fewer chunks.
                slug_prefix = f"{Path(file_path).parent.name}_"
                append_bm25_index(
                    self.bm25_file,
                    doc_ids,
                    documents,
                    metadatas,
                    stale_id_prefix=slug_prefix,
                )
        except Exception as e:  # noqa: BLE001 - IO / parse; many failure modes
            logger.debug("Failed to append BM25 index for %s: %s", file_path, e)


def build_index(
    config: ChirpSettings,
    force: bool = False,
    progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """Build the notes search index."""
    manager = IndexManager(config)
    return manager.build_index(force=force, progress_callback=progress_callback)
