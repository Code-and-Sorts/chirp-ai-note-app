import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import requests
from chromadb.config import Settings as ChromaSettings
from rich.console import Console

from config.settings import ChirpSettings
from notes_chat.types import Chunk, NoteMeta

console = Console()


class IndexManager:
    def __init__(self, config: ChirpSettings):
        self.config = config
        self.settings = config.notes_chat
        self.notes_root = config.directories.notes_root

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.settings.index_dir / "chroma"),
            settings=ChromaSettings(allow_reset=True),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="notes", metadata={"hnsw:space": "cosine"}
        )

        self.manifest_file = self.settings.index_dir / "manifest.json"
        self.bm25_file = self.settings.index_dir / "bm25.json"

    def build_index(
        self,
        force: bool = False,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Build or update the search index."""
        try:
            self.config.ensure_directories_exist()

            if force:
                self._reset_index()

            manifest = self._load_manifest()
            current_files = self._scan_notes_files()

            added_files = []
            modified_files = []
            removed_files = []

            for file_path, file_sig in current_files.items():
                if file_path not in manifest:
                    added_files.append(file_path)
                elif manifest[file_path] != file_sig:
                    modified_files.append(file_path)

            for file_path in manifest:
                if file_path not in current_files:
                    removed_files.append(file_path)

            total_changes = len(added_files) + len(modified_files) + len(removed_files)
            if total_changes == 0 and not force:
                return {
                    "success": True,
                    "message": "Index is up to date",
                    "files_processed": 0,
                }

            processed_count = 0

            for file_path in removed_files:
                self._remove_from_index(file_path)
                processed_count += 1
                if progress_callback:
                    progress_callback()

            for file_path in added_files + modified_files:
                if self._add_to_index(Path(file_path)):
                    processed_count += 1
                    if progress_callback:
                        progress_callback()

            new_manifest = current_files
            self._save_manifest(new_manifest)
            self._rebuild_bm25()

            return {
                "success": True,
                "files_processed": processed_count,
                "added": len(added_files),
                "modified": len(modified_files),
                "removed": len(removed_files),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _reset_index(self):
        """Reset the entire index."""
        try:
            self.chroma_client.delete_collection("notes")
        except Exception:
            pass

        self.collection = self.chroma_client.get_or_create_collection(
            name="notes", metadata={"hnsw:space": "cosine"}
        )

        if self.manifest_file.exists():
            self.manifest_file.unlink()
        if self.bm25_file.exists():
            self.bm25_file.unlink()

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
            with open(self.manifest_file) as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_manifest(self, manifest: dict[str, Any]):
        """Save the index manifest."""
        with open(self.manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

    def _add_to_index(self, file_path: Path) -> bool:
        """Add a notes file to the index."""
        try:
            content = file_path.read_text(encoding="utf-8")
            meta = self._extract_metadata(file_path, content)

            if not meta:
                return False

            chunks = self._chunk_content(file_path, content, meta)

            if not chunks:
                return False

            self._remove_from_index(str(file_path))

            embeddings = self._get_embeddings([chunk.content for chunk in chunks])

            if not embeddings:
                console.print(
                    f"[yellow]⚠️ Failed to get embeddings for {file_path.name}[/yellow]"
                )
                return False

            self.collection.add(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                embeddings=embeddings,  # type: ignore
                metadatas=[self._chunk_to_metadata(chunk) for chunk in chunks],
            )

            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to index {file_path.name}: {e}[/red]")
            return False

    def _remove_from_index(self, file_path: str):
        """Remove a file from the index."""
        try:
            results = self.collection.get(where={"path": file_path})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
        except Exception as e:
            console.print(
                f"[yellow]⚠️ Failed to remove {Path(file_path).name}: {e}[/yellow]"
            )

    def _extract_metadata(self, file_path: Path, content: str) -> NoteMeta | None:
        """Extract metadata from a notes file."""
        try:
            title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem

            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", file_path.name)
            if date_match:
                date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            else:
                stat = file_path.stat()
                date = datetime.fromtimestamp(stat.st_mtime)

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

        except Exception as e:
            console.print(
                f"[yellow]⚠️ Failed to extract metadata from {file_path.name}: {e}[/yellow]"
            )
            return None

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
                chunk_id = f"{file_path.stem}_{i:03d}"
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
                    chunk_id = f"{file_path.stem}_{i:03d}_{j:03d}"
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
        """Get embeddings from Ollama."""
        try:
            embeddings = []

            for text in texts:
                response = requests.post(
                    f"{self.config.models.ollama_url}/api/embeddings",
                    json={"model": self.settings.emb_model, "prompt": text},
                    timeout=30,
                )

                if response.status_code != 200:
                    return None

                result = response.json()
                embeddings.append(result["embedding"])

            return embeddings

        except Exception as e:
            console.print(f"[red]❌ Failed to get embeddings: {e}[/red]")
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
        """Rebuild BM25 index from Chroma data."""
        try:
            from notes_chat.bm25 import rebuild_bm25_index

            rebuild_bm25_index(self.collection, self.bm25_file)
        except Exception as e:
            console.print(f"[yellow]⚠️ Failed to rebuild BM25 index: {e}[/yellow]")


def build_index(
    config: ChirpSettings,
    force: bool = False,
    progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """Build the notes search index."""
    manager = IndexManager(config)
    return manager.build_index(force=force, progress_callback=progress_callback)
