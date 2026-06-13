"""Inventory smoke test for the notes-quality regression corpus (story 6.1).

Validates the corpus shape required by PRD Domain-Specific Requirements ->
Validation Methodology before story 6.6 replays the corpus against the MLX
pipeline: >=10 recordings, complete transcript/notes pairs, no committed
audio, and the bucket distribution declared in bucket_manifest.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

CORPUS_DIR = Path(__file__).parent / "notes_quality"
AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".aiff", ".ogg"}
MIN_RECORDINGS = 10
SHORT_MAX_WORDS = 300
LONG_MIN_WORDS = 3000
MIN_LENGTH_BUCKET_COUNTS = {"short": 3, "medium": 4, "long": 3}
REQUIRED_SPEAKER_BUCKETS = {"single", "two", "multi"}
REQUIRED_DOMAIN_BUCKETS = {"technical", "non-technical"}
REQUIRED_MANIFEST_FIELDS = {"speakers", "domain", "word_count_bucket"}


def corpus_slugs() -> list[str]:
    if not CORPUS_DIR.is_dir():
        return []
    return sorted(path.name for path in CORPUS_DIR.iterdir() if path.is_dir())


def transcript_word_count(slug: str) -> int:
    transcript = CORPUS_DIR / slug / "transcript.txt"
    assert transcript.is_file(), f"missing transcript for {slug}: {transcript}"
    return len(transcript.read_text(encoding="utf-8").split())


def length_bucket(word_count: int) -> str:
    if word_count < SHORT_MAX_WORDS:
        return "short"
    if word_count > LONG_MIN_WORDS:
        return "long"
    return "medium"


def load_manifest() -> dict[str, Any]:
    manifest_path = CORPUS_DIR / "bucket_manifest.toml"
    assert manifest_path.is_file(), f"missing manifest: {manifest_path}"
    with manifest_path.open("rb") as handle:
        return tomllib.load(handle)


def test_corpus_directory_exists() -> None:
    assert CORPUS_DIR.is_dir(), f"missing corpus directory: {CORPUS_DIR}"


def test_corpus_has_minimum_recordings() -> None:
    slugs = corpus_slugs()
    assert len(slugs) >= MIN_RECORDINGS, (
        f"corpus needs >={MIN_RECORDINGS} recordings, found {len(slugs)}: {slugs}"
    )


@pytest.mark.parametrize("slug", corpus_slugs())
@pytest.mark.parametrize(
    "filename", ["transcript.txt", "notes_before.md", "notes_after.md"]
)
def test_recording_file_present_and_utf8(slug: str, filename: str) -> None:
    # notes_after.md is required since story 6.6's regression comparison run;
    # a careless rebase that drops one fails here loudly.
    path = CORPUS_DIR / slug / filename
    assert path.is_file(), f"missing {path}"
    content = path.read_text(encoding="utf-8")
    assert content.strip(), f"empty {path}"


def test_no_audio_files_committed() -> None:
    audio_files = [
        path
        for path in CORPUS_DIR.rglob("*")
        if path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    assert not audio_files, f"audio files must not be committed: {audio_files}"


def test_length_bucket_distribution() -> None:
    counts = {"short": 0, "medium": 0, "long": 0}
    for slug in corpus_slugs():
        counts[length_bucket(transcript_word_count(slug))] += 1
    for bucket, minimum in MIN_LENGTH_BUCKET_COUNTS.items():
        assert counts[bucket] >= minimum, (
            f"need >={minimum} {bucket} recordings, got {counts[bucket]} "
            f"(distribution: {counts})"
        )


def test_manifest_matches_subdirectories() -> None:
    manifest = load_manifest()
    assert manifest.get("schema_version") == 1
    manifest_slugs = set(manifest.get("recordings", {}))
    directory_slugs = set(corpus_slugs())
    assert manifest_slugs == directory_slugs, (
        f"manifest/directory mismatch — only in manifest: "
        f"{sorted(manifest_slugs - directory_slugs)}, only on disk: "
        f"{sorted(directory_slugs - manifest_slugs)}"
    )


def test_manifest_entries_have_required_fields() -> None:
    recordings = load_manifest().get("recordings", {})
    missing = {
        slug: sorted(REQUIRED_MANIFEST_FIELDS - set(entry))
        for slug, entry in recordings.items()
        if not REQUIRED_MANIFEST_FIELDS <= set(entry)
    }
    assert not missing, f"manifest entries missing required fields: {missing}"


def test_manifest_covers_speaker_and_domain_buckets() -> None:
    recordings = load_manifest().get("recordings", {})
    speakers = {entry["speakers"] for entry in recordings.values()}
    domains = {entry["domain"] for entry in recordings.values()}
    assert REQUIRED_SPEAKER_BUCKETS <= speakers, (
        f"missing speaker buckets: {sorted(REQUIRED_SPEAKER_BUCKETS - speakers)}"
    )
    assert REQUIRED_DOMAIN_BUCKETS <= domains, (
        f"missing domain buckets: {sorted(REQUIRED_DOMAIN_BUCKETS - domains)}"
    )


def test_manifest_word_count_buckets_match_transcripts() -> None:
    recordings = load_manifest().get("recordings", {})
    mismatches = {}
    for slug, entry in recordings.items():
        derived = length_bucket(transcript_word_count(slug))
        if entry["word_count_bucket"] != derived:
            mismatches[slug] = (entry["word_count_bucket"], derived)
    assert not mismatches, f"manifest bucket != derived bucket: {mismatches}"
