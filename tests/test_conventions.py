"""Suite-hygiene invariants for the chirpd cutover (story 6.5, AC-11).

The LLM-touching tests migrated off Ollama-shaped fixtures in stories 6.2-6.5
must stay migrated: no `requests` mocks simulating LLM/embedding calls, no
Ollama API references, no `mlx_lm` mocks (use `FakeBackend` at the
`LLMBackend` boundary instead).

Story 7.5 retired the last deferred Ollama helpers in `notes_chat/prompting.py`
(and the `ollama_url` setting), so there is no longer any carve-out — every
migrated test file must be fully clean.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).parent

MIGRATED_FILES = sorted(
    [
        *TESTS_DIR.glob("test_note_generator*.py"),
        *TESTS_DIR.glob("test_retrieval*.py"),
        TESTS_DIR / "test_embedding_adapter.py",
        TESTS_DIR / "test_prompting.py",
        *(TESTS_DIR / "notes_chat").glob("test_*.py"),
    ]
)

REQUESTS_MOCK = re.compile(r"requests\.(post|get|put|delete)")
OLLAMA_SHAPE = re.compile(r"ollama|/api/generate|/api/embeddings", re.IGNORECASE)
MLX_IMPORT = re.compile(r"mlx_lm")


def _violations(pattern: re.Pattern[str]) -> list[str]:
    found = []
    for path in MIGRATED_FILES:
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                found.append(
                    f"{path.relative_to(TESTS_DIR.parent)}:{lineno}: {line.strip()}"
                )
    return found


def test_migrated_files_exist():
    assert len(MIGRATED_FILES) >= 10, MIGRATED_FILES


def test_no_requests_mocks_for_migrated_llm_paths():
    assert _violations(REQUESTS_MOCK) == []


def test_no_ollama_shaped_fixtures():
    assert _violations(OLLAMA_SHAPE) == []


def test_mlx_is_never_mocked_in_migrated_tests():
    assert _violations(MLX_IMPORT) == []
