"""Suite-hygiene invariants for the chirpd cutover (story 6.5, AC-11).

The LLM-touching tests migrated off Ollama-shaped fixtures in stories 6.2-6.5
must stay migrated: no `requests` mocks simulating LLM/embedding calls, no
Ollama API references, no `mlx_lm` mocks (use `FakeBackend` at the
`LLMBackend` boundary instead). `tests/test_prompting.py` is the one carve-out
while its deferred helpers remain on Ollama (story 6.5 AC-8), scoped per rule:

- `requests` mocks: allowed only on lines tagged `TODO(EPIC-INIT-AND-MIGRATION)`
  (inline or the line above the `@patch` decorator).
- Ollama references: the whole file is exempt until EPIC-INIT-AND-MIGRATION —
  the deferred helpers' imports, test names, and assertion strings necessarily
  mention Ollama on lines that carry no tag.
- `mlx_lm`: no exemption anywhere.
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

DEFERRED_TAG = "TODO(EPIC-INIT-AND-MIGRATION)"
DEFERRED_FILE = TESTS_DIR / "test_prompting.py"

REQUESTS_MOCK = re.compile(r"requests\.(post|get|put|delete)")
OLLAMA_SHAPE = re.compile(r"ollama|/api/generate|/api/embeddings")
MLX_IMPORT = re.compile(r"mlx_lm")


def _violations(pattern: re.Pattern[str], allow_deferred_tag: bool) -> list[str]:
    found = []
    for path in MIGRATED_FILES:
        lines = path.read_text().splitlines()
        for lineno, line in enumerate(lines, start=1):
            if not pattern.search(line):
                continue
            # A deferred-helper line is tagged inline or on the line above it
            # (the tag sits above `@patch("requests....")` decorators).
            tagged = DEFERRED_TAG in line or (
                lineno >= 2 and DEFERRED_TAG in lines[lineno - 2]
            )
            if allow_deferred_tag and path == DEFERRED_FILE and tagged:
                continue
            found.append(
                f"{path.relative_to(TESTS_DIR.parent)}:{lineno}: {line.strip()}"
            )
    return found


def test_migrated_files_exist():
    assert len(MIGRATED_FILES) >= 10, MIGRATED_FILES


def test_no_requests_mocks_for_migrated_llm_paths():
    # AC-1: `requests` may only simulate LLM calls on lines carrying the
    # deferred-helper tag in test_prompting.py.
    assert _violations(REQUESTS_MOCK, allow_deferred_tag=True) == []


def test_no_ollama_shaped_fixtures():
    # AC-2: test_prompting.py is wholly exempt (not just tagged lines) — its
    # deferred helpers' imports, test names, and assertion strings mention
    # Ollama on untagged lines until EPIC-INIT-AND-MIGRATION retires them.
    violations = [
        v
        for v in _violations(OLLAMA_SHAPE, allow_deferred_tag=True)
        if not v.startswith("tests/test_prompting.py")
    ]
    assert violations == []


def test_mlx_is_never_mocked_in_migrated_tests():
    # AC-10: unit tests fake at the LLMBackend boundary, never at mlx_lm.
    assert _violations(MLX_IMPORT, allow_deferred_tag=False) == []
