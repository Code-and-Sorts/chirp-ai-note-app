"""Test that CLI module imports quickly without loading heavy dependencies."""

import sys
from unittest.mock import patch


def test_cli_import_does_not_load_heavy_dependencies():
    """Verify that importing chirp.cli does not trigger heavy module imports (pyaudio, faster_whisper, chromadb, torch)."""
    heavy_modules = {"pyaudio", "faster_whisper", "chromadb", "torch"}

    with patch.dict(sys.modules, sys.modules.copy(), clear=True):
        if "chirp.cli" in sys.modules:
            del sys.modules["chirp.cli"]

        for name in list(sys.modules.keys()):
            for heavy in heavy_modules:
                if name == heavy or name.startswith(f"{heavy}."):
                    del sys.modules[name]
                    break

        import chirp.cli  # noqa: F401

        loaded_heavy = {
            name
            for name in sys.modules.keys()
            for heavy in heavy_modules
            if name == heavy or name.startswith(f"{heavy}.")
        }

        assert not loaded_heavy, (
            f"Heavy modules loaded at import time: {loaded_heavy}. "
            "Move these to function-level imports."
        )
