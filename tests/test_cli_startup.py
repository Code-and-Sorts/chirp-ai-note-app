"""Test that CLI module imports quickly without loading heavy dependencies."""

import sys


def test_cli_import_does_not_load_pyaudio():
    """Verify that importing chirp.cli does not trigger pyaudio import."""
    modules_before = set(sys.modules.keys())

    if "chirp.cli" in sys.modules:
        del sys.modules["chirp.cli"]

    import chirp.cli  # noqa: F401

    heavy_modules = {"pyaudio", "faster_whisper", "chromadb", "torch"}
    loaded_heavy = heavy_modules & (set(sys.modules.keys()) - modules_before)

    assert not loaded_heavy, (
        f"Heavy modules loaded at import time: {loaded_heavy}. "
        "Move these to function-level imports."
    )
