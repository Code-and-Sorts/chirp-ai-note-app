"""Quiet huggingface_hub's download progress bars and rate-limit warnings.

The tqdm progress bars print to stderr and bleed into the Rich Live pipeline
UI during `chirp transcribe`, and the "unauthenticated requests to the HF Hub"
warning is noise for local use. Both are silenced before model downloads run.
Explicit downloads (`chirp models add`) don't construct a transcriber, so they
keep their progress output.
"""

from __future__ import annotations


def quiet_huggingface_output() -> None:
    try:
        from huggingface_hub.utils import disable_progress_bars
        from huggingface_hub.utils import logging as hf_logging
    except ImportError:
        return
    try:
        disable_progress_bars()
        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - cosmetic; never break transcription
        return
