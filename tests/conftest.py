"""Shared test fixtures and configuration for the chirp test suite."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from unittest import mock

import freezegun
import pytest

# Stub `pyaudio` before any recorder modules are imported so that
# `recorder.device_manager` (which does `import pyaudio` at module level)
# can be safely imported on hosts that don't have PortAudio installed.
# Using MagicMock so tests that patch recorder.device_manager.pyaudio.PyAudio
# can set attributes on the stub without AttributeError.
if "pyaudio" not in sys.modules:
    sys.modules["pyaudio"] = mock.MagicMock()

# freezegun walks every imported module's __dir__ on entry. transformers v5
# has a lazy __dir__ that imports submodules unconditionally, and some of
# those submodules raise NameError at class-body evaluation time. Ignore
# transformers so freezegun never triggers that lazy load.
freezegun.configure(extend_ignore_list=["transformers"])


@pytest.fixture(autouse=True)
def _force_darwin_platform(request: pytest.FixtureRequest) -> Iterator[None]:
    """Patch sys.platform and platform.mac_ver so macOS version checks pass.

    Tests decorated with @pytest.mark.real_platform opt out and receive
    the host's real platform values.
    """
    if "real_platform" in request.keywords:
        yield
        return
    with (
        mock.patch.object(sys, "platform", "darwin"),
        mock.patch(
            "audio_capture.platform.mac_ver",
            return_value=("13.0.0", ("", "", ""), ""),
        ),
    ):
        yield
