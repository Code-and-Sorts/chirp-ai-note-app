"""Hatchling build hook: include CaptureAudio.app only when it exists.

The Swift helper is a build artifact (gitignored).  A plain force-include
entry errors at install time in environments that haven't run the Swift
build step (CI, dev installs, etc.).  This hook adds the bundle
conditionally so the wheel works when built locally after `make build`,
and installs cleanly otherwise.
"""

from __future__ import annotations

import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        app_src = os.path.join(
            os.path.dirname(__file__),
            "audio_capture",
            "CaptureAudio.app",
        )
        if os.path.isdir(app_src):
            build_data["force_include"]["audio_capture/CaptureAudio.app"] = (
                "audio_capture/CaptureAudio.app"
            )
