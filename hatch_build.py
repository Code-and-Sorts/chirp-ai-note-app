"""Hatchling build hook: bundle CaptureAudio.app and platform-tag the wheel.

The Swift helper is a build artifact (gitignored).  A plain force-include
entry errors at install time in environments that haven't run the Swift
build step (CI, dev installs, etc.).  This hook adds the bundle
conditionally so the wheel works when built locally after `make build`,
and installs cleanly otherwise.

When the bundle is present, the wheel ships a macOS-arm64 Mach-O executable
and therefore is not pure-Python: the hook marks it non-purelib and tags it
`py3-none-macosx_13_0_arm64` so pip refuses it on Linux/Windows/Intel (and on
macOS 11/12) instead of installing a binary that can only fail at runtime.
"""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        app_src = Path(__file__).parent / "audio_capture" / "CaptureAudio.app"
        if app_src.is_dir():
            build_data["force_include"]["audio_capture/CaptureAudio.app"] = (
                "audio_capture/CaptureAudio.app"
            )
            # py3-none keeps the wheel CPython-version-agnostic (the helper is a
            # subprocess, not a C extension); macosx_13_0 matches the helper's
            # build target and the macOS 13+ runtime requirement, so pip refuses
            # the wheel on macOS 11/12 rather than installing a binary that can't run.
            build_data["pure_python"] = False
            build_data["tag"] = "py3-none-macosx_13_0_arm64"
