"""Build the bundled Swift `capture_audio` helper.

Usage: ``python -m audio_capture.build``

Verifies that ``swiftc`` is available at Swift 5.9 or newer, then shells out
to the Makefile in ``audio_capture/swift`` to produce ``CaptureAudio.app``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

MIN_SWIFT_MAJOR = 5
MIN_SWIFT_MINOR = 9
SWIFT_DIR = Path(__file__).parent / "swift"


def _run(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError as exc:
        return 127, str(exc)


def _swift_version_ok() -> tuple[bool, str]:
    code, output = _run(["swiftc", "--version"])
    if code != 0:
        return False, output
    match = re.search(r"Swift version (\d+)\.(\d+)", output)
    if not match:
        return False, output
    major = int(match.group(1))
    minor = int(match.group(2))
    if (major, minor) < (MIN_SWIFT_MAJOR, MIN_SWIFT_MINOR):
        return False, output
    return True, output


def main() -> int:
    ok, version_output = _swift_version_ok()
    if not ok:
        sys.stderr.write(
            "xcode command-line tools missing or out of date — run "
            "`xcode-select --install` (Swift 5.9+ required)\n"
        )
        if version_output:
            sys.stderr.write(version_output)
        return 1

    env = os.environ.copy()
    arch = env.get("ARCH")
    make_args = ["make", "-C", str(SWIFT_DIR), "build"]
    if arch:
        make_args.append(f"ARCH={arch}")

    proc = subprocess.run(make_args, env=env)
    if proc.returncode != 0:
        return proc.returncode

    bundle = Path(__file__).parent / "CaptureAudio.app"
    binary = bundle / "Contents" / "MacOS" / "capture_audio"

    # Replace the linker-applied ad-hoc signature with a proper bundle-level
    # ad-hoc signature so Info.plist is bound and the bundle identifier
    # (com.codeandsorts.chirp.capture-audio) is the code identity TCC uses.
    # Without this, macOS attributes screen-recording / mic prompts to the
    # parent process (Terminal/IDE) and Chirp never appears in System
    # Settings → Privacy & Security.
    sign_proc = subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(bundle)]
    )
    if sign_proc.returncode != 0:
        sys.stderr.write("codesign failed; the bundle is unsigned\n")
        return sign_proc.returncode

    sys.stdout.write(f"built: {binary}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
