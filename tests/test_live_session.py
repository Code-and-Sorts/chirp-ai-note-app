"""Session-level tests for `recorder.live_session.LiveTranscriptionSession`.

These exercise the orchestration layer that sits on top of LiveAudioStream
(VAD chunker, transcriber, dashboard, capture_error propagation). Heavy
dependencies (LiveDashboard, LiveTranscriber, VADChunker, the real audio
stream) are mocked so the tests run on every platform without hardware.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from chirp.exceptions import RecordingError
from recorder.live_session import LiveTranscriptionSession


def _build_settings(notes_root: Path) -> mock.MagicMock:
    settings = mock.MagicMock()
    settings.directories.notes_root = notes_root
    settings.audio.sample_rate = 16000
    settings.audio.channels = 1
    return settings


def _make_fake_stream(capture_error: BaseException | None) -> mock.MagicMock:
    fake = mock.MagicMock()
    fake.capture_error = capture_error
    fake.mic_device_name = "MockMic"
    fake.sample_rate = 16000
    fake.channels = 1
    fake.frame_duration = 0.032
    fake.frames = []
    return fake


def test_run_raises_recording_error_when_audio_stream_capture_error_set(
    tmp_path: Path,
) -> None:
    # Mid-recording helper crash must surface as RecordingError before any
    # save/transcribe step runs — otherwise the live path would happily
    # produce a silently-truncated WAV plus a partial transcript.
    settings = _build_settings(tmp_path)
    fake_stream = _make_fake_stream(
        capture_error=RuntimeError("helper-boom-mid-stream")
    )

    with (
        mock.patch("recorder.live_session.LiveAudioStream", return_value=fake_stream),
        mock.patch("recorder.live_session.LiveDashboard"),
        mock.patch("recorder.live_session.LiveTranscriber"),
        mock.patch("recorder.vad_chunker.VADChunker"),
    ):
        session = LiveTranscriptionSession(
            settings=settings,
            device_manager=mock.MagicMock(),
            console=mock.MagicMock(),
            duration_minutes=None,
        )
        # Make `_wait_for_completion` exit immediately.
        session.stop_event.set()

        with pytest.raises(
            RecordingError, match="audio capture worker crashed mid-recording"
        ) as excinfo:
            session.run()

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "helper-boom-mid-stream" in str(excinfo.value.__cause__)
    fake_stream.save_recording.assert_not_called()


def test_run_succeeds_when_audio_stream_capture_error_is_none(
    tmp_path: Path,
) -> None:
    # Sanity counterpoint: a clean session (no capture_error) reaches
    # save_recording. Pins the contract that capture_error == None lets
    # the post-stop pipeline complete normally.
    settings = _build_settings(tmp_path)
    fake_stream = _make_fake_stream(capture_error=None)

    with (
        mock.patch("recorder.live_session.LiveAudioStream", return_value=fake_stream),
        mock.patch("recorder.live_session.LiveDashboard"),
        mock.patch("recorder.live_session.LiveTranscriber") as transcriber_cls,
        mock.patch("recorder.vad_chunker.VADChunker"),
    ):
        transcriber_cls.return_value.total_words = 0
        transcriber_cls.return_value.segments = []

        session = LiveTranscriptionSession(
            settings=settings,
            device_manager=mock.MagicMock(),
            console=mock.MagicMock(),
            duration_minutes=None,
            title="ok",
        )
        session.stop_event.set()
        result = session.run()

    fake_stream.save_recording.assert_called_once()
    assert result.audio_path.parent.parent == tmp_path
