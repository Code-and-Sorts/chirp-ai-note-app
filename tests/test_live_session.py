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
            console=mock.MagicMock(),
            duration_minutes=None,
        )
        # Make `_wait_for_completion` exit immediately.
        session.stop_event.set()

        with pytest.raises(
            RecordingError, match="live capture failed mid-recording"
        ) as excinfo:
            session.run()

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "helper-boom-mid-stream" in str(excinfo.value.__cause__)
    fake_stream.save_recording.assert_not_called()
    assert list(tmp_path.iterdir()) == [], (
        "note dir should be cleaned up on capture failure"
    )


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
            console=mock.MagicMock(),
            duration_minutes=None,
            title="ok",
        )
        session.stop_event.set()
        result = session.run()

    fake_stream.save_recording.assert_called_once()
    assert result.audio_path.parent.parent == tmp_path


def test_run_does_not_call_saver_or_transcriber_on_capture_error(
    tmp_path: Path,
) -> None:
    # M18: when audio_stream.capture_error is set before (or during) run(),
    # run() must raise RecordingError and leave saver/transcriber uncalled.
    settings = _build_settings(tmp_path)
    fake_stream = _make_fake_stream(capture_error=RuntimeError("pre-set-capture-error"))
    saver = mock.MagicMock()
    transcriber_mock = mock.MagicMock()
    transcriber_mock.total_words = 0
    transcriber_mock.segments = []

    with (
        mock.patch("recorder.live_session.LiveAudioStream", return_value=fake_stream),
        mock.patch("recorder.live_session.LiveDashboard"),
        mock.patch(
            "recorder.live_session.LiveTranscriber", return_value=transcriber_mock
        ),
        mock.patch("recorder.vad_chunker.VADChunker"),
    ):
        session = LiveTranscriptionSession(
            settings=settings,
            console=mock.MagicMock(),
            duration_minutes=None,
            title="abort-test",
        )
        session.stop_event.set()

        with pytest.raises(RecordingError, match="live capture failed mid-recording"):
            session.run()

    fake_stream.save_recording.assert_not_called()
    saver.assert_not_called()
    transcriber_mock.export_transcript.assert_not_called()


def test_capture_not_started_when_model_load_fails(tmp_path: Path) -> None:
    # AC-2 ordering: the Whisper model is loaded before capture opens, so a
    # download/load failure surfaces WhisperModelLoadError without ever
    # starting the mic/screen-recording helper.
    from chirp.exceptions import WhisperModelLoadError

    settings = _build_settings(tmp_path)
    fake_stream = _make_fake_stream(capture_error=None)

    with (
        mock.patch("recorder.live_session.LiveAudioStream", return_value=fake_stream),
        mock.patch("recorder.live_session.LiveDashboard"),
        mock.patch(
            "recorder.live_session.LiveTranscriber",
            side_effect=WhisperModelLoadError("no network"),
        ),
        mock.patch("recorder.vad_chunker.VADChunker"),
    ):
        session = LiveTranscriptionSession(
            settings=settings,
            console=mock.MagicMock(),
            duration_minutes=None,
        )

        with pytest.raises(WhisperModelLoadError):
            session.run()

    fake_stream.start.assert_not_called()


def test_slow_final_pass_is_included_in_export(tmp_path: Path) -> None:
    import threading

    from recorder.live_types import TranscriptSegment

    settings = _build_settings(tmp_path)
    fake_stream = _make_fake_stream(capture_error=None)

    class SlowTranscriber(threading.Thread):
        def __init__(self, *args, **kwargs):
            super().__init__(daemon=True)
            self._segments: list[TranscriptSegment] = []
            self._lock = threading.Lock()
            self.total_words = 0
            self._unbounded_join = threading.Event()

        def run(self):
            if self._unbounded_join.wait(timeout=5):
                with self._lock:
                    self._segments.append(
                        TranscriptSegment(text="final", start=0.0, end=1.0, words=1)
                    )
                    self.total_words = 1

        def join(self, timeout=None):
            if timeout is None:
                self._unbounded_join.set()
            super().join(timeout)

        @property
        def segments(self):
            with self._lock:
                return list(self._segments)

        def export_transcript(self, path):
            segments = self.segments
            path.write_text("\n".join(s.text for s in segments), encoding="utf-8")

        def close(self):
            pass

    with (
        mock.patch("recorder.live_session.LiveAudioStream", return_value=fake_stream),
        mock.patch("recorder.live_session.LiveDashboard"),
        mock.patch("recorder.live_session.LiveTranscriber", SlowTranscriber),
        mock.patch("recorder.vad_chunker.VADChunker"),
    ):
        session = LiveTranscriptionSession(
            settings=settings,
            console=mock.MagicMock(),
            duration_minutes=None,
            title="slow",
        )
        session.stop_event.set()
        result = session.run()

    assert result.total_words == 1
    assert result.transcript_path is not None
    assert result.transcript_path.read_text(encoding="utf-8") == "final"


def test_drop_counters_surface_in_result(tmp_path: Path) -> None:
    # AC-4: queue-full drops counted by the producers are reported back through
    # the session result so the CLI can warn the user.
    settings = _build_settings(tmp_path)
    fake_stream = _make_fake_stream(capture_error=None)
    fake_stream.dropped_frames = 4

    fake_chunker = mock.MagicMock()
    fake_chunker.dropped_chunks = 7

    with (
        mock.patch("recorder.live_session.LiveAudioStream", return_value=fake_stream),
        mock.patch("recorder.live_session.LiveDashboard"),
        mock.patch("recorder.live_session.LiveTranscriber") as transcriber_cls,
        mock.patch("recorder.vad_chunker.VADChunker", return_value=fake_chunker),
    ):
        transcriber_cls.return_value.total_words = 0
        transcriber_cls.return_value.segments = []

        session = LiveTranscriptionSession(
            settings=settings,
            console=mock.MagicMock(),
            duration_minutes=None,
            title="drops",
        )
        session.stop_event.set()
        result = session.run()

    assert result.dropped_frames == 4
    assert result.dropped_chunks == 7


def test_transcriber_closed_during_cleanup(tmp_path: Path) -> None:
    # AC-5: the live session releases the Whisper model on the clean path.
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
            console=mock.MagicMock(),
            duration_minutes=None,
            title="cleanup",
        )
        session.stop_event.set()
        session.run()

    transcriber_cls.return_value.close.assert_called()
