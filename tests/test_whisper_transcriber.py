from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from config.settings import ChirpSettings
from transcriber.whisper_transcriber import (
    WhisperTranscriber,
    _resolve_model_repo,
)

MLX_IMPORT = "transcriber.whisper_transcriber._import_mlx_whisper"
LOAD_AUDIO_IMPORT = "transcriber.whisper_transcriber.load_audio"
_FAKE_AUDIO_SAMPLES = 80000


def _make_mlx_module(*, segments=None, language="en"):
    """Build a stand-in for the lazily-imported mlx_whisper module."""
    module = Mock()
    module.load_models.load_model.return_value = Mock(name="whisper_model")
    module.transcribe.return_value = {
        "text": "".join(s.get("text", "") for s in (segments or [])),
        "language": language,
        "segments": segments if segments is not None else [],
    }
    return module


class TestWhisperTranscriber:
    @pytest.fixture(autouse=True)
    def _stub_load_audio(self):
        with patch(
            LOAD_AUDIO_IMPORT,
            return_value=np.zeros(_FAKE_AUDIO_SAMPLES, dtype=np.float32),
        ):
            yield

    @pytest.fixture
    def mock_settings(self):
        settings = Mock(spec=ChirpSettings)
        models = Mock()
        models.whisper = "small"
        models.num_predict = 4096
        settings.models = models
        audio = Mock()
        audio.vad_enabled = True
        vad_params = Mock()
        vad_params.model_dump.return_value = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 1000,
            "max_speech_duration_s": 30,
            "speech_pad_ms": 300,
        }
        audio.vad_parameters = vad_params
        settings.audio = audio
        return settings

    @pytest.fixture
    def mlx_module(self):
        return _make_mlx_module(
            segments=[
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": " This is a test transcription.",
                    "avg_logprob": -0.5,
                    "no_speech_prob": 0.1,
                }
            ]
        )

    @pytest.fixture
    def transcriber(self, mock_settings, mlx_module):
        with patch(MLX_IMPORT, return_value=mlx_module):
            instance = WhisperTranscriber(mock_settings)
            # Default to "no VAD regions" so transcribe tests never touch silero.
            with patch.object(instance, "_detect_speech", return_value=[]):
                yield instance

    def test_initialization_loads_model(self, mock_settings, mlx_module):
        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)

        assert transcriber.model is mlx_module.load_models.load_model.return_value
        assert transcriber.settings is mock_settings
        assert transcriber.model_repo == "mlx-community/whisper-small"

    def test_resolve_model_repo_maps_friendly_names(self):
        assert (
            _resolve_model_repo("large-v3-turbo")
            == "mlx-community/whisper-large-v3-turbo"
        )

    def test_resolve_model_repo_passes_through_org_repo(self):
        assert _resolve_model_repo("my-org/custom-whisper") == "my-org/custom-whisper"

    def test_get_optimal_device_apple_silicon(self, mock_settings, mlx_module):
        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)
        with patch(
            "transcriber.whisper_transcriber.platform.system", return_value="Darwin"
        ):
            assert transcriber._get_optimal_device() == "mlx"

    def test_get_optimal_device_non_darwin(self, mock_settings, mlx_module):
        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)
        with patch(
            "transcriber.whisper_transcriber.platform.system", return_value="Linux"
        ):
            assert transcriber._get_optimal_device() == "cpu"

    def test_get_compute_type_is_float16(self, transcriber):
        assert transcriber._get_compute_type() == "float16"

    def test_transcribe_file_file_not_found(self, transcriber):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.transcribe_file(Path("/non/existent/file.wav"))

    def test_transcribe_file_no_model_loaded(self, transcriber):
        transcriber.model = None
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="Whisper model not loaded"):
                transcriber.transcribe_file(Path("test.wav"))

    def test_transcribe_file_handles_transcription_error(self, transcriber, mlx_module):
        mlx_module.transcribe.side_effect = Exception("Transcription failed")
        with patch("pathlib.Path.exists", return_value=True):
            result = transcriber.transcribe_file(Path("test.wav"))

        assert result["success"] is False
        assert result["filename"] == "test.wav"
        assert result["full_text"] == ""
        assert result["segments"] == []
        assert result["error"] == "Transcription failed"
        assert "transcription_time" in result["metadata"]

    def test_transcribe_file_releases_mlx_cache(self, transcriber):
        with (
            patch("transcriber.whisper_transcriber._release_mlx_cache") as release,
            patch("pathlib.Path.exists", return_value=True),
        ):
            transcriber.transcribe_file(Path("test.wav"))

        release.assert_called_once()

    def test_transcribe_file_releases_mlx_cache_on_error(self, transcriber, mlx_module):
        mlx_module.transcribe.side_effect = Exception("Transcription failed")
        with (
            patch("transcriber.whisper_transcriber._release_mlx_cache") as release,
            patch("pathlib.Path.exists", return_value=True),
        ):
            transcriber.transcribe_file(Path("test.wav"))

        release.assert_called_once()

    def test_transcribe_file_includes_enhanced_metadata(
        self, tmp_path, mock_settings, mlx_module
    ):
        import tomli_w

        note_dir = tmp_path / "strategy-sync-2025-01-01"
        note_dir.mkdir()
        audio_path = note_dir / "audio.wav"
        audio_path.write_bytes(b"fake audio data")
        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump({"title": "Strategy Sync", "date": "2025-01-01T12:00:00"}, fh)

        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)
            with patch.object(transcriber, "_detect_speech", return_value=[]):
                result = transcriber.transcribe_file(audio_path)

        metadata = result["metadata"]
        assert metadata["meeting_name"] == "Strategy Sync"
        assert metadata["title"] == "Strategy Sync"
        assert metadata["duration"] == pytest.approx(5.0)
        assert metadata["language"] == "en"
        assert metadata["device"] == transcriber._get_optimal_device()
        assert metadata["compute_type"] == "float16"
        assert metadata["model_repo"] == "mlx-community/whisper-small"
        assert metadata["segment_count"] == 1
        assert metadata["word_count"] == 5
        assert metadata["recording_datetime"].startswith("2025-01-01T12:00:00")
        assert result["full_text"] == "This is a test transcription."

    def test_transcribe_file_maps_word_timestamps(self, tmp_path, mock_settings):
        audio_path = tmp_path / "20250101_120000_test.wav"
        audio_path.write_bytes(b"fake audio data")
        module = _make_mlx_module(
            segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Hello world",
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.05,
                    "words": [
                        {
                            "word": " Hello",
                            "start": 0.0,
                            "end": 0.5,
                            "probability": 0.9,
                        },
                        {
                            "word": " world",
                            "start": 0.5,
                            "end": 1.0,
                            "probability": 0.8,
                        },
                    ],
                }
            ]
        )

        with patch(MLX_IMPORT, return_value=module):
            transcriber = WhisperTranscriber(mock_settings)
            with patch.object(transcriber, "_detect_speech", return_value=[]):
                result = transcriber.transcribe_file(audio_path)

        words = result["segments"][0]["words"]
        assert [w["word"] for w in words] == [" Hello", " world"]
        assert words[0]["probability"] == 0.9

    def test_transcribe_batch_processes_multiple_files(self, transcriber):
        test_files = [Path("file1.wav"), Path("file2.wav")]
        with patch.object(transcriber, "transcribe_file") as mock_transcribe:
            mock_transcribe.return_value = {"success": True, "filename": "test.wav"}
            results = transcriber.transcribe_batch(test_files)

        assert len(results) == 2
        assert mock_transcribe.call_count == 2

    def test_transcribe_batch_handles_individual_file_errors(self, transcriber):
        test_files = [Path("file1.wav"), Path("file2.wav")]
        with patch.object(transcriber, "transcribe_file") as mock_transcribe:
            mock_transcribe.side_effect = [
                {"success": True, "filename": "file1.wav"},
                Exception("File processing failed"),
            ]
            results = transcriber.transcribe_batch(test_files)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[1]["error"] == "File processing failed"
        assert results[1]["filename"] == "file2.wav"

    def test_get_model_info_returns_configuration(self, transcriber):
        info = transcriber.get_model_info()
        assert info["model_name"] == "small"
        assert info["model_repo"] == "mlx-community/whisper-small"
        assert info["device"] == transcriber._get_optimal_device()
        assert info["compute_type"] == "float16"
        assert info["loaded"] is True

    def test_get_model_info_when_model_not_loaded(self, transcriber):
        transcriber.model = None
        assert transcriber.get_model_info()["loaded"] is False

    def test_read_audio_metadata_file_exists(self, tmp_path, transcriber):
        import tomli_w

        note_dir = tmp_path / "test-2025-09-17"
        note_dir.mkdir()
        audio_path = note_dir / "audio.wav"
        audio_path.write_bytes(b"")
        test_metadata = {"title": "Test Meeting Title", "date": "2025-09-17T14:30:00"}
        with (note_dir / "meta.toml").open("wb") as fh:
            tomli_w.dump(test_metadata, fh)

        result = transcriber._read_audio_metadata(audio_path)
        assert result == test_metadata

    def test_read_audio_metadata_file_not_exists(self, transcriber):
        with patch.object(Path, "exists", return_value=False):
            assert transcriber._read_audio_metadata(Path("test_audio.wav")) is None

    def test_vad_disabled_passes_whole_file(self, tmp_path, mock_settings, mlx_module):
        mock_settings.audio.vad_enabled = False
        audio_path = tmp_path / "20250101_120000_test.wav"
        audio_path.write_bytes(b"fake audio data")

        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)
            transcriber.transcribe_file(audio_path)

        assert mlx_module.transcribe.call_args.kwargs["clip_timestamps"] == "0"

    def test_vad_speech_regions_become_clip_timestamps(
        self, tmp_path, mock_settings, mlx_module
    ):
        audio_path = tmp_path / "20250101_120000_test.wav"
        audio_path.write_bytes(b"fake audio data")

        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)
            # 1.0s-3.0s of speech at 16 kHz.
            with patch.object(
                transcriber,
                "_detect_speech",
                return_value=[{"start": 16000, "end": 48000}],
            ):
                transcriber.transcribe_file(audio_path)

        assert mlx_module.transcribe.call_args.kwargs["clip_timestamps"] == [1.0, 3.0]

    def test_vad_no_speech_falls_back_to_whole_file(
        self, tmp_path, mock_settings, mlx_module
    ):
        audio_path = tmp_path / "20250101_120000_test.wav"
        audio_path.write_bytes(b"fake audio data")

        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)
            with patch.object(transcriber, "_detect_speech", return_value=[]):
                transcriber.transcribe_file(audio_path)

        assert mlx_module.transcribe.call_args.kwargs["clip_timestamps"] == "0"

    def test_detect_speech_passes_vad_parameters(self, mock_settings, mlx_module):
        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)

        get_speech_timestamps = Mock(return_value=[{"start": 0, "end": 16000}])
        with patch.dict(
            "sys.modules",
            {
                "silero_vad": Mock(
                    get_speech_timestamps=get_speech_timestamps,
                    load_silero_vad=Mock(return_value=Mock()),
                ),
                "torch": Mock(),
            },
        ):
            transcriber._detect_speech(np.zeros(16000, dtype=np.float32))

        call_kwargs = get_speech_timestamps.call_args.kwargs
        assert call_kwargs["sampling_rate"] == 16000
        assert call_kwargs["threshold"] == 0.5
        assert call_kwargs["min_speech_duration_ms"] == 250
        assert call_kwargs["speech_pad_ms"] == 300

    def test_detect_speech_converts_non_ndarray_audio_before_torch(
        self, mock_settings, mlx_module
    ):
        # mlx_whisper.audio.load_audio returns an mlx.core.array, not numpy.
        # torch.from_numpy only accepts an ndarray, so _detect_speech must
        # convert first. Reproduce the contract with a stand-in (real mlx/torch
        # cost ~5min on CI runners) so the regression stays guarded but fast.
        class _MlxLikeArray:
            def __init__(self, data: np.ndarray) -> None:
                self._data = data

            def __array__(self, dtype=None, copy=None) -> np.ndarray:
                return np.asarray(self._data, dtype=dtype)

        audio = _MlxLikeArray(np.zeros(16000, dtype=np.float32))
        captured: dict = {}

        def fake_from_numpy(arr):
            captured["arr"] = arr
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"expected np.ndarray (got {type(arr).__name__})")
            return arr

        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)

        with patch.dict(
            "sys.modules",
            {
                "torch": Mock(from_numpy=fake_from_numpy),
                "silero_vad": Mock(
                    get_speech_timestamps=Mock(
                        return_value=[{"start": 0, "end": 16000}]
                    ),
                    load_silero_vad=Mock(return_value=Mock()),
                ),
            },
        ):
            result = transcriber._detect_speech(audio)

        assert result == [{"start": 0, "end": 16000}]
        assert isinstance(captured["arr"], np.ndarray)

    def test_model_load_failure_raises_typed_actionable_error(self, mock_settings):
        from chirp.exceptions import WhisperModelLoadError

        module = _make_mlx_module()
        module.load_models.load_model.side_effect = OSError("connection reset")
        with patch(MLX_IMPORT, return_value=module):
            with pytest.raises(WhisperModelLoadError) as excinfo:
                WhisperTranscriber(mock_settings)

        message = str(excinfo.value)
        assert "Whisper model" in message
        assert "mlx-community/whisper-small" in message
        assert isinstance(excinfo.value.__cause__, OSError)

    def test_close_is_idempotent_and_nulls_model(self, mock_settings, mlx_module):
        with patch(MLX_IMPORT, return_value=mlx_module):
            transcriber = WhisperTranscriber(mock_settings)

        assert transcriber.model is not None
        transcriber.close()
        transcriber.close()
        assert transcriber.model is None
        assert transcriber._vad_model is None
