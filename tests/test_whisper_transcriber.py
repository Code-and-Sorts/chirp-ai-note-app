import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config.settings import ChirpSettings
from transcriber.whisper_transcriber import WhisperTranscriber


class TestWhisperTranscriber:
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
    def mock_whisper_model(self):
        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = " This is a test transcription."
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.1
        mock_segment.words = None

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 5.0

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        return mock_model

    def test_initialization_loads_model(self, mock_settings):
        with patch(
            "transcriber.whisper_transcriber.WhisperModel"
        ) as mock_whisper_model_class:
            mock_model = Mock()
            mock_whisper_model_class.return_value = mock_model

            with patch.object(
                WhisperTranscriber, "_get_optimal_device", return_value="cpu"
            ):
                with patch.object(
                    WhisperTranscriber, "_get_compute_type", return_value="int8"
                ):
                    with patch.object(
                        WhisperTranscriber, "_get_cpu_threads", return_value=4
                    ):
                        transcriber = WhisperTranscriber(mock_settings)

                        assert transcriber.model == mock_model
                        assert transcriber.settings == mock_settings

    @patch("platform.system")
    @patch("platform.processor")
    def test_get_optimal_device_apple_silicon(
        self, mock_processor, mock_system, mock_settings
    ):
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1"

        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)
            device = transcriber._get_optimal_device()

            assert device == "cpu"

    def test_get_optimal_device_with_cuda(self, mock_settings):
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            with patch.object(WhisperTranscriber, "_get_cpu_threads", return_value=4):
                with patch.object(
                    WhisperTranscriber, "_get_compute_type", return_value="float16"
                ):
                    transcriber = WhisperTranscriber(mock_settings)

                    with patch("platform.system", return_value="Linux"):
                        with patch("platform.processor", return_value="Intel"):
                            mock_torch = Mock()
                            mock_torch.cuda.is_available.return_value = True

                            with patch.dict("sys.modules", {"torch": mock_torch}):
                                device = transcriber._get_optimal_device()
                                assert device == "cuda"

    def test_get_optimal_device_fallback_cpu(self, mock_settings):
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            with patch.object(WhisperTranscriber, "_get_cpu_threads", return_value=4):
                with patch.object(
                    WhisperTranscriber, "_get_compute_type", return_value="int8"
                ):
                    transcriber = WhisperTranscriber(mock_settings)

                    with patch("platform.system", return_value="Linux"):
                        with patch("platform.processor", return_value="Intel"):
                            with patch.dict("sys.modules", {}, clear=False):
                                if "torch" in sys.modules:
                                    del sys.modules["torch"]
                                device = transcriber._get_optimal_device()
                                assert device == "cpu"

    @patch("platform.system")
    @patch("platform.processor")
    @patch("os.cpu_count")
    def test_get_compute_type_apple_silicon(
        self, mock_cpu_count, mock_processor, mock_system, mock_settings
    ):
        mock_cpu_count.return_value = 8
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M2"

        with patch("transcriber.whisper_transcriber.WhisperModel"):
            with patch.object(
                WhisperTranscriber, "_get_optimal_device", return_value="cpu"
            ):
                transcriber = WhisperTranscriber(mock_settings)
                compute_type = transcriber._get_compute_type()

                assert compute_type == "int8"

    @patch("os.cpu_count")
    def test_get_compute_type_cuda_device(self, mock_cpu_count, mock_settings):
        mock_cpu_count.return_value = 8
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            with patch.object(
                WhisperTranscriber, "_get_optimal_device", return_value="cuda"
            ):
                transcriber = WhisperTranscriber(mock_settings)
                compute_type = transcriber._get_compute_type()

                assert compute_type == "float16"

    @patch("os.cpu_count")
    def test_get_cpu_threads_normal_case(self, mock_cpu_count, mock_settings):
        mock_cpu_count.return_value = 8

        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)
            threads = transcriber._get_cpu_threads()

            assert threads == 6

    @patch("os.cpu_count")
    def test_get_cpu_threads_none_fallback(self, mock_cpu_count, mock_settings):
        mock_cpu_count.return_value = None

        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)
            threads = transcriber._get_cpu_threads()

            assert threads == 1

    @patch("os.cpu_count")
    @patch("platform.processor")
    @patch("platform.system")
    def test_transcribe_file_file_not_found(
        self, mock_system, mock_processor, mock_cpu_count, mock_settings
    ):
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1"
        mock_cpu_count.return_value = 8
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)
            non_existent_file = Path("/non/existent/file.wav")

            with pytest.raises(FileNotFoundError, match="Audio file not found"):
                transcriber.transcribe_file(non_existent_file)

    @patch("os.cpu_count")
    @patch("platform.processor")
    @patch("platform.system")
    def test_transcribe_file_no_model_loaded(
        self, mock_system, mock_processor, mock_cpu_count, mock_settings
    ):
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1"
        mock_cpu_count.return_value = 8
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)
            transcriber.model = None

            test_file = Path("test.wav")
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(RuntimeError, match="Whisper model not loaded"):
                    transcriber.transcribe_file(test_file)

    @patch("os.cpu_count")
    @patch("platform.processor")
    @patch("platform.system")
    def test_transcribe_file_handles_transcription_error(
        self, mock_system, mock_processor, mock_cpu_count, mock_settings
    ):
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1"
        mock_cpu_count.return_value = 8
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)
            mock_model = Mock()
            mock_model.transcribe.side_effect = Exception("Transcription failed")
            transcriber.model = mock_model

            test_file = Path("test.wav")
            with patch("pathlib.Path.exists", return_value=True):
                result = transcriber.transcribe_file(test_file)

                assert result["success"] is False
                assert result["filename"] == "test.wav"
                assert result["full_text"] == ""
                assert result["segments"] == []
                assert result["error"] == "Transcription failed"
                assert "transcription_time" in result["metadata"]

    def test_transcribe_file_includes_enhanced_metadata(
        self, tmp_path, mock_settings, mock_whisper_model
    ):
        audio_path = tmp_path / "20250101_120000_sample.wav"
        audio_path.write_bytes(b"fake audio data")

        metadata_content = {
            "title": "Strategy Sync",
            "recorded_at": "2025-01-01T12:00:00",
        }
        metadata_file = audio_path.with_suffix(f"{audio_path.suffix}.meta")
        metadata_file.write_text(json.dumps(metadata_content), encoding="utf-8")

        with patch("transcriber.whisper_transcriber.WhisperModel") as mock_model_cls:
            mock_model_cls.return_value = mock_whisper_model
            with patch.object(
                WhisperTranscriber, "_get_optimal_device", return_value="cpu"
            ):
                with patch.object(
                    WhisperTranscriber, "_get_compute_type", return_value="int8"
                ):
                    with patch.object(
                        WhisperTranscriber, "_get_cpu_threads", return_value=4
                    ):
                        transcriber = WhisperTranscriber(mock_settings)

        result = transcriber.transcribe_file(audio_path)

        metadata = result["metadata"]

        assert metadata["recording_id"] == "20250101_120000"
        assert metadata["meeting_name"] == "Strategy Sync"
        assert metadata["title"] == "Strategy Sync"
        assert metadata["duration"] == pytest.approx(5.0)
        assert metadata["segment_count"] == 1
        assert metadata["word_count"] == len("This is a test transcription.".split())
        assert metadata["recording_datetime"].startswith("2025-01-01T12:00:00")
        assert result["full_text"] == "This is a test transcription."

    @patch("os.cpu_count")
    @patch("platform.processor")
    @patch("platform.system")
    def test_transcribe_batch_processes_multiple_files(
        self, mock_system, mock_processor, mock_cpu_count, mock_settings
    ):
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1"
        mock_cpu_count.return_value = 8
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)

            test_files = [Path("file1.wav"), Path("file2.wav")]

            with patch.object(transcriber, "transcribe_file") as mock_transcribe:
                mock_transcribe.return_value = {"success": True, "filename": "test.wav"}

                results = transcriber.transcribe_batch(test_files)

                assert len(results) == 2
                assert mock_transcribe.call_count == 2

    @patch("os.cpu_count")
    @patch("platform.processor")
    @patch("platform.system")
    def test_transcribe_batch_handles_individual_file_errors(
        self, mock_system, mock_processor, mock_cpu_count, mock_settings
    ):
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1"
        mock_cpu_count.return_value = 8
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)

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

    def test_get_model_info_returns_configuration(self, mock_settings):
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            with patch.object(
                WhisperTranscriber, "_get_optimal_device", return_value="cpu"
            ):
                with patch.object(
                    WhisperTranscriber, "_get_compute_type", return_value="int8"
                ):
                    with patch.object(
                        WhisperTranscriber, "_get_cpu_threads", return_value=4
                    ):
                        transcriber = WhisperTranscriber(mock_settings)

                        info = transcriber.get_model_info()

                        assert info["model_name"] == "small"
                        assert info["device"] == "cpu"
                        assert info["compute_type"] == "int8"
                        assert info["cpu_threads"] == 4
                        assert info["loaded"] is True

    @patch("os.cpu_count")
    @patch("platform.processor")
    @patch("platform.system")
    def test_get_model_info_when_model_not_loaded(
        self, mock_system, mock_processor, mock_cpu_count, mock_settings
    ):
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "Apple M1"
        mock_cpu_count.return_value = 8
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)
            transcriber.model = None

            info = transcriber.get_model_info()

            assert info["loaded"] is False

    def test_read_audio_metadata_file_exists(self, mock_settings):
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)

            test_audio_file = Path("test_audio.wav")
            test_metadata = {
                "title": "Test Meeting Title",
                "recorded_at": "2025-09-17T14:30:00",
                "channels": 2,
                "sample_rate": 16000,
            }

            with patch("builtins.open", create=True):
                with patch("json.load", return_value=test_metadata):
                    with patch.object(Path, "exists", return_value=True):
                        result = transcriber._read_audio_metadata(test_audio_file)

                        assert result == test_metadata
                        assert result["title"] == "Test Meeting Title"

    def test_read_audio_metadata_file_not_exists(self, mock_settings):
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)

            test_audio_file = Path("test_audio.wav")

            with patch.object(Path, "exists", return_value=False):
                result = transcriber._read_audio_metadata(test_audio_file)

                assert result is None

    def test_read_audio_metadata_file_corrupted(self, mock_settings):
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)

            test_audio_file = Path("test_audio.wav")

            with patch("builtins.open", create=True):
                with patch("json.load", side_effect=Exception("JSON decode error")):
                    with patch.object(Path, "exists", return_value=True):
                        result = transcriber._read_audio_metadata(test_audio_file)

                        assert result is None

    def test_metadata_includes_title_when_provided(self, mock_settings):
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)

            test_metadata = {
                "title": "Important Meeting",
                "recorded_at": "2025-09-17T14:30:00",
            }

            with patch.object(
                transcriber, "_read_audio_metadata", return_value=test_metadata
            ):
                metadata = transcriber._read_audio_metadata(Path("test.wav"))
                assert metadata["title"] == "Important Meeting"

    def test_metadata_returns_none_when_no_title(self, mock_settings):
        with patch("transcriber.whisper_transcriber.WhisperModel"):
            transcriber = WhisperTranscriber(mock_settings)

            with patch.object(transcriber, "_read_audio_metadata", return_value=None):
                metadata = transcriber._read_audio_metadata(Path("test.wav"))
                assert metadata is None

    def test_transcribe_passes_vad_parameters(
        self, mock_settings, mock_whisper_model, tmp_path
    ):
        audio_path = tmp_path / "20250101_120000_test.wav"
        audio_path.write_bytes(b"fake audio data")

        with patch(
            "transcriber.whisper_transcriber.WhisperModel",
            return_value=mock_whisper_model,
        ):
            with patch.object(
                WhisperTranscriber, "_get_optimal_device", return_value="cpu"
            ):
                with patch.object(
                    WhisperTranscriber, "_get_compute_type", return_value="int8"
                ):
                    with patch.object(
                        WhisperTranscriber, "_get_cpu_threads", return_value=4
                    ):
                        transcriber = WhisperTranscriber(mock_settings)

        transcriber.transcribe_file(audio_path)

        mock_whisper_model.transcribe.assert_called_once()
        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is True
        assert call_kwargs["vad_parameters"] == {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 1000,
            "max_speech_duration_s": 30,
            "speech_pad_ms": 300,
        }
