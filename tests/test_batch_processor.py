import json
from unittest.mock import Mock, patch

from transcriber.batch_processor import BatchProcessor


def test_process_single_file_writes_metadata(tmp_path):
    settings = Mock()
    settings.directories = Mock()
    settings.directories.transcriptions = tmp_path

    transcription_result = {
        "success": True,
        "metadata": {
            "recording_id": "20250101_120000",
            "meeting_name": "Strategy Sync",
        },
    }

    mock_transcriber = Mock()
    mock_transcriber.transcribe_file.return_value = transcription_result

    mock_compressor = Mock()
    mock_compressor.compress_json.return_value = True

    with patch(
        "transcriber.batch_processor.WhisperTranscriber", return_value=mock_transcriber
    ):
        with patch("transcriber.batch_processor.PopupManager"):
            processor = BatchProcessor(settings)

    processor.compressor = mock_compressor

    audio_file = tmp_path / "20250101_120000.wav"
    audio_file.write_bytes(b"audio data")

    result = processor._process_single_file(audio_file)

    expected_dir = tmp_path / "20250101_120000"
    expected_output = expected_dir / "20250101_120000.json.gz"
    metadata_path = expected_dir / "metadata.json"

    mock_compressor.compress_json.assert_called_once_with(
        transcription_result, expected_output
    )

    assert metadata_path.exists()
    assert (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        == transcription_result["metadata"]
    )
    assert result["metadata_path"] == str(metadata_path)
    assert result["output_path"] == str(expected_output)
