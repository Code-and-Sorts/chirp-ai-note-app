from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from utils.file_utils import (
    generate_audio_filename,
    get_audio_files,
    get_file_size_mb,
    sanitize_filename,
)


class TestFileUtils:
    def test_generate_audio_filename_with_title(self):
        with patch("utils.file_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20231201_140000"

            result = generate_audio_filename("Test Meeting", "wav")

            assert result == "20231201_140000_Test Meeting.wav"

    def test_generate_audio_filename_without_title(self):
        with patch("utils.file_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20231201_140000"

            result = generate_audio_filename()

            assert result == "20231201_140000.wav"

    def test_sanitize_filename_removes_invalid_chars(self):
        filename = "Test<File>Name:With/Invalid\\Chars|?*"
        result = sanitize_filename(filename)

        assert result == "TestFileNameWithInvalidChars"

    def test_sanitize_filename_truncates_long_names(self):
        long_filename = "A" * 100
        result = sanitize_filename(long_filename)

        assert len(result) <= 50

    def test_get_audio_files_finds_audio_extensions(self):
        mock_directory = Mock()
        mock_directory.exists.return_value = True

        # Mock files with different extensions
        mock_files = [
            Mock(is_file=lambda: True, suffix=".wav", stat=lambda: Mock(st_mtime=1)),
            Mock(is_file=lambda: True, suffix=".mp3", stat=lambda: Mock(st_mtime=2)),
            Mock(is_file=lambda: True, suffix=".txt", stat=lambda: Mock(st_mtime=3)),
        ]
        mock_directory.iterdir.return_value = mock_files

        result = get_audio_files(mock_directory)

        assert len(result) == 2  # Only .wav and .mp3 files

    def test_get_audio_files_empty_directory(self):
        mock_directory = Mock()
        mock_directory.exists.return_value = False

        result = get_audio_files(mock_directory)

        assert result == []

    def test_get_file_size_mb_existing_file(self):
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_size = 1024 * 1024  # 1 MB

        result = get_file_size_mb(mock_path)

        assert result == 1.0

    def test_get_file_size_mb_nonexistent_file(self):
        mock_path = Mock()
        mock_path.exists.return_value = False

        result = get_file_size_mb(mock_path)

        assert result == 0.0
