import gzip
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from transcriber.compression import JSONCompressor


class TestCompressJson:
    def test_writes_gzip_file_and_returns_true(self, tmp_path):
        output_path = tmp_path / "output.json.gz"
        data = {"key": "value", "number": 42}

        result = JSONCompressor.compress_json(data, output_path)

        assert result is True
        assert output_path.exists()

    def test_compressed_file_is_valid_gzip(self, tmp_path):
        output_path = tmp_path / "output.json.gz"
        data = {"hello": "world"}

        JSONCompressor.compress_json(data, output_path)

        with gzip.open(output_path, "rb") as f:
            raw = f.read()

        assert json.loads(raw.decode("utf-8")) == data

    def test_creates_parent_directories(self, tmp_path):
        output_path = tmp_path / "nested" / "deep" / "output.json.gz"

        result = JSONCompressor.compress_json({"x": 1}, output_path)

        assert result is True
        assert output_path.parent.exists()

    def test_uses_compact_separators(self, tmp_path):
        output_path = tmp_path / "output.json.gz"
        data = {"a": 1, "b": 2}

        JSONCompressor.compress_json(data, output_path)

        with gzip.open(output_path, "rb") as f:
            raw = f.read().decode("utf-8")

        assert " " not in raw

    def test_preserves_non_ascii_characters(self, tmp_path):
        output_path = tmp_path / "output.json.gz"
        data = {"greeting": "こんにちは"}

        JSONCompressor.compress_json(data, output_path)

        with gzip.open(output_path, "rb") as f:
            result = json.loads(f.read().decode("utf-8"))

        assert result["greeting"] == "こんにちは"

    def test_returns_false_on_os_error(self, tmp_path):
        output_path = tmp_path / "output.json.gz"

        with patch("gzip.open", side_effect=OSError("disk full")):
            result = JSONCompressor.compress_json({"x": 1}, output_path)

        assert result is False

    def test_returns_false_when_mkdir_fails(self, tmp_path):
        output_path = tmp_path / "blocked" / "output.json.gz"

        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            result = JSONCompressor.compress_json({"x": 1}, output_path)

        assert result is False


class TestDecompressJson:
    def test_round_trips_data(self, tmp_path):
        path = tmp_path / "data.json.gz"
        original = {"name": "test", "values": [1, 2, 3]}

        JSONCompressor.compress_json(original, path)
        result = JSONCompressor.decompress_json(path)

        assert result == original

    def test_raises_file_not_found_when_missing(self, tmp_path):
        missing = tmp_path / "nonexistent.json.gz"

        with pytest.raises(FileNotFoundError, match="Compressed file not found"):
            JSONCompressor.decompress_json(missing)

    def test_raises_runtime_error_on_os_error(self, tmp_path):
        path = tmp_path / "bad.json.gz"
        path.write_bytes(b"not gzip data at all")

        with pytest.raises(RuntimeError, match="Failed to decompress JSON file"):
            JSONCompressor.decompress_json(path)

    def test_raises_runtime_error_on_invalid_json(self, tmp_path):
        path = tmp_path / "invalid.json.gz"
        with gzip.open(path, "wb") as f:
            f.write(b"not valid json {{{")

        with pytest.raises(RuntimeError, match="Failed to decompress JSON file"):
            JSONCompressor.decompress_json(path)

    def test_raises_runtime_error_on_unicode_decode_error(self, tmp_path):
        path = tmp_path / "bad_encoding.json.gz"
        with gzip.open(path, "wb") as f:
            f.write(b"\xff\xfe invalid utf-8")

        with pytest.raises(RuntimeError, match="Failed to decompress JSON file"):
            JSONCompressor.decompress_json(path)

    def test_returns_empty_dict_for_non_dict_json(self, tmp_path):
        path = tmp_path / "list.json.gz"
        with gzip.open(path, "wb") as f:
            f.write(json.dumps([1, 2, 3]).encode("utf-8"))

        result = JSONCompressor.decompress_json(path)

        assert result == {}

    def test_returns_dict_for_dict_json(self, tmp_path):
        path = tmp_path / "dict.json.gz"
        data = {"key": "val"}
        with gzip.open(path, "wb") as f:
            f.write(json.dumps(data).encode("utf-8"))

        result = JSONCompressor.decompress_json(path)

        assert result == data


class TestGetCompressionRatio:
    def test_returns_ratio(self):
        assert JSONCompressor.get_compression_ratio(1000, 400) == pytest.approx(0.4)

    def test_returns_zero_when_original_size_is_zero(self):
        assert JSONCompressor.get_compression_ratio(0, 400) == 0.0

    def test_returns_one_when_sizes_equal(self):
        assert JSONCompressor.get_compression_ratio(500, 500) == pytest.approx(1.0)

    def test_ratio_greater_than_one_allowed(self):
        assert JSONCompressor.get_compression_ratio(100, 200) == pytest.approx(2.0)


class TestGetFileSizes:
    def test_returns_zero_sizes_when_path_missing(self, tmp_path):
        missing = tmp_path / "ghost.json"

        result = JSONCompressor.get_file_sizes(missing)

        assert result == {"original": 0, "compressed": 0}

    def test_returns_original_size_as_utf8_byte_count(self, tmp_path):
        json_file = tmp_path / "data.json"
        content = '{"key": "value"}'
        json_file.write_text(content, encoding="utf-8")

        result = JSONCompressor.get_file_sizes(json_file)

        assert result["original"] == len(content.encode("utf-8"))

    def test_compressed_size_matches_stat(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text('{"a": 1}', encoding="utf-8")

        result = JSONCompressor.get_file_sizes(json_file)

        assert result["compressed"] == json_file.stat().st_size

    def test_original_size_zero_on_os_error_reading_file(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text("{}", encoding="utf-8")

        with patch("pathlib.Path.open", side_effect=OSError("read error")):
            result = JSONCompressor.get_file_sizes(json_file)

        assert result["original"] == 0

    def test_compressed_size_zero_on_stat_os_error(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text("{}", encoding="utf-8")

        original_stat = Path.stat

        call_count = {"n": 0}

        def stat_side_effect(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise OSError("stat error")
            return original_stat(self, *args, **kwargs)

        with patch.object(Path, "stat", stat_side_effect):
            result = JSONCompressor.get_file_sizes(json_file)

        assert result["compressed"] == 0

    def test_non_ascii_content_counted_in_utf8_bytes(self, tmp_path):
        json_file = tmp_path / "unicode.json"
        content = '{"emoji": "🐦"}'
        json_file.write_text(content, encoding="utf-8")

        result = JSONCompressor.get_file_sizes(json_file)

        assert result["original"] == len(content.encode("utf-8"))
