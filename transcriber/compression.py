import gzip
import json
from pathlib import Path
from typing import Any


class JSONCompressor:
    @staticmethod
    def compress_json(data: dict[str, Any], output_path: Path) -> bool:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            json_bytes = json_str.encode("utf-8")

            with gzip.open(output_path, "wb") as f:
                f.write(json_bytes)

            return True
        except Exception:
            return False

    @staticmethod
    def decompress_json(input_path: Path) -> dict[str, Any]:
        if not input_path.exists():
            raise FileNotFoundError(f"Compressed file not found: {input_path}")

        try:
            with gzip.open(input_path, "rb") as f:
                json_bytes = f.read()

            json_str = json_bytes.decode("utf-8")
            return json.loads(json_str)
        except Exception as e:
            raise RuntimeError(f"Failed to decompress JSON file: {str(e)}")

    @staticmethod
    def get_compression_ratio(original_size: int, compressed_size: int) -> float:
        if original_size == 0:
            return 0.0
        return compressed_size / original_size

    @staticmethod
    def get_file_sizes(json_path: Path) -> dict[str, int]:
        if not json_path.exists():
            return {"original": 0, "compressed": 0}

        try:
            with open(json_path, encoding="utf-8") as f:
                original_size = len(f.read().encode("utf-8"))
        except Exception:
            original_size = 0

        try:
            compressed_size = json_path.stat().st_size
        except Exception:
            compressed_size = 0

        return {"original": original_size, "compressed": compressed_size}
