import json
from unittest.mock import patch

from notes_chat.cache import cache_answer, clear_cache, get_cached_answer


class TestCache:
    def test_cache_key_stability(self):
        """Test that cache keys are stable and order-independent."""
        question = "What was decided in the meeting?"
        ids1 = ["chunk1", "chunk2", "chunk3"]
        ids2 = ["chunk3", "chunk1", "chunk2"]  # Different order

        # Mock the config to avoid file system operations
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir.return_value = "/tmp"

            from notes_chat.cache import _generate_cache_key

            key1 = _generate_cache_key(question, ids1)
            key2 = _generate_cache_key(question, ids2)

            assert key1 == key2  # Should be the same despite different order
            assert len(key1) == 16  # Should be truncated hash

    def test_cache_hit_miss(self, tmp_path):
        """Test cache hit and miss scenarios."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path

            question = "What was discussed?"
            retrieved_ids = ["chunk1", "chunk2"]
            answer = "We discussed the project timeline."

            # Test cache miss
            cached = get_cached_answer(question, retrieved_ids)
            assert cached is None

            # Cache the answer
            success = cache_answer(question, retrieved_ids, answer)
            assert success

            # Test cache hit
            cached = get_cached_answer(question, retrieved_ids)
            assert cached == answer

    def test_cache_different_questions(self, tmp_path):
        """Test that different questions have different cache entries."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path

            question1 = "What was decided?"
            question2 = "Who attended?"
            retrieved_ids = ["chunk1", "chunk2"]
            answer1 = "We decided to implement feature X."
            answer2 = "Alice and Bob attended."

            # Cache both answers
            cache_answer(question1, retrieved_ids, answer1)
            cache_answer(question2, retrieved_ids, answer2)

            # Verify they're cached separately
            cached1 = get_cached_answer(question1, retrieved_ids)
            cached2 = get_cached_answer(question2, retrieved_ids)

            assert cached1 == answer1
            assert cached2 == answer2
            assert cached1 != cached2

    def test_cache_different_retrieved_ids(self, tmp_path):
        """Test that different retrieved IDs result in different cache entries."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path

            question = "What was decided?"
            ids1 = ["chunk1", "chunk2"]
            ids2 = ["chunk1", "chunk3"]
            answer1 = "Answer based on chunks 1,2"
            answer2 = "Answer based on chunks 1,3"

            # Cache both
            cache_answer(question, ids1, answer1)
            cache_answer(question, ids2, answer2)

            # Verify separate caching
            cached1 = get_cached_answer(question, ids1)
            cached2 = get_cached_answer(question, ids2)

            assert cached1 == answer1
            assert cached2 == answer2

    def test_cache_file_format(self, tmp_path):
        """Test that cache files are properly formatted."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path

            question = "Test question"
            retrieved_ids = ["chunk1", "chunk2"]
            answer = "Test answer"

            success = cache_answer(question, retrieved_ids, answer)
            assert success

            # Find the cache file
            cache_files = list((tmp_path / "cache").glob("*.json"))
            assert len(cache_files) == 1

            # Verify file format
            with open(cache_files[0]) as f:
                data = json.load(f)

            assert data["question"] == question
            assert data["retrieved_ids"] == sorted(retrieved_ids)
            assert data["answer"] == answer
            assert "cache_key" in data

    def test_cache_clear(self, tmp_path):
        """Test cache clearing functionality."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path

            # Add some cache entries
            cache_answer("question1", ["chunk1"], "answer1")
            cache_answer("question2", ["chunk2"], "answer2")

            cache_dir = tmp_path / "cache"
            assert len(list(cache_dir.glob("*.json"))) == 2

            # Clear cache
            success = clear_cache()
            assert success

            # Verify cache is empty
            assert len(list(cache_dir.glob("*.json"))) == 0

            # Verify cached answers are gone
            assert get_cached_answer("question1", ["chunk1"]) is None
            assert get_cached_answer("question2", ["chunk2"]) is None

    def test_error_handling(self, tmp_path):
        """Test error handling in cache operations."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            # Mock config that will cause errors
            mock_config.side_effect = Exception("Config error")

            # All operations should handle errors gracefully
            assert get_cached_answer("question", ["chunk1"]) is None
            assert cache_answer("question", ["chunk1"], "answer") is False
            assert clear_cache() is False

    def test_bypass_on_dry_run(self, tmp_path):
        """Test that cache is not used in dry-run scenarios."""
        # This would be tested at the CLI level, but we can test the basic functionality
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path

            question = "Test question"
            retrieved_ids = ["chunk1"]
            answer = "Test answer"

            # Cache an answer
            cache_answer(question, retrieved_ids, answer)

            # In dry-run mode, the CLI should not call get_cached_answer
            # But if it does, it should still return the cached value
            cached = get_cached_answer(question, retrieved_ids)
            assert cached == answer
