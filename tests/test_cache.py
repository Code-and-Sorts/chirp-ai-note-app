import json
import time
from unittest.mock import patch

from notes_chat.cache import cache_answer, clear_cache, get_cached_answer


class TestCache:
    def test_cache_key_stability(self):
        """Test that cache keys are stable and order-independent."""
        question = "What was decided in the meeting?"
        ids1 = ["chunk1", "chunk2", "chunk3"]
        ids2 = ["chunk3", "chunk1", "chunk2"]

        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir.return_value = "/tmp"

            from notes_chat.cache import _generate_cache_key

            key1 = _generate_cache_key(question, ids1)
            key2 = _generate_cache_key(question, ids2)

            assert key1 == key2
            assert len(key1) == 16

    def test_cache_hit_miss(self, tmp_path):
        """Test cache hit and miss scenarios."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path

            question = "What was discussed?"
            retrieved_ids = ["chunk1", "chunk2"]
            answer = "We discussed the project timeline."

            cached = get_cached_answer(question, retrieved_ids)
            assert cached is None

            success = cache_answer(question, retrieved_ids, answer)
            assert success

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

            cache_answer(question1, retrieved_ids, answer1)
            cache_answer(question2, retrieved_ids, answer2)

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

            cache_answer(question, ids1, answer1)
            cache_answer(question, ids2, answer2)

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

            cache_files = list((tmp_path / "cache").glob("*.json"))
            assert len(cache_files) == 1

            with cache_files[0].open() as f:
                data = json.load(f)

            assert data["question"] == question
            assert data["retrieved_ids"] == sorted(retrieved_ids)
            assert data["answer"] == answer
            assert "cache_key" in data

    def test_cache_clear(self, tmp_path):
        """Test cache clearing functionality."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path

            cache_answer("question1", ["chunk1"], "answer1")
            cache_answer("question2", ["chunk2"], "answer2")

            cache_dir = tmp_path / "cache"
            assert len(list(cache_dir.glob("*.json"))) == 2

            success = clear_cache()
            assert success

            assert len(list(cache_dir.glob("*.json"))) == 0

            assert get_cached_answer("question1", ["chunk1"]) is None
            assert get_cached_answer("question2", ["chunk2"]) is None

    def test_error_handling(self, tmp_path):
        """Test error handling in cache operations."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.side_effect = Exception("Config error")

            assert get_cached_answer("question", ["chunk1"]) is None
            assert cache_answer("question", ["chunk1"], "answer") is False
            assert clear_cache() is False


class TestCacheKeying:
    def test_different_chat_model_changes_key(self):
        """A `chirp models default` chat switch must invalidate the key (AC-7)."""
        from notes_chat.cache import _generate_cache_key

        question = "What was decided?"
        ids = ["chunk1", "chunk2"]

        with patch(
            "notes_chat.cache._resolved_models", return_value=("model-a", "embed-x")
        ):
            key_a = _generate_cache_key(question, ids)
        with patch(
            "notes_chat.cache._resolved_models", return_value=("model-b", "embed-x")
        ):
            key_b = _generate_cache_key(question, ids)

        assert key_a != key_b

    def test_different_embed_model_changes_key(self):
        from notes_chat.cache import _generate_cache_key

        question = "What was decided?"
        ids = ["chunk1"]
        with patch(
            "notes_chat.cache._resolved_models", return_value=("model-a", "embed-x")
        ):
            key_x = _generate_cache_key(question, ids)
        with patch(
            "notes_chat.cache._resolved_models", return_value=("model-a", "embed-y")
        ):
            key_y = _generate_cache_key(question, ids)

        assert key_x != key_y

    def test_prompt_version_changes_key(self):
        """Bumping the prompt version must invalidate cached answers (AC-7)."""
        from notes_chat import cache as cache_module

        question = "What was decided?"
        ids = ["chunk1"]
        with patch(
            "notes_chat.cache._resolved_models", return_value=("model-a", "embed-x")
        ):
            with patch.object(cache_module, "PROMPT_VERSION", "1"):
                key_v1 = cache_module._generate_cache_key(question, ids)
            with patch.object(cache_module, "PROMPT_VERSION", "2"):
                key_v2 = cache_module._generate_cache_key(question, ids)

        assert key_v1 != key_v2

    def test_stale_answer_from_old_model_is_not_returned(self, tmp_path):
        """An answer cached under model-a is a miss after switching to model-b."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path
            question = "What was decided?"
            ids = ["chunk1"]

            with patch(
                "notes_chat.cache._resolved_models",
                return_value=("model-a", "embed-x"),
            ):
                cache_answer(question, ids, "old-model answer")
                assert get_cached_answer(question, ids) == "old-model answer"

            with patch(
                "notes_chat.cache._resolved_models",
                return_value=("model-b", "embed-x"),
            ):
                assert get_cached_answer(question, ids) is None


class TestCacheEviction:
    def test_ttl_expired_entry_is_a_miss(self, tmp_path):
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path
            cache_answer("q", ["c1"], "answer")

            with patch("notes_chat.cache.CACHE_TTL_SECONDS", -1):
                assert get_cached_answer("q", ["c1"]) is None
            assert list((tmp_path / "cache").glob("*.json")) == []

    def test_size_eviction_settles_at_cap_not_one(self, tmp_path):
        """Filling well past the cap settles the count AT the cap, not at 1.

        Regression for H2: `entries[:overflow]` with a negative overflow (when
        under the cap) is a negative slice that deletes almost everything, so the
        cache collapsed to a tiny size (e.g. 2). Each `cache_answer` after the
        first write would re-trigger that, leaving ~1 entry. The fix guards
        `overflow <= 0`.
        """
        cap = 10
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path
            with patch("notes_chat.cache.CACHE_MAX_ENTRIES", cap):
                for i in range(30):
                    cache_answer(f"q{i}", [f"c{i}"], f"a{i}")
                    time.sleep(0.005)

            files = list((tmp_path / "cache").glob("*.json"))
            # Settles at the cap, not collapsed to ~1 by the negative slice (H2).
            assert len(files) == cap

    def test_under_cap_does_not_evict(self, tmp_path):
        """When under the cap, nothing is evicted (the H2 negative-slice trap)."""
        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path
            with patch("notes_chat.cache.CACHE_MAX_ENTRIES", 500):
                for i in range(5):
                    cache_answer(f"q{i}", [f"c{i}"], f"a{i}")
                    time.sleep(0.005)

            files = list((tmp_path / "cache").glob("*.json"))
            assert len(files) == 5  # old negative-slice bug would leave ~2

    def test_force_reindex_clears_cache(self, tmp_path):
        """`chirp index --force` must invalidate the answer cache (AC-7)."""
        from typer.testing import CliRunner

        from notes_chat.cli import app

        with patch("notes_chat.cache.get_notes_config") as mock_config:
            mock_config.return_value.notes_chat.index_dir = tmp_path
            cache_answer("q1", ["c1"], "a1")
            cache_answer("q2", ["c2"], "a2")
            assert len(list((tmp_path / "cache").glob("*.json"))) == 2

        with (
            patch("notes_chat.cli.get_notes_config"),
            patch("notes_chat.index.build_index", return_value={"success": True}),
            patch("notes_chat.cache.get_notes_config") as cli_cache_config,
        ):
            cli_cache_config.return_value.notes_chat.index_dir = tmp_path
            CliRunner().invoke(app, ["index", "--force"])

        assert list((tmp_path / "cache").glob("*.json")) == []
