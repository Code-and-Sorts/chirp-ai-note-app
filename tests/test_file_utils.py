import time
from datetime import date
from unittest.mock import Mock

import tomli_w

from utils.file_utils import (
    NoteRecord,
    get_file_size_mb,
    list_notes,
    sanitize_filename,
    slugify,
)


class TestSanitizeFilename:
    def test_removes_invalid_chars(self):
        assert (
            sanitize_filename("Test<File>Name:With/Invalid\\Chars|?*")
            == "TestFileNameWithInvalidChars"
        )

    def test_truncates_long_names(self):
        assert len(sanitize_filename("A" * 100)) <= 50


class TestSlugify:
    def test_produces_kebab_case_with_date(self):
        assert (
            slugify("Project Kickoff", date(2026, 4, 20))
            == "project-kickoff-2026-04-20"
        )

    def test_strips_punctuation(self):
        assert (
            slugify("Q2 Planning: Day 1!", date(2026, 4, 20))
            == "q2-planning-day-1-2026-04-20"
        )

    def test_appends_numeric_suffix_on_collision(self, tmp_path):
        (tmp_path / "standup-2026-04-20").mkdir()
        assert slugify("standup", date(2026, 4, 20), tmp_path) == "standup-2026-04-20-2"

        (tmp_path / "standup-2026-04-20-2").mkdir()
        assert slugify("standup", date(2026, 4, 20), tmp_path) == "standup-2026-04-20-3"

    def test_fallback_for_empty_title(self):
        assert slugify("   ", date(2026, 4, 20)) == "note-2026-04-20"

    def test_punctuation_only_falls_back_to_note(self):
        assert slugify("!!!", date(2026, 4, 20)) == "note-2026-04-20"

    def test_ascii_folds_accented_characters(self):
        assert slugify("Café résumé", date(2026, 4, 20)) == "cafe-resume-2026-04-20"

    def test_unicode_only_title_falls_back_to_note(self):
        assert slugify("💯💯💯", date(2026, 4, 20)) == "note-2026-04-20"


class TestListNotes:
    def test_returns_empty_when_root_missing(self, tmp_path):
        assert list_notes(tmp_path / "missing") == []

    def test_returns_empty_when_root_is_a_file(self, tmp_path):
        bogus = tmp_path / "not-a-dir"
        bogus.write_text("oops")
        assert list_notes(bogus) == []

    def test_returns_records_sorted_by_created_at(self, tmp_path):
        first_dir = tmp_path / "first-2026-04-20"
        first_dir.mkdir()
        _write_meta(first_dir, title="First", iso="2026-04-20T09:00:00")

        second_dir = tmp_path / "second-2026-04-20"
        second_dir.mkdir()
        _write_meta(second_dir, title="Second", iso="2026-04-20T10:00:00")

        records = list_notes(tmp_path)

        assert [record.slug for record in records] == [
            "first-2026-04-20",
            "second-2026-04-20",
        ]

    def test_falls_back_to_mtime_when_meta_missing(self, tmp_path):
        older_dir = tmp_path / "older"
        older_dir.mkdir()
        newer_dir = tmp_path / "newer"
        newer_dir.mkdir()

        older_time = time.time() - 100
        import os

        os.utime(older_dir, (older_time, older_time))

        records = list_notes(tmp_path)
        assert [record.slug for record in records] == ["older", "newer"]

    def test_populates_artifact_paths(self, tmp_path):
        note_dir = tmp_path / "team-standup-2026-04-20"
        note_dir.mkdir()
        (note_dir / "audio.wav").write_bytes(b"")
        (note_dir / "transcript.txt").write_text("hello", encoding="utf-8")
        (note_dir / "notes.md").write_text("# hi", encoding="utf-8")
        _write_meta(
            note_dir, title="Team Standup", iso="2026-04-20T09:00:00", tags=["ops"]
        )

        records = list_notes(tmp_path)

        assert len(records) == 1
        record = records[0]
        assert isinstance(record, NoteRecord)
        assert record.audio == note_dir / "audio.wav"
        assert record.transcript == note_dir / "transcript.txt"
        assert record.notes == note_dir / "notes.md"
        assert record.meta == note_dir / "meta.toml"
        assert record.tags == ["ops"]
        assert record.title == "Team Standup"

    def test_skips_hidden_directories(self, tmp_path):
        (tmp_path / ".debug-live").mkdir()
        (tmp_path / ".DS_Store").mkdir()
        visible_dir = tmp_path / "visible-2026-04-20"
        visible_dir.mkdir()
        _write_meta(visible_dir, title="Visible", iso="2026-04-20T09:00:00")

        records = list_notes(tmp_path)
        assert [record.slug for record in records] == ["visible-2026-04-20"]


class TestGetFileSizeMb:
    def test_existing_file(self):
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_size = 1024 * 1024
        assert get_file_size_mb(mock_path) == 1.0

    def test_missing_file(self):
        mock_path = Mock()
        mock_path.exists.return_value = False
        assert get_file_size_mb(mock_path) == 0.0


def _write_meta(note_dir, title, iso, tags=None):
    payload = {
        "title": title,
        "date": iso,
        "tags": tags or [],
    }
    with (note_dir / "meta.toml").open("wb") as fh:
        tomli_w.dump(payload, fh)
