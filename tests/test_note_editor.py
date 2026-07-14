from notes.note_editor import ManualNoteEditor


def test_status_line_marks_readonly():
    editor = ManualNoteEditor("Sample", "# Heading\n", readonly=True)

    status = editor._status_line(80)

    assert "read-only" in status


def test_readonly_blocks_insert(monkeypatch):
    monkeypatch.setattr("notes.note_editor.curses.beep", lambda: None)

    editor = ManualNoteEditor("Sample", "Content", readonly=True)
    assert editor.mode == "view"

    result = editor._handle_view_mode("i")

    assert result is False
    assert editor.mode == "view"
    assert editor._readonly_notified is True
    assert editor.message.startswith("Read-only note")
