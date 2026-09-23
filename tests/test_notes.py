"""Tests for ingest/notes.py"""

from pathlib import Path

import pytest

from ingest.notes import _is_note


class TestIsNote:
    @pytest.mark.parametrize(
        "path_str",
        [
            "10-projects/foo.md",
            "20-areas/bar/baz.md",
            "30-resources/x/y/z.md",
        ],
    )
    def test_markdown_in_note_dirs_is_note(self, path_str):
        assert _is_note(Path(path_str)) is True

    @pytest.mark.parametrize(
        "path_str",
        [
            "00-inbox/scratch.md",
            "90-daily/2024-01-01.md",
            "_templates/note.md",
        ],
    )
    def test_markdown_in_skip_dirs_is_not_note(self, path_str):
        assert _is_note(Path(path_str)) is False

    def test_non_markdown_in_note_dir_is_not_note(self):
        assert _is_note(Path("10-projects/notes.txt")) is False

    def test_markdown_outside_any_known_dir_is_not_note(self):
        assert _is_note(Path("40-other/file.md")) is False

    def test_bare_filename_no_parent_dir_is_not_note(self):
        assert _is_note(Path("README.md")) is False


class TestIngestNotesNoFiles:
    def test_no_changed_files_logs_and_skips(self, mocker, caplog):
        import logging

        mocker.patch("ingest.notes._changed_files", return_value=[])
        mocker.patch("ingest.notes.head_sha", return_value="abc1234")
        mock_upsert = mocker.patch("ingest.notes.upsert")

        from ingest.notes import ingest_notes

        with caplog.at_level(logging.DEBUG, logger="ingest.notes"):
            ingest_notes()

        assert "no curated notes changed" in caplog.text
        mock_upsert.assert_not_called()


class TestIngestNotesFileArg:
    def test_nonexistent_explicit_file_is_skipped(self, mocker, tmp_path):
        missing = tmp_path / "10-projects" / "ghost.md"  # never created
        mocker.patch("ingest.notes.assert_safe_path", return_value=missing)
        mocker.patch("ingest.notes.head_sha", return_value="abc")
        mock_upsert = mocker.patch("ingest.notes.upsert")
        mock_ids = mocker.patch("ingest.notes.ids_by_payload")

        from ingest.notes import ingest_notes

        ingest_notes(file=str(missing))

        mock_upsert.assert_not_called()
        mock_ids.assert_not_called()

    def test_existing_explicit_file_is_ingested(self, mocker, tmp_path):
        note = tmp_path / "10-projects" / "real.md"
        note.parent.mkdir(parents=True)
        note.write_text("## Section\n" + "content " * 20)

        mocker.patch("ingest.notes.assert_safe_path", return_value=note)
        mocker.patch("ingest.notes.head_sha", return_value="abc")
        mocker.patch("ingest.notes.ids_by_payload", return_value=[])
        mocker.patch("ingest.notes.distill_text", return_value="distilled")
        mock_upsert = mocker.patch("ingest.notes.upsert")

        from ingest.notes import ingest_notes

        ingest_notes(file=str(note))

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert len(points) > 0
        assert points[0]["file_path"] == str(note)
        assert points[0]["source"] == "obsidian-note"


class TestIngestNotesStaleCleanup:
    def test_stale_ids_not_in_new_points_are_deleted(self, mocker, tmp_path):
        note = tmp_path / "10-projects" / "shrink.md"
        note.parent.mkdir(parents=True)
        note.write_text("## Section\nshort content")  # → 1 chunk now

        stale_id = "stale-id-from-before"
        mocker.patch("ingest.notes._changed_files", return_value=[note])
        mocker.patch("ingest.notes.head_sha", return_value="abc")
        mocker.patch("ingest.notes.ids_by_payload", return_value=[stale_id])
        mocker.patch("ingest.notes.distill_text", return_value="distilled")
        mocker.patch("ingest.notes.upsert")
        mock_delete = mocker.patch("ingest.notes.delete_by_ids")

        from ingest.notes import ingest_notes

        ingest_notes()

        mock_delete.assert_called_once()
        deleted_ids = mock_delete.call_args[0][1]
        assert stale_id in deleted_ids

    def test_no_stale_ids_skips_delete(self, mocker, tmp_path):
        note = tmp_path / "10-projects" / "stable.md"
        note.parent.mkdir(parents=True)
        note.write_text("## Section\nsome content")

        mocker.patch("ingest.notes._changed_files", return_value=[note])
        mocker.patch("ingest.notes.head_sha", return_value="abc")
        mocker.patch("ingest.notes.ids_by_payload", return_value=[])
        mocker.patch("ingest.notes.distill_text", return_value="distilled")
        mocker.patch("ingest.notes.upsert")
        mock_delete = mocker.patch("ingest.notes.delete_by_ids")

        from ingest.notes import ingest_notes

        ingest_notes()

        mock_delete.assert_not_called()


class TestIngestNotesParallelDistillOrder:
    def test_distilled_order_matches_chunk_order(self, mocker, tmp_path):
        """ThreadPoolExecutor.map preserves input order regardless of which
        worker finishes first — verify chunk_index/section stay aligned with
        their own distilled text, not shuffled by concurrency."""
        note = tmp_path / "10-projects" / "multi.md"
        note.parent.mkdir(parents=True)
        # Two sections → two chunks, each big enough to survive chunk_markdown.
        note.write_text("## First\n" + "alpha " * 60 + "\n\n## Second\n" + "beta " * 60)

        mocker.patch("ingest.notes._changed_files", return_value=[note])
        mocker.patch("ingest.notes.head_sha", return_value="abc")
        mocker.patch("ingest.notes.ids_by_payload", return_value=[])
        mock_upsert = mocker.patch("ingest.notes.upsert")
        mocker.patch("ingest.notes.delete_by_ids")

        # Distill deterministically based on input content so we can verify
        # each output landed on the chunk it actually came from.
        def fake_distill(text, purpose):
            return f"summary-of[{text[:5]}]"

        mocker.patch("ingest.notes.distill_text", side_effect=fake_distill)

        from ingest.notes import ingest_notes

        ingest_notes()

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert len(points) == 2
        # chunk_index must be strictly increasing in the emitted point order
        # (i.e. distilled text order tracks chunk order, not completion order).
        sections = [p["section"] for p in points]
        assert sections == ["First", "Second"]
        assert points[0]["text"] == "summary-of[First]"
        assert points[1]["text"] == "summary-of[Secon]"
