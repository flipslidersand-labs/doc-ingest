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
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc1234")
        mock_upsert = mocker.patch("core.qdrant.upsert")

        from ingest.notes import ingest_notes

        with caplog.at_level(logging.DEBUG, logger="core.ingest_pipeline"):
            ingest_notes()

        assert "no curated notes changed" in caplog.text
        mock_upsert.assert_not_called()


class TestIngestNotesFileArg:
    def test_nonexistent_explicit_file_removes_its_points_and_ingests_nothing(
        self, mocker, tmp_path
    ):
        """#217: 存在しないファイル (= 削除・リネームされた) は、検索に出続けないよう point を消す。"""
        missing = tmp_path / "10-projects" / "ghost.md"  # never created
        mocker.patch("core.ingest_pipeline.assert_safe_path", return_value=missing)
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mocker.patch("core.qdrant.ids_by_payload", return_value=["p1", "p2"])
        mock_delete = mocker.patch("core.qdrant.delete_by_ids")

        from ingest.notes import ingest_notes

        ingest_notes(file=str(missing))

        mock_upsert.assert_not_called()
        mock_delete.assert_called_once_with("notes", ["p1", "p2"])

    def test_nonexistent_file_without_points_deletes_nothing(self, mocker, tmp_path):
        missing = tmp_path / "10-projects" / "ghost.md"
        mocker.patch("core.ingest_pipeline.assert_safe_path", return_value=missing)
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.upsert")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mock_delete = mocker.patch("core.qdrant.delete_by_ids")

        from ingest.notes import ingest_notes

        ingest_notes(file=str(missing))

        mock_delete.assert_not_called()

    def test_existing_explicit_file_is_ingested(self, mocker, tmp_path):
        note = tmp_path / "10-projects" / "real.md"
        note.parent.mkdir(parents=True)
        note.write_text("## Section\n" + "content " * 20)

        mocker.patch("core.ingest_pipeline.assert_safe_path", return_value=note)
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("ingest.notes.distill_text", return_value="distilled")
        mock_upsert = mocker.patch("core.qdrant.upsert")

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
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[stale_id])
        mocker.patch("ingest.notes.distill_text", return_value="distilled")
        mocker.patch("core.qdrant.upsert")
        mock_delete = mocker.patch("core.qdrant.delete_by_ids")

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
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("ingest.notes.distill_text", return_value="distilled")
        mocker.patch("core.qdrant.upsert")
        mock_delete = mocker.patch("core.qdrant.delete_by_ids")

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
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mocker.patch("core.qdrant.delete_by_ids")

        # Distill deterministically based on input content so we can verify
        # each output landed on the chunk it actually came from.
        def fake_distill(text, purpose, **kwargs):
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


class TestIngestNotesIsLocalOnly:
    def test_notes_are_distilled_with_local_only(self, mocker, tmp_path):
        """非公開ノートは Anthropic に送らない: distill_text は local_only=True で呼ばれる。"""
        note = tmp_path / "10-projects" / "private.md"
        note.parent.mkdir(parents=True)
        note.write_text("## Section\n" + "secret " * 30)
        mocker.patch("ingest.notes._changed_files", return_value=[note])
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.upsert")
        mocker.patch("core.qdrant.delete_by_ids")
        mock_distill = mocker.patch("ingest.notes.distill_text", return_value="d")

        from ingest.notes import ingest_notes

        ingest_notes()

        mock_distill.assert_called()
        assert all(c.kwargs.get("local_only") is True for c in mock_distill.call_args_list)


class TestExplicitFile:
    """#264: 手動実行の file 指定経路 (assert_safe_path 経由) の接続点を固定する。"""

    def test_file_goes_through_assert_safe_path_and_skips_changed_files(self, mocker):
        resolved = Path("/resolved/10-projects/a.md")
        safe = mocker.patch("core.ingest_pipeline.assert_safe_path", return_value=resolved)
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        changed = mocker.patch("ingest.notes._changed_files")
        git_changed = mocker.patch("core.ingest_pipeline.changed_files")
        ingest = mocker.patch("core.ingest_pipeline.ingest_markdown_files")

        from ingest.notes import COLLECTION, ingest_notes

        ingest_notes(file="rel/a.md")

        safe.assert_called_once_with("rel/a.md")
        changed.assert_not_called()
        git_changed.assert_not_called()
        # 解決前の文字列ではなく、assert_safe_path が返したパスがそのまま渡る
        assert ingest.call_args.args[0] == [resolved]
        assert ingest.call_args.kwargs["collection"] == COLLECTION

    def test_path_outside_base_raises_before_any_write(self, mocker, tmp_path, monkeypatch):
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("## S\n" + "x" * 100)
        monkeypatch.setenv("DOC_INGEST_BASE_DIR", str(base))
        monkeypatch.chdir(base)
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        upsert = mocker.patch("core.qdrant.upsert")
        delete = mocker.patch("core.qdrant.delete_by_ids")
        mocker.patch("core.qdrant.ids_by_payload", return_value=["old"])
        distill = mocker.patch("ingest.notes.distill_text")

        from ingest.notes import ingest_notes

        with pytest.raises(PermissionError):
            ingest_notes(file="../outside.md")

        upsert.assert_not_called()
        delete.assert_not_called()
        distill.assert_not_called()

    def test_md_outside_note_dirs_is_ingested_when_given_explicitly(self, mocker, tmp_path):
        """現仕様: file 指定には NOTE_DIRS の絞り込みがかからない (00-inbox も入る)。"""
        from ingest.notes import _is_note, ingest_notes

        scratch = tmp_path / "00-inbox" / "scratch.md"
        scratch.parent.mkdir()
        scratch.write_text("## Section\n" + "x" * 100)
        assert _is_note(Path("00-inbox/scratch.md")) is False
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mocker.patch("ingest.notes.distill_text", return_value="distilled")

        ingest_notes(file=str(scratch))

        mock_upsert.assert_called_once()
        assert mock_upsert.call_args.args[0] == "notes"
