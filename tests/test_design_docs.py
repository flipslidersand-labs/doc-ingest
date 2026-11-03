"""Tests for ingest/design_docs.py"""

from pathlib import Path

import pytest

from ingest.design_docs import _changed_files


@pytest.fixture
def git_changed(mocker):
    """Stub what core.git.changed_files returns, as seen by core.ingest_pipeline."""

    def _set(paths):
        return mocker.patch(
            "core.ingest_pipeline.changed_files", return_value=[Path(p) for p in paths]
        )

    return _set


class TestChangedFiles:
    @pytest.mark.parametrize(
        "path",
        [
            "CLAUDE.md",
            "AGENTS.md",
            "docs/a.md",
            "adr/0001.md",
            "ADR/0001.md",
            # Path.match matches from the right, so nested dirs still count.
            "pkg/CLAUDE.md",
            "pkg/AGENTS.md",
            "sub/docs/a.md",
        ],
    )
    def test_design_docs_included(self, git_changed, path):
        git_changed([path])
        assert _changed_files() == [Path(path)]

    @pytest.mark.parametrize(
        "path",
        [
            "README.md",
            "src/main.go",
            "docs/a.txt",
            "CLAUDE.txt",
            "adr/0001.rst",
            "notes/a.md",
            # current spec: docs/*.md is one level only
            "docs/sub/a.md",
        ],
    )
    def test_non_design_docs_excluded(self, git_changed, path):
        git_changed([path])
        assert _changed_files() == []

    def test_mixed_input_keeps_only_design_docs_in_order(self, git_changed):
        git_changed(["src/main.go", "docs/a.md", "README.md", "ADR/0001.md"])
        assert _changed_files() == [Path("docs/a.md"), Path("ADR/0001.md")]

    def test_no_changes_returns_empty(self, git_changed):
        git_changed([])
        assert _changed_files() == []


class TestIngestDesignDocs:
    def test_no_files_logs_message(self, mocker, caplog):
        mocker.patch("ingest.design_docs._changed_files", return_value=[])
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc1234")
        import logging

        from ingest.design_docs import ingest_design_docs

        with caplog.at_level(logging.DEBUG, logger="core.ingest_pipeline"):
            ingest_design_docs()
        assert "no design docs changed" in caplog.text

    def test_upsert_before_delete_and_stale_removed(self, mocker, tmp_path):
        md = tmp_path / "CLAUDE.md"
        md.write_text("## Section\n" + "x" * 100)

        old_id = 9999
        mocker.patch("ingest.design_docs._changed_files", return_value=[md])
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[old_id])
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mock_delete_ids = mocker.patch("core.qdrant.delete_by_ids")
        mocker.patch("ingest.design_docs.distill_design_doc", return_value="distilled")

        from ingest.design_docs import ingest_design_docs

        ingest_design_docs()

        mock_upsert.assert_called_once()
        # old_id is not in new points → should be deleted
        mock_delete_ids.assert_called_once()
        deleted = mock_delete_ids.call_args[0][1]
        assert old_id in deleted

    def test_no_stale_ids_skips_delete(self, mocker, tmp_path):
        md = tmp_path / "CLAUDE.md"
        md.write_text("## Section\n" + "x" * 100)

        mocker.patch("ingest.design_docs._changed_files", return_value=[md])
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mock_delete_ids = mocker.patch("core.qdrant.delete_by_ids")
        mocker.patch("ingest.design_docs.distill_design_doc", return_value="distilled")

        from ingest.design_docs import ingest_design_docs

        ingest_design_docs()

        mock_upsert.assert_called_once()
        mock_delete_ids.assert_not_called()
