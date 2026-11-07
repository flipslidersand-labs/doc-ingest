"""Tests for ingest/design_docs.py"""

from pathlib import Path

import pytest

from ingest.design_docs import _changed_files
from tests.conftest import override_settings


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


class TestExplicitFile:
    """#264: 手動実行の file 指定経路 (assert_safe_path 経由) の接続点を固定する。"""

    def test_file_goes_through_assert_safe_path_and_skips_changed_files(self, mocker):
        resolved = Path("/resolved/CLAUDE.md")
        safe = mocker.patch("core.ingest_pipeline.assert_safe_path", return_value=resolved)
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        changed = mocker.patch("ingest.design_docs._changed_files")
        git_changed = mocker.patch("core.ingest_pipeline.changed_files")
        ingest = mocker.patch("core.ingest_pipeline.ingest_markdown_files")

        from ingest.design_docs import COLLECTION, ingest_design_docs

        ingest_design_docs(file="rel/CLAUDE.md")

        safe.assert_called_once_with("rel/CLAUDE.md")
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
        override_settings(doc_ingest_base_dir=str(base))
        monkeypatch.chdir(base)
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        upsert = mocker.patch("core.qdrant.upsert")
        delete = mocker.patch("core.qdrant.delete_by_ids")
        mocker.patch("core.qdrant.ids_by_payload", return_value=["old"])
        distill = mocker.patch("ingest.design_docs.distill_design_doc")

        from ingest.design_docs import ingest_design_docs

        with pytest.raises(PermissionError):
            ingest_design_docs(file="../outside.md")

        upsert.assert_not_called()
        delete.assert_not_called()
        distill.assert_not_called()

    def test_md_outside_design_patterns_is_ingested_when_given_explicitly(self, mocker, tmp_path):
        """現仕様: file 指定には DESIGN_PATTERNS の絞り込みがかからない (README.md も入る)。"""
        from ingest.design_docs import _is_design_doc, ingest_design_docs

        readme = tmp_path / "README.md"
        readme.write_text("## Section\n" + "x" * 100)
        assert _is_design_doc(readme) is False
        mocker.patch("core.ingest_pipeline.head_sha", return_value="abc")
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mocker.patch("ingest.design_docs.distill_design_doc", return_value="distilled")

        ingest_design_docs(file=str(readme))

        mock_upsert.assert_called_once()
        assert mock_upsert.call_args.args[0] == "design"
        assert mock_upsert.call_args.args[1][0]["source"] == "design-doc"
