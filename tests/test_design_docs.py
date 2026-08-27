"""Tests for ingest/design_docs.py"""

from unittest.mock import MagicMock


class TestChangedFiles:
    def test_returns_matching_design_docs(self, mocker, tmp_path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("content")

        mocker.patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=f"{claude_md}\n"),
        )
        import importlib

        import ingest.design_docs as m

        importlib.reload(m)

        result = m._changed_files()
        assert any(p.name == "CLAUDE.md" for p in result)

    def test_first_commit_error_returns_empty(self, mocker):
        mocker.patch(
            "subprocess.run",
            return_value=MagicMock(returncode=128, stdout=""),
        )

        import importlib

        import ingest.design_docs as m

        importlib.reload(m)
        result = m._changed_files()

        assert result == []

    def test_non_design_docs_excluded(self, mocker):
        mocker.patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout="src/main.go\nREADME.md\n"),
        )
        import importlib

        import ingest.design_docs as m

        importlib.reload(m)
        result = m._changed_files()
        assert result == []


class TestIngestDesignDocs:
    def test_no_files_logs_message(self, mocker, caplog):
        mocker.patch("ingest.design_docs._changed_files", return_value=[])
        mocker.patch("ingest.design_docs.head_sha", return_value="abc1234")
        import logging

        from ingest.design_docs import ingest_design_docs

        with caplog.at_level(logging.DEBUG, logger="ingest.design_docs"):
            ingest_design_docs()
        assert "no design docs changed" in caplog.text

    def test_upsert_before_delete_and_stale_removed(self, mocker, tmp_path):
        md = tmp_path / "CLAUDE.md"
        md.write_text("## Section\n" + "x" * 100)

        old_id = 9999
        mocker.patch("ingest.design_docs._changed_files", return_value=[md])
        mocker.patch("ingest.design_docs.head_sha", return_value="abc")
        mocker.patch("ingest.design_docs.ids_by_payload", return_value=[old_id])
        mock_upsert = mocker.patch("ingest.design_docs.upsert")
        mock_delete_ids = mocker.patch("ingest.design_docs.delete_by_ids")
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
        mocker.patch("ingest.design_docs.head_sha", return_value="abc")
        mocker.patch("ingest.design_docs.ids_by_payload", return_value=[])
        mock_upsert = mocker.patch("ingest.design_docs.upsert")
        mock_delete_ids = mocker.patch("ingest.design_docs.delete_by_ids")
        mocker.patch("ingest.design_docs.distill_design_doc", return_value="distilled")

        from ingest.design_docs import ingest_design_docs

        ingest_design_docs()

        mock_upsert.assert_called_once()
        mock_delete_ids.assert_not_called()
