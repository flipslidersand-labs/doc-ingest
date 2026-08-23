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

    def test_first_commit_fallback_uses_ls_files(self, mocker):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "HEAD~1" in cmd:
                return MagicMock(returncode=128, stdout="")
            return MagicMock(returncode=0, stdout="CLAUDE.md\n")

        mocker.patch("subprocess.run", side_effect=fake_run)

        import importlib

        import ingest.design_docs as m
        importlib.reload(m)
        m._changed_files()

        cmds = [" ".join(c) for c in calls]
        assert any("ls-files" in c for c in cmds)

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

    def test_delete_called_before_upsert(self, mocker, tmp_path):
        md = tmp_path / "CLAUDE.md"
        md.write_text("## Section\n" + "x" * 100)

        mocker.patch("ingest.design_docs._changed_files", return_value=[md])
        mocker.patch("ingest.design_docs.head_sha", return_value="abc")
        mock_delete = mocker.patch("ingest.design_docs.delete_by_payload")
        mock_upsert = mocker.patch("ingest.design_docs.upsert")
        mocker.patch("ingest.design_docs.distill_design_doc", return_value="distilled")

        from ingest.design_docs import ingest_design_docs
        ingest_design_docs()

        mock_delete.assert_called_once_with("design", "file_path", str(md))
        assert mock_upsert.call_count == 1
