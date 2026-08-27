"""Tests for cli.py — smoke tests via click.testing.CliRunner."""

from click.testing import CliRunner

from cli import main


class TestHelp:
    def test_root_help(self):
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_arxiv_help(self):
        assert CliRunner().invoke(main, ["arxiv", "--help"]).exit_code == 0

    def test_design_help(self):
        assert CliRunner().invoke(main, ["design", "--help"]).exit_code == 0

    def test_notes_help(self):
        assert CliRunner().invoke(main, ["notes", "--help"]).exit_code == 0

    def test_sync_help(self):
        assert CliRunner().invoke(main, ["sync", "--help"]).exit_code == 0

    def test_search_help(self):
        assert CliRunner().invoke(main, ["search", "--help"]).exit_code == 0

    def test_list_help(self):
        assert CliRunner().invoke(main, ["list", "--help"]).exit_code == 0

    def test_pdf_help(self):
        assert CliRunner().invoke(main, ["pdf", "--help"]).exit_code == 0


class TestSyncCommand:
    def test_sync_calls_sync_external(self, mocker):
        mock_sync = mocker.patch("cli.sync_external")
        CliRunner().invoke(main, ["sync"])
        mock_sync.assert_called_once_with(force=False, dry_run=False)

    def test_sync_force_flag(self, mocker):
        mock_sync = mocker.patch("cli.sync_external")
        CliRunner().invoke(main, ["sync", "--force"])
        mock_sync.assert_called_once_with(force=True, dry_run=False)

    def test_sync_dry_run_flag(self, mocker):
        mock_sync = mocker.patch("cli.sync_external")
        CliRunner().invoke(main, ["sync", "--dry-run"])
        mock_sync.assert_called_once_with(force=False, dry_run=True)


class TestSearchCommand:
    def test_search_calls_qdrant_search(self, mocker):
        mock_search = mocker.patch("core.qdrant.search", return_value=[])
        CliRunner().invoke(main, ["search", "Claude API"])
        mock_search.assert_called_once()
        args = mock_search.call_args[0]
        assert args[1] == "Claude API"

    def test_search_no_results_prints_message(self, mocker):
        mocker.patch("core.qdrant.search", return_value=[])
        result = CliRunner().invoke(main, ["search", "nothing"])
        assert "No results" in result.output

    def test_search_collection_option(self, mocker):
        mock_search = mocker.patch("core.qdrant.search", return_value=[])
        CliRunner().invoke(main, ["search", "q", "--collection", "research"])
        assert mock_search.call_args[0][0] == "research"

    def test_search_with_results_prints_source(self, mocker):
        mocker.patch(
            "core.qdrant.search",
            return_value=[
                {
                    "score": 0.9,
                    "source_url": "https://example.com",
                    "text": "some content",
                }
            ],
        )
        result = CliRunner().invoke(main, ["search", "q"])
        assert "https://example.com" in result.output


class TestArxivCommand:
    def test_arxiv_calls_ingest_arxiv(self, mocker):
        mock_ingest = mocker.patch("cli.ingest_arxiv")
        CliRunner().invoke(main, ["arxiv", "https://arxiv.org/abs/1706.03762"])
        mock_ingest.assert_called_once_with("https://arxiv.org/abs/1706.03762", tags=[])

    def test_arxiv_tags_parsed(self, mocker):
        mock_ingest = mocker.patch("cli.ingest_arxiv")
        CliRunner().invoke(main, ["arxiv", "https://arxiv.org/abs/1706.03762", "--tags", "ml,nlp"])
        mock_ingest.assert_called_once_with("https://arxiv.org/abs/1706.03762", tags=["ml", "nlp"])


class TestArxivValidation:
    def test_arxiv_rejects_no_scheme(self):
        result = CliRunner().invoke(main, ["arxiv", "arxiv.org/abs/1706.03762"])
        assert result.exit_code != 0
        assert "http" in result.output.lower() or "url" in result.output.lower()

    def test_arxiv_rejects_bare_id(self):
        result = CliRunner().invoke(main, ["arxiv", "1706.03762"])
        assert result.exit_code != 0

    def test_arxiv_accepts_https(self, mocker):
        mock_ingest = mocker.patch("cli.ingest_arxiv")
        result = CliRunner().invoke(main, ["arxiv", "https://arxiv.org/abs/1706.03762"])
        assert result.exit_code == 0
        mock_ingest.assert_called_once()

    def test_arxiv_accepts_http(self, mocker):
        mock_ingest = mocker.patch("cli.ingest_arxiv")
        result = CliRunner().invoke(main, ["arxiv", "http://example.com/paper"])
        assert result.exit_code == 0
        mock_ingest.assert_called_once()


class TestSearchLimitValidation:
    def test_search_limit_zero_rejected(self, mocker):
        mocker.patch("core.qdrant.search", return_value=[])
        result = CliRunner().invoke(main, ["search", "q", "--limit", "0"])
        assert result.exit_code != 0

    def test_search_limit_negative_rejected(self, mocker):
        mocker.patch("core.qdrant.search", return_value=[])
        result = CliRunner().invoke(main, ["search", "q", "--limit", "-1"])
        assert result.exit_code != 0

    def test_search_limit_over_100_rejected(self, mocker):
        mocker.patch("core.qdrant.search", return_value=[])
        result = CliRunner().invoke(main, ["search", "q", "--limit", "101"])
        assert result.exit_code != 0

    def test_search_limit_1_accepted(self, mocker):
        mocker.patch("core.qdrant.search", return_value=[])
        result = CliRunner().invoke(main, ["search", "q", "--limit", "1"])
        assert result.exit_code == 0

    def test_search_limit_100_accepted(self, mocker):
        mocker.patch("core.qdrant.search", return_value=[])
        result = CliRunner().invoke(main, ["search", "q", "--limit", "100"])
        assert result.exit_code == 0


class TestPdfValidation:
    def test_pdf_missing_file_shows_error(self, mocker):
        mocker.patch("ingest.pdf.ingest_pdf", side_effect=FileNotFoundError)
        result = CliRunner().invoke(main, ["pdf", "/nonexistent/path.pdf"])
        assert result.exit_code != 0
        assert "ファイルが見つかりません" in result.output or "Error" in result.output

    def test_pdf_calls_ingest_pdf(self, mocker):
        mock_ingest = mocker.patch("ingest.pdf.ingest_pdf")
        CliRunner().invoke(main, ["pdf", "/some/file.pdf"])
        mock_ingest.assert_called_once_with("/some/file.pdf", tags=[])
