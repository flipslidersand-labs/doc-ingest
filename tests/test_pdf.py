"""Tests for ingest/pdf.py"""
import httpx
import pytest
import respx

FAKE_TEXT = "Introduction\n\n" + "word " * 200


class TestExtractText:
    def test_delegates_to_pymupdf(self, mocker):
        mock_page = mocker.MagicMock()
        mock_page.get_text.return_value = "page one text"
        mock_doc = mocker.MagicMock()
        mock_doc.__iter__ = mocker.Mock(return_value=iter([mock_page]))
        mocker.patch("pymupdf.open", return_value=mock_doc)

        from ingest.pdf import extract_text
        result = extract_text(b"%PDF-fake")
        assert result == "page one text"
        mock_doc.close.assert_called_once()

    def test_multipage_joined_with_double_newline(self, mocker):
        pages = [mocker.MagicMock(), mocker.MagicMock()]
        pages[0].get_text.return_value = "page one"
        pages[1].get_text.return_value = "page two"
        mock_doc = mocker.MagicMock()
        mock_doc.__iter__ = mocker.Mock(return_value=iter(pages))
        mocker.patch("pymupdf.open", return_value=mock_doc)

        from ingest.pdf import extract_text
        result = extract_text(b"%PDF-fake")
        assert result == "page one\n\npage two"


class TestIngestPdf:
    def test_local_file_not_found_raises(self):
        from ingest.pdf import ingest_pdf
        with pytest.raises(FileNotFoundError):
            ingest_pdf("/nonexistent/path/file.pdf")

    def test_empty_pdf_skips_upsert(self, mocker, tmp_path, caplog):
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("ingest.pdf.extract_text", return_value="   ")
        mock_upsert = mocker.patch("ingest.pdf.upsert")

        import logging

        from ingest.pdf import ingest_pdf
        with caplog.at_level(logging.WARNING, logger="ingest.pdf"):
            ingest_pdf(str(pdf))

        mock_upsert.assert_not_called()
        assert "no text extracted" in caplog.text

    def test_local_pdf_upserted(self, mocker, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("ingest.pdf.extract_text", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, prompt: text)
        mocker.patch("ingest.pdf.delete_by_payload")
        mock_upsert = mocker.patch("ingest.pdf.upsert")

        from ingest.pdf import ingest_pdf
        ingest_pdf(str(pdf), tags=["research"])

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert len(points) > 0
        assert points[0]["source"] == "pdf"
        assert points[0]["tags"] == ["research"]

    @respx.mock
    def test_remote_url_fetches_and_upserts(self, mocker):
        respx.get("https://example.com/paper.pdf").mock(
            return_value=httpx.Response(200, content=b"%PDF-fake")
        )
        mocker.patch("ingest.pdf.extract_text", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, prompt: text)
        mocker.patch("ingest.pdf.delete_by_payload")
        mock_upsert = mocker.patch("ingest.pdf.upsert")

        from ingest.pdf import ingest_pdf
        ingest_pdf("https://example.com/paper.pdf")

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert points[0]["source_url"] == "https://example.com/paper.pdf"

    @respx.mock
    def test_remote_404_raises(self, mocker):
        respx.get("https://example.com/missing.pdf").mock(
            return_value=httpx.Response(404)
        )
        from ingest.pdf import ingest_pdf
        with pytest.raises(httpx.HTTPStatusError):
            ingest_pdf("https://example.com/missing.pdf")

    def test_deterministic_chunk_ids(self, mocker, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("ingest.pdf.extract_text", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, prompt: text)
        mocker.patch("ingest.pdf.delete_by_payload")
        mock_upsert = mocker.patch("ingest.pdf.upsert")

        from ingest.pdf import ingest_pdf
        ingest_pdf(str(pdf))
        ids1 = [p["id"] for p in mock_upsert.call_args[0][1]]

        mock_upsert.reset_mock()
        ingest_pdf(str(pdf))
        ids2 = [p["id"] for p in mock_upsert.call_args[0][1]]

        assert ids1 == ids2
