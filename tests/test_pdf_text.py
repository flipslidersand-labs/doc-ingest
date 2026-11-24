"""Tests for core/pdf_text.py"""

import pytest

from doc_ingest.core.pdf_text import extract_pdf_text


def _doc(mocker, pages, error_on=None):
    mock_pages = []
    for i, text in enumerate(pages):
        page = mocker.MagicMock()
        if i == error_on:
            page.get_text.side_effect = RuntimeError("boom")
        else:
            page.get_text.return_value = text
        mock_pages.append(page)
    doc = mocker.MagicMock()
    doc.__iter__ = mocker.Mock(return_value=iter(mock_pages))
    doc.__enter__ = mocker.Mock(return_value=doc)
    doc.__exit__ = mocker.Mock(return_value=False)
    return doc, mock_pages


def test_path_source_joins_pages(mocker, tmp_path):
    doc, _ = _doc(mocker, ["a", "b"])
    opener = mocker.patch("pymupdf.open", return_value=doc)
    assert extract_pdf_text(tmp_path / "x.pdf") == "a\n\nb"
    assert opener.call_args.args == (str(tmp_path / "x.pdf"),)


def test_bytes_source_opens_as_stream(mocker):
    doc, _ = _doc(mocker, ["a"])
    opener = mocker.patch("pymupdf.open", return_value=doc)
    assert extract_pdf_text(b"%PDF") == "a"
    assert opener.call_args.kwargs == {"stream": b"%PDF", "filetype": "pdf"}


def test_document_closed_when_get_text_raises(mocker):
    doc, _ = _doc(mocker, ["a", "b"], error_on=1)
    mocker.patch("pymupdf.open", return_value=doc)
    with pytest.raises(RuntimeError):
        extract_pdf_text(b"%PDF")
    doc.__exit__.assert_called_once()


def test_limit_truncates_and_stops_reading(mocker):
    doc, pages = _doc(mocker, ["x" * 5000] * 50)
    mocker.patch("pymupdf.open", return_value=doc)
    out = extract_pdf_text(b"%PDF", limit=8000)
    assert len(out) == 8000
    assert sum(1 for p in pages if p.get_text.called) <= 3


def test_no_limit_reads_everything(mocker):
    doc, pages = _doc(mocker, ["x" * 5000] * 5)
    mocker.patch("pymupdf.open", return_value=doc)
    assert len(extract_pdf_text(b"%PDF")) == 5000 * 5 + 2 * 4
    assert all(p.get_text.called for p in pages)
