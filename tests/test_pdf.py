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

    def test_multipage_joined_with_double_newline(self, mocker):
        pages = [mocker.MagicMock(), mocker.MagicMock()]
        pages[0].get_text.return_value = "page one"
        pages[1].get_text.return_value = "page two"
        mock_doc = mocker.MagicMock()
        mock_doc.__iter__ = mocker.Mock(return_value=iter(pages))
        mock_doc.__enter__ = mocker.Mock(return_value=mock_doc)
        mock_doc.__exit__ = mocker.Mock(return_value=False)
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
        mocker.patch("ingest.pdf._extract_text_from_path", return_value="   ")
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
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)
        mocker.patch("ingest.pdf.ids_by_payload", return_value=[])
        mocker.patch("ingest.pdf.delete_by_ids")
        mock_upsert = mocker.patch("ingest.pdf.upsert")

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf), tags=["research"])

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert len(points) > 0
        assert points[0]["source"] == "pdf"
        assert points[0]["tags"] == ["research"]

    def test_distill_failure_keeps_old_chunks(self, mocker, tmp_path):
        """要約が全バックエンド失敗で例外になっても、旧チャンクを先に消してはいけない。"""
        from core.distiller import DistillError

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=DistillError("down"))
        mock_delete = mocker.patch("ingest.pdf.ids_by_payload", return_value=[])
        mocker.patch("ingest.pdf.delete_by_ids")
        mock_upsert = mocker.patch("ingest.pdf.upsert")

        from ingest.pdf import ingest_pdf

        with pytest.raises(DistillError):
            ingest_pdf(str(pdf))

        mock_delete.assert_not_called()
        mock_upsert.assert_not_called()

    @pytest.fixture
    def public_dns(self, mocker):
        import socket

        return mocker.patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, None, None, None, ("93.184.216.34", 0))],
        )

    @respx.mock
    def test_remote_url_fetches_and_upserts(self, mocker, public_dns):
        respx.get("https://example.com/paper.pdf").mock(
            return_value=httpx.Response(200, content=b"%PDF-fake")
        )
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)
        mocker.patch("ingest.pdf.ids_by_payload", return_value=[])
        mocker.patch("ingest.pdf.delete_by_ids")
        mock_upsert = mocker.patch("ingest.pdf.upsert")

        from ingest.pdf import ingest_pdf

        ingest_pdf("https://example.com/paper.pdf")

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert points[0]["source_url"] == "https://example.com/paper.pdf"

    @respx.mock
    def test_remote_404_raises(self, mocker, public_dns):
        respx.get("https://example.com/missing.pdf").mock(return_value=httpx.Response(404))
        from ingest.pdf import ingest_pdf

        with pytest.raises(httpx.HTTPStatusError):
            ingest_pdf("https://example.com/missing.pdf")

    def test_deterministic_chunk_ids(self, mocker, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)
        mocker.patch("ingest.pdf.ids_by_payload", return_value=[])
        mocker.patch("ingest.pdf.delete_by_ids")
        mock_upsert = mocker.patch("ingest.pdf.upsert")

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf))
        ids1 = [p["id"] for p in mock_upsert.call_args[0][1]]

        mock_upsert.reset_mock()
        ingest_pdf(str(pdf))
        ids2 = [p["id"] for p in mock_upsert.call_args[0][1]]

        assert ids1 == ids2


class TestIngestPdfSafety:
    """#215: 旧チャンクの先行削除でデータを失わない / 非公開 PDF を Anthropic に送らない。"""

    def _patches(self, mocker, old_ids=()):
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.ids_by_payload", return_value=list(old_ids))
        order = mocker.Mock()
        mocker.patch("ingest.pdf.upsert", order.upsert)
        mocker.patch("ingest.pdf.delete_by_ids", order.delete)
        return order

    def test_embedding_failure_keeps_old_chunks(self, mocker, tmp_path):
        """upsert (= embed) が落ちても、旧チャンクは 1 件も消えない。"""
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-fake")
        order = self._patches(mocker, old_ids=["old-1", "old-2"])
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)
        order.upsert.side_effect = RuntimeError("embedding service down")

        from ingest.pdf import ingest_pdf

        with pytest.raises(RuntimeError):
            ingest_pdf(str(pdf))

        order.delete.assert_not_called()

    def test_stale_chunks_are_deleted_only_after_a_successful_upsert(self, mocker, tmp_path):
        from core.ids import make_id

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-fake")
        source_url = str(pdf.resolve())
        keep_id = make_id(f"{source_url}:0")
        order = self._patches(mocker, old_ids=[keep_id, "stale-id"])
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf))

        assert [c[0] for c in order.mock_calls] == ["upsert", "delete"]
        assert order.delete.call_args.args == ("research", ["stale-id"])

    def test_local_pdf_is_summarised_locally_only_by_default(self, mocker, tmp_path):
        pdf = tmp_path / "private.pdf"
        pdf.write_bytes(b"%PDF-fake")
        self._patches(mocker)
        distill = mocker.patch("ingest.pdf.distill_text", return_value="d")

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf))

        assert distill.call_count > 0
        assert all(c.kwargs["local_only"] is True for c in distill.call_args_list)

    def test_allow_cloud_opts_a_local_pdf_into_anthropic(self, mocker, tmp_path):
        pdf = tmp_path / "private.pdf"
        pdf.write_bytes(b"%PDF-fake")
        self._patches(mocker)
        distill = mocker.patch("ingest.pdf.distill_text", return_value="d")

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf), allow_cloud=True)

        assert all(c.kwargs["local_only"] is False for c in distill.call_args_list)

    @respx.mock
    def test_remote_pdf_is_a_public_document_so_cloud_is_allowed(self, mocker):
        respx.get("https://example.com/paper.pdf").mock(
            return_value=httpx.Response(200, content=b"%PDF-fake")
        )
        self._patches(mocker)
        distill = mocker.patch("ingest.pdf.distill_text", return_value="d")

        from ingest.pdf import ingest_pdf

        ingest_pdf("https://example.com/paper.pdf")

        assert all(c.kwargs["local_only"] is False for c in distill.call_args_list)


class TestIngestPdfUrlSafety:
    """#214: URL 入力は url_guard を通る (SSRF・リダイレクト・サイズ)。"""

    @pytest.fixture(autouse=True)
    def _no_real_network(self):
        """未登録のリクエストは接続せず即失敗させる。ガードが壊れても実ネットワークに出ない。"""
        with respx.mock:
            yield

    @pytest.fixture(autouse=True)
    def _no_side_effects(self, mocker):
        mocker.patch("ingest.pdf.ids_by_payload", return_value=[])
        mocker.patch("ingest.pdf.delete_by_ids")
        self.upsert = mocker.patch("ingest.pdf.upsert")
        mocker.patch("ingest.pdf.distill_text", return_value="d")
        self.extract = mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)

    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/a.pdf",  # 平文 http
            "https://10.0.0.1/a.pdf",
            "https://169.254.169.254/latest/meta-data",
            "https://[::ffff:127.0.0.1]/a.pdf",
            "https://100.64.0.1/a.pdf",  # CGNAT
        ],
    )
    def test_unsafe_urls_are_rejected_without_a_request(self, url):
        from core.url_guard import UnsafeURLError
        from ingest.pdf import ingest_pdf

        with pytest.raises(UnsafeURLError):
            ingest_pdf(url)
        self.extract.assert_not_called()
        self.upsert.assert_not_called()

    def test_redirect_to_an_internal_host_is_not_followed(self, mocker):
        import socket

        from core.url_guard import UnsafeURLError
        from ingest.pdf import ingest_pdf

        mocker.patch(
            "socket.getaddrinfo",
            side_effect=lambda host, *a, **k: [
                (
                    socket.AF_INET,
                    None,
                    None,
                    None,
                    ("93.184.216.34" if host == "example.com" else host, 0),
                )
            ],
        )
        respx.get("https://example.com/r.pdf").mock(
            return_value=httpx.Response(302, headers={"location": "https://192.168.1.1/x.pdf"})
        )
        internal = respx.get("https://192.168.1.1/x.pdf").mock(return_value=httpx.Response(200))
        with pytest.raises(UnsafeURLError):
            ingest_pdf("https://example.com/r.pdf")
        assert not internal.called
        self.upsert.assert_not_called()

    def test_oversized_pdf_is_rejected_and_the_temp_file_is_removed(self, mocker):
        import glob
        import tempfile

        from core.url_guard import UnsafeURLError
        from ingest.pdf import ingest_pdf

        def too_big(url, fileobj, **kw):
            fileobj.write(b"x" * 10)
            raise UnsafeURLError("response exceeds limit")

        mocker.patch("ingest.pdf.safe_download", side_effect=too_big)
        before = set(glob.glob(f"{tempfile.gettempdir()}/tmp*.pdf"))
        with pytest.raises(UnsafeURLError):
            ingest_pdf("https://example.com/huge.pdf")
        assert (
            set(glob.glob(f"{tempfile.gettempdir()}/tmp*.pdf")) == before
        )  # 一時ファイルが残らない
        self.upsert.assert_not_called()

    def test_local_paths_do_not_go_through_the_downloader(self, mocker, tmp_path):
        from ingest.pdf import ingest_pdf

        dl = mocker.patch("ingest.pdf.safe_download")
        pdf = tmp_path / "local.pdf"
        pdf.write_bytes(b"%PDF")
        ingest_pdf(str(pdf))
        dl.assert_not_called()
