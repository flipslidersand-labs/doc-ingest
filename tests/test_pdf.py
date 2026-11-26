"""Tests for ingest/pdf.py"""

import httpx
import pytest
import respx

from tests.conftest import override_settings

FAKE_TEXT = "Introduction\n\n" + "word " * 200


class TestExtractTextFromPath:
    """pymupdf でページを読み、`\n\n` で結合する (#227: 呼び出し元の無い bytes 版 extract_text は削除)。"""

    @staticmethod
    def _doc(mocker, pages):
        mock_pages = []
        for text in pages:
            page = mocker.MagicMock()
            page.get_text.return_value = text
            mock_pages.append(page)
        doc = mocker.MagicMock()
        doc.__iter__ = mocker.Mock(return_value=iter(mock_pages))
        doc.__enter__ = mocker.Mock(return_value=doc)
        doc.__exit__ = mocker.Mock(return_value=False)  # 実物の Document は __exit__ で close する
        return doc

    def test_single_page(self, mocker, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("pymupdf.open", return_value=self._doc(mocker, ["page one text"]))

        from ingest.pdf import _extract_text_from_path

        assert _extract_text_from_path(pdf) == "page one text"

    def test_multipage_joined_with_double_newline(self, mocker, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("pymupdf.open", return_value=self._doc(mocker, ["page one", "page two"]))

        from ingest.pdf import _extract_text_from_path

        assert _extract_text_from_path(pdf) == "page one\n\npage two"

    def test_the_document_is_closed(self, mocker, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF-fake")
        doc = self._doc(mocker, ["x"])
        mocker.patch("pymupdf.open", return_value=doc)

        from ingest.pdf import _extract_text_from_path

        _extract_text_from_path(pdf)
        doc.__exit__.assert_called_once()

    def test_missing_file_raises_before_opening(self, mocker, tmp_path):
        opener = mocker.patch("pymupdf.open")

        from ingest.pdf import _extract_text_from_path

        with pytest.raises(FileNotFoundError):
            _extract_text_from_path(tmp_path / "nope.pdf")
        opener.assert_not_called()


class TestIngestPdf:
    def test_local_file_not_found_raises(self):
        from ingest.pdf import ingest_pdf

        with pytest.raises(FileNotFoundError):
            ingest_pdf("/nonexistent/path/file.pdf")

    def test_empty_pdf_skips_upsert(self, mocker, tmp_path, caplog):
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("ingest.pdf._extract_text_from_path", return_value="   ")
        mock_upsert = mocker.patch("core.qdrant.upsert")

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
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.delete_by_ids")
        mock_upsert = mocker.patch("core.qdrant.upsert")

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf), tags=["research"])

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert len(points) > 0
        assert points[0]["source"] == "pdf"
        assert points[0]["trust"] == "external"
        assert points[0]["tags"] == ["research"]

    def test_distill_failure_keeps_old_chunks(self, mocker, tmp_path):
        """要約が全バックエンド失敗で例外になっても、旧チャンクを先に消してはいけない。"""
        from core.distiller import DistillError

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=DistillError("down"))
        mock_delete = mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.delete_by_ids")
        mock_upsert = mocker.patch("core.qdrant.upsert")

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
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.delete_by_ids")
        mock_upsert = mocker.patch("core.qdrant.upsert")

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
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.delete_by_ids")
        mock_upsert = mocker.patch("core.qdrant.upsert")

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
        mocker.patch("core.qdrant.ids_by_payload", return_value=list(old_ids))
        order = mocker.Mock()
        mocker.patch("core.qdrant.upsert", order.upsert)
        mocker.patch("core.qdrant.delete_by_ids", order.delete)
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
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.delete_by_ids")
        self.upsert = mocker.patch("core.qdrant.upsert")
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


def test_pdf_chunks_over_the_cap_are_not_distilled(mocker, tmp_path):
    """チャンク数の上限を超える PDF は、超過分を LLM 要約に回さない (#224)。"""
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(b"%PDF-fake")
    text = "\n\n".join(f"paragraph {i} " + "word " * 250 for i in range(10))
    override_settings(max_chunks_per_doc=2)
    mocker.patch("ingest.pdf._extract_text_from_path", return_value=text)
    distill = mocker.patch("ingest.pdf.distill_text", side_effect=lambda t, p, **kw: t)
    mocker.patch("core.qdrant.ids_by_payload", return_value=[])
    mocker.patch("core.qdrant.delete_by_ids")
    mock_upsert = mocker.patch("core.qdrant.upsert")

    from ingest.pdf import ingest_pdf

    ingest_pdf(str(pdf))

    assert distill.call_count == 2
    assert len(mock_upsert.call_args[0][1]) == 2


class TestHardening317:
    def test_zero_chunks_keeps_existing_points(self, mocker, tmp_path, caplog):
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-fake")
        mocker.patch("ingest.pdf._extract_text_from_path", return_value="text")
        mocker.patch("ingest.pdf.cap_chunks", return_value=[])
        replace = mocker.patch("ingest.pdf.replace_points")

        import logging

        from ingest.pdf import ingest_pdf

        with caplog.at_level(logging.WARNING, logger="ingest.pdf"):
            ingest_pdf(str(pdf))

        replace.assert_not_called()
        assert "keeping existing points" in caplog.text

    def test_local_pdf_over_size_limit_is_skipped_unread(self, mocker, tmp_path):
        pdf = tmp_path / "big.pdf"
        pdf.write_bytes(b"%PDF-" + b"x" * 100)
        override_settings(max_local_bytes=10)
        extract = mocker.patch("ingest.pdf._extract_text_from_path")
        replace = mocker.patch("ingest.pdf.replace_points")

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf))

        extract.assert_not_called()
        replace.assert_not_called()

    def test_url_query_and_fragment_are_not_stored_or_logged(self, mocker, caplog):
        mocker.patch("ingest.pdf.safe_download")
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)
        replace = mocker.patch("ingest.pdf.replace_points")

        import logging

        from ingest.pdf import ingest_pdf

        with caplog.at_level(logging.INFO, logger="ingest.pdf"):
            ingest_pdf("https://example.com/p.pdf?token=SECRET#frag")

        args = replace.call_args[0]
        assert all(p["source_url"] == "https://example.com/p.pdf" for p in args[3])
        assert "SECRET" not in caplog.text

    def test_urls_differing_only_by_query_do_not_overwrite_each_other(self, mocker):
        mocker.patch("ingest.pdf.safe_download")
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)
        store: dict[str, dict] = {}

        def fake_replace(collection, key, value, points, keep_extra_ids=()):
            new = {p["id"] for p in points}
            for i in [i for i, p in store.items() if p.get(key) == value and i not in new]:
                del store[i]
            store.update({p["id"]: p for p in points})

        mocker.patch("ingest.pdf.replace_points", side_effect=fake_replace)

        from core.ids import make_id
        from ingest.pdf import ingest_pdf

        ingest_pdf("https://example.com/dl?id=1")
        first = set(store)
        ingest_pdf("https://example.com/dl?id=2")

        assert first <= set(store)
        assert len(store) == 2 * len(first)
        # id は元の (query 付き) URL から作る = 旧実装と同じ
        assert make_id("https://example.com/dl?id=1:0") in first


class TestIngestPdfPathGuard:
    """#357: ローカル PDF も design/notes と同じく DOC_INGEST_BASE_DIR 配下に限る。"""

    @pytest.fixture
    def base(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        override_settings(doc_ingest_base_dir=str(base))
        return base

    @pytest.fixture
    def io(self, mocker):
        extract = mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.delete_by_ids")
        upsert = mocker.patch("core.qdrant.upsert")
        return extract, upsert

    def test_pdf_outside_base_is_rejected_before_reading(self, tmp_path, base, io):
        extract, upsert = io
        outside = tmp_path / "outside.pdf"
        outside.write_bytes(b"%PDF-fake")

        from ingest.pdf import ingest_pdf

        with pytest.raises(PermissionError, match="outside the allowed base"):
            ingest_pdf(str(outside))

        extract.assert_not_called()
        upsert.assert_not_called()

    def test_dotdot_escape_is_rejected(self, tmp_path, base, io):
        extract, upsert = io
        (tmp_path / "outside.pdf").write_bytes(b"%PDF-fake")

        from ingest.pdf import ingest_pdf

        with pytest.raises(PermissionError):
            ingest_pdf(str(base / ".." / "outside.pdf"))
        extract.assert_not_called()
        upsert.assert_not_called()

    def test_symlink_inside_base_pointing_outside_is_rejected(self, tmp_path, base, io):
        extract, upsert = io
        outside = tmp_path / "outside.pdf"
        outside.write_bytes(b"%PDF-fake")
        (base / "link.pdf").symlink_to(outside)

        from ingest.pdf import ingest_pdf

        with pytest.raises(PermissionError):
            ingest_pdf(str(base / "link.pdf"))
        extract.assert_not_called()
        upsert.assert_not_called()

    def test_outside_base_missing_file_is_still_a_permission_error(self, tmp_path, base, io):
        """存在確認より先に検査する (base 外のファイルの有無を FileNotFoundError で漏らさない)。"""
        from ingest.pdf import ingest_pdf

        with pytest.raises(PermissionError):
            ingest_pdf(str(tmp_path / "nope.pdf"))

    def test_pdf_inside_base_is_ingested(self, base, io):
        extract, upsert = io
        pdf = base / "sub" / "paper.pdf"
        pdf.parent.mkdir()
        pdf.write_bytes(b"%PDF-fake")

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf))

        upsert.assert_called_once()
        assert upsert.call_args[0][1][0]["source_url"] == str(pdf.resolve())

    def test_url_is_not_subject_to_the_path_guard(self, mocker, base, io):
        """URL は path_guard の対象外 (url_guard の対象)。base 外でも取り込める。"""
        download = mocker.patch("ingest.pdf.safe_download")
        _, upsert = io

        from ingest.pdf import ingest_pdf

        ingest_pdf("https://example.com/paper.pdf")

        download.assert_called_once()
        upsert.assert_called_once()


class TestIngestPdfSourceKind:
    """#357: URL かどうかは scheme で判定する (startswith("http") ではない)。"""

    @pytest.fixture
    def io(self, mocker):
        mocker.patch("ingest.pdf._extract_text_from_path", return_value=FAKE_TEXT)
        mocker.patch("ingest.pdf.distill_text", side_effect=lambda text, purpose, **kw: text)
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.delete_by_ids")
        mocker.patch("core.qdrant.upsert")
        return mocker.patch("ingest.pdf.safe_download")

    def test_uppercase_scheme_is_treated_as_a_url(self, io):
        from ingest.pdf import ingest_pdf

        ingest_pdf("HTTPS://example.com/paper.pdf")

        io.assert_called_once()

    def test_http_scheme_goes_to_url_guard_not_the_filesystem(self, io):
        from ingest.pdf import ingest_pdf

        ingest_pdf("http://example.com/paper.pdf")

        io.assert_called_once()

    def test_local_file_whose_name_starts_with_http_is_local(self, tmp_path, io):
        pdf = tmp_path / "http_notes.pdf"
        pdf.write_bytes(b"%PDF-fake")

        from ingest.pdf import ingest_pdf

        ingest_pdf(str(pdf))

        io.assert_not_called()

    def test_relative_local_file_named_http_is_local(self, tmp_path, io, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "httpd.pdf").write_bytes(b"%PDF-fake")

        from ingest.pdf import ingest_pdf

        ingest_pdf("httpd.pdf")

        io.assert_not_called()
