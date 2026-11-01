"""Tests for ingest/arxiv.py"""

import httpx
import pytest
import respx

ARXIV_API_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is All You Need</title>
    <summary>We propose a new architecture called the Transformer.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <published>2017-06-12T00:00:00Z</published>
  </entry>
</feed>"""

FAKE_PDF_BYTES = b"%PDF-1.4 fake"


class TestArxivId:
    def test_abs_url(self):
        from ingest.arxiv import _arxiv_id

        assert _arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"

    def test_pdf_url(self):
        from ingest.arxiv import _arxiv_id

        assert _arxiv_id("https://arxiv.org/pdf/1706.03762") == "1706.03762"

    def test_blog_url_returns_none(self):
        from ingest.arxiv import _arxiv_id

        assert _arxiv_id("https://example.com/blog/post") is None

    def test_versioned_id_strips_version_suffix(self):
        # regex [\d.]+ stops before 'v2' — version suffix is intentionally dropped
        from ingest.arxiv import _arxiv_id

        assert _arxiv_id("https://arxiv.org/abs/1706.03762v2") == "1706.03762"


class TestValidateArxivId:
    def test_valid_short_id(self):
        from ingest.arxiv import _validate_arxiv_id

        assert _validate_arxiv_id("1706.03762") == "1706.03762"

    def test_valid_long_id(self):
        from ingest.arxiv import _validate_arxiv_id

        assert _validate_arxiv_id("2301.12345") == "2301.12345"

    def test_valid_versioned_id(self):
        from ingest.arxiv import _validate_arxiv_id

        assert _validate_arxiv_id("1706.03762v2") == "1706.03762v2"

    def test_invalid_id_raises(self):
        from ingest.arxiv import _validate_arxiv_id

        with pytest.raises(ValueError, match="不正な arxiv_id"):
            _validate_arxiv_id("../../etc/passwd")

    def test_empty_id_raises(self):
        from ingest.arxiv import _validate_arxiv_id

        with pytest.raises(ValueError, match="不正な arxiv_id"):
            _validate_arxiv_id("")

    def test_id_with_spaces_raises(self):
        from ingest.arxiv import _validate_arxiv_id

        with pytest.raises(ValueError, match="不正な arxiv_id"):
            _validate_arxiv_id("1706.03762 extra")


class TestFetchArxiv:
    @respx.mock
    def test_parses_metadata(self):
        respx.get("https://export.arxiv.org/api/query?id_list=1706.03762").mock(
            return_value=httpx.Response(200, text=ARXIV_API_RESPONSE)
        )
        from ingest.arxiv import _fetch_arxiv

        meta = _fetch_arxiv("1706.03762")
        assert meta["title"] == "Attention Is All You Need"
        assert "Transformer" in meta["abstract"]
        assert "Ashish Vaswani" in meta["authors"]
        assert meta["published_date"] == "2017-06-12"

    @respx.mock
    def test_empty_feed_raises_value_error(self):
        """Empty feed (no entry elements) should raise ValueError immediately."""
        respx.get("https://export.arxiv.org/api/query?id_list=0000.00000").mock(
            return_value=httpx.Response(
                200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
            )
        )
        from ingest.arxiv import _fetch_arxiv

        with pytest.raises(ValueError, match="0000.00000 が見つかりません"):
            _fetch_arxiv("0000.00000")

    @respx.mock
    def test_invalid_id_rejected_before_request(self):
        """Malformed arxiv_id should raise before any HTTP call is made."""
        from ingest.arxiv import _fetch_arxiv

        with pytest.raises(ValueError, match="不正な arxiv_id"):
            _fetch_arxiv("bad/id/../etc")

    @respx.mock
    def test_cdata_and_entities_parsed_correctly(self):
        """xml.etree handles XML entities correctly unlike regex."""
        xml_with_entities = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>A &amp; B &lt;paper&gt;</title>
    <summary>Uses &lt;attention&gt; mechanism.</summary>
    <author><name>Author One</name></author>
    <published>2023-01-15T00:00:00Z</published>
  </entry>
</feed>"""
        respx.get("https://export.arxiv.org/api/query?id_list=2301.00001").mock(
            return_value=httpx.Response(200, text=xml_with_entities)
        )
        from ingest.arxiv import _fetch_arxiv

        meta = _fetch_arxiv("2301.00001")
        assert meta["title"] == "A & B <paper>"
        assert "<attention>" in meta["abstract"]


class TestFetchPdfText:
    @respx.mock
    def test_fetch_failure_returns_empty(self):
        respx.get("https://arxiv.org/pdf/1706.03762").mock(return_value=httpx.Response(404))
        from ingest.arxiv import _fetch_pdf_text

        result = _fetch_pdf_text("1706.03762")
        assert result == ""

    def test_network_error_returns_empty(self, mocker):
        mocker.patch("ingest.arxiv.httpx.get", side_effect=httpx.RequestError("timeout"))
        from ingest.arxiv import _fetch_pdf_text

        result = _fetch_pdf_text("1706.03762")
        assert result == ""


class TestIngestArxiv:
    @respx.mock
    def test_arxiv_url_calls_upsert(self, mocker):
        respx.get("https://export.arxiv.org/api/query?id_list=1706.03762").mock(
            return_value=httpx.Response(200, text=ARXIV_API_RESPONSE)
        )
        mocker.patch("ingest.arxiv._fetch_pdf_text", return_value="full paper text")
        mocker.patch("ingest.arxiv.distill_paper", return_value="distilled")
        mock_upsert = mocker.patch("ingest.arxiv.upsert")

        from ingest.arxiv import ingest_arxiv

        ingest_arxiv("https://arxiv.org/abs/1706.03762", tags=["ml"])

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert len(points) == 1
        assert points[0]["source"] == "arxiv"
        assert points[0]["arxiv_id"] == "1706.03762"
        assert points[0]["tags"] == ["ml"]

    @respx.mock
    def test_blog_url_calls_upsert(self, mocker):
        respx.get("https://example.com/blog/post").mock(
            return_value=httpx.Response(200, text="<html><body>great post</body></html>")
        )
        mocker.patch("ingest.arxiv.distill_webpage", return_value="distilled blog")
        mock_upsert = mocker.patch("ingest.arxiv.upsert")

        from ingest.arxiv import ingest_arxiv

        ingest_arxiv("https://example.com/blog/post")

        mock_upsert.assert_called_once()
        points = mock_upsert.call_args[0][1]
        assert points[0]["source"] == "blog"
        assert points[0]["source_url"] == "https://example.com/blog/post"

    @respx.mock
    def test_pdf_fetch_failure_falls_back_to_abstract(self, mocker):
        respx.get("https://export.arxiv.org/api/query?id_list=1706.03762").mock(
            return_value=httpx.Response(200, text=ARXIV_API_RESPONSE)
        )
        mocker.patch("ingest.arxiv._fetch_pdf_text", return_value="")
        mock_distill = mocker.patch("ingest.arxiv.distill_paper", return_value="abstract only")
        mock_upsert = mocker.patch("ingest.arxiv.upsert")

        from ingest.arxiv import ingest_arxiv

        ingest_arxiv("https://arxiv.org/abs/1706.03762")

        mock_distill.assert_called_once()
        mock_upsert.assert_called_once()

    @respx.mock
    def test_deterministic_id_for_same_arxiv(self, mocker):
        respx.get("https://export.arxiv.org/api/query?id_list=1706.03762").mock(
            return_value=httpx.Response(200, text=ARXIV_API_RESPONSE)
        )
        mocker.patch("ingest.arxiv._fetch_pdf_text", return_value="")
        mocker.patch("ingest.arxiv.distill_paper", return_value="d")
        mock_upsert = mocker.patch("ingest.arxiv.upsert")

        from ingest.arxiv import ingest_arxiv

        ingest_arxiv("https://arxiv.org/abs/1706.03762")
        id1 = mock_upsert.call_args[0][1][0]["id"]

        mock_upsert.reset_mock()
        ingest_arxiv("https://arxiv.org/abs/1706.03762")
        id2 = mock_upsert.call_args[0][1][0]["id"]

        assert id1 == id2


class TestArxivIdFormats:
    """#219: `.pdf` 付き URL で ID の末尾にドットが残り、_validate_arxiv_id がクラッシュしていた。"""

    @pytest.mark.parametrize(
        "url",
        [
            "https://arxiv.org/abs/1706.03762",
            "https://arxiv.org/pdf/1706.03762",
            "https://arxiv.org/pdf/1706.03762.pdf",
            "https://arxiv.org/pdf/1706.03762v5.pdf",
            "https://arxiv.org/abs/1706.03762v2",
            "https://arxiv.org/abs/1706.03762?context=cs",
            "https://arxiv.org/abs/1706.03762/",
            "https://www.arxiv.org/abs/1706.03762#section",
        ],
    )
    def test_all_common_url_shapes_give_the_bare_id(self, url):
        from ingest.arxiv import _arxiv_id, _validate_arxiv_id

        assert _arxiv_id(url) == "1706.03762"
        _validate_arxiv_id(_arxiv_id(url))  # 以前は '1706.03762.' で ValueError

    def test_five_digit_ids(self):
        from ingest.arxiv import _arxiv_id

        assert _arxiv_id("https://arxiv.org/pdf/2401.12345.pdf") == "2401.12345"
        assert _arxiv_id("https://arxiv.org/abs/2401.1234v3") == "2401.1234"

    @pytest.mark.parametrize(
        "url",
        [
            "https://arxiv.org/abs/1706.037621234",  # 桁が多すぎる: 黙って切り詰めない
            "https://arxiv.org/abs/170.03762",
            "https://arxiv.org/abs/",
            "https://example.com/arxiv.org/blog",
        ],
    )
    def test_malformed_ids_are_not_extracted(self, url):
        from ingest.arxiv import _arxiv_id

        assert _arxiv_id(url) is None

    @respx.mock
    def test_pdf_suffix_url_ingests_end_to_end(self, mocker):
        respx.get("https://export.arxiv.org/api/query?id_list=1706.03762").mock(
            return_value=httpx.Response(200, text=ARXIV_API_RESPONSE)
        )
        mocker.patch("ingest.arxiv._fetch_pdf_text", return_value="")
        mocker.patch("ingest.arxiv.distill_paper", return_value="d")
        up = mocker.patch("ingest.arxiv.upsert")

        from ingest.arxiv import ingest_arxiv

        ingest_arxiv("https://arxiv.org/pdf/1706.03762.pdf")

        assert up.call_args[0][1][0]["arxiv_id"] == "1706.03762"


ARTICLE_HTML = """<!doctype html><html><head><title>Post</title>
<style>.nav{color:red}</style><script>var tracker = 'SCRIPT-NOISE';</script></head>
<body><nav>Home | About | NAV-NOISE</nav>
<article><h1>Why retries need jitter</h1>
<p>Exponential backoff spreads retries over time, but when many clients fail together they also
retry together. Adding random jitter to each delay breaks that synchronisation and keeps the
recovering service from being hit by a thundering herd of simultaneous requests.</p>
<p>The simplest scheme is full jitter: pick a random delay between zero and the exponential cap.
It trades a slightly longer average wait for a much flatter load curve on the server side.</p>
</article><footer>Copyright FOOTER-NOISE</footer></body></html>"""


class TestIngestBlog:
    """#219: ブログ取り込みはリダイレクトを追い、生 HTML ではなく抽出した本文を要約する。"""

    @pytest.fixture(autouse=True)
    def _hermetic(self, mocker):
        """ホスト名は公開 IP に解決させる。IP リテラルはそのまま判定させる (内部アドレスを公開扱いにしない)。"""
        import ipaddress
        import socket

        def fake_getaddrinfo(host, *a, **k):
            try:
                ipaddress.ip_address(host)
                addr = host
            except ValueError:
                addr = "93.184.216.34"
            family = socket.AF_INET6 if ":" in addr else socket.AF_INET
            return [(family, None, None, None, (addr, 0))]

        mocker.patch("socket.getaddrinfo", side_effect=fake_getaddrinfo)
        self.distill = mocker.patch("ingest.arxiv.distill_webpage", return_value="distilled")
        self.upsert = mocker.patch("ingest.arxiv.upsert")
        with respx.mock:  # 未登録のリクエストは実ネットワークに出さず即失敗
            yield

    def _ingest(self, url="https://example.com/blog/post"):
        from ingest.arxiv import ingest_arxiv

        ingest_arxiv(url)

    def test_html_is_reduced_to_the_article_before_summarising(self):
        respx.get("https://example.com/blog/post").mock(
            return_value=httpx.Response(
                200, text=ARTICLE_HTML, headers={"content-type": "text/html; charset=utf-8"}
            )
        )
        self._ingest()
        body = self.distill.call_args[0][0]
        assert "full jitter" in body and "thundering herd" in body
        for noise in ("SCRIPT-NOISE", "NAV-NOISE", "FOOTER-NOISE", "<script", "<html", "<nav"):
            assert noise not in body

    def test_a_redirect_to_the_canonical_url_is_followed(self):
        respx.get("https://example.com/blog/post").mock(
            return_value=httpx.Response(301, headers={"location": "https://example.com/blog/post/"})
        )
        respx.get("https://example.com/blog/post/").mock(
            return_value=httpx.Response(
                200, text=ARTICLE_HTML, headers={"content-type": "text/html"}
            )
        )
        self._ingest()
        self.upsert.assert_called_once()
        assert self.upsert.call_args[0][1][0]["source_url"] == "https://example.com/blog/post"

    def test_a_redirect_to_an_internal_host_is_blocked_and_not_requested(self):
        from core.url_guard import UnsafeURLError

        respx.get("https://example.com/blog/post").mock(
            return_value=httpx.Response(302, headers={"location": "https://192.168.1.1/admin"})
        )
        internal = respx.get("https://192.168.1.1/admin").mock(return_value=httpx.Response(200))
        with pytest.raises(UnsafeURLError):
            self._ingest()
        assert not internal.called
        self.upsert.assert_not_called()

    def test_markdown_and_plain_text_are_passed_through_unchanged(self):
        md = "# Title\n\nSome *markdown* body that is not HTML.\n"
        respx.get("https://example.com/blog/post").mock(
            return_value=httpx.Response(200, text=md, headers={"content-type": "text/markdown"})
        )
        self._ingest()
        assert self.distill.call_args[0][0] == md

    def test_html_with_no_extractable_article_fails_loudly_and_ingests_nothing(self):
        respx.get("https://example.com/blog/post").mock(
            return_value=httpx.Response(
                200,
                text="<!doctype html><html><head><title>x</title></head><body></body></html>",
                headers={"content-type": "text/html"},
            )
        )
        with pytest.raises(ValueError, match="本文を抽出できません"):
            self._ingest()
        self.distill.assert_not_called()
        self.upsert.assert_not_called()

    def test_http_error_status_propagates(self):
        respx.get("https://example.com/blog/post").mock(return_value=httpx.Response(404))
        with pytest.raises(httpx.HTTPStatusError):
            self._ingest()
        self.upsert.assert_not_called()
