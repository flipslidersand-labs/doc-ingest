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
            return_value=httpx.Response(200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>')
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
        respx.get("https://arxiv.org/pdf/1706.03762").mock(
            return_value=httpx.Response(404)
        )
        from ingest.arxiv import _fetch_pdf_text
        result = _fetch_pdf_text("1706.03762")
        assert result == ""

    def test_network_error_returns_empty(self, mocker):
        mocker.patch("httpx.get", side_effect=httpx.RequestError("timeout"))
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
