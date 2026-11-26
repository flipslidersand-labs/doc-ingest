"""#358: 例外文言・CLI エラー・ログに署名付き URL の query / fragment を残さない。

HTTP リクエスト自体は元の URL のまま (query 付き) 送る。伏せるのは出力だけ。
"""

import io
import logging
import socket

import httpx
import pytest
import respx
from click.testing import CliRunner

from cli import main
from core.url_guard import UnsafeURLError, assert_safe_url, safe_download, safe_get

SECRET = "SIGNEDSECRET123"
QS = f"?X-Amz-Signature={SECRET}&token={SECRET}#frag-{SECRET}"


@pytest.fixture(autouse=True)
def _no_real_network(mocker):
    """未登録のリクエストは即失敗させ、名前は公開 IP に解決させる (IP リテラルはそのまま判定)。"""
    real = socket.getaddrinfo

    def fake(host, *a, **kw):
        try:
            socket.inet_pton(socket.AF_INET, host)
            return real(host, *a, **kw)
        except OSError:
            pass
        if host.startswith("["):
            return real(host, *a, **kw)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    mocker.patch("core.url_guard.socket.getaddrinfo", side_effect=fake)
    with respx.mock:
        yield


def _download(url, **kw):
    return safe_download(url, io.BytesIO(), **kw)


class TestUnsafeUrlErrorHidesQuery:
    def test_non_https_scheme(self):
        with pytest.raises(UnsafeURLError) as ei:
            assert_safe_url(f"http://example.com/a.pdf{QS}")
        assert SECRET not in str(ei.value)
        assert "http://example.com/a.pdf" in str(ei.value)

    def test_no_hostname(self):
        with pytest.raises(UnsafeURLError) as ei:
            assert_safe_url(f"https:///a.pdf{QS}")
        assert SECRET not in str(ei.value)

    def test_blocked_address(self):
        with pytest.raises(UnsafeURLError) as ei:
            assert_safe_url(f"https://10.0.0.1/a.pdf{QS}")
        assert SECRET not in str(ei.value)
        assert "blocked address" in str(ei.value)

    def test_redirect_without_location(self):
        respx.get("https://example.com/r").mock(return_value=httpx.Response(302))
        with pytest.raises(UnsafeURLError, match="no Location") as ei:
            safe_get(f"https://example.com/r{QS}")
        assert SECRET not in str(ei.value)

    def test_too_many_redirects_safe_get(self):
        respx.get(url__regex=r"https://example\.com/.*").mock(
            return_value=httpx.Response(302, headers={"location": f"https://example.com/loop{QS}"})
        )
        with pytest.raises(UnsafeURLError, match="too many redirects") as ei:
            safe_get(f"https://example.com/start{QS}")
        assert SECRET not in str(ei.value)

    def test_too_many_redirects_safe_download(self):
        respx.get(url__regex=r"https://example\.com/.*").mock(
            return_value=httpx.Response(302, headers={"location": f"https://example.com/loop{QS}"})
        )
        with pytest.raises(UnsafeURLError, match="too many redirects") as ei:
            _download(f"https://example.com/start{QS}")
        assert SECRET not in str(ei.value)

    def test_declared_size_too_large(self):
        respx.get("https://example.com/big").mock(
            return_value=httpx.Response(
                200, headers={"content-length": "1000"}, content=b"x" * 1000
            )
        )
        with pytest.raises(UnsafeURLError, match="too large") as ei:
            _download(f"https://example.com/big{QS}", max_bytes=100)
        assert SECRET not in str(ei.value)

    def test_streamed_size_exceeded(self):
        def chunks():
            for _ in range(50):
                yield b"x" * 10

        respx.get("https://example.com/s").mock(return_value=httpx.Response(200, content=chunks()))
        with pytest.raises(UnsafeURLError, match="exceeds") as ei:
            _download(f"https://example.com/s{QS}", max_bytes=100)
        assert SECRET not in str(ei.value)

    def test_request_itself_keeps_the_original_url(self):
        """伏せるのは出力だけ。取得先の URL は元のまま (署名付き URL が取れなくなってはいけない)。"""
        route = respx.get("https://example.com/a.pdf").mock(
            return_value=httpx.Response(200, content=b"ok")
        )
        assert _download(f"https://example.com/a.pdf{QS}") == 2
        assert SECRET in str(route.calls.last.request.url)


class TestCliHidesQuery:
    def test_pdf_http_404_has_no_query_and_no_traceback(self):
        respx.get("https://example.com/a.pdf").mock(return_value=httpx.Response(404))
        result = CliRunner().invoke(main, ["pdf", f"https://example.com/a.pdf{QS}"])
        assert result.exit_code != 0
        assert SECRET not in result.output
        assert "404" in result.output
        assert "https://example.com/a.pdf" in result.output
        assert "Traceback" not in result.output

    def test_pdf_http_404_after_redirect_hides_the_final_url_query(self):
        respx.get("https://example.com/a.pdf").mock(
            return_value=httpx.Response(302, headers={"location": f"https://cdn.example.com/x{QS}"})
        )
        respx.get("https://cdn.example.com/x").mock(return_value=httpx.Response(403))
        result = CliRunner().invoke(main, ["pdf", "https://example.com/a.pdf"])
        assert result.exit_code != 0
        assert SECRET not in result.output
        assert "403" in result.output

    def test_pdf_unsafe_url_has_no_query(self):
        result = CliRunner().invoke(main, ["pdf", f"https://10.0.0.1/a.pdf{QS}"])
        assert result.exit_code != 0
        assert SECRET not in result.output
        assert "取り込めない URL" in result.output

    def test_arxiv_blog_http_404_has_no_query_and_no_traceback(self):
        respx.get("https://example.com/post").mock(return_value=httpx.Response(404))
        result = CliRunner().invoke(main, ["arxiv", f"https://example.com/post{QS}"])
        assert result.exit_code != 0
        assert SECRET not in result.output
        assert "404" in result.output
        assert "Traceback" not in result.output

    def test_arxiv_unsafe_url_has_no_query(self):
        result = CliRunner().invoke(main, ["arxiv", f"http://example.com/post{QS}"])
        assert result.exit_code != 0
        assert SECRET not in result.output
        assert "https" in result.output.lower()


def test_arxiv_pdf_fetch_failure_log_has_no_query(caplog, mocker):
    """arXiv PDF が署名付き URL へリダイレクトして失敗しても、警告ログに query を出さない。"""
    from ingest.arxiv import _fetch_pdf_text

    respx.get("https://arxiv.org/pdf/1706.03762").mock(
        return_value=httpx.Response(302, headers={"location": f"https://cdn.example.com/x{QS}"})
    )
    respx.get("https://cdn.example.com/x").mock(return_value=httpx.Response(500))
    with caplog.at_level(logging.WARNING):
        assert _fetch_pdf_text("1706.03762") == ""
    assert "PDF fetch failed" in caplog.text
    assert SECRET not in caplog.text
