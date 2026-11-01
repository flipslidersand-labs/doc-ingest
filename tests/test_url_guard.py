"""Tests for core/url_guard.py"""

import socket

import httpx
import pytest
import respx

from core.url_guard import (
    MAX_REDIRECTS,
    UnsafeURLError,
    assert_safe_url,
    safe_download,
    safe_get,
)


class TestAssertSafeUrl:
    def test_private_ip_rfc1918_10(self):
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://10.0.0.1/secret")

    def test_private_ip_rfc1918_172(self):
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://172.16.0.1/secret")

    def test_private_ip_rfc1918_192(self):
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://192.168.1.1/admin")

    def test_loopback_ipv4(self):
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://127.0.0.1/")

    def test_loopback_localhost_resolves_to_blocked(self, mocker):
        mocker.patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, None, None, None, ("127.0.0.1", 0))],
        )
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://localhost/")

    def test_link_local(self):
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://169.254.169.254/latest/meta-data/")

    def test_http_scheme_rejected(self):
        with pytest.raises(UnsafeURLError, match="https scheme"):
            assert_safe_url("http://example.com/api")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(UnsafeURLError, match="https scheme"):
            assert_safe_url("ftp://example.com/file")

    def test_no_hostname_rejected(self):
        with pytest.raises(UnsafeURLError, match="no hostname"):
            assert_safe_url("https:///path")

    def test_unresolvable_host_rejected(self, mocker):
        mocker.patch(
            "socket.getaddrinfo",
            side_effect=socket.gaierror("Name or service not known"),
        )
        with pytest.raises(UnsafeURLError, match="Cannot resolve"):
            assert_safe_url("https://this-host-does-not-exist.invalid/")

    def test_public_ip_passes(self, mocker):
        mocker.patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, None, None, None, ("93.184.216.34", 0))],
        )
        assert_safe_url("https://example.com/api")  # should not raise

    def test_public_ip_literal_passes(self, mocker):
        mocker.patch(
            "socket.getaddrinfo",
            return_value=[(socket.AF_INET, None, None, None, ("1.1.1.1", 0))],
        )
        assert_safe_url("https://1.1.1.1/")  # should not raise


class TestBlockedAddressRanges:
    """#218: 列挙型のブロックリストから抜けていたアドレス。getaddrinfo の結果で判定する。"""

    @staticmethod
    def _resolve_to(mocker, addr):
        family = socket.AF_INET6 if ":" in addr else socket.AF_INET
        mocker.patch("socket.getaddrinfo", return_value=[(family, None, None, None, (addr, 0))])

    @pytest.mark.parametrize(
        "addr",
        [
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
            "::ffff:169.254.169.254",  # IPv4-mapped metadata
            "::ffff:10.0.0.5",  # IPv4-mapped private
            "::",  # unspecified (Linux では localhost に接続される)
            "0.0.0.0",
            "100.64.0.1",  # CGNAT (Tailscale)
            "100.127.255.254",
            "198.18.0.1",  # benchmark
            "224.0.0.1",  # multicast
            "ff02::1",  # IPv6 multicast
            "240.0.0.1",  # reserved
            "64:ff9b::7f00:1",  # NAT64
            "64:ff9b:1::1",  # NAT64 local-use
            "2002:7f00:1::",  # 6to4 → 127.0.0.1
            "2002:0a00:0001::",  # 6to4 → 10.0.0.1
            "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo
            "fe80::1",
            "fc00::1",
            "::1",
        ],
    )
    def test_blocked(self, mocker, addr):
        self._resolve_to(mocker, addr)
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://example.com/")

    @pytest.mark.parametrize(
        "addr",
        [
            "93.184.216.34",
            "1.1.1.1",
            "2606:4700:4700::1111",  # public IPv6
            "::ffff:93.184.216.34",  # IPv4-mapped but public
            "2002:5db8:d822::",  # 6to4 → 93.184.216.34 (public)
            "100.63.255.255",  # CGNAT の直前 (公開)
            "100.128.0.1",  # CGNAT の直後 (公開)
        ],
    )
    def test_public_addresses_still_pass(self, mocker, addr):
        self._resolve_to(mocker, addr)
        assert_safe_url("https://example.com/")  # should not raise

    def test_any_blocked_answer_among_several_blocks_the_url(self, mocker):
        """DNS が公開 IP と内部 IP を両方返すとき、1 つでも内部なら拒否する。"""
        mocker.patch(
            "socket.getaddrinfo",
            return_value=[
                (socket.AF_INET, None, None, None, ("93.184.216.34", 0)),
                (socket.AF_INET6, None, None, None, ("::ffff:127.0.0.1", 0, 0, 0)),
            ],
        )
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://example.com/")


class TestExternalSsrf:
    """Integration-level: _fetch_source rejects unsafe source URLs."""

    def test_private_url_in_source_is_blocked(self, mocker):
        source = {
            "url": "https://192.168.1.100/internal",
            "type": "external-api",
            "render": False,
            "check_interval": "weekly",
            "tags": [],
            "last_etag": "",
            "last_synced": "",
        }
        mocker.patch("ingest.external._load_sources", return_value=[source])
        mocker.patch("ingest.external._save_sources")
        mocker.patch("ingest.external.ids_by_payload", return_value=[])
        mocker.patch("ingest.external.upsert")

        from ingest.external import sync_external

        sync_external()

        import ingest.external as ext

        ext.upsert.assert_not_called()

    def test_http_url_in_source_is_blocked(self, mocker):
        source = {
            "url": "http://example.com/api",
            "type": "external-api",
            "render": False,
            "check_interval": "weekly",
            "tags": [],
            "last_etag": "",
            "last_synced": "",
        }
        mocker.patch("ingest.external._load_sources", return_value=[source])
        mocker.patch("ingest.external._save_sources")
        mocker.patch("ingest.external.ids_by_payload", return_value=[])
        mocker.patch("ingest.external.upsert")

        from ingest.external import sync_external

        sync_external()

        import ingest.external as ext

        ext.upsert.assert_not_called()


class TestSafeGet:
    """safe_get() manually follows redirects, re-validating every hop (#149)."""

    @respx.mock
    def test_no_redirect_passes_through(self):
        respx.get("https://example.com/api").mock(return_value=httpx.Response(200, text="ok"))
        resp = safe_get("https://example.com/api")
        assert resp.status_code == 200
        assert resp.text == "ok"

    @respx.mock
    def test_redirect_to_safe_host_is_followed(self):
        respx.get("https://example.com/old").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/new"})
        )
        respx.get("https://example.com/new").mock(return_value=httpx.Response(200, text="moved"))
        resp = safe_get("https://example.com/old")
        assert resp.status_code == 200
        assert resp.text == "moved"

    @respx.mock
    def test_redirect_to_private_ip_is_blocked(self):
        respx.get("https://example.com/redir").mock(
            return_value=httpx.Response(302, headers={"location": "https://192.168.1.1/internal"})
        )
        with pytest.raises(UnsafeURLError, match="blocked"):
            safe_get("https://example.com/redir")

    @respx.mock
    def test_redirect_downgrading_to_http_is_blocked(self):
        respx.get("https://example.com/redir").mock(
            return_value=httpx.Response(302, headers={"location": "http://example.com/plain"})
        )
        with pytest.raises(UnsafeURLError, match="https scheme"):
            safe_get("https://example.com/redir")

    @respx.mock
    def test_redirect_with_no_location_header_is_blocked(self):
        respx.get("https://example.com/redir").mock(return_value=httpx.Response(302))
        with pytest.raises(UnsafeURLError, match="no Location"):
            safe_get("https://example.com/redir")

    @respx.mock
    def test_relative_redirect_location_resolved_against_current_url(self):
        respx.get("https://example.com/a/old").mock(
            return_value=httpx.Response(302, headers={"location": "/a/new"})
        )
        respx.get("https://example.com/a/new").mock(return_value=httpx.Response(200, text="ok"))
        resp = safe_get("https://example.com/a/old")
        assert resp.status_code == 200

    @respx.mock
    def test_redirect_chain_within_limit_succeeds(self):
        respx.get("https://example.com/hop0").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/hop1"})
        )
        respx.get("https://example.com/hop1").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/hop2"})
        )
        respx.get("https://example.com/hop2").mock(return_value=httpx.Response(200, text="done"))
        resp = safe_get("https://example.com/hop0", max_redirects=3)
        assert resp.text == "done"

    @respx.mock
    def test_too_many_redirects_raises(self):
        for i in range(MAX_REDIRECTS + 1):
            respx.get(f"https://example.com/hop{i}").mock(
                return_value=httpx.Response(
                    302, headers={"location": f"https://example.com/hop{i + 1}"}
                )
            )
        with pytest.raises(UnsafeURLError, match="too many redirects"):
            safe_get("https://example.com/hop0")


class TestArxivSsrf:
    """ingest_arxiv tech-blog path rejects unsafe URLs."""

    def test_private_blog_url_raises(self):
        from ingest.arxiv import ingest_arxiv

        with pytest.raises(UnsafeURLError):
            ingest_arxiv("https://192.168.1.1/blog")

    def test_http_blog_url_raises(self):
        from ingest.arxiv import ingest_arxiv

        with pytest.raises(UnsafeURLError):
            ingest_arxiv("http://example.com/blog")


class TestSafeDownload:
    """#214: safe_download() は最初の URL も検証し、リダイレクトを各ホップ再検証し、サイズを制限する。"""

    @pytest.fixture(autouse=True)
    def _no_real_network(self):
        """未登録のリクエストは接続せず即失敗させる。ガードが壊れても実ネットワークに出ない。"""
        with respx.mock:
            yield

    @pytest.fixture(autouse=True)
    def _public_dns(self, mocker):
        """名前は公開 IP に解決させる (実 DNS に依存しない)。IP リテラルはそのまま判定される。"""
        real = socket.getaddrinfo

        def fake(host, *a, **kw):
            try:
                import ipaddress

                ipaddress.ip_address(host)
                return real(host, *a, **kw)
            except ValueError:
                return [(socket.AF_INET, None, None, None, ("93.184.216.34", 0))]

        mocker.patch("socket.getaddrinfo", side_effect=fake)

    @staticmethod
    def _dl(url, **kw):
        import io

        buf = io.BytesIO()
        n = safe_download(url, buf, **kw)
        return n, buf.getvalue()

    def test_downloads_the_body(self):
        respx.get("https://example.com/a.pdf").mock(
            return_value=httpx.Response(200, content=b"%PDF-1")
        )
        assert self._dl("https://example.com/a.pdf") == (6, b"%PDF-1")

    @pytest.mark.parametrize(
        "url",
        [
            "https://10.0.0.1/a.pdf",
            "https://169.254.169.254/latest",
            "https://[::ffff:127.0.0.1]/a",
        ],
    )
    def test_first_url_is_validated_before_any_request(self, url):
        with pytest.raises(UnsafeURLError, match="blocked"):
            self._dl(
                url
            )  # 検証より前に接続しようとすれば respx が AllMockedAssertionError で落とす

    def test_plain_http_is_rejected(self):
        with pytest.raises(UnsafeURLError, match="https scheme"):
            self._dl("http://example.com/a.pdf")

    def test_redirect_to_a_private_address_is_blocked_and_not_requested(self):
        respx.get("https://example.com/r").mock(
            return_value=httpx.Response(302, headers={"location": "https://192.168.1.1/internal"})
        )
        internal = respx.get("https://192.168.1.1/internal").mock(
            return_value=httpx.Response(200, content=b"secret")
        )
        with pytest.raises(UnsafeURLError, match="blocked"):
            self._dl("https://example.com/r")
        assert not internal.called

    def test_redirect_chain_to_a_safe_host_is_followed(self):
        respx.get("https://example.com/old").mock(
            return_value=httpx.Response(301, headers={"location": "/new.pdf"})
        )
        respx.get("https://example.com/new.pdf").mock(
            return_value=httpx.Response(200, content=b"ok")
        )
        assert self._dl("https://example.com/old") == (2, b"ok")

    def test_too_many_redirects(self):
        respx.get(url__regex=r"https://example\.com/.*").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/loop"})
        )
        with pytest.raises(UnsafeURLError, match="too many redirects"):
            self._dl("https://example.com/start")

    def test_redirect_without_location_is_rejected(self):
        respx.get("https://example.com/r").mock(return_value=httpx.Response(302))
        with pytest.raises(UnsafeURLError, match="no Location"):
            self._dl("https://example.com/r")

    def test_declared_content_length_over_the_limit_is_rejected_before_reading(self):
        respx.get("https://example.com/big").mock(
            return_value=httpx.Response(
                200, headers={"content-length": "1000"}, content=b"x" * 1000
            )
        )
        import io

        buf = io.BytesIO()
        with pytest.raises(UnsafeURLError, match="too large"):
            safe_download("https://example.com/big", buf, max_bytes=100)
        assert buf.getvalue() == b""  # 1 バイトも書いていない

    def test_streamed_body_over_the_limit_is_cut_off_without_a_content_length(self):
        """Content-Length が無い / 嘘でも、受信バイト数で止める。"""

        def chunks():
            for _ in range(50):
                yield b"x" * 10

        respx.get("https://example.com/stream").mock(
            return_value=httpx.Response(200, content=chunks())
        )
        import io

        buf = io.BytesIO()
        with pytest.raises(UnsafeURLError, match="exceeds"):
            safe_download("https://example.com/stream", buf, max_bytes=100)
        assert len(buf.getvalue()) <= 100

    def test_one_byte_over_the_limit_is_cut_off_in_the_stream_check(self):
        """境界: ちょうど 100 は通り、101 は止まる。Content-Length の無い (チャンク) 応答で検査する。"""
        import io

        def body(n):
            def gen():
                for _ in range(n):
                    yield b"x"

            return gen()

        respx.get("https://example.com/b100").mock(
            return_value=httpx.Response(200, content=body(100))
        )
        respx.get("https://example.com/b101").mock(
            return_value=httpx.Response(200, content=body(101))
        )
        assert safe_download("https://example.com/b100", io.BytesIO(), max_bytes=100) == 100
        with pytest.raises(UnsafeURLError, match="exceeds"):
            safe_download("https://example.com/b101", io.BytesIO(), max_bytes=100)

    def test_redirect_limit_bounds_the_number_of_requests(self):
        route = respx.get(url__regex=r"https://example\.com/.*").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/loop"})
        )
        with pytest.raises(UnsafeURLError, match="too many redirects"):
            self._dl("https://example.com/start")
        assert route.call_count == MAX_REDIRECTS  # それ以上は取りに行かない

    def test_a_chain_just_under_the_redirect_limit_succeeds(self):
        for n in range(MAX_REDIRECTS - 1):
            respx.get(f"https://example.com/h{n}").mock(
                return_value=httpx.Response(
                    302, headers={"location": f"https://example.com/h{n + 1}"}
                )
            )
        respx.get(f"https://example.com/h{MAX_REDIRECTS - 1}").mock(
            return_value=httpx.Response(200, content=b"end")
        )
        assert self._dl("https://example.com/h0") == (3, b"end")

    def test_body_exactly_at_the_limit_is_accepted(self):
        respx.get("https://example.com/edge").mock(
            return_value=httpx.Response(200, content=b"x" * 100)
        )
        assert self._dl("https://example.com/edge", max_bytes=100)[0] == 100

    def test_http_error_status_propagates(self):
        respx.get("https://example.com/missing").mock(return_value=httpx.Response(404))
        with pytest.raises(httpx.HTTPStatusError):
            self._dl("https://example.com/missing")

    def test_default_limit_is_50_mib_and_overridable_by_env(self, monkeypatch):
        import importlib

        import core.url_guard as g

        assert g.MAX_DOWNLOAD_BYTES == 50 * 1024 * 1024
        monkeypatch.setenv("MAX_DOWNLOAD_BYTES", "1234")
        assert importlib.reload(g).MAX_DOWNLOAD_BYTES == 1234
        monkeypatch.delenv("MAX_DOWNLOAD_BYTES")
        importlib.reload(g)
