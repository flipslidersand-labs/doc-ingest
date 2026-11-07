"""Tests for core/url_guard.py"""

import io
import socket
from pathlib import Path

import httpx
import pytest
import respx

from core import qdrant
from core.url_guard import (
    MAX_REDIRECTS,
    UnsafeURLError,
    assert_safe_url,
    safe_download,
    safe_get,
)
from tests.conftest import override_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


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
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.upsert")

        from ingest.external import sync_external

        sync_external()

        qdrant.upsert.assert_not_called()

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
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mocker.patch("core.qdrant.upsert")

        from ingest.external import sync_external

        sync_external()

        qdrant.upsert.assert_not_called()


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

    def test_default_limit_is_50_mib_and_overridable_by_env(self):
        from core.config import load_settings

        assert load_settings({}).max_download_bytes == 50 * 1024 * 1024
        assert load_settings({"MAX_DOWNLOAD_BYTES": "1234"}).max_download_bytes == 1234


class TestSafeGetBodyLimit:
    """#224: safe_get は本文を全量メモリに読んでいた。ストリームで読み、上限 (展開後) を超えたら拒否する。"""

    @pytest.fixture(autouse=True)
    def _no_real_network(self):
        with respx.mock:
            yield

    @pytest.fixture
    def public_dns(self, mocker):
        """ホスト名は公開 IP に解決し、IP リテラルはそのまま判定させる (内部アドレスを公開扱いにしない)。"""
        import ipaddress
        import socket

        def fake(host, *a, **kw):
            try:
                ipaddress.ip_address(host)
                addr = host
            except ValueError:
                addr = "93.184.216.34"
            return [(socket.AF_INET, None, None, None, (addr, 0))]

        return mocker.patch("socket.getaddrinfo", side_effect=fake)

    @staticmethod
    def _stream(n, chunk=1):
        def gen():
            for _ in range(n):
                yield b"x" * chunk

        return gen()

    def test_a_declared_content_length_over_the_limit_is_refused(self):
        respx.get("https://example.com/big").mock(
            return_value=httpx.Response(
                200, headers={"content-length": "1000"}, content=b"x" * 1000
            )
        )
        with pytest.raises(UnsafeURLError, match="too large"):
            safe_get("https://example.com/big", max_bytes=100)

    def test_a_chunked_body_without_content_length_is_cut_off(self):
        respx.get("https://example.com/s").mock(
            return_value=httpx.Response(200, content=self._stream(500))
        )
        with pytest.raises(UnsafeURLError, match="exceeds"):
            safe_get("https://example.com/s", max_bytes=100)

    def test_the_boundary_exactly_at_the_limit_passes_and_one_over_fails(self):
        respx.get("https://example.com/b100").mock(
            return_value=httpx.Response(200, content=self._stream(100))
        )
        respx.get("https://example.com/b101").mock(
            return_value=httpx.Response(200, content=self._stream(101))
        )
        assert len(safe_get("https://example.com/b100", max_bytes=100).content) == 100
        with pytest.raises(UnsafeURLError, match="exceeds"):
            safe_get("https://example.com/b101", max_bytes=100)

    def test_a_small_gzip_body_that_expands_past_the_limit_is_refused(self):
        """圧縮後は小さいが展開すると巨大 (zip 爆弾)。展開後のバイト数で数える。"""
        import gzip

        packed = gzip.compress(b"A" * 100_000)
        assert len(packed) < 1000  # 圧縮後は小さい
        respx.get("https://example.com/bomb").mock(
            return_value=httpx.Response(200, headers={"content-encoding": "gzip"}, content=packed)
        )
        with pytest.raises(UnsafeURLError, match="exceeds"):
            safe_get("https://example.com/bomb", max_bytes=10_000)

    def test_a_gzip_body_within_the_limit_is_returned_decoded_exactly_once(self):
        """再構築した Response が、展開済みの本文をもう一度展開しようとして壊れない。"""
        import gzip

        respx.get("https://example.com/gz").mock(
            return_value=httpx.Response(
                200,
                headers={"content-encoding": "gzip", "content-type": "text/plain; charset=utf-8"},
                content=gzip.compress("日本語の本文".encode()),
            )
        )
        resp = safe_get("https://example.com/gz")
        assert resp.text == "日本語の本文"
        assert "content-encoding" not in resp.headers  # 二重展開させない

    def test_the_default_limit_applies_when_max_bytes_is_not_given(self, monkeypatch):
        """呼び出し側が max_bytes を指定しなくても、共通の既定値 (MAX_DOWNLOAD_BYTES) が効く。
        既定の 50MiB は試せないので、モジュールの値を小さくして確かめる。"""
        override_settings(max_download_bytes=100)
        respx.get("https://example.com/small").mock(
            return_value=httpx.Response(200, content=b"x" * 100)
        )
        respx.get("https://example.com/over").mock(
            return_value=httpx.Response(200, content=b"x" * 101)
        )
        assert len(safe_get("https://example.com/small").content) == 100
        with pytest.raises(UnsafeURLError):
            safe_get("https://example.com/over")  # max_bytes を渡していない

    def test_an_explicit_max_bytes_overrides_the_default(self, monkeypatch):
        override_settings(max_download_bytes=10)
        respx.get("https://example.com/m").mock(return_value=httpx.Response(200, content=b"x" * 50))
        assert len(safe_get("https://example.com/m", max_bytes=100).content) == 50

    def test_the_response_keeps_what_callers_use(self):
        respx.get("https://example.com/r").mock(
            return_value=httpx.Response(
                404, headers={"etag": '"abc"', "content-type": "text/html"}, content=b"nope"
            )
        )
        resp = safe_get("https://example.com/r")
        assert resp.status_code == 404 and resp.headers["etag"] == '"abc"'
        assert resp.is_success is False and resp.text == "nope"
        assert str(resp.url) == "https://example.com/r"
        with pytest.raises(httpx.HTTPStatusError):
            resp.raise_for_status()

    def test_redirects_are_still_followed_and_validated(self, public_dns):
        respx.get("https://example.com/old").mock(
            return_value=httpx.Response(301, headers={"location": "/new"})
        )
        respx.get("https://example.com/new").mock(
            return_value=httpx.Response(200, content=b"moved")
        )
        assert safe_get("https://example.com/old").content == b"moved"
        respx.get("https://example.com/evil").mock(
            return_value=httpx.Response(302, headers={"location": "https://192.168.1.1/x"})
        )
        with pytest.raises(UnsafeURLError, match="blocked"):
            safe_get("https://example.com/evil")

    def test_the_limit_applies_to_the_final_hop_not_the_redirect_body(self, public_dns):
        respx.get("https://example.com/a").mock(
            return_value=httpx.Response(302, headers={"location": "/b"}, content=b"r" * 500)
        )
        respx.get("https://example.com/b").mock(return_value=httpx.Response(200, content=b"x" * 50))
        assert len(safe_get("https://example.com/a", max_bytes=100).content) == 50


def _addr(ip):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return (family, socket.SOCK_STREAM, 6, "", (ip, 443))


class TestDnsRebinding:
    """#224: 検証した後に DNS が内部アドレスへ変わっても、そこへは接続しない。"""

    def test_connect_to_a_name_that_now_resolves_private_is_refused(self, mocker):
        from core.url_guard import _PinnedBackend

        mocker.patch("socket.getaddrinfo", return_value=[_addr("169.254.169.254")])
        base = mocker.patch("httpcore.SyncBackend.connect_tcp")

        with pytest.raises(UnsafeURLError, match="blocked"):
            _PinnedBackend().connect_tcp("rebind.example", 443)

        base.assert_not_called()

    def test_a_name_with_one_private_record_is_refused(self, mocker):
        from core.url_guard import _PinnedBackend

        mocker.patch("socket.getaddrinfo", return_value=[_addr("93.184.216.34"), _addr("10.0.0.5")])
        base = mocker.patch("httpcore.SyncBackend.connect_tcp")

        with pytest.raises(UnsafeURLError, match="blocked"):
            _PinnedBackend().connect_tcp("mixed.example", 443)

        base.assert_not_called()

    def test_connects_to_the_validated_address_not_the_name(self, mocker):
        from core.url_guard import _PinnedBackend

        mocker.patch("socket.getaddrinfo", return_value=[_addr("93.184.216.34")])
        base = mocker.patch("httpcore.SyncBackend.connect_tcp", return_value="stream")

        assert _PinnedBackend().connect_tcp("example.com", 443, 5.0) == "stream"

        assert base.call_args.args[:2] == ("93.184.216.34", 443)

    def test_falls_back_to_the_next_validated_address(self, mocker):
        import httpcore

        from core.url_guard import _PinnedBackend

        mocker.patch(
            "socket.getaddrinfo", return_value=[_addr("93.184.216.34"), _addr("93.184.216.35")]
        )
        base = mocker.patch(
            "httpcore.SyncBackend.connect_tcp", side_effect=[httpcore.ConnectError("down"), "ok"]
        )

        assert _PinnedBackend().connect_tcp("example.com", 443) == "ok"

        assert [c.args[0] for c in base.call_args_list] == ["93.184.216.34", "93.184.216.35"]

    def test_download_is_refused_when_dns_changes_after_the_check(self, mocker):
        """assert_safe_url は公開 IP を見て通し、接続時の再解決は内部 IP を返す。"""
        answers = iter([[_addr("93.184.216.34")], [_addr("127.0.0.1")], [_addr("127.0.0.1")]])
        mocker.patch("socket.getaddrinfo", side_effect=lambda *a, **k: next(answers))
        base = mocker.patch("httpcore.SyncBackend.connect_tcp")

        with pytest.raises(UnsafeURLError, match="blocked"):
            safe_download("https://rebind.example/x", io.BytesIO())

        base.assert_not_called()

    def test_environment_proxy_is_not_used(self, monkeypatch):
        """プロキシ経由だと DNS 解決を検証できない。"""
        from core.url_guard import _client

        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")

        with _client(5) as client:
            assert not client._mounts

    def test_duplicate_records_are_tried_once(self, mocker):
        import httpcore

        from core.url_guard import _PinnedBackend

        mocker.patch("socket.getaddrinfo", return_value=[_addr("93.184.216.34")] * 3)
        base = mocker.patch(
            "httpcore.SyncBackend.connect_tcp", side_effect=httpcore.ConnectError("down")
        )

        with pytest.raises(httpcore.ConnectError):
            _PinnedBackend().connect_tcp("example.com", 443)

        assert base.call_count == 1
