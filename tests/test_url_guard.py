"""Tests for core/url_guard.py"""
import socket

import pytest

from core.url_guard import UnsafeURLError, assert_safe_url


class TestAssertSafeUrl:
    def test_private_ip_rfc1918_10(self):
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://10.0.0.1/secret")

    def test_private_ip_rfc1918_172(self):
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://172.16.0.1/secret")

    def test_private_ip_rfc1918_192(self):
        with pytest.raises(UnsafeURLError, match="blocked"):
            assert_safe_url("https://192.168.68.1/admin")

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


class TestExternalSsrf:
    """Integration-level: _fetch_source rejects unsafe source URLs."""

    def test_private_url_in_source_is_blocked(self, mocker):
        source = {
            "url": "https://192.168.68.100/internal",
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
