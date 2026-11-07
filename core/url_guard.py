"""URL safety guard: reject non-HTTPS and private/reserved addresses."""

import ipaddress
import socket
from collections.abc import Iterator
from typing import BinaryIO
from urllib.parse import urljoin, urlparse

import httpcore
import httpx

from core.config import get_settings

MAX_REDIRECTS = 5

# NAT64 (RFC 6052 / 8215): 内部の IPv4 を埋め込んで到達できる。is_global が True を返すので別に拒否する。
_NAT64 = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


def _blocked_reason(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Why *ip* must not be fetched, or None if it is a public unicast address.

    An allow-list on ``is_global`` instead of a hand-written CIDR deny-list: private, loopback,
    link-local, CGNAT (100.64/10), benchmark (198.18/15), documentation, ... are all non-global.
    IPv6 forms that carry an IPv4 address are unwrapped so ``::ffff:127.0.0.1`` is judged as
    127.0.0.1; tunnels (Teredo, NAT64) are refused outright.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        for embedded in (ip.ipv4_mapped, ip.sixtofour):
            if embedded is not None:
                return _blocked_reason(embedded)
        if ip.teredo is not None:
            return "Teredo tunnel"
        if any(ip in net for net in _NAT64):
            return "NAT64"
    if ip.is_multicast:
        return "multicast"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_reserved:
        return "reserved"
    if not ip.is_global:
        return "non-public (private/loopback/link-local/shared/benchmark)"
    return None


class UnsafeURLError(ValueError):
    pass


def _safe_addresses(hostname: str, port: int | None = None) -> list[str]:
    """Resolve *hostname* and return its addresses, raising UnsafeURLError if any is blocked.

    Every address must be public: a name with one public and one private record is refused,
    since the connection could be made to either.
    """
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Cannot resolve {hostname!r}: {exc}") from exc
    addresses: list[str] = []
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        reason = _blocked_reason(ip)
        if reason is not None:
            raise UnsafeURLError(f"{hostname!r} resolves to blocked address {ip} ({reason})")
        if sockaddr[0] not in addresses:
            addresses.append(sockaddr[0])
    return addresses


def assert_safe_url(url: str) -> None:
    """Raise UnsafeURLError if url is not HTTPS or resolves to a private address."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise UnsafeURLError(f"URL must use https scheme: {url!r}")
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURLError(f"URL has no hostname: {url!r}")
    try:
        _safe_addresses(hostname)
    except UnsafeURLError as exc:
        raise UnsafeURLError(f"{exc}: {url!r}") from exc


class _PinnedBackend(httpcore.SyncBackend):
    """Connect only to addresses that were validated at connect time (DNS rebinding, #224).

    ``assert_safe_url`` resolves the name once and httpx resolves it again when it connects, so a
    DNS server that answers a public address first and a private one second (TTL 0) passes the
    check and is then fetched. Resolving, validating and connecting in one place closes that gap;
    TLS still verifies the certificate against the original hostname.
    """

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        last: Exception | None = None
        for ip in _safe_addresses(host, port):
            try:
                return super().connect_tcp(ip, port, timeout, local_address, socket_options)
            except httpcore.ConnectError as exc:
                last = exc
        raise last or UnsafeURLError(f"{host!r} has no address")


def _client(timeout: float) -> httpx.Client:
    """An httpx client that cannot be steered to a non-public address by DNS.

    Passing *transport* explicitly also makes httpx ignore ``HTTPS_PROXY``: through a proxy the
    connection goes to the proxy and the proxy does the DNS lookup, which this guard cannot check.
    """
    transport = httpx.HTTPTransport()
    transport._pool._network_backend = _PinnedBackend()  # httpx has no public hook for this
    return httpx.Client(transport=transport, timeout=timeout)


def _capped_chunks(resp: httpx.Response, limit: int, url: str) -> Iterator[bytes]:
    """Yield the decoded body of a streaming *resp*, refusing to go past *limit* bytes.

    The limit is checked against ``Content-Length`` up front and against the **decoded** stream
    as it arrives, so a missing or lying header — or a small gzip body that expands to gigabytes —
    cannot bypass it.
    """
    declared = resp.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise UnsafeURLError(f"response too large ({declared} bytes > {limit}): {url!r}")
    written = 0
    for chunk in resp.iter_bytes():
        written += len(chunk)
        if written > limit:
            raise UnsafeURLError(f"response exceeds {limit} bytes: {url!r}")
        yield chunk


# 本文を読み込み済みの Response に作り直すとき、元のエンコード関連のヘッダは外す
# (本文は展開済み。残すと Response が二重に展開しようとして壊れる)。
_BODY_HEADERS = {"content-encoding", "content-length", "transfer-encoding"}


def safe_get(
    url: str,
    *,
    headers: dict | None = None,
    timeout: float = 30,
    max_redirects: int = MAX_REDIRECTS,
    max_bytes: int | None = None,
) -> httpx.Response:
    """GET *url* with manually-followed, re-validated redirects and a body size limit.

    ``httpx``'s built-in ``follow_redirects=True`` only validates the
    starting URL — a redirect response's ``Location`` is never checked, so a
    URL that passed ``assert_safe_url`` at call time can still redirect the
    request to a private address or metadata endpoint (#149). This calls
    ``assert_safe_url`` on every redirect hop before following it.

    The body is streamed and refused past *max_bytes* (default ``MAX_DOWNLOAD_BYTES``,
    decoded size) instead of being read whole into memory (#224). Raises
    :class:`UnsafeURLError` when it is exceeded. Non-2xx responses are returned, not raised.

    Does not itself validate *url* — call ``assert_safe_url(url)`` first if
    *url* is not already trusted.
    """
    limit = get_settings().max_download_bytes if max_bytes is None else max_bytes
    current = url
    for _ in range(max_redirects):
        with (
            _client(timeout) as client,
            client.stream("GET", current, headers=headers, follow_redirects=False) as resp,
        ):
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise UnsafeURLError(f"redirect with no Location header: {current!r}")
                next_url = urljoin(current, location)
                assert_safe_url(next_url)
                current = next_url
                continue
            body = b"".join(_capped_chunks(resp, limit, current))
            kept = [(k, v) for k, v in resp.headers.multi_items() if k.lower() not in _BODY_HEADERS]
            return httpx.Response(
                resp.status_code, headers=kept, content=body, request=resp.request
            )
    raise UnsafeURLError(f"too many redirects (> {max_redirects}) starting from {url!r}")


def safe_download(
    url: str,
    fileobj: BinaryIO,
    *,
    headers: dict | None = None,
    timeout: float = 60,
    max_bytes: int | None = None,
    max_redirects: int = MAX_REDIRECTS,
) -> int:
    """Stream *url* into *fileobj* and return the number of bytes written.

    Unlike :func:`safe_get` this validates the **first** URL too (it is the trust boundary for
    user-supplied URLs), re-validates every redirect hop, and stops once more than *max_bytes*
    (default ``MAX_DOWNLOAD_BYTES``) have been received — checked against ``Content-Length``
    up front and against the decoded stream, so a missing or lying header cannot bypass it.
    Raises :class:`UnsafeURLError` for a blocked/oversized/too-redirected URL and lets
    ``httpx.HTTPStatusError`` propagate for non-2xx.
    """
    limit = get_settings().max_download_bytes if max_bytes is None else max_bytes
    current = url
    assert_safe_url(current)
    for _ in range(max_redirects):
        with (
            _client(timeout) as client,
            client.stream("GET", current, headers=headers, follow_redirects=False) as resp,
        ):
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise UnsafeURLError(f"redirect with no Location header: {current!r}")
                next_url = urljoin(current, location)
                assert_safe_url(next_url)
                current = next_url
                continue
            resp.raise_for_status()
            written = 0
            for chunk in _capped_chunks(resp, limit, current):
                written += len(chunk)
                fileobj.write(chunk)
            return written
    raise UnsafeURLError(f"too many redirects (> {max_redirects}) starting from {url!r}")
