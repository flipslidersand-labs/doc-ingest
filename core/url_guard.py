"""URL safety guard: reject non-HTTPS and private/reserved addresses."""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx

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


def assert_safe_url(url: str) -> None:
    """Raise UnsafeURLError if url is not HTTPS or resolves to a private address."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise UnsafeURLError(f"URL must use https scheme: {url!r}")
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURLError(f"URL has no hostname: {url!r}")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Cannot resolve {hostname!r}: {exc}") from exc
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        reason = _blocked_reason(ip)
        if reason is not None:
            raise UnsafeURLError(f"URL resolves to blocked address {ip} ({reason}): {url!r}")


def safe_get(
    url: str,
    *,
    headers: dict | None = None,
    timeout: float = 30,
    max_redirects: int = MAX_REDIRECTS,
) -> httpx.Response:
    """GET *url* with manually-followed, re-validated redirects.

    ``httpx``'s built-in ``follow_redirects=True`` only validates the
    starting URL — a redirect response's ``Location`` is never checked, so a
    URL that passed ``assert_safe_url`` at call time can still redirect the
    request to a private address or metadata endpoint (#149). This calls
    ``assert_safe_url`` on every redirect hop before following it.

    Does not itself validate *url* — call ``assert_safe_url(url)`` first if
    *url* is not already trusted.
    """
    current = url
    for _ in range(max_redirects):
        resp = httpx.get(current, headers=headers, timeout=timeout, follow_redirects=False)
        if not resp.is_redirect:
            return resp
        location = resp.headers.get("location")
        if not location:
            raise UnsafeURLError(f"redirect with no Location header: {current!r}")
        next_url = urljoin(current, location)
        assert_safe_url(next_url)
        current = next_url
    raise UnsafeURLError(f"too many redirects (> {max_redirects}) starting from {url!r}")
