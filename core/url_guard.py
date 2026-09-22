"""URL safety guard: reject non-HTTPS and private/reserved addresses."""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx

MAX_REDIRECTS = 5

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


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
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                raise UnsafeURLError(f"URL resolves to blocked address {ip} ({network}): {url!r}")


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
