"""URL safety guard: reject non-HTTPS and private/reserved addresses."""
import ipaddress
import socket
from urllib.parse import urlparse

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
                raise UnsafeURLError(
                    f"URL resolves to blocked address {ip} ({network}): {url!r}"
                )
