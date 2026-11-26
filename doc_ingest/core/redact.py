"""Keep secrets out of logs: strip URLs from exception text before logging it.

httpx's ``HTTPStatusError`` message is ``Client error '404 Not Found' for url '<the full URL>'``.
For a Discord webhook the URL *is* the credential (anyone holding it can post to the channel),
and for sources that carry a token in the query string it is too. Log the reason, not the URL.
"""

import re

# 空白・引用符・山括弧・閉じ括弧で URL を打ち切る。クエリやパスのトークンごと消す。
_URL_RE = re.compile(r"""https?://[^\s'"<>)\]]+""", re.IGNORECASE)


def redact_urls(text: object) -> str:
    """Return ``str(text)`` with every ``http(s)://…`` replaced by ``<url>``."""
    return _URL_RE.sub("<url>", str(text))
