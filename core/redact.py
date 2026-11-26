"""Keep secrets out of logs: strip URLs from exception text before logging it.

httpx's ``HTTPStatusError`` message is ``Client error '404 Not Found' for url '<the full URL>'``.
For a Discord webhook the URL *is* the credential (anyone holding it can post to the channel),
and for sources that carry a token in the query string it is too. Log the reason, not the URL.
"""

import re

# URL は空白でだけ終わる。`)` `]` `'` `"` `<` `>` は userinfo・path に生で入りうる(`https://u:pa)ss@h/`)ので、
# そこで打ち切ると後半(パスワードの続き)がログに残る。線形時間(入れ子の量指定子なし)。
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
# URL の直後に続く文の記号。URL の末尾に食い込んでいたら外して残す(`(https://h/x).` → `(<url>).`)。
_TRAILING = ".,;:!?)]}'\">"


def _redact(m: re.Match[str]) -> str:
    url = m.group(0)
    return "<url>" + url[len(url.rstrip(_TRAILING)) :]


def redact_urls(text: object) -> str:
    """Return ``str(text)`` with every ``http(s)://…`` replaced by ``<url>``."""
    return _URL_RE.sub(_redact, str(text))
