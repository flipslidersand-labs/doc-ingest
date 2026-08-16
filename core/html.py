"""HTML → Markdown extraction for external doc sources.

External API docs are served as HTML pages wrapped in large navigation /
boilerplate. Feeding the raw HTML to the markdown chunker produces one giant
low-signal chunk. trafilatura extracts the main content and emits markdown,
which the section chunker can then split on real headings.
"""
import trafilatura

# Content-Type substrings that indicate HTML we should convert.
_HTML_HINTS = ("text/html", "application/xhtml")


def looks_like_html(content_type: str, body: str) -> bool:
    """Heuristic: is this response an HTML page rather than markdown/plain text?"""
    ct = (content_type or "").lower()
    if any(h in ct for h in _HTML_HINTS):
        return True
    head = body[:512].lstrip().lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def extract_markdown(html: str) -> str | None:
    """Extract main-content markdown from an HTML page.

    Returns None when trafilatura cannot find extractable content, so callers
    can decide whether to fall back to the raw body.
    """
    return trafilatura.extract(
        html,
        output_format="markdown",
        include_links=False,
        include_tables=True,
        include_comments=False,
    )
