"""Tests for core/html.py"""
from core.html import extract_markdown, looks_like_html

SIMPLE_HTML = """<!DOCTYPE html>
<html><head><title>Test</title></head>
<body><article><h1>Hello</h1><p>This is real content for extraction.</p></article></body>
</html>"""


class TestLooksLikeHtml:
    def test_text_html_content_type(self):
        assert looks_like_html("text/html; charset=utf-8", "") is True

    def test_xhtml_content_type(self):
        assert looks_like_html("application/xhtml+xml", "") is True

    def test_json_content_type(self):
        assert looks_like_html("application/json", "{}") is False

    def test_empty_content_type_doctype_body(self):
        assert looks_like_html("", "<!DOCTYPE html><html>") is True

    def test_empty_content_type_html_tag_body(self):
        assert looks_like_html("", "<html><body>hi</body></html>") is True

    def test_plain_markdown_body(self):
        assert looks_like_html("", "## Section\n\nsome text") is False

    def test_content_type_takes_precedence_over_body(self):
        # text/html even with empty body → True
        assert looks_like_html("text/html", "") is True

    def test_whitespace_before_doctype(self):
        assert looks_like_html("", "  \n<!doctype html><html>") is True

    def test_empty_both(self):
        assert looks_like_html("", "") is False


class TestExtractMarkdown:
    def test_returns_string_for_real_html(self):
        result = extract_markdown(SIMPLE_HTML)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_none_for_js_only_page(self):
        js_only = "<html><body><script>app.init()</script></body></html>"
        result = extract_markdown(js_only)
        # trafilatura returns None when no extractable content
        assert result is None or result.strip() == ""

    def test_extracts_main_content(self):
        result = extract_markdown(SIMPLE_HTML)
        assert result is not None
        assert "Hello" in result or "real content" in result
