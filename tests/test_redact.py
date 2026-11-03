"""core/redact: ログに出す例外テキストから URL (= 認証情報になりうる) を伏せる (#224)。"""

import httpx
import pytest

from core.redact import redact_urls


class TestRedactUrls:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("https://discord.com/api/webhooks/1/SECRET", "<url>"),
            ("http://example.com/a?token=SECRET&x=1", "<url>"),
            (
                "Client error '404' for url 'https://h/p/SECRET'",
                "Client error '404' for url '<url>'",
            ),
            ('see "https://h/SECRET" now', 'see "<url>" now'),
            ("(https://h/SECRET)", "(<url>)"),
            ("<https://h/SECRET>", "<<url>>"),
            ("a https://one/S1 and http://two/S2 b", "a <url> and <url> b"),
            ("HTTPS://H/SECRET", "<url>"),
            ("no url here, HTTP 500", "no url here, HTTP 500"),
            ("", ""),
        ],
    )
    def test_cases(self, text, expected):
        assert redact_urls(text) == expected

    def test_accepts_an_exception(self):
        assert redact_urls(RuntimeError("fail at https://h/SECRET")) == "fail at <url>"

    def test_keeps_the_reason_and_status(self):
        err = httpx.HTTPStatusError(
            "Client error '404 Not Found' for url 'https://discord.com/api/webhooks/1/SECRET'",
            request=httpx.Request("POST", "https://discord.com/api/webhooks/1/SECRET"),
            response=httpx.Response(404),
        )
        out = redact_urls(err)
        assert "404 Not Found" in out and "SECRET" not in out

    def test_the_query_string_secret_goes_too(self):
        assert "SECRET" not in redact_urls("GET https://api.example.com/v1/x?api_key=SECRET failed")
