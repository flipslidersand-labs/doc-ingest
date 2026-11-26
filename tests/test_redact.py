"""core/redact: ログに出す例外テキストから URL (= 認証情報になりうる) を伏せる (#224)。"""

import time

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


class TestRedactUrlsDelimiters:
    """#385: ``)`` ``]`` ``'`` ``"`` ``<`` ``>`` を生で含む userinfo・path の後半を残さない。"""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # URL の途中にある区切り文字は URL の一部として伏せる
            ("https://u:pa)ss@h/", "<url>"),
            ("https://u:pa]ss@h/", "<url>"),
            ("https://u:pa'ss@h/", "<url>"),
            ('https://u:pa"ss@h/', "<url>"),
            ("https://u:pa<ss@h/", "<url>"),
            ("https://u:pa>ss@h/", "<url>"),
            ("https://h/a)b/SECRET", "<url>"),
            ("GET https://u:pa)ss@h/x failed", "GET <url> failed"),
            # 末尾に付く句読点・括弧・引用符は URL に食い込ませず残す
            ("see https://h/SECRET.", "see <url>."),
            ("see https://h/SECRET, then", "see <url>, then"),
            ("at https://h/SECRET: failed", "at <url>: failed"),
            ("(https://h/SECRET)", "(<url>)"),
            ("[https://h/SECRET]", "[<url>]"),
            ("'https://h/SECRET'", "'<url>'"),
            ("for url 'https://h/SECRET').", "for url '<url>')."),
            ('("https://h/SECRET");', '("<url>");'),
            ("<https://h/SECRET>", "<<url>>"),
            ("{https://h/SECRET}", "{<url>}"),
            ("really https://h/SECRET?!", "really <url>?!"),
        ],
    )
    def test_cases(self, text, expected):
        assert redact_urls(text) == expected

    def test_no_userinfo_fragment_survives(self):
        for pw in ("pa)ss", "pa]ss", "pa'ss", 'pa"ss', "pa<ss", "pa>ss"):
            out = redact_urls(f"fail https://u:{pw}@h/ done")
            assert "ss@" not in out and "pa" not in out, (pw, out)
            assert out == "fail <url> done"

    def test_long_input_stays_linear(self):
        for text in ("https://" + ")" * 100_000, "https://h/" + "a)" * 50_000, "http://)" * 20_000):
            start = time.monotonic()
            redact_urls(text)
            assert time.monotonic() - start < 2.0
