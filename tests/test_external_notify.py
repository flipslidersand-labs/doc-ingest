"""Tests for ingest.external._notify_discord (HTTP layer stubbed with respx)."""

import json
import logging

import httpx
import pytest
import respx

from ingest.external import _notify_discord

WEBHOOK = "https://discord.example.invalid/api/webhooks/0000/dummy-token"
TS = "2026-01-01 00:00 UTC"


@pytest.fixture
def webhook(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", WEBHOOK)


def _content(route) -> str:
    return json.loads(route.calls.last.request.content)["content"]


class TestNoWebhook:
    @respx.mock
    @pytest.mark.parametrize("unset", [True, False])
    def test_no_request_when_unset_or_empty(self, monkeypatch, unset):
        if unset:
            monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        else:
            monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
        route = respx.post(url__regex=r".*").mock(return_value=httpx.Response(204))
        _notify_discord(1, 2, [], [], TS)
        assert not route.called


class TestSend:
    @respx.mock
    def test_posts_json_to_webhook_url(self, webhook):
        route = respx.post(WEBHOOK).mock(return_value=httpx.Response(204))
        _notify_discord(3, 42, [], [], TS)
        assert route.call_count == 1
        req = route.calls.last.request
        assert req.headers["content-type"] == "application/json"
        assert set(json.loads(req.content)) == {"content"}

    @respx.mock
    def test_success_message(self, webhook):
        route = respx.post(WEBHOOK).mock(return_value=httpx.Response(204))
        _notify_discord(3, 42, [], [], TS)
        lines = _content(route).split("\n")
        assert lines == [
            f"✅ doc-ingest sync 完了 ({TS})",
            "synced: 3 sources / 42 chunks",
            "skipped: 0",
            "failed: 0",
        ]

    @respx.mock
    def test_failed_switches_status_and_lists_urls(self, webhook):
        route = respx.post(WEBHOOK).mock(return_value=httpx.Response(204))
        _notify_discord(1, 5, ["https://a.example/s"], ["https://b.example/f"], TS)
        lines = _content(route).split("\n")
        assert lines[0].startswith("⚠️")
        assert lines[2] == "skipped: 1 (https://a.example/s)"
        assert lines[3] == "failed: 1 (https://b.example/f)"

    @respx.mock
    def test_three_items_not_truncated(self, webhook):
        route = respx.post(WEBHOOK).mock(return_value=httpx.Response(204))
        _notify_discord(0, 0, ["a", "b", "c"], [], TS)
        assert _content(route).split("\n")[2] == "skipped: 3 (a, b, c)"

    @respx.mock
    def test_more_than_three_items_truncated_with_ellipsis(self, webhook):
        route = respx.post(WEBHOOK).mock(return_value=httpx.Response(204))
        items = [f"u{i}" for i in range(7)]
        _notify_discord(0, 0, items, items, TS)
        lines = _content(route).split("\n")
        # count is the full length; only the first 3 are listed
        assert lines[2] == "skipped: 7 (u0, u1, u2…)"
        assert lines[3] == "failed: 7 (u0, u1, u2…)"
        assert "u3" not in _content(route)

    @respx.mock
    def test_special_characters_are_json_escaped_and_preserved(self, webhook):
        route = respx.post(WEBHOOK).mock(return_value=httpx.Response(204))
        weird = 'https://x.example/?q="a"&b=<c>\\n日本語'
        _notify_discord(1, 1, [weird], [], TS)
        assert weird in _content(route)

    @respx.mock
    def test_message_length_is_not_capped(self, webhook):
        # Documents current behaviour: no Discord 2000-char limit handling exists.
        route = respx.post(WEBHOOK).mock(return_value=httpx.Response(204))
        long_url = "https://x.example/" + "a" * 3000
        _notify_discord(1, 1, [long_url], [], TS)
        assert len(_content(route)) > 2000


class TestErrorsSwallowed:
    @respx.mock
    @pytest.mark.parametrize("status", [400, 404, 429, 500])
    def test_http_error_status_not_raised_and_logged(self, webhook, caplog, status):
        respx.post(WEBHOOK).mock(return_value=httpx.Response(status))
        with caplog.at_level(logging.WARNING):
            _notify_discord(1, 1, [], [], TS)  # must not raise
        assert any("Discord notify failed" in r.getMessage() for r in caplog.records)

    @respx.mock
    @pytest.mark.parametrize(
        "exc",
        [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), RuntimeError("boom")],
    )
    def test_transport_exception_not_raised(self, webhook, exc):
        respx.post(WEBHOOK).mock(side_effect=exc)
        _notify_discord(1, 1, [], [], TS)  # must not raise

    @respx.mock
    def test_timeout_is_set(self, webhook, mocker):
        post = mocker.patch("ingest.external.httpx.post")
        _notify_discord(1, 1, [], [], TS)
        assert post.call_args.kwargs["timeout"] == 10
        assert post.call_args.args[0] == WEBHOOK
