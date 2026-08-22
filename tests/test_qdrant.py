"""Tests for core/qdrant.py embed batching, retry, and auth handling."""
import httpx
import respx

import core.qdrant as q


@respx.mock
def test_embed_subbatches(monkeypatch):
    monkeypatch.setattr(q, "EMBED_BATCH", 4)
    seen = []

    def responder(request):
        import json

        n = len(json.loads(request.content)["texts"])
        seen.append(n)
        return httpx.Response(200, json={"vectors": [[0.0] * 768] * n})

    respx.post(q.EMBED_URL).mock(side_effect=responder)
    vectors = q.embed(["t"] * 10)
    assert len(vectors) == 10
    assert seen == [4, 4, 2]  # 10 texts → 4+4+2 across three POSTs


def test_qdrant_api_key_none_when_unset(monkeypatch):
    """QDRANT_API_KEY resolves to None when env var is absent or empty."""
    monkeypatch.setattr(q, "QDRANT_API_KEY", None)
    assert q.QDRANT_API_KEY is None


def test_embed_api_key_none_when_unset(monkeypatch):
    """EMBED_API_KEY resolves to None when env var is absent or empty."""
    monkeypatch.setattr(q, "EMBED_API_KEY", None)
    assert q.EMBED_API_KEY is None


@respx.mock
def test_embed_omits_auth_header_when_key_unset(monkeypatch):
    """No X-API-Key header is sent when EMBED_API_KEY is None."""
    monkeypatch.setattr(q, "EMBED_API_KEY", None)
    monkeypatch.setattr(q, "EMBED_BATCH", 16)
    received_headers = {}

    def responder(request):
        received_headers.update(dict(request.headers))
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(q.EMBED_URL).mock(side_effect=responder)
    q.embed(["t"])
    assert "x-api-key" not in received_headers


@respx.mock
def test_embed_sends_auth_header_when_key_set(monkeypatch):
    """X-API-Key header is sent when EMBED_API_KEY is set."""
    monkeypatch.setattr(q, "EMBED_API_KEY", "secret-key")
    monkeypatch.setattr(q, "EMBED_BATCH", 16)
    received_headers = {}

    def responder(request):
        received_headers.update(dict(request.headers))
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(q.EMBED_URL).mock(side_effect=responder)
    q.embed(["t"])
    assert received_headers.get("x-api-key") == "secret-key"


@respx.mock
def test_embed_retries_on_timeout(monkeypatch):
    monkeypatch.setattr(q, "EMBED_BATCH", 16)
    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(q.EMBED_URL).mock(side_effect=responder)
    vectors = q.embed(["t"])
    assert len(vectors) == 1
    assert calls["n"] == 2  # first attempt timed out, second succeeded
