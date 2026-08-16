"""Tests for core/qdrant.py embed batching and retry."""
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
