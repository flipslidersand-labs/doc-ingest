"""Tests for core/embedding.py: batching, retry, auth, primary→fallback routing (#155)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import core.embedding as emb


@respx.mock
def test_embed_subbatches(monkeypatch):
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 4)
    seen = []

    def responder(request):
        import json

        n = len(json.loads(request.content)["texts"])
        seen.append(n)
        return httpx.Response(200, json={"vectors": [[0.0] * 768] * n})

    respx.post(emb.EMBED_URL).mock(side_effect=responder)
    vectors = emb.embed(["t"] * 10)
    assert len(vectors) == 10
    assert seen == [4, 4, 2]  # 10 texts → 4+4+2 across three POSTs


def test_embed_api_key_none_when_unset(monkeypatch):
    """EMBED_API_KEY resolves to None when env var is absent or empty."""
    monkeypatch.setattr(emb, "EMBED_API_KEY", None)
    assert emb.EMBED_API_KEY is None


@respx.mock
def test_embed_omits_auth_header_when_key_unset(monkeypatch):
    """No X-API-Key header is sent when EMBED_API_KEY is None."""
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_API_KEY", None)
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    received_headers = {}

    def responder(request):
        received_headers.update(dict(request.headers))
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(emb.EMBED_URL).mock(side_effect=responder)
    emb.embed(["t"])
    assert "x-api-key" not in received_headers


@respx.mock
def test_embed_sends_auth_header_when_key_set(monkeypatch):
    """X-API-Key header is sent when EMBED_API_KEY is set."""
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_API_KEY", "secret-key")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    received_headers = {}

    def responder(request):
        received_headers.update(dict(request.headers))
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(emb.EMBED_URL).mock(side_effect=responder)
    emb.embed(["t"])
    assert received_headers.get("x-api-key") == "secret-key"


@respx.mock
def test_embed_retries_on_timeout(monkeypatch):
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(emb.EMBED_URL).mock(side_effect=responder)
    mock_sleep = MagicMock()
    with patch("time.sleep", mock_sleep):
        vectors = emb.embed(["t"])
    assert len(vectors) == 1
    assert calls["n"] == 2  # first attempt timed out, second succeeded
    mock_sleep.assert_called_once_with(1)  # 2**0 = 1s after first failure


PRIMARY_URL = "http://primary:9093/embed/batch"


FALLBACK_URL = "http://fallback:9092/embed/batch"


@respx.mock
def test_fallback_on_connect_error(monkeypatch):
    """When primary is unreachable, fallback URL is used."""
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    monkeypatch.setattr(emb, "EMBED_URL", PRIMARY_URL)
    monkeypatch.setattr(emb, "EMBED_FALLBACK_URL", FALLBACK_URL)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 1)

    respx.post(PRIMARY_URL).mock(side_effect=httpx.ConnectError("refused"))
    respx.post(FALLBACK_URL).mock(return_value=httpx.Response(200, json={"vectors": [[0.1] * 768]}))

    with patch("time.sleep"):
        vecs = emb.embed(["test"])
    assert len(vecs) == 1


@respx.mock
def test_fallback_on_501(monkeypatch):
    """When primary returns 501 (collection not served), fallback is tried immediately."""
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    monkeypatch.setattr(emb, "EMBED_URL", PRIMARY_URL)
    monkeypatch.setattr(emb, "EMBED_FALLBACK_URL", FALLBACK_URL)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 3)

    primary_calls = {"n": 0}

    def primary_responder(request):
        primary_calls["n"] += 1
        return httpx.Response(501, json={"detail": "not served"})

    respx.post(PRIMARY_URL).mock(side_effect=primary_responder)
    respx.post(FALLBACK_URL).mock(return_value=httpx.Response(200, json={"vectors": [[0.2] * 768]}))

    with patch("time.sleep"):
        vecs = emb.embed(["snippet code"])
    assert len(vecs) == 1
    assert primary_calls["n"] == 1  # no retry on 501


@respx.mock
def test_no_fallback_when_unset(monkeypatch):
    """When EMBED_FALLBACK_URL is empty, error propagates as before."""
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    monkeypatch.setattr(emb, "EMBED_URL", PRIMARY_URL)
    monkeypatch.setattr(emb, "EMBED_FALLBACK_URL", "")
    monkeypatch.setattr(emb, "EMBED_RETRIES", 1)

    respx.post(PRIMARY_URL).mock(side_effect=httpx.ConnectError("refused"))

    import pytest

    with patch("time.sleep"), pytest.raises(httpx.ConnectError):
        emb.embed(["test"])


@respx.mock
def test_primary_used_when_available(monkeypatch):
    """When primary is healthy, fallback URL is never called."""
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    monkeypatch.setattr(emb, "EMBED_URL", PRIMARY_URL)
    monkeypatch.setattr(emb, "EMBED_FALLBACK_URL", FALLBACK_URL)

    fallback_calls = {"n": 0}

    def fallback_responder(request):
        fallback_calls["n"] += 1
        return httpx.Response(200, json={"vectors": [[0.3] * 768]})

    respx.post(PRIMARY_URL).mock(return_value=httpx.Response(200, json={"vectors": [[0.4] * 768]}))
    respx.post(FALLBACK_URL).mock(side_effect=fallback_responder)

    vecs = emb.embed(["test"])
    assert len(vecs) == 1
    assert fallback_calls["n"] == 0


class TestEmbedBatchValidation:
    """core.embedding is a shared module object (sys.modules-cached) across every
    test file, and importlib.reload() executes top-to-bottom in-place —  a
    raised exception mid-reload still leaves EMBED_BATCH set to the bad
    value. Always reload back to a clean default afterward so other tests
    (in this file and others) don't inherit a broken module."""

    @pytest.fixture(autouse=True)
    def _restore_qdrant_module(self, monkeypatch):
        import importlib

        yield
        monkeypatch.delenv("EMBED_BATCH", raising=False)
        importlib.reload(emb)

    def test_zero_raises_at_import(self, monkeypatch):
        import importlib

        monkeypatch.setenv("EMBED_BATCH", "0")
        with pytest.raises(ValueError, match="EMBED_BATCH must be >= 1"):
            importlib.reload(emb)

    def test_negative_raises_at_import(self, monkeypatch):
        import importlib

        monkeypatch.setenv("EMBED_BATCH", "-1")
        with pytest.raises(ValueError, match="EMBED_BATCH must be >= 1"):
            importlib.reload(emb)

    def test_valid_value_does_not_raise(self, monkeypatch):
        import importlib

        monkeypatch.setenv("EMBED_BATCH", "32")
        importlib.reload(emb)
        assert emb.EMBED_BATCH == 32


@respx.mock
def test_embed_retries_on_http_500_then_succeeds(monkeypatch):
    """501 以外の HTTP エラーはリトライ対象(501 だけが即フォールバック)。"""
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 3)
    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(emb.EMBED_URL).mock(side_effect=responder)
    with patch("time.sleep") as sleep:
        assert len(emb.embed(["t"])) == 1
    assert calls["n"] == 2
    sleep.assert_called_once_with(1)


@respx.mock
def test_embed_http_500_exhausted_raises_status_error(monkeypatch):
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 3)
    monkeypatch.setattr(emb, "EMBED_FALLBACK_URL", "")
    route = respx.post(emb.EMBED_URL).mock(return_value=httpx.Response(500))
    with patch("time.sleep"), pytest.raises(httpx.HTTPStatusError):
        emb.embed(["t"])
    assert route.call_count == 3


@respx.mock
def test_embed_fallback_use_is_logged(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    monkeypatch.setattr(emb, "EMBED_URL", PRIMARY_URL)
    monkeypatch.setattr(emb, "EMBED_FALLBACK_URL", FALLBACK_URL)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 1)
    respx.post(PRIMARY_URL).mock(side_effect=httpx.ConnectError("refused"))
    respx.post(FALLBACK_URL).mock(return_value=httpx.Response(200, json={"vectors": [[0.1] * 768]}))
    with patch("time.sleep"), caplog.at_level(logging.WARNING, logger="core.embedding"):
        emb.embed(["t"])
    assert "using fallback" in caplog.text
    assert "unreachable" in caplog.text


@respx.mock
def test_embed_non_http_error_is_not_retried(monkeypatch):
    """想定外の例外(壊れたレスポンス等)は握りつぶさず即座に伝播する。"""
    monkeypatch.setattr(emb, "EMBED_BACKEND", "http")
    monkeypatch.setattr(emb, "EMBED_BATCH", 16)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 3)
    route = respx.post(emb.EMBED_URL).mock(return_value=httpx.Response(200, json={"nope": 1}))
    with patch("time.sleep") as sleep, pytest.raises(KeyError):
        emb.embed(["t"])
    assert route.call_count == 1
    sleep.assert_not_called()
