"""Tests for core/qdrant.py embed batching, retry, and auth handling."""
import threading
import time
from unittest.mock import MagicMock, patch

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
    mock_sleep = MagicMock()
    with patch("time.sleep", mock_sleep):
        vectors = q.embed(["t"])
    assert len(vectors) == 1
    assert calls["n"] == 2  # first attempt timed out, second succeeded
    mock_sleep.assert_called_once_with(1)  # 2**0 = 1s after first failure


# ---------------------------------------------------------------------------
# Tests for #81: thread-safe client() singleton and _collection_exists cache
# ---------------------------------------------------------------------------

def test_client_singleton_threadsafe(monkeypatch):
    """Concurrent calls to client() must produce exactly one QdrantClient."""
    # Reset module-level state so the test is independent.
    monkeypatch.setattr(q, "_client", None)

    created = []
    original_init = q.QdrantClient.__init__

    def slow_init(self, *args, **kwargs):
        time.sleep(0.05)  # simulate slow network handshake
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(q.QdrantClient, "__init__", slow_init)

    results = []

    def call_client():
        results.append(q.client())

    threads = [threading.Thread(target=call_client) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All calls must return the same object.
    assert len({id(r) for r in results}) == 1
    # QdrantClient.__init__ must have been called exactly once.
    assert len(created) == 1


def test_ensure_collection_calls_get_collections_once(monkeypatch):
    """ensure_collection() fetches get_collections at most once for a known name."""
    # Reset cache state.
    monkeypatch.setattr(q, "_known_collections", set())

    mock_col = MagicMock()
    mock_col.name = "test-col"

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [mock_col]
    monkeypatch.setattr(q, "_client", mock_client)

    # First call — should hit get_collections once and populate cache.
    q.ensure_collection("test-col")
    assert mock_client.get_collections.call_count == 1
    assert mock_client.create_collection.call_count == 0

    # Second call — cache hit, get_collections must NOT be called again.
    q.ensure_collection("test-col")
    assert mock_client.get_collections.call_count == 1


def test_collection_exists_uses_cache(monkeypatch):
    """_collection_exists() skips network call when collection is cached."""
    monkeypatch.setattr(q, "_known_collections", {"cached-col"})
    mock_client = MagicMock()
    monkeypatch.setattr(q, "_client", mock_client)

    assert q._collection_exists("cached-col") is True
    mock_client.get_collections.assert_not_called()


def test_collection_exists_cache_miss_fetches_remote(monkeypatch):
    """_collection_exists() fetches from Qdrant on cache miss."""
    monkeypatch.setattr(q, "_known_collections", set())

    mock_col = MagicMock()
    mock_col.name = "remote-col"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [mock_col]
    monkeypatch.setattr(q, "_client", mock_client)

    assert q._collection_exists("remote-col") is True
    mock_client.get_collections.assert_called_once()

    # Second call must use the cache.
    assert q._collection_exists("remote-col") is True
    mock_client.get_collections.assert_called_once()
