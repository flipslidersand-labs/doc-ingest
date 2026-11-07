"""Tests for core/embedding.py: batching, retry, auth, primary→fallback routing (#155)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import core.embedding as emb
from core.config import get_settings
from tests.conftest import override_settings


@respx.mock
def test_embed_subbatches(monkeypatch):
    override_settings(embed_backend="http")
    override_settings(embed_batch=4)
    seen = []

    def responder(request):
        import json

        n = len(json.loads(request.content)["texts"])
        seen.append(n)
        return httpx.Response(200, json={"vectors": [[0.0] * 768] * n})

    respx.post(get_settings().embed_url).mock(side_effect=responder)
    vectors = emb.embed(["t"] * 10)
    assert len(vectors) == 10
    assert seen == [4, 4, 2]  # 10 texts → 4+4+2 across three POSTs


def test_embed_api_key_none_when_unset(monkeypatch):
    """EMBED_API_KEY resolves to None when env var is absent or empty."""
    override_settings(embed_api_key=None)
    assert get_settings().embed_api_key is None


@respx.mock
def test_embed_omits_auth_header_when_key_unset(monkeypatch):
    """No X-API-Key header is sent when EMBED_API_KEY is None."""
    override_settings(embed_backend="http")
    override_settings(embed_api_key=None)
    override_settings(embed_batch=16)
    received_headers = {}

    def responder(request):
        received_headers.update(dict(request.headers))
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(get_settings().embed_url).mock(side_effect=responder)
    emb.embed(["t"])
    assert "x-api-key" not in received_headers


@respx.mock
def test_embed_sends_auth_header_when_key_set(monkeypatch):
    """X-API-Key header is sent when EMBED_API_KEY is set."""
    override_settings(embed_backend="http")
    override_settings(embed_api_key="secret-key")
    override_settings(embed_batch=16)
    received_headers = {}

    def responder(request):
        received_headers.update(dict(request.headers))
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(get_settings().embed_url).mock(side_effect=responder)
    emb.embed(["t"])
    assert received_headers.get("x-api-key") == "secret-key"


@respx.mock
def test_embed_retries_on_timeout(monkeypatch):
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(get_settings().embed_url).mock(side_effect=responder)
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
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    override_settings(embed_url=PRIMARY_URL)
    override_settings(embed_fallback_url=FALLBACK_URL)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 1)

    respx.post(PRIMARY_URL).mock(side_effect=httpx.ConnectError("refused"))
    respx.post(FALLBACK_URL).mock(return_value=httpx.Response(200, json={"vectors": [[0.1] * 768]}))

    with patch("time.sleep"):
        vecs = emb.embed(["test"])
    assert len(vecs) == 1


@respx.mock
def test_fallback_on_501(monkeypatch):
    """When primary returns 501 (collection not served), fallback is tried immediately."""
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    override_settings(embed_url=PRIMARY_URL)
    override_settings(embed_fallback_url=FALLBACK_URL)
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
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    override_settings(embed_url=PRIMARY_URL)
    override_settings(embed_fallback_url="")
    monkeypatch.setattr(emb, "EMBED_RETRIES", 1)

    respx.post(PRIMARY_URL).mock(side_effect=httpx.ConnectError("refused"))

    import pytest

    with patch("time.sleep"), pytest.raises(httpx.ConnectError):
        emb.embed(["test"])


@respx.mock
def test_primary_used_when_available(monkeypatch):
    """When primary is healthy, fallback URL is never called."""
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    override_settings(embed_url=PRIMARY_URL)
    override_settings(embed_fallback_url=FALLBACK_URL)

    fallback_calls = {"n": 0}

    def fallback_responder(request):
        fallback_calls["n"] += 1
        return httpx.Response(200, json={"vectors": [[0.3] * 768]})

    respx.post(PRIMARY_URL).mock(return_value=httpx.Response(200, json={"vectors": [[0.4] * 768]}))
    respx.post(FALLBACK_URL).mock(side_effect=fallback_responder)

    vecs = emb.embed(["test"])
    assert len(vecs) == 1
    assert fallback_calls["n"] == 0


@respx.mock
def test_embed_retries_on_http_500_then_succeeds(monkeypatch):
    """501 以外の HTTP エラーはリトライ対象(501 だけが即フォールバック)。"""
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 3)
    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(get_settings().embed_url).mock(side_effect=responder)
    with patch("time.sleep") as sleep:
        assert len(emb.embed(["t"])) == 1
    assert calls["n"] == 2
    sleep.assert_called_once_with(1)


@respx.mock
def test_embed_http_500_exhausted_raises_status_error(monkeypatch):
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 3)
    override_settings(embed_fallback_url="")
    route = respx.post(get_settings().embed_url).mock(return_value=httpx.Response(500))
    with patch("time.sleep"), pytest.raises(httpx.HTTPStatusError):
        emb.embed(["t"])
    assert route.call_count == 3


@respx.mock
def test_embed_fallback_use_is_logged(monkeypatch, caplog):
    import logging

    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    override_settings(embed_url=PRIMARY_URL)
    override_settings(embed_fallback_url=FALLBACK_URL)
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
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 3)
    route = respx.post(get_settings().embed_url).mock(
        return_value=httpx.Response(200, json={"nope": 1})
    )
    with patch("time.sleep") as sleep, pytest.raises(KeyError):
        emb.embed(["t"])
    assert route.call_count == 1
    sleep.assert_not_called()


@respx.mock
def test_http_mode_is_sent_as_given_and_defaults_to_index(monkeypatch):
    """#222: 検索クエリは mode="search" で送る (サービスが query: を付ける)。既定は "index"。"""
    import json

    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    modes = []

    def responder(request):
        modes.append(json.loads(request.content)["mode"])
        return httpx.Response(200, json={"vectors": [[0.0] * 768]})

    respx.post(get_settings().embed_url).mock(side_effect=responder)
    emb.embed(["doc"])
    emb.embed(["doc"], mode="index")
    emb.embed(["a query"], mode="search")
    assert modes == ["index", "index", "search"]


@respx.mock
def test_search_mode_survives_the_fallback_url(monkeypatch):
    import json

    primary, fallback = "http://primary:9093/embed/batch", "http://fallback:9092/embed/batch"
    override_settings(embed_backend="http")
    override_settings(embed_batch=16)
    override_settings(embed_url=primary)
    override_settings(embed_fallback_url=fallback)
    monkeypatch.setattr(emb, "EMBED_RETRIES", 1)
    respx.post(primary).mock(side_effect=httpx.ConnectError("refused"))
    seen = []

    def fb(request):
        seen.append(json.loads(request.content)["mode"])
        return httpx.Response(200, json={"vectors": [[0.1] * 768]})

    respx.post(fallback).mock(side_effect=fb)
    with patch("time.sleep"):
        emb.embed(["q"], mode="search")
    assert seen == ["search"]


def test_an_unknown_mode_is_rejected_before_any_request(monkeypatch):
    monkeypatch.setattr(emb, "_embed_batch", lambda b, m: pytest.fail("must not be called"))
    with pytest.raises(ValueError, match="mode"):
        emb.embed(["x"], mode="query")  # type: ignore[arg-type]  # "search" が正しい綴り


def test_embed_passes_the_mode_to_every_sub_batch(monkeypatch):
    seen = []
    override_settings(embed_backend="http")
    override_settings(embed_batch=2)
    monkeypatch.setattr(emb, "_embed_batch", lambda b, m: seen.append(m) or [[0.0]] * len(b))
    emb.embed(["a", "b", "c", "d", "e"], mode="search")
    assert seen == ["search", "search", "search"]


class TestTransportErrorsAreRetriedAndFallBack:
    """#229: ReadError / RemoteProtocolError などはリトライもフォールバックも対象外だった。"""

    PRIMARY, FALLBACK = "http://primary:9093/embed/batch", "http://fallback:9092/embed/batch"

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        override_settings(embed_backend="http")
        override_settings(embed_batch=16)
        override_settings(embed_url=self.PRIMARY)
        override_settings(embed_fallback_url=self.FALLBACK)
        monkeypatch.setattr(emb, "EMBED_RETRIES", 2)

    @respx.mock
    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ReadError("connection reset"),
            httpx.RemoteProtocolError("server disconnected"),
            httpx.WriteError("broken pipe"),
            httpx.ProxyError("proxy"),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_falls_back_after_retrying_the_primary(self, exc):
        primary = respx.post(self.PRIMARY).mock(side_effect=exc)
        respx.post(self.FALLBACK).mock(
            return_value=httpx.Response(200, json={"vectors": [[0.1] * 4]})
        )
        with patch("time.sleep"):
            assert emb.embed(["t"]) == [[0.1] * 4]
        assert primary.call_count == 2  # EMBED_RETRIES 回リトライしてから

    @respx.mock
    def test_without_a_fallback_the_original_error_is_raised(self, monkeypatch):
        override_settings(embed_fallback_url="")
        respx.post(self.PRIMARY).mock(side_effect=httpx.ReadError("reset"))
        with patch("time.sleep"), pytest.raises(httpx.ReadError):
            emb.embed(["t"])

    @respx.mock
    def test_a_decoding_error_is_not_retried(self):
        """通信系ではない (壊れた応答) ので、リトライしても同じ。即座に伝播する。"""
        route = respx.post(self.PRIMARY).mock(side_effect=httpx.DecodingError("bad gzip"))
        with patch("time.sleep") as sleep, pytest.raises(httpx.DecodingError):
            emb.embed(["t"])
        assert route.call_count == 1
        sleep.assert_not_called()


class TestVectorCountIsChecked:
    """#229: サーバーが入力より少ないベクトルを返すと、呼び出し側の zip が後ろを黙って落とした。"""

    @respx.mock
    def test_http_short_response_raises(self, monkeypatch):
        override_settings(embed_backend="http")
        override_settings(embed_batch=16)
        respx.post(get_settings().embed_url).mock(
            return_value=httpx.Response(200, json={"vectors": [[0.0]]})
        )
        with pytest.raises(emb.EmbeddingError, match="1 vectors for 3 texts"):
            emb.embed(["a", "b", "c"])

    def test_onnx_short_response_raises(self, monkeypatch):
        override_settings(embed_backend="onnx_gpu")
        override_settings(embed_batch=16)
        monkeypatch.setattr(emb, "_embed_batch_onnx", lambda b, m: [[0.0]])
        with pytest.raises(emb.EmbeddingError):
            emb.embed(["a", "b"])

    def test_each_sub_batch_is_checked(self, monkeypatch):
        override_settings(embed_backend="http")
        override_settings(embed_batch=2)
        monkeypatch.setattr(emb, "_embed_batch", lambda b, m: [[0.0]] * (len(b) - (len(b) == 1)))
        with pytest.raises(emb.EmbeddingError, match="0 vectors for 1 texts"):
            emb.embed(["a", "b", "c"])  # 2 + 1: 2 つ目のバッチだけ不足

    def test_an_exact_count_passes(self, monkeypatch):
        override_settings(embed_backend="http")
        override_settings(embed_batch=2)
        monkeypatch.setattr(emb, "_embed_batch", lambda b, m: [[0.0]] * len(b))
        assert len(emb.embed(["a", "b", "c", "d", "e"])) == 5

    def test_no_texts_is_fine(self):
        assert emb.embed([]) == []
