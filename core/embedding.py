"""Embedding acquisition: HTTP embedding service (with fallback + retry).

Split out of core/qdrant.py (#155): everything about *getting vectors* lives here;
core/qdrant.py is the thin vector-DB client that calls embed().
"""

import logging
from typing import Literal

from core.config import get_settings
from core.retry import with_retry

_log = logging.getLogger(__name__)

EMBED_RETRIES = 3

# e5 は文書に "passage: "、検索クエリに "query: " を付けて学習されている。HTTP embedding-svc は
# mode="index" / "search" でこれを付け替えてから推論する。
# 文書側 (upsert) は "index"、検索クエリ (search) は "search" (#222: 従来はクエリも "index" だった)。
_PURPOSES = frozenset({"index", "search"})  # prefix は サーバー側で付くので値は持たない
Mode = Literal["index", "search"]


class EmbeddingError(RuntimeError):
    """埋め込みの応答が不正 (ベクトル数が入力と合わない等)。"""


def _post_embed(url: str, texts: list[str], mode: Mode = "index") -> list[list[float]]:
    """POST to *url* and return vectors. Raises on network error or non-2xx."""
    import httpx

    cfg = get_settings()
    resp = httpx.post(
        url,
        json={"texts": texts, "collection": cfg.embed_collection, "mode": mode},
        headers={"X-API-Key": cfg.embed_api_key} if cfg.embed_api_key else {},
        timeout=cfg.embed_timeout,
    )
    resp.raise_for_status()
    return resp.json()["vectors"]


def _embed_retryable(e: Exception) -> bool:
    import httpx

    if isinstance(e, httpx.HTTPStatusError):
        # 501 = collection not served by this backend → no retry, move to the next url
        return e.response.status_code != 501
    return isinstance(e, httpx.TransportError)


def _embed_batch(texts: list[str], mode: Mode = "index") -> list[list[float]]:
    from functools import partial

    import httpx

    cfg = get_settings()

    urls = [u for u in [cfg.embed_url, cfg.embed_fallback_url] if u]
    last_exc: Exception = RuntimeError("no EMBED_URL configured")
    for url in urls:

        def _log_retry(attempt: int, e: Exception, url: str = url) -> None:
            _log.warning(
                "embed retry %d/%d — %d texts → %s (url=%s)",
                attempt + 1,
                EMBED_RETRIES,
                len(texts),
                type(e).__name__,
                url,
            )

        try:
            vectors = with_retry(
                partial(_post_embed, url, texts, mode),
                retries=EMBED_RETRIES,
                is_retryable=_embed_retryable,
                on_retry=_log_retry,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 501:
                _log.info("embed: primary returned 501, switching to fallback")
            last_exc = e
        except httpx.TransportError as e:  # timeout / connect / read / protocol / proxy ...
            _log.warning("embed: %s unreachable (%s), trying fallback", url, type(e).__name__)
            last_exc = e
        else:
            if url != cfg.embed_url:
                _log.warning("embed: using fallback %s (primary unavailable)", url)
            return vectors
    raise last_exc


def embed(texts: list[str], mode: Mode = "index") -> list[list[float]]:
    """Embed texts in bounded sub-batches so a single POST stays within timeout.

    *mode* is ``"index"`` for documents (e5 ``passage:``) and ``"search"`` for queries (``query:``).
    """
    if mode not in _PURPOSES:
        raise ValueError(f"mode must be one of {sorted(_PURPOSES)}, got {mode!r}")
    cfg = get_settings()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), cfg.embed_batch):
        batch = texts[i : i + cfg.embed_batch]
        got = _embed_batch(batch, mode)
        if len(got) != len(batch):
            # 少なく返ると、呼び出し側の zip が後ろの文書を黙って落とす (ペイロードの欠損)
            raise EmbeddingError(f"embedding returned {len(got)} vectors for {len(batch)} texts")
        vectors.extend(got)
    return vectors
