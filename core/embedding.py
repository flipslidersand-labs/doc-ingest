"""Embedding acquisition: HTTP embedding service (with fallback + retry) or local ONNX.

Split out of core/qdrant.py (#155): everything about *getting vectors* lives here;
core/qdrant.py is the thin vector-DB client that calls embed().
"""

import logging
import os
import threading
from typing import Literal

from core.config import get_settings
from core.retry import with_retry

_log = logging.getLogger(__name__)

EMBED_RETRIES = 3

# EMBED_BACKEND=onnx_gpu: bypass HTTP and infer locally with onnxruntime CUDA EP.
# Requires: ONNX_MODEL_PATH, LD_LIBRARY_PATH with CUDA/cuDNN libs (see .env.example).
# e5 は文書に "passage: "、検索クエリに "query: " を付けて学習されている。HTTP embedding-svc は
# mode="index" / "search" でこれを付け替えてから推論するので、ローカル ONNX 経路も同じ表で揃える。
# 付けないと同じテキストでも cosine が 0.96〜0.99 ずれ、HTTP と混在すると検索順位が変わる (#156 で実測)。
# 文書側 (upsert) は "index"、検索クエリ (search) は "search" (#222: 従来はクエリも "index" だった)。
E5_PREFIX = {"index": "passage: ", "search": "query: "}
Mode = Literal["index", "search"]

# Lazy ONNX session + tokenizer — initialized once on first embed call.
_onnx_sess = None
_onnx_tok = None
_onnx_input_names: frozenset[str] = frozenset()
_onnx_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    """埋め込みの応答が不正 (ベクトル数が入力と合わない等)。"""


def _onnx_init() -> None:
    """Lazily initialize ONNX session + tokenizer (called once, under lock)."""
    global _onnx_sess, _onnx_tok, _onnx_input_names
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_model_path = get_settings().onnx_model_path
    if not onnx_model_path:
        raise RuntimeError("ONNX_MODEL_PATH is not set. Cannot use EMBED_BACKEND=onnx_gpu.")
    model_dir = os.path.dirname(os.path.abspath(onnx_model_path))
    _log.info("Loading ONNX model from %s …", onnx_model_path)
    _onnx_sess = ort.InferenceSession(
        onnx_model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _onnx_input_names = frozenset(i.name for i in _onnx_sess.get_inputs())
    active_ep = _onnx_sess.get_providers()[0]
    if active_ep == "CUDAExecutionProvider":
        _log.info("ONNX session ready — active EP: %s", active_ep)
    else:
        # CUDA/cuDNN のライブラリ不足などで onnxruntime は黙って CPU に落ちる。気づけるよう警告する。
        _log.warning(
            "ONNX session ready but CUDAExecutionProvider is NOT active (using %s) — "
            "running on CPU. Check LD_LIBRARY_PATH / CUDA 12 / cuDNN 9.",
            active_ep,
        )
    _log.info("Loading tokenizer from %s …", model_dir)
    _onnx_tok = AutoTokenizer.from_pretrained(model_dir)
    _log.info("Tokenizer ready")


def _onnx_resources() -> tuple:
    """Return cached (session, tokenizer, input_names), initializing on first call."""
    if _onnx_sess is not None:
        return _onnx_sess, _onnx_tok, _onnx_input_names
    with _onnx_lock:
        if _onnx_sess is None:
            _onnx_init()
    return _onnx_sess, _onnx_tok, _onnx_input_names


def _embed_batch_onnx(texts: list[str], mode: Mode = "index") -> list[list[float]]:
    """Embed texts locally using onnxruntime (CUDA EP on dev-nodee GTX1080)."""
    import numpy as np

    sess, tok, input_names = _onnx_resources()
    texts = [E5_PREFIX[mode] + t for t in texts]
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")

    # Always keep attention_mask from enc (may be absent from ONNX input_names).
    mask = enc["attention_mask"].astype(np.float32)  # (batch, seq)
    feed = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}
    out = sess.run(None, feed)[0]  # (batch, seq, hidden)

    # Mean-pool over non-padding tokens then L2-normalize (e5 convention).
    pooled = (out * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
    return pooled.tolist()


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
    if mode not in E5_PREFIX:
        raise ValueError(f"mode must be one of {sorted(E5_PREFIX)}, got {mode!r}")
    cfg = get_settings()
    batch_fn = _embed_batch_onnx if cfg.embed_backend == "onnx_gpu" else _embed_batch
    vectors: list[list[float]] = []
    for i in range(0, len(texts), cfg.embed_batch):
        batch = texts[i : i + cfg.embed_batch]
        got = batch_fn(batch, mode)
        if len(got) != len(batch):
            # 少なく返ると、呼び出し側の zip が後ろの文書を黙って落とす (ペイロードの欠損)
            raise EmbeddingError(f"embedding returned {len(got)} vectors for {len(batch)} texts")
        vectors.extend(got)
    return vectors
