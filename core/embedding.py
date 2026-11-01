"""Embedding acquisition: HTTP embedding service (with fallback + retry) or local ONNX.

Split out of core/qdrant.py (#155): everything about *getting vectors* lives here;
core/qdrant.py is the thin vector-DB client that calls embed().
"""

import os
import threading

from core.logging import get_logger
from core.retry import with_retry

_log = get_logger(__name__)

EMBED_URL = os.getenv("EMBED_URL", "http://localhost:9092/embed/batch")
# EMBED_FALLBACK_URL: secondary endpoint used when EMBED_URL is unreachable or returns 501.
# Unset by default — when absent, behaviour is identical to before this change.
# C案: EMBED_URL=dev-nodee:9093 (GPU primary), EMBED_FALLBACK_URL=MINIPC:9092 (INT8 fallback)
EMBED_FALLBACK_URL = os.getenv("EMBED_FALLBACK_URL", "")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or None  # None → no X-API-Key header sent
EMBED_COLLECTION = os.getenv("EMBED_COLLECTION", "sessions")  # embedding-svc のモデルルーティング用
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "180"))  # CPU e5 のコールドロードを許容
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))  # 1 POST あたりの最大テキスト数
if EMBED_BATCH < 1:
    # embed()'s range(0, len(texts), EMBED_BATCH) raises an opaque ValueError
    # for 0 and silently returns an empty batch list for negative values
    # (upsert() then zips 0 vectors against N points, dropping payloads
    # silently) — fail loudly and immediately at import time instead (#135).
    raise ValueError(f"EMBED_BATCH must be >= 1, got {EMBED_BATCH}")
EMBED_RETRIES = 3

# EMBED_BACKEND=onnx_gpu: bypass HTTP and infer locally with onnxruntime CUDA EP.
# Requires: ONNX_MODEL_PATH, LD_LIBRARY_PATH with CUDA/cuDNN libs (see .env.example).
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "http")  # "http" | "onnx_gpu"
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "")
# HTTP embedding-svc は mode="index" (_post_embed が固定で送る) で e5 の "passage: " を付けてから推論する。
# ローカル ONNX 経路もそれに揃える。付けないと同じテキストでも cosine が 0.96〜0.99 ずれ、
# HTTP で作った collection と混在すると検索順位が変わる (#156 で実測)。
ONNX_TEXT_PREFIX = "passage: "

# Lazy ONNX session + tokenizer — initialized once on first embed call.
_onnx_sess = None
_onnx_tok = None
_onnx_input_names: frozenset[str] = frozenset()
_onnx_lock = threading.Lock()


def _onnx_init() -> None:
    """Lazily initialize ONNX session + tokenizer (called once, under lock)."""
    global _onnx_sess, _onnx_tok, _onnx_input_names
    import onnxruntime as ort
    from transformers import AutoTokenizer

    if not ONNX_MODEL_PATH:
        raise RuntimeError("ONNX_MODEL_PATH is not set. Cannot use EMBED_BACKEND=onnx_gpu.")
    model_dir = os.path.dirname(os.path.abspath(ONNX_MODEL_PATH))
    _log.info("Loading ONNX model from %s …", ONNX_MODEL_PATH)
    _onnx_sess = ort.InferenceSession(
        ONNX_MODEL_PATH,
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


def _embed_batch_onnx(texts: list[str]) -> list[list[float]]:
    """Embed texts locally using onnxruntime (CUDA EP on dev-nodee GTX1080)."""
    import numpy as np

    sess, tok, input_names = _onnx_resources()
    texts = [ONNX_TEXT_PREFIX + t for t in texts]
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")

    # Always keep attention_mask from enc (may be absent from ONNX input_names).
    mask = enc["attention_mask"].astype(np.float32)  # (batch, seq)
    feed = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}
    out = sess.run(None, feed)[0]  # (batch, seq, hidden)

    # Mean-pool over non-padding tokens then L2-normalize (e5 convention).
    pooled = (out * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
    return pooled.tolist()


def _post_embed(url: str, texts: list[str]) -> list[list[float]]:
    """POST to *url* and return vectors. Raises on network error or non-2xx."""
    import httpx

    resp = httpx.post(
        url,
        json={"texts": texts, "collection": EMBED_COLLECTION, "mode": "index"},
        headers={"X-API-Key": EMBED_API_KEY} if EMBED_API_KEY else {},
        timeout=EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["vectors"]


def _embed_retryable(e: Exception) -> bool:
    import httpx

    if isinstance(e, httpx.HTTPStatusError):
        # 501 = collection not served by this backend → no retry, move to the next url
        return e.response.status_code != 501
    return isinstance(e, (httpx.TimeoutException, httpx.ConnectError))


def _embed_batch(texts: list[str]) -> list[list[float]]:
    from functools import partial

    import httpx

    urls = [u for u in [EMBED_URL, EMBED_FALLBACK_URL] if u]
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
                partial(_post_embed, url, texts),
                retries=EMBED_RETRIES,
                is_retryable=_embed_retryable,
                on_retry=_log_retry,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 501:
                _log.info("embed: primary returned 501, switching to fallback")
            last_exc = e
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            _log.warning("embed: %s unreachable (%s), trying fallback", url, type(e).__name__)
            last_exc = e
        else:
            if url != EMBED_URL:
                _log.warning("embed: using fallback %s (primary unavailable)", url)
            return vectors
    raise last_exc


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in bounded sub-batches so a single POST stays within timeout."""
    batch_fn = _embed_batch_onnx if EMBED_BACKEND == "onnx_gpu" else _embed_batch
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(batch_fn(texts[i : i + EMBED_BATCH]))
    return vectors
