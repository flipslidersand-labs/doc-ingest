"""Qdrant client wrapper targeting MINIPC e5 embedding service."""

import os
import threading
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from core.logging import get_logger

_log = get_logger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None  # None → unauthenticated (dev/localhost)
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:9092/embed/batch")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or None  # None → no X-API-Key header sent
EMBED_COLLECTION = os.getenv("EMBED_COLLECTION", "sessions")  # embedding-svc のモデルルーティング用
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "180"))  # CPU e5 のコールドロードを許容
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))  # 1 POST あたりの最大テキスト数
EMBED_RETRIES = 3
VECTOR_SIZE = 768  # multilingual-e5-base

# EMBED_BACKEND=onnx_gpu: bypass HTTP and infer locally with onnxruntime CUDA EP.
# Requires: ONNX_MODEL_PATH, LD_LIBRARY_PATH with CUDA/cuDNN libs (see .env.example).
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "http")  # "http" | "onnx_gpu"
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "")

# Lazy ONNX session — initialized once on first embed call.
_onnx_sess = None
_onnx_lock = threading.Lock()

_client: QdrantClient | None = None
_client_lock = threading.Lock()

# Process-level cache of known collection names.
# Populated lazily on first ensure_collection / existence check call.
# Guards against N+1 get_collections() round-trips when syncing N sources.
_known_collections: set[str] = set()
_known_collections_lock = threading.Lock()


def client() -> QdrantClient:
    global _client
    if _client is None:  # fast path (no lock)
        with _client_lock:
            if _client is None:  # double-checked locking
                _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def _collection_exists(name: str) -> bool:
    """Return True if *name* exists in Qdrant, using the process-level cache."""
    with _known_collections_lock:
        if name in _known_collections:
            return True
    # Cache miss — fetch from Qdrant and warm the cache for all collections.
    fetched = {col.name for col in client().get_collections().collections}
    with _known_collections_lock:
        _known_collections.update(fetched)
        return name in _known_collections


def ensure_collection(name: str) -> None:
    with _known_collections_lock:
        if name in _known_collections:
            return
    # Not cached yet — check remotely and create if needed.
    fetched = {col.name for col in client().get_collections().collections}
    with _known_collections_lock:
        _known_collections.update(fetched)
        if name in _known_collections:
            return
    c = client()
    c.create_collection(
        name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    c.create_payload_index(name, "ingested_at", field_schema=PayloadSchemaType.DATETIME)
    with _known_collections_lock:
        _known_collections.add(name)


def _onnx_session():
    """Return (or lazily initialize) the ONNX InferenceSession for GPU inference."""
    global _onnx_sess
    if _onnx_sess is not None:
        return _onnx_sess
    with _onnx_lock:
        if _onnx_sess is not None:
            return _onnx_sess
        import onnxruntime as ort

        if not ONNX_MODEL_PATH:
            raise RuntimeError("ONNX_MODEL_PATH is not set. Cannot use EMBED_BACKEND=onnx_gpu.")
        _log.info("Loading ONNX model from %s …", ONNX_MODEL_PATH)
        _onnx_sess = ort.InferenceSession(
            ONNX_MODEL_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active = _onnx_sess.get_providers()[0]
        _log.info("ONNX session ready — active EP: %s", active)
        return _onnx_sess


def _embed_batch_onnx(texts: list[str]) -> list[list[float]]:
    """Embed texts locally using onnxruntime (CUDA EP on dev-nodee GTX1080)."""
    import numpy as np
    from transformers import AutoTokenizer

    sess = _onnx_session()

    # Tokenize — use the model directory that contains tokenizer_config.json
    model_dir = os.path.dirname(ONNX_MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")

    input_names = {i.name for i in sess.get_inputs()}
    feed = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}
    out = sess.run(None, feed)[0]  # (batch, seq, 768)

    # Mean-pool over non-padding tokens then L2-normalize (e5 convention).
    mask = feed["attention_mask"].astype(np.float32)  # (batch, seq)
    pooled = (out * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = pooled / np.maximum(norms, 1e-9)
    return pooled.tolist()


def _embed_batch(texts: list[str]) -> list[list[float]]:
    import time

    import httpx

    for attempt in range(EMBED_RETRIES):
        try:
            resp = httpx.post(
                EMBED_URL,
                json={"texts": texts, "collection": EMBED_COLLECTION, "mode": "index"},
                headers={"X-API-Key": EMBED_API_KEY} if EMBED_API_KEY else {},
                timeout=EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["vectors"]
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt == EMBED_RETRIES - 1:
                raise
            _log.warning(
                "embed retry %d/%d — %d texts → %s",
                attempt + 1,
                EMBED_RETRIES,
                len(texts),
                type(e).__name__,
            )
            time.sleep(2**attempt)  # 1s, 2s
    raise RuntimeError("unreachable")


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in bounded sub-batches so a single POST stays within timeout."""
    batch_fn = _embed_batch_onnx if EMBED_BACKEND == "onnx_gpu" else _embed_batch
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(batch_fn(texts[i : i + EMBED_BATCH]))
    return vectors


def upsert(collection: str, points: list[dict[str, Any]]) -> None:
    ensure_collection(collection)
    vectors = embed([p["text"] for p in points])
    structs = [
        PointStruct(
            id=p["id"],
            vector=vec,
            payload={k: v for k, v in p.items() if k != "id"},
        )
        for p, vec in zip(points, vectors)
    ]
    client().upsert(collection_name=collection, points=structs)


def search(collection: str, query: str, limit: int = 5) -> list[dict]:
    ensure_collection(collection)
    vector = embed([query])[0]
    hits = client().query_points(collection_name=collection, query=vector, limit=limit).points
    return [
        {
            "score": round(hit.score, 4),
            "id": hit.id,
            **(hit.payload or {}),
        }
        for hit in hits
    ]


def delete_by_payload(collection: str, key: str, value: str) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    if not _collection_exists(collection):
        return
    client().delete(
        collection_name=collection,
        points_selector=Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))]),
    )


def ids_by_payload(collection: str, key: str, value: str) -> list[int]:
    """Return all point IDs where payload[key] == value (scrolls to exhaustion)."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    if not _collection_exists(collection):
        return []
    ids: list[int] = []
    offset = None
    filt = Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])
    while True:
        records, offset = client().scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.extend(r.id for r in records)
        if offset is None:
            break
    return ids


def delete_by_ids(collection: str, ids: list[int]) -> None:
    """Delete points by explicit ID list."""
    if not ids:
        return
    if not _collection_exists(collection):
        return
    client().delete(collection_name=collection, points_selector=ids)


def list_collections() -> list[dict]:
    from qdrant_client.models import OrderBy

    cols = client().get_collections().collections
    result = []
    for col in sorted(cols, key=lambda c: c.name):
        info = client().get_collection(col.name)
        # Use order_by with limit=1 to fetch only the single most-recent record.
        # scroll(limit=50) over SHA-256-keyed points gave no ordering guarantee,
        # so max(ingested_at) over the first 50 results was incorrect (#91).
        records, _ = client().scroll(
            collection_name=col.name,
            limit=1,
            order_by=OrderBy(key="ingested_at", direction="desc"),
            with_payload=["ingested_at"],
        )
        last = records[0].payload.get("ingested_at", "") if records and records[0].payload else ""
        result.append(
            {
                "name": col.name,
                "points": info.points_count,
                "last_ingested": last[:19].replace("T", " ") if last else "—",
            }
        )
    return result
