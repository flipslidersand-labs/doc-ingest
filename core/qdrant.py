"""Qdrant client wrapper targeting MINIPC e5 embedding service."""
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.logging import get_logger

_log = get_logger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:9092/embed/batch")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_COLLECTION = os.getenv("EMBED_COLLECTION", "sessions")  # embedding-svc のモデルルーティング用
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "180"))  # CPU e5 のコールドロードを許容
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))  # 1 POST あたりの最大テキスト数
EMBED_RETRIES = 3
VECTOR_SIZE = 768  # multilingual-e5-base

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    return _client


def ensure_collection(name: str) -> None:
    c = client()
    existing = {col.name for col in c.get_collections().collections}
    if name not in existing:
        c.create_collection(
            name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _embed_batch(texts: list[str]) -> list[list[float]]:
    import time

    import httpx

    for attempt in range(EMBED_RETRIES):
        try:
            resp = httpx.post(
                EMBED_URL,
                json={"texts": texts, "collection": EMBED_COLLECTION, "mode": "index"},
                headers={"X-API-Key": EMBED_API_KEY},
                timeout=EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["vectors"]
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt == EMBED_RETRIES - 1:
                raise
            _log.warning("embed retry %d/%d — %d texts → %s", attempt + 1, EMBED_RETRIES, len(texts), type(e).__name__)
            time.sleep(2**attempt)  # 1s, 2s
    raise RuntimeError("unreachable")


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in bounded sub-batches so a single POST stays within timeout."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(_embed_batch(texts[i : i + EMBED_BATCH]))
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
    hits = client().query_points(
        collection_name=collection, query=vector, limit=limit
    ).points
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

    existing = {col.name for col in client().get_collections().collections}
    if collection not in existing:
        return
    client().delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value))]
        ),
    )


def ids_by_payload(collection: str, key: str, value: str) -> list[int]:
    """Return all point IDs where payload[key] == value (scrolls to exhaustion)."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    existing = {col.name for col in client().get_collections().collections}
    if collection not in existing:
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
    existing = {col.name for col in client().get_collections().collections}
    if collection not in existing:
        return
    client().delete(collection_name=collection, points_selector=ids)


def list_collections() -> list[dict]:
    cols = client().get_collections().collections
    result = []
    for col in sorted(cols, key=lambda c: c.name):
        info = client().get_collection(col.name)
        records, _ = client().scroll(
            collection_name=col.name,
            limit=50,
            with_payload=["ingested_at"],
        )
        timestamps = [r.payload.get("ingested_at", "") for r in records if r.payload]
        last = max(timestamps, default="")
        result.append({
            "name": col.name,
            "points": info.points_count,
            "last_ingested": last[:19].replace("T", " ") if last else "—",
        })
    return result
