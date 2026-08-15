"""Qdrant client wrapper targeting MINIPC e5 embedding service."""
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.68.63:6333")
EMBED_URL = os.getenv("EMBED_URL", "http://192.168.68.63:9092/embed")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
VECTOR_SIZE = 1024  # e5-large

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def ensure_collection(name: str) -> None:
    c = client()
    existing = {col.name for col in c.get_collections().collections}
    if name not in existing:
        c.create_collection(
            name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def embed(texts: list[str]) -> list[list[float]]:
    import httpx

    resp = httpx.post(
        EMBED_URL,
        json={"texts": texts},
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


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


def delete_by_payload(collection: str, key: str, value: str) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client().delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value))]
        ),
    )
