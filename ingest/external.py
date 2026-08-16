"""Sync external API docs with ETag-based freshness detection."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from core.chunker import chunk_markdown
from core.qdrant import delete_by_payload, upsert

COLLECTION = "external-docs"
SOURCES_FILE = Path(__file__).parent.parent / "sources" / "external.yaml"


def _load_sources() -> list[dict]:
    return yaml.safe_load(SOURCES_FILE.read_text()) or []


def _save_sources(sources: list[dict]) -> None:
    SOURCES_FILE.write_text(yaml.dump(sources, allow_unicode=True))


def sync_external(force: bool = False) -> None:
    sources = _load_sources()
    now = datetime.now(timezone.utc).isoformat()
    changed = False

    for source in sources:
        url = source["url"]
        stored_etag = source.get("last_etag", "")

        headers = {}
        if stored_etag and not force:
            headers["If-None-Match"] = stored_etag

        try:
            resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        except httpx.RequestError as e:
            print(f"skip {url}: {e}")
            continue

        if resp.status_code == 304:
            print(f"up-to-date {url}")
            continue

        resp.raise_for_status()
        new_etag = resp.headers.get("etag", "")

        # delete stale chunks before re-ingesting
        delete_by_payload(COLLECTION, "source_url", url)

        chunks = chunk_markdown(resp.text, source_url=url)
        points = [
            {
                "id": int(
                    hashlib.sha256(f"{url}:{c['chunk_index']}".encode()).hexdigest(), 16
                ) % (2**63),
                "text": c["text"],
                "source": "external-api",
                "source_url": url,
                "section": c["section"],
                "doc_type": source.get("type", "external-api"),
                "tags": source.get("tags", []),
                "ingested_at": now,
            }
            for c in chunks
        ]
        if points:
            upsert(COLLECTION, points)

        source["last_etag"] = new_etag
        source["last_synced"] = now
        changed = True
        print(f"synced {url} ({len(points)} chunks) → {COLLECTION}")

    if changed:
        _save_sources(sources)
