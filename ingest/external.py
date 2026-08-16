"""Sync external API docs with ETag-based freshness detection."""
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from core.chunker import chunk_markdown
from core.qdrant import delete_by_payload, upsert

COLLECTION = "external-docs"
SOURCES_FILE = Path(__file__).parent.parent / "sources" / "external.yaml"
MAX_RETRIES = 3


def _load_sources() -> list[dict]:
    return yaml.safe_load(SOURCES_FILE.read_text()) or []


def _save_sources(sources: list[dict]) -> None:
    SOURCES_FILE.write_text(yaml.dump(sources, allow_unicode=True))


def _fetch_with_retry(url: str, headers: dict) -> httpx.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
            if resp.status_code < 500:
                return resp
            print(f"[retry {attempt+1}/{MAX_RETRIES}] {url} → HTTP {resp.status_code}")
        except httpx.RequestError as e:
            print(f"[retry {attempt+1}/{MAX_RETRIES}] {url} → {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(2**attempt)  # 1s, 2s, 4s
    return None


def sync_external(force: bool = False, dry_run: bool = False) -> None:
    sources = _load_sources()
    now = datetime.now(timezone.utc).isoformat()
    changed = False

    for source in sources:
        url = source["url"]
        stored_etag = source.get("last_etag", "")

        headers = {}
        if stored_etag and not force:
            headers["If-None-Match"] = stored_etag

        resp = _fetch_with_retry(url, headers)
        if resp is None:
            print(f"failed {url} (gave up after {MAX_RETRIES} attempts)")
            source["failed_at"] = now
            changed = True
            continue

        if resp.status_code == 304:
            print(f"up-to-date {url}")
            continue

        resp.raise_for_status()
        new_etag = resp.headers.get("etag", "")
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

        if dry_run:
            print(f"dry-run {url} → {len(chunks)} chunks (not written)")
            continue

        delete_by_payload(COLLECTION, "source_url", url)
        if points:
            upsert(COLLECTION, points)

        source["last_etag"] = new_etag
        source["last_synced"] = now
        source.pop("failed_at", None)
        changed = True
        print(f"synced {url} ({len(points)} chunks) → {COLLECTION}")

    if changed and not dry_run:
        _save_sources(sources)
