"""Sync external API docs with ETag-based freshness detection."""
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from ruamel.yaml import YAML

from core.chunker import chunk_markdown
from core.html import extract_markdown, looks_like_html
from core.qdrant import delete_by_payload, upsert

COLLECTION = "external-docs"
SOURCES_FILE = Path(__file__).parent.parent / "sources" / "external.yaml"
MAX_RETRIES = 3
JINA_BASE = "https://r.jina.ai/"

_yaml = YAML()
_yaml.preserve_quotes = True


def _load_sources():
    return _yaml.load(SOURCES_FILE.read_text()) or []


def _save_sources(sources) -> None:
    with SOURCES_FILE.open("w") as f:
        _yaml.dump(sources, f)


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

        use_jina = source.get("render", False)
        fetch_url = JINA_BASE + url if use_jina else url

        resp = _fetch_with_retry(fetch_url, headers)
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

        body = resp.text
        if use_jina:
            # Jina Reader returns clean markdown — skip HTML extraction
            if not body.strip():
                print(f"warn {url} → Jina returned empty body, skipping")
                continue
        elif looks_like_html(resp.headers.get("content-type", ""), body):
            extracted = extract_markdown(body)
            if extracted:
                body = extracted
            else:
                print(f"warn {url} → HTML extraction empty, skipping (raw HTML not ingested)")
                continue
        chunks = chunk_markdown(body, source_url=url)
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

        try:
            delete_by_payload(COLLECTION, "source_url", url)
            if points:
                upsert(COLLECTION, points)
        except Exception as e:  # noqa: BLE001 - isolate one source's failure from the rest
            print(f"failed {url} → {type(e).__name__}: {e} (continuing)")
            source["failed_at"] = now
            changed = True
            continue

        source["last_etag"] = new_etag
        source["last_synced"] = now
        source.pop("failed_at", None)
        changed = True
        print(f"synced {url} ({len(points)} chunks) → {COLLECTION}")

    if changed and not dry_run:
        _save_sources(sources)
