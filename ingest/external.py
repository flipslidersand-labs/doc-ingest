"""Sync external API docs with ETag-based freshness detection."""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import httpx
from ruamel.yaml import YAML

from core.chunker import chunk_markdown
from core.html import extract_markdown, looks_like_html
from core.ids import make_id
from core.logging import get_logger
from core.qdrant import delete_by_ids, ids_by_payload, upsert

_log = get_logger(__name__)

COLLECTION = "external-docs"
SOURCES_FILE = Path(__file__).parent.parent / "sources" / "external.yaml"
MAX_RETRIES = 3
JINA_BASE = "https://r.jina.ai/"
SYNC_WORKERS = int(os.getenv("SYNC_WORKERS", "4"))
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
            _log.debug("retry %d/%d %s → HTTP %d", attempt + 1, MAX_RETRIES, url, resp.status_code)
        except httpx.RequestError as e:
            _log.debug("retry %d/%d %s → %s", attempt + 1, MAX_RETRIES, url, e)
        if attempt < MAX_RETRIES - 1:
            time.sleep(2**attempt)  # 1s, 2s, 4s
    return None


def _fetch_source(source: dict, force: bool, now: str) -> dict:
    """Fetch + extract content for one source. Thread-safe: reads source, never writes it."""
    url = source["url"]
    stored_etag = source.get("last_etag", "")
    headers: dict[str, str] = {}
    if stored_etag and not force:
        headers["If-None-Match"] = stored_etag

    use_jina = source.get("render", False)
    fetch_url = JINA_BASE + url if use_jina else url

    resp = _fetch_with_retry(fetch_url, headers)
    if resp is None:
        _log.warning("failed %s (gave up after %d attempts)", url, MAX_RETRIES)
        return {"status": "failed", "url": url, "source": source, "new_etag": "", "points": []}

    if resp.status_code == 304:
        _log.info("up-to-date %s", url)
        return {"status": "skipped", "url": url, "source": source, "new_etag": "", "points": []}

    if not resp.is_success:
        _log.warning("failed %s → HTTP %d (continuing)", url, resp.status_code)
        return {"status": "failed", "url": url, "source": source, "new_etag": "", "points": []}

    new_etag = resp.headers.get("etag", "")
    body = resp.text

    if use_jina:
        if not body.strip():
            _log.warning("Jina returned empty body, skipping: %s", url)
            return {"status": "skipped", "url": url, "source": source, "new_etag": "", "points": []}
    elif looks_like_html(resp.headers.get("content-type", ""), body):
        extracted = extract_markdown(body)
        if extracted:
            body = extracted
        else:
            _log.warning("HTML extraction empty, skipping (raw HTML not ingested): %s", url)
            return {"status": "skipped", "url": url, "source": source, "new_etag": "", "points": []}

    chunks = chunk_markdown(body, source_url=url)
    points = [
        {
            "id": make_id(f"{url}:{c['chunk_index']}"),
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
    return {"status": "ready", "url": url, "source": source, "new_etag": new_etag, "points": points}


def _notify_discord(
    synced: int,
    total_chunks: int,
    skipped: list[str],
    failed: list[str],
    ts: str,
) -> None:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        return
    status = "✅" if not failed else "⚠️"
    lines = [
        f"{status} doc-ingest sync 完了 ({ts})",
        f"synced: {synced} sources / {total_chunks} chunks",
        f"skipped: {len(skipped)}" + (f" ({', '.join(skipped[:3])}{'…' if len(skipped) > 3 else ''})" if skipped else ""),
        f"failed: {len(failed)}" + (f" ({', '.join(failed[:3])}{'…' if len(failed) > 3 else ''})" if failed else ""),
    ]
    try:
        resp = httpx.post(webhook_url, json={"content": "\n".join(lines)}, timeout=10)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        _log.warning("Discord notify failed: %s", e)


def sync_external(force: bool = False, dry_run: bool = False) -> None:
    sources = _load_sources()
    now = datetime.now(UTC).isoformat()
    changed = False

    synced_count = 0
    total_chunks = 0
    skipped_urls: list[str] = []
    failed_urls: list[str] = []

    # Phase 1: parallel fetch + content extraction (I/O bound)
    fetch_results: list[dict] = [{}] * len(sources)
    with ThreadPoolExecutor(max_workers=SYNC_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_fetch_source, source, force, now): i
            for i, source in enumerate(sources)
        }
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                fetch_results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                url = sources[i]["url"]
                _log.warning("failed %s → %s: %s (continuing)", url, type(e).__name__, e)
                fetch_results[i] = {
                    "status": "failed", "url": url,
                    "source": sources[i], "new_etag": "", "points": [],
                }

    # Phase 2: sequential upsert + metadata update (preserves source order)
    for result in fetch_results:
        if not result:
            continue
        source = result["source"]
        url = result["url"]
        status = result["status"]

        if status == "failed":
            source["failed_at"] = now
            changed = True
            failed_urls.append(url)
            continue

        if status == "skipped":
            skipped_urls.append(url)
            continue

        # status == "ready"
        points = result["points"]
        new_etag = result["new_etag"]

        if dry_run:
            _log.info("dry-run %s → %d chunks (not written)", url, len(points))
            continue

        try:
            old_ids = ids_by_payload(COLLECTION, "source_url", url)
            if points:
                upsert(COLLECTION, points)
            new_ids = {p["id"] for p in points}
            stale = [i for i in old_ids if i not in new_ids]
            if stale:
                delete_by_ids(COLLECTION, stale)
        except Exception as e:  # noqa: BLE001 - isolate one source's failure from the rest
            _log.warning("failed %s → %s: %s (continuing)", url, type(e).__name__, e)
            source["failed_at"] = now
            changed = True
            failed_urls.append(url)
            continue

        if new_etag:
            source["last_etag"] = new_etag
        source["last_synced"] = now
        source.pop("failed_at", None)
        changed = True
        synced_count += 1
        total_chunks += len(points)
        _log.info("synced %s (%d chunks) → %s", url, len(points), COLLECTION)

    if not dry_run:
        if changed:
            _save_sources(sources)
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        _notify_discord(synced_count, total_chunks, skipped_urls, failed_urls, ts)
