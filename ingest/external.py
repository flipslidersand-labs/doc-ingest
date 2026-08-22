"""Sync external API docs with ETag-based freshness detection."""
import hashlib
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from ruamel.yaml import YAML

from core.chunker import chunk_markdown
from core.html import extract_markdown, looks_like_html
from core.logging import get_logger
from core.qdrant import delete_by_ids, ids_by_payload, upsert

_log = get_logger(__name__)

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
            _log.debug("retry %d/%d %s → HTTP %d", attempt + 1, MAX_RETRIES, url, resp.status_code)
        except httpx.RequestError as e:
            _log.debug("retry %d/%d %s → %s", attempt + 1, MAX_RETRIES, url, e)
        if attempt < MAX_RETRIES - 1:
            time.sleep(2**attempt)  # 1s, 2s, 4s
    return None


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
            _log.warning("failed %s (gave up after %d attempts)", url, MAX_RETRIES)
            source["failed_at"] = now
            changed = True
            failed_urls.append(url)
            continue

        if resp.status_code == 304:
            _log.info("up-to-date %s", url)
            skipped_urls.append(url)
            continue

        if not resp.is_success:
            _log.warning("failed %s → HTTP %d (continuing)", url, resp.status_code)
            source["failed_at"] = now
            changed = True
            failed_urls.append(url)
            continue

        new_etag = resp.headers.get("etag", "")

        body = resp.text
        if use_jina:
            if not body.strip():
                _log.warning("Jina returned empty body, skipping: %s", url)
                skipped_urls.append(url)
                continue
        elif looks_like_html(resp.headers.get("content-type", ""), body):
            extracted = extract_markdown(body)
            if extracted:
                body = extracted
            else:
                _log.warning("HTML extraction empty, skipping (raw HTML not ingested): %s", url)
                skipped_urls.append(url)
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
            _log.info("dry-run %s → %d chunks (not written)", url, len(chunks))
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

    if changed and not dry_run:
        _save_sources(sources)

    if not dry_run:
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        _notify_discord(synced_count, total_chunks, skipped_urls, failed_urls, ts)
