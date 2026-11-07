"""Sync external API docs with ETag-based freshness detection."""

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from ruamel.yaml import YAML

from core.chunker import cap_chunks, chunk_markdown
from core.config import get_settings
from core.html import extract_markdown, looks_like_html
from core.ids import make_id
from core.qdrant import replace_points
from core.redact import redact_urls
from core.retry import with_retry
from core.schema import Collection
from core.url_guard import UnsafeURLError, assert_safe_url, safe_get

_log = logging.getLogger(__name__)

COLLECTION = Collection.EXTERNAL_DOCS.value
SOURCES_FILE = Path(__file__).parent.parent / "sources" / "external.yaml"
MAX_RETRIES = 3
# 生チャンクをそのまま埋め込むので、e5 の 512 token 上限 (超過は黙って切り詰め) に収める。
# チャンカーの既定 400 は tiktoken 基準で、e5 のトークナイザではコードの多い文章が最大 1.41 倍になり
# 13/412 チャンクが 512 を超えた (#211)。300 なら e5 の最大 422 で、文脈を ~50 tokens 前置しても収まる。
EXTERNAL_MAX_SECTION_TOKENS = 300
JINA_BASE = "https://r.jina.ai/"
_yaml = YAML()
_yaml.preserve_quotes = True


def _load_sources():
    return _yaml.load(SOURCES_FILE.read_text()) or []


def _save_sources(sources) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=SOURCES_FILE.parent, delete=False, suffix=".tmp"
    ) as f:
        _yaml.dump(sources, f)
        tmp = Path(f.name)
    tmp.replace(SOURCES_FILE)  # POSIX atomic rename


class _ServerError(Exception):
    """5xx response — retried like a network error."""


def _fetch_with_retry(url: str, headers: dict) -> httpx.Response | None:
    def attempt() -> httpx.Response:
        resp = safe_get(url, headers=headers, timeout=30)
        if resp.status_code >= 500:
            raise _ServerError(f"HTTP {resp.status_code}")
        return resp

    def log_retry(n: int, e: Exception) -> None:
        _log.debug("retry %d/%d %s → %s", n + 1, MAX_RETRIES, url, redact_urls(e))

    try:
        return with_retry(
            attempt,
            retries=MAX_RETRIES,
            is_retryable=lambda e: isinstance(e, (_ServerError, httpx.RequestError)),
            on_retry=log_retry,
        )
    except UnsafeURLError as e:
        _log.warning("blocked %s: %s", url, redact_urls(e))
    except (_ServerError, httpx.RequestError):
        pass  # gave up — the caller logs the failure
    return None


@dataclass
class FetchResult:
    """Phase 1 (取得) の結果。Phase 2 (反映) はこれだけを見る。"""

    status: Literal["ready", "skipped", "failed"]
    url: str
    source: dict
    new_etag: str = ""
    points: list[dict] = field(default_factory=list)

    @classmethod
    def skipped(cls, url: str, source: dict) -> "FetchResult":
        return cls("skipped", url, source)

    @classmethod
    def failed(cls, url: str, source: dict) -> "FetchResult":
        return cls("failed", url, source)

    @classmethod
    def ready(cls, url: str, source: dict, new_etag: str, points: list[dict]) -> "FetchResult":
        return cls("ready", url, source, new_etag, points)


def _fetch_source(source: dict, force: bool, now: str) -> FetchResult:
    """Fetch + extract content for one source. Thread-safe: reads source, never writes it."""
    url = source["url"]
    try:
        assert_safe_url(url)
    except UnsafeURLError as exc:
        _log.warning("blocked unsafe URL: %s", exc)
        return FetchResult.failed(url, source)
    stored_etag = source.get("last_etag", "")
    headers: dict[str, str] = {}
    if stored_etag and not force:
        headers["If-None-Match"] = stored_etag

    use_jina = source.get("render", False)
    fetch_url = JINA_BASE + url if use_jina else url

    resp = _fetch_with_retry(fetch_url, headers)
    if resp is None:
        _log.warning("failed %s (gave up after %d attempts)", url, MAX_RETRIES)
        return FetchResult.failed(url, source)

    if resp.status_code == 304:
        _log.info("up-to-date %s", url)
        return FetchResult.skipped(url, source)

    if not resp.is_success:
        _log.warning("failed %s → HTTP %d (continuing)", url, resp.status_code)
        return FetchResult.failed(url, source)

    new_etag = resp.headers.get("etag", "")
    body = resp.text

    if use_jina:
        if not body.strip():
            _log.warning("Jina returned empty body, skipping: %s", url)
            return FetchResult.skipped(url, source)
    elif looks_like_html(resp.headers.get("content-type", ""), body):
        extracted = extract_markdown(body)
        if extracted:
            body = extracted
        else:
            _log.warning("HTML extraction empty, skipping (raw HTML not ingested): %s", url)
            return FetchResult.skipped(url, source)

    chunks = cap_chunks(chunk_markdown(body, max_section_tokens=EXTERNAL_MAX_SECTION_TOKENS), url)
    points = [
        {
            "id": make_id(f"{url}:{c['chunk_index']}"),
            "text": c["text"],
            "source": "external-api",
            "trust": "external",
            "source_url": url,
            "section": c["section"],
            "doc_type": source.get("type", "external-api"),
            "tags": source.get("tags", []),
            "ingested_at": now,
        }
        for c in chunks
    ]
    return FetchResult.ready(url, source, new_etag, points)


def _notify_discord(
    synced: int,
    total_chunks: int,
    skipped: list[str],
    failed: list[str],
    ts: str,
) -> None:
    webhook_url = get_settings().discord_webhook_url
    if not webhook_url:
        return
    status = "✅" if not failed else "⚠️"
    lines = [
        f"{status} doc-ingest sync 完了 ({ts})",
        f"synced: {synced} sources / {total_chunks} chunks",
        f"skipped: {len(skipped)}"
        + (f" ({', '.join(skipped[:3])}{'…' if len(skipped) > 3 else ''})" if skipped else ""),
        f"failed: {len(failed)}"
        + (f" ({', '.join(failed[:3])}{'…' if len(failed) > 3 else ''})" if failed else ""),
    ]
    try:
        resp = httpx.post(webhook_url, json={"content": "\n".join(lines)}, timeout=10)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        # webhook の URL はそれ自体が認証情報。httpx の例外メッセージは URL を含むので伏せる
        _log.warning("Discord notify failed: %s", redact_urls(e))


@dataclass
class _Summary:
    """Phase 2 の集計。通知とソース保存の要否に使う。"""

    synced: int = 0
    total_chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    changed: bool = False


def _apply_result(result: FetchResult, now: str, dry_run: bool, summary: _Summary) -> None:
    """ソース1件の結果を Qdrant とソースのメタデータ (source dict) に反映し、summary に積む。"""
    source = result.source
    url = result.url

    if result.status == "failed":
        source["failed_at"] = now
        summary.changed = True
        summary.failed.append(url)
        return

    if result.status == "skipped":
        summary.skipped.append(url)
        return

    # status == "ready"
    points = result.points
    if dry_run:
        _log.info("dry-run %s → %d chunks (not written)", url, len(points))
        return

    try:
        replace_points(COLLECTION, "source_url", url, points)
    except Exception as e:  # noqa: BLE001 - isolate one source's failure from the rest
        _log.warning("failed %s → %s: %s (continuing)", url, type(e).__name__, redact_urls(e))
        source["failed_at"] = now
        summary.changed = True
        summary.failed.append(url)
        return

    if result.new_etag:
        source["last_etag"] = result.new_etag
    source["last_synced"] = now
    source.pop("failed_at", None)
    summary.changed = True
    summary.synced += 1
    summary.total_chunks += len(points)
    _log.info("synced %s (%d chunks) → %s", url, len(points), COLLECTION)


def _finish_sync(sources: list, summary: _Summary) -> None:
    """ソースの書き戻しと、全体のサマリ通知。"""
    if summary.changed:
        _save_sources(sources)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    _notify_discord(summary.synced, summary.total_chunks, summary.skipped, summary.failed, ts)


def sync_external(force: bool = False, dry_run: bool = False) -> None:
    sources = _load_sources()
    now = datetime.now(UTC).isoformat()

    # Phase 1: parallel fetch + content extraction (I/O bound)
    fetch_results: list[FetchResult | None] = [None] * len(sources)
    with ThreadPoolExecutor(max_workers=get_settings().sync_workers) as pool:
        future_to_idx = {
            pool.submit(_fetch_source, source, force, now): i for i, source in enumerate(sources)
        }
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                fetch_results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                url = sources[i]["url"]
                _log.warning(
                    "failed %s → %s: %s (continuing)", url, type(e).__name__, redact_urls(e)
                )
                fetch_results[i] = FetchResult.failed(url, sources[i])

    # Phase 2: sequential upsert + metadata update (preserves source order)
    summary = _Summary()
    for result in fetch_results:
        if result is not None:
            _apply_result(result, now, dry_run, summary)

    if not dry_run:
        _finish_sync(sources, summary)
