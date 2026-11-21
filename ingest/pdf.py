"""Ingest local or remote PDF files into Qdrant research collection."""

import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from core.chunker import cap_chunks, chunk_text
from core.config import get_settings
from core.distiller import distill_text
from core.ids import make_id
from core.qdrant import replace_points
from core.schema import Collection
from core.url_guard import safe_download, strip_url

_log = logging.getLogger(__name__)

COLLECTION = Collection.RESEARCH.value


def _extract_text_from_path(path: Path) -> str:
    """Open PDF from a file path (avoids loading full bytes into memory)."""
    if not path.exists():
        raise FileNotFoundError(f"No such file: '{path}'")
    doc = pymupdf.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def ingest_pdf(source: str, tags: list[str] | None = None, allow_cloud: bool = False) -> None:
    """Ingest a local PDF path or remote PDF URL.

    A local file may be private, so by default it is summarised by the local LLM only and is
    never sent to Anthropic (``allow_cloud=True`` opts in). A URL is a public document.
    """
    tags = tags or []
    now = datetime.now(UTC).isoformat()
    source_url = source  # 表示用 (query 除去済み)
    doc_key = source  # 識別用 (元の URL / 解決済みパス。point id の元で、query も含む)
    is_url = source.startswith("http")
    local_only = not is_url and not allow_cloud

    if is_url:
        # 署名付き URL のトークンをログ・payload・point id に残さない (query/fragment を落とす)
        source_url = strip_url(source)
        # URL は利用者入力。https 以外・内部アドレス・内部へのリダイレクトを拒否し、サイズに上限を設ける (#214)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            safe_download(source, tmp)
            tmp.flush()
            text = _extract_text_from_path(Path(tmp.name))
    else:
        local_path = Path(source)
        max_bytes = get_settings().max_local_bytes
        if local_path.exists() and local_path.stat().st_size > max_bytes:
            # 読み込む前に見送る。既存の point は消さない
            _log.warning("skipping %s: larger than MAX_LOCAL_BYTES=%d", local_path, max_bytes)
            return
        source_url = doc_key = str(local_path.resolve())
        text = _extract_text_from_path(local_path)

    if not text.strip():
        _log.warning("no text extracted from %s", source_url)
        return

    chunks = cap_chunks(chunk_text(text), source_url)
    if not chunks:
        # 空で上書きしない: replace_points(..., []) は既存 point を全削除する
        _log.warning("no chunks produced for %s; keeping existing points", source_url)
        return
    doc_id = make_id(doc_key)
    points = [
        {
            "id": make_id(f"{doc_key}:{c['chunk_index']}"),
            "text": distill_text(
                c["text"], "この資料の要点・実装に使える知見を200字以内で。", local_only=local_only
            ),
            "source": "pdf",
            "trust": "external",
            "source_url": source_url,
            "doc_id": doc_id,
            "chunk_index": c["chunk_index"],
            "tags": tags,
            "ingested_at": now,
        }
        for c in chunks
    ]
    # 旧チャンクを先に消さない: distill / 埋め込み (upsert) のどちらが失敗しても旧データが残る
    replace_points(COLLECTION, "doc_id", doc_id, points)
    _log.info("ingested %s (%d chunks) → %s", source_url, len(points), COLLECTION)
