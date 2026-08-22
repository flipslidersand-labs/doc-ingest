"""Ingest local or remote PDF files into Qdrant research collection."""
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from core.chunker import chunk_text
from core.distiller import distill_design_doc
from core.logging import get_logger
from core.qdrant import delete_by_payload, upsert

_log = get_logger(__name__)

COLLECTION = "research"


def extract_text(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def ingest_pdf(source: str, tags: list[str] | None = None) -> None:
    """Ingest a local PDF path or remote PDF URL."""
    tags = tags or []
    now = datetime.now(UTC).isoformat()
    source_url = source

    if source.startswith("http"):
        import httpx
        resp = httpx.get(source, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        pdf_bytes = resp.content
    else:
        pdf_bytes = Path(source).read_bytes()
        source_url = str(Path(source).resolve())

    text = extract_text(pdf_bytes)
    if not text.strip():
        _log.warning("no text extracted from %s", source)
        return

    delete_by_payload(COLLECTION, "source_url", source_url)

    chunks = chunk_text(text, source_url=source_url)
    points = [
        {
            "id": int(
                hashlib.sha256(f"{source_url}:{c['chunk_index']}".encode()).hexdigest(), 16
            ) % (2**63),
            "text": distill_design_doc(c["text"]),
            "source": "pdf",
            "source_url": source_url,
            "chunk_index": c["chunk_index"],
            "tags": tags,
            "ingested_at": now,
        }
        for c in chunks
    ]
    if points:
        upsert(COLLECTION, points)
    _log.info("ingested %s (%d chunks) → %s", source, len(points), COLLECTION)
