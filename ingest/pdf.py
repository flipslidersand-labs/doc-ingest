"""Ingest local or remote PDF files into Qdrant research collection."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from core.chunker import chunk_text
from core.distiller import distill_design_doc
from core.qdrant import delete_by_payload, upsert

COLLECTION = "research"


def extract_text(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def ingest_pdf(source: str, tags: list[str] | None = None) -> None:
    """Ingest a local PDF path or remote PDF URL."""
    tags = tags or []
    now = datetime.now(timezone.utc).isoformat()
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
        print(f"warning: no text extracted from {source}")
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
    print(f"ingested {source} ({len(points)} chunks) → {COLLECTION}")
