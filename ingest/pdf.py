"""Ingest local or remote PDF files into Qdrant research collection."""
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from core.chunker import chunk_text
from core.distiller import distill_design_doc
from core.ids import make_id
from core.logging import get_logger
from core.qdrant import delete_by_payload, upsert

_log = get_logger(__name__)

COLLECTION = "research"


def extract_text(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _extract_text_from_path(path: Path) -> str:
    """Open PDF from a file path (avoids loading full bytes into memory)."""
    if not path.exists():
        raise FileNotFoundError(f"No such file: '{path}'")
    doc = pymupdf.open(str(path))
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

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            with httpx.stream("GET", source, timeout=60, follow_redirects=True) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    tmp.write(chunk)
            tmp.flush()
            text = _extract_text_from_path(Path(tmp.name))
    else:
        local_path = Path(source)
        source_url = str(local_path.resolve())
        text = _extract_text_from_path(local_path)

    if not text.strip():
        _log.warning("no text extracted from %s", source)
        return

    delete_by_payload(COLLECTION, "source_url", source_url)

    chunks = chunk_text(text, source_url=source_url)
    points = [
        {
            "id": make_id(f"{source_url}:{c['chunk_index']}"),
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
