"""Ingest arxiv papers and tech blog posts into Qdrant research collection."""
import hashlib
import re
from datetime import UTC, datetime

import httpx
import pymupdf

from core.distiller import distill_paper, distill_webpage
from core.logging import get_logger
from core.qdrant import upsert

_log = get_logger(__name__)

COLLECTION = "research"


def _arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\d.]+)", url)
    return m.group(1) if m else None


def _fetch_arxiv(arxiv_id: str) -> dict:
    resp = httpx.get(
        f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.text
    title = re.search(r"<title>([^<]+)</title>", text)
    summary = re.search(r"<summary>([^<]+)</summary>", text, re.DOTALL)
    authors = re.findall(r"<name>([^<]+)</name>", text)
    published = re.search(r"<published>([^<]+)</published>", text)
    return {
        "title": title.group(1).strip() if title else "",
        "abstract": summary.group(1).strip() if summary else "",
        "authors": authors,
        "published_date": published.group(1)[:10] if published else "",
    }


def _fetch_pdf_text(arxiv_id: str) -> str:
    try:
        resp = httpx.get(
            f"https://arxiv.org/pdf/{arxiv_id}",
            timeout=60,
            follow_redirects=True,
        )
        resp.raise_for_status()
        doc = pymupdf.open(stream=resp.content, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)[:8000]
    except Exception as e:  # noqa: BLE001
        _log.warning("PDF fetch failed (%s), using abstract only", e)
        return ""


def ingest_arxiv(url: str, tags: list[str] | None = None) -> None:
    tags = tags or []
    now = datetime.now(UTC).isoformat()

    arxiv_id = _arxiv_id(url)
    if arxiv_id:
        meta = _fetch_arxiv(arxiv_id)
        body = _fetch_pdf_text(arxiv_id)
        distilled = distill_paper(meta["abstract"], body=body)
        points = [
            {
                "id": int(hashlib.sha256(arxiv_id.encode()).hexdigest(), 16) % (2**63),
                "text": distilled,
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "title": meta["title"],
                "authors": meta["authors"],
                "published_date": meta["published_date"],
                "tags": tags,
                "ingested_at": now,
            }
        ]
    else:
        # tech blog
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        distilled = distill_webpage(resp.text)
        uid = int(hashlib.sha256(url.encode()).hexdigest(), 16) % (2**63)
        points = [
            {
                "id": uid,
                "text": distilled,
                "source": "blog",
                "source_url": url,
                "tags": tags,
                "ingested_at": now,
            }
        ]

    upsert(COLLECTION, points)
    _log.info("ingested %s → %s", url, COLLECTION)
