"""Ingest arxiv papers and tech blog posts into Qdrant research collection."""

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import httpx
import pymupdf

from core.distiller import distill_paper, distill_webpage
from core.ids import make_id
from core.logging import get_logger
from core.qdrant import upsert

_log = get_logger(__name__)

COLLECTION = "research"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def _arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\d.]+)", url)
    return m.group(1) if m else None


def _validate_arxiv_id(arxiv_id: str) -> str:
    """Validate arxiv_id format before use in URL to prevent injection."""
    if not _ARXIV_ID_RE.fullmatch(arxiv_id):
        raise ValueError(f"不正な arxiv_id フォーマット: {arxiv_id!r}")
    return arxiv_id


def _fetch_arxiv(arxiv_id: str) -> dict:
    _validate_arxiv_id(arxiv_id)
    resp = httpx.get(
        f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
        timeout=30,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    entries = root.findall(f"{{{_ATOM_NS}}}entry")
    if not entries:
        raise ValueError(f"{arxiv_id} が見つかりません")
    entry = entries[0]

    def _text(tag: str) -> str:
        el = entry.find(f"{{{_ATOM_NS}}}{tag}")
        return el.text.strip() if el is not None and el.text else ""

    authors = [
        name.text.strip()
        for author in entry.findall(f"{{{_ATOM_NS}}}author")
        for name in author.findall(f"{{{_ATOM_NS}}}name")
        if name.text
    ]
    published_raw = _text("published")
    return {
        "title": _text("title"),
        "abstract": _text("summary"),
        "authors": authors,
        "published_date": published_raw[:10] if published_raw else "",
    }


def _fetch_pdf_text(arxiv_id: str) -> str:
    try:
        resp = httpx.get(
            f"https://arxiv.org/pdf/{arxiv_id}",
            timeout=60,
            follow_redirects=True,
        )
        resp.raise_for_status()
        with pymupdf.open(stream=resp.content, filetype="pdf") as doc:
            pages = [page.get_text() for page in doc]
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
                "id": make_id(arxiv_id),
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
        uid = make_id(url)
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
