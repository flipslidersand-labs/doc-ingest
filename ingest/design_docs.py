"""Ingest own design docs (CLAUDE.md / ADR / docs/) into Qdrant design collection."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from core.chunker import chunk_markdown
from core.distiller import distill_design_doc
from core.git import changed_files, head_sha
from core.ids import make_id
from core.logging import get_logger
from core.path_guard import assert_safe_path
from core.qdrant import delete_by_ids, ids_by_payload, upsert

_log = get_logger(__name__)

COLLECTION = "design"
DESIGN_PATTERNS = ("CLAUDE.md", "AGENTS.md", "docs/*.md", "adr/*.md", "ADR/*.md")
DISTILL_WORKERS = int(os.getenv("DISTILL_WORKERS", "4"))


def _changed_files() -> list[Path]:
    return [
        p
        for p in changed_files()
        if p.suffix == ".md" and any(p.match(pat) for pat in DESIGN_PATTERNS)
    ]


def ingest_design_docs(file: str | None = None) -> None:
    now = datetime.now(UTC).isoformat()
    sha = head_sha()

    if file:
        files = [assert_safe_path(file)]
    else:
        files = _changed_files()

    if not files:
        _log.debug("no design docs changed")
        return

    for path in files:
        if not path.exists():
            continue
        old_ids = ids_by_payload(COLLECTION, "file_path", str(path))
        text = path.read_text()
        chunks = chunk_markdown(text, source_url=str(path))
        with ThreadPoolExecutor(max_workers=DISTILL_WORKERS) as ex:
            distilled_list = list(ex.map(lambda c: distill_design_doc(c["text"]), chunks))
        points = [
            {
                "id": make_id(f"{path}:{chunk['chunk_index']}"),
                "text": distilled,
                "source": "design-doc",
                "file_path": str(path),
                "section": chunk["section"],
                "git_sha": sha,
                "ingested_at": now,
            }
            for chunk, distilled in zip(chunks, distilled_list)
        ]
        upsert(COLLECTION, points)
        stale = [i for i in old_ids if i not in {p["id"] for p in points}]
        if stale:
            delete_by_ids(COLLECTION, stale)
        _log.info("ingested %s (%d chunks) → %s", path, len(points), COLLECTION)
