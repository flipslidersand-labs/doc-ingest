"""Shared markdown ingest pipeline for `notes` / `design` collections.

Flow per file: fetch old ids → chunk → distill in parallel → upsert → delete stale ids.
Callers only decide which files to ingest and how to distill a chunk.
"""

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.chunker import chunk_markdown
from core.ids import make_id
from core.logging import get_logger
from core.qdrant import delete_by_ids, ids_by_payload, upsert

_log = get_logger(__name__)

DISTILL_WORKERS = int(os.getenv("DISTILL_WORKERS", "4"))


def ingest_markdown_files(
    files: Iterable[Path],
    *,
    collection: str,
    source_label: str,
    distill_fn: Callable[[str], str],
    git_sha: str,
    now: str,
) -> None:
    for path in files:
        if not path.exists():
            continue
        old_ids = ids_by_payload(collection, "file_path", str(path))
        chunks = chunk_markdown(path.read_text(), source_url=str(path))
        with ThreadPoolExecutor(max_workers=DISTILL_WORKERS) as ex:
            distilled_list = list(ex.map(lambda c: distill_fn(c["text"]), chunks))
        points = [
            {
                "id": make_id(f"{path}:{chunk['chunk_index']}"),
                "text": distilled,
                "source": source_label,
                "file_path": str(path),
                "section": chunk["section"],
                "git_sha": git_sha,
                "ingested_at": now,
            }
            for chunk, distilled in zip(chunks, distilled_list)
        ]
        upsert(collection, points)
        stale = [i for i in old_ids if i not in {p["id"] for p in points}]
        if stale:
            delete_by_ids(collection, stale)
        _log.info("ingested %s (%d chunks) → %s", path, len(points), collection)
