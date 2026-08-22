"""Ingest own design docs (CLAUDE.md / ADR / docs/) into Qdrant design collection."""
from datetime import UTC, datetime
from pathlib import Path

from core.chunker import chunk_markdown
from core.distiller import distill_design_doc
from core.git import changed_files, head_sha
from core.ids import make_id
from core.logging import get_logger
from core.qdrant import delete_by_payload, upsert

_log = get_logger(__name__)

COLLECTION = "design"
DESIGN_PATTERNS = ("CLAUDE.md", "AGENTS.md", "docs/*.md", "adr/*.md", "ADR/*.md")


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
        files = [Path(file)]
    else:
        files = _changed_files()

    if not files:
        _log.debug("no design docs changed")
        return

    for path in files:
        if not path.exists():
            continue
        delete_by_payload(COLLECTION, "file_path", str(path))
        text = path.read_text()
        chunks = chunk_markdown(text, source_url=str(path))
        points = []
        for chunk in chunks:
            distilled = distill_design_doc(chunk["text"])
            uid = make_id(f"{path}:{chunk['chunk_index']}")
            points.append(
                {
                    "id": uid,
                    "text": distilled,
                    "source": "design-doc",
                    "file_path": str(path),
                    "section": chunk["section"],
                    "git_sha": sha,
                    "ingested_at": now,
                }
            )
        upsert(COLLECTION, points)
        _log.info("ingested %s (%d chunks) → %s", path, len(points), COLLECTION)
