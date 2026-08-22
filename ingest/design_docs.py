"""Ingest own design docs (CLAUDE.md / ADR / docs/) into Qdrant design collection."""
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from core.chunker import chunk_markdown
from core.distiller import distill_design_doc
from core.git import changed_files, head_sha
from core.qdrant import delete_by_payload, upsert

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
        print("no design docs changed")
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
            uid = int(
                hashlib.sha256(f"{path}:{chunk['chunk_index']}".encode()).hexdigest(), 16
            ) % (2**63)
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
        print(f"ingested {path} ({len(points)} chunks) → {COLLECTION}")
