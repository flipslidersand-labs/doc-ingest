"""Ingest curated Obsidian vault notes into Qdrant `notes` collection.

Targets a private thought vault (e.g. ~/notes). Only curated directories are
ingested — scratch dirs (00-inbox, 90-daily) are skipped to avoid polluting the
collection with unfinished thinking.
"""
from datetime import UTC, datetime
from pathlib import Path

from core.chunker import chunk_markdown
from core.distiller import distill_text
from core.git import changed_files, head_sha
from core.ids import make_id
from core.logging import get_logger
from core.qdrant import delete_by_payload, upsert

_log = get_logger(__name__)

COLLECTION = "notes"

# Curated top-level dirs to ingest. Scratch dirs are intentionally excluded.
NOTE_DIRS = ("10-projects", "20-areas", "30-resources")
SKIP_DIRS = ("00-inbox", "90-daily", "_templates")


def _is_note(p: Path) -> bool:
    if p.suffix != ".md":
        return False
    parts = p.parts
    if parts and parts[0] in SKIP_DIRS:
        return False
    return bool(parts) and parts[0] in NOTE_DIRS


def _changed_files() -> list[Path]:
    return [p for p in changed_files() if _is_note(p)]


def ingest_notes(file: str | None = None) -> None:
    now = datetime.now(UTC).isoformat()
    sha = head_sha()

    if file:
        files = [Path(file)]
    else:
        files = _changed_files()

    if not files:
        _log.debug("no curated notes changed")
        return

    for path in files:
        if not path.exists():
            continue
        delete_by_payload(COLLECTION, "file_path", str(path))
        text = path.read_text()
        chunks = chunk_markdown(text, source_url=str(path))
        points = []
        for chunk in chunks:
            distilled = distill_text(chunk["text"], "この Obsidian ノートの要点・決定理由・制約を200字以内で。")
            uid = make_id(f"{path}:{chunk['chunk_index']}")
            points.append(
                {
                    "id": uid,
                    "text": distilled,
                    "source": "obsidian-note",
                    "file_path": str(path),
                    "section": chunk["section"],
                    "git_sha": sha,
                    "ingested_at": now,
                }
            )
        upsert(COLLECTION, points)
        _log.info("ingested %s (%d chunks) → %s", path, len(points), COLLECTION)
