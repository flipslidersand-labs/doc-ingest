"""Ingest curated Obsidian vault notes into Qdrant `notes` collection.

Targets a private thought vault (e.g. ~/notes). Only curated directories are
ingested — scratch dirs (00-inbox, 90-daily) are skipped to avoid polluting the
collection with unfinished thinking.
"""
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from core.chunker import chunk_markdown
from core.distiller import distill_design_doc
from core.qdrant import delete_by_payload, upsert

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
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True
        )
    return [p for p in (Path(x) for x in result.stdout.splitlines()) if _is_note(p)]


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip()


def ingest_notes(file: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    sha = _git_sha()

    if file:
        files = [Path(file)]
    else:
        files = _changed_files()

    if not files:
        print("no curated notes changed")
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
                hashlib.sha256(f"{path}:{chunk['chunk_index']}".encode()).hexdigest(),
                16,
            ) % (2**63)
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
        print(f"ingested {path} ({len(points)} chunks) → {COLLECTION}")
