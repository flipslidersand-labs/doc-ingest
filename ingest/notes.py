"""Ingest curated Obsidian vault notes into Qdrant `notes` collection.

Targets a private thought vault (e.g. ~/notes). Only curated directories are
ingested — scratch dirs (00-inbox, 90-daily) are skipped to avoid polluting the
collection with unfinished thinking.
"""

from pathlib import Path

from core.distiller import distill_text
from core.ingest_pipeline import changed_matching, ingest_changed_or_file

COLLECTION = "notes"

# Curated top-level dirs to ingest. Scratch dirs are intentionally excluded.
NOTE_DIRS = ("10-projects", "20-areas", "30-resources")


def _is_note(p: Path) -> bool:
    if p.suffix != ".md":
        return False
    parts = p.parts
    return bool(parts) and parts[0] in NOTE_DIRS


def _changed_files() -> list[Path]:
    return changed_matching(_is_note)


def ingest_notes(file: str | None = None) -> None:
    ingest_changed_or_file(
        file,
        changed=_changed_files,
        collection=COLLECTION,
        source_label="obsidian-note",
        distill_fn=lambda text: distill_text(
            text, "この Obsidian ノートの要点・決定理由・制約を200字以内で。", local_only=True
        ),
        nothing_msg="no curated notes changed",
    )
