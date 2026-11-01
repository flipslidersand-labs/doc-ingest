"""Ingest curated Obsidian vault notes into Qdrant `notes` collection.

Targets a private thought vault (e.g. ~/notes). Only curated directories are
ingested — scratch dirs (00-inbox, 90-daily) are skipped to avoid polluting the
collection with unfinished thinking.
"""

from datetime import UTC, datetime
from pathlib import Path

from core.distiller import distill_text
from core.git import changed_files, head_sha
from core.ingest_pipeline import ingest_markdown_files
from core.logging import get_logger
from core.path_guard import assert_safe_path

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
        files = [assert_safe_path(file)]
    else:
        files = _changed_files()

    if not files:
        _log.debug("no curated notes changed")
        return

    ingest_markdown_files(
        files,
        collection=COLLECTION,
        source_label="obsidian-note",
        distill_fn=lambda text: distill_text(
            text, "この Obsidian ノートの要点・決定理由・制約を200字以内で。", local_only=True
        ),
        git_sha=sha,
        now=now,
    )
