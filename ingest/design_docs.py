"""Ingest own design docs (CLAUDE.md / ADR / docs/) into Qdrant design collection."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from core.distiller import distill_design_doc
from core.git import changed_files, head_sha
from core.ingest_pipeline import ingest_markdown_files
from core.path_guard import assert_safe_path

_log = logging.getLogger(__name__)

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
        files = [assert_safe_path(file)]
    else:
        files = _changed_files()

    if not files:
        _log.debug("no design docs changed")
        return

    ingest_markdown_files(
        files,
        collection=COLLECTION,
        source_label="design-doc",
        distill_fn=distill_design_doc,
        git_sha=sha,
        now=now,
    )
