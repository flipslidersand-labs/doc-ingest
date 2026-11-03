"""Ingest own design docs (CLAUDE.md / ADR / docs/) into Qdrant design collection."""

from pathlib import Path

from core.distiller import distill_design_doc
from core.ingest_pipeline import changed_matching, ingest_changed_or_file

COLLECTION = "design"
DESIGN_PATTERNS = ("CLAUDE.md", "AGENTS.md", "docs/*.md", "adr/*.md", "ADR/*.md")


def _is_design_doc(p: Path) -> bool:
    return p.suffix == ".md" and any(p.match(pat) for pat in DESIGN_PATTERNS)


def _changed_files() -> list[Path]:
    return changed_matching(_is_design_doc)


def ingest_design_docs(file: str | None = None) -> None:
    ingest_changed_or_file(
        file,
        changed=_changed_files,
        collection=COLLECTION,
        source_label="design-doc",
        distill_fn=distill_design_doc,
        nothing_msg="no design docs changed",
    )
