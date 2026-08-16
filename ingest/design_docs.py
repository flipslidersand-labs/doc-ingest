"""Ingest own design docs (CLAUDE.md / ADR / docs/) into Qdrant design collection."""
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from core.chunker import chunk_markdown
from core.distiller import distill_design_doc
from core.qdrant import delete_by_payload, upsert

COLLECTION = "design"
DESIGN_PATTERNS = ("CLAUDE.md", "AGENTS.md", "docs/*.md", "adr/*.md", "ADR/*.md")


def _changed_files() -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # first commit: list all tracked files instead
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
        )
    paths = [Path(p) for p in result.stdout.splitlines()]
    return [
        p
        for p in paths
        if p.suffix == ".md" and any(p.match(pat) for pat in DESIGN_PATTERNS)
    ]


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ingest_design_docs(file: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    sha = _git_sha()

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
