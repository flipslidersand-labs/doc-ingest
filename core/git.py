"""Git utilities shared across ingest modules."""

import subprocess
from pathlib import Path


def changed_files(repo_path: str | Path = ".") -> list[Path]:
    """Return files changed in the last commit; fall back to all tracked files."""
    r = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--name-only", "HEAD~1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        r = subprocess.run(
            ["git", "-C", str(repo_path), "ls-files"],
            capture_output=True,
            text=True,
            check=False,
        )
    return [Path(p) for p in r.stdout.splitlines() if p]


def head_sha(repo_path: str | Path = ".") -> str:
    """Return the short SHA of HEAD."""
    r = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip()
