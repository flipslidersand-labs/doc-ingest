"""Git utilities shared across ingest modules."""

import logging
import subprocess
from pathlib import Path

_log = logging.getLogger(__name__)


def changed_files(repo_path: str | Path = ".") -> list[Path]:
    """Return files changed in the last commit.

    Returns an empty list when ``git diff HEAD~1`` fails (e.g. initial commit,
    shallow clone).  Callers must not treat the return value as "all tracked
    files" in the error case, so we no longer fall back to ``git ls-files``.
    """
    r = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--name-only", "HEAD~1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        _log.warning(
            "git diff HEAD~1 failed (returncode=%d, stderr=%r); "
            "returning empty list to avoid full-repo ingest",
            r.returncode,
            r.stderr.strip(),
        )
        return []
    return [Path(p) for p in r.stdout.splitlines() if p]


def head_sha(repo_path: str | Path = ".") -> str:
    """Return the short SHA of HEAD, or empty string on error."""
    r = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        _log.warning(
            "git rev-parse --short HEAD failed (returncode=%d, stderr=%r)",
            r.returncode,
            r.stderr.strip(),
        )
        return ""
    return r.stdout.strip()
