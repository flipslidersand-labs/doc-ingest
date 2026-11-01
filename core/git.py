"""Git utilities shared across ingest modules."""

import logging
import subprocess
from pathlib import Path

_log = logging.getLogger(__name__)


def changed_files(repo_path: str | Path = ".") -> list[Path]:
    """Return files changed in the last commit — including deleted ones and renamed-away paths.

    Returns an empty list when ``git diff HEAD~1`` fails (e.g. initial commit,
    shallow clone).  Callers must not treat the return value as "all tracked
    files" in the error case, so we no longer fall back to ``git ls-files``.

    * ``-z``: NUL-separated and never quoted. Without it git prints non-ASCII paths as
      ``"10-projects/\\346\\227…md"`` (``core.quotepath``), which names no file — Japanese
      notes were silently skipped.
    * ``--no-renames``: a rename is reported as delete(old) + add(new). By default only the new
      path is listed, so the old path's points could never be removed (#217).
    """
    r = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--name-only", "-z", "--no-renames", "HEAD~1"],
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
    return [Path(p) for p in r.stdout.split("\0") if p]


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
