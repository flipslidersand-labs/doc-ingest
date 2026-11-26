"""Git utilities shared across ingest modules."""

import logging
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

_log = logging.getLogger(__name__)


def changed_files(repo_path: str | Path = ".") -> list[Path]:
    """Return files changed in the last commit (HEAD~1 vs HEAD, ignoring the working tree) — including deleted ones and renamed-away paths.

    The paths are relative to the **repository root**, whatever the cwd is (``git diff
    --name-only`` without ``--relative``). Filter on them as such; to open them use
    :func:`resolve_from_cwd` — from a subdirectory the bare paths name nothing (#354).

    Returns an empty list when ``git diff HEAD~1 HEAD`` fails (e.g. initial commit,
    shallow clone).  Callers must not treat the return value as "all tracked
    files" in the error case, so we no longer fall back to ``git ls-files``.

    * ``-z``: NUL-separated and never quoted. Without it git prints non-ASCII paths as
      ``"10-projects/\\346\\227…md"`` (``core.quotepath``), which names no file — Japanese
      notes were silently skipped.
    * ``--no-renames``: a rename is reported as delete(old) + add(new). By default only the new
      path is listed, so the old path's points could never be removed (#217).
    """
    r = subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "HEAD~1",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        _log.warning(
            "git diff HEAD~1 HEAD failed (returncode=%d, stderr=%r); "
            "returning empty list to avoid full-repo ingest",
            r.returncode,
            r.stderr.strip(),
        )
        return []
    return [Path(p) for p in r.stdout.split("\0") if p]


def repo_root(repo_path: str | Path = ".") -> Path | None:
    """The working-tree root of the repo containing *repo_path*, or None if git can't tell."""
    # フック内で git が設定する GIT_DIR 等を引き継ぐと -C が無視される (#314、core/doc_key.py と同じ)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    r = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    top = r.stdout.strip()
    return Path(top).resolve() if r.returncode == 0 and top else None


def resolve_from_cwd(paths: Iterable[Path], repo_path: str | Path = ".") -> list[Path]:
    """Turn repo-root-relative *paths* (from :func:`changed_files`) into paths usable from the cwd.

    At the repo root (the post-commit hook) they are returned unchanged. From a subdirectory they
    become absolute, so a file that exists is found and a deleted one is still missing. If the
    root can't be determined, returns an empty list: guessing "relative to the cwd" would make
    every existing file look deleted and its points get removed (#354).
    """
    paths = list(paths)
    root = repo_root(repo_path)
    if root is None:
        _log.warning(
            "could not determine the git repo root; skipping %d changed path(s)", len(paths)
        )
        return []
    if str(repo_path) == "." and root == Path.cwd().resolve():
        return paths
    return [root / p for p in paths]


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
