"""Stable identity for an ingested document: ``<repo>/<path-in-repo>`` (#216).

The post-commit hook runs in each repo's root and sees *relative* paths (``CLAUDE.md``),
while a manual run sees *absolute* paths. Keying points by the bare path made two repos'
``CLAUDE.md`` collide (same point id; the stale-id sweep then deleted the other repo's
chunks) and made hook/manual runs of the same file produce different ids.

``repo`` is the main working tree's directory name (``git rev-parse --git-common-dir``'s
parent), so a linked worktree resolves to the same repo as the main checkout.
Files outside any git repo fall back to their absolute path (repo = "").
"""

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple


class DocKey(NamedTuple):
    repo: str
    key: str


def _git(cwd: Path, *args: str) -> str | None:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


@lru_cache(maxsize=256)
def _repo_info(directory: Path) -> tuple[Path, str] | None:
    """(worktree toplevel, main repo name) for *directory*, or None outside a git repo."""
    top = _git(directory, "rev-parse", "--show-toplevel")
    if top is None:
        return None
    common = _git(directory, "rev-parse", "--path-format=absolute", "--git-common-dir")
    main_dir = Path(common).parent if common and Path(common).name == ".git" else Path(top)
    return Path(top).resolve(), main_dir.name


def _nearest_existing_dir(path: Path) -> Path:
    for p in (path.parent, *path.parent.parents):
        if p.is_dir():
            return p
    return Path.cwd()


def doc_key(path: str | Path) -> DocKey:
    """Return the identity of *path* (it need not exist, e.g. a file deleted in the commit)."""
    resolved = Path(path).resolve()
    info = _repo_info(_nearest_existing_dir(resolved))
    if info is not None:
        top, repo = info
        try:
            return DocKey(repo, f"{repo}/{resolved.relative_to(top).as_posix()}")
        except ValueError:
            pass  # the file resolves outside the repo (symlink) → absolute fallback
    return DocKey("", str(resolved))


def escapes_repo(path: str | Path) -> bool:
    """True if *path* sits inside a git repo but resolves, through symlinks, to somewhere outside it.

    The repo root is found by walking up the *lexical* path to the nearest directory holding a
    ``.git`` (file or dir), without following symlinks. Using ``git rev-parse`` from the path's
    resolved location would miss ``docs -> ~/.ssh``: the resolved parent is not in the repo at all.
    A path outside every git repo cannot "escape" one, so it is not flagged here.
    """
    lexical = Path(os.path.abspath(path))
    top = next((p for p in lexical.parents if os.path.lexists(p / ".git")), None)
    if top is None:
        return False
    try:
        lexical.resolve().relative_to(top.resolve())
    except ValueError:
        return True
    return False
