"""Path traversal guard for local file ingestion.

Resolves a candidate path and verifies it lives inside *allowed_base*
(default: cwd, overridable via DOC_INGEST_BASE_DIR).  Raises
``PermissionError`` for paths that escape the base directory.

Also provides ``assert_https_url`` which rejects plain ``http://`` URLs.
"""

from __future__ import annotations

import os
from pathlib import Path


def _allowed_base() -> Path:
    """Return the resolved base directory for file access."""
    env = os.environ.get("DOC_INGEST_BASE_DIR")
    if env:
        return Path(env).resolve()
    return Path.cwd().resolve()


def assert_safe_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve *path* and assert it is inside *base*.

    Parameters
    ----------
    path:
        The candidate file path supplied by the caller.
    base:
        Allowed base directory.  Defaults to ``DOC_INGEST_BASE_DIR`` env var
        or ``Path.cwd()`` when not provided.

    Returns
    -------
    Path
        The resolved, validated ``Path`` object.

    Raises
    ------
    PermissionError
        When the resolved path is outside *base*.
    """
    allowed = base if base is not None else _allowed_base()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(allowed):
        raise PermissionError(
            f"Path '{resolved}' is outside the allowed base directory '{allowed}'. "
            "Set DOC_INGEST_BASE_DIR to extend the allowed root."
        )
    return resolved


def assert_https_url(url: str) -> None:
    """Raise ``ValueError`` if *url* is not an ``https://`` URL."""
    if not url.startswith("https://"):
        raise ValueError(
            f"Only https:// URLs are permitted; got '{url}'. "
            "Plain http:// is rejected to prevent credential interception."
        )
