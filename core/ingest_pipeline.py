"""Shared markdown ingest pipeline for `notes` / `design` collections.

Flow per file: fetch old ids → chunk → distill in parallel → upsert → delete stale ids.
Callers only decide which files to ingest and how to distill a chunk.
"""

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.chunker import cap_chunks, chunk_markdown
from core.config import get_settings
from core.distiller import DistillError
from core.doc_key import doc_key, escapes_repo, lexical_doc_key
from core.git import changed_files, head_sha
from core.ids import make_id
from core.path_guard import _allowed_base, assert_safe_path
from core.qdrant import replace_points

_log = logging.getLogger(__name__)

# git 管理外のファイルの repo。空にすると scripts/migrate_doc_keys.py が「未移行」と見分けられない。
NO_REPO = "-"


@dataclass(frozen=True)
class IngestResult:
    """Failures of one ingest run, counted separately (#350).

    files_failed: files that raised (not ingested). The CLI exits non-zero for these.
    chunks_skipped: chunks whose distillation failed; their old point is kept. Warning only.
    """

    files_failed: int = 0
    chunks_skipped: int = 0


def _distill_or_none(distill_fn: Callable[[str], str], text: str) -> str | None:
    try:
        return distill_fn(text)
    except DistillError as e:
        _log.warning("distill failed: %s", e)
        return None


def _ingest_one(
    path: Path,
    *,
    ex: ThreadPoolExecutor,
    collection: str,
    source_label: str,
    distill_fn: Callable[[str], str],
    git_sha: str,
    now: str,
) -> int:
    """Ingest one file. Returns the number of chunks skipped because distillation failed."""
    # 識別子は <repo>/<リポ内の相対パス>。フック(相対パス)と手動(絶対パス)で同じになり、
    # 別リポの同名ファイルとは衝突しない (#216)。
    repo, key = doc_key(path)
    unsafe = path.exists() and (path.is_symlink() or escapes_repo(path))
    if unsafe:
        # symlink は文書ではなく「指す先のパス」を持つだけ。リポ外 (~/.ssh など) の中身を LLM と
        # Qdrant に入れない。親ディレクトリが symlink でリポ外へ出る場合も同様 (#224)。
        # key は解決後 (= リポ外の実パス) になりうるので、原因の symlink があるリポ内のパスを出す
        _log.warning("skipping %s: symlink or resolves outside the repo", path)
        # 消すのは symlink 自身の point だけ。解決後の key だとリンク先の別文書の point を消す (#296)
        key = lexical_doc_key(path).key
    if unsafe or not path.exists():
        # 削除・リネームされた、または文書として無効になったファイル: その識別子の point を消す (#217)
        replace_points(collection, "file_path", key, [])
        _log.info("removed %s from %s", key, collection)
        return 0
    max_bytes = get_settings().max_local_bytes
    if path.stat().st_size > max_bytes:
        # 巨大ファイルは読み込む前に見送る。既存の point は消さない (一時的な肥大化でデータを失わない)
        _log.warning("skipping %s: larger than MAX_LOCAL_BYTES=%d", key, max_bytes)
        return 0
    chunks = cap_chunks(chunk_markdown(path.read_text()), key)
    distilled_list = list(ex.map(lambda c: _distill_or_none(distill_fn, c["text"]), chunks))
    # 要約に失敗したチャンクは埋め込まず、既存の point も消さない(一時障害でデータを失わない)
    failed_ids = {
        make_id(f"{key}:{c['chunk_index']}") for c, d in zip(chunks, distilled_list) if d is None
    }
    points = [
        {
            "id": make_id(f"{key}:{chunk['chunk_index']}"),
            "text": distilled,
            "source": source_label,
            "trust": "internal",
            "file_path": key,
            "repo": repo or NO_REPO,
            "section": chunk["section"],
            "git_sha": git_sha,
            "ingested_at": now,
        }
        for chunk, distilled in zip(chunks, distilled_list)
        if distilled is not None
    ]
    replace_points(collection, "file_path", key, points, keep_extra_ids=failed_ids)
    _log.info("ingested %s (%d chunks) → %s", key, len(points), collection)
    if failed_ids:
        _log.warning("%s: distill failed for %d chunk(s), skipped", key, len(failed_ids))
    return len(failed_ids)


def ingest_markdown_files(
    files: Iterable[Path],
    *,
    collection: str,
    source_label: str,
    distill_fn: Callable[[str], str],
    git_sha: str,
    now: str,
) -> IngestResult:
    """Files that raised are logged and counted so the remaining files are still ingested;
    chunks skipped because distillation failed are counted apart from them (#350)."""
    files_failed = 0
    chunks_skipped = 0
    with ThreadPoolExecutor(max_workers=get_settings().distill_workers) as ex:
        for path in files:
            try:
                chunks_skipped += _ingest_one(
                    path,
                    ex=ex,
                    collection=collection,
                    source_label=source_label,
                    distill_fn=distill_fn,
                    git_sha=git_sha,
                    now=now,
                )
            except Exception as e:  # noqa: BLE001
                files_failed += 1
                _log.warning("failed to ingest %s: %s: %s", path, type(e).__name__, e)
                _log.debug("traceback for %s", path, exc_info=True)
    return IngestResult(files_failed=files_failed, chunks_skipped=chunks_skipped)


def changed_matching(is_target: Callable[[Path], bool]) -> list[Path]:
    """The files changed in the last commit for which *is_target* is true."""
    return [p for p in changed_files() if is_target(p)]


def _target_relative(file: str) -> Path:
    """*file* as the post-commit hook would name it: relative to its git repo root.

    Found lexically (no symlink following, like ``escapes_repo``). Outside any repo, relative to
    the allowed base directory; failing that, unchanged.
    """
    lexical = Path(os.path.abspath(file))
    top = next((p for p in lexical.parents if os.path.lexists(p / ".git")), None)
    for root in (top, _allowed_base()):
        if root is not None and lexical.is_relative_to(root):
            return lexical.relative_to(root)
    return Path(file)


def ingest_changed_or_file(
    file: str | None,
    *,
    is_target: Callable[[Path], bool],
    changed: Callable[[], list[Path]],
    collection: str,
    source_label: str,
    distill_fn: Callable[[str], str],
    nothing_msg: str,
) -> IngestResult:
    """Ingest *file* or, without it, what *changed* returns.

    *file* must pass ``assert_safe_path`` and the same *is_target* test as the changed files
    (#360). Otherwise ``--file`` would pull in scratch notes or non-markdown files the hook
    skips. A rejected file is logged and counted as one failed file (nothing is read or written).

    The entry point shared by the markdown collections (design, notes): they differ only in
    which files count (*changed*), the collection, the source label and the distiller.
    Returns the failure counts so the CLI can exit non-zero for failed files (#332, #350).
    """
    now = datetime.now(UTC).isoformat()
    sha = head_sha()
    if file:
        # 検証だけに使い、取り込みには元 (未解決) のパスを渡す。解決後のパスだと symlink 判定が
        # 効かず、指す先の中身が取り込まれる (#315)
        assert_safe_path(file)
        if not is_target(_target_relative(file)):
            _log.warning("skipping %s: not a target of the %s collection", file, collection)
            return IngestResult(files_failed=1)
        files = [Path(file)]
    else:
        files = changed()
    if not files:
        _log.debug(nothing_msg)
        return IngestResult()
    result = ingest_markdown_files(
        files,
        collection=collection,
        source_label=source_label,
        distill_fn=distill_fn,
        git_sha=sha,
        now=now,
    )
    if result.files_failed or result.chunks_skipped:
        _log.warning(
            "%d file(s) failed, %d chunk(s) skipped (distill failed) while ingesting into %s"
            " (see warnings above)",
            result.files_failed,
            result.chunks_skipped,
            collection,
        )
    return result
