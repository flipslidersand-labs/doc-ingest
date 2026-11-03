"""Shared markdown ingest pipeline for `notes` / `design` collections.

Flow per file: fetch old ids → chunk → distill in parallel → upsert → delete stale ids.
Callers only decide which files to ingest and how to distill a chunk.
"""

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.chunker import cap_chunks, chunk_markdown
from core.distiller import DistillError
from core.doc_key import doc_key, escapes_repo
from core.ids import make_id
from core.qdrant import delete_by_ids, ids_by_payload, upsert

_log = logging.getLogger(__name__)

# llama-server は 1 台で複数スロットが n_ctx を共有し、長い入力の並列は全滅する。既定は直列。
DISTILL_WORKERS = int(os.getenv("DISTILL_WORKERS", "1"))
# ローカルの markdown 1 ファイルの上限 (バイト)。これを超えるファイルは読まずに見送る (#224)。
MAX_LOCAL_BYTES = int(os.getenv("MAX_LOCAL_BYTES", str(10 * 1024 * 1024)))
# git 管理外のファイルの repo。空にすると scripts/migrate_doc_keys.py が「未移行」と見分けられない。
NO_REPO = "-"


def _distill_or_none(distill_fn: Callable[[str], str], text: str) -> str | None:
    try:
        return distill_fn(text)
    except DistillError as e:
        _log.warning("distill failed: %s", e)
        return None


def ingest_markdown_files(
    files: Iterable[Path],
    *,
    collection: str,
    source_label: str,
    distill_fn: Callable[[str], str],
    git_sha: str,
    now: str,
) -> int:
    """Returns the number of chunks skipped because distillation failed."""
    failed = 0
    for path in files:
        # 識別子は <repo>/<リポ内の相対パス>。フック(相対パス)と手動(絶対パス)で同じになり、
        # 別リポの同名ファイルとは衝突しない (#216)。
        repo, key = doc_key(path)
        unsafe = path.exists() and (path.is_symlink() or escapes_repo(path))
        if unsafe:
            # symlink は文書ではなく「指す先のパス」を持つだけ。リポ外 (~/.ssh など) の中身を LLM と
            # Qdrant に入れない。親ディレクトリが symlink でリポ外へ出る場合も同様 (#224)。
            # key は解決後 (= リポ外の実パス) になりうるので、原因の symlink があるリポ内のパスを出す
            _log.warning("skipping %s: symlink or resolves outside the repo", path)
        if unsafe or not path.exists():
            # 削除・リネームされた、または文書として無効になったファイル: その識別子の point を消す (#217)
            gone = ids_by_payload(collection, "file_path", key)
            if gone:
                delete_by_ids(collection, gone)
                _log.info("removed %d point(s) of %s → %s", len(gone), key, collection)
            continue
        if path.stat().st_size > MAX_LOCAL_BYTES:
            # 巨大ファイルは読み込む前に見送る。既存の point は消さない (一時的な肥大化でデータを失わない)
            _log.warning("skipping %s: larger than MAX_LOCAL_BYTES=%d", key, MAX_LOCAL_BYTES)
            continue
        old_ids = ids_by_payload(collection, "file_path", key)
        chunks = cap_chunks(chunk_markdown(path.read_text()), key)
        with ThreadPoolExecutor(max_workers=DISTILL_WORKERS) as ex:
            distilled_list = list(ex.map(lambda c: _distill_or_none(distill_fn, c["text"]), chunks))
        # 要約に失敗したチャンクは埋め込まず、既存の point も消さない(一時障害でデータを失わない)
        failed_ids = {
            make_id(f"{key}:{c['chunk_index']}")
            for c, d in zip(chunks, distilled_list)
            if d is None
        }
        failed += len(failed_ids)
        points = [
            {
                "id": make_id(f"{key}:{chunk['chunk_index']}"),
                "text": distilled,
                "source": source_label,
                "file_path": key,
                "repo": repo or NO_REPO,
                "section": chunk["section"],
                "git_sha": git_sha,
                "ingested_at": now,
            }
            for chunk, distilled in zip(chunks, distilled_list)
            if distilled is not None
        ]
        upsert(collection, points)
        keep = {p["id"] for p in points} | failed_ids
        stale = [i for i in old_ids if i not in keep]
        if stale:
            delete_by_ids(collection, stale)
        _log.info("ingested %s (%d chunks) → %s", key, len(points), collection)
        if failed_ids:
            _log.warning("%s: distill failed for %d chunk(s), skipped", key, len(failed_ids))
    return failed
