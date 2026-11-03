#!/usr/bin/env python3
"""Delete pre-#216 points (no `repo` payload) from the `design` / `notes` collections.

Before #216 the point id and `file_path` were the bare path (relative from the post-commit
hook, absolute from a manual run), so they collide across repos and never match the new
`<repo>/<path>` key. Those points are orphans after upgrading; delete them, then re-ingest.

  .venv/bin/python scripts/migrate_doc_keys.py                 # dry-run: count + sample file_paths
  .venv/bin/python scripts/migrate_doc_keys.py --apply         # actually delete
  .venv/bin/python scripts/migrate_doc_keys.py design --apply  # one collection only

Run on the machine that owns the Qdrant credentials (reads QDRANT_URL / QDRANT_API_KEY).
Re-ingest afterwards (see README): unchanged docs are NOT picked up by the post-commit hook.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import (
    Filter,
    FilterSelector,
    IsEmptyCondition,
    PayloadField,
)

COLLECTIONS = ("design", "notes")


def _unmigrated() -> Filter:
    return Filter(must=[IsEmptyCondition(is_empty=PayloadField(key="repo"))])


def migrate(client, collections, apply: bool, out=print) -> int:
    total = 0
    for name in collections:
        flt = _unmigrated()
        n = client.count(collection_name=name, count_filter=flt, exact=True).count
        points, _ = client.scroll(
            collection_name=name, scroll_filter=flt, limit=10, with_payload=["file_path"]
        )
        out(f"[{name}] 未移行 {n} 件")
        for p in points:
            out(f"    {(p.payload or {}).get('file_path', '?')}")
        if n > len(points):
            out(f"    … ほか {n - len(points)} 件")
        if apply and n:
            client.delete(collection_name=name, points_selector=FilterSelector(filter=flt))
            out(f"[{name}] {n} 件を削除しました")
        total += n
    if not apply:
        out("dry-run です。削除するには --apply を付けてください。")
    return total


def main(argv=None, client=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("collections", nargs="*", help="対象 (design / notes)。省略で両方")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    bad = [c for c in args.collections if c not in COLLECTIONS]
    if bad:
        ap.error(f"対象外のコレクション: {bad} (許可: {list(COLLECTIONS)})")
    if client is None:
        from core.qdrant import client as get_client

        client = get_client()
    migrate(client, args.collections or COLLECTIONS, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
