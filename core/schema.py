"""コレクション名と payload の形を一箇所に集約する。

payload キーの改名はしない。読み出し側は旧キー(source_url / file_path / arxiv_id)を併記する。
"""

from enum import StrEnum
from typing import TypedDict


class Collection(StrEnum):
    EXTERNAL_DOCS = "external-docs"
    RESEARCH = "research"
    DESIGN = "design"
    NOTES = "notes"


def collection_names() -> list[str]:
    return [c.value for c in Collection]


class BasePayload(TypedDict, total=False):
    """各 ingest が書く payload の共通部分(出典キーはソースごとに異なる)。"""

    text: str
    section: str
    trust: str
    source: str
    origin: str  # 将来の統一出典キー。現状は未書き込み
    ingested_at: str
    source_url: str  # external / pdf / arxiv(blog)
    file_path: str  # design / notes
    arxiv_id: str  # arxiv(論文)


# 出典取得の優先順。旧キー併記は payload 改名 PR まで維持する。
_ORIGIN_KEYS = ("origin", "source_url", "file_path", "arxiv_id")


def payload_origin(payload: dict) -> str:
    """出典(origin)は取り込み元の URL かパス。新キー優先で旧キーにフォールバックする。"""
    for k in _ORIGIN_KEYS:
        v = payload.get(k)
        if v:
            return str(v)
    return ""
