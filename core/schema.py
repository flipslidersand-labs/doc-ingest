"""コレクション名と、payload からの出典取得を一箇所に集約する。

payload キーの改名はしない。出典キーは source_url / file_path / arxiv_id。
"""

from enum import StrEnum


class Collection(StrEnum):
    EXTERNAL_DOCS = "external-docs"
    RESEARCH = "research"
    DESIGN = "design"
    NOTES = "notes"


def collection_names() -> list[str]:
    return [c.value for c in Collection]


# 出典取得の優先順。
_ORIGIN_KEYS = ("source_url", "file_path", "arxiv_id")


def payload_origin(payload: dict) -> str:
    """出典は取り込み元の URL かパス。source_url、file_path、arxiv_id の順に最初の非空を返す。"""
    for k in _ORIGIN_KEYS:
        v = payload.get(k)
        if v:
            return str(v)
    return ""
