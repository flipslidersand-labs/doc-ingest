"""PDF からテキストを取り出す共通処理 (ingest/pdf.py と ingest/arxiv.py が使う)。"""

from pathlib import Path

import pymupdf


def extract_pdf_text(source: Path | bytes, limit: int | None = None) -> str:
    """ページ本文を ``\\n\\n`` で結合して返す。

    ``source`` が Path ならファイルから、bytes ならメモリ上のストリームから開く。
    ``limit`` を渡すと、その文字数に届いた時点でページ読みを止め、``limit`` 文字に切り詰める。
    例外時も含め、document は必ず close される。
    """
    if isinstance(source, bytes):
        doc = pymupdf.open(stream=source, filetype="pdf")
    else:
        doc = pymupdf.open(str(source))
    pages: list[str] = []
    n = 0
    with doc:
        for page in doc:
            pages.append(page.get_text())
            n += len(pages[-1]) + 2
            if limit is not None and n >= limit:
                break
    text = "\n\n".join(pages)
    return text if limit is None else text[:limit]
