"""Markdown section-level chunker and plain text chunker."""
import re

# Section bodies larger than this are sub-split so a single embedding does not
# exceed the embedding model's window (multilingual-e5-base ~512 tokens).
MAX_SECTION_CHARS = 1500


def _split_by_size(text: str, max_chars: int) -> list[str]:
    """Group paragraphs into pieces each <= max_chars.

    A single paragraph longer than max_chars is hard-split on a character
    window so no piece exceeds the limit.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        # Hard-split a paragraph that alone exceeds the limit.
        while len(para) > max_chars:
            if current:
                pieces.append("\n\n".join(current))
                current, size = [], 0
            pieces.append(para[:max_chars])
            para = para[max_chars:].strip()
        if current and size + len(para) > max_chars:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_text(text: str, source_url: str = "", chunk_size: int = 800) -> list[dict]:
    """Split plain text into fixed-size chunks by paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current, idx = [], [], 0
    for para in paragraphs:
        current.append(para)
        if sum(len(p) for p in current) >= chunk_size:
            body = "\n\n".join(current)
            chunks.append({"text": body, "section": "", "chunk_index": idx, "source_url": source_url})
            idx += 1
            current = []
    if current:
        body = "\n\n".join(current)
        if len(body) >= 50:
            chunks.append({"text": body, "section": "", "chunk_index": idx, "source_url": source_url})
    return chunks


def chunk_markdown(
    text: str, source_url: str = "", max_section_chars: int = MAX_SECTION_CHARS
) -> list[dict]:
    sections = re.split(r"(?m)^#{2,3} ", text)
    chunks = []
    idx = 0
    for section in sections:
        body = section.strip()
        if len(body) < 50:
            continue
        heading = body.split("\n", 1)[0].strip()
        pieces = (
            [body]
            if len(body) <= max_section_chars
            else _split_by_size(body, max_section_chars)
        )
        for piece in pieces:
            if len(piece) < 50:
                continue
            chunks.append(
                {
                    "text": piece,
                    "section": heading,
                    "chunk_index": idx,
                    "source_url": source_url,
                }
            )
            idx += 1
    return chunks
