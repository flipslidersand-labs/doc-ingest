"""Markdown section-level chunker and plain text chunker."""

import re

import tiktoken

# multilingual-e5-base context window: 512 tokens. Leave headroom for the
# "query: " / "passage: " prefix that the embedding service prepends.
MAX_SECTION_TOKENS = 400

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _hard_split_tokens(text: str, max_tokens: int) -> list[str]:
    """Split a single oversized text into pieces each <= max_tokens using the tokenizer."""
    ids = _enc.encode(text)
    return [_enc.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Group paragraphs into pieces each <= max_tokens.

    A single paragraph longer than max_tokens is hard-split via the tokenizer
    so every output piece is guaranteed to fit.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        para_tokens = _count_tokens(para)
        # Hard-split a paragraph that alone exceeds the limit.
        if para_tokens > max_tokens:
            if current:
                pieces.append("\n\n".join(current))
                current, size = [], 0
            pieces.extend(_hard_split_tokens(para, max_tokens))
            continue
        if current and size + para_tokens > max_tokens:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += para_tokens
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_text(text: str, source_url: str = "", chunk_tokens: int = 200) -> list[dict]:
    """Split plain text into chunks bounded by token count."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current, size, idx = [], [], 0, 0
    for para in paragraphs:
        t = _count_tokens(para)
        current.append(para)
        size += t
        if size >= chunk_tokens:
            body = "\n\n".join(current)
            chunks.append(
                {"text": body, "section": "", "chunk_index": idx, "source_url": source_url}
            )
            idx += 1
            current, size = [], 0
    if current:
        body = "\n\n".join(current)
        if _count_tokens(body) >= 10:
            chunks.append(
                {"text": body, "section": "", "chunk_index": idx, "source_url": source_url}
            )
    return chunks


def chunk_markdown(
    text: str, source_url: str = "", max_section_tokens: int = MAX_SECTION_TOKENS
) -> list[dict]:
    sections = re.split(r"(?m)^#{2,3} ", text)
    chunks = []
    idx = 0
    for section in sections:
        body = section.strip()
        if _count_tokens(body) < 10:
            continue
        heading = body.split("\n", 1)[0].strip()
        pieces = (
            [body]
            if _count_tokens(body) <= max_section_tokens
            else _split_by_tokens(body, max_section_tokens)
        )
        for piece in pieces:
            if _count_tokens(piece) < 10:
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
