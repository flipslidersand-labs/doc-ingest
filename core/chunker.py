"""Markdown section-level chunker and plain text chunker."""
import re


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


def chunk_markdown(text: str, source_url: str = "") -> list[dict]:
    sections = re.split(r"(?m)^#{2,3} ", text)
    chunks = []
    for i, section in enumerate(sections):
        body = section.strip()
        if len(body) < 50:
            continue
        heading = body.split("\n", 1)[0].strip()
        chunks.append(
            {
                "text": body,
                "section": heading,
                "chunk_index": i,
                "source_url": source_url,
            }
        )
    return chunks
