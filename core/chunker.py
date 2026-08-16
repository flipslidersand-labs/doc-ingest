"""Markdown section-level chunker (splits on h2/h3 headings)."""
import re


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
