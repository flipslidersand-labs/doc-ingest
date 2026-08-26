"""Tests for core/chunker.py"""

import tiktoken

from core.chunker import chunk_markdown, chunk_text

_enc = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(_enc.encode(text))


class TestChunkMarkdown:
    def test_splits_on_h2(self):
        md = "## Section A\n" + "x " * 30 + "\n\n## Section B\n" + "y " * 30
        chunks = chunk_markdown(md)
        assert len(chunks) == 2
        assert chunks[0]["section"] == "Section A"
        assert chunks[1]["section"] == "Section B"

    def test_splits_on_h3(self):
        md = "### Deep\n" + "z " * 30
        chunks = chunk_markdown(md)
        assert len(chunks) == 1
        assert chunks[0]["section"] == "Deep"

    def test_skips_short_sections(self):
        md = "## Tiny\nhi\n\n## Real\n" + "a " * 30
        chunks = chunk_markdown(md)
        assert len(chunks) == 1
        assert chunks[0]["section"] == "Real"

    def test_single_line_heading_no_slice_bug(self):
        # regression: heading_end=-1 caused body[:-1] before fix
        md = "## " + "a " * 30
        chunks = chunk_markdown(md)
        assert len(chunks) == 1

    def test_source_url_propagated(self):
        md = "## S\n" + "b " * 30
        chunks = chunk_markdown(md, source_url="http://example.com")
        assert chunks[0]["source_url"] == "http://example.com"

    def test_empty_text_returns_empty(self):
        assert chunk_markdown("") == []

    def test_oversized_section_is_subsplit(self):
        # one section whose body far exceeds the cap → multiple chunks
        body = "## Big\n" + "\n\n".join(["word " * 60] * 10)
        chunks = chunk_markdown(body, max_section_tokens=100)
        assert len(chunks) > 1
        assert all(_tokens(c["text"]) <= 100 for c in chunks)
        assert all(c["section"] == "Big" for c in chunks)

    def test_subsplit_chunk_indices_unique(self):
        body = "## Big\n" + "\n\n".join(["word " * 60] * 10)
        indices = [c["chunk_index"] for c in chunk_markdown(body, max_section_tokens=100)]
        assert indices == list(range(len(indices)))

    def test_single_giant_paragraph_hard_split(self):
        # a single paragraph with no blank lines still gets bounded
        chunks = chunk_markdown("## H\n" + "word " * 300, max_section_tokens=100)
        assert all(_tokens(c["text"]) <= 100 for c in chunks)

    def test_small_section_not_split(self):
        chunks = chunk_markdown("## S\n" + "word " * 30, max_section_tokens=400)
        assert len(chunks) == 1


class TestChunkText:
    def test_splits_by_paragraph(self):
        para = "word " * 20  # ~20 tokens
        text = "\n\n".join([para] * 6)
        chunks = chunk_text(text, chunk_tokens=30)
        assert len(chunks) >= 2

    def test_skips_short_remainder(self):
        # single very short paragraph — under 10-token threshold
        chunks = chunk_text("hi", chunk_tokens=100)
        assert chunks == []

    def test_chunk_index_increments(self):
        text = "\n\n".join(["word " * 20] * 5)
        chunks = chunk_text(text, chunk_tokens=30)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_url_propagated(self):
        text = "\n\n".join(["word " * 20] * 3)
        chunks = chunk_text(text, source_url="test.pdf")
        assert all(c["source_url"] == "test.pdf" for c in chunks)
