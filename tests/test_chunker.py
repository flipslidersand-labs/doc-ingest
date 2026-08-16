"""Tests for core/chunker.py"""
from core.chunker import chunk_markdown, chunk_text


class TestChunkMarkdown:
    def test_splits_on_h2(self):
        md = "## Section A\n" + "x" * 60 + "\n\n## Section B\n" + "y" * 60
        chunks = chunk_markdown(md)
        assert len(chunks) == 2
        assert chunks[0]["section"] == "Section A"
        assert chunks[1]["section"] == "Section B"

    def test_splits_on_h3(self):
        md = "### Deep\n" + "z" * 60
        chunks = chunk_markdown(md)
        assert len(chunks) == 1
        assert chunks[0]["section"] == "Deep"

    def test_skips_short_sections(self):
        md = "## Tiny\nhi\n\n## Real\n" + "a" * 60
        chunks = chunk_markdown(md)
        assert len(chunks) == 1
        assert chunks[0]["section"] == "Real"

    def test_single_line_heading_no_slice_bug(self):
        # regression: heading_end=-1 caused body[:-1] before fix
        md = "## " + "a" * 60
        chunks = chunk_markdown(md)
        assert len(chunks) == 1
        # section should be the full text (no trailing char missing)
        assert chunks[0]["section"] == "a" * 60

    def test_source_url_propagated(self):
        md = "## S\n" + "b" * 60
        chunks = chunk_markdown(md, source_url="http://example.com")
        assert chunks[0]["source_url"] == "http://example.com"

    def test_empty_text_returns_empty(self):
        assert chunk_markdown("") == []


class TestChunkText:
    def test_splits_by_paragraph(self):
        # each paragraph is ~60 chars; chunk_size=80 → 2 paras per chunk → 3 chunks
        para = "word " * 12  # ~60 chars
        text = ("\n\n".join([para] * 6))
        chunks = chunk_text(text, chunk_size=80)
        assert len(chunks) >= 2

    def test_skips_short_remainder(self):
        # single very short paragraph should be excluded
        chunks = chunk_text("hi", chunk_size=100)
        assert chunks == []

    def test_chunk_index_increments(self):
        text = "\n\n".join(["word " * 20] * 5)
        chunks = chunk_text(text, chunk_size=50)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_url_propagated(self):
        text = "\n\n".join(["word " * 20] * 3)
        chunks = chunk_text(text, source_url="test.pdf")
        assert all(c["source_url"] == "test.pdf" for c in chunks)
