"""Tests for core/chunker.py"""

from typing import ClassVar

import pytest
import tiktoken

from core.chunker import chunk_markdown, chunk_text
from tests.conftest import override_settings

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

    def test_chunk_dict_has_exactly_the_documented_keys(self):
        """#227: 誰も読まない source_url を持たせない。呼び出し側が持つ出所は自分で付ける。"""
        chunks = chunk_markdown("## S\n" + "b " * 30)
        assert set(chunks[0]) == {"text", "section", "chunk_index"}

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

    def test_normal_paragraph_flushed_before_oversized_paragraph(self):
        # A normal-sized paragraph followed by one that alone exceeds
        # max_tokens must flush the pending buffer as its own piece
        # *before* hard-splitting the oversized one (core/chunker.py:38-40).
        small_para = "word " * 10
        giant_para = "word " * 300
        body = "## H\n" + small_para + "\n\n" + giant_para
        chunks = chunk_markdown(body, max_section_tokens=100)

        assert all(_tokens(c["text"]) <= 100 for c in chunks)
        # The small paragraph must survive as its own flushed piece (with the
        # section heading prefix chunk_markdown adds to a section's first
        # piece), not get merged into / lost by the giant paragraph's
        # hard-split — i.e. exactly one piece has word-count == 10.
        word_counts = [c["text"].count("word") for c in chunks]
        assert word_counts.count(10) == 1
        # No content is dropped: every word from both paragraphs reappears.
        assert sum(word_counts) == 10 + 300


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

    def test_chunk_dict_has_exactly_the_documented_keys(self):
        text = "\n\n".join(["word " * 20] * 3)
        chunks = chunk_text(text)
        assert chunks and all(set(c) == {"text", "section", "chunk_index"} for c in chunks)


class TestHardSplitKeepsWholeCharacters:
    """#220: トークン境界で切って decode すると、マルチバイト文字の途中で切れて U+FFFD になった。"""

    SAMPLES: ClassVar[dict[str, str]] = {
        "japanese": "これは日本語のテストです。" * 30,
        "emoji_zwj": "👨‍👩‍👧‍👦🎉🚀🙂" * 40,
        "mixed": "Hello 世界 😀 テスト " * 25,
        "rare_kanji": "𠮷野家の𩸽" * 30,
        "ascii": "word " * 200,
    }

    @pytest.mark.parametrize("name", list(SAMPLES))
    @pytest.mark.parametrize("max_tokens", [5, 20, 100])
    def test_pieces_join_back_to_the_original_text_without_replacement_chars(
        self, name, max_tokens
    ):
        from core.chunker import _hard_split_tokens

        text = self.SAMPLES[name]
        pieces = _hard_split_tokens(text, max_tokens)
        assert "".join(pieces) == text  # 文字の欠落も重複も無い
        assert "�" not in "".join(pieces)

    @pytest.mark.parametrize("name", list(SAMPLES))
    @pytest.mark.parametrize("max_tokens", [5, 20, 100])
    def test_every_piece_fits_the_token_limit(self, name, max_tokens):
        from core.chunker import _count_tokens, _hard_split_tokens

        for piece in _hard_split_tokens(self.SAMPLES[name], max_tokens):
            assert _count_tokens(piece) <= max_tokens

    def test_each_piece_is_valid_utf8_on_its_own(self):
        from core.chunker import _hard_split_tokens

        for piece in _hard_split_tokens(self.SAMPLES["japanese"], 7):
            piece.encode("utf-8")  # 例外にならない
            assert "�" not in piece

    def test_a_tiny_limit_still_makes_progress_and_keeps_characters_whole(self):
        """1 文字が複数トークンにまたがる状況で上限が 1 でも、無限ループせず文字を壊さない。"""
        from core.chunker import _hard_split_tokens

        text = "日本語𠮷😀" * 3
        pieces = _hard_split_tokens(text, 1)
        assert "".join(pieces) == text and "�" not in "".join(pieces)

    def test_random_text_roundtrips(self):
        import random

        from core.chunker import _count_tokens, _hard_split_tokens

        rng = random.Random(7)
        alphabet = list("あいうえお漢字日本語abcXYZ 012\n😀🎉é")
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 300)))
            limit = rng.choice([3, 7, 15, 50])
            pieces = _hard_split_tokens(text, limit)
            assert "".join(pieces) == text
            assert all(_count_tokens(p) <= limit for p in pieces)

    def test_empty_text_gives_no_pieces(self):
        from core.chunker import _hard_split_tokens

        assert _hard_split_tokens("", 10) == []

    def test_a_giant_japanese_paragraph_through_chunk_markdown_stays_intact(self):
        """入口 (chunk_markdown) からも、巨大な 1 段落の日本語が化けない。"""
        from core.chunker import chunk_markdown

        para = "これは日本語の長い段落です。" * 400  # 1 つの段落で上限を大きく超える
        chunks = chunk_markdown("## 節\n" + para)
        assert len(chunks) > 1
        joined = "".join(c["text"] for c in chunks)
        assert "�" not in joined


class TestCapChunks:
    """1 文書のチャンク数に上限をかける (#224)。"""

    def test_default_cap_is_1000(self):
        from core.config import load_settings

        assert load_settings({}).max_chunks_per_doc == 1000
        assert load_settings({"MAX_CHUNKS_PER_DOC": "7"}).max_chunks_per_doc == 7

    def test_exactly_at_the_cap_is_returned_unchanged_without_a_warning(self, mocker, caplog):
        from core.chunker import cap_chunks

        override_settings(max_chunks_per_doc=3)
        chunks = [{"chunk_index": i} for i in range(3)]

        with caplog.at_level("WARNING"):
            assert cap_chunks(chunks, "doc") == chunks

        assert "MAX_CHUNKS_PER_DOC" not in caplog.text

    def test_over_the_cap_keeps_the_first_chunks_and_warns(self, mocker, caplog):
        from core.chunker import cap_chunks

        override_settings(max_chunks_per_doc=3)
        chunks = [{"chunk_index": i} for i in range(10)]

        with caplog.at_level("WARNING"):
            capped = cap_chunks(chunks, "big.md")

        assert [c["chunk_index"] for c in capped] == [0, 1, 2]
        assert "big.md" in caplog.text and "MAX_CHUNKS_PER_DOC" in caplog.text
