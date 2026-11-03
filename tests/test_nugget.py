"""Tests for core.nugget BM25 extraction."""

import pytest

from core.nugget import (
    _bm25_scores,
    _split_sentences,
    _tokenize,
    apply_nuggets,
    extract_nuggets,
)

TEXT = (
    "Python supports list comprehensions for concise loops. "
    "The weather is nice today. "
    "BM25 ranks documents by term frequency and inverse document frequency. "
    "I enjoy coffee in the morning. "
    "Sparse retrieval methods include TF-IDF and BM25."
)


def test_extract_returns_top_k_sentences():
    nugget = extract_nuggets("BM25 retrieval", TEXT, top_k=2)
    assert "BM25" in nugget


def test_extract_fewer_sentences_than_top_k():
    short = "Only one sentence here."
    result = extract_nuggets("query", short, top_k=5)
    assert result == short


def test_extract_empty_text_returns_empty():
    assert extract_nuggets("query", "") == ""


def test_apply_nuggets_replaces_text():
    results = [{"score": 0.9, "text": TEXT, "source_url": "http://example.com"}]
    out = apply_nuggets("BM25 retrieval", results, top_k=2)
    assert len(out) == 1
    assert out[0]["text"] != TEXT
    assert "BM25" in out[0]["text"]
    assert out[0]["nugget_source_len"] == len(TEXT)


def test_apply_nuggets_preserves_other_fields():
    results = [{"score": 0.5, "text": TEXT, "section": "intro"}]
    out = apply_nuggets("query", results)
    assert out[0]["section"] == "intro"
    assert out[0]["score"] == 0.5


def test_apply_nuggets_empty_list():
    assert apply_nuggets("query", []) == []


# ---------------------------------------------------------------------------
# #228: 日本語 (文分割・トークン化)・IDF (文書頻度)・タイブレーク
# ---------------------------------------------------------------------------
JA = (
    "決定理由は、既存のインフラで運用実績があることです。"
    "制約として、単一障害点になる点が挙げられます。"
    "代替案として Memcached を検討しましたが、永続化が無いため却下しました。"
    "移行は段階的に行い、まず読み取りだけを切り替えます。"
)


class TestSplitSentences:
    def test_japanese_sentences_split_without_spaces(self):
        """以前は空白が無いと 1 文のままで、--nugget は日本語では何もしなかった。"""
        assert len(_split_sentences(JA)) == 4
        assert _split_sentences(JA)[0].endswith("あることです。")

    @pytest.mark.parametrize("mark", ["。", "！", "？"])
    def test_each_japanese_terminator_splits(self, mark):
        assert _split_sentences(f"前の文{mark}次の文") == [f"前の文{mark}", "次の文"]

    def test_english_still_splits_on_terminator_plus_space(self):
        assert _split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]

    def test_decimals_and_versions_are_not_split(self):
        assert _split_sentences("Pi is 3.14 and v1.2 works.") == ["Pi is 3.14 and v1.2 works."]

    def test_newlines_split(self):
        assert _split_sentences("- 一つ目\n- 二つ目\n\n三つ目") == [
            "- 一つ目",
            "- 二つ目",
            "三つ目",
        ]

    def test_mixed_japanese_and_english(self):
        assert len(_split_sentences("Redis を採用した。Memcached is rejected. 理由は永続化。")) == 3

    def test_empty_and_whitespace(self):
        assert _split_sentences("") == [] and _split_sentences("  \n ") == []


class TestTokenize:
    def test_ascii_words_are_lowercased(self):
        assert _tokenize("BM25 Ranks Documents") == ["bm25", "ranks", "documents"]

    def test_japanese_runs_become_character_bigrams(self):
        assert _tokenize("ベクトル") == ["ベク", "クト", "トル"]

    def test_a_single_japanese_character_is_kept(self):
        assert _tokenize("に") == ["に"]

    def test_full_width_is_normalised_to_half_width(self):
        assert _tokenize("ＢＭ２５") == _tokenize("BM25") == ["bm25"]

    def test_mixed_scripts(self):
        assert _tokenize("ベクトルDB") == ["ベク", "クト", "トル", "db"]

    def test_punctuation_is_dropped(self):
        assert _tokenize("、。！ ?") == []


class TestScoring:
    def test_a_rare_term_outweighs_a_common_one(self):
        """#228: IDF は文書頻度から。以前は文内の出現回数を df と取り違え、稀な語もありふれた語も
        同じ重み (この例では同点) だった。"""
        sentences = ["rare alpha", "common beta", "common gamma", "common delta"]
        scores = _bm25_scores("common rare", sentences)
        assert scores[0] > max(scores[1:])

    def test_repeating_a_rare_term_inside_one_sentence_does_not_lower_its_weight(self):
        sentences = ["rare rare rare alpha", "common beta", "common gamma", "common delta"]
        scores = _bm25_scores("rare", sentences)
        assert scores[0] > 0 and scores[1] == scores[2] == scores[3] == 0

    def test_document_frequency_counts_sentences_not_occurrences(self):
        """df は「その語を含む文の数」。1 文の中で繰り返されても 1 と数える。
        (スコア経由だと tf の飽和で区別できないので、df を直接確かめる。)"""
        from core.nugget import _doc_freq

        df = _doc_freq([["rare", "rare", "rare", "x"], ["common", "y"], ["common", "z"]])
        assert df["rare"] == 1  # 3 回出るが 1 文だけ
        assert df["common"] == 2
        assert df["missing"] == 0

    def test_a_term_in_every_sentence_gets_a_small_weight(self):
        sentences = ["common one", "common two", "common three"]
        assert max(_bm25_scores("common", sentences)) < 0.5  # df == N → idf は小さい

    def test_no_match_scores_zero(self):
        assert _bm25_scores("zzz", ["a b", "c d"]) == [0.0, 0.0]

    def test_empty_sentences(self):
        assert _bm25_scores("q", []) == []


class TestJapaneseRetrieval:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("単一障害点になる制約", "単一障害点"),
            ("Memcached を却下した理由", "Memcached"),
            ("どう移行するか 読み取り", "読み取り"),
            ("運用実績", "運用実績"),
            ("永続化が無い", "永続化"),
        ],
    )
    def test_the_matching_sentence_comes_first(self, query, expected):
        top = extract_nuggets(query, JA, top_k=1)
        assert expected in top
        assert top != JA  # 以前は全文がそのまま返っていた

    def test_top_k_limits_the_number_of_sentences(self):
        assert len(_split_sentences(extract_nuggets("制約", JA, top_k=2))) == 2


class TestTieBreaking:
    def test_when_nothing_matches_the_first_sentences_come_back_in_document_order(self):
        """以前は同点を文の辞書順の逆で並べ、一致が無いと先頭ではなく辞書順で最後の文が返った。"""
        text = "apple is here. zebra is here. mango is here."
        assert extract_nuggets("qqq", text, top_k=1) == "apple is here."
        assert extract_nuggets("qqq", text, top_k=2) == "apple is here. zebra is here."

    def test_equal_scores_keep_document_order(self):
        text = "zebra token. apple token."
        assert extract_nuggets("token", text, top_k=2) == "zebra token. apple token."

    def test_a_query_of_only_punctuation_does_not_crash(self):
        assert extract_nuggets("、。", JA, top_k=1) == _split_sentences(JA)[0]
