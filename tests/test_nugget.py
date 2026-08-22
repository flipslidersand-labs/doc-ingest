"""Tests for core.nugget BM25 extraction."""
from core.nugget import apply_nuggets, extract_nuggets

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
