"""BM25-based nugget extraction: pull relevant sentences from Qdrant chunks."""

import math
import re
from collections import Counter

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _bm25_scores(query: str, sentences: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not sentences:
        return []
    tokenized = [_tokenize(s) for s in sentences]
    avgdl = sum(len(t) for t in tokenized) / len(tokenized)
    q_terms = _tokenize(query)
    scores = []
    for doc in tokenized:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            idf = math.log(1 + (len(sentences) - f + 0.5) / (f + 0.5))
            num = f * (k1 + 1)
            den = f + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * (num / den)
        scores.append(score)
    return scores


def extract_nuggets(query: str, text: str, top_k: int = 3) -> str:
    """Return top_k most query-relevant sentences from text joined as a string."""
    sentences = _split_sentences(text)
    if not sentences:
        return text
    scores = _bm25_scores(query, sentences)
    ranked = sorted(zip(scores, sentences), reverse=True)
    return " ".join(s for _, s in ranked[:top_k])


def apply_nuggets(query: str, results: list[dict], top_k: int = 3) -> list[dict]:
    """Replace 'text' field in each result dict with extracted nuggets."""
    out = []
    for r in results:
        text = r.get("text", "")
        nugget = extract_nuggets(query, text, top_k=top_k)
        out.append({**r, "text": nugget, "nugget_source_len": len(text)})
    return out
