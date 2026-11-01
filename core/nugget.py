"""BM25-based nugget extraction: pull relevant sentences from Qdrant chunks."""

import math
import re
import unicodedata
from collections import Counter

# 文の区切り。日本語の。！？は直後に空白が無くても区切る。ASCII の . ! ? は直後に空白がある
# ときだけ ("3.14" や "v1.2" を割らない)。改行も区切り (要約は箇条書きや段落で書かれる)。
_SENT_RE = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+|\n+")

# 空白で語が区切られない文字の連なり (ひらがな・カタカナ・長音・漢字・々)。bigram に分ける。
_CJK_RUN = r"[぀-ヿ㐀-鿿豈-﫿々]+"
_TOKEN_RE = re.compile(rf"[a-z0-9_]+|{_CJK_RUN}")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s and s.strip()]


def _tokenize(text: str) -> list[str]:
    """ASCII の語はそのまま、日本語は辞書なしで文字 bigram (1 文字だけの連なりはそのまま)。

    NFKC で全角英数・半角カナを正規化してから小文字にする。形態素解析器に依存しないので、
    語の切れ目とずれる bigram も出るが、BM25 の IDF が共通の bigram (助詞の連なり) を弱める。
    """
    tokens: list[str] = []
    for run in _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower()):
        if run[0].isascii() or len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _doc_freq(tokenized: list[list[str]]) -> Counter:
    """語ごとの文書頻度 = その語を含む「文の数」。同じ文の中で何回出ても 1 と数える。"""
    return Counter(term for doc in tokenized for term in set(doc))


def _bm25_scores(query: str, sentences: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not sentences:
        return []
    tokenized = [_tokenize(s) for s in sentences]
    n = len(sentences)
    avgdl = sum(len(t) for t in tokenized) / n
    # IDF は「その語を含む文の数 (文書頻度)」から。以前は文内の出現回数 f を df と取り違えていて、
    # 稀な語もありふれた語も同じ重みになり、同じ文内で繰り返される語ほど軽くなっていた (#228)。
    df = _doc_freq(tokenized)
    q_terms = _tokenize(query)
    scores = []
    for doc in tokenized:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            num = f * (k1 + 1)
            den = f + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * (num / den)
        scores.append(score)
    return scores


def extract_nuggets(query: str, text: str, top_k: int = 3) -> str:
    """Return top_k most query-relevant sentences from text joined as a string.

    Ties (including "nothing matched") keep document order, not the sentences' sort order.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text
    scores = _bm25_scores(query, sentences)
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    return " ".join(sentences[i] for i in ranked[:top_k])


def apply_nuggets(query: str, results: list[dict], top_k: int = 3) -> list[dict]:
    """Replace 'text' field in each result dict with extracted nuggets."""
    out = []
    for r in results:
        text = r.get("text", "")
        nugget = extract_nuggets(query, text, top_k=top_k)
        out.append({**r, "text": nugget, "nugget_source_len": len(text)})
    return out
