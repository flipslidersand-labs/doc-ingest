"""LLM distillation via Anthropic API (Haiku for cost efficiency)."""
import os

import anthropic

_client: anthropic.Anthropic | None = None

MODEL = "claude-haiku-4-5-20251001"


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def distill_paper(abstract: str, body: str = "") -> str:
    prompt = (
        f"この論文の実装に直接使える知見・手法・数値を200字以内で。理論背景は省く。\n\n"
        f"Abstract:\n{abstract}\n\nBody (抜粋):\n{body[:3000]}"
    )
    msg = _get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def distill_design_doc(section: str) -> str:
    prompt = (
        f"この設計の決定理由・制約・代替案を200字以内で。\n\n{section[:4000]}"
    )
    msg = _get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def distill_webpage(content: str) -> str:
    prompt = (
        f"この技術記事の実装に使える要点を200字以内で。\n\n{content[:5000]}"
    )
    msg = _get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
