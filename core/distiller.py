"""LLM distillation: local llama-server (OpenAI-compatible) → Anthropic Haiku fallback.

Both failing raises DistillError."""

import json
import logging
import os
import urllib.request

_log = logging.getLogger(__name__)

# llama-server の OpenAI 互換 API (/v1/chat/completions)。Ollama は撤去済み。
# 入力は最大 5000 字なので n_ctx が十分な台を指す (yuki gemma-4-12b: localhost:18080, n_ctx 32768)。
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "")  # llama-server は未指定で可
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "120"))
# 思考するモデルは max_tokens が小さいと本文が空になる (40 で空、1500 で正答: llm CLI の実測)。
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "1500"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


class DistillError(RuntimeError):
    """要約バックエンドがすべて失敗した。原文を要約の代わりに埋め込まないため例外にする。"""


def _local(prompt: str) -> str:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": LOCAL_LLM_MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }
    if LOCAL_LLM_MODEL:
        body["model"] = LOCAL_LLM_MODEL
    req = urllib.request.Request(
        f"{LOCAL_LLM_URL.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=LOCAL_LLM_TIMEOUT) as resp:
        choice = json.loads(resp.read())["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    if not text:
        # 思考でトークンを使い切ると本文が空になる。空文字を要約として埋め込まない。
        raise ValueError(f"empty response (finish_reason={choice.get('finish_reason')})")
    return text


def _anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _generate(prompt: str, *, local_only: bool = False) -> str:
    """local_only=True は非公開ソース用: Anthropic へ送らず、ローカル LLM が失敗したら DistillError。"""
    try:
        return _local(prompt)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        if local_only:
            raise DistillError(
                f"local LLM failed ({e}); cloud fallback disabled for private source"
            ) from e
        _log.warning("local LLM unavailable (%s), trying Anthropic Haiku", e)

    if not ANTHROPIC_API_KEY:
        raise DistillError("local LLM failed and ANTHROPIC_API_KEY is not set")
    try:
        return _anthropic(prompt)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        raise DistillError(f"local LLM and Anthropic both failed ({e})") from e


def distill_paper(abstract: str, body: str = "") -> str:
    prompt = (
        f"この論文の実装に直接使える知見・手法・数値を200字以内で。理論背景は省く。\n\n"
        f"Abstract:\n{abstract}\n\nBody (抜粋):\n{body[:3000]}"
    )
    return _generate(prompt)


def distill_design_doc(section: str) -> str:
    """設計 doc は非公開なので常にローカル LLM のみ。"""
    prompt = f"この設計の決定理由・制約・代替案を200字以内で。\n\n{section[:4000]}"
    return _generate(prompt, local_only=True)


def distill_text(text: str, purpose: str, *, local_only: bool = False) -> str:
    """Generic distillation for any text with a caller-specified purpose."""
    prompt = f"{purpose}\n\n{text[:4000]}"
    return _generate(prompt, local_only=local_only)


def distill_webpage(content: str) -> str:
    prompt = f"この技術記事の実装に使える要点を200字以内で。\n\n{content[:5000]}"
    return _generate(prompt)
