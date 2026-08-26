"""LLM distillation: Ollama → Anthropic Haiku fallback → raw text."""

import json
import os
import urllib.request

from core.logging import get_logger

_log = get_logger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def _ollama(prompt: str) -> str:
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        return json.loads(resp.read())["response"].strip()


def _anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _generate(prompt: str, source_text: str = "") -> str:
    try:
        return _ollama(prompt)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # noqa: BLE001
        _log.warning("Ollama unavailable (%s), trying Anthropic Haiku", e)

    if ANTHROPIC_API_KEY:
        try:
            return _anthropic(prompt)
        except (MemoryError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001
            _log.warning("Anthropic unavailable (%s), using raw text fallback", e)

    fallback = source_text[:500] if source_text else prompt[:500]
    _log.warning("raw text fallback used for '%s'", (source_text or prompt)[:60])
    return fallback


def distill_paper(abstract: str, body: str = "") -> str:
    prompt = (
        f"この論文の実装に直接使える知見・手法・数値を200字以内で。理論背景は省く。\n\n"
        f"Abstract:\n{abstract}\n\nBody (抜粋):\n{body[:3000]}"
    )
    return _generate(prompt, source_text=abstract)


def distill_design_doc(section: str) -> str:
    prompt = f"この設計の決定理由・制約・代替案を200字以内で。\n\n{section[:4000]}"
    return _generate(prompt, source_text=section)


def distill_text(text: str, purpose: str) -> str:
    """Generic distillation for any text with a caller-specified purpose."""
    prompt = f"{purpose}\n\n{text[:4000]}"
    return _generate(prompt, source_text=text)


def distill_webpage(content: str) -> str:
    prompt = f"この技術記事の実装に使える要点を200字以内で。\n\n{content[:5000]}"
    return _generate(prompt, source_text=content)
