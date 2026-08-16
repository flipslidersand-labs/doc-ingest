"""LLM distillation: Ollama → Anthropic Haiku fallback → raw text."""
import json
import os
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


<<<<<<< HEAD
def _generate(prompt: str, source_text: str = "") -> str:
    payload = json.dumps({"model": MODEL, "prompt": prompt, "stream": False}).encode()
=======
def _ollama(prompt: str) -> str:
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
>>>>>>> 9ddbfc0 (feat(distiller): Anthropic Haiku fallback を追加 (closes #10))
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["response"].strip()
    except Exception as e:
        print(f"[distiller] Ollama unavailable ({e}), using raw text fallback")
        return source_text[:500] if source_text else prompt[:500]


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
    except Exception as e:
        print(f"[distiller] Ollama unavailable ({e}), trying Anthropic Haiku")

    if ANTHROPIC_API_KEY:
        try:
            return _anthropic(prompt)
        except Exception as e:
            print(f"[distiller] Anthropic unavailable ({e}), using raw text fallback")

    return source_text[:500] if source_text else prompt[:500]


def distill_paper(abstract: str, body: str = "") -> str:
    prompt = (
        f"この論文の実装に直接使える知見・手法・数値を200字以内で。理論背景は省く。\n\n"
        f"Abstract:\n{abstract}\n\nBody (抜粋):\n{body[:3000]}"
    )
    return _generate(prompt, source_text=abstract)


def distill_design_doc(section: str) -> str:
    prompt = f"この設計の決定理由・制約・代替案を200字以内で。\n\n{section[:4000]}"
    return _generate(prompt, source_text=section)


def distill_webpage(content: str) -> str:
    prompt = f"この技術記事の実装に使える要点を200字以内で。\n\n{content[:5000]}"
    return _generate(prompt, source_text=content)
