"""LLM distillation via Ollama (local inference)."""
import os
import urllib.request
import json

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def _generate(prompt: str) -> str:
    payload = json.dumps({"model": MODEL, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["response"].strip()


def distill_paper(abstract: str, body: str = "") -> str:
    prompt = (
        f"この論文の実装に直接使える知見・手法・数値を200字以内で。理論背景は省く。\n\n"
        f"Abstract:\n{abstract}\n\nBody (抜粋):\n{body[:3000]}"
    )
    return _generate(prompt)


def distill_design_doc(section: str) -> str:
    prompt = f"この設計の決定理由・制約・代替案を200字以内で。\n\n{section[:4000]}"
    return _generate(prompt)


def distill_webpage(content: str) -> str:
    prompt = f"この技術記事の実装に使える要点を200字以内で。\n\n{content[:5000]}"
    return _generate(prompt)
