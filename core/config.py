"""Central configuration: one frozen Settings object loaded from the environment (#261).

.env is loaded once, inside load_settings(), instead of as an import side effect.
Modules call get_settings() when they need a value; tests swap the cached object
(see tests/conftest.py override_settings) instead of patching module constants.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # embedding (core/embedding.py)
    embed_url: str = "http://localhost:9092/embed/batch"
    # Secondary endpoint used when embed_url is unreachable or returns 501. Empty = none.
    embed_fallback_url: str = ""
    embed_api_key: str | None = None  # None -> no X-API-Key header sent
    embed_collection: str = "sessions"  # embedding-svc のモデルルーティング用
    embed_timeout: float = 180.0  # CPU e5 のコールドロードを許容
    embed_batch: int = 16  # 1 POST あたりの最大テキスト数
    embed_backend: str = "http"  # "http" | "onnx_gpu"
    onnx_model_path: str = ""
    # vector DB (core/qdrant.py)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None  # None -> unauthenticated (dev/localhost)
    # distiller (core/distiller.py)
    local_llm_url: str = "http://localhost:8080"
    local_llm_model: str = ""  # llama-server は未指定で可
    local_llm_timeout: float = 120.0
    local_llm_max_tokens: int = 1500
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    def __post_init__(self) -> None:
        if self.embed_batch < 1:
            # range(0, n, 0) raises an opaque ValueError and a negative step silently
            # yields no batches (upsert() would then drop payloads) — fail loudly (#135).
            raise ValueError(f"EMBED_BATCH must be >= 1, got {self.embed_batch}")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env`. With env=None, load .env once and read os.environ."""
    if env is None:
        load_dotenv()
        env = os.environ
    d = Settings()

    def s(name: str, default: str) -> str:
        return env.get(name, default)

    return Settings(
        embed_url=s("EMBED_URL", d.embed_url),
        embed_fallback_url=s("EMBED_FALLBACK_URL", d.embed_fallback_url),
        embed_api_key=env.get("EMBED_API_KEY") or None,
        embed_collection=s("EMBED_COLLECTION", d.embed_collection),
        embed_timeout=float(s("EMBED_TIMEOUT", "180")),
        embed_batch=int(s("EMBED_BATCH", "16")),
        embed_backend=s("EMBED_BACKEND", d.embed_backend),
        onnx_model_path=s("ONNX_MODEL_PATH", d.onnx_model_path),
        qdrant_url=s("QDRANT_URL", d.qdrant_url),
        qdrant_api_key=env.get("QDRANT_API_KEY") or None,
        local_llm_url=s("LOCAL_LLM_URL", d.local_llm_url),
        local_llm_model=s("LOCAL_LLM_MODEL", d.local_llm_model),
        local_llm_timeout=float(s("LOCAL_LLM_TIMEOUT", "120")),
        local_llm_max_tokens=int(s("LOCAL_LLM_MAX_TOKENS", "1500")),
        anthropic_api_key=s("ANTHROPIC_API_KEY", d.anthropic_api_key),
        anthropic_model=s("ANTHROPIC_MODEL", d.anthropic_model),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide Settings, loading it on first use."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
