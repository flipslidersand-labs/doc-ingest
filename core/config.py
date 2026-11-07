"""Central configuration: one frozen Settings object loaded from the environment (#261).

.env is loaded once, inside load_settings(), instead of as an import side effect.
Modules call get_settings() when they need a value; tests swap the cached object
(see tests/conftest.py override_settings) instead of patching module constants.
"""

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # embedding (core/embedding.py)
    embed_url: str = "http://localhost:9092/embed/batch"
    # Secondary endpoint used when embed_url is unreachable or returns 501. Empty = none.
    embed_fallback_url: str = ""
    embed_api_key: str | None = field(default=None, repr=False)  # None -> no X-API-Key header sent
    embed_collection: str = "sessions"  # embedding-svc のモデルルーティング用
    embed_timeout: float = 180.0  # CPU e5 のコールドロードを許容
    embed_batch: int = 16  # 1 POST あたりの最大テキスト数
    embed_backend: str = "http"  # "http" | "onnx_gpu"
    onnx_model_path: str = ""
    # vector DB (core/qdrant.py)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = field(
        default=None, repr=False
    )  # None -> unauthenticated (dev/localhost)
    # distiller (core/distiller.py)
    local_llm_url: str = "http://localhost:8080"
    local_llm_model: str = ""  # llama-server は未指定で可
    local_llm_timeout: float = 120.0
    local_llm_max_tokens: int = 1500
    anthropic_api_key: str = field(default="", repr=False)
    anthropic_model: str = "claude-haiku-4-5-20251001"
    # ingest pipeline (core/ingest_pipeline.py)
    # llama-server は 1 台で複数スロットが n_ctx を共有し、長い入力の並列は全滅する。既定は直列。
    distill_workers: int = 1
    # external sync (ingest/external.py)
    sync_workers: int = 4
    # The URL itself is a credential. Empty -> no Discord notification.
    discord_webhook_url: str = field(default="", repr=False)
    # size limits (core/chunker.py, core/ingest_pipeline.py, core/url_guard.py)
    # 1 文書から作るチャンク数の上限。巨大・悪意ある入力で LLM 要約と埋め込みが際限なく走るのを防ぐ (#224)。
    max_chunks_per_doc: int = 1000
    # ローカルの markdown 1 ファイルの上限 (バイト)。これを超えるファイルは読まずに見送る (#224)。
    max_local_bytes: int = 10 * 1024 * 1024
    # safe_download の既定の上限 (展開後のバイト数)。巨大な応答でディスク/メモリを埋めさせない。
    max_download_bytes: int = 50 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.embed_batch < 1:
            # range(0, n, 0) raises an opaque ValueError and a negative step silently
            # yields no batches (upsert() would then drop payloads) — fail loudly (#135).
            raise ValueError(f"EMBED_BATCH must be >= 1, got {self.embed_batch}")
        # ThreadPoolExecutor(max_workers<=0) raises ValueError; surface it at load time.
        if self.distill_workers < 1:
            raise ValueError(f"DISTILL_WORKERS must be >= 1, got {self.distill_workers}")
        if self.sync_workers < 1:
            raise ValueError(f"SYNC_WORKERS must be >= 1, got {self.sync_workers}")
        # 0 以下だと全チャンク/全ファイル/全ダウンロードを拒否してしまう。
        for env_name, value in (
            ("MAX_CHUNKS_PER_DOC", self.max_chunks_per_doc),
            ("MAX_LOCAL_BYTES", self.max_local_bytes),
            ("MAX_DOWNLOAD_BYTES", self.max_download_bytes),
        ):
            if value < 1:
                raise ValueError(f"{env_name} must be >= 1, got {value}")


def _int(env: Mapping[str, str], name: str, default: str | int) -> int:
    raw = env.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _float(env: Mapping[str, str], name: str, default: str) -> float:
    raw = env.get(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


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
        embed_timeout=_float(env, "EMBED_TIMEOUT", "180"),
        embed_batch=_int(env, "EMBED_BATCH", "16"),
        embed_backend=s("EMBED_BACKEND", d.embed_backend),
        onnx_model_path=s("ONNX_MODEL_PATH", d.onnx_model_path),
        qdrant_url=s("QDRANT_URL", d.qdrant_url),
        qdrant_api_key=env.get("QDRANT_API_KEY") or None,
        local_llm_url=s("LOCAL_LLM_URL", d.local_llm_url),
        local_llm_model=s("LOCAL_LLM_MODEL", d.local_llm_model),
        local_llm_timeout=_float(env, "LOCAL_LLM_TIMEOUT", "120"),
        local_llm_max_tokens=_int(env, "LOCAL_LLM_MAX_TOKENS", "1500"),
        anthropic_api_key=s("ANTHROPIC_API_KEY", d.anthropic_api_key),
        anthropic_model=s("ANTHROPIC_MODEL", d.anthropic_model),
        distill_workers=_int(env, "DISTILL_WORKERS", "1"),
        sync_workers=_int(env, "SYNC_WORKERS", "4"),
        discord_webhook_url=s("DISCORD_WEBHOOK_URL", d.discord_webhook_url),
        max_chunks_per_doc=_int(env, "MAX_CHUNKS_PER_DOC", str(d.max_chunks_per_doc)),
        max_local_bytes=_int(env, "MAX_LOCAL_BYTES", str(d.max_local_bytes)),
        max_download_bytes=_int(env, "MAX_DOWNLOAD_BYTES", str(d.max_download_bytes)),
    )


_settings: Settings | None = None
_lock = threading.Lock()


def get_settings() -> Settings:
    """Return the process-wide Settings, loading it on first use."""
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = load_settings()
    return _settings
