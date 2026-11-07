"""Tests for core/config.py (#261)."""

import pytest

from core import config
from core.config import Settings, load_settings


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)


def test_defaults_when_env_empty():
    s = load_settings({})
    assert s == Settings()
    assert s.embed_url == "http://localhost:9092/embed/batch"
    assert s.embed_batch == 16
    assert s.embed_timeout == 180.0
    assert s.embed_backend == "http"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.local_llm_url == "http://localhost:8080"
    assert s.local_llm_max_tokens == 1500
    assert s.anthropic_model == "claude-haiku-4-5-20251001"


def test_reads_environment_values():
    s = load_settings(
        {
            "EMBED_URL": "http://e:1/x",
            "EMBED_TIMEOUT": "7.5",
            "EMBED_BATCH": "32",
            "EMBED_BACKEND": "onnx_gpu",
            "ONNX_MODEL_PATH": "/m/model.onnx",
            "QDRANT_URL": "http://q:6333",
            "LOCAL_LLM_MAX_TOKENS": "99",
            "ANTHROPIC_API_KEY": "k",
        }
    )
    assert (s.embed_url, s.embed_timeout, s.embed_batch) == ("http://e:1/x", 7.5, 32)
    assert (s.embed_backend, s.onnx_model_path) == ("onnx_gpu", "/m/model.onnx")
    assert s.qdrant_url == "http://q:6333"
    assert s.local_llm_max_tokens == 99
    assert s.anthropic_api_key == "k"


def test_empty_api_keys_become_none():
    s = load_settings({"EMBED_API_KEY": "", "QDRANT_API_KEY": ""})
    assert s.embed_api_key is None
    assert s.qdrant_api_key is None
    s = load_settings({"EMBED_API_KEY": "a", "QDRANT_API_KEY": "b"})
    assert (s.embed_api_key, s.qdrant_api_key) == ("a", "b")


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_embed_batch_must_be_positive(bad):
    with pytest.raises(ValueError, match="EMBED_BATCH must be >= 1"):
        load_settings({"EMBED_BATCH": bad})


def test_settings_is_frozen():
    with pytest.raises(AttributeError):
        Settings().embed_batch = 2  # type: ignore[misc]


def test_get_settings_loads_once_and_caches(monkeypatch):
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("EMBED_BATCH", "5")
    first = config.get_settings()
    assert first.embed_batch == 5
    monkeypatch.setenv("EMBED_BATCH", "6")
    assert config.get_settings() is first


def test_remaining_fields_read_from_environment():
    s = load_settings(
        {
            "EMBED_FALLBACK_URL": "http://fb:9092/embed/batch",
            "EMBED_COLLECTION": "docs",
            "LOCAL_LLM_URL": "http://llm:8080",
            "LOCAL_LLM_MODEL": "gemma",
            "LOCAL_LLM_TIMEOUT": "9.5",
            "ANTHROPIC_MODEL": "claude-x",
        }
    )
    assert s.embed_fallback_url == "http://fb:9092/embed/batch"
    assert s.embed_collection == "docs"
    assert s.local_llm_url == "http://llm:8080"
    assert s.local_llm_model == "gemma"
    assert s.local_llm_timeout == 9.5
    assert s.anthropic_model == "claude-x"


def test_repr_hides_api_keys():
    values = ("alpha-one", "beta-two", "gamma-three")
    s = Settings(**dict(zip(("embed_api_key", "qdrant_api_key", "anthropic_api_key"), values)))
    r = repr(s)
    for v in values:
        assert v not in r
    assert "embed_url" in r


def test_workers_and_webhook_read_from_environment():
    d = load_settings({})
    assert (d.distill_workers, d.sync_workers, d.discord_webhook_url) == (1, 4, "")
    s = load_settings(
        {
            "DISTILL_WORKERS": "2",
            "SYNC_WORKERS": "8",
            "DISCORD_WEBHOOK_URL": "https://dummy.invalid/h",
        }
    )
    assert (s.distill_workers, s.sync_workers) == (2, 8)
    assert s.discord_webhook_url == "https://dummy.invalid/h"


@pytest.mark.parametrize("name", ["DISTILL_WORKERS", "SYNC_WORKERS"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_workers_below_one_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        load_settings({name: value})


def test_repr_hides_webhook_url():
    assert "dummy-hook" not in repr(Settings(discord_webhook_url="https://x.invalid/dummy-hook"))


def test_get_settings_loads_once_under_concurrency(monkeypatch):
    import threading
    import time

    calls = []

    def slow_load():
        calls.append(1)
        time.sleep(0.05)
        return Settings()

    monkeypatch.setattr(config, "load_settings", slow_load)
    config._settings = None
    threads = [threading.Thread(target=config.get_settings) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["EMBED_BATCH", "SYNC_WORKERS", "MAX_LOCAL_BYTES"])
def test_non_integer_error_names_the_variable(name):
    with pytest.raises(ValueError, match=name):
        load_settings({name: "abc"})


def test_non_float_error_names_the_variable():
    with pytest.raises(ValueError, match="EMBED_TIMEOUT"):
        load_settings({"EMBED_TIMEOUT": "abc"})


@pytest.mark.parametrize("name", ["MAX_CHUNKS_PER_DOC", "MAX_LOCAL_BYTES", "MAX_DOWNLOAD_BYTES"])
@pytest.mark.parametrize("value", ["0", "-5"])
def test_size_limits_below_one_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        load_settings({name: value})


def test_doc_ingest_base_dir_maps_from_env():
    assert load_settings({"DOC_INGEST_BASE_DIR": "/x"}).doc_ingest_base_dir == "/x"


def test_doc_ingest_base_dir_defaults_to_empty():
    assert load_settings({}).doc_ingest_base_dir == ""
