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
            "QDRANT_URL": "http://q:6333",
            "LOCAL_LLM_MAX_TOKENS": "99",
            "ANTHROPIC_API_KEY": "k",
        }
    )
    assert (s.embed_url, s.embed_timeout, s.embed_batch) == ("http://e:1/x", 7.5, 32)
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


def test_removed_embed_backend_warns_once_per_load(caplog):
    """#320: 撤去済みの EMBED_BACKEND が設定されていたら警告する。読み込みは失敗させない。"""
    with caplog.at_level("WARNING", logger="core.config"):
        s = load_settings({"EMBED_BACKEND": "onnx_gpu"})
    assert s == Settings()
    msgs = [r.getMessage() for r in caplog.records if "EMBED_BACKEND" in r.getMessage()]
    assert len(msgs) == 1
    assert "HTTP" in msgs[0]


def test_no_warning_without_embed_backend(caplog):
    with caplog.at_level("WARNING", logger="core.config"):
        load_settings({})
    assert not [r for r in caplog.records if "EMBED_BACKEND" in r.getMessage()]


# --- #362 safety net: env var -> attribute -> default -> type, for every Settings field ---
# Written from the pre-refactor load_settings(); literals are deliberately NOT derived from
# Settings so that a change of any env name, default or type fails here.
_MB = 1024 * 1024
# (env name, attribute, default, type, sample env value, parsed sample)
_MAPPING = [
    ("EMBED_URL", "embed_url", "http://localhost:9092/embed/batch", str, "http://e:1/x", "http://e:1/x"),
    ("EMBED_FALLBACK_URL", "embed_fallback_url", "", str, "http://fb:1/y", "http://fb:1/y"),
    ("EMBED_API_KEY", "embed_api_key", None, type(None), "ek", "ek"),
    ("EMBED_COLLECTION", "embed_collection", "sessions", str, "docs", "docs"),
    ("EMBED_TIMEOUT", "embed_timeout", 180.0, float, "7.5", 7.5),
    ("EMBED_BATCH", "embed_batch", 16, int, "32", 32),
    ("QDRANT_URL", "qdrant_url", "http://localhost:6333", str, "http://q:1", "http://q:1"),
    ("QDRANT_API_KEY", "qdrant_api_key", None, type(None), "qk", "qk"),
    ("LOCAL_LLM_URL", "local_llm_url", "http://localhost:8080", str, "http://l:1", "http://l:1"),
    ("LOCAL_LLM_MODEL", "local_llm_model", "", str, "gemma", "gemma"),
    ("LOCAL_LLM_TIMEOUT", "local_llm_timeout", 120.0, float, "9.5", 9.5),
    ("LOCAL_LLM_MAX_TOKENS", "local_llm_max_tokens", 1500, int, "99", 99),
    ("ANTHROPIC_API_KEY", "anthropic_api_key", "", str, "ak", "ak"),
    ("ANTHROPIC_MODEL", "anthropic_model", "claude-haiku-4-5-20251001", str, "claude-x", "claude-x"),
    ("DISTILL_WORKERS", "distill_workers", 1, int, "2", 2),
    ("SYNC_WORKERS", "sync_workers", 4, int, "8", 8),
    ("DISCORD_WEBHOOK_URL", "discord_webhook_url", "", str, "https://d.invalid/h", "https://d.invalid/h"),
    ("MAX_CHUNKS_PER_DOC", "max_chunks_per_doc", 1000, int, "5", 5),
    ("MAX_LOCAL_BYTES", "max_local_bytes", 10 * _MB, int, "6", 6),
    ("MAX_DOWNLOAD_BYTES", "max_download_bytes", 50 * _MB, int, "7", 7),
    ("DOC_INGEST_BASE_DIR", "doc_ingest_base_dir", "", str, "/x", "/x"),
]  # fmt: skip


def test_mapping_table_covers_every_settings_field():
    import dataclasses

    assert len(_MAPPING) == 21
    assert [m[1] for m in _MAPPING] == [f.name for f in dataclasses.fields(Settings)]


@pytest.mark.parametrize("env_name,attr,default,typ,_raw,_parsed", _MAPPING)
def test_every_default_value_and_type(env_name, attr, default, typ, _raw, _parsed):
    for s in (load_settings({}), Settings()):
        value = getattr(s, attr)
        assert value == default
        assert type(value) is typ


def test_all_fields_set_via_environment_at_once():
    env = {m[0]: m[4] for m in _MAPPING}
    s = load_settings(env)
    for env_name, attr, _default, _typ, _raw, parsed in _MAPPING:
        value = getattr(s, attr)
        assert value == parsed, env_name
        assert type(value) is type(parsed), env_name


@pytest.mark.parametrize("env_name,attr,_d,_t,raw,parsed", _MAPPING)
def test_each_field_set_alone_changes_only_that_field(env_name, attr, _d, _t, raw, parsed):
    import dataclasses

    s = load_settings({env_name: raw})
    base = load_settings({})
    changed = [
        f.name for f in dataclasses.fields(Settings) if getattr(s, f.name) != getattr(base, f.name)
    ]
    assert changed == [attr]
    assert getattr(s, attr) == parsed


def test_unrelated_and_lowercase_env_names_are_ignored():
    assert load_settings({"embed_url": "http://x", "OTHER": "1", "PATH": "/bin"}) == Settings()


@pytest.mark.parametrize("env_name", ["EMBED_API_KEY", "QDRANT_API_KEY"])
def test_optional_str_fields_map_empty_to_none(env_name):
    attr = env_name.lower()
    assert getattr(load_settings({env_name: ""}), attr) is None
    assert getattr(load_settings({}), attr) is None


@pytest.mark.parametrize("env_name", ["EMBED_URL", "EMBED_FALLBACK_URL", "LOCAL_LLM_MODEL"])
def test_plain_str_fields_keep_empty_string(env_name):
    assert getattr(load_settings({env_name: ""}), env_name.lower()) == ""


_INT_ENVS = [m[0] for m in _MAPPING if m[3] is int]
_FLOAT_ENVS = [m[0] for m in _MAPPING if m[3] is float]


@pytest.mark.parametrize("env_name", _INT_ENVS)
@pytest.mark.parametrize("bad", ["abc", "1.5", ""])
def test_invalid_int_message_is_exact(env_name, bad):
    with pytest.raises(ValueError) as ei:
        load_settings({env_name: bad})
    assert str(ei.value) == f"{env_name} must be an integer, got {bad!r}"


@pytest.mark.parametrize("env_name", _FLOAT_ENVS)
@pytest.mark.parametrize("bad", ["abc", ""])
def test_invalid_float_message_is_exact(env_name, bad):
    with pytest.raises(ValueError) as ei:
        load_settings({env_name: bad})
    assert str(ei.value) == f"{env_name} must be a number, got {bad!r}"


@pytest.mark.parametrize(
    "env_name",
    [
        "EMBED_BATCH",
        "DISTILL_WORKERS",
        "SYNC_WORKERS",
        "MAX_CHUNKS_PER_DOC",
        "MAX_LOCAL_BYTES",
        "MAX_DOWNLOAD_BYTES",
    ],
)
def test_range_error_message_is_exact(env_name):
    with pytest.raises(ValueError) as ei:
        load_settings({env_name: "0"})
    assert str(ei.value) == f"{env_name} must be >= 1, got 0"


def test_float_and_int_accept_the_same_spellings_as_before():
    s = load_settings({"EMBED_TIMEOUT": "180", "LOCAL_LLM_TIMEOUT": "1e2", "EMBED_BATCH": " 3 "})
    assert (s.embed_timeout, s.local_llm_timeout, s.embed_batch) == (180.0, 100.0, 3)
    assert type(s.embed_timeout) is float
