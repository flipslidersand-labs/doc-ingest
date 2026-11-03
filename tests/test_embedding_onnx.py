"""EMBED_BACKEND=onnx_gpu 経路 (#156)。GPU・onnxruntime・transformers なしで動く。

実機の onnxruntime/モデルは使わず、トークナイザとセッションを fake に差し替えて
前処理 (プレフィックス)・プーリング・入力フィルタ・初期化・ディスパッチの契約を固定する。
"""

import logging
import sys
import types

import numpy as np
import pytest

import core.embedding as emb

HIDDEN = 4


class FakeTok:
    """1 単語 = 1 トークン。短い文は 0 パディング(attention_mask=0)。"""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding and truncation and max_length == 512 and return_tensors == "np"
        self.seen = list(texts)
        lens = [len(t.split()) for t in texts]
        width = max(lens)
        mask = np.array([[1] * n + [0] * (width - n) for n in lens])
        return {
            "input_ids": mask * 7,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(mask),
        }


class FakeSess:
    """隠れ状態 = (文 b, 位置 t, 次元 d) ごとに決定的な値。パディング位置は巨大値にする。"""

    def __init__(self, input_names=("input_ids", "attention_mask"), providers=None):
        self._names = input_names
        self._providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.feed = None

    def get_inputs(self):
        return [types.SimpleNamespace(name=n) for n in self._names]

    def get_providers(self):
        return self._providers

    def run(self, _outputs, feed):
        self.feed = feed
        mask = feed["attention_mask"]
        b, t = mask.shape
        h = np.zeros((b, t, HIDDEN), dtype=np.float32)
        for i in range(b):
            for j in range(t):
                h[i, j] = (i + 1) * (j + 1) * np.arange(1, HIDDEN + 1) if mask[i, j] else 1e6
        return [h]


@pytest.fixture
def onnx(monkeypatch):
    tok, sess = FakeTok(), FakeSess()
    monkeypatch.setattr(emb, "_onnx_sess", sess)
    monkeypatch.setattr(emb, "_onnx_tok", tok)
    monkeypatch.setattr(emb, "_onnx_input_names", frozenset(sess._names))
    return tok, sess


def test_every_text_gets_passage_prefix_like_the_http_service(onnx):
    tok, _ = onnx
    emb._embed_batch_onnx(["alpha beta", "gamma"])
    assert tok.seen == ["passage: alpha beta", "passage: gamma"]


def test_search_mode_uses_the_query_prefix_and_index_mode_the_passage_prefix(onnx):
    """#222: クエリは "query: "、文書は "passage: "。mode 省略は文書 (従来どおり)。"""
    tok, _ = onnx
    emb._embed_batch_onnx(["alpha"], "search")
    assert tok.seen == ["query: alpha"]
    emb._embed_batch_onnx(["alpha"], "index")
    assert tok.seen == ["passage: alpha"]
    emb._embed_batch_onnx(["alpha"])
    assert tok.seen == ["passage: alpha"]


def test_mean_pool_ignores_padding_and_is_l2_normalized(onnx):
    # トークン数は "passage:" を含めて 3 と 2。短い方の後ろは 1e6 のパディング。
    out = np.array(emb._embed_batch_onnx(["alpha beta", "gamma"]))
    assert out.shape == (2, HIDDEN)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    d = np.arange(1, HIDDEN + 1)
    for i, n_tok in enumerate([3, 2]):
        mean = np.mean([(i + 1) * (j + 1) * d for j in range(n_tok)], axis=0)
        assert np.allclose(out[i], mean / np.linalg.norm(mean), atol=1e-6)


def test_feed_only_contains_model_inputs_as_int64(onnx):
    _, sess = onnx
    emb._embed_batch_onnx(["alpha"])
    assert set(sess.feed) == {"input_ids", "attention_mask"}  # token_type_ids は除外
    assert all(v.dtype == np.int64 for v in sess.feed.values())


def test_returns_plain_python_lists(onnx):
    out = emb._embed_batch_onnx(["alpha"])
    assert isinstance(out, list) and isinstance(out[0], list) and isinstance(out[0][0], float)


def test_embed_dispatches_to_onnx_in_batches(monkeypatch):
    sizes = []

    def fake_onnx(batch, mode):
        sizes.append(len(batch))
        return [[0.0] * HIDDEN for _ in batch]

    monkeypatch.setattr(emb, "EMBED_BACKEND", "onnx_gpu")
    monkeypatch.setattr(emb, "EMBED_BATCH", 2)
    monkeypatch.setattr(emb, "_embed_batch_onnx", fake_onnx)
    monkeypatch.setattr(emb, "_embed_batch", lambda b, m: pytest.fail("HTTP path must not be used"))
    assert len(emb.embed(["a", "b", "c", "d", "e"])) == 5
    assert sizes == [2, 2, 1]


def _install_fake_runtime(monkeypatch, providers):
    sess = FakeSess(providers=providers)
    created = {}

    def inference_session(path, providers):
        created["path"], created["providers"] = path, providers
        return sess

    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = inference_session
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda d: FakeTok())
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    for name, val in (
        ("_onnx_sess", None),
        ("_onnx_tok", None),
        ("_onnx_input_names", frozenset()),
    ):
        monkeypatch.setattr(emb, name, val)
    return created


def test_init_requires_model_path(monkeypatch):
    monkeypatch.setattr(emb, "ONNX_MODEL_PATH", "")
    _install_fake_runtime(monkeypatch, ["CUDAExecutionProvider"])
    with pytest.raises(RuntimeError, match="ONNX_MODEL_PATH"):
        emb._onnx_init()


def test_init_requests_cuda_then_cpu_and_stays_quiet_when_cuda_is_active(monkeypatch, caplog):
    monkeypatch.setattr(emb, "ONNX_MODEL_PATH", "/m/model.onnx")
    created = _install_fake_runtime(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    with caplog.at_level(logging.INFO, logger="core.embedding"):
        emb._onnx_init()
    assert created["path"] == "/m/model.onnx"
    assert created["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert emb._onnx_input_names == frozenset({"input_ids", "attention_mask"})
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_init_warns_loudly_when_it_silently_fell_back_to_cpu(monkeypatch, caplog):
    """CUDA ライブラリ不足だと onnxruntime は黙って CPU に落ちる。気づけること。"""
    monkeypatch.setattr(emb, "ONNX_MODEL_PATH", "/m/model.onnx")
    _install_fake_runtime(monkeypatch, ["CPUExecutionProvider"])
    with caplog.at_level(logging.INFO, logger="core.embedding"):
        emb._onnx_init()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "NOT active" in warnings[0].getMessage()


def test_resources_initialize_once(monkeypatch):
    calls = []

    def fake_init():
        calls.append(1)
        monkeypatch.setattr(emb, "_onnx_sess", "S")
        monkeypatch.setattr(emb, "_onnx_tok", "T")
        monkeypatch.setattr(emb, "_onnx_input_names", frozenset({"x"}))

    monkeypatch.setattr(emb, "_onnx_sess", None)
    monkeypatch.setattr(emb, "_onnx_init", fake_init)
    assert emb._onnx_resources() == ("S", "T", frozenset({"x"}))
    assert emb._onnx_resources() == ("S", "T", frozenset({"x"}))
    assert len(calls) == 1
