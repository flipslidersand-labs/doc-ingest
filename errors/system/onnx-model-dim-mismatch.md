---
title: "ONNX モデルの次元数が Qdrant collection と不一致"
tags: [onnx, embedding, qdrant, e5]
severity: high
date: "2026-09-05"
---

## 症状

Phase 1-3 で `e5-onnx/model_encoder_only.onnx` を使って動作確認したが、
本番 Qdrant collections は 768-dim で作成済み。
モデルの hidden_size を確認すると 384 (e5-small) だった。

## 原因

`export_onnx.sh` の `MODEL_ID="intfloat/multilingual-e5-small"` が研究用に使われていた。
MINIPC embedding-svc は `intfloat/multilingual-e5-base` (768-dim) を使っているため、
GPU backend 用に別途 e5-base をエクスポートする必要があった。

## 解決策

```bash
# config.json の hidden_size で次元数を確認
jq '._name_or_path // .model_type, .hidden_size' e5-onnx/config.json

# e5-base を再エクスポート (--library transformers で encoder-only 直接生成)
optimum-cli export onnx \
  --model intfloat/multilingual-e5-base \
  --task feature-extraction \
  --library transformers \
  ./e5-base-onnx/
```

## 予防

- 新しい ONNX モデルを使う前に必ず `config.json` の `hidden_size` を確認する
- 本番 Qdrant の collection vector_size と照合してから embed テストを行う
- export スクリプトに MODEL_ID とエクスポート先の dim を明記する
