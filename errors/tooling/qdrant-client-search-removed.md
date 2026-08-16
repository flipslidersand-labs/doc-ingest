---
title: "qdrant-client 1.19 で QdrantClient.search() が廃止され AttributeError"
tags: [qdrant, qdrant-client, python, breaking-change]
severity: high
date: "2026-08-17"
---

## 症状

検索実行時に落ちる。ingest/upsert は正常なため検索経路だけが本番で全滅する。

```
AttributeError: 'QdrantClient' object has no attribute 'search'
```

## 原因

- 依存ピンが `qdrant-client>=1.9`（上限なし）で最新の 1.19 が入る
- qdrant-client は `QdrantClient.search()` を廃止し `query_points()` に統合
- 旧 API `client.search(collection_name=..., query_vector=..., limit=...)` が消えた
- ユニットテストが client をモックしていると API 削除を検知できない（本番で初めて顕在化）

## 解決策

`query_points()` へ移行する。戻り値は `QueryResponse`、ヒットは `.points`。

```python
# before
hits = client.search(collection_name=col, query_vector=vec, limit=limit)
# after
hits = client.query_points(collection_name=col, query=vec, limit=limit).points
# hit.score / hit.id / hit.payload は同じ。payload は None ガード推奨
```

ピンも `qdrant-client>=1.10`（query_points 導入版）以上に上げる。

## 予防

- モックベースのテストは API 削除を捕捉できない。ライブラリ跨ぎの最小 smoke test を持つ
- ingest だけでなく **search 経路も** デプロイ後に 1 度実行して確認する
