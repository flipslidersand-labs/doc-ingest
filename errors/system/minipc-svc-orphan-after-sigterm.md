---
title: "MINIPC embedding-svc が systemd と断絶したままオーファン生存"
tags: [systemd, minipc, embedding-svc, orphan]
severity: medium
date: "2026-09-05"
---

## 症状

`systemctl status embedding-svc` が `inactive (dead)` を示していたにもかかわらず、
`curl http://localhost:9092/health` が 200 を返し、プロセスが 2.3GB RSS で生存していた。

## 原因

SIGTERM で正常終了した際に systemd の管理下から外れたが、
プロセス本体（pid 1975424）は 8/22 から生き続けていた。
systemd の `Restart=on-failure` は SIGTERM（正常終了）では発動しないため、
再起動されずオーファン状態になった。

## 解決策

```bash
# ポート占有プロセスを確認
ss -tlnp sport = :9092

# オーファンプロセスを停止してから systemd 経由で再起動
sudo kill $(lsof -t -i:9092)
sleep 2
sudo systemctl restart embedding-svc
```

## 予防

- `systemctl status` だけでなく `/health` エンドポイントでサービスの実態を確認する
- uptime-kuma 等のモニタリングで HTTP レベルの死活監視を行う（現在 MINIPC:3002 で稼働中）
- 意図的に停止する場合は `systemctl stop` ではなく pid を明示的に kill してから systemd 経由で再起動する
