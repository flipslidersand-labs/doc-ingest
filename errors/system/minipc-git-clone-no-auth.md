---
title: "MINIPC での git clone が SSH 未設定で失敗"
tags: [git, ssh, minipc]
severity: low
date: "2026-08-17"
---

## 症状

MINIPC で `git clone https://github.com/...` を実行すると
"fatal: could not read Username" エラー。
手動コピーで配置されたファイルは `.git` がなくリポジトリでない。

## 原因

MINIPC に GitHub SSH キーが設定されていない。
HTTPS クローンはインタラクティブ認証が必要だが cron/SSH セッションでは使えない。

## 解決策

`scripts/deploy-minipc.sh` でローカルから転送する (2026-11-02 更新: #195 で方式が変わった)。

```bash
DEPLOY_DRY_RUN=1 bash scripts/deploy-minipc.sh   # 先に差分だけ確認 (送信も install もしない)
bash scripts/deploy-minipc.sh                    # 本番へ
```

- 送るのは `git archive origin/main` (コミット済みの内容だけ)。未コミット変更・`.claude/worktrees`・coverage は混ざらない
- **`.git` は送らない**。MINIPC 側の `.git`・`logs/`・`.worktrees/`・`.venv`・`.env` は除外して保護する
  (旧版の手順は `.git` ごと rsync していたが、ローカルの作業ツリーを丸ごと本番に混入させていた)
- 転送後に `pip install -e .` まで行う

## 予防

- MINIPC に `~/.ssh/id_ed25519` を生成して GitHub に登録する（恒久対策。登録すれば MINIPC で `git pull` できる）
- デプロイは上のスクリプトに一本化する（手で rsync しない）
