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

`rsync` でローカルから転送する（`.git` も含めると git リポジトリとして機能する）:

```bash
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' --exclude='.env' \
  /path/to/local/project/ \
  minipc:~/project/
ssh minipc "cd ~/project && pip install -e . -q"
```

## 予防

- MINIPC に `~/.ssh/id_ed25519` を生成して GitHub に登録する（恒久対策）
- または rsync デプロイをスクリプト化して `make deploy` コマンドにする
