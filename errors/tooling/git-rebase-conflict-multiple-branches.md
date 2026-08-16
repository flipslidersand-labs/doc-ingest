---
title: "複数ブランチのリベースで nested conflict マーカーが発生"
tags: [git, rebase, conflict]
severity: medium
date: "2026-08-17"
---

## 症状

複数の feature ブランチを squash-merge した後、別ブランチをリベースすると
`<<<<<<< HEAD` が二重にネストした形で出現する:

```
<<<<<<< HEAD
<<<<<<< HEAD
def _generate(prompt):
    ...
=======
def _ollama(prompt):
    ...
>>>>>>> 9ddbfc0
=======
def _ollama(prompt):
    ...
>>>>>>> 027bec4
```

## 原因

squash-merge 済みの内容がリベース時に再度コンフリクト対象になる。
cherry-pick の重複適用が原因 (git の `skipped previously applied commit` 警告が出る)。

## 解決策

コンフリクトファイルを「正しい最終状態」で上書きして解決する。
`git mergetool` や手動編集よりも Write ツールで直接正しい内容を書き込む方が速い:

```bash
# 正しい内容を Write した後
git add <file>
GIT_EDITOR=true git rebase --continue
git push --force-with-lease origin <branch>
```

## 予防

- 多数のブランチを順次マージする場合はマージ後に随時 `git fetch origin main && git rebase origin/main` を実行
- squash-merge 済みブランチは即削除して混入を防ぐ
