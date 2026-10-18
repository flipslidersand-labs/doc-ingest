#!/usr/bin/env bash
# pre-commit を導入し、実行結果の記録(precommit-metrics)を有効にする。何度実行しても同じ状態になる。
# 使い方: bash scripts/install-precommit.sh   (クローンごとに 1 回。pre-commit install の代わり)
set -euo pipefail
src="$(cd "$(dirname "$0")/precommit-metrics" && pwd)"
hooks="$(git rev-parse --git-path hooks)"   # worktree でも共通の hooks ディレクトリを指す
hook="$hooks/pre-commit"

if [ -f "$hook" ] && grep -q 'precommit-metrics-wrapper' "$hook"; then
  echo "導入済み: ラッパーを最新に更新します"
else
  # 注意: ラッパーが入った状態で pre-commit install を再実行すると、ラッパーが pre-commit.legacy に退避され二重実行になる。
  pre-commit install
  mv "$hook" "$hooks/pre-commit.real"
fi
cp "$src/commit-wrapper.sh" "$hook"
cp "$src/pclog.sh" "$hooks/pclog.sh"
chmod +x "$hook" "$hooks/pclog.sh"
echo "OK: $hook (記録先: ${PRECOMMIT_METRICS_LOG:-~/.local/state/precommit-metrics/events.jsonl})"
