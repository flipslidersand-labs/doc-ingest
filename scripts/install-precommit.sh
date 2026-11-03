#!/usr/bin/env bash
# pre-commit を導入し、実行結果の記録(precommit-metrics)を有効にする。何度実行しても同じ状態になる。
# 使い方: bash scripts/install-precommit.sh   (クローンごとに 1 回。pre-commit install の代わり)
#
# 方式(#189): <git-common-dir>/managed-hooks に置き、core.hooksPath で指す。
#  - pre-commit install は hooksPath があると拒否されるので、再実行してもラッパーは上書きされない
#  - 場所は .git 配下の絶対パス: 全 worktree・全ブランチで同じ。旧ブランチでもフックが消えない
#  - pre-commit 以外のフック(pre-push の secret-sweep など)は .git/hooks の同名ファイルへ転送する
# 更新するには、このスクリプトを再実行する(最新を copy する)。
set -euo pipefail
src="$(cd "$(dirname "$0")/precommit-metrics" && pwd)"
command -v pre-commit >/dev/null 2>&1 || { echo "pre-commit が見つかりません (pip install pre-commit)" >&2; exit 1; }
common="$(git rev-parse --path-format=absolute --git-common-dir)"
legacy="$common/hooks"
dir="$common/managed-hooks"

# 旧方式(.git/hooks/pre-commit をラッパー化)の導入物だけを片づける。pre-commit 自身が作ったものは触らない。
if [ -f "$legacy/pre-commit" ] && grep -q 'precommit-metrics-wrapper' "$legacy/pre-commit"; then
  rm -f "$legacy/pre-commit" "$legacy/pre-commit.real" "$legacy/pclog.sh"
  echo "旧方式のラッパーを削除しました"
fi

mkdir -p "$dir"
cp "$src/commit-wrapper.sh" "$dir/pre-commit"
cp "$src/pclog.sh" "$dir/pclog.sh"
cp "$src/forward.sh" "$dir/_forward"
chmod +x "$dir/pre-commit" "$dir/pclog.sh" "$dir/_forward"
# git が呼ぶクライアントフック(pre-commit 以外)。.git/hooks に同名があれば転送される
for h in applypatch-msg pre-applypatch post-applypatch prepare-commit-msg commit-msg post-commit \
         pre-rebase post-checkout post-merge pre-push post-rewrite pre-auto-gc; do
  ln -sf _forward "$dir/$h"
done
git config core.hooksPath "$dir"
echo "OK: core.hooksPath=$dir (記録先: ${PRECOMMIT_METRICS_LOG:-~/.local/state/precommit-metrics/events.jsonl})"
