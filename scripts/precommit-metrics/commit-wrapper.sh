#!/usr/bin/env bash
# precommit-metrics-wrapper (install-precommit.sh がこの印で導入済みか判定する)
# .git/hooks/pre-commit として置く。本物は pre-commit.real に退避。出力を表示しつつ pclog.sh で記録する。
# 自動修正で失敗した場合(files were modified)は、全体を stage 済みのファイルだけ再 stage して 1 回だけ再実行する(ADR-016 §3)。
# 部分 stage のファイル(未 stage 差分あり)は触らない。git stash は使わない。
# 設定のないブランチ(導入 PR のマージ前など)では何もせず通す(ADR-016: 環境の都合で commit を止めない)。hook の cwd は作業ツリー直下。
[ -f .pre-commit-config.yaml ] || exit 0
hd="$(dirname "$0")"; real="$hd/pre-commit.real"

run_once() {
  local tmp rc; tmp="$(mktemp)"
  "$real" "$@" 2>&1 | tee "$tmp"; rc=${PIPESTATUS[0]}
  bash "$hd/pclog.sh" < "$tmp"
  grep -q 'files were modified by this hook' "$tmp" && MODIFIED=1 || MODIFIED=0
  rm -f "$tmp"; return $rc
}

# 実行前に「全体が stage 済み」のファイルを控える(staged かつ未 stage 差分なし)
safe="$(comm -23 <(git diff --cached --name-only --diff-filter=AM | sort) <(git diff --name-only | sort))"

run_once "$@"; rc=$?
if [ $rc -ne 0 ] && [ "$MODIFIED" = 1 ] && [ -n "$safe" ]; then
  changed="$(comm -12 <(printf '%s\n' "$safe") <(git diff --name-only | sort))"
  if [ -n "$changed" ]; then
    echo "[precommit-wrapper] 自動修正を再 stage して 1 回だけ再実行します:" $changed >&2
    mapfile -t files <<< "$changed"
    git add -- "${files[@]}" || exit $rc
    run_once "$@"; rc=$?
  fi
fi
exit $rc
