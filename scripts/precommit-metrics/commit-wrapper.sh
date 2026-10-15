#!/usr/bin/env bash
# precommit-metrics-wrapper (install-precommit.sh がこの印で導入済みか判定する)
# .git/hooks/pre-commit として置く。本物は pre-commit.real に退避。出力を表示しつつ pclog.sh で記録する。
real="$(dirname "$0")/pre-commit.real"; tmp="$(mktemp)"
"$real" "$@" 2>&1 | tee "$tmp"; rc=${PIPESTATUS[0]}
bash "$(dirname "$0")/pclog.sh" < "$tmp"; rm -f "$tmp"; exit $rc
