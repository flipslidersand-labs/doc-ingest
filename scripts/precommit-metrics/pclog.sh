#!/usr/bin/env bash
# pre-commit の出力(stdin)を読み、フックごとに JSONL を追記する。公式フックのまま使える。
# result: pass / fixed(Failed + files were modified) / fail(Failed + exit code) / skipped
# 記録先: $PRECOMMIT_METRICS_LOG(既定: ~/.local/state/precommit-metrics/events.jsonl)。記録に失敗してもコミットは止めない。
log="${PRECOMMIT_METRICS_LOG:-$HOME/.local/state/precommit-metrics/events.jsonl}"
repo="$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)")")"; host="$(hostname)"
mkdir -p "$(dirname "$log")" 2>/dev/null
sed -E 's/\x1b\[[0-9;]*m//g' | awk -v repo="$repo" -v host="$host" '
function flush() { if (name != "") { printf "{\"ts\":\"%s\",\"repo\":\"%s\",\"host\":\"%s\",\"source\":\"local\",\"hook\":\"%s\",\"result\":\"%s\",\"rc\":%d}\n", ts, repo, host, name, res, rc; name="" } }
BEGIN { "date -u +%Y-%m-%dT%H:%M:%SZ" | getline ts }
/^[^ -].*\.\.\.\.+.*(Passed|Failed|Skipped)$/ {
  flush(); line=$0; sub(/\.\.\.\.+.*/, "", line); name=line; rc=0
  if ($0 ~ /Passed$/) res="pass"; else if ($0 ~ /Skipped$/) res="skipped"; else res="fail"
  next }
/^- exit code: / { rc=$4; next }
/^- files were modified by this hook/ { if (res=="fail") res="fixed"; next }
END { flush() }' >> "$log" 2>/dev/null || true
