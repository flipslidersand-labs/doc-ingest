#!/usr/bin/env bash
# managed-hooks の pre-commit 以外のフック (pre-push など) を <git-common-dir>/hooks/<同名> へ転送する。
# core.hooksPath を設定すると .git/hooks のフックが無言で止まる (secret-sweep の pre-push など) のを防ぐ (#189)。
# このファイルは _forward として置かれ、各フック名のシンボリックリンクから呼ばれる。stdin と引数はそのまま渡す。
name="$(basename "$0")"
target="$(git rev-parse --path-format=absolute --git-common-dir)/hooks/$name"
[ -x "$target" ] && exec "$target" "$@"
exit 0
