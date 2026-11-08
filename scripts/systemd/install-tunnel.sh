#!/usr/bin/env bash
# 逆トンネルの恒久 user unit を配備する(冪等)。transient 版が残っていれば止めて置き換える。
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest="$HOME/.config/systemd/user"
mkdir -p "$dest"
install -m 0644 "$here/minipc-llm-reverse-tunnel.service" "$dest/"
systemctl --user stop minipc-llm-reverse-tunnel.service 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now minipc-llm-reverse-tunnel.service
loginctl show-user "$USER" -p Linger | grep -q 'Linger=yes' || echo "WARN: linger 無効。loginctl enable-linger $USER が必要" >&2
systemctl --user is-active minipc-llm-reverse-tunnel.service
