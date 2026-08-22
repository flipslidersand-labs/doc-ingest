#!/usr/bin/env bash
set -euo pipefail

REMOTE="minipc"
REMOTE_DIR="~/doc-ingest"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[deploy] syncing ${LOCAL_DIR} → ${REMOTE}:${REMOTE_DIR}"
rsync -av \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='.env' \
  "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

echo "[deploy] installing package"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && .venv/bin/pip install -e . -q"

if [[ "${1:-}" == "--sync" ]]; then
  echo "[deploy] running sync --force"
  ssh "${REMOTE}" "cd ${REMOTE_DIR} && .venv/bin/doc-ingest sync --force"
fi

echo "[deploy] done"
