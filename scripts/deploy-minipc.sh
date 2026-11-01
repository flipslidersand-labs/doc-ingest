#!/usr/bin/env bash
set -euo pipefail

REMOTE="minipc"
REMOTE_DIR="~/doc-ingest"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REF="${DEPLOY_REF:-origin/main}"

# 未コミット変更・.git・worktree・coverage 等を送らないため、コミット済みの REF だけを展開して送る (#195)
git -C "${LOCAL_DIR}" fetch -q origin
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
git -C "${LOCAL_DIR}" archive "${REF}" | tar -x -C "${STAGE}"

echo "[deploy] syncing ${REF} ($(git -C "${LOCAL_DIR}" rev-parse --short "${REF}")) → ${REMOTE}:${REMOTE_DIR}"
# --delete で消えて困る MINIPC 側の実体 (.git / logs / .worktrees / venv / 設定 / キャッシュ) は除外して保護する
rsync -av --delete \
  --exclude='.git' \
  --exclude='.worktrees' \
  --exclude='logs' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='.env' \
  --exclude='.ruff_cache' \
  --exclude='.pytest_cache' \
  ${DEPLOY_DRY_RUN:+--dry-run} \
  "${STAGE}/" "${REMOTE}:${REMOTE_DIR}/"

if [[ -n "${DEPLOY_DRY_RUN:-}" ]]; then
  echo "[deploy] dry-run のため install/sync は行わない"
  exit 0
fi

echo "[deploy] installing package"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && .venv/bin/pip install -e . -q"

if [[ "${1:-}" == "--sync" ]]; then
  echo "[deploy] running sync --force"
  ssh "${REMOTE}" "cd ${REMOTE_DIR} && .venv/bin/doc-ingest sync --force"
fi

echo "[deploy] done"
