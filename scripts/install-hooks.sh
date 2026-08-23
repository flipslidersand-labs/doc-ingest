#!/usr/bin/env bash
# Install doc-ingest post-commit hook into one or more git repos.
#
# Usage:
#   bash scripts/install-hooks.sh                        # use DEFAULT_REPOS below
#   bash scripts/install-hooks.sh /path/to/repo ...     # explicit repos
#   bash scripts/install-hooks.sh --dry-run             # preview without writing
#   bash scripts/install-hooks.sh --uninstall           # remove from all repos

set -euo pipefail

DOC_INGEST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MARKER="# doc-ingest-hook"

# Repos to install into when no arguments are given
DEFAULT_REPOS=(
  "$HOME/projects/flipslidersand/mesh-drop"
  "$HOME/projects/forge"
  "$HOME/projects/flipslidersand/fluxion"
)

# The snippet appended to (or placed in) the hook
read -r -d '' SNIPPET << 'SNIPPET' || true
# doc-ingest-hook
_DOC_INGEST_DIR="${DOC_INGEST_DIR:-$HOME/projects/doc-ingest}"
if [ -f "$_DOC_INGEST_DIR/.env" ]; then
  set -a && source "$_DOC_INGEST_DIR/.env" && set +a
fi
PYTHONPATH="$_DOC_INGEST_DIR" "$_DOC_INGEST_DIR/.venv/bin/doc-ingest" design 2>/dev/null || true
SNIPPET

# --- parse flags ---
DRY_RUN=0
UNINSTALL=0
REPOS=()

for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=1 ;;
    --uninstall) UNINSTALL=1 ;;
    *)           REPOS+=("$arg") ;;
  esac
done

if [ ${#REPOS[@]} -eq 0 ]; then
  REPOS=("${DEFAULT_REPOS[@]}")
fi

# --- main loop ---
for repo in "${REPOS[@]}"; do
  hook="$repo/.git/hooks/post-commit"

  if [ ! -d "$repo/.git" ]; then
    echo "skip  $repo  (not a git repo)"
    continue
  fi

  if [ "$UNINSTALL" -eq 1 ]; then
    if [ ! -f "$hook" ] || ! grep -qF "$MARKER" "$hook" 2>/dev/null; then
      echo "skip  $repo  (not installed)"
      continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "dry   $repo  (would remove hook snippet)"
    else
      # remove the snippet block: from MARKER line to end of SNIPPET
      # use python for reliable multi-line removal
      python3 - "$hook" "$MARKER" <<'PY'
import sys, re
path, marker = sys.argv[1], sys.argv[2]
text = open(path).read()
# remove from marker to next blank line + trailing newline
text = re.sub(r'\n?' + re.escape(marker) + r'.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
open(path, 'w').write(text.strip() + '\n' if text.strip() else '')
PY
      echo "removed $hook"
    fi
    continue
  fi

  if grep -qF "$MARKER" "$hook" 2>/dev/null; then
    echo "skip  $repo  (already installed)"
    continue
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ -f "$hook" ]; then
      echo "dry   $repo  (would append to existing hook)"
    else
      echo "dry   $repo  (would create new hook)"
    fi
    continue
  fi

  if [ -f "$hook" ]; then
    printf '\n%s\n' "$SNIPPET" >> "$hook"
    echo "appended $hook"
  else
    printf '#!/usr/bin/env bash\nset -e\n\n%s\n' "$SNIPPET" > "$hook"
    chmod +x "$hook"
    echo "created $hook"
  fi
done
