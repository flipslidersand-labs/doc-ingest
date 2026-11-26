#!/usr/bin/env bash
# Install doc-ingest post-commit hook into one or more git repos.
#
# Usage:
#   bash scripts/install-hooks.sh                        # use DEFAULT_REPOS below
#   bash scripts/install-hooks.sh /path/to/repo ...     # explicit repos
#   bash scripts/install-hooks.sh --dry-run             # preview without writing
#   bash scripts/install-hooks.sh --uninstall           # remove from all repos
#   bash scripts/install-hooks.sh --notes [/path/to/vault]   # Obsidian vault: run `notes` (default ~/notes)

set -euo pipefail

DOC_INGEST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MARKER="# doc-ingest-hook"

# Repos to install into when no arguments are given.
# NOTE: mesh-drop and fluxion moved from the flipslidersand org to
# flipslidersand-labs on 2026-09-15. If repos move org/path again, update
# this list (or pass explicit paths as arguments instead).
DEFAULT_REPOS=(
  "$HOME/projects/flipslidersand-labs/mesh-drop"
  "$HOME/projects/forge"
  "$HOME/projects/flipslidersand-labs/fluxion"
)

# The snippet appended to (or placed in) the hook
# Loads .env without shell evaluation (fixes #80/#129: arbitrary command
# execution via source). Only KEY=VALUE lines are accepted; comments and
# blank lines are skipped. Values are read as literal strings — no $(…),
# backtick, or variable expansion occurs.
#
# #356: __DOC_INGEST_ROOT__ is replaced at install time with this checkout's root (shell-quoted),
# so the hook works wherever the repo was cloned. DOC_INGEST_DIR in the hook's env still wins.
# A failing/unlaunchable doc-ingest never fails the commit (exit 0), but is appended to
# ${XDG_STATE_HOME:-$HOME/.local/state}/doc-ingest/post-commit.log (private perms; timestamp,
# repo, exit code, last 5 stderr lines only - never .env values). No blank lines inside the
# snippet: --uninstall removes from the marker up to the first blank line.
read -r -d '' SNIPPET << 'SNIPPET' || true
# doc-ingest-hook
_DOC_INGEST_DEFAULT=__DOC_INGEST_ROOT__
_DOC_INGEST_DIR="${DOC_INGEST_DIR:-$_DOC_INGEST_DEFAULT}"
if [ -f "$_DOC_INGEST_DIR/.env" ]; then
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in
      ''|'#'*) continue ;;
    esac
    if printf '%s\n' "$_line" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*='; then
      _k="${_line%%=*}"
      _v="${_line#*=}"
      # strip one pair of matching surrounding quotes (python-dotenv does not override
      # existing env vars, so a leftover quote would win). Never eval the value.
      case "$_v" in
        \"?*\") _v="${_v#\"}"; _v="${_v%\"}" ;;
        \'?*\') _v="${_v#\'}"; _v="${_v%\'}" ;;
        \"\"|\'\') _v="" ;;
      esac
      export "$_k=$_v"
    fi
  done < "$_DOC_INGEST_DIR/.env"
fi
_di_err="$(mktemp "${TMPDIR:-/tmp}/doc-ingest-hook.XXXXXX" 2>/dev/null)" || _di_err=/dev/null
_di_rc=0
PYTHONPATH="$_DOC_INGEST_DIR" "$_DOC_INGEST_DIR/.venv/bin/doc-ingest" design 2>"$_di_err" || _di_rc=$?
if [ "$_di_rc" -ne 0 ]; then
  (
    umask 077
    _di_log="${XDG_STATE_HOME:-$HOME/.local/state}/doc-ingest/post-commit.log"
    mkdir -p "${_di_log%/*}" || exit 0
    {
      printf '%s post-commit repo=%s exit=%s cmd=%s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$PWD" "$_di_rc" "$_DOC_INGEST_DIR/.venv/bin/doc-ingest"
      tail -n 5 "$_di_err" | sed 's/^/  | /'
    } >> "$_di_log"
    if [ "$(wc -c < "$_di_log")" -gt 65536 ]; then
      tail -n 200 "$_di_log" > "$_di_log.tmp" && mv "$_di_log.tmp" "$_di_log"
    fi
  ) 2>/dev/null || true
fi
[ "$_di_err" = /dev/null ] || rm -f "$_di_err"
SNIPPET

# --- parse flags ---
DRY_RUN=0
UNINSTALL=0
NOTES=0
REPOS=()

for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --notes)     NOTES=1 ;;
    *)           REPOS+=("$arg") ;;
  esac
done

if [ "$NOTES" -eq 1 ]; then
  # Obsidian vault: same safe .env loading as the design hook, but runs `notes`.
  # A separate marker lets both hooks live in one post-commit file.
  MARKER="# doc-ingest-notes-hook"
  SNIPPET="${SNIPPET/doc-ingest-hook/doc-ingest-notes-hook}"
  SNIPPET="${SNIPPET/doc-ingest\" design /doc-ingest\" notes }"
  DEFAULT_REPOS=("$HOME/notes")
fi

# Embed this checkout's root (the repo that holds this script) as the hook's default
# location. Split on the placeholder rather than ${var/pat/rep}: bash 5.2 treats `&` in the
# replacement specially, and printf %q escapes it.
_PLACEHOLDER="__DOC_INGEST_ROOT__"
case "$SNIPPET" in
  *"$_PLACEHOLDER"*) ;;
  *) echo "install-hooks.sh: internal error: root placeholder missing from snippet" >&2; exit 1 ;;
esac
SNIPPET="${SNIPPET%%"$_PLACEHOLDER"*}$(printf '%q' "$DOC_INGEST_DIR")${SNIPPET#*"$_PLACEHOLDER"}"

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

  # The old README recipe wrote a hand-made hook that `source`d .env (arbitrary command
  # execution, #80/#221). Appending ours would run doc-ingest twice, so ask for removal instead.
  if grep -qE '^[^#]*(^|[ ;{])\. +"?\$_?D(OC_INGEST_DIR)?/\.env' "$hook" 2>/dev/null; then
    echo "legacy $repo  (unsafe hand-written hook sources .env: remove $hook, then re-run)" >&2
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
