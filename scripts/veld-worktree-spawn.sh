#!/usr/bin/env bash
# veld-worktree-spawn.sh — spawn a sibling git worktree per agent x branch.
#
# Convention:
#   <parent-of-main-worktree>/<repo-name>-<branch-slug>
# e.g. main worktree /repos/veld + branch w5/journaled-writer
#      -> /repos/veld-w5-journaled-writer
#
# Branch resolution:
#   - local branch exists:         git worktree add <path> <branch>
#   - origin/<branch> exists:      git worktree add <path> -b <branch> --track origin/<branch>
#   - neither:                     git worktree add <path> -b <branch> origin/main
#
# Conflicts:
#   - target path already exists -> abort, suggest `git worktree remove`.
#
# Slugging: lowercased, [^a-z0-9]+ -> '-', trimmed. '/', '_', '.' all become '-'.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: veld-worktree-spawn.sh <branch> [--agent <id>] [--register] [--help]

  <branch>     Branch to check out in the new worktree (e.g. w5/journaled-writer).
  --agent      Agent id to tag the session with. Defaults to a random 8-char hex.
  --register   POST a session record to Veld's /api/remember (kind=session).
               Honours $VELD_API (default http://127.0.0.1:8080).
  --help       Show this help.

Creates a sibling worktree at <parent>/<repo>-<branch-slug>, copies .claude/,
.vscode/, .mcp.json, and sleight/ (if present), and prints the cd command.
EOF
}

BRANCH=""
AGENT=""
REGISTER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)    AGENT="${2:-}"; shift 2 ;;
    --agent=*)  AGENT="${1#*=}"; shift ;;
    --register) REGISTER=1; shift ;;
    --help|-h)  usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "error: unknown flag: $1" >&2; usage >&2; exit 2 ;;
    *)  if [[ -z "$BRANCH" ]]; then BRANCH="$1"; shift; else
          echo "error: unexpected positional arg: $1" >&2; exit 2; fi ;;
  esac
done

if [[ -z "$BRANCH" ]]; then usage >&2; exit 1; fi

fail() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m%s\033[0m\n' "$*"; }
note() { printf '\033[90m%s\033[0m\n' "$*"; }

# Refuse outside a git repo.
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  fail "not inside a git repository"
fi

# Main worktree = first 'worktree' entry from `git worktree list --porcelain`.
MAIN_WT="$(git worktree list --porcelain | awk '/^worktree /{print substr($0,10); exit}')"
[[ -z "$MAIN_WT" ]] && MAIN_WT="$(git rev-parse --show-toplevel)"
# Resolve to absolute.
MAIN_WT="$(cd "$MAIN_WT" && pwd)"

PARENT_DIR="$(dirname "$MAIN_WT")"
REPO_NAME="$(basename "$MAIN_WT")"

# Slug: lowercase, non-alnum -> '-', squeeze, trim.
SLUG="$(printf '%s' "$BRANCH" | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
[[ -z "$SLUG" ]] && fail "branch '$BRANCH' slugs to empty string"

TARGET="$PARENT_DIR/$REPO_NAME-$SLUG"

if [[ -e "$TARGET" ]]; then
  fail "target path already exists: $TARGET
       remove it first with: git worktree remove \"$TARGET\""
fi

# Agent id default.
if [[ -z "$AGENT" ]]; then
  if [[ -r /dev/urandom ]]; then
    AGENT="$(LC_ALL=C tr -dc 'a-f0-9' </dev/urandom | head -c 8)"
  else
    AGENT="$(printf '%08x' "$RANDOM$RANDOM")"
  fi
fi

# Branch resolution.
HAS_LOCAL=0;  git show-ref --verify --quiet "refs/heads/$BRANCH"          && HAS_LOCAL=1  || true
HAS_REMOTE=0; git show-ref --verify --quiet "refs/remotes/origin/$BRANCH" && HAS_REMOTE=1 || true

info "main worktree : $MAIN_WT"
info "new worktree  : $TARGET"
info "branch        : $BRANCH  (local=$HAS_LOCAL, origin=$HAS_REMOTE)"
info "agent id      : $AGENT"

if   [[ $HAS_LOCAL  -eq 1 ]]; then git worktree add -- "$TARGET" "$BRANCH"
elif [[ $HAS_REMOTE -eq 1 ]]; then git worktree add -b "$BRANCH" --track -- "$TARGET" "origin/$BRANCH"
else                                git worktree add -b "$BRANCH" -- "$TARGET" "origin/main"
fi

# Idempotent per-file copy: skip if dest exists.
copy_if_missing() {
  local src="$1" dst="$2" label="$3"
  [[ -e "$src" ]] || return 0
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    # -print0 + while-read for path safety; skip nested worktrees dir.
    ( cd "$src" && find . -mindepth 1 \( -path './worktrees' -o -path './worktrees/*' \) -prune -o -print0 ) \
      | while IFS= read -r -d '' rel; do
          rel="${rel#./}"
          local s="$src/$rel" d="$dst/$rel"
          if [[ -d "$s" ]]; then
            mkdir -p "$d"
          elif [[ -e "$d" ]]; then
            note "[skip] $label/$rel exists"
          else
            mkdir -p "$(dirname "$d")"
            cp -p "$s" "$d"
          fi
        done
  else
    if [[ -e "$dst" ]]; then note "[skip] $label exists"
    else cp -p "$src" "$dst"; fi
  fi
}

info "copying agent config -> $TARGET"
copy_if_missing "$MAIN_WT/.claude"   "$TARGET/.claude"   ".claude"
copy_if_missing "$MAIN_WT/.vscode"   "$TARGET/.vscode"   ".vscode"
copy_if_missing "$MAIN_WT/sleight"   "$TARGET/sleight"   "sleight"
copy_if_missing "$MAIN_WT/.mcp.json" "$TARGET/.mcp.json" ".mcp.json"

# Optional Veld session registration.
if [[ $REGISTER -eq 1 ]]; then
  API="${VELD_API:-http://127.0.0.1:8080}"
  STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  PAYLOAD=$(cat <<EOF
{"kind":"session","tags":["agent-session","agent:$(esc "$AGENT")","branch:$(esc "$BRANCH")"],"content":"Spawned worktree $(esc "$TARGET") on branch $(esc "$BRANCH") for agent $(esc "$AGENT")","facets":{"agent_session":{"worktree_path":"$(esc "$TARGET")","branch":"$(esc "$BRANCH")","agent_id":"$(esc "$AGENT")","started_at":"$STARTED_AT","parent_repo":"$(esc "$MAIN_WT")"}}}
EOF
)
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 5 -X POST -H 'Content-Type: application/json' \
         --data "$PAYLOAD" "$API/api/remember" >/dev/null; then
      info "registered session with Veld at $API"
    else
      printf '\033[33mwarn:\033[0m failed to register session. Worktree is still ready.\n' >&2
    fi
  else
    printf '\033[33mwarn:\033[0m curl not found; skipped --register.\n' >&2
  fi
fi

echo
printf '\033[32mWorktree ready.\033[0m\n'
echo "Next:"
echo "  cd \"$TARGET\" && claude"
