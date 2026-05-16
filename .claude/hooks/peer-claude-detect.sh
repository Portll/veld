#!/usr/bin/env bash
# peer-claude-detect.sh — SessionStart hook.
#
# Detects other live Claude Code sessions working on the same git project and
# emits a system-reminder recommending a sibling worktree if any are found.
#
# Anchor incident: 2026-05-15. A peer Claude window on antidote-dev `main` ran
# a pre-rebase auto-stash that swept the current session's 1500+ LOC of
# uncommitted DevNotes work into stash@{0}. Recovery via stash@{0}^3 worked
# but the contamination would have been prevented entirely by working in a
# sibling worktree from the start.
#
# Implementation: reuses presence_siblings() from claude-agent-lib.sh which
# already filters by project_hash, drops stale presence files (os.kill(pid, 0)),
# and excludes self. If siblings exist on the current project, emits a
# <system-reminder> block to stdout suggesting a sibling-worktree workflow.
#
# Output contract:
#   - When peers detected: <system-reminder> block to stdout.
#   - When no peers detected: silent (no output).
#   - On error (lib unsourceable, etc.): silent (no output).
#   - Exit 0 always — never blocks the session.

# Run only if we have a project directory and the agent lib is available.
[ -n "${CLAUDE_PROJECT_DIR:-}" ] || exit 0
[ -f "$HOME/.claude/hooks/claude-agent-lib.sh" ] || exit 0

# claude-agent-lib.sh expects CLAUDE_PROJECT_DIR + sets CLAUDE_AGENT_PID +
# CLAUDE_PROJECT_HASH. The library is idempotent so sourcing it here is safe
# even if another hook (e.g., the veld session-start) already sourced it.
# shellcheck disable=SC1091
source "$HOME/.claude/hooks/claude-agent-lib.sh" 2>/dev/null || exit 0

# Gather sibling list. If presence_siblings function is missing or fails,
# bail silently.
type presence_siblings >/dev/null 2>&1 || exit 0
siblings_output=$(presence_siblings 2>/dev/null || true)

# If "No active sibling agents." (empty case), say nothing.
if [ -z "$siblings_output" ] || echo "$siblings_output" | grep -q "No active sibling agents"; then
  exit 0
fi

# Resolve a friendly project name and a suggested worktree path.
project_basename=$(basename "${CLAUDE_PROJECT_DIR:-.}")
sibling_path="${CLAUDE_PROJECT_DIR}-$(date +%y%m%d)"

cat <<EOF
<system-reminder>
Peer Claude session(s) detected on this project (${project_basename}):

$(echo "$siblings_output" | sed 's/^/  /')

To avoid cross-window contamination (pre-rebase auto-stash, push races,
file-lock conflicts), consider working in a sibling git worktree on a
feature branch instead of directly on the shared branch:

  cd "$(dirname "$CLAUDE_PROJECT_DIR")"
  git -C "$CLAUDE_PROJECT_DIR" fetch origin main
  git -C "$CLAUDE_PROJECT_DIR" worktree add "$sibling_path" -b feat/<topic> origin/main
  cd "$sibling_path"

Pre-flight before pushing: scripts/guards/check-origin-race.sh (if the
project has it) validates that no peer push intervened between fetch and push.

If you have read-only intent (lookup, review, browse), this reminder is safe
to ignore — only state-changing work risks contamination.
</system-reminder>
EOF

exit 0
