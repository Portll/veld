#!/usr/bin/env bash
# plan-mode-status.sh — UserPromptSubmit hook.
#
# Emits a deterministic 1-line status to stdout reflecting whether plan-mode is
# active for the current project. Reads ~/.claude/state/plan-mode-<hash>.json
# maintained by the plan-mode-track skill (Claude is expected to invoke that
# skill on plan-mode entry/exit).
#
# Anchor incident: 2026-05-15. ExitPlanMode tool call returned a
# "stream closed before response received" error but plan-mode HAD actually
# exited. Claude retried, was blocked by the lingering harness lock, retried
# again, then finally tested with a non-plan edit — wasted ~3 turns realising
# plan-mode was already off. This hook makes the truth visible on every turn.
#
# Output contract:
#   - When plan-mode active for this project: one line to stdout
#       "Plan mode: ACTIVE | plan file: <path> | entered: <iso>"
#   - When state file is stale (> 24h since entered_at): one line
#       "Plan mode: OFF (stale state file — manually delete ~/.claude/state/plan-mode-<hash>.json)"
#   - When no state file or active: false: one line
#       "Plan mode: OFF"
#   - On read error (malformed JSON, perms): silent (no output).
#   - Exit 0 always.

[ -n "${CLAUDE_PROJECT_DIR:-}" ] || exit 0

STATE_DIR="$HOME/.claude/state"
PROJECT_HASH=$(echo "$CLAUDE_PROJECT_DIR" | md5 | cut -c1-12)
STATE_FILE="$STATE_DIR/plan-mode-${PROJECT_HASH}.json"

if [ ! -f "$STATE_FILE" ]; then
  echo "Plan mode: OFF"
  exit 0
fi

# Parse + freshness-check + emit. python3 because we're already using it
# elsewhere in this hook directory; jq would be a new dependency.
python3 - "$STATE_FILE" <<'PY' 2>/dev/null || echo "Plan mode: OFF"
import json, sys, datetime, os
path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
    if not data.get('active', False):
        print('Plan mode: OFF')
        sys.exit(0)
    entered = data.get('entered_at', '')
    plan_file = data.get('plan_file_path', '<unknown>')
    try:
        e = datetime.datetime.fromisoformat(entered.replace('Z', '+00:00'))
        age_h = (datetime.datetime.now(datetime.timezone.utc) - e).total_seconds() / 3600
        if age_h > 24:
            print(f'Plan mode: OFF (stale state file — manually delete {os.path.basename(path)})')
            sys.exit(0)
    except Exception:
        pass
    print(f'Plan mode: ACTIVE | plan file: {plan_file} | entered: {entered}')
except Exception:
    sys.exit(2)
PY
exit 0
