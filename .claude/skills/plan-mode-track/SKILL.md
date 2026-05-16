---
name: plan-mode-track
description: Mark plan-mode entry/exit so the plan-mode-status.sh UserPromptSubmit hook can report state to every subsequent turn. Invoke immediately AFTER the harness enters plan-mode and immediately AFTER ExitPlanMode succeeds (or anytime ExitPlanMode returns an ambiguous error).
---

# Plan-Mode Tracker

Keeps `~/.claude/state/plan-mode-<project_hash>.json` in sync with the harness's actual plan-mode state. Without this, the `plan-mode-status.sh` `UserPromptSubmit` hook can't tell Claude whether plan-mode is active.

## When to invoke

| Trigger | Args | Effect |
|---|---|---|
| Harness just entered plan-mode (system-reminder "Plan mode is active" appeared) | `enter <plan-file-absolute-path>` | Writes the state file with `active: true`. |
| `ExitPlanMode` tool returned success | `exit` | Deletes the state file. |
| `ExitPlanMode` tool returned an ambiguous error (e.g., "stream closed before response received") | `verify` | Probes — if the next non-plan edit succeeds, calls `exit`; otherwise leaves state alone. |

## Args contract

- `enter <plan-file>` — `plan-file` is the absolute path to the plan markdown file the harness named.
- `exit` — no args.
- `verify` — no args; performs a one-shot probe and may transition to `exit` if successful.

## Commands

### `enter <plan-file>`

```bash
PROJ=$(pwd)
HASH=$(echo "$PROJ" | md5 | cut -c1-12)
mkdir -p ~/.claude/state
cat > ~/.claude/state/plan-mode-${HASH}.json <<JSON
{
  "active": true,
  "plan_file_path": "<plan-file>",
  "entered_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "project_hash": "${HASH}"
}
JSON
```

### `exit`

```bash
PROJ=$(pwd)
HASH=$(echo "$PROJ" | md5 | cut -c1-12)
rm -f ~/.claude/state/plan-mode-${HASH}.json
```

### `verify`

1. Attempt a no-op `Read` on any non-plan file (e.g., the project's `package.json` or `README.md`).
2. If the read succeeded without a plan-mode lock error → run the `exit` commands above.
3. If the read was blocked by the plan-mode lock → state file is correct as-is, do nothing.

## Why this exists

Claude Code's harness doesn't expose plan-mode state to hook scripts. So the only way the `UserPromptSubmit` hook can tell Claude "you're in/out of plan mode" is via a state file that Claude itself maintains. This skill is the discipline layer.

The 2026-05-15 DevNotes session anchor incident: `ExitPlanMode` returned `Tool permission stream closed before response received`, plan-mode had actually exited successfully, but Claude wasted three turns retrying before discovering it. With this skill + the hook, the very next `UserPromptSubmit` would have surfaced `Plan mode: OFF` and the truth would have been obvious instantly.

## Stale-file safety

If Claude crashes between `enter` and `exit`, the state file lingers. The hook treats files older than 24h as "stale" and reports `Plan mode: OFF (stale state file — manually delete …)`. Users can delete the file at any time without affecting harness behavior.

## Not a substitute for the harness lock

The actual plan-mode write-lock is enforced by Claude Code's harness, not this skill. This skill is purely informational — it surfaces state, never gates behavior.
