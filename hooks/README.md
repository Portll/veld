# Veld Hooks

Claude Code / Claude CLI lifecycle hooks that wire the editor into the Veld
memory backend. The entry point is [`memory-hook.ts`](./memory-hook.ts); it is
executed by `bun run` for every lifecycle event registered in
[`claude-settings.json`](./claude-settings.json) (and the project-local
`.claude/settings.json` shim).

Events handled: `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `PreToolUse`,
`PostToolUse`, `PostToolUseFailure`, `SubagentStop`, `Stop`.

| Event | What the hook does |
| --- | --- |
| `SessionStart`, `UserPromptSubmit` | Surfaces memories as `hookSpecificOutput.additionalContext` JSON on stdout. The prompt is also ingested. |
| `PreToolUse` | Surfaces memories for the tool about to run. This is the only surfacing per tool use, except for an orchestration subagent, which surfaces after it completes. |
| `PostToolUse` | Records the action for feedback attribution. Stores a memory only for a shell command with an error on stderr, and completes `[ORCH-TODO:…]` todos when an `Agent` (or legacy `Task`) run completes. |
| `PostToolUseFailure` | Stores a failed `Bash`/`PowerShell` command (Claude Code reports a non-zero exit here, not in `PostToolUse`). |
| `SubagentStop` | Stores the subagent's `last_assistant_message`. |
| `SessionEnd` | Removes the marker file and the session's state file. |

A timeout is not an outage. When veld does not answer in time the hook injects
nothing. Only a refused connection or an HTTP error produces a `<veld-warning>`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VELD_HOOK_TIMEOUT_MS` | `4000` | Per-request budget on tool events. Keep it below the 5 s hook `timeout`. |
| `VELD_HOOK_PROMPT_TIMEOUT_MS` | `8000` | Budget for `SessionStart` and `UserPromptSubmit`. Keep it below their 15 s and 10 s hook `timeout`. |
| `VELD_HOOK_DEBOUNCE_MS` | `500` | Repeated identical surfacing within this window is answered from the state file. `0` disables it. |
| `VELD_HOOK_STATE_DIR` | `<tmpdir>/veld-hook` | Where per-session state lives. |

Hook `timeout` values in `claude-settings.json` are seconds, as Claude Code
reads them. If a hook outlives its `timeout`, Claude Code discards its output.

## Session state file

Claude Code starts a new process for every event, so the hook keeps what must
span events in `${VELD_HOOK_STATE_DIR}/<session_id>.json`. That covers the
episode id and sequence, the previous memory id, the last surfaced ids and
output for implicit feedback, the reinforcement counts, the pending tool
actions, the debounce cache and the marker path. Writes are atomic (tmp +
rename, mode `0600`). Read and write errors fall back to fresh state. Files
untouched for a day are pruned at `SessionStart`.

## Agent session marker file

On `SessionStart` the hook drops a small JSON file in the current working
directory so external tooling (notably the agent-session detection helper)
can identify which chat brand is driving this process.

**Path:** `${cwd}/.veld-agent-session.<pid>` — e.g.
`./.veld-agent-session.12345`.

**Format:**

```json
{
  "agent_id":   "Claude",
  "started_at": "2026-05-27T05:00:00.000Z",
  "pid":        12345,
  "binary":     "bun.exe"
}
```

| Field        | Source                                                                                   | Notes                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `agent_id`   | Hard-coded `"Claude"`                                                                    | Chat brand. Identical for Claude Code, Claude CLI, and Claude Desktop — they share a brand. |
| `started_at` | `new Date().toISOString()`                                                               | ISO-8601 UTC.                                                                               |
| `pid`        | `process.pid`                                                                            | The bun/node process running the hook.                                                      |
| `binary`     | basename of `process.argv[0]` (or `"claude-desktop"` if `CLAUDE_DESKTOP` env var is set) | Best-effort launcher detection; diagnostics only, not load-bearing.                         |

**Lifecycle:**

1. Written atomically on every `SessionStart` (write to `<path>.tmp`, then
   `renameSync`). The write is idempotent: re-running the hook in the same
   process overwrites the file in place.
2. Removed on `SessionEnd`. Deletion is best-effort — failures are logged to
   stderr and swallowed. The file is gitignored (`.veld-agent-session.*` in
   the repo root `.gitignore`), so leaks are not critical.

**Why not finer-grained brand IDs?** Claude Code, Claude CLI, and Claude
Desktop all share the user-id `claude-code` in Veld (see project
`CLAUDE.md` › Client Integration). They are the same chat brand to the
backend, so `agent_id` is fixed at `"Claude"`. The `binary` field is the
only field that distinguishes launchers, and it is advisory only.

## Tests

`memory-hook.test.ts` covers the pure helpers, the marker and state file
lifecycles, and the settings files. It also runs the hook as a subprocess per
event against a mock veld server on an ephemeral port. Run with:

```sh
bun test hooks/
```
