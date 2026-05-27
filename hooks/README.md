# Veld Hooks

Claude Code / Claude CLI lifecycle hooks that wire the editor into the Veld
memory backend. The entry point is [`memory-hook.ts`](./memory-hook.ts); it is
executed by `bun run` for every lifecycle event registered in
[`claude-settings.json`](./claude-settings.json) (and the project-local
`.claude/settings.json` shim).

Events handled: `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `PreToolUse`,
`PostToolUse`, `SubagentStop`, `Stop`.

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

`memory-hook.test.ts` covers the pure formatting helpers and the marker
file lifecycle (write → readback → JSON shape → cleanup). Run with:

```
bun test hooks/
```
