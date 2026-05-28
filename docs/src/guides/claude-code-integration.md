# Claude Code integration

Veld plugs into Claude Code via two mechanisms — the MCP server (46 tools)
and hooks (automatic pre/post-tool memory). Both are configured through a
single `.claude/settings.json` file in your project. For most use cases,
`veld claude [args...]` is the easiest path — it launches Claude Code with
the Veld proxy already wired up.

## Quick setup — explicit

```sh
# 1. Start the veld HTTP daemon (once per machine)
veld server &

# 2. Install the MCP server globally
npm install -g @veld/memory-mcp

# 3. In your project, configure Claude Code (or skip this and use `veld claude`)
mkdir -p .claude
cat > .claude/settings.json <<'EOF'
{
  "mcpServers": {
    "veld": {
      "command": "veld-mcp",
      "args": ["--user-id", "claude-code"]
    }
  },
  "hooks": {
    "PreToolUse": "node ./hooks/memory-hook.ts pre",
    "PostToolUse": "node ./hooks/memory-hook.ts post",
    "SessionStart": "node ./hooks/memory-hook.ts session-start",
    "SessionEnd": "node ./hooks/memory-hook.ts session-end",
    "UserPromptSubmit": "node ./hooks/memory-hook.ts prompt",
    "Stop": "node ./hooks/memory-hook.ts stop"
  }
}
EOF
```

## What the integration does

| Layer | Action |
|---|---|
| **Session start** | Surface relevant memories from past sessions; check `list_todos` |
| **User prompt submit** | Retrieve memories related to the new prompt; inject into context |
| **Pre-tool** | Surface memories about the file/function/topic the tool is about to touch |
| **Post-tool** | Record the action (Edit, Write, Bash, Read, Grep, Glob, etc.) as a memory |
| **Session end / Stop** | Encode important context for future sessions |

The hooks are implemented in [`hooks/memory-hook.ts`](https://github.com/Portll/veld/blob/main/hooks/memory-hook.ts).

## Quick setup — `veld claude`

If you don't want to edit `.claude/settings.json` yourself, veld has a
launcher subcommand:

```sh
veld claude          # launches Claude Code with veld proxy on default port
veld claude --help   # forwards --help to claude itself
veld claude --port 3031  # launches with veld on a different port
```

This is the easiest path for one-off sessions or when you don't want the
hook layer fully wired (you still get MCP tools).

## Explicit tools

When the automatic layer isn't enough, Claude Code can call MCP tools directly.
See the [MCP tools reference](../reference/mcp-tools.md) for the full list.

Most-used:

- `remember(content, importance)` — emphasize a memory
- `recall(query)` — semantic search
- `recall_by_tags(tags)` — tag-filtered search
- `forget(memory_id)` — suppress a memory
- `proactive_context()` — get contextually relevant memories
- `add_todo(content)` — track work across sessions

## User ID

The default `user_id` for Claude Code is `claude-code`. This is also the
default for the Claude CLI, so the two share memory continuity. Use a
different `user_id` if you want isolated memory streams.

## Standalone (external project)

For projects that don't have a `.claude/settings.json` of their own (e.g.,
a casual exploration session), use the standalone hook config:

```sh
export VELD_HOOKS_DIR=/path/to/veld/hooks
cp $VELD_HOOKS_DIR/claude-settings.json ~/.claude/settings.json
```

The standalone config uses `$VELD_HOOKS_DIR` so a single veld install can
serve many projects.

## See also

- [VS Code Copilot integration](vscode-copilot.md)
- [Cursor integration](cursor-integration.md)
- [Tuning retrieval](tuning-retrieval.md)
- [MCP tools reference](../reference/mcp-tools.md)
