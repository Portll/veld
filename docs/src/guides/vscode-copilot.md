# VS Code Copilot integration

GitHub Copilot's Agent mode supports MCP. Veld registers 47 tools that Copilot
can call.

## Setup

In your project, create `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "veld": {
      "command": "veld-mcp",
      "args": ["--user-id", "vscode-copilot"]
    }
  }
}
```

And `.github/copilot-instructions.md` for workspace-level switches:

```markdown
# Veld memory available

Use `remember(content)` to persist important context.
Use `recall(query)` to retrieve past memories before answering.
Use `proactive_context()` to surface contextually relevant memories.

See https://portll.github.io/veld/ for the full tool list.
```

## User ID

Default `vscode-copilot`. Distinct from `claude-code` — these are two
separate memory streams unless you intentionally set them to the same ID.

## What's different from Claude Code

VS Code Copilot does not currently support all six hook lifecycle events the
way Claude Code does. The hooks layer (`hooks/memory-hook.ts`) is therefore
not active for Copilot. Copilot only uses the **MCP tool** layer.

In practice this means:
- Pre/post-tool memory surfacing is **manual** — Copilot calls `recall`
  / `proactive_context` itself when prompted by workspace instructions.
- Session-start memory injection happens via Copilot's own workspace
  context loading mechanism + the `.github/copilot-instructions.md` file.

## See also

- [Claude Code integration](claude-code-integration.md)
- [MCP tools reference](../reference/mcp-tools.md)
