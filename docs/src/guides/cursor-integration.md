# Cursor integration

Cursor supports MCP via project-level `.cursor/mcp.json` or global
`~/.cursor/mcp.json`. Veld registers as an MCP server.

## Setup

Project-level:

```json
{
  "mcpServers": {
    "veld": {
      "command": "veld-mcp",
      "args": ["--user-id", "cursor"]
    }
  }
}
```

The `veld-mcp` binary ships with the `@veld/memory-mcp` npm package:

```sh
npm install -g @veld/memory-mcp
```

Alternatively, use the Rust MCP server directly (no Node required):

```json
{
  "mcpServers": {
    "veld": {
      "command": "veld",
      "args": ["serve"]
    }
  }
}
```

`veld serve` is the MCP stdio transport (not the HTTP daemon; that's
`veld server`). It speaks the same MCP protocol via `rmcp` and exposes
the same tools. Useful when you don't want to install the `@veld/memory-mcp`
npm package.

## Cursor Directory listing

Veld is listed in the Cursor Directory as `veld-1`:
<https://cursor.directory/plugins/veld-1>.

Install from the directory for the easiest setup.

## See also

- [Claude Code integration](claude-code-integration.md)
- [VS Code Copilot integration](vscode-copilot.md)
- [MCP tools reference](../reference/mcp-tools.md)
