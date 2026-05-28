# Quickstart

Get veld running and store your first memory in under 60 seconds.

```admonish tip
If you only want to use veld with Claude Code and don't care about the
underlying details, skip to step 5 and use `veld claude` — it handles
the daemon, MCP wiring, and hooks for you.
```

## Prerequisites

- Rust toolchain (`rustup` — stable channel).
- An embedding server: [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), or any OpenAI-compatible embedding endpoint.

## 1. Install

```sh
cargo install veld
```

For the MCP server (Claude Code / VS Code Copilot integration):

```sh
npm install -g @veld/memory-mcp
```

## 2. Start the server

```sh
veld server
```

Veld starts on `http://127.0.0.1:3030`. (Note: `veld serve` is the MCP stdio
transport — different command. `veld server` is the HTTP daemon.) Data is stored at:

| Platform | Default path |
|---|---|
| Linux | `~/.local/share/veld/` |
| macOS | `~/Library/Application Support/veld/` |
| Windows | `%APPDATA%\veld\` |

Verify it's running:

```sh
curl http://127.0.0.1:3030/health
# {"status":"ok"}
```

## 3. Store a memory

```sh
curl -X POST http://127.0.0.1:3030/api/remember \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "me",
    "content": "The auth middleware uses API-key headers, not cookies.",
    "importance": 0.8
  }'
```

## 4. Retrieve memories

```sh
curl "http://127.0.0.1:3030/api/recall?user_id=me&query=auth+middleware"
```

## 5. Connect Claude Code

Add veld to your project's `.claude/settings.json`:

```json
{
  "mcpServers": {
    "veld": {
      "command": "veld-mcp",
      "args": ["--user-id", "claude-code"]
    }
  }
}
```

Claude Code now has access to 46 memory tools (`remember`, `recall`, `proactive_context`, etc.). On session start, run `list_todos` and `proactive_context` to resume where you left off.

**Easier path:** `veld claude [args...]` launches Claude Code with the Veld
proxy already wired up — no `.claude/settings.json` editing required.

## 6. Run as a persistent service

So veld survives reboots and session logouts, install it as a service. See
[Deploying](guides/deploying.md) for the full service-unit recipes — the
`packaging/{linux,macos,windows}/` directories ship the templates.

```sh
# Easiest: let veld do the setup
veld init       # creates config, generates API key, downloads ONNX runtime
veld doctor     # diagnoses common issues (port conflicts, ONNX, storage)
```

## Next steps

- [Architecture overview](architecture/overview.md) — understand how veld's retrieval pipeline works.
- [Claude Code integration guide](guides/claude-code-integration.md) — full walkthrough with hook configuration.
- [Configuration reference](reference/config.md) — all environment variables and config keys.
