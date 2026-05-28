<!-- This page is hand-authored. It will be replaced by `gen-cli-ref` once
     that generator is implemented (decision 0004). Source: src/cli.rs. -->

# CLI reference

Veld ships a single binary, `veld`, with multiple subcommands. Each subcommand
has its own flags (with both long-form and corresponding `VELD_*` env-var
overrides). For the canonical source, see [src/cli.rs](https://github.com/Portll/veld/blob/main/src/cli.rs).

## Subcommands

| Command | Purpose | Common flags |
|---|---|---|
| `veld server` | Start the HTTP API server | `--host`, `--port`, `--storage`, `--storage-backend`, `--production`, `--rate-limit`, `--max-concurrent` |
| `veld tui` | Launch the terminal dashboard | `--api-url`, `--api-key` |
| `veld serve` | Run as MCP server (stdio transport) | `--api-url`, `--api-key`, `--user-id`, `--mcp-enabled` |
| `veld init` | First-time setup — config, API key, ONNX runtime | — |
| `veld status` | Check server health and status | `--api-url`, `--api-key` |
| `veld doctor` | Diagnose common issues (storage, ONNX, port, server health) | — |
| `veld hook session-start \| prompt \| commit` | Output Claude Code hook JSON | (per subcommand) |
| `veld claude [args...]` | Launch Claude Code with the Veld proxy wired up | `--port`, plus trailing args forwarded to `claude` |
| `veld version` | Print version and build information | — |

## `veld server` — HTTP daemon

The most common subcommand. Starts the HTTP API on `127.0.0.1:3030` by
default.

```sh
veld server                                  # defaults: localhost:3030, redb backend
veld server --host 0.0.0.0 --port 8080       # bind to a different address
veld server --storage-backend rocksdb        # force the legacy backend
veld server --production                     # deny-all CORS fallback, backup auto-enable, safety warnings
veld server --rate-limit 0                   # disable rate limiting
```

All flags also accept env-var equivalents — see [config reference](config.md)
for the full list (`VELD_HOST`, `VELD_PORT`, `VELD_MEMORY_PATH`,
`VELD_STORAGE_BACKEND`, `VELD_ENV`, `VELD_RATE_LIMIT`, `VELD_MAX_CONCURRENT`).

## `veld serve` — MCP stdio transport

Not the HTTP daemon; this is the MCP transport for Claude Code / Cursor /
VS Code Copilot. Runs as a child process of the MCP client over stdio.

```sh
veld serve                                   # defaults to claude-code user_id
veld serve --user-id vscode-copilot          # different user id
```

## `veld claude` — launch Claude Code with veld wired

Launches Claude Code with the Veld proxy already in place. Easiest path for
one-off sessions:

```sh
veld claude                  # default port 3030
veld claude --port 3031      # custom veld port
veld claude --version        # forwards to claude --version
veld claude [-- any args here]  # all trailing args go to claude
```

## `veld init` and `veld doctor`

`veld init` runs the first-time setup wizard: creates `~/.veld/config.toml`,
generates an API key, downloads the ONNX MiniLM runtime. Idempotent — safe
to re-run.

`veld doctor` diagnoses common issues:

- Storage directory exists and is writable
- ONNX runtime is installed
- Port 3030 is available
- Server is reachable

Run it when something feels broken before opening an issue.

## See also

- [Configuration reference](config.md) — all env vars
- [Quickstart](../quickstart.md) — first-time setup
- [Deploying](../guides/deploying.md) — running as a service
