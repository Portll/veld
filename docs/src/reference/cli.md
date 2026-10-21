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
| `veld restart` | Stop a server gracefully and bring it back the way the monitor does | `--port` |
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

## `veld restart` — restart a server without killing its pid

```sh
veld restart                 # the configured port: VELD_PORT, else config.toml's `port`, else 3030
veld restart --port 3031     # another instance
```

`veld restart` finds the instance registered on the port in the
[instance registry](../architecture/instance-registry.md) and restarts it
through the code the monitor's **Restart server** uses (`restart_instance` in
`status-core`):

1. It stops the server with SIGTERM (forced on Windows) and waits up to 120 s
   for the graceful shutdown to finish.
2. If launchd names the process as a job's main process, it starts that job
   again with `launchctl kickstart` (never `-k`, which would kill it mid-flush).
3. Otherwise, if a supervisor such as Veld Desktop brings the server back
   within 10 s, it leaves the server to that supervisor.
4. Otherwise it relaunches the server from the launch recipe saved for it in
   the monitor (`launch-recipes.json` under `<config>/veld/monitor/`), else from
   the registry's recipe. The new process's environment is layered, later
   layers winning: `~/.config/veld/server.env`, the stopped process's own
   `VELD_*` variables, the recipe's variables, then the registry's host, port
   and data path. Its output is appended to `<config>/veld/logs/veld-<port>.log`.

`<config>/veld` is the directory that holds `config.toml`. The values the
command itself read from `config.toml` are not passed to the new process, which
reads the file itself; variables exported in the calling shell are inherited.

If nothing is registered on the port but a launch recipe is saved for it, the
server is started from that recipe. A server that answers `/health` without
being registered (one too old to register) is refused; stop it with
`veld stop --port N` first.

On success it prints one paragraph: what brought the server back (the launchd
job, a supervisor, or the new pid), where the output goes, and any `VELD_*`
variable the old process was started with that the new one could not be given.
If the server is not answering `/health` again within 90 s, it exits non-zero
with the error chain.

## See also

- [Configuration reference](config.md) — all env vars
- [Quickstart](../quickstart.md) — first-time setup
- [Deploying](../guides/deploying.md) — running as a service
