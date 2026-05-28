# Deploying veld

Veld is a single binary. Deployment is "put the binary somewhere and run it
as a service."

```admonish warning
`v0.7.6-unstable` is not yet production-hardened. For desktop/developer use
the current branch works well; for production multi-tenant deployment, wait
for the v0.9 release. See [PROGRESS.md](https://github.com/Portll/veld/blob/main/PROGRESS.md)
for status.
```

## Local desktop

The simplest deployment: a launchd/systemd/service unit that runs
`veld server` (the HTTP daemon) in the background. Note: `veld serve` is
the MCP stdio transport — different command.

| Platform | Approach |
|---|---|
| Linux | Systemd user unit (see `packaging/linux/veld.service`) |
| macOS | launchd agent (see `packaging/macos/net.portll.veld.plist`) |
| Windows | NSSM-managed service (`packaging/windows/install-service.ps1`) |

Each platform's packaging directory has the artifact and an install script.

## Docker

Pre-built image:

```sh
docker run -d \
  --name veld \
  -p 3030:3030 \
  -v veld_data:/data \
  varunveld/veld
```

Data persists in the named volume. Config via env vars (`VELD_*` —
see [Configuration reference](../reference/config.md)).

## Cloud / multi-node

For multi-node deployments (rare for veld since it's edge-first), enable the
`telemetry` feature for distributed tracing:

```sh
cargo build --release --features telemetry
```

This pulls in OpenTelemetry exporters (~200 additional crates). Configure
via `OTEL_*` env vars.

For multi-tenant collective learning, enable `multi-tenant`:

```sh
cargo build --release --features multi-tenant
```

See [Multi-tenant](multi-tenant.md) for the operational model.

## Embedding backend

Veld doesn't include an embedding model by default. Configure an external
embedding endpoint via `VELD_EMBED_URL`:

| Backend | URL |
|---|---|
| LM Studio | `http://127.0.0.1:1234/v1` |
| Ollama | `http://127.0.0.1:11434/v1` |
| vLLM | `http://your-vllm:8000/v1` |
| OpenAI-compatible | any |

Veld auto-detects the backend type from the URL response shape.

## Reverse proxy

If you expose veld beyond localhost, put it behind a reverse proxy with TLS:

- **Caddy** — automatic HTTPS via Let's Encrypt
- **nginx** — manual cert management
- **Cloudflare Tunnel** — no public port

Veld checks the bind host via `is_local_bind_host()`; binding to
`0.0.0.0` is a deliberate choice that disables some safety defaults.

## See also

- [Configuration reference](../reference/config.md) — env vars
- [Multi-tenant](multi-tenant.md) — when one veld serves many users
- [Zenoh / robotics](zenoh-robotics.md) — multi-node edge deployments
