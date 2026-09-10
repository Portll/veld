# Deploying veld

Veld is a single binary. Deployment is "put the binary somewhere and run it
as a service."

> **⚠️ Production readiness**
>
> `v0.7.6-unstable` is not yet production-hardened. For desktop/developer use
> the current branch works well; for production multi-tenant deployment, wait
> for the v0.9 release. See [PROGRESS.md](https://github.com/Portll/veld/blob/main/PROGRESS.md)
> for status.

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

Veld picks its PRIMARY embedder at startup, in this order, and reports the choice on
`GET /health` as `embedder` (`id`, `model`, `dimension`, `kind`, `reason`):

1. `VELD_EMBEDDER`, when set: `nomic` (local ONNX), `minilm` (local ONNX), `http`
   (the endpoint below), or an `http(s)://host:port[#model]` URL. An explicit choice
   that cannot be honoured fails startup with the endpoint named; it never falls back.
2. Nomic-embed-text-v1.5 ONNX (768d) when its model files are present
   (`VELD_AUTO_DOWNLOAD_MODELS=true` fetches them).
3. The OpenAI-compatible HTTP endpoint, when it answers a probe for the configured
   model. This is the same nomic model served by a process that already holds it.
4. MiniLM-L6 ONNX (384d). With no model files at all veld refuses to start rather than
   run on hash vectors (`VELD_ALLOW_SIMPLIFIED_EMBEDDINGS=1` overrides, for tests only).

The HTTP endpoint is any server speaking `/v1/models` and `/v1/embeddings`. Give the
BASE URL, without `/v1`; the client appends the path:

| Backend | Start it | `VELD_EMBEDDING_API_URL` |
|---|---|---|
| LM Studio | load `text-embedding-nomic-embed-text-v1.5` (bundled) | `http://127.0.0.1:1234` (default) |
| llama.cpp | `llama-server -m nomic-embed-text-v1.5.Q4_K_M.gguf --embedding --pooling mean --alias text-embedding-nomic-embed-text-v1.5 --port 8080` | `http://127.0.0.1:8080` |
| vLLM / LocalAI / llamafile | serve the model under the same name | its base URL |

`VELD_EMBEDDING_API_MODEL` (default `text-embedding-nomic-embed-text-v1.5`) must match an
id the server lists under `/v1/models`; the probe refuses otherwise and prints what the
server does list. `--alias` is how llama-server is given that name. Nomic needs its task
prefixes (`search_document: ` / `search_query: `); they are applied automatically when the
model name contains `nomic`, and `VELD_EMBEDDING_API_TASK_PREFIXES=0|1` forces either way.
`VELD_EMBEDDING_API_TIMEOUT_MS` (default 30000) covers a cold model load, which is the
slow case, not the embed. Ollama is not supported: its native API is not OpenAI-shaped and
was removed on 2026-08-27.

**Switching embedders re-embeds the store.** The embedder id and dimension are recorded in
`.veld_embedder` / `.veld_embedding_dim` beside each tenant's storage; when they differ from
the active embedder, every memory is re-embedded at startup (crash-resumable; the marker
advances only after a clean pass). Once a corpus marker names an HTTP endpoint, later boots
use that endpoint or refuse, so an endpoint that happens to be down cannot silently swap
the store back to a local model and trigger a second full re-embed. Budget the first boot
after a switch accordingly: a 2.5 GB tenant is minutes to tens of minutes against a local
server.

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
