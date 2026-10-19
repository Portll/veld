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
| llama.cpp | `scripts/embedder-server.sh`, as a LaunchAgent from `packaging/macos/net.portll.veld-embedder.plist` (below) | `http://127.0.0.1:8095` |
| vLLM / LocalAI / llamafile | serve the model under the same name | its base URL |

`VELD_EMBEDDING_API_MODEL` (default `text-embedding-nomic-embed-text-v1.5`) must match an
id the server lists under `/v1/models`; the probe refuses otherwise and prints what the
server does list. `--alias` is how llama-server is given that name. Nomic needs its task
prefixes (`search_document: ` / `search_query: `); they are applied automatically when the
model name contains `nomic`, and `VELD_EMBEDDING_API_TASK_PREFIXES=0|1` forces either way.
`VELD_EMBEDDING_API_TIMEOUT_MS` (default 30000) covers a cold model load, which is the
slow case, not the embed. Ollama is not supported: its native API is not OpenAI-shaped and
was removed on 2026-08-27.

**A text longer than the model's window is embedded in pieces.** `VELD_EMBEDDING_API_MAX_TOKENS`
is the window (default 2048 for nomic, 8192 for bge-m3, no limit for other models; 0 turns it
off). A longer text is cut on `chunk_text`'s boundaries, each piece is embedded, and the text gets
the normalised mean of the pieces. A server keeps only the opening of an over-long input, and LM
Studio does so without an error. Serving nomic past its trained 2048 tokens does not help: on 22
long documents, plain 8192 context halved recall@1 (0.34 to 0.14), and YaRN ×4 reached 0.39,
against 0.52 for the mean of pieces. Where the server answers `POST /tokenize` (llama-server),
veld counts exactly; elsewhere it assumes 2 bytes per token, the densest this store's text runs.
Only a text with more bytes than the window is counted at all.

### The embedder on its own llama-server

LM Studio serves embeddings from the same queue as any chat model it holds, so a busy LLM delays
every remember and recall. `scripts/embedder-server.sh` runs the embedding model on its own
llama-server, configured by `~/.config/veld/embedder.env`:

```sh
EMBEDDER_MODEL_PATH=/path/to/nomic-embed-text-v1.5.Q4_K_M.gguf
EMBEDDER_ALIAS=text-embedding-nomic-embed-text-v1.5   # = VELD_EMBEDDING_API_MODEL
EMBEDDER_WINDOW=2048     # the model's trained window; 8192 for bge-m3
EMBEDDER_POOLING=mean    # cls for bge-m3
```

The script sizes llama-server for embedding: `-c` is the window times the slots, because
llama-server divides `-c` among its slots, and `-b`/`-ub` are the window, because an embedding
input must fit in one micro-batch. With llama-server's defaults any input over 512 tokens is
refused. `packaging/macos/net.portll.veld-embedder.plist` runs it under launchd; then set
`VELD_EMBEDDING_API_URL=http://127.0.0.1:8095`.

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

## Readiness while a tenant opens

A tenant's stores open on its first request, not at boot, and a large one takes a
while: 11.7 s for the live server's largest tenant on 2026-10-19, and 10 min for the
same tenant on a redb replica that had to rebuild its Vamana index. Requests for that
tenant wait for the open. The remember, recall, upsert, list and memory CRUD routes
and the streaming websockets wait on the blocking pool, so `/health`, `/health/live`
and requests for other tenants keep answering while it runs.

`GET /health/ready` answers `503 not_ready` with `reason: "tenant_opening"` while any
request is waiting on an open, and reports how many in `tenant_open_waiters`. It goes
back to `200 ready` when the open finishes. A hard write latch (below) takes precedence
in `reason`.

## Disk full: the write latch

RocksDB stops accepting writes after a background IO error. For "no space left on
device" it never clears that by itself, so writes keep failing after the volume has
space back. Measured 2026-10-11: 11,088 consecutive failed writes against a 221 MB
store on a volume with 253 GiB free, cleared only by restarting the process, while
`/health` reported `healthy` the whole time.

Veld now notices and says so:

| Surface | Latched (soft) | Latched (hard) |
|---|---|---|
| `GET /health` | `status: "degraded"` plus a `write_latch` array, one entry per latched store (`store`, `condition`, `since`, `failed_writes`, `recovery_attempts`, `hard`) | same, with `"hard": true` |
| `GET /health/ready` | `200 ready` — recovery is still in progress | `503 not_ready`, `reason: "storage_write_latched"`, `hard_latched_stores: <n>` |
| `GET /health/live` | `200 alive` | `200 alive` — the process is alive either way |
| Write API calls | — | `503 STORAGE_WRITE_LATCHED` instead of `500 INTERNAL_ERROR` |

**The latch is per store, not per process.** Veld runs one RocksDB per tenant
(`base/<user_id>`) and RocksDB's own latch is per database, so a write landing on
one tenant's store says nothing about another's. One latched store out of twelve is
`degraded`, and the array names which.

**`/health` and `/health/ready` are unauthenticated** (`PROBE_PATHS`), so they
carry no store paths. Each entry names its store by `store` — the first 16 hex
characters of the path's SHA-256, stable across responses and restarts so you can
tell two latched stores apart and line them up with the log. The path itself is a
per-user directory and would disclose both your filesystem layout and a tenant id;
the RocksDB message would additionally disclose the failing file and the memory's
UUID. Both go to the WARN line raised when a store latches and the ERROR line
raised when it goes hard — `~/Library/Logs/veld/server.log` on macOS.

On a latched write veld checks free space on that store's filesystem. If space is
back it flushes and retries the write once; some RocksDB write errors are soft and
clear on a successful retry, and that retry is the unlatch. If the retry also fails
the latch is **hard** for that store, logged once at ERROR with the remedy, and
reported as above. Attempts and cooldowns are counted per store.

**A hard latch needs a process restart, and veld cannot fix it in-process.**
RocksDB's error handler bypasses its retryable-IO auto-resume path for the NoSpace
subcode (`db/error_handler.cc:446`), leaving `SstFileManager::StartErrorRecovery()`
as the only route to auto-recovery — and that requires an `SstFileManager` attached
to the options, for which the C API has no setter (`db/error_handler.cc:541`).
`DB::Resume()` is not exposed by rust-rocksdb 0.24. Reopening the database in the
same process cannot work either: RocksDB holds a directory lock and live handles are
still outstanding.

| Variable | Default | Effect |
|---|---|---|
| `VELD_WRITE_LATCH_MIN_FREE_BYTES` | `268435456` (256 MiB) | Below this, the disk is treated as genuinely full and the write is not retried |
| `VELD_WRITE_LATCH_MAX_ATTEMPTS` | `3` | Flush+retry attempts per latch |
| `VELD_WRITE_LATCH_COOLDOWN_SECS` | `30` | Minimum gap between recovery probes, so a latched store under load does not spend every request on a doomed flush |
| `VELD_WRITE_LATCH_RESTART` | off | `1` makes a hard latch with space back exit `75` (`EX_TEMPFAIL`) so a supervisor restarts into a reopened store. Off by default — exiting is the operator's choice |

Set `VELD_WRITE_LATCH_RESTART=1` only where something restarts the process: a
launchd `KeepAlive`, a systemd `Restart=on-failure`, or a container orchestrator.

## See also

- [Configuration reference](../reference/config.md) — env vars
- [Multi-tenant](multi-tenant.md) — when one veld serves many users
- [Zenoh / robotics](zenoh-robotics.md) — multi-node edge deployments
