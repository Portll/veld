# Veld — Agentic Memory: Project Instructions

> Product: **Veld — Agentic Memory**. Crate / package IDs remain `veld`.
> Repository: https://github.com/Portll/veld • License: Veld Evaluation Licence (PolyForm Internal Use by agreement; AGPL-3.0-or-later intended at 2.0 — see LICENSING.md) • Cargo version: see `Cargo.toml` (`0.7.x` line at time of writing).
> Branch status (per [PROGRESS.md](PROGRESS.md)): first packaged release cut (repo moved to `Portll/veld`, CI added); the `0.7.x` line on `main` is stabilizing toward a clean `v0.8`. `v0.9` is the public-release target; redb stays a second engine until the gates in [ADR 0007](docs/src/decisions/0007-rocksdb-default-engine.md) hold.

---

## Code Standards

- Production-grade code only. **No** TODOs, placeholders, mocks, or stubs in committed code.
- Take a bird's-eye view before changing anything: understand architecture, data flow, and the affected scoring/retrieval layers before fixing.
- **Do not** run `cargo build` or trunk-serve. The user keeps these running in the background. Tests run locally under the tiers in **Build & Test Discipline** below.
- On **macOS**, prefer `./scripts/cargo-dev.sh ...` so Cargo inherits the local `libclang` workaround needed by the current RocksDB build path. (See `scripts/setup-libclang-env.sh`.)
- **No git-commit signatures** — no "Generated with Claude Code", no "Co-Authored-By" lines. Clean commit messages only.
- Maintain the per-user lock pattern (see [src/handlers/consolidation.rs](src/handlers/consolidation.rs) `CONSOLIDATION_LOCKS`) for any new write path that mutates shared per-user state.
- When you must serialize a new field on `Memory`, make it backwards-compatible: `Option<T>` with `#[serde(default)]`, or wire a migration in [src/storage/](src/storage/).

---

## You Have Persistent Memory

You are running with Veld, a persistent cognitive memory system. This is not a tool you query — it is part of how you think.

### How it works

**Automatic Layer (invisible to you):**
- When a session starts, relevant memories from past sessions are surfaced.
- When the user asks something, related memories are retrieved and injected into your context.
- When you finish responding, important context is encoded into memory.
- When a command fails or a subagent finishes, the outcome is recorded. Routine edits and reads are not.

Context is surfaced before a tool runs and returned to the model as `additionalContext`; failures and subagent completions are recorded after. You don't see the API calls — you just receive the context. Implementation: [hooks/memory-hook.ts](hooks/memory-hook.ts).

**Emphasis Layer (explicit tools):**
- `remember` — mark something HIGH IMPORTANCE (preserved longer, resists decay).
- `forget` — delete a memory by id. It is a hard delete, journaled, so replicas drop it too.
- `remember` is an emphasis signal, not an on/off switch: memory happens anyway.

### What this means for you

1. **You have continuity.** Past sessions exist. Decisions, learnings, patterns persist. You are not starting fresh.
2. **Don't "check" memory.** Relevant context appears automatically. Trust the system.
3. **Use `remember` sparingly** — only for genuinely important things. Most is captured automatically.
4. **Use `forget` for corrections.** If you stored something wrong, forget it.
5. **You know this user.** Their preferences, codebase, and working style are in your context.

---

## MCP / Memory Tools

The TypeScript MCP server ([mcp-server/index.ts](mcp-server/index.ts), published as `@veld/memory-mcp`) exposes **46 tools** over the HTTP API. The Rust binary `veld serve` ([src/cli.rs](src/cli.rs), using `rmcp` from [src/mcp/mod.rs](src/mcp/mod.rs)) also serves an MCP stdio transport.

**CLI subcommands** (from [src/cli.rs](src/cli.rs)):

| Command | Purpose |
|---|---|
| `veld server` | Start the HTTP API server (defaults: `127.0.0.1:3030`, `rocksdb` backend when the build has it; see Storage) |
| `veld tui` | Launch the terminal dashboard |
| `veld serve` | Run as MCP server (stdio transport) — note: *not* the HTTP daemon |
| `veld init` | First-time setup — config, API key, ONNX runtime |
| `veld status` | Check server health |
| `veld doctor` | Diagnose common issues (storage, ONNX, port) |
| `veld restart [--port N]` | Stop a server gracefully and bring it back as the monitor's Restart does: its launchd job or supervisor, else a relaunch from its launch recipe |
| `veld hook session-start \| prompt \| commit` | Output Claude Code hook JSON |
| `veld claude [args...]` | Launch Claude Code with the Veld proxy wired up |
| `veld version` | Print version and build info | Representative tools:

| Category | Tools |
|---|---|
| Core memory | `remember`, `recall`, `forget`, `read_memory`, `list_memories`, `memory_stats`, `memory_health` |
| Proactive context | `proactive_context`, `context_summary`, `quick_recall`, `query`, `topic`, `what_i_know`, `recent_memories`, `count`, `session_summary` |
| Index & backups | `verify_index`, `repair_index`, `backup_create`, `backup_list`, `backup_verify`, `backup_purge`, `backup_restore` |
| Consolidation | `consolidation_report` |
| Token budget | `token_status`, `reset_token_session` |
| Reminders | `set_reminder`, `list_reminders`, `dismiss_reminder` |
| Todos | `add_todo`, `list_todos`, `update_todo`, `complete_todo`, `delete_todo`, `reorder_todo`, `todo_stats`, `list_subtasks`, `pending_work` |
| Todo comments | `add_todo_comment`, `list_todo_comments`, `update_todo_comment`, `delete_todo_comment` |
| Projects | `add_project`, `list_projects`, `archive_project`, `delete_project` |
| Seeding | `seed_project` |

The HTTP API beneath the MCP tools is larger (~202 routes registered in [src/handlers/router.rs](src/handlers/router.rs)). Endpoints not exposed via MCP include direct CRUD (`/api/memory/{id}` PUT/DELETE), entity resolution (`/api/entity/*`), prompt generation (`/api/prompt/gen`), context blocks (`/api/context/blocks`), tier moves (`/api/memory/tier`), anchor (`/api/anchor` — decay resistance), sleep-phase consolidation (`/api/consolidation/sleep`), external-dimension push (`/api/sleight/dimensions`), and webhooks (`/webhook/linear`, `/webhook/github`).

For work that spans sessions, check `list_todos` at session start.

---

## Architecture

Veld is an **edge-native cognitive memory system**. Single binary, runs offline, no external databases, no API keys. Optimised for resource-constrained devices (Zenoh/ROS2-native for robotics).

### Retrieval (the actual hot path)

The retrieval pipeline is **multi-layer hybrid search**, not a single algorithm. Layers (numbering follows the codebase's "Layer 3.5 / 4.527 / 5.85" convention from [PROGRESS.md](PROGRESS.md) — non-integer numbers are real, not typos):

- **Vector search**: HNSW (default), Vamana/DiskANN, SPANN, and PQ codecs all live in [src/vector_db/](src/vector_db/). HNSW is the active retrieval engine.
- **BM25 keyword search** via Tantivy (full-text index).
- **Graph spreading activation** over Hebbian edges in [src/memory/graph_retrieval.rs](src/memory/graph_retrieval.rs).
- **Cross-encoder reranking** ([src/embeddings/cross_encoder.rs](src/embeddings/cross_encoder.rs)) at ~18% blend, top-20 budget.
- **Dual-embedder competition**: MiniLM 384d + Nomic 768d via LM Studio / Ollama / vLLM HTTP API (auto-detect). `CompetitiveEmbedder` uses `Arc<dyn Embedder>`.
- **20-signal scoring** ([src/memory/types.rs](src/memory/types.rs) `ScoringSignals`): base score, recency, arousal, source credibility, temporal match, session boost, access count, graph strength, calibrated confidence (Bayesian α/β), confidence observations, feedback momentum (EMA), cross-encoder, importance, entity match, tag match, episode coherence, source-type multiplier, emotional valence intensity, sequence proximity, activation level, temporal fact density, entity density.
- **External-dimension modulation** from Spine's evaluation engine via `ExternalDimensionScores` (density, coherence, closure, confidence, isotropy) — modulates retrieval rank when fresh.
- **Signal attribution** per memory ([src/memory/types.rs](src/memory/types.rs) `SignalAttribution`): which signals contributed to a memory's rank. Drives adaptive weight learning.

### Storage

Storage abstraction layer: [src/storage/mod.rs](src/storage/mod.rs) defines `PrimaryMemoryStore`, `GraphStore`, `KeyValueStore` traits with `StorageCapabilities` per backend.

- **Two engines, both built by default** (`default = ["rocksdb", "storage-redb"]` in [Cargo.toml](Cargo.toml)). **RocksDB is the default engine** ([ADR 0007](docs/src/decisions/0007-rocksdb-default-engine.md)); **redb** (single-file embedded) is selected at runtime with `VELD_STORAGE_BACKEND=redb` (or `--storage-backend redb`). `legacy-rocksdb` is an alias of the `rocksdb` feature.
- **Redb-only build**: `--no-default-features --features storage-redb` carries no RocksDB code; a request for the rocksdb backend then refuses at boot and names the feature.
- **Selection logic**: `default_requested_storage_backend()` returns RocksDb whenever the `rocksdb` feature is on, and `effective_storage_backend_for_current_build(Redb)` returns Redb when `storage-redb` is compiled in ([src/config.rs](src/config.rs)). Engine implementations: [src/storage/legacy_rocksdb.rs](src/storage/legacy_rocksdb.rs) and [src/storage/redb.rs](src/storage/redb.rs).
- **Canonical reference**: [src/storage/README.md](src/storage/README.md) (file map, engine layout, `veld migrate`, what only RocksDB does).
- **Default data path**:
  - Linux: `~/.local/share/veld/`
  - macOS: `~/Library/Application Support/veld/`
  - Windows: `%APPDATA%\veld\`
  - Falls back to `./veld_data/` if a legacy directory already exists in the CWD (versions ≤ 0.1.80 compatibility).

### Memory model

- **Tiered memory** ([src/memory/types.rs](src/memory/types.rs) `MemoryTier`): `Working` → `Session` → `LongTerm` → `Archive`. Tier promotion driven by age × importance × access. Agent-directed moves via `/api/memory/tier`.
- **Anchor**: agent-pinned memories resist decay (`/api/anchor`).
- **Facets** ([src/memory/facets.rs](src/memory/facets.rs)): `Who`, `What`, `When`, `Where`, `Why`, plus `RecordKind`, `ContentKind`, `CausalLink`, `EngramBinding`, `Prediction`, `AgentRef`, `Place`. The page-type taxonomy needed for a wiki view already exists here.
- **Context blocks** ([src/memory/context_blocks.rs](src/memory/context_blocks.rs)): Letta-style mutable agent state (`key`, `content`, `max_tokens`, `version`). Distinct from memories — these are persistently editable.
- **Prospective memory**: `ExperienceType::Intention` for future reminders; filtered from normal recall, surfaces via dedicated reminder queries or spreading activation.
- **Calibrated confidence**: Bayesian α/β per memory; retrieval gate at 0.85–1.0.
- **Decay**: multi-time-scale ([src/decay.rs](src/decay.rs), [src/decay_scales.rs](src/decay_scales.rs)).

### Consolidation

- **Background pipeline** ([src/handlers/consolidation.rs](src/handlers/consolidation.rs) `consolidate_memories`): fact extraction → maintenance (replay + tier consolidation + decay) → graph strengthening. Spawned as detached `tokio::task` (survives HTTP timeout). Guarded by per-user `CONSOLIDATION_LOCKS`.
- **Sleep-phase consolidation**: `/api/consolidation/sleep` — deeper replay pass.
- **Gap topology** ([src/memory/gap_topology/](src/memory/gap_topology/)): detector, scoring, Voronoi decomposition. Surfaces knowledge gaps in the graph.
- **Session detection** ([src/memory/wavelet_sessions.rs](src/memory/wavelet_sessions.rs)): gap-based segmentation — a fixed 2h inter-memory gap threshold (`detect_sessions`) plus a partition-theoretic adaptive threshold (`detect_sessions_adaptive`: largest relative jump in the sorted gap distribution, with a Hardy-Ramanujan `(1 − 1/24)` modular correction, clamped to `[300s, 28800s]`). **No wavelet transform despite the module name** — the file name is legacy. Powers ordinal-session resolution in recall ("two sessions ago").

### Ingest

- **Multi-format text extraction** ([src/ingest/](src/ingest/)): generic extractors + Google Drive + GitHub. PDF support behind the `pdf` feature flag.
- **Project seeding** ([src/handlers/seed.rs](src/handlers/seed.rs)): cold-start ingestion of a project's contents into the memory store.
- **Webhook ingest**: Linear and GitHub webhooks (authenticated + rate-limited).

### Event log

- **Intent log** ([src/intent_log/](src/intent_log/)): event-sourced journal with typed `IntentPayload` (bincode encoded). Append-only. Recent work (commit `3173a16`): typed payload with bincode encode/decode.

### Other subsystems

- **MIF** ([src/mif/](src/mif/)): Memory Interchange Format — import/export adapters for Mem0, Veld native, markdown, generic. With PII handling.
- **Knowledge graph**: `EntityNode`, `RelationshipEdge`, `EpisodicNode` ([src/graph_memory.rs](src/graph_memory.rs)). Hebbian edge strengthening via `strengthen_memory_edges` during consolidation.
- **Embeddings cache**: HTTP backend with LRU; optional Zenoh-shared cache when the `zenoh` feature is on.
- **Compression pipeline** ([src/memory/compression.rs](src/memory/compression.rs)): semantic consolidation into facts; size-based compression.
- **Streaming ingest** ([src/streaming.rs](src/streaming.rs), 1577 lines): WebSocket `/api/stream` for high-throughput memory ingestion. Distinct from the SSE endpoints in `handlers/webhooks.rs` (`/api/context/sse`, `/api/events/sse`).
- **Metrics** ([src/metrics.rs](src/metrics.rs), 989 lines): Prometheus metric registrations (`veld_*` prefix). `/metrics` route in [src/handlers/router.rs](src/handlers/router.rs); visibility controlled by `VELD_METRICS_PUBLIC`.
- **Visualization / graph viewer** ([src/handlers/visualization.rs](src/handlers/visualization.rs), 405 lines + [src/handlers/graph_view.html](src/handlers/graph_view.html)): in-browser knowledge-graph viewer. Routes: `GET /graph/view` (public HTML shell), `GET /api/brain/{user_id}`, `/api/visualization/*`.
- **LLM-Wiki** ([src/wiki/](src/wiki/)): wiki render as an **intent-log projection** (`WikiProjection`), exposed via authed `` `POST /api/wiki/render` `` ([src/handlers/router.rs:272](src/handlers/router.rs#L272)). Bidirectional **edit-back**: Obsidian edits flow back to the canonical store, with an optional default-off live watcher ([src/wiki/watcher.rs](src/wiki/watcher.rs)). Consumes the facet page-type taxonomy in [src/memory/facets.rs](src/memory/facets.rs). Now shipped — was the "LLM-Wiki Phase 3" plan.
- **Veld GUI / monitor** ([monitor/](monitor/), Tauri desktop app): first-run setup panel, multi-instance discovery + tabs, runtime **dials** panel (`get_dials`/`set_dials`), the server graph viewer embedded in-window, a **Security** overlay (keys, logins) and a **Data** overlay ([monitor/src/data.js](monitor/src/data.js)): backups (`/api/backup/*`), MIF export/import to and from files, and a backend swap (dry-run + `/api/admin/storage/migrate`, then `storage_backend` written to `config.toml` for the next start). **Restart server** (Data › Backend, and the tray menu) stops the instance; a server that is a launchd job's main process is started again with `launchctl kickstart` (never `-k`, which kills it after the job's exit timeout, mid-flush); otherwise it waits for a supervisor to bring it back (a new registry entry on the port), else relaunches it from a saved recipe or the registry's `launch` recipe, layering the operator's `~/.config/veld/server.env`, the live process's `VELD_*` environment and the recipe's variables ([status-core/src/restart.rs](status-core/src/restart.rs)). Saved recipes live in `<config>/veld/monitor/launch-recipes.json`. On this machine Veld Desktop supervises the server and respawns it 2 s after exit; its `server.env` pins `VELD_STORAGE_BACKEND`, which beats `config.toml`, so "Set next start" rewrites both. Companion crates: [status-core/](status-core/), [status-tui/](status-tui/), [status-widget/](status-widget/).
- **Live replication** ([src/replication.rs](src/replication.rs), routes in [src/handlers/replication.rs](src/handlers/replication.rs)): a follower tails a leader's per-user intent logs over `` `GET /api/admin/replication/records` `` (admin key + API key) and applies each frame to its store, then journals it locally; one direction, memories only (todos, reminders, projects are not journaled). Follow at boot with `VELD_REPLICATE_FROM` + `VELD_REPLICATE_ADMIN_KEY`, or at runtime via `` `POST /api/admin/replication/follow` `` (the monitor's Data › Replication view). Cursor per source under `<data>/replication/`. Verified 2026-10-19 with two scratch servers: ids preserved, forgets propagate, lag one poll interval.
- **Instance registry** ([src/instance_registry.rs](src/instance_registry.rs)): **server-authored** registry of live veld instances (replaces port-guessing). Consumed by the monitor, the statusline hook, and Claude session wiring. See [docs/src/architecture/instance-registry.md](docs/src/architecture/instance-registry.md).
- **Epistemic hygiene / injection boundary**: source-trust is surfaced at recall, and `injection_suspected` is flagged for non-trusted-source content (W-006). See [docs/src/architecture/epistemic-hygiene.md](docs/src/architecture/epistemic-hygiene.md).
- **Store↔log durability**: reconcile-on-startup, LSN ordering, and bulk-forget journaling keep the primary store and the intent log consistent; a per-user write-path serialization lock (W-005) generalizes the `CONSOLIDATION_LOCKS` pattern to all write paths.
- **Auth**: API-key middleware ([src/auth.rs](src/auth.rs)) plus Phase C user-auth (password + TOTP + recovery codes) at `/api/user_auth/*`.
- **Multi-tenant** ([src/extensions/](src/extensions/), feature `multi-tenant`): hosaka collective-store, PII policy, maintenance.
- **Fortress** ([src/fortress/](src/fortress/), feature `fortress`): fractal binary obfuscation, anti-debug, integrity checks for distribution builds.
- **Zenoh transport** ([src/zenoh_transport/](src/zenoh_transport/), feature `zenoh`): ROS2/robotics-native pub/sub.
- **Telemetry** (feature `telemetry`): distributed tracing for cloud/multi-node deployments.

### Alignment (recent work)

Cross-embedder alignment, now on a **v2 engine**: whitened-Procrustes + CSLS (beyond the original Procrustes + Ridge fitters), with payload hardening, conditioning, and perf-flag config. Fit/eval bins, retrieval integration. Binaries: `alignment-collect`, `alignment-fit`, `alignment-eval` (see [src/bin/](src/bin/)). Evaluations at [evaluations/alignment/](evaluations/alignment/).

---

## Codebase Layout

> **Canonical exhaustive module list:** [docs/src/architecture/module-index.md](docs/src/architecture/module-index.md) — auto-generated from `src/lib.rs` by `docs/generators/src/bin/gen-module-index.rs`. The table below is a curated "where to look first" map, **not** a complete inventory. If you're building an inventory or comparing veld to a fork, walk the module index, not this table.

| Path | Purpose |
|---|---|
| [src/](src/) | Rust core — memory system, HTTP server, embeddings, graph, vector DB |
| [src/cli.rs](src/cli.rs) | Main `veld` binary entrypoint (`serve` + `hook` subcommands) |
| [src/bin/](src/bin/) | Alignment tooling binaries |
| [src/mcp/](src/mcp/) | Rust MCP server (rmcp) — alternative path to the HTTP API |
| [src/handlers/](src/handlers/) | HTTP route handlers; one module per domain |
| [src/handlers/router.rs](src/handlers/router.rs) | Route table — single source of truth for endpoints |
| [src/handlers/state.rs](src/handlers/state.rs) | `MultiUserMemoryManager` — per-user state, locks |
| [src/memory/](src/memory/) | Memory core — types, retrieval, graph, consolidation, gap topology |
| [src/storage/](src/storage/) | Storage trait abstraction + Redb + legacy RocksDB |
| [src/vector_db/](src/vector_db/) | HNSW, Vamana, SPANN, PQ codecs |
| [src/embeddings/](src/embeddings/) | Embedder traits, cross-encoder, NER, keywords, chunking |
| [src/intent_log/](src/intent_log/) | Event-sourced journal with typed bincode payloads |
| [src/mif/](src/mif/) | Memory Interchange Format — import/export adapters |
| [src/ingest/](src/ingest/) | Multi-format text extraction; GitHub / Google Drive ingestors |
| [src/config.rs](src/config.rs) | Configuration; storage path resolution; backend selection |
| [src/auth.rs](src/auth.rs) / [src/user_auth/](src/user_auth/) | API-key middleware + Phase C user auth (TOTP) |
| [src/extensions/](src/extensions/) | Multi-tenant feature gated extensions |
| [src/fortress/](src/fortress/) | Fortress feature — obfuscation, anti-debug |
| [src/zenoh_transport/](src/zenoh_transport/) | Zenoh feature — robotics pub/sub |
| [mcp-server/](mcp-server/) | TypeScript MCP server (`@veld/memory-mcp`) — wraps the HTTP API for Claude/Cursor/Copilot |
| [tui/](tui/) | Rust TUI dashboard |
| [monitor/](monitor/) | Tauri desktop GUI (Veld GUI) — multi-instance dashboard, dials panel, embedded graph viewer |
| [src/wiki/](src/wiki/) | LLM-Wiki — wiki render as an intent-log projection + Obsidian edit-back |
| [src/instance_registry.rs](src/instance_registry.rs) | Server-authored live-instance registry (replaces port-guessing) |
| [hooks/](hooks/) | Claude Code hooks — automatic memory pre/post tool |
| [python/](python/) | Python bindings (maturin/PyO3, feature `python`) |
| [evaluations/](evaluations/) | Benchmark + alignment evaluation artifacts |
| [packaging/](packaging/) | Distribution packaging (NSIS installer, etc.) |
| [PROGRESS.md](PROGRESS.md) | Shipped vs in-flight work; the canonical "what is real" doc |
| [BENCHMARKS.md](BENCHMARKS.md) | Benchmark results |
| [SECURITY.md](SECURITY.md) | Security policy |

---

## Build & Test Discipline

- **Don't** run `cargo build`, `trunk serve`, `cargo run`, release builds, executed benches, or anything that binds :3030 or :3031. The user runs these, and the live servers on this machine serve every session.
- **Tests run in three tiers** (operator-approved 2026-10-22):

  | Tier | What | When |
  |---|---|---|
  | 0 | `cargo check`, `cargo clippy`, `cargo fmt --check`, `cargo test --no-run`, `cargo bench --no-run` | any time |
  | 1 | `scripts/test-local.sh <selectors>`: the unit tests of the modules a change touches, plus `--test handler_tests`, as separate calls (a test-name filter applies to every target in the call) | before every commit |
  | 2 | `scripts/ci-local.sh`: CI's version and ledger checks, then `scripts/test-local.sh --full`: fmt, clippy (default, redb-only, zenoh), and on default, redb-only and rocksdb-only `cargo nextest run --profile ci`, the doctests and a bench build | push gate only; one at a time per machine |

  Run tests through [scripts/test-local.sh](scripts/test-local.sh), not bare `cargo test`. It runs them niced with `RUST_TEST_THREADS=4` and a private target dir. It gives them a throwaway `HOME`, sharing only the model cache, and strips every inherited `VELD_*`/`SHODH_*` variable, so a test that forgets its own storage path cannot reach a live tenant. Tier 2 holds a machine-wide lock: full matrices run side by side pushed the load to 37 on 2026-10-21 and made tests flake. Tier 2 skips only the tests listed in [scripts/test-local.known-red](scripts/test-local.known-red), each with an owner and a task.
- **Push gate**: a push to `main`, a `v*` tag or a branch with an open PR, and `gh pr create`, need a green tier 2 for the pushed tree. spine's `bin/ci-verdict.mjs --repo <worktree> --rev <sha>` runs `scripts/ci-local.sh` on a clean checkout of that commit and records the verdict in `.claude/store/ci-verdicts.jsonl`; the `spine-push-gate.mjs` hook refuses the push without a green one. The override, `CI_LOCAL_SKIP='<reason>' git push …`, is recorded in `.claude/store/push-gate.jsonl` and is the visible exception, not a habit. GitHub CI stays the authority for what is on origin.
- **On macOS**, prefix Cargo calls with `./scripts/cargo-dev.sh` so libclang/RocksDB build path picks up the env shim.
- **Local install builds** use `cargo build --profile local --bin veld` in a private `CARGO_TARGET_DIR` (no LTO, incremental codegen); `--release` (fat LTO, one unit) is for tagged releases. The Python cdylib is the `veld-py` crate, built only by maturin; `build.rs` reruns only when git HEAD moves, so a no-change rebuild is a no-op.
- **Per-user locking**: any new write handler that mutates shared per-user state should follow the `CONSOLIDATION_LOCKS` pattern in [src/handlers/consolidation.rs](src/handlers/consolidation.rs):

  ```rust
  static FOO_LOCKS: std::sync::LazyLock<
      dashmap::DashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>,
  > = std::sync::LazyLock::new(dashmap::DashMap::new);
  ```

- **Background work**: long-running handlers spawn a detached `tokio::task` and return 202 Accepted. This avoids the 60s HTTP timeout killing pipelines mid-flight.
- **Storage migrations**: when adding a `Memory` field, deserialize with `#[serde(default)]`; existing on-disk records must round-trip.
- **Public route safety**: only public-router probes (`/health/*`) may live without auth. The `public_router_has_no_per_user_handlers` test enforces no public handler reads `?user_id=`. Do not bypass it.

---

## Storage Backend Reality (read before touching storage)

Both engines are real and built by default. The canonical description is [src/storage/README.md](src/storage/README.md); read it first. Specifically:

- `default_requested_storage_backend()` returns `RocksDb` whenever the `rocksdb` feature is on ([src/config.rs](src/config.rs)); redb only when that feature is off.
- `effective_storage_backend_for_current_build(Redb)` returns `Redb` when `storage-redb` is compiled in. A rocksdb request in a build without the `rocksdb` feature is refused at boot, by name.
- Boot guard ([src/storage/boot_guard.rs](src/storage/boot_guard.rs)): veld refuses to serve an empty store while the other engine holds data for that user, and names `VELD_STORAGE_BACKEND=redb` or `veld migrate` in the refusal.
- Only RocksDB has the BackupEngine ([src/backup.rs](src/backup.rs)) and the RocksDB `MemoryStorage` ([src/memory/storage.rs](src/memory/storage.rs)). redb backups go through export and import (VELDPORT dumps). Encryption-v2 and the semantic merge pass run on both engines.
- redb becomes the default only when the gates in [ADR 0007](docs/src/decisions/0007-rocksdb-default-engine.md) hold. Closed: merge pass and tombstone GC on redb, encryption-v2 on redb. Open: backup/restore equality, recall-gate equality on both engines, and a 14-day live soak on redb with no boot-guard refusal and no decode failure.

When writing new storage code: target the trait surface (`PrimaryMemoryStore`, `GraphStore`, `KeyValueStore`) in [src/storage/mod.rs](src/storage/mod.rs), not the legacy concrete types. A capability added for RocksDB needs a redb implementation or a named refusal (ADR 0007).

---

## Client Integration

Veld hooks are configured for three clients:

### VS Code Copilot Agent
- `.vscode/mcp.json` — registers Veld MCP server (46 tools).
- `.github/copilot-instructions.md` — workspace instructions with tool/skill switches.
- User ID: `vscode-copilot`.

### Claude Code
- `.claude/settings.json` — hooks + MCP server config (relative paths, no env vars needed).
- [hooks/memory-hook.ts](hooks/memory-hook.ts) — implementation (all 6 lifecycle events, all tool types).
- User ID: `claude-code`.

### Claude CLI
- Same `.claude/settings.json` — Claude CLI reads project settings from this path.
- Same hooks fire for CLI usage.
- User ID: `claude-code` (shared with Claude Code for memory continuity).

### Standalone Hook Config (for external projects)
- [hooks/claude-settings.json](hooks/claude-settings.json) — portable config using `$VELD_HOOKS_DIR` env var.
- Copy to `~/.claude/settings.json` or project `.claude/settings.json`.
- Set `VELD_HOOKS_DIR` to point to veld's `hooks/` directory.

---

## Encoding Conventions

How to use the memory tools well — what to `remember`, what to leave to the automatic hook layer, and how to tag.

### When to call `remember` explicitly

The hook layer records command failures and subagent completions automatically, and surfaces context at session start, on each prompt and before tools. Reach for explicit `remember(content, importance)` only when:

- A **decision** was made that future sessions need to honour (e.g., "we chose Redb over RocksDB because …" — see [decision 0001](docs/src/decisions/0001-redb-migration.md)).
- A **user preference** was stated (e.g., "always run cargo check, never cargo build").
- A **load-bearing invariant** was discovered (e.g., "the `public_router_has_no_per_user_handlers` test is the security contract").
- A **non-obvious cause-of-failure** was diagnosed and would otherwise be re-discovered.

Do not `remember` routine tool patterns, file paths, or anything visible from `git log`/`git blame`. The hook layer already covers those.

### Importance band

| Importance | Use for |
|---|---|
| `0.9–1.0` | Anchorable invariants (won't decay; consume sparingly — these compete for the top of recall) |
| `0.7–0.8` | Decisions, user preferences, security contracts |
| `0.4–0.6` | Diagnostic notes, useful-to-have context |
| ≤ `0.3` | Don't bother — leave it to the hook layer |

### Tag taxonomy

Tags are the cheap "what kind of thing" signal (scoring signal 15). Keep them short and reusable. Veld doesn't enforce a taxonomy, but converging on a small set makes tag-match scoring effective:

- **Domain**: `auth`, `storage`, `retrieval`, `consolidation`, `ingest`, `docs`, `tui`, `ci`, `release`
- **Kind**: `decision`, `bug`, `preference`, `invariant`, `gotcha`, `todo`
- **Status**: `wip`, `blocked`, `done` (mostly for todos)

A memory typically carries one domain + one kind tag (`["storage", "decision"]`); add more only if you'll search by them.

Veld adds derived tags of its own (NER, YAKE, ingest hints, cold-start extraction). After redaction it drops any derived tag without a letter, under two characters, or carrying code punctuation, then keeps the rarest by graph mention count up to `DERIVED_TAG_BUDGET` ([src/entity_cap.rs](src/entity_cap.rs)). The default budget is 50, the record-wide limit, so it cuts nothing: on the recall gate a budget of 12 lost eleven gold documents and 24 lost three (PR #88). `VELD_DERIVED_TAG_BUDGET` lowers it for a store that wants to measure. Authored tags are never counted against it.

### Citation format

Reference code locations as `[path/to/file.rs:42](src/path/to/file.rs#L42)`. Reference decisions as `[0001-redb-migration.md](docs/src/decisions/0001-redb-migration.md)`. Reference HTTP routes as `` `POST /api/remember` `` (backticks, full path). The wiki-render layer (LLM-Wiki Phase 3) consumes this format directly; sticking to it now means no migration cost later.

---

## Scale & Migration

How veld behaves as data grows, and the migration paths between tiers / backends / schemas.

### Retrieval scale (cheap → expensive)

| Memory count | Default retrieval path |
|---|---|
| < 1 000 | Working/Session brute-force cosine (Layer 3.5) covers everything |
| 1 000 – 50 000 | HNSW + BM25 (Tantivy) hybrid kicks in for the LongTerm tier |
| 50 000 – 500 000 | Cross-encoder rerank budget (top-20, 18% blend) is the dominant cost; consider tuning [`CognitiveDials`](docs/src/architecture/dials.md) to reduce rerank K |
| 500 000+ | Vamana / SPANN large-collection indexes available in [src/vector_db/](src/vector_db/); SQLite slow store for gap topology queries |

### Tier promotion (drives what's hot vs cold)

`Working → Session → LongTerm → Archive`. Promotion is `age × importance × access_count` against the `TIER_PROMOTION_*` constants in [src/constants.rs](src/constants.rs). Archive-tier memories stay indexed in HNSW but are excluded from default `/api/recall`; surface them explicitly with `/api/recall?include_archived=true`.

### Storage backend migration (RocksDB and redb, on demand)

`veld migrate` copies one engine's stores into the other inside the same data directory and never modifies the source. Each unit is verified by frame count and SHA-256 digest. It needs a build with both engines (the default build). Try it first with a dry run, which writes nothing:

```bash
veld migrate --from rocksdb --to redb --user all --dry-run
```

`veld export` and `veld import` use the same VELDPORT codec ([src/storage/portable.rs](src/storage/portable.rs)). Migrating an encrypted tenant needs `VELD_MASTER_PASSPHRASE` (or `VELD_KMS_WRAP_KEY`), including for the dry run. The boot guard ([src/storage/boot_guard.rs](src/storage/boot_guard.rs)) refuses to serve an empty store while the other engine holds data, so a forgotten migration fails loudly instead of serving nothing. Back up before migrating: `POST /api/backup/create` covers RocksDB; for redb use `veld export`.

Engine speed is unmeasured in this document. The comparison benches are [benches/storage_engine_benchmarks.rs](benches/storage_engine_benchmarks.rs), being added on the redb-parity branch; do not quote a figure until they have run.

See [decision 0001](docs/src/decisions/0001-redb-migration.md) (amended) and [decision 0007](docs/src/decisions/0007-rocksdb-default-engine.md) for context; src/storage/README.md is canonical.

### Record-schema migration

When you add a field on `Memory`, make it backwards-compatible: `Option<T>` with `#[serde(default)]`. The legacy `bincode1` format support in [Cargo.toml](Cargo.toml) handles pre-2.0 memories. Forward-incompatible changes need an explicit migration in [src/storage/](src/storage/); these are rare and should land paired with a `## Schema Version & CHANGELOG` entry once that section exists (LLM-Wiki Phase 1).

### Distribution-binary scaling

The default release binary is ~30 MB. `--features embedded-docs` adds ~3-5 MB for the mdBook output. `--features fortress` strips tracing and adds obfuscation (smaller binary, harder to debug). `--features telemetry` adds OpenTelemetry exporters (~200 extra crates) — only enable in cloud / multi-node deployments where you want distributed tracing.

---

## Distribution

| Surface | Identifier |
|---|---|
| Rust crate | `veld` on crates.io |
| Node / MCP | `@veld/memory-mcp` on npm |
| Python | `veld` on PyPI |
| Docker | `ghcr.io/portll/veld` |
| MCP Registry | `veld` |
| Cursor Directory | `veld-1` |

---

## Where to find things

- **Live work / status**: [PROGRESS.md](PROGRESS.md) — shipped layers, retrieval benchmarks, in-flight stabilization.
- **Plans (uncommitted, local)**: `~/.claude/plans/` — current items include the LLM-wiki dual-pathway plan (`veld-llm-wiki-dual-pathway-plan.md`) and the docs sidecar plan (`veld-docs-sidecar-plan.md`).
- **Benchmarks**: [BENCHMARKS.md](BENCHMARKS.md), [benchmark_report.json](benchmark_report.json).
- **Security policy**: [SECURITY.md](SECURITY.md).
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- **Alignment work**: [ALIGNMENT_IMPLEMENTATION.md](ALIGNMENT_IMPLEMENTATION.md), [EMBEDDING_ALIGNMENT.md](EMBEDDING_ALIGNMENT.md), [evaluations/alignment/](evaluations/alignment/).
- **Rectification / remediation**: [RECTIFICATION.md](RECTIFICATION.md), [REMEDIATION_PLAN.md](REMEDIATION_PLAN.md).

The memory system you're using IS this codebase. Meta, but useful context.
