# Veld — Agentic Memory: Project Instructions

> Product: **Veld — Agentic Memory**. Crate / package IDs remain `veld`.
> Repository: https://github.com/Portll/veld • License: BUSL-1.1 • Cargo version: see `Cargo.toml` (`0.7.x` line at time of writing).
> Branch status (per [PROGRESS.md](PROGRESS.md)): `v0.7.6-unstable` stabilizing toward a clean `v0.8` cut; `v0.9` is the public-release / redb-integration target.

---

## Code Standards

- Production-grade code only. **No** TODOs, placeholders, mocks, or stubs in committed code.
- Take a bird's-eye view before changing anything: understand architecture, data flow, and the affected scoring/retrieval layers before fixing.
- **Do not** run `cargo build` or trunk-serve. The user keeps these running in the background. You may run `cargo check` and `cargo clippy` (compiler frontend only, no binary output).
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
- When you use tools (Edit, Write, Bash, Read, Grep, Glob, etc.) patterns are recorded.

Hooks fire for all tool types. Context is surfaced pre-tool and actions are recorded post-tool. You don't see the API calls — you just receive the context. Implementation: [hooks/memory-hook.ts](hooks/memory-hook.ts).

**Emphasis Layer (explicit tools):**
- `remember` — mark something HIGH IMPORTANCE (preserved longer, resists decay).
- `forget` — suppress a memory (reduce its importance, accelerate decay).
- These are emphasis signals, not on/off switches. Memory happens anyway.

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
| `veld server` | Start the HTTP API server (defaults: `127.0.0.1:3030`, `redb` backend) |
| `veld tui` | Launch the terminal dashboard |
| `veld serve` | Run as MCP server (stdio transport) — note: *not* the HTTP daemon |
| `veld init` | First-time setup — config, API key, ONNX runtime |
| `veld status` | Check server health |
| `veld doctor` | Diagnose common issues (storage, ONNX, port) |
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
- **External-dimension modulation** from Sleight's evaluation engine via `ExternalDimensionScores` (density, coherence, closure, confidence, isotropy) — modulates retrieval rank when fresh.
- **Signal attribution** per memory ([src/memory/types.rs](src/memory/types.rs) `SignalAttribution`): which signals contributed to a memory's rank. Drives adaptive weight learning.

### Storage

Storage abstraction layer: [src/storage/mod.rs](src/storage/mod.rs) defines `PrimaryMemoryStore`, `GraphStore`, `KeyValueStore` traits with `StorageCapabilities` per backend.

- **Default-requested backend**: Redb (single-file embedded). Feature flag: `storage-redb` (default).
- **Effective runtime backend during current build**: still **RocksDB** ([src/storage/legacy_rocksdb.rs](src/storage/legacy_rocksdb.rs)). See `effective_storage_backend_for_current_build` in [src/config.rs](src/config.rs) — the redb runtime lands as part of the v0.9 work.
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
- **Wavelet sessions** ([src/memory/wavelet_sessions.rs](src/memory/wavelet_sessions.rs)): session-segmentation analysis.

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
- **Streaming** ([src/streaming.rs](src/streaming.rs)): SSE for context status (`/api/context/sse`).
- **Auth**: API-key middleware ([src/auth.rs](src/auth.rs)) plus Phase C user-auth (password + TOTP + recovery codes) at `/api/user_auth/*`.
- **Multi-tenant** ([src/extensions/](src/extensions/), feature `multi-tenant`): hosaka collective-store, PII policy, maintenance.
- **Fortress** ([src/fortress/](src/fortress/), feature `fortress`): fractal binary obfuscation, anti-debug, integrity checks for distribution builds.
- **Zenoh transport** ([src/zenoh_transport/](src/zenoh_transport/), feature `zenoh`): ROS2/robotics-native pub/sub.
- **Telemetry** (feature `telemetry`): distributed tracing for cloud/multi-node deployments.

### Alignment (recent work)

Cross-embedder alignment: Procrustes + Ridge fitters, fit/eval bins, retrieval integration. Binaries: `alignment-collect`, `alignment-fit`, `alignment-eval` (see [src/bin/](src/bin/)). Evaluations at [evaluations/alignment/](evaluations/alignment/).

---

## Codebase Layout

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
| [hooks/](hooks/) | Claude Code hooks — automatic memory pre/post tool |
| [python/](python/) | Python bindings (maturin/PyO3, feature `python`) |
| [evaluations/](evaluations/) | Benchmark + alignment evaluation artifacts |
| [packaging/](packaging/) | Distribution packaging (NSIS installer, etc.) |
| [PROGRESS.md](PROGRESS.md) | Shipped vs in-flight work; the canonical "what is real" doc |
| [BENCHMARKS.md](BENCHMARKS.md) | Benchmark results |
| [SECURITY.md](SECURITY.md) | Security policy |

---

## Build & Test Discipline

- **Don't** run `cargo build`, `trunk serve`, `cargo run`, or any binary-producing command. The user runs these.
- **You may** run `cargo check`, `cargo check --all-features`, `cargo clippy`, `cargo clippy --all-features`, and `cargo test --no-run` (compile-only test build).
- **On macOS**, prefix Cargo calls with `./scripts/cargo-dev.sh` so libclang/RocksDB build path picks up the env shim.
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

The codebase contains the *abstraction* for Redb (default-requested), but the *runtime engine* during current builds is still RocksDB. Specifically:

- `default_requested_storage_backend()` → `Redb` in [src/config.rs](src/config.rs:57).
- `effective_storage_backend_for_current_build(Redb)` currently returns `RocksDb` ([src/config.rs](src/config.rs:61)).
- The redb path lands as part of `v0.9` work (per [PROGRESS.md](PROGRESS.md)).

When writing new storage code: target the trait surface (`PrimaryMemoryStore`, `GraphStore`, `KeyValueStore`) in [src/storage/mod.rs](src/storage/mod.rs), not the legacy concrete types. New code lands ready for the redb cutover.

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

## Distribution

| Surface | Identifier |
|---|---|
| Rust crate | `veld` on crates.io |
| Node / MCP | `@veld/memory-mcp` on npm |
| Python | `veld` on PyPI |
| Docker | `varunveld/veld` |
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
