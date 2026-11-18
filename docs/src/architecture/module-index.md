<!-- GENERATED FILE — do not edit by hand.
     Source: src/lib.rs
     Generator: docs/generators/src/bin/gen-module-index.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-module-index`) -->

# Module index

Top-level Rust modules in veld. **52** modules (plus **4** feature-gated).

Each entry's summary is the first paragraph of that module's `//!` doc comment. Click into the crate docs (built by `cargo doc`) for full API.

## Always-on

| Module | Summary |
|---|---|
| `ab_testing` | A/B Testing Infrastructure for Relevance Scoring |
| `auth` | — |
| `backup` | P2: Backup & Restore System |
| `config` | Configuration management for Veld |
| `constants` | Documented constants for the memory system |
| `datasets` | Datasets — structured tabular storage with row-level graph links. |
| `decay` | Hybrid Decay Model (SHO-103) |
| `decay_scales` | Fourier-Learned Decay Scales (SHO-FFT) |
| `declared_entities` | Entities a writer declares on a memory, typed, as opposed to entities veld derives from the content. |
| `embeddings` | Embedding infrastructure (model loading, inference, trait definitions). Embedding generation module |
| `entity_cap` | Which entities survive `MAX_ENTITIES_PER_MEMORY`, and why it is not the first fifty. |
| `entity_frequency` | Entity mentions per time bucket, computed from episodes. |
| `entity_hygiene` | Lexical hygiene for graph entities (which are also the tag namespace). |
| `errors` | Enterprise-grade error handling with structured error types and codes Provides detailed error information for debugging and client error … |
| `graph_memory` | Graph Memory System - Inspired by Graphiti |
| `handlers` | HTTP API Handlers - Modular organization of the REST API |
| `ingest` | Multi-format text extraction pipeline |
| `instance_registry` | Instance registry — how a running `veld server` makes itself discoverable. |
| `integrations` | External integrations for syncing data sources to Veld memory |
| `intent_log` | Durable, checksummed append-only intent log (W5 scaffold). |
| `keyring` | The keyring: one keystore holding every factor an operator unlocks with, which opens every other keystore on the server. |
| `keystore` | Encryption-at-rest v2 — envelope keystore (crypto core, phase P1). |
| `legacy_field_encryption` | Read side of the retired field-level encryption, kept only to migrate it. |
| `llm_hosts` | The local LLM hosts veld will talk to, read from `manifests/llm-hosts.json`. |
| `mcp` | Unified veld binary - MCP server + Claude Code hooks |
| `memory` | Veld memory substrate with product semantics (consolidation, importance, decay). Memory substrate for Veld context management |
| `metrics` | Production-grade metrics with Prometheus |
| `middleware` | P1.3: HTTP request tracking middleware for observability |
| `mif` | Memory Interchange Format (MIF) v2 |
| `oblivious` | Oblivious-access seam (encryption-v2 P6). |
| `path_guard` | Path guards for identifiers and directories that come from callers. |
| `pir` | Private Information Retrieval (encryption-v2 P7). |
| `query_parsing` | Modular Query Parsing System |
| `query_planner` | W6 query planner — foundation tier. |
| `rate_limit_governance` | Resettable rate limiting governance — a thin wrapper around `governor::RateLimiter` that: |
| `recall_harness` | Recall Harness |
| `redaction` | Entity redaction — classify credential-shaped strings before they enter the graph. |
| `relevance` | Proactive Memory Surfacing (SHO-29) |
| `replication` | Live replication between veld instances. |
| `server` | Server bootstrap module — starts the Veld HTTP API server. |
| `server_runtime` | Server runtime orchestration API. |
| `similarity` | Vector similarity search for semantic retrieval |
| `storage` | Storage abstraction layer for backend-agnostic persistence. |
| `streaming` | Streaming Memory Ingestion for Implicit Learning |
| `tag_cores` | Stamped tag cores: tags that sit on exactly the same records. |
| `tls` | Native TLS: an optional second listener serving the HTTP API over rustls. |
| `tracing_setup` | P1.6: Distributed tracing with OpenTelemetry (OPTIONAL) |
| `user_auth` | Self-hosted user authentication (Phase C of the auth roll-out). |
| `user_memory` | User memory substrate API. |
| `validation` | Input validation for enterprise security Prevents injection attacks, ensures data integrity, protects against ReDoS |
| `vector_db` | Vector index infrastructure (Vamana, SPANN). Vector database module with pluggable index backends |
| `wiki` | Wiki render projection (LLM-Wiki) — the wiki view as an intent-log projection. |

## Feature-gated

These modules are compiled only when their feature flag is enabled (e.g., `cargo build --features python`).

| Module | Summary |
|---|---|
| `extensions` | — |
| `fortress` | Fortress: fractal binary obfuscation for distribution builds. |
| `python` | — |
| `zenoh_transport` | Zenoh transport layer: infrastructure for routing Veld operations over the Zenoh protocol. Zenoh Transport Layer |

---

*This table is regenerated from `src/lib.rs`. For deeper API reference, see the rustdoc output stitched into the docs site at `/api/`.*
