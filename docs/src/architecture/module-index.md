<!-- GENERATED FILE — do not edit by hand.
     Source: src/lib.rs
     Generator: docs/generators/src/bin/gen-module-index.rs
     Regenerate: cd docs/generators && cargo run --bin gen-module-index -->

# Module index

Top-level Rust modules in veld. **33** modules (plus **4** feature-gated).

Each entry's summary is the first paragraph of that module's `//!` doc comment. Click into the crate docs (built by `cargo doc`) for full API.

## Always-on

| Module | Summary |
|---|---|
| `ab_testing` | A/B Testing Infrastructure for Relevance Scoring |
| `auth` | — |
| `backup` | P2: Backup & Restore System |
| `config` | Configuration management for Veld |
| `constants` | Documented constants for the memory system |
| `decay` | Hybrid Decay Model (SHO-103) |
| `decay_scales` | Fourier-Learned Decay Scales (SHO-FFT) |
| `earth` | Transitional Earth substrate API. |
| `embeddings` | Earth layer: embedding infrastructure (model loading, inference, trait definitions). Embedding generation module |
| `encryption` | Field-level encryption for sensitive memory content. |
| `errors` | Enterprise-grade error handling with structured error types and codes Provides detailed error information for debugging and client error … |
| `graph_memory` | Graph Memory System - Inspired by Graphiti |
| `handlers` | HTTP API Handlers - Modular organization of the REST API |
| `ingest` | Multi-format text extraction pipeline |
| `integrations` | External integrations for syncing data sources to Veld memory |
| `intent_log` | Durable, checksummed append-only intent log (W5 scaffold). |
| `mcp` | Unified veld binary - MCP server + Claude Code hooks |
| `memory` | Veld layer: Earth substrate with product semantics (consolidation, importance, decay). Earth substrate for Veld context management |
| `metrics` | Production-grade metrics with Prometheus |
| `middleware` | P1.3: HTTP request tracking middleware for observability |
| `mif` | Memory Interchange Format (MIF) v2 |
| `query_parsing` | Modular Query Parsing System |
| `rate_limit_governance` | Resettable rate limiting governance — a thin wrapper around `governor::RateLimiter` that: |
| `relevance` | Proactive Memory Surfacing (SHO-29) |
| `roots` | Transitional Roots orchestration API. |
| `server` | Server bootstrap module — starts the Veld HTTP API server. |
| `similarity` | Vector similarity search for semantic retrieval |
| `storage` | Storage abstraction layer for backend-agnostic persistence. |
| `streaming` | Streaming Memory Ingestion for Implicit Learning |
| `tracing_setup` | P1.6: Distributed tracing with OpenTelemetry (OPTIONAL) |
| `user_auth` | Self-hosted user authentication (Phase C of the auth roll-out). |
| `validation` | Input validation for enterprise security Prevents injection attacks, ensures data integrity, protects against ReDoS |
| `vector_db` | Earth layer: vector index infrastructure (Vamana, SPANN). Vector database module with pluggable index backends |

## Feature-gated

These modules are compiled only when their feature flag is enabled (e.g., `cargo build --features python`).

| Module | Summary |
|---|---|
| `extensions` | — |
| `fortress` | Fortress: fractal binary obfuscation for distribution builds. |
| `python` | — |
| `zenoh_transport` | Roots layer: transport infrastructure. Zenoh protocol routing for distributed Earth coordination. Zenoh Transport Layer for Roots |

---

*This table is regenerated from `src/lib.rs`. For deeper API reference, see the rustdoc output stitched into the docs site at `/api/`.*
