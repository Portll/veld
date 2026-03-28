//! LOCOMO-style Retrieval Quality Benchmark
//!
//! Evaluates shodh-memory retrieval across 4 LOCOMO query categories:
//! - Single-hop: direct fact retrieval from a single memory
//! - Temporal: time-referenced queries requiring temporal reasoning
//! - Multi-hop: queries requiring combining information from multiple memories
//! - Open-domain: broad, context-dependent queries
//!
//! The corpus simulates a realistic multi-session software project conversation
//! spanning 4 sessions (kickoff, architecture, debugging, review).
//!
//! Metrics: MRR, Recall@5, Precision@5, Absence Violations
//! Reference: mem0 scored 66.9% on the original LOCOMO benchmark.
//!
//! Run with: cargo test locomo_benchmark -- --ignored --nocapture

use std::collections::HashMap;
use std::time::Instant;

use chrono::{Duration, Utc};
use shodh_memory::memory::types::{Experience, ExperienceType, Query};
use shodh_memory::memory::{MemoryConfig, MemoryId, MemorySystem};
use tempfile::TempDir;

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct LocomoMemory {
    content: &'static str,
    experience_type: ExperienceType,
    tags: Vec<&'static str>,
    importance: f32,
    /// Session number (1-4) — used to set created_at timestamps
    session: u8,
}

struct LocomoQuery {
    query: &'static str,
    query_type: &'static str,
    /// Indices into the memory corpus that SHOULD appear in top-5
    expected_memory_indices: Vec<usize>,
    /// Indices that should NOT appear in results (negative test)
    absence_indices: Vec<usize>,
}

struct PerQueryResult {
    query_text: String,
    query_type: String,
    mrr: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    precision_at_5: f32,
    absence_violations: usize,
    latency_ms: u64,
    retrieved_indices: Vec<usize>,
    expected_indices: Vec<usize>,
}

struct TypeSummary {
    query_type: String,
    mrr: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    precision_at_5: f32,
    absence_violations: usize,
    query_count: usize,
    avg_latency_ms: u64,
}

// =============================================================================
// CORPUS: 60 memories across 4 sessions
// =============================================================================

fn locomo_corpus() -> Vec<LocomoMemory> {
    vec![
        // =====================================================================
        // SESSION 1: Project Kickoff (15 memories)
        // =====================================================================

        // 0: Project initiation
        LocomoMemory {
            content: "Project kickoff meeting held on Monday. The project is called Nexus, a \
                      real-time collaboration platform for distributed engineering teams. \
                      Target launch is Q3. The executive sponsor is VP of Engineering Diana Chen.",
            experience_type: ExperienceType::Conversation,
            tags: vec!["kickoff", "nexus", "project"],
            importance: 0.9,
            session: 1,
        },
        // 1: Team assignment
        LocomoMemory {
            content: "Team assignments for Project Nexus: Raj Patel is the project lead and \
                      backend architect. Maria Gonzalez handles frontend and UX. Tomasz Kowalski \
                      is the infrastructure and DevOps engineer. Sarah Kim covers QA and testing.",
            experience_type: ExperienceType::Decision,
            tags: vec!["team", "roles", "nexus"],
            importance: 0.85,
            session: 1,
        },
        // 2: Backend language decision
        LocomoMemory {
            content: "Decided to use Rust for the backend services. Raj argued Rust gives us \
                      memory safety and performance for the real-time collaboration engine. \
                      We considered Go and TypeScript but Rust won because of its type system \
                      and ability to handle high-concurrency WebSocket connections without a GC.",
            experience_type: ExperienceType::Decision,
            tags: vec!["backend", "rust", "language", "decision"],
            importance: 0.9,
            session: 1,
        },
        // 3: Database decision
        LocomoMemory {
            content: "Selected PostgreSQL as the primary database for user data, project \
                      metadata, and access control. Chose it over MySQL for its JSON support \
                      and advanced indexing. Redis will handle ephemeral session state and \
                      presence information.",
            experience_type: ExperienceType::Decision,
            tags: vec!["database", "postgresql", "redis", "decision"],
            importance: 0.85,
            session: 1,
        },
        // 4: Testing framework decision
        LocomoMemory {
            content: "Chose a testing strategy: Rust's built-in test framework for unit tests, \
                      with proptest for property-based testing of the CRDT engine. Integration \
                      tests will use testcontainers to spin up PostgreSQL and Redis. Sarah Kim \
                      will also set up Playwright for end-to-end frontend tests.",
            experience_type: ExperienceType::Decision,
            tags: vec!["testing", "proptest", "playwright", "qa"],
            importance: 0.75,
            session: 1,
        },
        // 5: Timeline and milestones
        LocomoMemory {
            content: "Established project milestones: Alpha by end of month 2 with basic \
                      real-time editing. Beta by month 4 with presence, permissions, and \
                      offline sync. GA by month 6 in Q3. Each milestone has a demo to Diana \
                      Chen and the leadership team.",
            experience_type: ExperienceType::Task,
            tags: vec!["timeline", "milestones", "planning"],
            importance: 0.8,
            session: 1,
        },
        // 6: Initial architecture sketch
        LocomoMemory {
            content: "Architecture sketch from kickoff: a WebSocket gateway for real-time \
                      sync, a REST API for CRUD operations, a CRDT engine for conflict-free \
                      collaborative editing, and a notification service. All services \
                      communicate via NATS for internal messaging.",
            experience_type: ExperienceType::Decision,
            tags: vec!["architecture", "websocket", "crdt", "nats"],
            importance: 0.85,
            session: 1,
        },
        // 7: Frontend technology
        LocomoMemory {
            content: "Maria chose React with TypeScript for the frontend. The editor component \
                      will use TipTap (ProseMirror-based) for rich text editing. State \
                      management via Zustand because it integrates cleanly with the CRDT layer. \
                      Tailwind CSS for styling.",
            experience_type: ExperienceType::Decision,
            tags: vec!["frontend", "react", "typescript", "editor"],
            importance: 0.7,
            session: 1,
        },
        // 8: Security requirements
        LocomoMemory {
            content: "Security requirements discussed: OAuth2 with PKCE for authentication, \
                      role-based access control with workspace-level and document-level \
                      permissions. All data encrypted at rest with AES-256 and in transit \
                      with TLS 1.3. SOC2 compliance is a hard requirement before GA.",
            experience_type: ExperienceType::Decision,
            tags: vec!["security", "oauth", "rbac", "soc2"],
            importance: 0.9,
            session: 1,
        },
        // 9: Infrastructure plan
        LocomoMemory {
            content: "Tomasz presented the infrastructure plan: Kubernetes on AWS EKS, with \
                      separate namespaces for staging and production. CI/CD via GitHub Actions. \
                      Monitoring with Prometheus and Grafana. Log aggregation through Loki. \
                      Target of 99.9% uptime SLA.",
            experience_type: ExperienceType::Task,
            tags: vec!["infrastructure", "kubernetes", "aws", "devops"],
            importance: 0.8,
            session: 1,
        },
        // 10: Competitive analysis
        LocomoMemory {
            content: "Discussed competitors: Google Docs dominates general collaboration, \
                      Notion is strong for knowledge management, and Linear is the benchmark \
                      for developer tools UX. Our differentiator is native code-aware \
                      collaboration with real-time editing of code, docs, and diagrams in one \
                      workspace, optimized for engineering teams.",
            experience_type: ExperienceType::Conversation,
            tags: vec!["competition", "strategy", "differentiation"],
            importance: 0.7,
            session: 1,
        },
        // 11: Risk assessment
        LocomoMemory {
            content: "Key risks identified during kickoff: CRDT implementation complexity \
                      could delay alpha, Rust hiring pool is smaller than Go/TypeScript, \
                      WebSocket scaling beyond 10K concurrent connections needs load testing \
                      early, and SOC2 compliance audit timeline is uncertain.",
            experience_type: ExperienceType::Observation,
            tags: vec!["risks", "planning", "crdt"],
            importance: 0.8,
            session: 1,
        },
        // 12: API design principles
        LocomoMemory {
            content: "Agreed on API design principles: REST for resource CRUD, WebSocket for \
                      real-time updates, consistent error envelope with correlation IDs, \
                      pagination via cursor-based tokens, rate limiting at 1000 req/min per \
                      workspace. API versioning through URL path (/v1/).",
            experience_type: ExperienceType::Decision,
            tags: vec!["api", "design", "rest", "websocket"],
            importance: 0.75,
            session: 1,
        },
        // 13: Data model discussion
        LocomoMemory {
            content: "Core data model: Workspaces contain Projects, Projects contain Documents. \
                      Documents are versioned with CRDT operations stored as an append-only log. \
                      Users belong to Workspaces with roles (owner, admin, editor, viewer). \
                      Invitations use time-limited tokens.",
            experience_type: ExperienceType::Decision,
            tags: vec!["data-model", "workspace", "documents"],
            importance: 0.75,
            session: 1,
        },
        // 14: Sprint cadence
        LocomoMemory {
            content: "Adopted two-week sprints with planning on Monday and retro on Friday. \
                      Daily standups at 10am UTC. Raj manages the backlog in Linear. Demo \
                      to stakeholders every other sprint. Definition of done includes tests, \
                      docs, and code review by at least one other team member.",
            experience_type: ExperienceType::Decision,
            tags: vec!["process", "sprints", "agile"],
            importance: 0.65,
            session: 1,
        },

        // =====================================================================
        // SESSION 2: Architecture Deep-Dive (15 memories)
        // =====================================================================

        // 15: CRDT algorithm choice
        LocomoMemory {
            content: "After evaluating Yjs, Automerge, and Diamond Types, we decided to build \
                      our own CRDT engine in Rust. The existing libraries are either JavaScript-only \
                      or lack the performance characteristics we need for large documents. Raj will \
                      implement a Peritext-inspired algorithm for rich text with formatting marks.",
            experience_type: ExperienceType::Decision,
            tags: vec!["crdt", "algorithm", "architecture", "peritext"],
            importance: 0.9,
            session: 2,
        },
        // 16: Database schema migration strategy
        LocomoMemory {
            content: "Decided on sqlx with compile-time checked queries for database access. \
                      Migrations managed with sqlx-migrate. Schema changes require a migration \
                      file, review by Raj, and testing against a staging database snapshot \
                      before production rollout. Zero-downtime migrations using the expand-contract \
                      pattern.",
            experience_type: ExperienceType::Decision,
            tags: vec!["database", "migrations", "sqlx"],
            importance: 0.75,
            session: 2,
        },
        // 17: WebSocket scaling architecture
        LocomoMemory {
            content: "Designed the WebSocket gateway: sticky sessions via a consistent hash ring \
                      on workspace ID, NATS JetStream for cross-node message fanout, connection \
                      limits of 5000 per pod with horizontal autoscaling. Heartbeat interval at \
                      30 seconds, reconnect with exponential backoff on the client side.",
            experience_type: ExperienceType::Decision,
            tags: vec!["websocket", "scaling", "nats", "architecture"],
            importance: 0.85,
            session: 2,
        },
        // 18: Decision to switch from PostgreSQL to MongoDB for document storage
        LocomoMemory {
            content: "Major architecture change: switching document storage from PostgreSQL to \
                      MongoDB. The CRDT operation log doesn't fit relational tables well — each \
                      operation is a deeply nested JSON structure with variable-length arrays. \
                      MongoDB's flexible schema and native BSON support handle this naturally. \
                      PostgreSQL remains for user accounts, permissions, and workspace metadata.",
            experience_type: ExperienceType::Decision,
            tags: vec!["database", "mongodb", "postgresql", "architecture-change"],
            importance: 0.95,
            session: 2,
        },
        // 19: Caching strategy
        LocomoMemory {
            content: "Caching strategy: Redis for hot document snapshots (latest materialized \
                      CRDT state), session tokens, and presence data. TTL of 1 hour for document \
                      snapshots, 24 hours for session tokens. Cache invalidation on write via \
                      publish/subscribe through NATS. No local in-process caches to avoid \
                      consistency issues in the multi-pod setup.",
            experience_type: ExperienceType::Decision,
            tags: vec!["caching", "redis", "consistency"],
            importance: 0.75,
            session: 2,
        },
        // 20: Authentication service design
        LocomoMemory {
            content: "Authentication service architecture: OAuth2 code flow with Google and \
                      GitHub as identity providers, with email/password as fallback. JWT access \
                      tokens with 15-minute expiry, refresh tokens with 30-day rotation. Token \
                      validation middleware shared across all services as a Rust library crate.",
            experience_type: ExperienceType::Decision,
            tags: vec!["authentication", "oauth", "jwt", "security"],
            importance: 0.8,
            session: 2,
        },
        // 21: Trade-offs discussion about Rust backend
        LocomoMemory {
            content: "Revisited the Rust backend decision. Trade-offs discussed: compile times \
                      slow down iteration (5-minute incremental builds), hiring is harder but \
                      the existing team is strong, the borrow checker catches real bugs but has \
                      a learning curve for Maria's frontend-to-backend contributions. Net \
                      assessment: keeping Rust because performance gains outweigh velocity cost \
                      for our real-time use case.",
            experience_type: ExperienceType::Conversation,
            tags: vec!["rust", "trade-offs", "backend", "decision"],
            importance: 0.8,
            session: 2,
        },
        // 22: Observability stack
        LocomoMemory {
            content: "Finalized the observability stack: OpenTelemetry for distributed tracing, \
                      Prometheus for metrics, Grafana for dashboards, and PagerDuty for alerting. \
                      Every service emits structured JSON logs with trace_id, span_id, and \
                      service_name fields. Alert thresholds: error rate above 0.5% for 3 minutes \
                      pages on-call.",
            experience_type: ExperienceType::Decision,
            tags: vec!["observability", "monitoring", "tracing", "alerting"],
            importance: 0.75,
            session: 2,
        },
        // 23: Permission model deep-dive
        LocomoMemory {
            content: "Designed granular permission model: workspace-level roles (owner, admin, \
                      member), project-level overrides (can restrict member to viewer), and \
                      document-level sharing links (view, comment, edit) with optional password \
                      protection and expiry. Permission checks happen in middleware before any \
                      handler logic executes.",
            experience_type: ExperienceType::Decision,
            tags: vec!["permissions", "security", "rbac"],
            importance: 0.8,
            session: 2,
        },
        // 24: Impact of MongoDB switch on timeline
        LocomoMemory {
            content: "The switch from PostgreSQL to MongoDB for document storage pushes the \
                      alpha milestone back by two weeks. Raj needs time to rewrite the document \
                      persistence layer and Tomasz needs to set up MongoDB in the Kubernetes \
                      cluster with a replica set. Revised alpha target: end of month 2.5 instead \
                      of month 2.",
            experience_type: ExperienceType::Observation,
            tags: vec!["timeline", "mongodb", "impact", "delay"],
            importance: 0.85,
            session: 2,
        },
        // 25: Offline sync design
        LocomoMemory {
            content: "Offline sync design: client stores CRDT operations in IndexedDB when \
                      disconnected, replays them on reconnect via the WebSocket channel. \
                      Conflict resolution is automatic because CRDTs are convergent. The client \
                      shows a visual indicator when in offline mode and queues outbound operations \
                      with local timestamps.",
            experience_type: ExperienceType::Decision,
            tags: vec!["offline", "sync", "crdt", "design"],
            importance: 0.8,
            session: 2,
        },
        // 26: Error handling convention
        LocomoMemory {
            content: "Established error handling conventions for the Rust services: thiserror \
                      for domain errors, anyhow for application-level propagation. Each service \
                      defines a ServiceError enum. HTTP handlers map errors to standard response \
                      codes. Internal errors never leak to clients — log full context server-side \
                      and return a correlation ID to the caller.",
            experience_type: ExperienceType::Learning,
            tags: vec!["error-handling", "rust", "convention"],
            importance: 0.7,
            session: 2,
        },
        // 27: Load testing plan
        LocomoMemory {
            content: "Planned load testing approach: use k6 to simulate 10,000 concurrent \
                      WebSocket connections with realistic editing patterns. Test scenarios \
                      include burst typing, large paste operations, and concurrent cursors in \
                      the same document section. Tomasz will run these against staging every \
                      sprint.",
            experience_type: ExperienceType::Task,
            tags: vec!["load-testing", "k6", "performance"],
            importance: 0.7,
            session: 2,
        },
        // 28: Code review process
        LocomoMemory {
            content: "Formalized code review process: all PRs require one approval, security-sensitive \
                      changes need two. Raj reviews all CRDT and backend PRs, Maria reviews frontend. \
                      Maximum PR size is 400 lines — larger changes must be split. CI must pass \
                      before review is requested. Average review turnaround target: 4 hours.",
            experience_type: ExperienceType::Decision,
            tags: vec!["code-review", "process", "team"],
            importance: 0.65,
            session: 2,
        },
        // 29: Search architecture
        LocomoMemory {
            content: "Designed full-text search: Meilisearch as the search engine, fed via an \
                      async indexing pipeline that listens to document change events from NATS. \
                      Search results include snippets with highlighted matches. Index is rebuilt \
                      nightly from the MongoDB document snapshots as a consistency safety net.",
            experience_type: ExperienceType::Decision,
            tags: vec!["search", "meilisearch", "architecture"],
            importance: 0.7,
            session: 2,
        },

        // =====================================================================
        // SESSION 3: Bugs and Fixes (15 memories)
        // =====================================================================

        // 30: First bug — CRDT merge panic
        LocomoMemory {
            content: "First major bug: the CRDT merge function panics when two users simultaneously \
                      delete the same text range. The issue is an out-of-bounds index in the \
                      tombstone array. Raj traced it to a missing bounds check when the delete \
                      range spans a previously deleted region. Fix: add a saturating subtraction \
                      and skip already-tombstoned characters.",
            experience_type: ExperienceType::Error,
            tags: vec!["bug", "crdt", "panic", "merge"],
            importance: 0.9,
            session: 3,
        },
        // 31: WebSocket connection leak
        LocomoMemory {
            content: "Found a WebSocket connection leak: when clients disconnect abruptly (network \
                      drop, laptop close), the server-side cleanup task was not firing because \
                      the heartbeat timeout was racing with the TCP keepalive. Connections \
                      accumulated until the pod hit its 5000 connection limit. Fix: added an \
                      explicit server-side ping/pong with a 45-second timeout independent of TCP.",
            experience_type: ExperienceType::Error,
            tags: vec!["bug", "websocket", "connection-leak", "fix"],
            importance: 0.85,
            session: 3,
        },
        // 32: MongoDB write concern issue
        LocomoMemory {
            content: "Discovered that MongoDB operations were using the default write concern \
                      (w:1) which means writes acknowledge after hitting the primary only. During \
                      a primary failover in staging, we lost 12 seconds of CRDT operations. \
                      Changed to w:majority for all document writes. Accepted the ~5ms latency \
                      increase for durability.",
            experience_type: ExperienceType::Error,
            tags: vec!["bug", "mongodb", "write-concern", "durability"],
            importance: 0.85,
            session: 3,
        },
        // 33: Performance regression
        LocomoMemory {
            content: "Performance regression: document load time increased from 200ms to 2.5 seconds \
                      after adding permission checks. Root cause: the permission middleware was making \
                      3 sequential database queries (workspace role, project override, document share). \
                      Fix: batch the permission queries into a single SQL statement with a UNION and \
                      cache the result for 60 seconds per user-document pair.",
            experience_type: ExperienceType::Error,
            tags: vec!["performance", "regression", "permissions", "fix"],
            importance: 0.8,
            session: 3,
        },
        // 34: Formatting preservation bug
        LocomoMemory {
            content: "Bug in rich text formatting: bold and italic marks were lost when merging \
                      concurrent insertions at a formatting boundary. The Peritext algorithm \
                      implementation had an off-by-one error in the mark expansion logic. Sarah's \
                      proptest suite caught this — the test generated a sequence of interleaved \
                      format-then-insert operations that exposed the boundary condition.",
            experience_type: ExperienceType::Error,
            tags: vec!["bug", "formatting", "crdt", "peritext"],
            importance: 0.8,
            session: 3,
        },
        // 35: Sarah's testing catches bugs
        LocomoMemory {
            content: "Sarah Kim's property-based testing strategy is proving invaluable. In two \
                      weeks of running proptest against the CRDT engine, it found 5 bugs that \
                      manual testing missed: the merge panic, the formatting boundary issue, \
                      a Unicode surrogate pair split, a cursor position drift after undo, and \
                      an empty document edge case. Decided to increase proptest iteration count \
                      from 256 to 1024.",
            experience_type: ExperienceType::Observation,
            tags: vec!["testing", "proptest", "sarah", "crdt", "bugs"],
            importance: 0.85,
            session: 3,
        },
        // 36: Authentication token bug
        LocomoMemory {
            content: "Found a security bug: refresh tokens were not being invalidated on password \
                      change. A user who changed their password could still have old sessions \
                      active indefinitely via refresh tokens. Fix: added a token_version counter \
                      on the user record that increments on password change. JWT validation now \
                      checks token_version matches.",
            experience_type: ExperienceType::Error,
            tags: vec!["security", "bug", "authentication", "tokens"],
            importance: 0.9,
            session: 3,
        },
        // 37: NATS message ordering issue
        LocomoMemory {
            content: "Encountered a NATS message ordering issue: document change events were \
                      arriving out of order at the search indexer, causing stale document \
                      content in search results. Root cause: NATS subjects were not partitioned \
                      by document ID, so messages for the same document went to different \
                      consumers. Fix: use document ID as the NATS subject suffix for ordering \
                      guarantees within a document.",
            experience_type: ExperienceType::Error,
            tags: vec!["bug", "nats", "ordering", "search"],
            importance: 0.75,
            session: 3,
        },
        // 38: Memory usage optimization
        LocomoMemory {
            content: "Optimized CRDT memory usage: the operation log was growing unbounded for \
                      active documents. Implemented periodic compaction that snapshots the current \
                      state and prunes operations older than 24 hours (keeping enough for offline \
                      clients to catch up). Reduced memory per-document from ~50MB for large docs \
                      to ~5MB. Tomasz confirmed staging memory usage dropped 60%.",
            experience_type: ExperienceType::Learning,
            tags: vec!["optimization", "crdt", "memory", "compaction"],
            importance: 0.8,
            session: 3,
        },
        // 39: CI pipeline flaky tests
        LocomoMemory {
            content: "The CI pipeline had 3 flaky tests causing spurious failures. Root causes: \
                      two tests had race conditions due to shared test database state (fixed with \
                      per-test database schemas), and one test assumed sub-millisecond timing \
                      that fails under CI load (fixed with retry and wider timing bounds). CI \
                      reliability improved from 72% to 98% green rate.",
            experience_type: ExperienceType::Error,
            tags: vec!["ci", "flaky-tests", "fix", "reliability"],
            importance: 0.7,
            session: 3,
        },
        // 40: Client reconnection bug
        LocomoMemory {
            content: "Bug in client reconnection logic: after reconnecting, the client replayed \
                      all buffered operations but also re-applied operations it had already sent \
                      before disconnecting, causing duplicate text insertion. Fix: added a \
                      sequence number to each operation and the server deduplicates based on \
                      client_id + sequence_number pairs.",
            experience_type: ExperienceType::Error,
            tags: vec!["bug", "reconnection", "duplicate", "client"],
            importance: 0.8,
            session: 3,
        },
        // 41: Raj's CRDT debugging approach
        LocomoMemory {
            content: "Raj developed a CRDT visualization tool that renders the operation DAG \
                      as a graphviz diagram, showing causal relationships between operations \
                      from different users. This made debugging merge conflicts dramatically \
                      easier — previously it took hours to trace through operation logs manually.",
            experience_type: ExperienceType::Learning,
            tags: vec!["crdt", "debugging", "tooling", "raj"],
            importance: 0.7,
            session: 3,
        },
        // 42: Rate limiting incident
        LocomoMemory {
            content: "Rate limiting misconfiguration: the 1000 req/min limit was applied per \
                      user instead of per workspace, so a single heavy user in a large workspace \
                      could exhaust the quota for everyone. Changed to per-workspace rate limiting \
                      with per-user sub-limits at 200 req/min. Added rate limit headers \
                      (X-RateLimit-Remaining, X-RateLimit-Reset) to all API responses.",
            experience_type: ExperienceType::Error,
            tags: vec!["rate-limiting", "bug", "api", "fix"],
            importance: 0.75,
            session: 3,
        },
        // 43: Testing strategy evolution
        LocomoMemory {
            content: "Testing strategy evolved significantly after the CRDT bugs. Added a \
                      fuzz testing layer using cargo-fuzz targeting the merge and apply_op \
                      functions. Property tests expanded to cover all formatting mark types. \
                      Sarah set up a nightly chaos test that randomly kills pods during load \
                      testing to verify the system recovers gracefully.",
            experience_type: ExperienceType::Decision,
            tags: vec!["testing", "fuzz", "chaos", "evolution"],
            importance: 0.8,
            session: 3,
        },
        // 44: Encryption at rest implementation
        LocomoMemory {
            content: "Implemented encryption at rest for document content in MongoDB using \
                      envelope encryption. Each document has a unique data encryption key (DEK) \
                      encrypted by a key encryption key (KEK) stored in AWS KMS. Decrypt \
                      latency adds ~2ms per document load which is acceptable.",
            experience_type: ExperienceType::Task,
            tags: vec!["encryption", "security", "mongodb", "kms"],
            importance: 0.8,
            session: 3,
        },

        // =====================================================================
        // SESSION 4: Review and Plan Changes (15 memories)
        // =====================================================================

        // 45: Progress review
        LocomoMemory {
            content: "Progress review at the 10-week mark: alpha is 80% complete. Real-time \
                      editing works with up to 50 concurrent users on the same document. \
                      Permission system is functional. The CRDT engine has been stabilized \
                      after the bug fixes. Main remaining alpha work: offline sync and the \
                      notification system.",
            experience_type: ExperienceType::Observation,
            tags: vec!["progress", "review", "alpha", "status"],
            importance: 0.85,
            session: 4,
        },
        // 46: Architecture decision reversed — back to PostgreSQL for some data
        LocomoMemory {
            content: "Reversed part of the MongoDB decision: moving workspace analytics and \
                      audit logs back to PostgreSQL. MongoDB is great for CRDT operations but \
                      its aggregation pipeline is too complex for the analytics queries we need. \
                      The team realized that time-series analytics on user activity patterns \
                      are much simpler with SQL window functions.",
            experience_type: ExperienceType::Decision,
            tags: vec!["architecture-change", "mongodb", "postgresql", "reversed"],
            importance: 0.9,
            session: 4,
        },
        // 47: Raj proposes architecture that will cause issues
        LocomoMemory {
            content: "Raj proposed a microservice split: breaking the monolithic backend into \
                      separate services for auth, documents, collaboration, and notifications. \
                      Maria raised concerns about increased deployment complexity and inter-service \
                      latency. Tomasz agreed to prototype the split but warned it could introduce \
                      distributed transaction issues.",
            experience_type: ExperienceType::Conversation,
            tags: vec!["architecture", "microservices", "raj", "discussion"],
            importance: 0.8,
            session: 4,
        },
        // 48: Bugs caused by the microservice split prototype
        LocomoMemory {
            content: "The microservice split prototype exposed several bugs: auth tokens were \
                      not properly propagated between services, the document service needed to \
                      call auth for permission checks adding 50ms latency per request, and the \
                      NATS event schema diverged between services within a week. Decided to \
                      revert to the monolith for alpha and revisit microservices for beta.",
            experience_type: ExperienceType::Error,
            tags: vec!["microservices", "bugs", "revert", "monolith"],
            importance: 0.85,
            session: 4,
        },
        // 49: Timeline adjustment
        LocomoMemory {
            content: "Timeline revised again: alpha pushed to end of month 3 due to the MongoDB \
                      switch delay and the failed microservice prototype. Beta target remains \
                      month 5. GA pushed from month 6 to month 7 to accommodate the extra \
                      stabilization time. Diana Chen approved the revised timeline but emphasized \
                      no further slips.",
            experience_type: ExperienceType::Decision,
            tags: vec!["timeline", "delay", "revised", "planning"],
            importance: 0.85,
            session: 4,
        },
        // 50: What went well assessment
        LocomoMemory {
            content: "Things that went well: the Rust CRDT engine is performant and correct \
                      after stabilization. Property-based testing caught bugs that would have \
                      been production incidents. The team's communication cadence (daily standups, \
                      sprint demos) kept everyone aligned. Tomasz's infrastructure automation \
                      made environment provisioning painless.",
            experience_type: ExperienceType::Observation,
            tags: vec!["retrospective", "positive", "team"],
            importance: 0.75,
            session: 4,
        },
        // 51: What didn't go well
        LocomoMemory {
            content: "Things that didn't go well: the database technology flip-flopping wasted \
                      3 weeks of development time. The microservice split attempt was premature \
                      and cost a full sprint. Some architectural decisions were made too quickly \
                      without enough evaluation (the MongoDB switch was decided in a single \
                      meeting). Need more rigorous decision-making with written proposals.",
            experience_type: ExperienceType::Observation,
            tags: vec!["retrospective", "negative", "decisions"],
            importance: 0.8,
            session: 4,
        },
        // 52: Decision-making process improvement
        LocomoMemory {
            content: "New decision-making process adopted: any architecture decision that affects \
                      more than one service requires a written RFC with problem statement, options \
                      evaluated, trade-offs, and a decision record. RFCs need 48 hours of review \
                      before implementation begins. Raj and Tomasz are required reviewers for \
                      all infrastructure RFCs.",
            experience_type: ExperienceType::Decision,
            tags: vec!["process", "rfc", "decision-making", "improvement"],
            importance: 0.8,
            session: 4,
        },
        // 53: Performance benchmarks
        LocomoMemory {
            content: "Performance benchmarks for the alpha build: document load time 180ms (p50), \
                      320ms (p99). Real-time sync latency 45ms (p50), 120ms (p99) for character-level \
                      operations. WebSocket gateway handles 8000 concurrent connections per pod. \
                      MongoDB document write latency 15ms (p50), 45ms (p99) with w:majority.",
            experience_type: ExperienceType::Discovery,
            tags: vec!["performance", "benchmarks", "metrics"],
            importance: 0.8,
            session: 4,
        },
        // 54: Focus areas for next phase
        LocomoMemory {
            content: "Priorities for the next phase: complete offline sync (Raj, 3 weeks), \
                      implement notification service with email and in-app channels (Maria, 2 weeks), \
                      SOC2 audit preparation including penetration testing (Tomasz, ongoing), \
                      expand test coverage to 85% with emphasis on integration tests (Sarah, \
                      continuous). Beta feature: collaborative diagrams using Excalidraw integration.",
            experience_type: ExperienceType::Task,
            tags: vec!["priorities", "next-phase", "planning"],
            importance: 0.85,
            session: 4,
        },
        // 55: Technical debt inventory
        LocomoMemory {
            content: "Technical debt identified: the CRDT compaction logic needs refactoring (tight \
                      coupling to MongoDB client), error handling in the WebSocket gateway is \
                      inconsistent (mix of anyhow and manual error types), the permission cache \
                      lacks proper invalidation on role changes, and the CI pipeline still uses \
                      a deprecated GitHub Actions runner image.",
            experience_type: ExperienceType::Observation,
            tags: vec!["technical-debt", "refactoring", "maintenance"],
            importance: 0.75,
            session: 4,
        },
        // 56: Hiring discussion
        LocomoMemory {
            content: "Discussed hiring: need one more backend engineer to handle the beta feature \
                      load. Rust experience preferred but strong Go/C++ engineers willing to learn \
                      are acceptable. Also need a dedicated security engineer for the SOC2 push. \
                      Diana approved both headcount requests.",
            experience_type: ExperienceType::Conversation,
            tags: vec!["hiring", "team", "growth"],
            importance: 0.7,
            session: 4,
        },
        // 57: Monitoring gaps found
        LocomoMemory {
            content: "Discovered monitoring gaps during the review: no alerts on MongoDB replica \
                      set lag, no dashboard for CRDT operation throughput, and the WebSocket \
                      connection count metric was not exposed to Prometheus. Tomasz added these \
                      to the observability backlog with high priority.",
            experience_type: ExperienceType::Observation,
            tags: vec!["monitoring", "gaps", "observability"],
            importance: 0.7,
            session: 4,
        },
        // 58: Client SDK decision
        LocomoMemory {
            content: "Decided to build official client SDKs in TypeScript and Python for the \
                      public API. The TypeScript SDK will be auto-generated from the OpenAPI \
                      spec using openapi-typescript-codegen. The Python SDK will be hand-written \
                      because the WebSocket layer needs custom handling. Both SDKs target beta \
                      release.",
            experience_type: ExperienceType::Decision,
            tags: vec!["sdk", "typescript", "python", "api"],
            importance: 0.7,
            session: 4,
        },
        // 59: Lessons learned summary
        LocomoMemory {
            content: "Key lessons learned: invest heavily in property-based testing for \
                      algorithmic code, avoid technology switches without written evaluation, \
                      prototype risky architecture changes before committing, and keep the \
                      deployment architecture simple until scale demands complexity. The team \
                      is stronger and more disciplined after these early missteps.",
            experience_type: ExperienceType::Learning,
            tags: vec!["lessons", "retrospective", "growth"],
            importance: 0.85,
            session: 4,
        },
    ]
}

// =============================================================================
// QUERIES: 20 benchmark queries (5 per type)
// =============================================================================

fn locomo_queries() -> Vec<LocomoQuery> {
    vec![
        // =====================================================================
        // SINGLE-HOP: Direct fact retrieval
        // =====================================================================
        LocomoQuery {
            query: "What programming language did we choose for the backend?",
            query_type: "single_hop",
            expected_memory_indices: vec![2],     // Rust decision
            absence_indices: vec![7, 10, 44],     // frontend, competitors, encryption
        },
        LocomoQuery {
            query: "Who is the project lead?",
            query_type: "single_hop",
            expected_memory_indices: vec![1],     // Raj Patel is project lead
            absence_indices: vec![10, 29, 44],    // competitors, search, encryption
        },
        LocomoQuery {
            query: "What database are we using?",
            query_type: "single_hop",
            expected_memory_indices: vec![3, 18], // PostgreSQL original + MongoDB switch
            absence_indices: vec![10, 14, 27],    // competitors, sprints, load testing
        },
        LocomoQuery {
            query: "What was the first bug we encountered?",
            query_type: "single_hop",
            expected_memory_indices: vec![30],    // CRDT merge panic
            absence_indices: vec![0, 5, 14],      // kickoff, timeline, sprints
        },
        LocomoQuery {
            query: "What testing framework did we pick?",
            query_type: "single_hop",
            expected_memory_indices: vec![4],     // proptest + playwright
            absence_indices: vec![0, 10, 29],     // kickoff, competitors, search
        },

        // =====================================================================
        // TEMPORAL: Time-referenced queries
        // =====================================================================
        LocomoQuery {
            query: "What decisions did we make last week?",
            query_type: "temporal",
            // Session 4 decisions (most recent — closest to "last week")
            expected_memory_indices: vec![46, 49, 52],
            absence_indices: vec![0, 2, 3],       // session 1 items
        },
        LocomoQuery {
            query: "What happened during the second meeting?",
            query_type: "temporal",
            // Session 2 = architecture deep-dive
            expected_memory_indices: vec![15, 17, 18],
            absence_indices: vec![30, 45, 48],    // session 3/4 items
        },
        LocomoQuery {
            query: "When did we switch from PostgreSQL to MongoDB?",
            query_type: "temporal",
            expected_memory_indices: vec![18],    // the switch decision
            absence_indices: vec![0, 5, 30],      // kickoff, timeline, first bug
        },
        LocomoQuery {
            query: "What was the most recent architecture change?",
            query_type: "temporal",
            // Session 4: analytics back to PostgreSQL, microservice revert
            expected_memory_indices: vec![46, 48],
            absence_indices: vec![6, 12, 3],      // session 1 architecture
        },
        LocomoQuery {
            query: "What bugs did we find during the debugging phase?",
            query_type: "temporal",
            // Session 3 = debugging phase. CRDT merge panic (30) and WebSocket leak (31)
            // are the two earliest bugs found during that sprint.
            expected_memory_indices: vec![30, 31],
            absence_indices: vec![0, 5, 14],      // non-bug items
        },

        // =====================================================================
        // MULTI-HOP: Combining information from multiple memories
        // =====================================================================
        LocomoQuery {
            query: "Why did we change the database AND what was the impact on the timeline?",
            query_type: "multi_hop",
            expected_memory_indices: vec![18, 24], // MongoDB switch + timeline impact
            absence_indices: vec![7, 10, 14],      // frontend, competitors, sprints
        },
        LocomoQuery {
            query: "Who proposed the architecture that caused the most bugs?",
            query_type: "multi_hop",
            // Raj proposed microservice split (47), which caused bugs (48)
            expected_memory_indices: vec![47, 48],
            absence_indices: vec![0, 5, 10],       // kickoff, timeline, competitors
        },
        LocomoQuery {
            query: "What decisions were reversed and why?",
            query_type: "multi_hop",
            // MongoDB partial reversal (46), microservice revert (48)
            expected_memory_indices: vec![46, 48],
            absence_indices: vec![0, 5, 14],       // kickoff basics
        },
        LocomoQuery {
            query: "How did the testing strategy change based on the bugs we found?",
            query_type: "multi_hop",
            // Original testing (4), Sarah's testing value (35), evolution (43)
            expected_memory_indices: vec![35, 43],
            absence_indices: vec![0, 10, 29],      // non-testing items
        },
        LocomoQuery {
            query: "What trade-offs did we discuss about the backend language choice?",
            query_type: "multi_hop",
            // Original Rust decision (2) + revisited trade-offs (21)
            expected_memory_indices: vec![2, 21],
            absence_indices: vec![7, 10, 44],      // frontend, competitors, encryption
        },

        // =====================================================================
        // OPEN-DOMAIN: Broad, context-dependent queries
        // =====================================================================
        LocomoQuery {
            query: "Summarize the project so far",
            query_type: "open_domain",
            // Should surface high-level status items
            expected_memory_indices: vec![0, 45, 49],
            absence_indices: vec![39, 42],         // CI flaky tests, rate limiting detail
        },
        LocomoQuery {
            query: "What are the biggest risks?",
            query_type: "open_domain",
            expected_memory_indices: vec![11],     // risk assessment
            absence_indices: vec![14, 29],         // sprints, search
        },
        LocomoQuery {
            query: "What should we focus on next?",
            query_type: "open_domain",
            expected_memory_indices: vec![54, 55], // priorities + tech debt
            absence_indices: vec![0, 10],          // kickoff, competitors
        },
        LocomoQuery {
            query: "What went well and what didn't?",
            query_type: "open_domain",
            expected_memory_indices: vec![50, 51], // went well + didn't go well
            absence_indices: vec![3, 12],          // database choice, API design
        },
        LocomoQuery {
            query: "What patterns do you see in our decision-making?",
            query_type: "open_domain",
            // Decision reversal pattern, process improvement, lessons learned
            expected_memory_indices: vec![52, 59],
            absence_indices: vec![3, 9],           // database detail, infra plan
        },
    ]
}

// =============================================================================
// SCORING FUNCTIONS
// =============================================================================

/// MRR: 1/rank of the first relevant result (0 if none in results).
fn reciprocal_rank(retrieved: &[usize], relevant: &[usize]) -> f32 {
    for (rank, idx) in retrieved.iter().enumerate() {
        if relevant.contains(idx) {
            return 1.0 / (rank as f32 + 1.0);
        }
    }
    0.0
}

/// Recall@k: fraction of relevant memories found in top-k results.
fn recall_at_k(retrieved: &[usize], relevant: &[usize], k: usize) -> f32 {
    if relevant.is_empty() {
        return 1.0;
    }
    let top_k: Vec<usize> = retrieved.iter().take(k).copied().collect();
    let hits = relevant.iter().filter(|r| top_k.contains(r)).count();
    hits as f32 / relevant.len() as f32
}

/// Precision@k: fraction of top-k results that are relevant.
fn precision_at_k(retrieved: &[usize], relevant: &[usize], k: usize) -> f32 {
    let top_k: Vec<usize> = retrieved.iter().take(k).copied().collect();
    if top_k.is_empty() {
        return 0.0;
    }
    let hits = top_k.iter().filter(|r| relevant.contains(r)).count();
    hits as f32 / k as f32
}

/// Count how many absence_indices appear in retrieved results.
fn count_absence_violations(retrieved: &[usize], absence: &[usize]) -> usize {
    retrieved
        .iter()
        .filter(|r| absence.contains(r))
        .count()
}

/// Map MemoryIds returned by recall back to corpus indices.
fn map_results_to_indices(
    results: &[std::sync::Arc<shodh_memory::memory::types::Memory>],
    stored_ids: &[MemoryId],
) -> Vec<usize> {
    results
        .iter()
        .filter_map(|mem| stored_ids.iter().position(|id| *id == mem.id))
        .collect()
}

// =============================================================================
// REPORTING
// =============================================================================

fn aggregate_by_type(results: &[PerQueryResult]) -> Vec<TypeSummary> {
    let types = ["single_hop", "temporal", "multi_hop", "open_domain"];
    types
        .iter()
        .map(|qt| {
            let matching: Vec<&PerQueryResult> =
                results.iter().filter(|r| r.query_type == *qt).collect();
            let n = matching.len();
            if n == 0 {
                return TypeSummary {
                    query_type: qt.to_string(),
                    mrr: 0.0,
                    recall_at_5: 0.0,
                    recall_at_10: 0.0,
                    precision_at_5: 0.0,
                    absence_violations: 0,
                    query_count: 0,
                    avg_latency_ms: 0,
                };
            }
            let nf = n as f32;
            TypeSummary {
                query_type: qt.to_string(),
                mrr: matching.iter().map(|r| r.mrr).sum::<f32>() / nf,
                recall_at_5: matching.iter().map(|r| r.recall_at_5).sum::<f32>() / nf,
                recall_at_10: matching.iter().map(|r| r.recall_at_10).sum::<f32>() / nf,
                precision_at_5: matching.iter().map(|r| r.precision_at_5).sum::<f32>() / nf,
                absence_violations: matching.iter().map(|r| r.absence_violations).sum(),
                query_count: n,
                avg_latency_ms: (matching.iter().map(|r| r.latency_ms).sum::<u64>() as f32 / nf)
                    as u64,
            }
        })
        .collect()
}

fn print_report(
    per_query: &[PerQueryResult],
    type_summaries: &[TypeSummary],
    overall_mrr: f32,
    overall_recall: f32,
    overall_recall_10: f32,
    overall_precision: f32,
    total_absence_violations: usize,
    total_absence_checks: usize,
    avg_latency_ms: u64,
) {
    println!();
    println!("==========================================================================");
    println!("  LOCOMO-Style Retrieval Quality Benchmark");
    println!("  Corpus: 60 memories across 4 sessions");
    println!("  Queries: 20 (5 per type)");
    println!("  Reference: mem0 scored 66.9% on LOCOMO");
    println!("==========================================================================");
    println!();

    // --- Per-query detail ---
    println!("Per-Query Detail");
    println!("{}", "\u{2500}".repeat(78));
    for r in per_query {
        let hit = if r.mrr > 0.0 { "HIT " } else { "MISS" };
        let absence_note = if r.absence_violations > 0 {
            format!(" [!{} absence violations]", r.absence_violations)
        } else {
            String::new()
        };
        println!(
            "  [{:<10}] [{}] MRR={:.2}  R@5={:.2}  R@10={:.2}  P@5={:.2}  {}ms{}",
            r.query_type, hit, r.mrr, r.recall_at_5, r.recall_at_10, r.precision_at_5, r.latency_ms, absence_note
        );
        let display_query = if r.query_text.len() > 72 {
            format!("{}...", &r.query_text[..69])
        } else {
            r.query_text.clone()
        };
        println!("    Q: \"{}\"", display_query);
        println!(
            "    expected={:?}  retrieved={:?}",
            r.expected_indices, r.retrieved_indices
        );
    }
    println!();

    // --- Summary table ---
    println!(
        "{:<14} {:>6} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "Query Type", "N", "MRR", "R@5", "R@10", "P@5", "Abs.Viol", "Lat(ms)"
    );
    println!("{}", "\u{2501}".repeat(78));
    for s in type_summaries {
        println!(
            "{:<14} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>8}",
            s.query_type,
            s.query_count,
            s.mrr,
            s.recall_at_5,
            s.recall_at_10,
            s.precision_at_5,
            s.absence_violations,
            s.avg_latency_ms
        );
    }
    println!("{}", "\u{2501}".repeat(88));
    println!(
        "{:<14} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>8}",
        "OVERALL",
        per_query.len(),
        overall_mrr,
        overall_recall,
        overall_recall_10,
        overall_precision,
        format!("{}/{}", total_absence_violations, total_absence_checks),
        avg_latency_ms
    );
    println!();

    // --- Composite score ---
    // Weighted composite comparable to LOCOMO's scoring:
    // 40% MRR + 30% Recall@5 + 20% Precision@5 + 10% absence compliance
    let absence_compliance = if total_absence_checks > 0 {
        1.0 - (total_absence_violations as f32 / total_absence_checks as f32)
    } else {
        1.0
    };
    let composite =
        0.30 * overall_mrr + 0.20 * overall_recall + 0.15 * overall_recall_10 + 0.20 * overall_precision + 0.15 * absence_compliance;
    println!(
        "Composite Score: {:.1}%  (MRR={:.1}% R@5={:.1}% R@10={:.1}% P@5={:.1}% AbsCompl={:.1}%)",
        composite * 100.0,
        overall_mrr * 100.0,
        overall_recall * 100.0,
        overall_recall_10 * 100.0,
        overall_precision * 100.0,
        absence_compliance * 100.0,
    );
    println!("  (Reference: mem0 = 66.9% on LOCOMO)");
    println!();
}

// =============================================================================
// BENCHMARK TEST
// =============================================================================

#[test]
#[ignore] // Run with: cargo test locomo_benchmark -- --ignored --nocapture
fn locomo_benchmark() {
    // --- Setup ---
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 200,
        session_memory_size_mb: 100,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.1,
    };
    let system = MemorySystem::new(config, None).expect("Failed to create memory system");

    // --- Store corpus with session-based timestamps ---
    let corpus = locomo_corpus();
    let mut stored_ids: Vec<MemoryId> = Vec::with_capacity(corpus.len());

    // Assign timestamps: session 1 = 4 weeks ago, session 2 = 3 weeks ago,
    // session 3 = 2 weeks ago, session 4 = 1 week ago
    let now = Utc::now();
    let session_offsets: HashMap<u8, Duration> = [
        (1, Duration::weeks(4)),
        (2, Duration::weeks(3)),
        (3, Duration::weeks(2)),
        (4, Duration::weeks(1)),
    ]
    .into_iter()
    .collect();

    let store_start = Instant::now();
    for (i, mem) in corpus.iter().enumerate() {
        let session_offset = session_offsets
            .get(&mem.session)
            .copied()
            .unwrap_or(Duration::zero());
        // Within a session, space memories 5 minutes apart for ordering
        let intra_session_offset = Duration::minutes(5 * (i as i64 % 15));
        let created_at = now - session_offset + intra_session_offset;

        let mut metadata = HashMap::new();
        metadata.insert("importance_hint".to_string(), mem.importance.to_string());
        metadata.insert("session".to_string(), mem.session.to_string());

        let experience = Experience {
            content: mem.content.to_string(),
            experience_type: mem.experience_type.clone(),
            entities: mem.tags.iter().map(|t| t.to_string()).collect(),
            metadata,
            ..Default::default()
        };

        let id = system
            .remember(experience, Some(created_at))
            .expect("Failed to store memory");
        stored_ids.push(id);
    }
    let store_duration = store_start.elapsed();
    println!(
        "\nStored {} memories in {}ms",
        stored_ids.len(),
        store_duration.as_millis()
    );

    // --- Run queries ---
    let queries = locomo_queries();
    let mut per_query_results: Vec<PerQueryResult> = Vec::with_capacity(queries.len());

    for bq in &queries {
        let query = Query {
            query_text: Some(bq.query.to_string()),
            max_results: 10,
            ..Default::default()
        };

        let query_start = Instant::now();
        let results = system.recall(&query).expect("Recall failed");
        let latency = query_start.elapsed();

        let retrieved_indices = map_results_to_indices(&results, &stored_ids);

        let mrr = reciprocal_rank(&retrieved_indices, &bq.expected_memory_indices);
        let r5 = recall_at_k(&retrieved_indices, &bq.expected_memory_indices, 5);
        let r10 = recall_at_k(&retrieved_indices, &bq.expected_memory_indices, 10);
        let p5 = precision_at_k(&retrieved_indices, &bq.expected_memory_indices, 5);
        let abs_v = count_absence_violations(&retrieved_indices, &bq.absence_indices);

        per_query_results.push(PerQueryResult {
            query_text: bq.query.to_string(),
            query_type: bq.query_type.to_string(),
            mrr,
            recall_at_5: r5,
            recall_at_10: r10,
            precision_at_5: p5,
            absence_violations: abs_v,
            latency_ms: latency.as_millis() as u64,
            retrieved_indices,
            expected_indices: bq.expected_memory_indices.clone(),
        });
    }

    // --- Aggregate ---
    let type_summaries = aggregate_by_type(&per_query_results);

    let n = per_query_results.len() as f32;
    let overall_mrr = per_query_results.iter().map(|r| r.mrr).sum::<f32>() / n;
    let overall_recall = per_query_results.iter().map(|r| r.recall_at_5).sum::<f32>() / n;
    let overall_recall_10 = per_query_results.iter().map(|r| r.recall_at_10).sum::<f32>() / n;
    let overall_precision = per_query_results.iter().map(|r| r.precision_at_5).sum::<f32>() / n;
    let total_absence_violations: usize =
        per_query_results.iter().map(|r| r.absence_violations).sum();
    let total_absence_checks: usize = queries.iter().map(|q| q.absence_indices.len()).sum();
    let avg_latency_ms =
        (per_query_results.iter().map(|r| r.latency_ms).sum::<u64>() as f32 / n) as u64;

    // --- Report ---
    print_report(
        &per_query_results,
        &type_summaries,
        overall_mrr,
        overall_recall,
        overall_recall_10,
        overall_precision,
        total_absence_violations,
        total_absence_checks,
        avg_latency_ms,
    );
}
