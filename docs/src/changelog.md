<!-- GENERATED FILE — do not edit by hand.
     Source: git log + tags
     Generator: docs/generators/src/bin/gen-changelog.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-changelog`) -->

# Changelog

Generated from git history. Sections are conventional-commit prefixes; hashes link to the commit on origin.

## Unreleased

### build

- build(windows-service): harden -Configure race + add shawl/WinSW backends (4de853f1)

### chore

- chore(memory): delete orphan src/memory/maintenance.rs (undeclared dead code; FIX-R1 expiry now ported into live run_maintenance) (2adb909d)
- chore: bump version 0.7.20 -> 0.7.21 (835e5869)
- chore(clippy): resolve all lib warnings (fixes + targeted allows) (3ecfb810)
- chore(lib): apply cargo fix — drop unused import + mark unused binding (7bdf9182)
- chore(docs): move veld-vs-shodh-comparison.html to ../veld-docs/ (42dbe936)
- chore: remove duplicate rlm_refiner.rs; fix comparison HTML; update generated docs (42ab7f8c)
- chore(gitignore): exclude entire evaluations/ directory (52ea6d23)
- chore(gitignore): exclude benchmark result snapshots from git (19115938)
- chore(locomo): warm-veld result snapshot (MRR 0.917) (6dae51f6)
- chore(carry): include llm_reranker.rs to unblock compile (55fbb700)

### ci

- ci(docs): GitHub Actions workflow for mdBook build and GitHub Pages deploy (59f918ba)
- ci(docs): GitHub Actions workflow for mdBook build and GitHub Pages deploy (1dc193f9)
- ci: install NSIS via choco before makensis-action (3e7560f6)

### docs

- docs(refresh): post-dials regeneration + dials + instance-registry pages (25378f65)
- docs(reference): regenerate config.md (92 env vars) (ed3eaa78)
- docs(guides): add relational-backends operator guide (22edb1e2)
- docs(comparison): correct and extend veld-vs-shodh-comparison.html (5c891432)
- docs(claude-md): correct streaming/metrics/visualization; pointer to exhaustive module-index (c0e89637)
- docs(sidecar): polish generators, content, and plugin set (56ab8cfd)
- docs(meta): update README, CLAUDE.md, and .gitignore for docs sidecar (0fcd6187)
- docs(generators): Phase B+C generator workspace (8 Rust binaries) (5a85fc6b)
- docs: Phase A+B mdBook skeleton with all reference stubs (fdfd3d1d)
- docs(claude): expand CLAUDE.md with architecture, MCP table, codebase layout (69440ca3)
- docs(facts): SECURITY.md + decision record 0005 (ae691d0b)
- docs(claude-md): correct streaming/metrics/visualization; pointer to exhaustive module-index (622ab451)
- docs(sidecar): polish generators, content, and plugin set (91e6bc0a)
- docs(meta): update README, CLAUDE.md, and .gitignore for docs sidecar (80610483)
- docs(generators): Phase B+C generator workspace (8 Rust binaries) (1abc5b2b)
- docs: Phase A+B mdBook skeleton with all reference stubs (d0c3f5e0)
- docs(claude): expand CLAUDE.md with architecture, MCP table, codebase layout (27741328)

### feat

- feat(c3): surface source-trust at recall (injection boundary, W-006) (eb61da59)
- feat(c1): per-user write-path serialization lock (W-005) (9991b4ee)
- feat(p4): iterative re-cueing (pattern completion) + clippy sweep (21e69d9e)
- feat(r3): populate EngramBinding conjunctive binding + surface reliability (7284f21f)
- feat(r6): bi-temporal as-of point-in-time queries (6e9d6845)
- feat(discovery): server-authored instance registry (replaces port-guessing) (53f287f6)
- feat(hooks): statusline accepts URL list as CLI arg; wire dual-server in settings (e918f300)
- feat(hooks): wire Claude session → veld so the monitor sees live sessions (e4f05fd5)
- feat(monitor): polish dials UI — ranges, tooltips, descriptions (71259664)
- feat(monitor): Dials panel in Tauri tray monitor (get_dials/set_dials commands + status-core helpers + frontend) (0b6407f9)
- feat(dials): persist runtime dials via KeyValueStore (rocksdb kv CF accessor + load-on-ctor/persist-on-set) (fce2627e)
- feat(storage): implement real redb KeyValueStore (was a stub) + round-trip/scan/persist test (13a18841)
- feat(tui): add Dials tab (view + nav); documents reconsolidation/retrieval dials and how to set them (5c10a806)
- feat(dials): MCP get_dials/set_dials tools (rmcp Rust + TS); client get/put (020ad5fc)
- feat(dials): GET/PUT /api/dials handlers (protected, per-user, sanitized) (f947dbdc)
- feat(dials): per-query retrieval overrides (refiner/rerank/candidate/rrf) via attached dials handle (9b213465)
- feat(reconsolidation): Layers 2 & 3 re-encode — entity enrich + embedding nudge (dial-gated, single re-index) (7a9bd588)
- feat(reconsolidation): Layer 1 re-encode — strengthen graph edges to retrieval-context entities (dial-gated) (25cef3f3)
- feat(reconsolidation): run_reconsolidation_expiry Layer 0 (drain+decay), gated; called from run_maintenance (3ce3f26c)
- feat(reconsolidation): accumulate retrieval_context across labile window; window length from dial (776f3db6)
- feat(dials): seed CognitiveDials from VELD_* env at construction (0168e5dd)
- feat(dials): wire CognitiveDials runtime state onto MemorySystem (4655cbee)
- feat(dials): add CognitiveDials runtime tuning struct (inert by default) (7b670eeb)
- feat(e6): full TCM context_drift — graded temporal-context contiguity (2a41609c)
- feat(cross-encoder): default to mxbai-xsmall + add jina-tiny/turbo MiniLM updates (76c25717)
- feat(m4): return query-level entity coverage + weighted-available FOK (cebea009)
- feat(m3.1): cross-embedder agreement signal for query-level FOK (14a29bf5)
- feat(cross-encoder): register mxbai-rerank-{xsmall,base}-v1 small-model options (563991a2)
- feat(cross-encoder): registry-driven catalogue + BGE-reranker-v2-m3 default (3738e4e9)
- feat(m3): return query-level feeling-of-knowing (FOK) on recall (bc11c106)
- feat(cross-encoder): pluggable trait + ensemble support (05a34dae)
- feat(m5): wire gap_topology persistence + scope-filtered query-in-gap (b0be9c0c)
- feat(w4): MssqlRelationalStore backend via tiberius (gated behind mssql feature) (a0af2a0b)
- feat(w7): dataset row CDC events (7759077a)
- feat(w7): dataset MCP tools (rmcp Rust + TypeScript) (a0c14745)
- feat(w7): tabular ingest mode — CSV/JSON -> relational dataset (ee16799c)
- feat(w4): wire a configured relational backend at startup (cutover activation) (e58d5d03)
- feat(storage): ErasedRelationalStore — type-erase any backend to Error=BoxError (eab2305c)
- feat(w4): memories projection auto-uses the configured relational store (439c5a1f)
- feat(w4): SupabaseRelationalStore — thin Postgres wrapper (gated) (2292ccaa)
- feat(w4): PostgresRelationalStore backend (gated behind postgres feature) (4b38bc51)
- feat(w4): wire the memories projection through the RelationalStore trait (gated) (df0e6c36)
- feat(w4): BlockingRelationalStore — sync->async bridge for the projection cutover (fcf8ab30)
- feat(w4): extend RelationalSlowStoreAdapter to the memories projection table (a8b34af2)
- feat(w6): /api/query/plan + /api/query/execute planner HTTP surface (8321b7c9)
- feat(w6): RealGraphQuerier over entity/episode topology (1fb07adf)
- feat(w6): real VamanaProvider for MultiUserMemoryManager via live writer (02e9773a)
- feat(w6): RealRelationalQuerier + RealVectorQuerier + StubGraphQuerier (no HTTP) (85ae7b43)
- feat(sleep-time): V1 stage 2 + V2 stage 1 — orchestrator, worker, handlers, R27/R42/R43/R46/R48/R66 (c09ae4f9)
- feat(vamana): public memory_id_for(vector_id) reverse accessor (a471b354)
- feat(tui): bind ? and F1 to open the embedded docs in the browser (5936c1f9)
- feat(server): host docs sidecar at /docs with optional embedded-docs feature (07b4e120)
- feat(locomo): single-call combined judge + interleaved recall/judge + Llama serverless (5a98a092)
- feat(memory): sleep-time V1 scaffold — async LLM context-block rewriter (ba1835ef)
- feat(retrieval): RLM refiner — optional LLM-driven final-stage reranker (093b986e)
- feat(alignment): expand corpus to 7 more domains (PM/DB/analytics/devops/sec/test/ai_loop) (6d53dd66)
- feat(alignment): alignment-eval dispatches on method + reads pairs.vec cache (2f054e8b)
- feat(facts): destructive purge_facts under FACT_PURGE_LOCKS + reaper (8cbf775b)
- feat(facts): facts_preview_purge — bucketed, audit-logged, read-only (c810f150)
- feat(facts): purged_at field + is_active filter + fact_narratives MCP tool (3a76c445)
- feat(w4): RelationalSlowStoreAdapter — parallel path proving trait against slow-store DDL (e0eb87b1)
- feat(w7): RelationalDatasetStore + LinkStore + HTTP CRUD (f175643b)
- feat(tui): bind ? and F1 to open the embedded docs in the browser (8aa8bfb7)
- feat(server): host docs sidecar at /docs with optional embedded-docs feature (0dfa4907)
- feat(locomo): single-call combined judge + interleaved recall/judge + Llama serverless (7b185843)
- feat(w6): query planner data model + selectivity planner + executor skeleton (foundation) (8d537fed)
- feat(w7): Dataset schema + DatasetStore trait + in-memory impl (foundation) (60fd8bef)
- feat(w4): RelationalStore trait + SqliteRelationalStore (foundation) (7469590d)
- feat(vamana): multi-embedding-kind projections + persist memory_id map via CheckpointStore side data (a7bb7c4e)
- feat(bm25): unify retrieval-side BM25Index with projection-owned Arc<BM25Index> (8e51b687)
- feat(memory): sleep-time V1 scaffold — async LLM context-block rewriter (52e5cdad)
- feat(retrieval): RLM refiner — optional LLM-driven final-stage reranker (e5730c3a)
- feat(intent-log): BM25 as third real TypedProjection (c746d30e)
- feat(intent-log): Vamana as second real TypedProjection (b3983017)
- feat(intent-log): SQLite as first real TypedProjection — journaled writes + replay (c6736478)
- feat(user-auth): admin promote/demote + persistent throttle + session revocation (be11ae9f)
- feat(memory): wire AgentSession into writer paths with detection helper (8d1480cf)
- feat(alignment): expand corpus to 7 more domains (PM/DB/analytics/devops/sec/test/ai_loop) (95ad2e6a)
- feat: wire ACL into reads, LLM into consolidation, schema-constrained extraction (#3) (d30e6718)
- feat(alignment): alignment-eval dispatches on method + reads pairs.vec cache (03cbe708)
- feat(alignment): corpus expansion tooling (MDN/cpython/Rust/React/TS) + multi-lambda Ridge sweep (8dfe2183)
- feat(alignment): Procrustes + Ridge fitters, fit/eval bins, retrieval integration (#2) (68ca2e3f)
- feat(hooks): write .veld-agent-session.<pid> marker on SessionStart (91f6c79a)
- feat: agentic-memory feature parity — bi-temporal, self-paging, HNSW, ranker, connectors, LLM hook, ACL (#1) (1d7c36b7)
- feat(user-auth): password + TOTP + recovery codes behind VELD_USER_AUTH_ENABLED (Phase C) (d2ea6669)
- feat(backup): include intent log + cf_user_auth; add --max-lsn PITR (2a350614)
- feat(intent-log): explicit format_version header + per-record schema_version (f7f9bfc3)
- feat(intent-log): Prometheus metrics + tracing spans (observability) (8f670a2a)
- feat(tui): git tab — branches, worktrees, recent commits (Phase A) (0bfb5683)
- feat(facets): AgentSession — worktree/branch/agent identity (8b82fda4)
- feat(scripts): veld-worktree-spawn — sibling-worktree-per-agent automation (d2d71595)
- feat(intent-log): JournaledWriter — log-first writer-side primitive (e2fbaa94)
- feat(intent-log): typed IntentPayload with bincode encode/decode (3173a160)
- feat(intent-log): Projection trait + replay driver + checkpoint store (eff7987d)

### fix

- fix(mcp-server): make the server typecheckable and tsc-clean (870a5b05)
- fix(tests): initialize SemanticFact purge fields in adaptive_memory_tests (07efd44b)
- fix(relational): bridge_block_on returns Result instead of panicking (1bd5cc1e)
- fix(datasets): dialect-correct relational dataset layer for Postgres and SQL Server (498cbe91)
- fix(m5): clippy explicit_auto_deref at gap-persist call sites (acc42b2d)
- fix(w4): dialect-correct memories DDL/upsert for Postgres/Supabase/MSSQL (38a9667b)
- fix(cross-encoder): track HF main + pre-warm on startup (face12ed)
- fix(tests): complete BoxError migration + facts/visibility in test harness (99719f93)
- fix(build): unblock lib + bin compile (40 errors) (ec1be949)
- fix(retrieval): restore rlm_refiner.rs accidentally dropped in HTML commit (f1759895)
- fix(relational): switch type-erased Error from anyhow::Error to BoxError (StdError-compatible) (130695f2)
- fix(alignment): swap langchain → huggingface/transformers for ai_loop source (50d61252)
- fix(facts): add purged_at/purge_reason to inline-test SemanticFact constructions (afd99b4c)
- fix(w4): close pluggability gaps — Row::from_owned, RelationalBackend::Custom, docstring example, Cargo dupe (51fb2ab1)
- fix(memory): finish AgentSession writer-read — marker file + writer paths + e2e test (68628d28)
- fix(alignment): swap langchain → huggingface/transformers for ai_loop source (4da67519)
- fix: brand whitelist + argon2 semaphore + 503-on-disabled-auth + recovery-code charset audit (ddc6e2da)
- fix(alignment): paths-relative-to-CWD in TOML specs; add pairs.vec cache (7822fc57)
- fix(intent-log): repair conflict-resolution build breaks (b93dfe85)
- fix(user-auth): session middleware sets Extension<AuthenticatedUser> (f3a71b07)

### misc

- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (d86677b1)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (47ad6a40)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (65686cfa)
- Merge branch 'main' into reconcile/unify-mainline (e0fae005)
- Merge remote-tracking branch 'origin/main' into remediation/prediction-error (af1d35d0)
- Merge origin/main into local main: facts + sleep-time + relational + intent-log (9e0ca473)

### monitor

- monitor: working Tauri GUI — IPC, tray, X-quit, diagnostics (#4) (dd99c57f)

### nomic

- nomic: unblock concurrent inference + dynamic seq-len + split metrics + hook debounce (#6) (3c342b16)

### packaging

- packaging: add cross-platform service installation scripts (37238cf7)
- packaging: add cross-platform service installation scripts (da8c3ee5)

### perf

- perf(retrieval): reuse stored embeddings in bi-encoder rerank (4ceaf465)
- perf(retrieval): reuse stored embeddings in bi-encoder rerank (636e0f09)

### refactor

- refactor(refiner)!: rename RLM->LLM refiner; consolidate duplicate module (30ec59b5)

### test

- test(reconsolidation): Layer-0 expiry gate + drain unit tests (passing) (eb1ab427)
- test(stress): gate brutal_stress_tests behind a stress feature (746c2a40)

## v0.7.18

### ci

- ci: install NSIS via choco before makensis-action (07343231)

### feat

- feat(embeddings): alignment scaffold + collect bin (Phase 1) (fac25c5d)

## v0.7.17

### feat

- feat(intent-log): durable checksummed append-only log (W5 scaffold) (b2eada2a)

### fix

- fix(security): TLS posture warning + keyed-store eviction sweeper (e09d7468)
- fix(security): structural circuit breaker on the public router (669a58e5)

### misc

- Fix veld hook commit decode error + add schema regression tests (346d5fee)

## v0.7.14

### build

- build(release): autobump pre-commit hook + finish v0.7.10 packaging (baa78bf4)

### feat

- feat(memory): add Orientation to the facets re-export (61867691)
- feat(memory): geospatial enrichment — LocalFrame, GeoFix, Orientation (37a0a481)
- feat(python): memory-to-dict reads via resolved_* (W3 readers) (2858e897)
- feat(retrieval+storage): spatial sort + RocksDB indices via resolved_* (W3 readers) (a1c6ae58)
- feat(query): Query::matches reads via resolved_* accessors (W3 readers) (feafb771)

### fix

- fix(security): three W2 hardening items (rate-limit clamp, audit, fixed-buffer compare) (73b853c0)

### misc

- Fix CI release + memory hook auth + add bump-version tests (e373a5cb)
- Merge branch 'w3/reader-switch' (7b517ccd)

## v0.7.8

First tagged release tracked by this changelog. For pre-v0.7.8 history see `git log`.
