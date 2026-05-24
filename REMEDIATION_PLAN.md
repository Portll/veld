# Veld — Remediation & Architecture Plan

Consolidated from the security-review, shipcheck, breakers, and portll-security
audits run 2026-05-21, plus the architecture direction set in the same session.
This is the canonical execution plan.

## Status — 2026-05-21

Done and pushed to `main`:

| Commit | Work |
|--------|------|
| `8f5528c` | Nomic promoted to primary embedder — Matryoshka truncation + asymmetric query encoding |
| `8a68811` | Resettable GCRA rate limiter + admin reset endpoint (ported from `feat/integrate-this`) |
| `6ee4589` | Security fixes — system-level-only `/health/ready`; admin-key prefix removed from logs |
| `b5a80a0` | Bifocal+ evaluation artefact of this plan |
| `78f70fd` | CI workflow on `main` (build/clippy/test, Linux/macOS/Windows); RECTIFICATION.md residual fixes |

**Build verification now runs on GitHub Actions** — local disk constraints no longer block it.

## Findings ledger

| Finding | Source | Severity | Status |
|---------|--------|----------|--------|
| `health_ready` unauthenticated per-tenant disclosure | security-review | Medium | Fixed `6ee4589` |
| Admin-key prefix in WARN log | breakers | Medium | Fixed `6ee4589` |
| `health_index` — same disclosure class, twin of `health_ready` | portll-security | Medium | **W2** |
| No structural rule keeping per-user handlers off the public router | portll-security | — | **W2 (A3)** |
| `VELD_RATE_LIMIT` unbounded → `build_quota` startup panic | breakers | Low | **W2** |
| Cross-tenant `resolve_request_user_id` rejection not audited | portll-security | Low-Med | **W2** |
| No TLS enforcement/warning for non-localhost binds | portll-security | Medium | **W2** |
| `admin.rs` key-length timing oracle | breakers | Low | **W2** |
| `health_index` raw internal error to unauthenticated caller | portll-security | Low-Med | **W2** |
| Keyed rate-limit store unbounded growth | breakers | Low | **W2** |
| `BlockingApiClient` timeout-build fallback drops the timeout | breakers | Low | **W2** |
| RECTIFICATION.md backlog (~40 items) | portll-security | — | **Done** — agents verified; lanes 2/4/5 already fixed by prior commits, lanes 1/3 residual fixed `78f70fd` |
| Core model bloat — `Experience`/`Memory`/`Query` kitchen-sink structs | gap analysis | — | **W3** |
| Two unrelated git histories (`main` 19 / `feat/integrate-this` 846) | preeminence analysis | — | **W1** |

## Workstreams

Post-hardening sequence (Bifocal+ pass moved W5 before W4):
`W0 → {W1, W2} → W3 → W5 → W4 → W6 → W7`.

### W0 — Build verification — **DONE**
GitHub Actions CI on `main`: version-check, format, clippy, build (3-OS matrix), test.
Runners carry the disk + libclang the local machine lacks. Recommend marking the
`summary` job a required status check in branch protection.

### W1 — Repo coherence
`main` (19 commits) and `feat/integrate-this` (846) have **unrelated histories** — zero
common ancestor. Keep `main` as canonical; import the feature branch's unique *content*
(not history); retire the branch. One Veld lineage.

### W2 — Security remediation
Close the remaining ledger items. Headline: `health_index` is the unfixed twin of the
fixed `health_ready` bug — strip its `?user_id=` branch from the public router; add
per-user health on an authenticated `/api/health/*` route resolving `user_id` via
`resolve_request_user_id` (the proven `delete_memory` pattern from `ba4c508`).
**A3 — structural circuit breaker:** build `build_public_routes` from an explicit
`PUBLIC_PATHS` const + a test asserting no public handler reads `?user_id=` for
per-tenant data. Stops the next occurrence. Plus: clamp `VELD_RATE_LIMIT`; audit
cross-tenant rejections; TLS posture warning; scrub public error bodies; fixed-buffer
key compare; keyed-store eviction.

### W2b — RECTIFICATION.md burn-down — **DONE**
Five non-overlapping agent lanes. Lanes 2 (auth), 4 (MCP/hooks), 5 (errors/audit) were
already fully fixed by prior commits. Lanes 1 (TUI UTF-8) and 3 (graph concurrency)
had residual items — fixed in `78f70fd`. Two graph perf items (merge three locks into
one struct; `Vec`→`HashMap` embedding cache) deferred as refactors needing compiler
verification.

### W3 — Minimal core + facet refactor — **IN PROGRESS**
Shrink `Experience`/`Memory`/`Query`; move domain data into typed facets attached only
when relevant, so adding a domain never grows the core type. See the design section
below. Scaffold landed in `src/memory/facets.rs`.

**2026-05-21 revision.** Refined by the neuroscience-driven 5-W design
([docs/neuroscience-5w-memory-design.md](docs/neuroscience-5w-memory-design.md)):
the first-pass `RepositoryContext` has been absorbed as `Place::Repo` inside a
new layered `WhereFacet`. WHAT and WHEN will fold into the minimal core;
WHERE / WHO / WHY / EngramBinding become the optional facets. Subsequent W3
steps assume the 5-W layout.

### W5 — Log-structured projection layer (cross-store consistency)
Veld already treats RocksDB as truth and Vamana/BM25/SQLite as rebuildable projections.
Formalize it: a durable, checksummed intent-log; projections are checkpointed and
replayed from the log; corruption recovers by truncating to the last valid entry.
Replaces today's ad-hoc best-effort sync. Precedes W4 so Postgres writes are
idempotent and replayable.

### W4 — `RelationalStore` trait + Postgres / Supabase
Abstract the SQLite slow store behind a `RelationalStore` trait; implement PostgreSQL
(pgvector + Apache AGE) — a lever to collapse relational + vector + graph toward one
engine. Supabase = hosted Postgres deployment target. MySQL not targeted.

### W6 — Query planner
A cross-store planner that *joins* relational ∧ vector ∧ graph predicates rather than
only rank-fusing (RRF). Per-stage result caps bound fan-out.

### W7 — First-class tabular datasets
A `Dataset` type (schema + rows) stored as real tables in Postgres; rows link to graph
entities by reference. Flat/tabular data survives ingestion as flat data instead of
being shredded into per-row memories.

## W3 design — `RecordKind` + facets

**Problem.** `Experience` carries five embedding kinds, multimodal refs, causal chains,
emotional/source/episode context, and robotics/mission fields; `Memory` adds ~18 more;
`Query` has ~40. Every record pays for every domain — robotics was the first domain so
it got hard-coded into the core types.

**Direction.** A minimal invariant core (`id`, `content`, `created_at`, tenant) plus
typed facets attached only when present. `RichContext` already does this for context
(`ProjectContext`, `CodeContext`, …); the refactor extends the pattern and retires the
flat robotics fields.

**`RecordKind`.** `store Plan / Prompt / Learning / Memory` is modelled as one core
record with a `RecordKind` discriminant, not four top-level types — four types would
fragment retrieval into four stores. Kind-specific data lives in facets
(`PlanFacet`, `PromptFacet`, `LearningFacet`); retrieval stays unified and filters by
kind.

**`RepositoryContext`.** Version-control identity (repo, branch, commit, files,
symbols, PR) — distinct from `CodeContext` (the live editing cursor). It is the
structured form of what `veld hook commit` currently stuffs into free-text + tags, and
the context a coding agent's memory actually needs.

**Scaffold (landed).** `src/memory/facets.rs` defines `RecordKind`, `RepositoryContext`,
`PlanFacet`/`PlanStep`/`PlanStatus`, `PromptFacet`, `LearningFacet` — standalone,
serde-round-tripped, unit-tested.

**Wiring (W3 step 2 — next).**
1. `#[serde(default)] pub kind: RecordKind` on `Memory` (existing rows → `Memory`).
2. `#[serde(default)] pub repository: RepositoryContext` on `RichContext`
   (4 construction sites: `memory/context.rs`, `memory/types.rs`,
   `handlers/remember.rs`, `tests/adaptive_memory_tests.rs`).
3. Attach kind facets behind a `RecordKind`-tagged optional field.
4. Migrate the flat robotics fields on `Experience`/`Query` into a `RobotContext`
   facet behind `#[deprecated]` serde aliases for one release — not a hard cut.
