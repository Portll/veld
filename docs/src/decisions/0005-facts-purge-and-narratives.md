---
id: 0005
title: Facts purge, narratives, and the purged_at field
status: Accepted
date: 2026-05-29
---

# 0005 — Facts purge, narratives, and the `purged_at` field

## Context

`veld` ports two MCP tools from the `shodh-memory` ancestor:

- **`fact_narratives`** — read-only clustered view of the semantic-fact corpus.
- **`facts_preview_purge`** + **`facts_purge`** — bucketed preview and
  destructive substring purge.

The destructive surface raised five distinct correctness questions that the
eval chain (bifocal+, foureyes, overloop ×2, breakers ×2) interrogated before
any code landed:

1. **Field overloading.** Should purge reuse the existing `valid_until`
   bi-temporal field, or introduce a separate `purged_at`?
2. **Filter enforcement.** Where is the "exclude purged facts" check applied?
3. **Information leakage.** Does the preview surface become an oracle for
   fact existence?
4. **Cross-coupling.** Does purge interact with alignment-fit, sleep-time
   rewriter, Hosaka collective, or context blocks in ways that require
   coordinated cleanup?
5. **Retention.** How long do soft-purged records survive before hard delete?

## Decision

### 1. `purged_at` is orthogonal to `valid_until`

A purged fact carries `purged_at: Some(now())` AND `purge_reason: Some(...)`
on the existing `SemanticFact` record. `valid_until` continues to track
*world-truth invalidation* (set by `detect_and_resolve_contradictions`).
The two semantics are kept separate to avoid a maintainer asking "did this
record become false in the world, or was it administratively removed?".

`PurgeReason` is a closed enum — `UserRequest | AdminCleanup |
PatternMatch{pattern_hash} | ConfidenceFloor{threshold}` — with NO `Other`
variant. Free-form operator notes belong in the audit log keyed by purge
event id, not on the fact record (which is replicated through MIF / backups).

### 2. Filter at the `SemanticFactStore` impl layer

Every public reader method on `SemanticFactStore` routes through the
`is_active(fact, now)` predicate, which returns `false` for either
`valid_until <= now` or `purged_at IS NOT NULL`. The `include_inactive`
parameter on `*_filtered` methods opts in to the full set for forensic and
MIF-export paths.

`as_of(at)` ALWAYS strips `purged_at IS NOT NULL` regardless of the `at`
argument. Time-travel queries cannot become an oracle for purged content.

The enumeration test `tests/fact_narratives_tests.rs::
reader_methods_exclude_purged_facts` asserts that every reader method honors
the filter. New reader methods that bypass `is_active` fail this test.

### 3. Two surfaces, two threat models

| Route                          | Counts        | Audit | Destructive |
|--------------------------------|---------------|-------|-------------|
| `POST /api/facts/preview-purge` | bucketed      | yes   | no          |
| `POST /api/facts/purge`         | exact         | yes   | yes (soft)  |

The preview returns bucketed counts (`None | Few | Some | Many`) to defeat
the existence-oracle: a probing agent cannot enumerate the user's fact
corpus by varying the pattern and reading the count.

The destructive route returns exact counts because it is the explicit
operator path — the bucketing trade-off (operator UX vs preview oracle) is
already paid at the preview boundary.

Both reject unknown JSON fields via `#[serde(deny_unknown_fields)]`. A client
sending `{"dry_run": false}` to `preview-purge` receives a 400 — the
TIER-CREEP guard from breakers (renaming or hot-patching the constraint
requires explicit code review).

### 4. Cross-coupling — what is needed, what is not

| Coupling                  | Action taken                                       |
|---------------------------|----------------------------------------------------|
| Contradiction-detector    | No-op; already reads via `is_active` filter        |
| Sleep-time rewriter       | Out of scope (V1 scaffold only); DIRTY events emitted to audit log for V2 to consume |
| Alignment-fit             | No-op; alignment is fit on a curated `pairs.jsonl`, not the fact corpus |
| Hosaka collective         | **No-op** — collective_store aggregates retrieval weights, not individual facts. Purging in user A does NOT propagate to user B's collective view. |
| Context blocks            | UUID-regex scan; matching candidate ids emit `context_block.dirty` audit events |
| MIF export / backup       | Tombstoned records survive in `include_inactive=true` paths; restore from a pre-purge backup will re-introduce purged facts. Documented in `SECURITY.md`. |

### 5. Retention

Two windows, governed by env vars (default 30 / 90 days):

- `VELD_PURGE_RETENTION_DAYS` — soft-purged record lifetime before reaper
  hard-deletes. Set to a negative value to disable reaping (keep audit
  trail forever).
- `VELD_CONTRADICTION_RETENTION_DAYS` — bi-temporal `valid_until` lifetime
  (existing behavior, unchanged).

Precedence: when both `purged_at` AND `valid_until` are set, the reaper
uses the `purged_at` window (administrative removal supersedes
contradiction history).

The reaper hooks into the existing heavy-cycle maintenance loop
alongside `decay_facts_for_all_users` — no separate scheduler.

## Consequences

**Positive:**
- Field separation is unambiguous; maintainer never has to ask which write
  path set `valid_until`.
- Single `is_active` enforcement point; enumeration test catches drift.
- Preview surface is information-stingy by design.
- Hosaka work avoided (~150 LOC) once the actual structure of
  `collective_store` was inspected — the cross-user replication concern
  was based on a wrong model.

**Negative / deferred:**
- Sleep-time V2 must consume the `context_block.dirty` audit events
  emitted by destructive purges — until it does, dirty blocks accumulate
  without active repair. Acceptable because sleep-time is V1 scaffold.
- `ActiveFact` newtype enforcement (compile-time guarantee that callers
  cannot receive purged records) is deferred — would require ~30
  callsite refactors and is a quality improvement, not a correctness one.
- Restore from a pre-purge backup re-introduces purged facts. Backup-
  intent preservation is a separate PR (audit-log replay during restore).

## References

- Eval artifacts: `evaluations/bifocal-revised-plan-p2-2026-05-29.json`,
  `evaluations/breakers-revised-plan-p1-2026-05-29.json`,
  `evaluations/breakers-revised-plan-p2-final-2026-05-29.json`,
  `evaluations/overloop-revised-plan-p1-2026-05-29.json`,
  `evaluations/overloop-revised-plan-p2-2026-05-29.json`.
- Implementation: commits `23b44ce` (Phase A), `2565507` (Phase B),
  `4a9b308` (Phase C).
