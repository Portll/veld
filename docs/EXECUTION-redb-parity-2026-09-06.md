# redb parity: what remains, dependency-ordered

Execution prompt of 2026-09-06, written at `c5a560b`. Every figure below was
measured today against this tree, not carried forward from the earlier handoffs.
Where a number moved in the *wrong* direction it says so, because a migration
that reports only its wins is how the last stretch of one gets mis-scoped.

Companion: `docs/HANDOFF-storage-remaining-2026-09-02.md` holds the design
rationale for the trait split and the batch API. This document holds only the
work that is left.

---

## Where this starts

**redb boots.** As of `4d25af6` a request for redb is honoured
(`effective_storage_backend_for_current_build`), all four construction paths are
wired — shared bundle, backup engine, `MemorySystem`, `GraphMemory` — and both
halves of a redb tenant are exercised by executed tests, not compiles.

**What still refuses, and must keep refusing:** the five backup *operations*
(`create`, `list`, `verify`, `purge`, `restore`). The comprehensive backup is a
rocksdb `BackupEngine` facility with no redb analogue; its redb route is
export/import (R-6a/b, landed). Each refuses by operation name. Do not
substitute `BackupStore::create_backup` — it is 4/4 on redb and *not*
comprehensive (no secondary stores, no intent log), so the swap makes the gate
pass and silently downgrades the backup. That is `4d1ec01` returning under a new
name.

### The rocksdb population, measured

```
rocksdb:: references outside src/storage/   168
rocksdb:: references inside  src/storage/    18
```

**168 is up from the 152 recorded on 2026-09-02, and that is expected.** The
migration *adds* rocksdb-typed references on purpose: `SharedEngineDb`,
`require_rocksdb`, `rocksdb_for_backup`, the `Option<Arc<DB>>` fields. Each one
converts an assumption into a refusal that names itself. Counting them as debt
would argue for deleting the guards, so partition by removal route instead of
tracking the total:

| route | approx. refs | what happens to them |
|---|---|---|
| A. Dies with the module | ~60 | `cf_descriptors`, `Options`, `DBCompressionType`, `Cache` — open-time schema and tuning inside stores already ported. Falls out when the rocksdb path is cfg-gated. |
| B. Deliberate guards — **keep** | ~25 | `SharedEngineDb`, `require_rocksdb`, `rocksdb_for_backup`, `backup.rs`. These are the refusals. They stay under `#[cfg(feature = "storage-rocksdb")]`. |
| C. Permanently rocksdb, cfg-gated | ~35 | `migrate_from_separate_dbs` in five stores: rocksdb-to-rocksdb imports from superseded directories. No meaning on another backend; gate, do not port. |
| D. Genuinely unported | ~35 | Step 1 below. |
| E. Tests | ~13 | Follow their subject. |

---

## The work

### 1. Port the four `sleep_time` stores and `merge.rs`

**Blocked by:** nothing. **Unblocks:** 4 (the cfg-gate).

These were never in the earlier port scope and are the last stores holding a
raw handle:

| module | `Arc<DB>` field | raw `self.db.` calls |
|---|---|---|
| `memory/sleep_time/queue.rs` | yes | 6 |
| `memory/sleep_time/supersession.rs` | yes | 4 |
| `memory/sleep_time/policies.rs` | yes | 0 |
| `memory/sleep_time/graduation.rs` | yes | 0 |
| `memory/merge.rs` | no | 6 |

Follow `memory/files.rs` (`26d2473`) for a multi-family store and
`memory/temporal_facts.rs` (`4a0c0c1`) for a default-keyspace one. `policies`
and `graduation` hold the field without using it — check whether they need the
handle at all before giving them a trait object.

**The trap that has bitten twice here:** `supersession.rs:204` and
`todos.rs:274` were once reported as multi-CF batches and are not — both are two
operations on the *same* family. Before preserving a batch as cross-family,
check which families it actually names. And the inverse still holds for
`prospective.rs`: `update()` and `delete()` *are* single cross-family commits,
and splitting them opens a window where a task and its indices disagree, with
nothing at the call site looking different.

**Verification owed:** each module's own suite, plus
`storage::backend_coverage_matrix` unchanged.

### 2. Finish the recall harness

**Blocked by:** nothing. **Unblocks:** the CI gate in 3.

Four of twelve modules are live (`metrics`, `fixtures`, `report`). Enabling the
rest against `LayerMode` (landed `046cd25`) took the compile errors 45 → 16, and
the remainder is specific:

- **`runner`** — five instrumentation sites needing two capture modules veld
  lacks: `memory::gold_funnel` (`begin`/`take`) and `memory::fusion_features`
  (`begin`/`take` + a jsonl dump). These are *diagnostic capture, not ranking*,
  so porting them does not move retrieval behaviour. Port them for real; a
  no-op capture that returns empty diagnostics is a harness reporting on
  nothing.
- **Six ablation harnesses** (`bridge`, `forgetting`, `lineage`, `multihop`,
  `ontology`, `temporal`) — need only `runner`'s helpers (`build_manager`,
  `ingest_corpus`, `run_smoke_suite_with_ranks`, `RunInputs`, `EVAL_USER`,
  `HARNESS_CLOCK_ANCHOR`). They unblock together with it.
- **`bridge_harness`** — five `UserMemory` → `MemorySystem` unwrappings;
  upstream has no such wrapper. `as_memory_system()`.
- **`decay_sim`** — `RelationshipEdge::synthetic_for_sim`, an upstream test
  constructor.

**Verification owed, and it is the point:** `runner` must be *executed* against
`tests/recall/corpora/` and produce a report, not merely compile. ~2,900 lines
landed unrun is the same defect as a CI gate that cannot build.

### 3. Make the recall gate real

**Blocked by:** 2.

veld's `recall-guard.yml` curls a live `127.0.0.1:3030` and cannot run on a
hosted runner. It has **never executed** — zero runs on record — while upstream's
LoCoMo gate ran four times in the week to 2026-09-06. Import upstream's
`recall.yml` (`on: pull_request`, builds `recall-eval`, compares against a
checked-in baseline) *once `recall-eval` builds here*, and either retire
`recall-guard.yml` or keep it explicitly as the live-instance latency probe it
actually is.

Record a first baseline from an executed run. A baseline copied from upstream
would be a number this codebase has never produced.

### 4. R-3: cfg-gate rocksdb out of the default build

**Blocked by:** 1, and by keeping route B intact.

Introduce `storage-rocksdb` as a feature; make `storage-redb` the default. Route
A falls out, route C gates, route B moves behind the feature and keeps refusing.

**Verification is not "it builds".** A build succeeding without libclang does
not prove the default feature set dropped rocksdb: check that no dev-dependency,
workspace member, or transitively-enabled feature pulls it back in
(`cargo tree -e features -i rocksdb` with default features). The payoff is
concrete — plain `cargo build --release` currently fails on macOS because
librocksdb-sys runs bindgen and libclang is not found, and the release build is
~9m dominated by rocksdb.

### 5. Backup parity, or a documented asymmetry

**Blocked by:** 4 (or run in parallel — it touches only `backup.rs` and the five
refusals).

Two honest routes, and this is a decision rather than a refactor:

- **Widen `BackupStore`** so the comprehensive call is expressible on both
  backends, with redb's implementation built on export/import.
- **Declare the asymmetry**: rocksdb deployments get incremental engine
  backups, redb deployments get export/import, and `/health` says which.

Either is defensible. What is not defensible is the five refusals quietly
remaining once redb is the default — an operator whose backups have never run
should be told at boot, not at restore.

### 6. Optional, and listed last on purpose

`run_merge_pass` / `gc_merge_tombstones` on redb — the last 2 of 44 in the
coverage matrix. Both sit behind a default-off flag on a maintenance cycle, and
porting them needs `has_any_inbound_refs`, which chains into three more keyspace
scans. It looks like the obvious remaining gap and moves nothing toward live.

---

## Standing rules for this migration

- **A guard needs a second witness that cannot share its failure mode.** The
  coverage matrix (44 probes, rocksdb 44/44, redb 42/44) is executable, not a
  static count of `unbuilt(`. Keep it that way, and grow it with the trait.
- **Ascending scan order is trait contract**, asserted hard in the conformance
  script — the due-index early exits depend on it, and two backends agreeing on a
  wrong order would pass an equality comparison.
- **redb permits one opener per file.** Anything constructed by path cannot
  coexist with an open store over the same file; `with_store` / `from_database`
  is the constructor a running server uses.
- **Do not re-add the hand-written `starts_with` prefix breaks.** `scan_cf` and
  `scan_prefix` enforce the bound. But do not remove an *ordering* early exit
  either — the due-index breaks are exits at the first future timestamp, not
  prefix guards, and they are load-bearing.
- **Measure before citing.** Every number in this document has a command behind
  it; re-run it rather than quoting this file in three weeks.
