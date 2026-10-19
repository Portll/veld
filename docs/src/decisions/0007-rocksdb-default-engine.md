---
id: 0007
title: RocksDB is the default engine and the port target
status: accepted
date: 2026-10-03
amends: 0001
---

# 0007 — RocksDB is the default engine and the port target

Amends [0001](0001-redb-migration.md): redb stops being the target default for v0.9. It stays a
second engine behind the storage traits, built by default, and becomes the default again only when
the gates below hold. [0006](0006-backend-parity-and-on-demand-migration.md) stands: the traits,
`veld migrate` and the boot guard are how both engines stay usable.

## Context

- veld follows upstream shodh-memory (`docs/upstream/PLAN.md`). Upstream runs on RocksDB and has no
  storage abstraction, so a port tested on RocksDB is tested against upstream's own behaviour.
- redb lacks four capabilities RocksDB has in veld today: the semantic merge pass and its tombstone
  GC (`PrimaryMemoryStore::run_merge_pass`, `gc_merge_tombstones` return an error on redb),
  record-level encryption (encryption-v2; redb refuses encrypted records), the backup engine and
  point-in-time restore in `src/backup.rs`, and a measured recall equivalence.
- The live store ran on redb from 2026-10-02 08:28Z to 2026-10-03 and moved back to RocksDB through
  `veld migrate --from redb --to rocksdb` with writes stopped, every unit verified by frame count and
  digest.

## Decision

1. **RocksDB is the default engine** for a new data directory, and the engine ports are written and
   proven on.
2. **Both engines are built by default.** The cargo features are `rocksdb` and `storage-redb`;
   `legacy-rocksdb` stays as an alias of `rocksdb`. CI tests the default build and a redb-only build
   (`--no-default-features --features storage-redb`), so neither engine rots.
3. **Every port goes through the storage traits.** Where a trait lacks a capability, the trait gains
   it with a RocksDB implementation, and redb returns an explicit refusal that a test names.
4. **redb becomes the default again only when each of these holds**, each shown by a named test or a
   recorded measurement:
   - the merge pass and tombstone GC run on redb;
   - encryption-v2 records are written and read on redb;
   - backup and point-in-time restore on redb produce the same restored store as the RocksDB engine;
   - recall-gate scores are equal on both engines over the same corpus and embedder;
   - a 14-day live soak on redb with no boot-guard refusal and no decode failure.

## Consequences

- The build keeps needing libclang for RocksDB, which 0001 set out to remove. The redb-only build
  stays available for anyone who wants it.
- Upstream ports land without an engine translation step. The ratchet test on direct `rocksdb::` use
  outside `src/storage/` still applies.
- A capability added for RocksDB without a redb implementation shows up as a named refusal, so the
  distance to the gates is visible.

## Backup and restore gate

`tests/backup_restore_parity.rs` is the named test for the backup gate. It writes the same corpus
to both engines, backs it up, verifies the backup, changes the live stores, then restores into a
fresh directory. Each restored store must equal its own engine's snapshot from before the backup,
and the two restored stores must equal each other. It covers the `BackupStore` trait (the primary
store) and the tenant backup behind `/api/backup/*` (primary store, graph and audit log), each with
a plaintext and an encryption-v2 corpus, plus list and purge retention. Point-in-time restore to a
journal LSN (`max_lsn`) is outside this test.

## Recall-gate equivalence

`tests/recall_engine_parity.rs` is the named test for the recall gate. Each of its tests builds
three tenants in one process from one corpus, user id, embedder (MiniLM in hash mode, which needs
no model files or network) and frozen scoring clock: a RocksDB tenant, a second RocksDB tenant as a
same-engine control, and a redb tenant. Every probe runs on all three and must give the same ranked
corpus keys, the same scores, tiers and importances within 1e-6, and the same `SignalAttribution`
and `ScoringFeatures` for each returned memory. A difference between the two RocksDB tenants is
reported as nondeterminism; the cross-engine comparison is not loosened for it.

- `locomo_gate_cases_rank_and_score_identically_on_both_engines`: the LoCoMo gate corpus and its
  100 cases, ingested and queried as `recall-eval` does.
- `structured_recall_is_identical_on_both_engines_through_maintenance_and_forgetting`: 490 gate
  turns given episodes, tags, parents, types and a nine-month timeline, probed by 35 recalls
  (factual, exploratory, time-scoped, ordinal-session, episode, tag, date-range, filtered and
  filter-only), `recall_by_tags`, `recall_by_date` and `proactive_context`, after ingest, after a
  heavy maintenance cycle, and after soft and hard forgetting.
- `sealed_structured_recall_is_identical_on_both_engines`: the same scenario over a 164-memory
  corpus sealed under encryption-v2.

The normalisations each comparison applies are listed at the top of the test file: memory ids
become corpus keys, results tied on score are compared in key order, storage-order reads are
compared as sets, and wall-clock fields and fact ids are dropped.
