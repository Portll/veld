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
