---
id: 0001
title: Redb migration target
status: accepted
date: 2026-05-28
---

# 0001 — Redb migration target

## Context

Veld currently uses RocksDB as its persistent store. RocksDB is fast and
battle-tested, but it imposes a heavyweight build dependency: the RocksDB C++
library, linked through `rocksdb-rs`, requires `libclang` on macOS,
platform-specific build flags, and produces a large binary. This conflicts with
veld's "single binary, runs offline, no external dependencies" identity.

[Redb](https://github.com/cberner/redb) is a single-file embedded key-value
store written entirely in Rust. It supports transactional batches, snapshots,
prefix scans, and in-place migration. The build is one Rust dependency.

## Decision

**Migrate veld from RocksDB to Redb as the default storage backend.** Land the
migration in v0.9. Until then, the storage abstraction (`PrimaryMemoryStore`,
`GraphStore`, `KeyValueStore` traits in `src/storage/mod.rs`) presents Redb as
the *requested* default while the *effective* backend remains RocksDB.

Code under `src/storage/legacy_rocksdb.rs` is the active implementation. Code
under `src/storage/redb.rs` is built behind the `storage-redb` feature flag.
All new write paths target the trait surface so the eventual cutover is a
backend swap, not a rewrite.

## Consequences

- **Pro:** removes the libclang/RocksDB build pain on macOS (the
  `./scripts/cargo-dev.sh` workaround stops being necessary post-migration).
- **Pro:** binary size shrinks materially.
- **Pro:** Redb's single-file format is easier to back up, ship between machines,
  and inspect (the file format is documented and stable).
- **Con:** Redb is younger than RocksDB and has a smaller battle-test corpus.
  Concurrent-write performance under heavy load is the primary risk; benchmarks
  during the v0.9 work will gate the cutover.
- **Con:** the migration carries the risk of subtle semantic differences (write
  amplification, fsync behaviour, transaction isolation). Each veld write path
  must be re-tested against Redb before the default flips.

## Status of the abstraction layer

`StorageCapabilities::for_backend()` advertises `supports_transactional_batch:
true` for both backends. New atomicity-requiring features (e.g., the LLM-Wiki
ingest envelope — see [0002](0002-llm-wiki-dual-pathway.md)) should rely on this
capability rather than on RocksDB-specific batch semantics.

## Migration mechanics

Migration is **not** a CLI subcommand today. The flow is:

1. Stop the veld HTTP server (`veld server`).
2. Snapshot the current state via `POST /api/backup/create` while the server
   is still up, or copy `~/.local/share/veld/` (Linux) / equivalent.
3. Once `effective_storage_backend_for_current_build()` flips to `Redb` (v0.9
   target), restart with `VELD_STORAGE_BACKEND=redb veld server`. Cold-start
   re-indexing happens on first boot — there is no explicit "migrate"
   command. The trait abstraction means each operation walks the legacy
   store on read and writes into Redb on update.
4. Rollback: stop the server, restore the backup, restart with
   `VELD_STORAGE_BACKEND=rocksdb` (which remains a valid choice while the
   legacy code path exists).

A dedicated `veld migrate` CLI subcommand is on the v0.9 roadmap but not
yet implemented.

## Related

- See `PROGRESS.md` "Post-v0.8 Target: v0.9 redb integration and public release
  hardening" for the in-flight status.
- The `effective_storage_backend_for_current_build()` function in
  `src/config.rs` is the single switch that flips Redb on once the migration
  passes its acceptance tests.
