# src/storage

The engine-neutral storage layer: the traits the rest of veld reads and writes through, their
redb and RocksDB implementations, and the tools that move data between engines.

| File | What it holds |
| :--- | :--- |
| `mod.rs` | The traits: `ColumnStore`, `KeyValueStore`, `PrimaryMemoryStore`, `GraphStore`, `AuditStore`, `BackupStore`, `MigrationStore`, `BatchWriter`, `StorageFactory`; `SharedEngineDb`. |
| `redb.rs` | The redb implementation of every trait. |
| `legacy_rocksdb.rs` | The RocksDB implementation. Compiled only with `legacy-rocksdb`. |
| `portable.rs` | The VELDPORT interchange codec used by export, import and migrate. |
| `migrate/` | `veld export`, `veld import` and `veld migrate`; each unit is verified by frame count and SHA-256 digest. |
| `boot_guard.rs` | Refuses to serve an empty store while the other engine holds data. |
| `relational/` | The relational store trait and its SQLite, Postgres and SQL Server backends. |
| `test_support.rs` | One store per enabled engine, so a test runs its assertions on each. |

## Engines

- **redb** is the default (`storage-redb`).
- **RocksDB** is built only with `legacy-rocksdb`. Without it the binary contains no RocksDB code,
  and a request for the rocksdb backend refuses at boot, naming the feature.

## Moving data between engines

`veld migrate` copies one engine's stores into the other in the same data directory and never
modifies the source. It needs a build with both engines:

```bash
veld migrate --from rocksdb --to redb --user all --dry-run
```

## What only RocksDB does

- The RocksDB backup engine in `src/backup.rs` (redb backups go through export and import).
- The RocksDB `MemoryStorage` in `src/memory/storage.rs`.
- Record-level at-rest encryption (encryption-v2); redb refuses encrypted records.

On redb, `PrimaryMemoryStore::run_merge_pass` and `gc_merge_tombstones` return an error
(semantic merge is off by default).
