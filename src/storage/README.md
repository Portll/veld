# src/storage

The engine-neutral storage layer: the traits the rest of veld reads and writes through, their
redb and RocksDB implementations, and the tools that move data between engines.

| File | What it holds |
| :--- | :--- |
| `mod.rs` | The traits: `ColumnStore`, `KeyValueStore`, `PrimaryMemoryStore`, `GraphStore`, `AuditStore`, `BackupStore`, `MigrationStore`, `BatchWriter`, `StorageFactory`; `SharedEngineDb`. |
| `redb.rs` | The redb implementation of every trait. |
| `legacy_rocksdb.rs` | The RocksDB implementation. Compiled with the `rocksdb` feature. |
| `portable.rs` | The VELDPORT interchange codec used by export, import and migrate. |
| `migrate/` | `veld export`, `veld import` and `veld migrate`; each unit is verified by frame count and SHA-256 digest. |
| `boot_guard.rs` | Refuses to serve an empty store while the other engine holds data. |
| `relational/` | The relational store trait and its SQLite, Postgres and SQL Server backends. |
| `test_support.rs` | One store per enabled engine, so a test runs its assertions on each. |

## Engines

- **RocksDB** is the default engine (`rocksdb`, [ADR 0007](../../docs/src/decisions/0007-rocksdb-default-engine.md)).
- **redb** (`storage-redb`) is built by default too and selected with `VELD_STORAGE_BACKEND=redb`.
- `--no-default-features --features storage-redb` builds redb alone: the binary contains no RocksDB
  code, and a request for the rocksdb backend refuses at boot, naming the feature.

## Moving data between engines

`veld migrate` copies one engine's stores into the other in the same data directory and never
modifies the source. It needs a build with both engines:

```bash
veld migrate --from rocksdb --to redb --user all --dry-run
```

## What only RocksDB does

- The RocksDB backup engine in `src/backup.rs` (redb backups go through export and import).
- The RocksDB `MemoryStorage` in `src/memory/storage.rs`.

Record-level encryption (encryption-v2) runs on both engines, and `veld migrate` moves an
encrypted tenant in either direction. Sealed records travel as stored and are never opened to disk;
the keystore is copied from `<user>/storage/keystore.json` (RocksDB) to `<user>/keystore.json` (redb)
or back, and the keystore generation sentinel goes with it. Every sealed record is opened under the
source keystore before the copy and under the copied keystore after it, so migrate, and its dry run,
needs `VELD_MASTER_PASSPHRASE` (or `VELD_KMS_WRAP_KEY`) and refuses without it. A target that
already holds a different keystore is refused. `veld export` still refuses an encrypted RocksDB
store, because a dump does not carry a keystore.

The semantic merge pass and its tombstone GC (`memory::merge`) are one implementation over
`PrimaryMemoryStore` and run on both engines; RocksDB answers the GC's citation check from its
`inbound_ref:` index, redb by reading the memories and facts.
