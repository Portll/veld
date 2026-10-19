# Storage

Veld's storage is behind a trait abstraction in
[src/storage/mod.rs](https://github.com/Portll/veld/blob/main/src/storage/mod.rs):
`PrimaryMemoryStore`, `GraphStore`, `KeyValueStore`. Two backends implement
these traits.

> **Two engines, RocksDB by default**
>
> Both engines are built by default and RocksDB is the default engine
> ([decision 0007](../decisions/0007-rocksdb-default-engine.md)). Select redb at
> runtime with `VELD_STORAGE_BACKEND=redb`. redb becomes the default only when
> the gates in 0007 hold. The canonical description lives next to the code in
> [src/storage/README.md](https://github.com/Portll/veld/blob/main/src/storage/README.md).

| Backend | Crate | Status |
|---|---|---|
| **RocksDB** | `rocksdb-rs` (RocksDB C++ via FFI) | Default engine. Cargo feature `rocksdb` (alias `legacy-rocksdb`), on by default. |
| **Redb** | `redb` (pure Rust) | Second engine, built by default (feature `storage-redb`), chosen with `VELD_STORAGE_BACKEND=redb`. |

`--no-default-features --features storage-redb` builds redb alone: the binary
contains no RocksDB code, and a request for the rocksdb backend refuses at boot,
naming the feature.

See [decision 0001](../decisions/0001-redb-migration.md) (amended) and
[decision 0007](../decisions/0007-rocksdb-default-engine.md) for why.

## What's stored where

| Data | Trait | Description |
|---|---|---|
| Memory records | `PrimaryMemoryStore` | The canonical record — content, embedding, facets, tier, importance, calibrated confidence, all 20 signal inputs |
| Vector index | `PrimaryMemoryStore::put_vector_mapping` + `src/vector_db/` | HNSW (default), Vamana, SPANN, or PQ — pluggable |
| Knowledge graph | `GraphStore` | `EntityNode`, `RelationshipEdge`, `EpisodicNode` — Hebbian edges |
| Intent log | `KeyValueStore` (append-only) | Event-sourced journal of every agent action; bincode-encoded `IntentPayload` |
| Context blocks | `KeyValueStore` | Letta-style mutable agent state |
| Backups | filesystem | `/api/backup/create` writes a snapshot to disk (RocksDB BackupEngine); on redb, backups are VELDPORT dumps through `veld export` and `veld import` |

## Storage paths

| Platform | Path |
|---|---|
| Linux | `~/.local/share/veld/` |
| macOS | `~/Library/Application Support/veld/` |
| Windows | `%APPDATA%\veld\` |
| Legacy (≤ 0.1.80) | `./veld_data/` (used if it exists in CWD; printed warning) |

Override with `VELD_MEMORY_PATH` env var or `--data-dir` flag.

## Capabilities

`StorageCapabilities::for_backend()` advertises what each backend supports:

| Capability | RocksDB | Redb |
|---|---|---|
| `embedded` | ✓ | ✓ |
| `default_target` | ✓ | ✗ |
| `legacy_compatibility` | ✓ | ✗ |
| `supports_prefix_scan` | ✓ | ✓ |
| `supports_transactional_batch` | ✓ | ✓ |
| `supports_snapshots` | ✓ | ✓ |
| `supports_migrate_in_place` | ✗ | ✗ |
| `supports_shared_multi_tenant_store` | ✓ | ✓ |

Beyond that flag set, the engines differ in a few places:

| Feature | RocksDB | Redb |
|---|---|---|
| Record-level encryption (encryption-v2) | ✓ | ✓ |
| Semantic merge pass and tombstone GC | ✓ | ✓ |
| BackupEngine ([src/backup.rs](https://github.com/Portll/veld/blob/main/src/backup.rs)) | ✓ | ✗ (VELDPORT dumps via export and import) |
| `MemoryStorage` ([src/memory/storage.rs](https://github.com/Portll/veld/blob/main/src/memory/storage.rs)) | ✓ | ✗ |

New code should target the trait surface, not RocksDB-specific concrete types.
Anything that depends on backend-specific behaviour (e.g., RocksDB column
families) must be hidden behind a trait method that Redb also implements, or
return an explicit refusal on redb that a test names.

## Vector index pluggability

`src/vector_db/` ships four ANN implementations:

- **HNSW** — Hierarchical Navigable Small World. Veld's default.
- **Vamana / DiskANN** — disk-resident graph index. Good for very large
  collections.
- **SPANN** — Search-Partition Approximate Nearest Neighbour. Two-level
  partitioned index.
- **PQ** — Product Quantization codec, used for compression alongside any of
  the above.

The vector index is partially independent of the primary store — the mapping
between memory ID and vector ID lives in `PrimaryMemoryStore::put_vector_mapping`.

## Migration story

`veld migrate` copies one engine's stores into the other inside the same data
directory and never modifies the source; each unit is verified by frame count
and SHA-256 digest. It needs a build with both engines. A dry run writes nothing:

```bash
veld migrate --from rocksdb --to redb --user all --dry-run
```

Migrating an encrypted tenant needs `VELD_MASTER_PASSPHRASE` (or
`VELD_KMS_WRAP_KEY`), dry run included. The boot guard refuses to serve an empty
store while the other engine holds data for that user, and its message names
`VELD_STORAGE_BACKEND=redb` or `veld migrate`.

Engine performance is unmeasured here. The comparison benches are
[benches/storage_engine_benchmarks.rs](https://github.com/Portll/veld/blob/main/benches/storage_engine_benchmarks.rs),
being added on the redb-parity branch.

When you write new storage code:
1. Target the trait surface.
2. If you must use a RocksDB-specific feature, hide it behind a capability
   check (`if capabilities.supports_X { ... }`) and give redb either an
   implementation or a named refusal.
3. Test against both backends; `src/storage/test_support.rs` provides one store
   per enabled engine.

## See also

- [Decision 0001, Redb migration](../decisions/0001-redb-migration.md) (amended by 0007)
- [Decision 0007, RocksDB is the default engine](../decisions/0007-rocksdb-default-engine.md)
- [Memory tiers](memory-tiers.md) — what `MemoryTier::Archive` means for storage
- [Consolidation](consolidation.md) — when stored memories are mutated
