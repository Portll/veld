# Storage

Veld's storage is behind a trait abstraction in
[src/storage/mod.rs](https://github.com/Portll/veld/blob/main/src/storage/mod.rs):
`PrimaryMemoryStore`, `GraphStore`, `KeyValueStore`. Two backends implement
these traits.

> **⚠️ Storage backend reality**
>
> The default-*requested* backend is Redb, but the *effective* runtime engine
> is still RocksDB until v0.9 — see [decision 0001](../decisions/0001-redb-migration.md).
> Code that targets the trait surface lands ready for the eventual cutover;
> code that uses RocksDB concrete types directly will need to be reworked.

| Backend | Crate | Status |
|---|---|---|
| **RocksDB** | `rocksdb-rs` (RocksDB C++ via FFI) | Current effective runtime. Legacy. |
| **Redb** | `redb` (pure Rust) | Default-*requested* target. Effective in v0.9. |

See [decision 0001](../decisions/0001-redb-migration.md) for why.

## What's stored where

| Data | Trait | Description |
|---|---|---|
| Memory records | `PrimaryMemoryStore` | The canonical record — content, embedding, facets, tier, importance, calibrated confidence, all 20 signal inputs |
| Vector index | `PrimaryMemoryStore::put_vector_mapping` + `src/vector_db/` | HNSW (default), Vamana, SPANN, or PQ — pluggable |
| Knowledge graph | `GraphStore` | `EntityNode`, `RelationshipEdge`, `EpisodicNode` — Hebbian edges |
| Intent log | `KeyValueStore` (append-only) | Event-sourced journal of every agent action; bincode-encoded `IntentPayload` |
| Context blocks | `KeyValueStore` | Letta-style mutable agent state |
| Backups | filesystem | `/api/backup/create` writes a snapshot to disk |

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
| `default_target` | ✗ | ✓ |
| `legacy_compatibility` | ✓ | ✗ |
| `supports_prefix_scan` | ✓ | ✓ |
| `supports_transactional_batch` | ✓ | ✓ |
| `supports_snapshots` | ✓ | ✓ |
| `supports_migrate_in_place` | ✓ | ✓ |
| `supports_shared_multi_tenant_store` | ✓ | ✓ |

New code should target the trait surface, not RocksDB-specific concrete types.
Anything that depends on backend-specific behaviour (e.g., RocksDB column
families) must be hidden behind a trait method that Redb also implements.

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

Until v0.9 lands, `effective_storage_backend_for_current_build()` in
[src/config.rs](https://github.com/Portll/veld/blob/main/src/config.rs)
silently returns `RocksDb` even when `Redb` is the requested default. This
is intentional — the trait abstraction is being landed first so the eventual
cutover is purely a one-line change.

When you write new storage code:
1. Target the trait surface.
2. If you must use a RocksDB-specific feature, hide it behind a capability
   check (`if capabilities.supports_X { ... }`) and provide a fallback
   Redb-compatible path.
3. Test against both backends if possible.

## See also

- [Decision 0001 — Redb migration](../decisions/0001-redb-migration.md)
- [Memory tiers](memory-tiers.md) — what `MemoryTier::Archive` means for storage
- [Consolidation](consolidation.md) — when stored memories are mutated
