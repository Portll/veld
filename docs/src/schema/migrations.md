# Migrations

Veld migrates data in two dimensions independently:

1. **Storage backend** — RocksDB → Redb (see [decision 0001](../decisions/0001-redb-migration.md)).
2. **Record schema** — versioned `Memory` shape evolution (see [changelog](changelog.md)).

## Storage migration: RocksDB → Redb

A dedicated `veld migrate` CLI subcommand is on the v0.9 roadmap but is not
yet implemented. Today, the migration story is:

1. **Stop the veld HTTP server** (`veld server`).
2. **Back up** by calling `POST /api/backup/create` *before* stopping the
   server, or by copying the storage directory directly. See
   [Configuration reference](../reference/config.md) for the default path
   per platform.
3. **Restart with the new backend** once `effective_storage_backend_for_current_build()`
   in [src/config.rs](https://github.com/Portll/veld/blob/main/src/config.rs)
   resolves Redb as effective:

   ```sh
   VELD_STORAGE_BACKEND=redb veld server
   ```

4. **Cold-start re-indexing** happens on first boot. The trait abstraction
   means each operation walks the legacy store on read and writes into Redb
   on update. There is no batch "migrate everything now" step today.

**Rollback:** restart with `VELD_STORAGE_BACKEND=rocksdb` (the legacy path
remains valid). If data is missing after rollback, restore the backup via
`POST /api/backup/restore`.

`v0.9` will gate the cutover behind a fitness suite — the migration must pass
acceptance benchmarks before the default flips.

See [decision 0001](../decisions/0001-redb-migration.md) for context.

## Schema migration: record schema bumps

When `schema_version` increments (e.g., v0 → v1), records are migrated in
two ways depending on the change:

| Change type | Migration approach |
|---|---|
| Additive field (new optional field) | `#[serde(default)]` — no migration needed; existing records deserialize cleanly |
| Renamed field | `#[serde(alias = "old_name")]` — both names accepted during one schema cycle |
| Removed field | Field becomes `Option<_>` for one schema cycle; removed entirely the next |
| Type change | Migration prose in the CHANGELOG entry; runs during consolidation maintenance |

Schema migrations are **not** all-at-once. A consolidation pass after a
schema bump touches each memory it processes and upgrades it incrementally.
Cold memories (never accessed) get migrated lazily on first access.

## See also

- [Decision 0001 — Redb migration](../decisions/0001-redb-migration.md)
- [Schema changelog](changelog.md)
- [Page contract](page-contract.md)
