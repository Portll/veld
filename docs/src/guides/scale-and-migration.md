<!-- GENERATED FILE — do not edit by hand.
     Source: CLAUDE.md (## Scale & Migration)
     Generator: docs/generators/src/bin/gen-claude-sections.rs
     Regenerate: bash docs/regenerate.sh
     (or, inside docs/generators/, run `cargo run --bin gen-claude-sections`) -->

# Scale & Migration

> Mirrored from CLAUDE.md. Edit the source there, not this file.

How veld behaves as data grows, and the migration paths between tiers / backends / schemas.

### Retrieval scale (cheap → expensive)

| Memory count | Default retrieval path |
|---|---|
| < 1 000 | Working/Session brute-force cosine (Layer 3.5) covers everything |
| 1 000 – 50 000 | HNSW + BM25 (Tantivy) hybrid kicks in for the LongTerm tier |
| 50 000 – 500 000 | With reranking on (it is off by default), the cross-encoder budget (top 30 of a 50-candidate pool, 50% blend) is the dominant cost; consider tuning [`CognitiveDials`](docs/src/architecture/dials.md) to reduce rerank K |
| 500 000+ | Vamana / SPANN large-collection indexes available in [src/vector_db/](src/vector_db/); SQLite slow store for gap topology queries |

### Tier promotion (drives what's hot vs cold)

`Working → Session → LongTerm → Archive`. Promotion is `age × importance × access_count` against the `TIER_PROMOTION_*` constants in [src/constants.rs](src/constants.rs). Archive-tier memories stay indexed in HNSW but are excluded from default `/api/recall`; surface them explicitly with `/api/recall?include_archived=true`.

### Storage backend migration (RocksDB and redb, on demand)

`veld migrate` copies one engine's stores into the other inside the same data directory and never modifies the source. Each unit is verified by frame count and SHA-256 digest. It needs a build with both engines (the default build). Try it first with a dry run, which writes nothing:

```bash
veld migrate --from rocksdb --to redb --user all --dry-run
```

`veld export` and `veld import` use the same VELDPORT codec ([src/storage/portable.rs](src/storage/portable.rs)). Migrating an encrypted tenant needs `VELD_MASTER_PASSPHRASE` (or `VELD_KMS_WRAP_KEY`), including for the dry run. The boot guard ([src/storage/boot_guard.rs](src/storage/boot_guard.rs)) refuses to serve an empty store while the other engine holds data, so a forgotten migration fails loudly instead of serving nothing. Back up before migrating: `POST /api/backup/create` covers RocksDB; for redb use `veld export`.

Engine speed is unmeasured in this document. The comparison benches are [benches/storage_engine_benchmarks.rs](benches/storage_engine_benchmarks.rs), being added on the redb-parity branch; do not quote a figure until they have run.

See [decision 0001](docs/src/decisions/0001-redb-migration.md) (amended) and [decision 0007](docs/src/decisions/0007-rocksdb-default-engine.md) for context; src/storage/README.md is canonical.

### Record-schema migration

`Memory` is bincode-encoded by position, and so is every type reachable from it, `Experience` included. `#[serde(default)]` does nothing there: a field added, removed or reordered on any of those types makes every record written before it undecodable (4a54053 appended two fields to `AgentSession` and 24 live records stopped decoding). `tests/persisted_layout.rs` snapshots the traced layout and fails at the commit that changes it.

In order of preference:

1. **Keep the data off the record.** Write-time-only data travels as a parameter, as the graph build's `Option<&TagSources>` does. Data that must persist can live beside the record in the user's key-value keyspace, keyed by memory id, as `memory::tag_sources` does.
2. **If it must be on the record**, add a legacy decode arm for the old shape (the `Legacy*` structs in [src/memory/storage.rs](src/memory/storage.rs)), then accept the new layout as the header of `tests/persisted_layout.rs` describes.

The legacy `bincode1` format support in [Cargo.toml](Cargo.toml) handles pre-2.0 memories. These changes are rare and should land paired with a `## Schema Version & CHANGELOG` entry once that section exists (LLM-Wiki Phase 1).

### Distribution-binary scaling

The default release binary is ~30 MB. `--features embedded-docs` adds ~3-5 MB for the mdBook output. `--features fortress` strips tracing and adds obfuscation (smaller binary, harder to debug). `--features telemetry` adds OpenTelemetry exporters (~200 extra crates) — only enable in cloud / multi-node deployments where you want distributed tracing.

---
