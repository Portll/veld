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
| 50 000 – 500 000 | Cross-encoder rerank budget (top-20, 18% blend) is the dominant cost; consider tuning [`CognitiveDials`](docs/src/architecture/dials.md) to reduce rerank K |
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

When you add a field on `Memory`, make it backwards-compatible: `Option<T>` with `#[serde(default)]`. The legacy `bincode1` format support in [Cargo.toml](Cargo.toml) handles pre-2.0 memories. Forward-incompatible changes need an explicit migration in [src/storage/](src/storage/); these are rare and should land paired with a `## Schema Version & CHANGELOG` entry once that section exists (LLM-Wiki Phase 1).

### Distribution-binary scaling

The default release binary is ~30 MB. `--features embedded-docs` adds ~3-5 MB for the mdBook output. `--features fortress` strips tracing and adds obfuscation (smaller binary, harder to debug). `--features telemetry` adds OpenTelemetry exporters (~200 extra crates) — only enable in cloud / multi-node deployments where you want distributed tracing.

---
