---
id: 0009
title: Embeddings out of the memory record, into a vector table of their own
status: proposed
date: 2026-10-22
---

# 0009 — Embeddings out of the memory record, into a vector table of their own

Burndown task 6.12(c) of plan `veld-audit-remediation-20261022` asked whether a memory's
embeddings should leave its record for a column family (RocksDB) or table (redb) of their own,
stored as raw little-endian `f32` keyed by the 16-byte memory id. It was to be built only if the
move could be made lazily and backward-compatibly on both engines, with the persisted-layout
tests updated on purpose and the boot guard unaffected; otherwise this record was to be written
instead. This is that record. Nothing in the tree implements it yet.

## Context

- `Experience` carries five vector fields inline
  ([src/memory/types.rs:932](../../../src/memory/types.rs#L932) to `:962`): `embeddings`,
  `embeddings_secondary` and the three media vectors. The record is the whole `Memory`,
  bincode-encoded ([src/memory/storage.rs:1711](../../../src/memory/storage.rs#L1711)
  `serialize_memory`; [src/storage/redb.rs:995](../../../src/storage/redb.rs#L995) `seal`), so a
  768-dimension primary plus a 384-dimension secondary puts about 4.6 KB of floats in every record.
- Most writes to a stored memory change metadata only: access counts
  (`update_access`, [src/memory/storage.rs:3811](../../../src/memory/storage.rs#L3811)),
  importance, tier, anchoring, tags, the interference decay. Each one rewrites the floats with it,
  and on RocksDB each rewrite is compacted again later. Reads that need no vector (listing,
  filtering, the index lookups that hydrate candidates) decode them all the same.
- Floats compress poorly under either engine's block compression, so the codec change of 6.14
  does not recover this.

## The design, if it is built

1. **Storage.** A `vectors` family/table beside `memories`, keyed by the 16-byte memory id. Each
   value is a small versioned envelope (version byte, then per field a tag, a `u32` length and the
   `f32` values little-endian) rather than bare floats, so the secondary and media vectors share
   one row and a later field needs no new table. On an encrypted tenant the envelope is sealed
   with the record cryptor of the current keystore epoch, exactly as the record is: an embedding
   leaks enough of its text to be reconstructed in part, so a plaintext side row would undo
   encryption-v2 for every sealed record. "Raw little-endian `f32`" holds only for plaintext
   tenants.
2. **Record.** `Experience` keeps its fields, so the bincode layout pinned by
   `tests/persisted_layout.rs` does not move. A migrated record stores them as `None`; the side
   row holds the values. The side envelope gets its own pinned layout entry in that test.
3. **Reads.** `get`, `get_many`, the scans and `search` hydrate from the side row in the same
   read (one more key in RocksDB's batched multi-get; the same read transaction on redb). A record
   that still carries inline vectors wins over any side row, which is what makes the move lazy.
   A `get_without_vectors` family of reads serves the callers that need none, which is where the
   read-side saving comes from.
4. **Writes.** `store`/`update` already read the previous version on RocksDB to remove its index
   rows (`queue_replaced_index_removals`) and run in one transaction on redb. They compare the
   vectors with the stored ones and rewrite the side row only when they changed, in the same batch
   or transaction as the record, so a metadata update writes the record alone. `delete` removes
   both.
5. **Migration.** Lazy: any write of a memory that still carries inline vectors moves them. An
   optional idempotent sweep behind a done-marker moves the rest, in batches through
   `update_many`, the way the `compressed_data` strip of 6.12(a) runs.
6. **Everything that copies records.** `veld export`/`veld import` (VELDPORT) and `veld migrate`
   carry the table and verify it by frame count and SHA-256 digest like every other unit; the
   RocksDB BackupEngine covers it as one more family; redb backups go through export and import
   already. The intent log is untouched: `Remember`/`Update` frames are snapshots of a hydrated
   `Memory`, so frames, replication followers and the SQLite mirror keep their vectors.
7. **Boot guard.** Unaffected: it asks whether a store holds memories, which are still the
   16-byte keys of `memories` on both engines.

## Why it is not built in this change

- **Every `Memory` read path moves.** Hydration has to reach `get`, `get_many`, `get_all`,
  `search`, the merge pass, recall's candidate hydration, export and the RocksDB scans that decode
  records in place (`get_stats`, the forget passes, `remove_matching`). One path missed returns
  memories without vectors and, silently, re-embeds or drops them from vector recall.
- **A downgrade reads migrated records as unembedded.** A build without the table sees `None`
  vectors: the index rebuild leaves those memories to repair, which re-embeds every one. That is
  recoverable but expensive, and nothing in the record tells an old build it is missing data.
  The move wants a store-level format stamp an older build refuses, as `index_meta` does for redb's
  derived indexes, before the first record migrates.
- **The asked-for format conflicts with encryption-v2**, as point 1 explains, so the format itself
  is a decision for the operator, not a detail of the port.
- **The gain is unmeasured.** `benches/storage_engine_benchmarks.rs` measures trait operations;
  the update-heavy write mix this targets (access bumps, consolidation rewrites) has no bench yet.

## Gates before building it

- A bench of the metadata-update mix on both engines, before and after, on a 100k-memory corpus.
- The store-level format stamp, and the refusal an older build gives when it sees it.
- Sealed side rows on encrypted tenants, with `veld migrate` re-verifying them under the copied
  keystore as it does records.
- Persisted-layout entries for the side envelope; differential tests that a record written either
  way reads back the same `Memory` on both engines; recall-gate equality before and after the
  sweep.

## Consequences

Until this is built, every metadata update keeps rewriting the record's floats. The other storage
work of the same plan does not depend on it.
