# Storage migration: where it actually stands, and the order the evidence forces

Handoff from the sapper sweep of 2026-09-01. Everything below was measured, not estimated; every
number states where it came from. Pick this up cold.

## The one-sentence state

The storage abstraction was **designed correctly and stalled at the `MemorySystem` boundary**: the
traits, the rocksdb adapters and a 1,040-line redb implementation all exist, and the engine selector
is a `const fn` that returns RocksDb from both arms, so nothing has ever been able to select redb.

## What landed

**`6fc3eee` — R-1.** Corrects the `Cargo.toml` note that said "138 source refs to cfg-gate". Measured
198, of which 191 are outside `src/storage/`. It also claimed rocksdb was unconditional only on
non-Windows; there is a second, equally ungated block at `Cargo.toml:511`, so it is unconditional
everywhere. Nothing else has landed.

## The order is A THEN B, not B then A

I recommended B-then-A and the evidence refutes me. Recorded here rather than quietly reversed,
because the reason is the useful part.

**B** = make `MemorySystem` hold `Arc<dyn PrimaryMemoryStore>` instead of `Arc<MemoryStorage>`.
**A** = extend the trait surface (batch, directional cursor, column family).

B looked cheaper. It is bounded — see below — but **two of its three hard escapes need A's batch API
first**, so B cannot complete before A. Do A, then B.

## B, measured

`MemorySystem` holds `long_term_memory: Arc<MemoryStorage>` (`src/memory/mod.rs:222`), taken by
`with_storage_and_bm25_index(config, storage: Arc<MemoryStorage>, …)` (`mod.rs:592-594`).

    24 distinct methods called on it
    94 call sites
     2 files only — src/memory/mod.rs (77), src/memory/recall.rs (17)

Top methods by frequency: `update` 18, `get` 18, `get_all` 8, `delete` 7, `store` 6,
`increment_retrieval_count` 5, `get_all_ids` 4, `save_interference_records` 3, `get_stats` 3,
`db` 3, then a tail of ~14 used once or twice.

`PrimaryMemoryStore` (`src/storage/mod.rs:85`) currently has ~7 methods. **B is therefore ~17 new
trait methods, implemented on both backends, and 94 call sites confined to 2 files.** That is a real
job but a bounded and reviewable one — far smaller than the "191 call sites" headline suggests,
because those 191 are the *gating* problem (R-3), not the *boundary* problem.

### The three escapes, which are the whole difficulty

`.db()` returns `Arc<rocksdb::DB>` (`src/memory/storage.rs:3477`) and is called three times:

1. **`mod.rs:4157`** — GDPR semantic-fact deletion, using `rocksdb::WriteBatch` across key prefixes.
   **Needs A's batch API.** This is the escape that inverts the order.
2. **`mod.rs:4248`** — `pub fn get_db() -> Arc<rocksdb::DB>`, documented "primarily intended for
   backup/restore". A **public** accessor leaking the engine type out of the crate. Should be
   absorbed by `BackupStore`; check for external callers before removing.
3. **`mod.rs:4441`** — `acl::MemoryAclStore::new(self.long_term_memory.db())`; `acl.rs:66` takes
   `Arc<DB>`. Needs porting to a trait (probably `KeyValueStore`, see the model below).

### Follow the model that already exists

`MemoryStorage::key_value_store()` (`src/memory/storage.rs:3489`) already returns
`Arc<dyn crate::storage::KeyValueStore>`, and its doc comment says callers use it so they "stay
portable across the redb cutover". **That is exactly the right pattern and it is already in the
tree** — the ACL store (escape 3) should be moved onto it, and new trait methods should follow its
shape rather than inventing another.

## A, measured

All 7 traits in `src/storage/mod.rs` (`KeyValueStore`, `PrimaryMemoryStore`, `GraphStore`,
`AuditStore`, `BackupStore`, `MigrationStore`, `StorageFactory`) expose only
get/put/delete/scan_prefix/flush-shaped operations. **None exposes batching, directional iteration,
or column families.** The call sites need all three:

- `rocksdb::WriteBatch` — atomic multi-key writes (`supersession.rs:143,204,241,316`,
  `facts.rs:827`, `mod.rs:4158`, …)
- `rocksdb::IteratorMode` / `Direction::Forward` — directional cursors
  (`facts.rs:318,377,431,504,574,920`, `supersession.rs:293`)
- `ColumnFamily` / `ColumnFamilyDescriptor` / `Options` (`supersession.rs:24,89`)

**`StorageCapabilities::supports_transactional_batch` is already declared `true` for BOTH backends**
(`src/storage/mod.rs:32`, set at `:46` and `:56`, asserted at `:182`) **with no API behind it.** A
should start by making that flag mean something.

## What is NOT true, corrected twice during the sweep

- ~~"Windows already excludes rocksdb"~~ — `Cargo.toml:511` gives Windows its own rocksdb block
  (same crate, minus jemalloc). rocksdb compiles on every platform.
- ~~"A half-migrated R-2 risks split-brain between engines"~~ — **false, and this is good news.**
  `legacy_rocksdb.rs:22-33,103-114` shows the adapters wrap the SAME store
  (`RocksDbPrimaryMemoryStore { inner: MemoryStorage }`, plus `from_inner`). Moving a call site onto
  a trait does not change which engine holds the data. There is no dual-write and no atomic cut.
- ~~"R-6 is a data migration to run carefully"~~ — **there is no migration to run.**
  `src/storage/redb.rs:992-1004`: `import_portable` and `migrate_in_place` both return
  `Err(not_implemented(...))` (7 such calls in the file), and the rocksdb side does not implement
  those methods at all. R-6 was struck; see below.

## The ordering hazard, which is the one thing that can take the fleet down

**R-4 (make `config.rs:64-71` honour `requested`) must NEVER land without R-5 (wire the 9
`StorageBackend::Redb => Err(...)` arms in `src/handlers/state.rs`).**

R-4 is a four-line change and looks like the satisfying fix. Landing it alone makes `effective`
become redb, which immediately hits nine bootstrap refusals; startup fails; Veld Desktop respawns
the child 5×/min, trips its crash-loop breaker and stops. **The operator sees a dead server with
`failReason: crash-loop` and no obvious link to a one-line config change.** No data loss — the store
is never opened — but total loss of service. Land R-4 and R-5 as one commit, or gate R-4 so redb
cannot be selected while any refusal arm remains.

## R-6 is refuted and replaced

Do not attempt `migrate_in_place`. The replacement, in order:

    R-6a  implement export_portable (rocksdb side)        additive, non-destructive
    R-6b  implement import_portable (redb side) + verify  into a SCRATCH target
    R-6c  export -> import -> verify -> switch pointer    reversible by config revert
    R-6d  migrate_in_place                                DO NOT IMPLEMENT

Rationale: in-place conversion necessarily creates a window where the rocksdb bytes are consumed and
the redb store is not yet proven readable — under a supervisor that kills and retries 5×/min. Export
→ import → verify → switch keeps rollback to a single config value.

Before any of it: the backup path is **not yet trustworthy**. `create_backup_comprehensive`
(`src/backup.rs:326`) takes an explicit `secondary_stores: &[SecondaryStoreRef]` list, and two
commits inside 48 hours fixed exactly this class — `4d1ec01` "a backup that omitted a store said
success" and `6f01987` "restore: deleting the graph before knowing the replacement copied is not a
restore". Two fixes landing is evidence it is improving, not that it is proven. It has not been
exercised at 2.0 GB across 22 tenants.

Also: stop the server and confirm `shared/LOCK` is released before any tool opens the store. A second
opener fails on it — observed.

## Scale, for sizing anything that touches data

    2.0 GB   total under ~/Library/Application Support/veld
    1.7 GB   the `portll` tenant alone
   ~4,900    memories
  ~16,500    vectors
       22    tenant directories

redb's 5 tests (`src/storage/redb.rs:541-887`) all run against **fresh temp databases**. None reads
rocksdb-written data, none tests concurrency or crash recovery, none approaches this scale. Anything
that wires redb into production is trusting unit round-trips on empty stores.

## Build note

Plain `cargo build --release` FAILS on macOS — libclang is not found, because librocksdb-sys runs
bindgen. Use `./scripts/cargo-dev.sh` (it sources `scripts/setup-libclang-env.sh`). Release build is
~9m10s, dominated by rocksdb. **Removing that toolchain from the default build is the actual payoff
of this whole migration**, and it only arrives at R-3 — after A and B.

## A's shape is decided, and not by the obvious answer

Measured after the handoff was first written, before implementing anything.

**What the batch API has to carry.** Every `rocksdb::WriteBatch` operation in `src/`:

    54  batch.put_cf(...)     column-family scoped
    52  batch.delete_cf(...)  column-family scoped
    15  batch.put(...)
    12  batch.delete(...)
     7  batch.is_empty()
     0  batch.iter()          -- see below; the earlier count of 1 was a name collision
    34  db.write(batch)   +  8  db.write_opt(...)

**80% of batch operations are column-family scoped.** Batch and column-family are therefore not two
independent extensions that can be done in either order — batch requires CF.

> **Corrected on implementation, 2026-09-01.** The first pass of this section reported 71/53 and 42
> writes. Re-measured with anchored patterns: 54/52 and 34. The ratio, which is the load-bearing
> part, is unchanged (82% → 80%) and so is the conclusion. But the raw counts were the numbers a
> reviewer would check, and they were wrong — the original grep matched `put_cf` on receivers other
> than a `batch`.

**The obvious API is wrong, and here is the measurement that says so.** Both `KeyValueStore`
implementations are ALREADY scoped to a single column family: `RocksDbColumnFamilyStore` holds one
`cf` and routes every operation through `self.cf()?` (`legacy_rocksdb.rs:231-247`), and
`RedbKeyValueStore` binds one `KV_TABLE` (`redb.rs:481-493`). So hanging `write_batch` off
`KeyValueStore` looks natural — the store already knows its CF, and no per-op CF field is needed.

That shape cannot express what 2 call sites already do:

    src/memory/prospective.rs:298-362     index_cf + tasks_cf in one batch
    src/memory/prospective.rs:382-408     index_cf + tasks_cf in one batch

Those two write across column families atomically. A store-scoped batch would force them into two
separate batches — which is not the same program. It would turn an atomic cross-CF update into two
updates with a window between them, and nothing at the call site would look different.

> **Corrected on implementation, 2026-09-01.** This said **4**, naming
> `supersession.rs:204` and `todos.rs:274` as well. Both are **single-CF**: `supersession.rs:204-206`
> is `delete_cf(self.cf(), f)` then `delete_cf(self.cf(), r)` — the same family twice — and
> `todos.rs:274-289` is two `put_cf(index_cf, …)`. The original count treated "two CF-scoped ops in
> one block" as "two column families" without comparing the family arguments.
>
> **The design does not change, and that is the point worth keeping.** One multi-CF site is
> sufficient to force a DB-scoped handle; the argument never depended on how many there were. But
> the count was offered as the evidence, and halving it does not weaken a conclusion that was never
> resting on it — which is worth knowing before someone re-derives 2 and assumes the design was
> wrong too.

**So the batch handle must be DB-scoped, not store-scoped**, and `StorageCapabilities::
supports_transactional_batch` must mean "atomic ACROSS column families", because that is the
property those four sites depend on. A version that quietly means "atomic within one CF" would be
true, useless, and indistinguishable from the real thing at the call site.

Suggested shape, for whoever implements A:

    pub enum BatchOp { Put { cf: Option<CfName>, key: Vec<u8>, value: Vec<u8> },
                       Delete { cf: Option<CfName>, key: Vec<u8> } }
    pub struct WriteBatch { ops: Vec<BatchOp> }   // is_empty()/len() cover the 7 is_empty() calls

    // on a DB-scoped handle, NOT on KeyValueStore:
    fn write_batch(&self, batch: &WriteBatch) -> Result<()>;

The one test that makes the capability flag mean something: a batch whose second op must fail leaves
the FIRST op unapplied. Assert it per backend. Without that, `supports_transactional_batch: true`
stays what it is today — a declaration with nothing behind it, which is the habit named below.

~~`batch.iter()` (1 call site) is the only operation this shape does not cover. Find it before
committing to the design.~~

**Resolved 2026-09-01: there is no such call site.** `src/embeddings/alignment.rs:69` is

    fn project_batch(&self, batch: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        batch.iter().map(|v| self.project(v)).collect()
    }

— a slice of embedding vectors, not a `rocksdb::WriteBatch`. The measurement matched on the
variable NAME. So the shape above covers every real operation and the design has no gap.

Worth keeping rather than deleting, because the instruction "find it before committing to the
design" is what made it look already checked. A named open question reads as diligence whether or
not anyone answers it, and this one was load-bearing enough to have blocked the design if taken on
trust in either direction.

## The pattern worth naming

Three independent instances found in one sweep of a declaration written before its wiring:

1. `legacy-rocksdb = []` — an empty feature the TODO says to gate on
2. `StorageCapabilities::supports_transactional_batch: true` — no API behind it
3. `multi-tenant` absent from `default` while nine handlers document tenant binding, and
   `resolve_request_user_id` returns the caller's own `user_id` on a stock build

Not a code defect; a habit. Worth checking for whenever a capability here is *claimed*.
