# Storage migration: where it actually stands, and the order the evidence forces

Handoff from the sapper sweep of 2026-09-01. Everything below was measured, not estimated; every
number states where it came from. Pick this up cold.

## The one-sentence state

The storage abstraction was **designed correctly and stalled at the `MemorySystem` boundary**: the
traits, the rocksdb adapters and a 1,040-line redb implementation all exist, and the engine selector
is a `const fn` that returns RocksDb from both arms, so nothing has ever been able to select redb.

## Where redb actually is — MEASURED TWICE, 2026-09-02

Do not re-derive this by reading the source; there is an executable check.
`cargo test --lib storage::backend_coverage_matrix -- --nocapture` CALLS every
`PrimaryMemoryStore` method on a live store of each backend and prints the gap.

    trait                    rocksdb      redb
    PrimaryMemoryStore         33/33     29/33
    GraphStore                 10/10     10/10
    AuditStore                   5/5       5/5
    KeyValueStore                6/6       6/6
    BackupStore                  4/4       1/4
    MigrationStore               4/4       2/4
    StorageFactory               7/7       7/7
    TOTAL                      69/69     57/69   (at dc8351d)

Two witnesses, agreeing on the count AND the membership: the runtime matrix above,
and an independent parse of the impl blocks counting `unbuilt(`. A static count
alone is weak — it cannot see a method that compiles, is reached, and fails for
another reason, and it counts a refusal written in a comment.

The matrix asserts `rocks_ok >= redb_ok`, because redb reporting MORE working
methods than rocksdb means the probe is wrong rather than the backends; and that the
gap list length equals `33 - redb_ok`, so a method failing on BOTH cannot hide inside
a gap attributed to redb alone. It fails if a method silently stops working and
equally if one silently starts.

**The remaining gap is not evenly distributed, and only one part of it is hard.**
The `BackupStore` three are the whole difficulty: rocksdb's is
`rocksdb::backup::BackupEngine`, a native facility with no redb analogue, so they
have to be rebuilt on portable export/import — which now exists. Everything else is
either mechanical or already answered by design.

## THE TRAITS WERE NEVER THE BOTTLENECK — measured 2026-09-02

redb reached 66/69 on the storage traits tonight. That is not the distance to live,
and treating it as the headline is the mistake this section exists to prevent.

**What the shared-store bootstrap actually needs.** `handlers/state.rs:721` builds a
bundle for a user, and five of its members are not in the trait set at all:

    ContextBlockStore    src/memory/context_blocks.rs    holds a db field
    FileMemoryStore      src/memory/files.rs             holds a db field
    ProspectiveStore     src/memory/prospective.rs       holds a db field
    TodoStore            src/memory/todos.rs             holds a db field
    FeedbackStore        src/memory/feedback.rs          (shared-db variant)

None has a redb implementation, and none is reachable through `StorageFactory`.
`redb.rs` defines eight store types; not one of these is among them. So the 69-method
trait surface is complete-ish while the bootstrap still cannot be satisfied.

**The other three arms, and why one of them is B.**

- `state.rs:1028` calls `UserMemory::with_storage_and_bm25_index(config,
  Arc::new(primary_store.into_inner()), …)` — the concrete type, via `into_inner()`.
  redb literally cannot be passed. **This arm IS B**, and it cannot be wired before
  the field flip.
- `state.rs:3030` / `:3182` are backup create and restore, calling
  `create_comprehensive_backup_with_staged_intent_log`. redb now has a working
  `BackupStore` (4/4), but it is NOT comprehensive — no secondary stores, no intent
  log. Wiring the simple one into the comprehensive call site would be `4d1ec01`
  again: a backup that omitted a store reporting success.

**The dependency chain, in the only order that works:**

    secondary stores get traits + redb impls   ─┐
    backup path expressed without get_db()     ─┤
                                                ├→ B (field flip, state.rs:1028)
                                                └→ R-5 (the 9 arms) → R-4 → selectable

**`rocksdb::` refs outside `src/storage/`: 191 at the start of this work, 150 now.**
That figure is the R-3 headline and it moves slowly on purpose — every one of the 41
removed came with a behaviour check, not a type substitution.

## What landed

**`6fc3eee` — R-1.** Corrects the `Cargo.toml` note that said "138 source refs to cfg-gate". Measured
198, of which 191 are outside `src/storage/`. It also claimed rocksdb was unconditional only on
non-Windows; there is a second, equally ungated block at `Cargo.toml:511`, so it is unconditional
everywhere.

**`8d93d55` / `73d1af1` / `bfa9782` — B, two of three escapes.** The trait gained the 21 methods
`MemorySystem` calls; the GDPR erasure path and the ACL store came off `.db()`. The third escape,
`get_db()`, is the backup path — see the R-6a/b section below for why that reorders the plan.

**`3cbcf0f` / `ad835ab` / `2e8a969` — R-6a/b/c.** Portable export (rocksdb), import (redb, refusing a
non-empty target), and a re-runnable both-directions verify. The dump format carries a record count
AND a digest, because a digest agrees with itself over however few frames survive a truncation.

**`73cbed4` — the corpus marker records WHICH embedder, not just its dimension.** Two models at 384
left `recorded == target` true, so the migration short-circuited over a corpus embedded by something
else, and the per-memory length check was blind for the same reason.

**`dc8351d` / `466a1f0` — redb 22/33 → 29/33** on `PrimaryMemoryStore`.

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

1. ~~**`mod.rs:4157`** — GDPR semantic-fact deletion.~~ **CLOSED, `73d1af1`.** Went through
   `scan_prefix_keys` + A's batch API. Two defects went with it: the prefix bound was hand-written in
   each of two loops (rocksdb's iterator runs PAST the prefix, and a missing break deletes keys
   nobody asked about — in an erasure path), and `.flatten()` was dropping enumeration errors, so a
   prefix that could not be read left its facts in place and the step still logged success.
2. **`mod.rs:4248`** — `pub fn get_db() -> Arc<rocksdb::DB>`. **STILL OPEN and it is the last one.**
   The handoff said "check for external callers before removing"; measured 2026-09-02, it has
   **three**: `handlers/state.rs:2990`, `:3020`, and `legacy_rocksdb.rs:330`. All three are the
   backup path, which is why this escape does not fall to a refactor — see the R-6a/b section.
3. ~~**`mod.rs:4441`** — `acl::MemoryAclStore::new(self.long_term_memory.db())`.~~ **CLOSED,
   `bfa9782`.** It used four operations and never needed a database handle.

   **The obvious port was wrong and would not have failed.** This document said to move it onto
   `MemoryStorage::key_value_store()`. That handle binds the `kv` column family, and ACL records are
   written with bare `db.put` — the DEFAULT keyspace. Routing them through it reads and writes a
   different keyspace: every existing grant comes back absent, every new one lands where nothing
   looks, and no error is raised at any point. A migration wearing a refactor's clothes. Hence
   `RocksDbDefaultKeyValueStore` and `MemoryStorage::default_key_value_store()`.

### Follow the model that already exists

`MemoryStorage::key_value_store()` (`src/memory/storage.rs:3489`) already returns
`Arc<dyn crate::storage::KeyValueStore>`, and its doc comment says callers use it so they "stay
portable across the redb cutover". **The SHAPE is right and it is already in the
tree** — new trait methods should follow it rather than inventing another. But note which keyspace a
given handle binds before routing anything onto it: `key_value_store()` is bound to `kv`, and escape
3 above is what happens when that is assumed rather than checked.

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

## R-6a/b IS A PREREQUISITE FOR R-3, NOT A FOLLOW-ON — corrected 2026-09-02

This document ordered R-6 last, after R-3. That is backwards, and B is where it shows.

Two of the three `.db()` escapes closed cleanly (`73d1af1` GDPR erasure, `bfa9782` ACL store).
The third is `get_db()`, whose three callers are all the backup path, and **`src/backup.rs` is built
on `rocksdb::backup::BackupEngine`** — rocksdb's own incremental backup facility:
`restore_from_latest_backup(…, &rocksdb::backup::RestoreOptions::default(), …)` at `backup.rs:668,677`.
16 `rocksdb::` refs in that file: 6 × `DB`, 4 × `Options`, 2 × `rocksdb::backup`, 1 × `IteratorMode`.

redb has no equivalent. So porting the backup path is not a refactor that moves a call site behind a
trait — it is **reimplementing backup as export/import**, which is exactly R-6a and R-6b.

The consequence that matters for sequencing: **finishing B does not unlock R-3.** A trait carrying
`Option<Arc<rocksdb::DB>>` for the backup path pins rocksdb into the default build just as firmly as
the call sites did, so cfg-gating it out remains impossible. 158 `rocksdb::` refs still stand outside
`src/storage/`.

Real order:

    R-6a  export_portable (rocksdb side)      additive, non-destructive
    R-6b  import_portable (redb side)         into a SCRATCH target
    ...   backup/restore re-expressed on top of those, replacing rocksdb::backup
    B     get_db() removable only then; field flip becomes mechanical
    R-3   cfg-gate rocksdb — reachable only after the above

**Do not port the backup engine under time pressure.** It took two correctness fixes inside 48 hours
(`4d1ec01` a backup that omitted a store said success; `6f01987` restore deleted the graph before
knowing the replacement copied), and it has never been exercised at 2.0 GB across 22 tenants. Two
fixes landing is evidence it is improving, not that it is proven.

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

## The exit segfault: FIXED, and my first diagnosis was wrong

`cargo test --lib memory::` ran 700 green then died with SIGSEGV on ~half of parallel
runs. Fixed 2026-09-02; five consecutive runs clean.

**It was never rocksdb.** This section previously read "consistent with many rocksdb
instances being dropped at process exit" — an inference from "many DBs, dies at exit",
written before anyone looked. The macOS crash report names the actual thread:

    cross-encoder-prewarm-mxbai-rerank-xsmall-v1
    EXC_BAD_ACCESS, KERN_INVALID_ADDRESS at 0x17
    libonnxruntime: OrtApis::CreateSession → InferenceSession::Initialize

`CrossEncoder::prewarm` spawns a DETACHED thread — `.spawn(…).ok()` drops the
JoinHandle, nothing joins it. A server runs for hours and the pre-warm finishes; a
test binary exits seconds after start and tears down mid-CreateSession.

`prewarm` now returns early under `cfg!(test)`. Skipped rather than joined: a
pre-warm you wait for is not a pre-warm, and `ensure_loaded()` is there for a caller
that genuinely needs the model.

**Worth keeping as a method note.** Three cheap steps, in order: single-threaded runs
were 3/3 clean (a race, not a deterministic bug); a base worktree at `85dbc52`
reproduced it (not this programme's work); the crash report named the frame (not the
subsystem I had assumed). The third step is the one that mattered, and it was
available from the start — `~/Library/Logs/DiagnosticReports/`.

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
