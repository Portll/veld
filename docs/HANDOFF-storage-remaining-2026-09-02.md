# Storage migration: what remains, in the only order that works

Handoff of 2026-09-02. Every number below was measured today, not carried forward.
Reproduction commands are given because several figures in the previous handoff were
estimates that turned out wrong — see *Corrections* at the end.

Companion document: `STORAGE-MIGRATION-HANDOFF-2026-09-01.md`, which holds the
design rationale (why the batch API is DB-scoped, why R-6a/b precedes R-3). This one
holds only the remaining work.

---

## Status

### Trait coverage — reproduce with the test, do not count `unbuilt(` by hand

```
cargo test --lib storage::backend_coverage_matrix -- --nocapture
```

It CALLS every `PrimaryMemoryStore` method on a live store of each backend and prints
the gap. A static count is a weaker witness: it cannot see a method that compiles, is
reached and fails for another reason, and it counts a refusal written in a comment.

| trait | rocksdb | redb |
|---|---|---|
| `PrimaryMemoryStore` | 33/33 | 31/33 |
| `GraphStore` | 10/10 | 10/10 |
| `AuditStore` | 5/5 | 5/5 |
| `KeyValueStore` | 6/6 | 6/6 |
| `ColumnStore` | 7/7 | 7/7 |
| `BackupStore` | 4/4 | 4/4 |
| `MigrationStore` | 4/4 | 3/4 |
| **TOTAL** | **76/76** | **73/76** |

redb refuses three, and none blocks boot:

- `run_merge_pass`, `gc_merge_tombstones` — behind `if is_heavy` in a maintenance
  cycle, and `run_merge_pass` is a no-op when `SEMANTIC_MERGE_ENABLED == false`,
  which is the default.
- `migrate_in_place` — do-not-implement by design; it refuses and names the
  reversible route instead.

### Secondary stores — the surface that actually gates the bootstrap

`handlers/state.rs:721` builds a per-user bundle whose members are **not in the trait
set**. This, not trait coverage, is what stops redb being selectable.

| store | families | state |
|---|---|---|
| `context_blocks` | 1 | ported → `KeyValueStore` |
| `feedback` | 1 | ported → `KeyValueStore` |
| `files` | 2 | ported → `ColumnStore` |
| **`prospective`** | **2** | **holds `Arc<DB>`** |
| **`todos`** | **3** | **holds `Arc<DB>`** |

### The three gates

```
config.rs  effective_storage_backend_for_current_build   both arms return RocksDb
state.rs   StorageBackend::Redb => Err(...)              9 arms
get_db()   callers                                       3, all the backup path
```

`rocksdb::` references outside `src/storage/`: **152** (191 when this work started).

---

## Remaining work, dependency-ordered

Each item states what unblocks it and what it unblocks. Do not reorder without
re-reading *Why the order is forced*.

### 1. Port `ProspectiveStore` onto `ColumnStore`

**Blocked by:** nothing. **Unblocks:** step 3.

Two families (`CF_PROSPECTIVE`, `CF_PROSPECTIVE_INDEX`), ~35 column-family
operations. Follow `files.rs` (`26d2473`) as the worked example.

**The one thing to be careful about:** `prospective.rs:298` and `:382` each write
`index_cf` and `tasks_cf` in ONE batch. These are the two call sites the DB-scoped
batch API exists for. A port that splits them into two batches turns an atomic
cross-family update into two updates with a window between them, and **nothing at the
call site would look different**. `ColumnStore::write_batch` preserves it; a
per-family handle would not.

### 2. Port `TodoStore` onto `ColumnStore`

**Blocked by:** nothing. **Unblocks:** step 3.

Three families (`CF_TODOS`, `CF_TODO_INDEX`, `CF_PROJECTS`), ~63 operations — the
largest of the five. Same pattern.

### 3. Give the five stores redb implementations

**Blocked by:** 1 and 2. **Unblocks:** step 6.

Once all five take `ColumnStore` / `KeyValueStore`, `RedbColumnStore` already
satisfies them — the work is wiring, not new storage code. Verify with the
conformance test (`storage::column_store_conformance`), which runs one script against
both backends and compares; do not write a second suite per backend, because two
suites passing separately is exactly what a divergence looks like from the inside.

### 4. Express the backup path without `get_db()`

**Blocked by:** nothing. **Unblocks:** step 5.

Three callers: `state.rs:2990`, `state.rs:3020` (that one is `GraphMemory`, a
different type), `legacy_rocksdb.rs:330`.

`state.rs:2990` passes the raw DB to
`create_comprehensive_backup_with_staged_intent_log`. **Do not substitute
`BackupStore::create_backup` for it.** redb's `BackupStore` is 4/4 and is *not*
comprehensive — no secondary stores, no intent log. Swapping them makes the gate pass
and silently downgrades the backup, which is `4d1ec01` ("a backup that omitted a
store said success") returning under a different name.

The honest options are to widen `BackupStore` to carry the comprehensive call, or to
keep an explicit engine-typed accessor named for what it is. Either is a decision, not
a refactor.

### 5. B — flip `MemorySystem::long_term_memory` to `Arc<dyn PrimaryMemoryStore>`

**Blocked by:** 4. **Unblocks:** step 6.

`memory/mod.rs:222` holds `Arc<MemoryStorage>`; 94 call sites across two files
(`memory/mod.rs` 77, `memory/recall.rs` 17). The trait already carries all 24 methods
those sites use, so the flip is mechanical **once `get_db()` is gone** — it is the
last thing pinning the concrete type.

### 6. R-5 — wire the nine bootstrap arms

**Blocked by:** 3 and 5. **Unblocks:** step 7.

`state.rs:1028` is B itself: it calls
`with_storage_and_bm25_index(config, Arc::new(primary_store.into_inner()), …)`, the
concrete type. `:721` needs step 3. `:3030` and `:3182` need step 4.

### 7. R-4 — make `config.rs` honour `requested`

**Blocked by:** 6. **Unblocks:** redb selectable.

Four lines. **Must land in the same commit as step 6, or gated so redb cannot be
selected while any refusal arm remains.** Alone it makes `effective` become redb,
which hits nine bootstrap refusals; startup fails; Veld Desktop respawns the child
5×/min, trips its crash-loop breaker, and stops. The operator sees a dead server with
`failReason: crash-loop` and no obvious link to a one-line config change.

### 8. R-3 — cfg-gate rocksdb out of the default build

**Blocked by:** 7, and by the backup path from step 4.

152 `rocksdb::` references outside `src/storage/`. This is where the payoff is: plain
`cargo build --release` currently fails on macOS because librocksdb-sys runs bindgen
and libclang is not found, and the release build is ~9m10s dominated by rocksdb.

**Verification is not "it builds".** A build succeeding without libclang does not
prove the default feature set dropped rocksdb — check that no dev-dependency,
workspace member or transitively-enabled feature pulls it back in.

### 9. Optional: `run_merge_pass` / `gc_merge_tombstones` on redb

**Blocked by:** nothing. **Unblocks:** nothing.

Listed last because it looks like the obvious remaining gap and is not. Both are
behind a default-off flag on a maintenance cycle. Porting them needs
`has_any_inbound_refs`, which chains into `has_inbound_refs`,
`semantic_fact_cites_memory` and `temporal_fact_cites_memory` — three more keyspace
scans. A subsystem port for zero movement toward live.

---

## Open defect: a test that passes alone and fails in company

`memory::storage::tests::test_deserialize_with_fallback_bincode1_minimal_fixture`

```
cargo test --lib test_deserialize_with_fallback_bincode1     → passes, exit 0
cargo test --lib memory::                                     → fails intermittently
```

```
left:  MemoryId(00000000-0000-002f-4bff-823e814a0ab3)
right: MemoryId(2f4bff82-3e81-4a0a-b300-3337eb6ff43c)
```

The left value is the right one shifted: `2f 4b ff 82 3e 81 4a 0a` appears in both,
displaced by four bytes and zero-padded. That is a decoder reading from the wrong
offset, not a wrong fixture.

**It is inter-test interference and it predates today's ports.** Observed failing
during the `BackupStore` work, before any secondary-store change; absent from five
consecutive runs afterwards. Passing alone and failing in company means it depends on
state another test mutates — a shared bincode config or a decoder global.

Not yet diagnosed. The method that worked on the segfault below applies: run it
adjacent to candidate neighbours rather than reasoning about which global it might be.

## Closed today, recorded because the diagnosis was wrong first

The suite used to SIGSEGV on ~half of parallel runs, always **after**
`test result: ok`. It was recorded in the previous handoff as "consistent with many
rocksdb instances being dropped at process exit" — an inference written before anyone
read the crash report.

It was not rocksdb. `~/Library/Logs/DiagnosticReports/` named the thread:

```
cross-encoder-prewarm-mxbai-rerank-xsmall-v1
EXC_BAD_ACCESS, KERN_INVALID_ADDRESS at 0x17
libonnxruntime: OrtApis::CreateSession → InferenceSession::Initialize
```

`CrossEncoder::prewarm` spawned with `.spawn(…).ok()`, dropping the `JoinHandle`.
Nothing joined it. A server runs for hours and the pre-warm finishes; a test binary
exits seconds after start and tears down mid-session-build. Fixed in `074a6ee`
(`cfg!(test)` early return at both spawn sites); five consecutive runs clean.

**The reusable part:** three steps, ascending in cost and descending in guesswork —
single-threaded runs clean (a race), a base worktree reproducing it (not this work),
then the crash report naming the frame. Only the third said anything unguessable, and
it was available from the first minute.

---

## Why the order is forced

The previous handoff put R-6 last, after R-3. That is backwards, and B is where it
shows. The third `.db()` escape is the backup path, and `src/backup.rs` is built on
`rocksdb::backup::BackupEngine` — a native incremental facility with no redb
analogue. Porting it is not moving a call site behind a trait; it is reimplementing
backup as export/import, which **is** R-6a and R-6b. Those landed today
(`3cbcf0f`, `ad835ab`, `2e8a969`), which is why step 4 is now reachable at all.

Finishing B does not by itself unlock R-3: a trait carrying
`Option<Arc<rocksdb::DB>>` for the backup path pins rocksdb into the default build
exactly as the call sites did.

---

## Corrections carried forward

The previous handoff's measured figures held. Its *characterised* ones did not, and
the pattern is worth stating because it will recur: **what was measured was right;
what was described without running a command was wrong.**

- "4 multi-CF batch blocks" → **2**. `supersession.rs:204` and `todos.rs:274` are two
  operations on the *same* family.
- "`batch.iter()`, 1 uncovered call site" → **0**. It was
  `embeddings/alignment.rs:69`, `project_batch(&self, batch: &[Vec<f32>])` — a slice
  of embedding vectors, matched on the variable name.
- "check for external callers before removing `get_db()`" → there are **three**, all
  the backup path, which is why it does not fall to a refactor.
- "move the ACL store onto `key_value_store()`" → **would have silently orphaned
  every grant.** That handle binds the `kv` family; ACL records are written with bare
  `db.put`, the default keyspace. No error at any point. Hence
  `RocksDbDefaultKeyValueStore`. **Check which keyspace a handle binds before routing
  anything onto it.**
- "the suite segfault is rocksdb teardown" → it was a detached ONNX pre-warm thread.

## Traps for whoever picks this up

- **`test result: ok` with a non-zero exit is not a pass.** Read the summary line and
  the exit code as separate facts.
- **A wrapper's exit code is not the command's.** `timeout … | tail` reports `tail`'s
  status; capture the inner one explicitly.
- **`pgrep -f "cargo"` matches any process whose path contains `cargo`**, including
  `~/.cargo-target/debug/…`. It will tell you a build is running when none is.
- **redb permits one opener per database file.** Any store constructed by path cannot
  coexist with an open store over the same file — `from_database(handle)` is the
  constructor a running server uses. rocksdb reached the same conclusion and makes
  `open_backup_store` refuse outright.
- **Eight hand-written `starts_with` prefix breaks have been removed** from ported
  call sites. rocksdb's iterator genuinely runs past a prefix, so each was
  load-bearing, and dropping one is silent. The trait methods enforce the bound; do
  not re-add the guard, and do not remove one without a trait method behind it.
