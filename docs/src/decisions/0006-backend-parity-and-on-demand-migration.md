---
id: 0006
title: Backend parity and on-demand migration
status: proposed
date: 2026-09-11
---

# 0006 — Backend parity and on-demand migration

Supersedes the "Migration mechanics" section of [0001](0001-redb-migration.md), which describes a
mechanism that is not implemented. Everything else in 0001 stands.

## Context — an outage, and what it proved

On 2026-09-10 a release build of veld was deployed to the live server. It reported
`status: healthy`, `users_count: 25`, and returned **zero records** for every recall. Rolled back
inside a minute; no data was lost or touched.

The cause was not a bug in the new code. `Cargo.toml` carries `default = ["storage-redb"]`, the
previous binary had been built without that feature, and `effective_storage_backend_for_current_build`
had therefore been silently downgrading every redb request to RocksDB. The new binary honoured the
request that config had been making all along, opened an empty redb store, and served it.

That falsifies 0001's stated migration mechanic:

> *"Cold-start re-indexing happens on first boot — there is no explicit 'migrate' command. The trait
> abstraction means each operation walks the legacy store on read and writes into Redb on update."*

There is no read-through. Flipping the backend does not migrate anything; it presents an empty
store as a healthy one. A cutover plan that rests on a mechanism nobody implemented is worse than
no plan, because it reads as a decision already taken.

## What is already built — the surprising part

The remediation is much smaller than the outage suggests, because almost all of it exists:

- **Trait parity is complete.** Both backends implement all nine traits — `AuditStore`,
  `BackupStore`, `BatchWriter`, `ColumnStore`, `GraphStore`, `KeyValueStore`, `MigrationStore`,
  `PrimaryMemoryStore`, `StorageFactory`. The lists are identical.
- **The interchange format exists.** `src/storage/portable.rs` is a shared codec with a `VELDPORT`
  magic and a trailer carrying the record count and a SHA-256 digest of every frame, checked before
  a reader returns anything. Its header states the reason precisely: a frame-by-frame reader hits
  EOF on a frame boundary and "reports success over a store missing everything after the cut".
- **Both directions are implemented against that one codec.** `export_portable` exists on the
  RocksDB and redb `MigrationStore` impls, and the redb side additionally verifies via
  `read_trailer` before import.

What is missing is **only the wiring**. Measured: nothing outside `src/storage/` calls
export or import. `MigrationStore` appears outside that directory exactly once, as an unused import
in `user_memory.rs`. There is no CLI subcommand and no HTTP route. A complete, integrity-checked
migration facility is present in the binary and unreachable from outside it.

## Scope

Parity **and** on-demand migration. Four work packages, ordered so each is useful alone.

### WP1 — Refuse the false clean (small, and the one that had to exist yesterday)

A backend serving an empty store while the other backend holds data must not report healthy.

- At startup, after the backend resolves, compare **content**, not file presence. If the selected
  store is empty and the other is non-empty, refuse to serve and name the migrate command.
  Emptiness that blocks a boot is a stronger claim than emptiness that fills a dashboard, so it is
  decided by a key count, never by `Path::exists`.
- **File existence is already known to be worthless here.** Four `.redb` files exist on the live
  box — `shared.redb`, `audit.redb`, `portll/storage.redb`, `portll/graph/graph.redb` — and all four
  are exactly 3,686,400 bytes, redb's empty-file allocation. A check keyed on existence concludes
  redb is initialised and skips migration forever.
- `/health` reports per-backend record counts beside the backend names. `status: healthy` next to
  `users_count: 25` over an empty store is the defect, and the count is what distinguishes them.

### WP2 — Wire the migration that already exists

- CLI: `veld export --user <id|all> --out <path>`, `veld import --in <path>`,
  `veld migrate --from <backend> --to <backend> [--user <id|all>] [--dry-run]`.
- HTTP routes for the same three, admin-key gated.
- **Verification is part of the command, not a follow-up.** Post-import, re-derive the count and
  digest and compare against the trailer; refuse to report success on a mismatch. The codec already
  carries both — the failure mode to avoid is a migration that reports success because it read
  until EOF.
- **The source is never deleted.** Migration writes the target and leaves the origin intact; removal
  is a separate, later, explicit act.
- **`--dry-run` reports counts per user without writing.** The payload is small enough to make this
  cheap: 195 MB of RocksDB across 21 users, of which `portll` is 169 MB.
- A re-POST of every record is **not** an acceptable fallback and must not be offered: it mints new
  ids and timestamps and destroys the history the records exist to carry.

### WP3 — Finish trait adoption, and ratchet it

Trait parity is complete; **adoption** is not. Measured 2026-09-11: 185 `rocksdb::` type uses in
`src/`, of which 18 are inside `src/storage/` and **167 are outside it** — `memory` 91, `handlers`
30, `backup` 16, `memory/sleep_time` 13, `user_auth` 8, `graph_memory` 7, `user_memory` 2. Against
that, 70 references to the storage traits outside `src/storage/`. So the abstraction is real and
half-adopted.

- Burn the 167 down module by module, largest first; `memory` alone is 55% of it.
- **Add a ratchet test**: no new `rocksdb::` outside `src/storage/`, baselined at the current count,
  allowed only to fall. Without it this erodes continuously — see WP4.
- `backup`'s 16 uses are the known gap: the comprehensive backup is a RocksDB `BackupEngine`
  facility, which is why five backup operations refuse under redb. WP2's export/import is the
  substrate that closes it, so WP3 should take `backup` after WP2 lands rather than before.

### WP4 — Upstream discipline

Upstream `shodh-memory` has **131 `rocksdb::` uses and zero redb**, and no storage abstraction. redb
is entirely veld-local. Therefore "matching upstream" can only mean functional parity, never backend
parity, and **every upstream merge re-introduces RocksDB-typed code by construction**. The WP3
ratchet is what makes upstream merges safe to take; without it, parity decays every time we sync.

## Consequences

- veld gains a real cutover story, and 0001's v0.9 target becomes reachable rather than assumed.
- The build-dependency win 0001 was chasing (no libclang, no `scripts/cargo-dev.sh`, smaller binary)
  stays blocked until WP3 completes, because RocksDB cannot be dropped while 167 uses remain.
- WP1 costs a startup refusal that will, correctly, stop a misconfigured server from booting. That
  is the intended behaviour and should not be softened to a warning: a warning is what the current
  `/health` already effectively is.

## Not decided here

**An external interchange format.** The operator has asked to evaluate "OMIR" as the interchange
representation. That name resolves to no standard I can find — the nearest actual artifact is the
**Open Memory Protocol** (a portable JSON format for moving memories between servers). This is
recorded as a backlog item, not a decision, and the name needs confirming before any spec work: the
internal `portable.rs` codec is sufficient for backend-to-backend migration, and an external
interchange format is a different problem with different constraints (lossy by nature, cross-vendor,
versioned independently). Adopting one should not block WP1–WP4.
