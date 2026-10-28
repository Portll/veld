# 0007 — redb live soak log

Evidence for the last open gate in [ADR 0007](0007-rocksdb-default-engine.md): fourteen days of
redb serving live data with no boot-guard refusal and no decode failure.

## Vehicle

The `:3031` replica (launchd agent `net.portll.veld-redb-replica`) serves redb from
`~/Library/Application Support/veld-replica-redb` and follows the RocksDB leader on `:3030`
through `/api/admin/replication/*`. It carries every memory the leader writes, so the engine sees
production traffic without the leader switching engines.

A daily check (`~/.config/veld/redb-soak-check.sh`, launchd agent
`net.portll.veld-redb-soak-check`, 04:20 local) appends one line to
`~/Library/Logs/veld/redb-soak-20261021.jsonl` with, for the last day of `replica-redb.log`:
boot-guard refusals, decode failures, panics, leader sync failures, the data directory size, and
the replica's user count beside the leader's.

The gate judges refusals and decode failures only. Panics and leader sync failures are recorded
so a reader can tell a redb fault from a leader fault.

## Clock

| | |
|---|---|
| Start | 2026-10-21T13:29Z |
| Build | main `90572ea9`, built 2026-10-21T13:02:45Z, carrying `feat/redb-parity` (`c76548ec`) and the index tables, content-hash layout and parity fixes from the redb-equivalence plan |
| End of window | 2026-11-04 |

Earlier replica time (2026-10-19 to 13:29Z today) ran the 0.8.4+1 build of 2026-10-20 and does
not count: it predates the parity branch.

## Day 0

Restart was a clean stop (`launchctl kill TERM`, exit 0 after a 64 s flush) and a `kickstart`.
First reading at 13:29:48Z: refusals 0, decode failures 0, panics 0 since the restart,
replication following, 46 users on the replica against 63 on the leader (the difference is
bench, probe and test tenants the follower does not carry), data 18.7 GiB.

Known on the leader side, not counted against redb: the leader's replication feed for the
`portll` tenant timed out four times today before the leader's own install at 13:02Z.

## Daily entries

Append one line per day from the JSONL file; a day with a refusal or a decode failure resets the
clock and names the cause.

| Day | Date | Refusals | Decode failures | Panics | Note |
|---|---|---|---|---|---|
| 0 | 2026-10-21 | 0 | 0 | 0 | clock start |
| 1 | 2026-10-22 | 0 | 0 | 0 | data directory 18.7 GiB to 2.7 GiB: the checkpoint store compacted the `portll` and `spine` projection side-data files at first open on the new build (log 13:35Z, 14:06Z, 14:46Z); user count unchanged at 46 |
