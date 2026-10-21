# Live replication

A follower instance tails a leader's per-user intent logs over HTTP and applies every
frame to its own store, then journals the same frame locally so its projections and its own
log follow. The intent log holds a full memory snapshot per `Remember`/`Update` and a
`Forget`/`SetAnchored` otherwise (`Anchor` in older logs), so a follower that starts empty converges on the leader's
memories by replaying from LSN 0 with ids and timestamps intact. Embeddings travel inside
the snapshot: the follower never re-embeds.

Code: `src/replication.rs` (feed, follower loop, apply), `src/handlers/replication.rs`
(routes). Shipped in 0.8.0; coalescing and batching in 0.8.2; per-tenant isolation,
divergence detection and transport rules in 0.8.3.

## Routes

All under `/api/admin/replication/`, on the protected router (per-user API key) with the
admin key checked in-handler (`X-Admin-API-Key`, the leader's `VELD_ADMIN_API_KEY`). The
feed carries every tenant's memories in the clear, which the per-user key alone must not
reach.

| Route | Role | Purpose |
|---|---|---|
| `GET status` | both | `users: [{user_id, log_bytes}]` this instance can feed; `following` when it tails a leader |
| `GET records?user_id&after_lsn&limit` | leader | Frames after `after_lsn`, decrypted, base64, with `head_lsn`, `more` and `log_created`. 404 for a tenant with no log; it never creates one |
| `POST follow {source, admin_key, api_key?}` | follower | Start tailing; refused when the source is this instance (any loopback name with this port) or already follows it |
| `POST unfollow` | follower | Stop; a poll in progress stops at its next page; the cursor is kept |

At boot: `VELD_REPLICATE_FROM`, `VELD_REPLICATE_ADMIN_KEY`, `VELD_REPLICATE_API_KEY`,
`VELD_REPLICATE_INTERVAL_SECS` (default 2).

## Transport and keys

The admin key and API key ride every request. A leader on another host must be `https`;
plain `http` is accepted only for loopback, or with `VELD_REPLICATE_INSECURE_HTTP=1` for a
private link. URLs carrying credentials are refused. When no API key is given, the
follower offers its own key only to a leader on this host; a remote leader must be given
its key. A leader's error body is echoed back in at most 300 characters, and a reply over
256 MB is refused.

## Follower loop

Every interval: `GET status` on the leader; then each tenant is synced on its own:

- **Skipped** when its log length is unchanged since its last clean sync, while it waits out
  its backoff after a failure, or while it is diverged.
- **Paged** from its cursor through `records`. A page is **coalesced** before it is applied:
  for each memory id, only the last `Remember`/`Update` snapshot in the page and the frames
  after it (`Forget`, `SetAnchored`, `Anchor`) survive. Logs are Update-heavy because every access and
  consolidation re-journals a memory, so this removes more than half the writes. Surviving
  snapshots are upserted through `MemorySystem::upsert_memories` with one BM25 commit per
  page; then each surviving frame is journaled locally with `journal_and_apply`. A frame
  naming another tenant, or a snapshot whose memory id differs from its frame's, fails the
  page.
- **Cursor saved after every page** under `<data>/replication/<source>.json`, so a
  restarted follower resumes at the last applied page. An unparseable cursor is renamed
  to `.json.unreadable`, logged, and the tenants replay from the start.
- **Failures are per tenant.** A tenant whose page fails records the full error chain,
  retries on its own backoff (the interval doubled per failure, capped at five minutes),
  and does not hold up the tenants after it. Before 0.8.3 one failing tenant stopped every
  tenant sorted after it; on 2026-10-18 that held portll still for two hours.
- **Divergence.** Each page reports the leader's next LSN and its log's identity (the
  creation time in its header). If the next LSN is not above what the follower applied,
  or the identity changed, the leader's log was restored, rolled back or recreated, and
  the follower's copy no longer matches it. The tenant is marked `diverged`, logged once
  at error level, and not synced again. To resync it, stop the follower, remove the
  tenant's directory on the follower, and start it: a diverged tenant with no local data
  replays from the start.
- **Leader unreachable.** The loop backs off the same way and logs a warning only when the
  error changes.
- **Shutdown.** At the shutdown signal the follower starts no request and no page; a page
  already being applied finishes and saves its cursor. The follower runs on the server's
  task tracker, so the shutdown drain (up to 10 s) waits for that page before the store
  flush and the vector-index save. Before this, it kept applying through both: on
  2026-10-19 the replica applied frames for 51 s after "Server shutdown complete".

A `follow` or `unfollow` bumps a generation number; a poll started under an older
generation stops at its next page and writes nothing, so switching targets cannot mix
two leaders' cursors.

Status reports, per tenant: applied LSN, the leader's next LSN, `lag` (leader frames not
yet applied: `leader_head_lsn - 1 - applied_lsn`, or the whole log when nothing is
applied), frames consumed and journaled, failures, last error and divergence; and overall:
pages applied, polls (passes over every tenant) finished, when the current pass started
(`pass_started`), the tenant it is syncing (`syncing`) and the last leader error. An
initial sync is a single pass that can last hours, so `polls` stays at 0 through it;
`pages`, `syncing` and `last_applied` show it moving.

## Limits, stated

- **One direction.** A follower is read-mostly; writes made on it are not sent back. Two
  instances following each other would echo frames forever, so `follow` refuses a source
  that reports following us. A three-node cycle is not detected.
- **Memories only.** Todos, reminders, projects, context blocks and dials are not journaled
  and do not replicate. The knowledge graph is rebuilt by the follower's own consolidation.
- **Same embedding space.** A snapshot whose vector dimension differs from the follower's
  index is stored but not vector-indexed (BM25 and graph still find it). The embedding
  spaces plan (`docs/EXECUTION-embedding-spaces-2026-10-19.md`, step 4) lifts this.
- **Frames leave the leader decrypted** and are sealed again by the follower's keystore on
  append.
- **Upgrade followers before the leader** when the log gains a frame type. Anchoring now
  journals `SetAnchored`, which carries the flag and the importance after the change; a
  follower built before it cannot decode that frame, so the tenant's page fails and retries
  on its backoff until the follower is upgraded. Older logs carry `Anchor`, with the
  importance only: for those the follower reads a rise as an anchor and a fall as its
  release, and a release that leaves importance unchanged is not seen.

## Measured

2026-10-19, redb replica of the live RocksDB server, same host, 60 tenants, 584 MB of logs:

| Follower | Consumed | Journaled | Note |
|---|---|---|---|
| per-frame apply (0.8.1) | 1.1 frames/s | = consumed | fsync-bound at 2% CPU: store commit, reindex, Tantivy commit, journal sync per frame |
| coalesced pages (0.8.2) | 22 frames/s | 45% of consumed | one BM25 commit per 500-frame page |

The remaining per-memory cost is the store commit and the journal fsync; the durability
class work in the redb-equivalence plan (task 2.2) is the next lever.

A launchd-managed follower needs a file limit: two Tantivy indexes per tenant hold 20 to 30
descriptors each. The server raises its own soft limit to the hard limit at start and logs
the result once logging is up.
