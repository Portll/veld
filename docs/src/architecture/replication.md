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
| `GET status` | both | `instance_id`, this process's id, new at every start; `users: [{user_id, log_bytes}]` this instance can feed; `followers` that polled it in the last 10 minutes; `following` when it tails a leader |
| `GET records?user_id&after_lsn&limit` | leader | Frames after `after_lsn`, decrypted, base64, with `head_lsn`, `more` and `log_created`. 404 for a tenant with no log; it never creates one |
| `POST follow {source, admin_key, api_key?}` | follower | Start tailing; refused when the source is this instance (any loopback name with this port, or a source whose status reports this process's `instance_id`) or already follows it. A leader whose status carries no `instance_id` (an older build) is followed |
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

`VELD_CA_CERT` adds a PEM bundle of trust roots to the built-in ones, for a leader whose
certificate a private CA signed. `VELD_REPLICATE_CLIENT_CERT` and
`VELD_REPLICATE_CLIENT_KEY`, set together, present a client certificate to a leader that
requires one. The client is built when a follow starts, so a missing, unreadable or
unparseable file fails the follow with the variable and the path named. See
[Network access](../guides/network-access.md).

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

## A follower is a read-only replica

While an instance follows a leader, only the frames it applies write to it:

- **Client writes are refused** with `503 READ_ONLY_REPLICA` and `Retry-After: 30`.
  `src/handlers/replica.rs` classifies every protected route in `ROUTES`, one layer on the
  protected router applies the table, and a test holds the table to the routes
  `build_protected_routes` mounts. The body names the leader (`leader: {instance_id, url}`)
  to a caller whose API key was verified. A webhook arrives without one and is refused
  without the leader named.
- **Reads of stores replication does not carry** answer `503 NOT_REPLICATED` rather than
  empty or stale data: todos and their comments, projects and project files, reminders,
  context blocks, the knowledge graph and tag provenance, facts, lineage, sessions, A/B
  tests, datasets, consolidation events and Claude session status. Memory reads (recall,
  list, get, search, export) are served.
- **Recall makes no usage writes**, as under `VELD_RECALL_READONLY=1`: no access counts,
  coactivation edges, interference, retrieval counter, reconsolidation boost or feedback.
  `proactive_context` does not auto-ingest, and its `ingested_memory_id` is empty.
- **Writers outside the router are refused too.** `/api/stream` refuses the upgrade, and
  a socket opened before a runtime follow answers its next message with an error frame,
  `{type: "error", code: "READ_ONLY_REPLICA", fatal: true}`, then closes with code 1013
  (try again later). The zenoh remember, forget, stream and mission start and end
  handlers drop the sample and log the refusal under the same code; a zenoh auto-topic
  keeps its session and ingests again after an unfollow. Zenoh recall stores no pending
  feedback. `src/replication.rs` `Replication::client_writes_allowed` is the one check
  they share.
- **Maintenance changes no record.** Decay write-back, tier promotion, compression,
  consolidation, fact extraction, graph strengthening and decay, the sleep-time passes and
  reminder cleanup are skipped; the heavy cycle's flush, cache pruning and vector-index
  compaction (a rebuild of an index past its rebuild or deleted-vector threshold) still
  run. Reminders do not fire and the embedding backlog does not drain. The sleep-time
  workers claim nothing. Work that began before a runtime follow stops writing: a
  maintenance cycle before its next tenant or write phase, a detached `/api/consolidate`
  or `/api/consolidation/sleep` pass before its next phase, and a sleep-time pass before
  it applies its rewrite (its queue item is released for after an unfollow).
- **Writable on a follower, because no leader owns them:** the user-auth surface (this
  instance's logins, sessions and keyring), `follow` and `unfollow` (unfollowing is how
  an operator promotes a follower), backup create and purge, and vector-index rebuild
  and repair (`/api/index/rebuild`, `/api/index/repair`), which write the index and its
  id mappings from the stored records and no record.

`GET /api/replication/role` takes any API key, and neither the admin key nor a tenant. It
reports `{role: leader|follower|standalone, instance_id, leader_instance_id?,
leader_url?, lag_frames?, lag_seconds?, leader_contact_age_seconds?, followers_count?,
max_lag_frames?}`. A leader is an instance that follows nothing and has been polled in
the last 10 minutes; it reports how many followers did and the largest of their summed
`lag_frames`. A follower reports `lag_frames`, the sum of its tenants' `lag`, and two
times, whole seconds:

- `lag_seconds` is how far behind the leader the follower was at its last catch-up: from
  the moment the last pass that left every tenant at its head asked the leader for its
  tenants, to the end of the last pass (a pass ends only once it has read the leader's
  status). It is absent until a pass has caught up since following began. It holds its
  value while the leader is unreachable, so it says how current the follower was when
  the leader went, which is what failover eligibility is decided on.
- `leader_contact_age_seconds` is the time since a request to the leader last
  succeeded, its status or a page of its feed, measured when the response is built. It
  grows while the leader is unreachable, and stays within a poll interval through an
  initial sync, whose pages each reach the leader. It is absent until the first request
  succeeds after following began.

A follower's responses to requests whose API key was verified carry them as
`X-Veld-Replica-Lag-Seconds` and `X-Veld-Replica-Contact-Age-Seconds`, each only when
known. `/health/ready` is unchanged: it is public and publishes no topology.

## The leader's view of its followers

A follower sends `X-Veld-Instance-Id` (its `instance_id`) on every `status` and `records`
request, and `X-Veld-Follower-Url` with its TLS URL when it has a TLS listener. The
leader keeps, in memory, one entry per follower that named itself, at most 64 (a new one
evicts the one polled longest ago), and drops a follower that has not polled for 10
minutes. `status` lists them:

```
followers: [{instance_id, url?, remote_addr, first_seen, last_poll,
             users: {user_id: {requested_lsn, leader_head_lsn, lag_frames}}}]
```

- `url` is what the follower says it is: kept only if it parses as an http(s) URL of at
  most 512 bytes with no credentials, shown and never dialled. `remote_addr` is the
  socket's peer.
- `requested_lsn` is the first LSN the follower asked for. Its `after_lsn` is its cursor,
  the last frame it applied, so it asks for `after_lsn + 1`, or 0 when it has applied
  nothing.
- `lag_frames = leader_head_lsn - requested_lsn` (0 when it asks from the head or past
  it): the frames the leader held at or above `requested_lsn`, which is what that request
  is served with no page limit. It is measured at the request and includes the frames
  the reply carried, and a follower does not ask again for a tenant whose log is
  unchanged, so a caught-up tenant shows its last page's size until its next change.
  The follower's own `lag` (`leader_head_lsn - 1 - applied_lsn`) is the same count.

## Limits, stated

- **One direction.** A follower is read-only (above). Two instances following each other
  would echo frames forever, so `follow` refuses a source that reports following us. A
  three-node cycle is not detected.
- **Memories only.** Todos, reminders, projects, context blocks, the knowledge graph,
  facts, lineage, sessions, A/B tests, datasets and dials are not journaled and do not
  replicate. Consolidation does not run on a follower, so its knowledge graph is not
  rebuilt either: recall's graph legs find there only what the instance held before it
  followed.
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
