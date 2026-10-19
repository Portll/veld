# Live replication

A follower instance tails a leader's per-user intent logs over HTTP and applies every
frame to its own store, then journals the same frame locally so its projections and its own
log follow. The intent log holds a full memory snapshot per `Remember`/`Update` and a
`Forget`/`Anchor` otherwise, so a follower that starts empty converges on the leader's
memories by replaying from LSN 0 with ids and timestamps intact. Embeddings travel inside
the snapshot: the follower never re-embeds.

Code: `src/replication.rs` (feed, follower loop, apply), `src/handlers/replication.rs`
(routes). Shipped in 0.8.0; coalescing and batching in 0.8.2.

## Routes

All under `/api/admin/replication/`, on the protected router (per-user API key) with the
admin key checked in-handler (`X-Admin-API-Key`, the leader's `VELD_ADMIN_API_KEY`). The
feed carries every tenant's memories in the clear, which the per-user key alone must not
reach.

| Route | Role | Purpose |
|---|---|---|
| `GET status` | both | `users: [{user_id, log_bytes}]` this instance can feed; `following` when it tails a leader |
| `GET records?user_id&after_lsn&limit` | leader | Frames after `after_lsn`, decrypted, base64, `head_lsn`, `more` |
| `POST follow {source, admin_key, api_key?}` | follower | Start tailing; refused when the source is this instance or already follows it |
| `POST unfollow` | follower | Stop; the cursor is kept |

At boot: `VELD_REPLICATE_FROM`, `VELD_REPLICATE_ADMIN_KEY`, `VELD_REPLICATE_API_KEY`
(default: this server's own first key), `VELD_REPLICATE_INTERVAL_SECS` (default 2).

## Follower loop

Every interval: `GET status` on the leader; for each user whose `log_bytes` changed since
the last quiet poll, page through `records` from the saved cursor. A page is **coalesced**
before it is applied: for each memory id, only the last `Remember`/`Update` snapshot in the
page and the frames after it (`Forget`, `Anchor`) survive, since earlier snapshots of the
same id are superseded. Logs are Update-heavy because every access and consolidation
re-journals a memory, so this removes more than half the writes. Surviving snapshots are
upserted through `MemorySystem::upsert_memories` with one BM25 commit per page; then each
surviving frame is journaled locally with `journal_and_apply`.

The cursor (`applied_lsn` and `source_bytes` per user) is persisted after every page under
`<data>/replication/<source>.json`, so a restarted follower resumes.

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
  append. The transport is whatever the two base URLs are.

## Measured

2026-10-19, redb replica of the live RocksDB server, same host, 60 tenants, 584 MB of logs:

| Follower | Consumed | Journaled | Note |
|---|---|---|---|
| per-frame apply (0.8.1) | 1.1 frames/s | = consumed | fsync-bound at 2% CPU: store commit, reindex, Tantivy commit, journal sync per frame |
| coalesced pages (0.8.2) | 22 frames/s | 45% of consumed | one BM25 commit per 500-frame page |

The remaining per-memory cost is the store commit and the journal fsync; the durability
class work in the redb-equivalence plan (task 2.2) is the next lever.

A launchd-managed follower needs a file limit: two Tantivy indexes per tenant hold 20 to 30
descriptors each. The server raises its own soft limit to the hard limit at start (0.8.2).
