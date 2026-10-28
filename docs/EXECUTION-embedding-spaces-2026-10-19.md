# Embedding spaces: twelve steps, dependency-ordered

Execution plan of 2026-10-19, written at `c92cd53c` (v0.8.1). It makes "which embedder
produced this vector" a first-class fact, so a corpus can hold vectors from more than one
model and more than one dimension, be searched across them, migrate between them in the
background, replicate between instances with different embedders, and pay for N spaces
less than N times.

## Why now

Every stored vector is anonymous. `Experience` holds `embeddings` and `embeddings_secondary`
(`src/memory/types.rs:893,899`) with no model id or dimension; the process has one primary
index and one secondary index sized to the embedders it booted with
(`src/memory/retrieval.rs:259,577`); the corpus's embedder identity is a single marker file
`.veld_embedder` (`src/embeddings/mod.rs:131`). Where that assumption leaks:

- The index refuses a mismatched dimension at insert (`src/vector_db/hnsw.rs:323`).
  Measured 2026-10-19: 160 `Failed to reindex memory … Failed to add vector to index` on the
  live server since September, and the same on the redb replica while applying older
  snapshots. Those memories are searchable by BM25 and graph only.
- A model change strands every vector: the re-embed pass runs at init and blocks probes
  for hours (write-latch plan, task 4).
- The secondary Vamana is rebuilt from nothing when its dimension changes
  (`retrieval.rs:583`).
- Replication (`src/replication.rs`) carries vectors inside the snapshot; a follower with a
  different embedder cannot index them and today fails silently per memory.
- The alignment engine (`src/embeddings/alignment*.rs`, wired at `src/memory/recall.rs:1649`)
  maps secondary into primary for exactly one pair and cannot self-start (recall-orch task 5).

## Rules that hold throughout

- Never pad, truncate or stack vectors of different spaces into one index. Cosine across
  spaces is noise that the index would report as similarity.
- The stored vector is the truth; an aligned vector is a read-side bridge with a confidence
  and is never written back as the memory's vector.
- A vector that cannot be indexed is recorded (backlog row with a target space), never
  dropped on the floor with a warning.
- Every step lands with a test on the recall gate (`scripts/recall_diff.py`, no-net-loss) and
  `cargo check --all-features`, `clippy`, `test --no-run`, both feature builds
  (`default`, `--no-default-features --features storage-redb`).

## Dependency graph

```
1 name the space
├── 2 one index per space ──┬── 5 background migration ── 11 PQ for non-primary spaces
│                           ├── 6 per-space fusion + calibration
│                           └── 10 query once, search everywhere
├── 3 alignment as a bridge (needs 1; useful from 2)
├── 7 cache / backlog / chunker / registry (needs 1; required before 5)
├── 8 space on every boundary (needs 1; feed + boot guard first)
└── 9 deferred, batched embedding (needs 7's backlog target space)
4 cross-instance replication (needs 2, 3 or 9, and 8's feed)
12 measurement: baseline before 9, then per step, per space
```

Three waves: **representation** (1, 2, 3), **lifecycle** (5, 7, 8, 4), **performance**
(9, 10, 11), with 6 and 12 cutting across. 12's baseline is the first thing to run.

## Steps

### 1. Name the space

**Goal.** Every vector knows its embedder id and dimension.

- `Experience` gains `embedding_space: Option<String>` and
  `embedding_space_secondary: Option<String>`, `#[serde(default)]`. Absent means "the
  corpus marker at write time"; a one-time pass stamps the marker's id onto absent fields
  (`src/memory/storage.rs` legacy decode arms are the model for a migration that never
  rewrites what it cannot read).
- A space id is `<model>@<dimension>`, derived from `PrimaryEmbedderInfo` at probe time
  (`src/embeddings/mod.rs:130`), never typed by hand.
- `reindex_memory` (`retrieval.rs:1223`) checks the vector's space against the index's; a
  mismatch records a backlog row with the target space instead of warning.
- `/health` reports `embedding_spaces: [{id, model, dimension, memories}]` beside
  `tenants_with_data`.

**Done when** the 160 silent failures become backlog rows, counted on `/health`, and a
memory written under one marker and read under another reports both spaces.

### 2. One index per space

**Goal.** The primary/secondary pair becomes a map from space id to index.

- `retrieval.rs`: `vector_index` and `sec_index` → `indexes: HashMap<SpaceId, Index>`,
  each with its own id mapping, persistence file (`vamana_persist.rs`) and projection
  (`vamana_projection.rs`, keyed by space in its checkpoint name).
- `search_ids` / `search_ids_secondary` (`retrieval.rs:1454,1564`) → `search_space(space,
  query_vec, k)`; callers fan out over the spaces the embedder can produce a query for and
  fuse by RRF with the existing BM25 and graph legs.
- A memory may have vectors in several spaces at once; the mapping is per (space, memory,
  chunk), which also fixes the drift bug where chunk and soft-deleted vectors count
  against single mappings (embedder-long-inputs task 6).
- The dual-embedder competition (`competitive.rs`) becomes "two spaces", not a special
  path.

**Done when** a corpus with minilm@384 and nomic@768 vectors is searched in both and the
gate matches or beats the current dual-embedder baseline.

### 3. Alignment as a bridge, not a replacement

**Goal.** A fitted linear map between two spaces lets a query or a foreign vector be
searched in a space it was not embedded in.

- Absorbs recall-orch task 5 (alignment decision). Seed the pairs corpus from the live
  store: memories that already hold vectors in both spaces are the pairs.
- `alignment_config.rs` keyed by `(from_space, to_space)`; maps stored in the space
  registry (step 7); `has_alignment()` becomes `alignment_between(a, b)`.
- An aligned result carries a confidence (fit residual) that fusion discounts (step 6).
- v2 (whitened-Procrustes + CSLS) wired into `alignment_fit.rs`; fit, eval and the
  retrieval hook all take space ids.

**Done when** a query embedded in nomic@768 retrieves from the minilm@384 index through
the map with measured recall on the gate, and the number is in BENCHMARKS.md.

### 4. Cross-instance replication across spaces

**Goal.** A follower whose embedder differs from the leader's still converges.

- The feed (`src/replication.rs::feed`) carries the space ids with each snapshot (they are
  inside the snapshot after step 1; the page header also lists the spaces present).
- `apply_replicated_page`: a snapshot whose space the follower indexes goes straight in;
  otherwise its vector stays stored, a backlog row targets the follower's space (step 9
  embeds it), and the follower searches it through alignment (step 3) until then.
- The redb replica on :3031 is the test bed: run it with a different `VELD_EMBEDDER` than
  :3030 and compare recall on identical data.

**Done when** the replica converges with a different embedder and `/health` on both shows
the spaces and counts.

### 5. Space-aware migration as a background job

**Goal.** Changing embedders is a resumable drain, not a boot pass.

- `SpaceMigration { from, to, cursor, started_at }` persisted in the shared store; one per
  tenant; progress on `/health` and the monitor's Backend view.
- The backlog drain (`embedding_backlog.rs:250`) walks memories whose vectors exist only in
  `from`, embeds into `to` (batched, step 9), writes the vector and its space, advances
  the cursor. Killable at any point, resumes at the cursor (write-latch plan task 4).
- Chunk vectors migrate with their parent.
- Retiring a space is a separate, explicit command (`veld spaces retire <id>`) refused
  while any memory has only that space.

**Done when** a model change on a 10k-memory corpus keeps `/health/ready` at 200
throughout and finishes in the background with the cursor visible.

### 6. Per-space fusion and calibration

**Goal.** Fusion does not favour the space with the sharper cosine distribution.

- Per-space score normalisation fitted from the gate (stored in the registry), per-space
  weight on the dials (`src/memory/dials.rs`), alignment confidence as a discount.
- `SignalAttribution` (`types.rs`) records the space that produced each candidate.
- The gate runs per space and per combination; "dual embedder wins" is a measured row.

**Done when** the attribution shows which space won each hit and the per-space weights
are dials, not constants.

### 7. Cache, backlog, chunker and registry keyed by space

**Goal.** Nothing that touches a vector is ambiguous about its space.

- `content_cache` (`src/memory/mod.rs:331`) keyed by `(space, sha256)`; today a model change
  serves stale vectors.
- Backlog rows carry a target space (`embedding_backlog.rs:156`).
- Chunk size and overlap come from the space's model (`chunking.rs`), not one constant.
- A space registry file beside `.veld_embedder`: every space the corpus holds, model,
  dimension, codec, count, and the alignment maps between pairs. Written by the server,
  read by the monitor and `veld doctor`.

**Done when** two spaces coexist in a corpus with no cross-contamination in the cache and
the registry lists both.

### 8. Space on every boundary a vector crosses

**Goal.** No surface drops the space silently.

- Intent-log snapshot (free after 1), replication feed (step 4), MIF export/import
  (`src/mif/`), portable dump (`src/storage/portable.rs`), backups, Zenoh messages
  (`src/zenoh_transport/`), Python and MCP surfaces: each carries the space id or
  declares the vector dropped.
- MIF import from another vendor marks vectors as a foreign space and queues re-embed.
- Boot guard (`src/storage/boot_guard.rs`): a corpus whose vectors are all in a space this
  build can neither produce nor align refuses to report healthy, like the empty-redb rule.

**Done when** a round trip through each surface preserves the space id, with a test per
surface, and the boot guard test covers the orphaned-space case.

### 9. Deferred, batched embedding

**Goal.** The request path never waits on an embedder.

- A write stores the memory, indexes BM25, and records a backlog row per target space;
  the drain batches 32 to 128 texts per call to the HTTP embedder
  (`http_embedder.rs`) or ONNX session, chunks included in the same batch.
- One process-wide embedder and cross-encoder, global ORT thread pool (perf-phase2
  task 5), shared by every space's drain.
- Recall of a not-yet-embedded memory works through BM25 and graph, as it already does
  during an outage.

**Done when** write p95 no longer depends on embedder latency and drain throughput per
batch size is in BENCHMARKS.md.

### 10. Query once, search everywhere

**Goal.** Read cost grows sub-linearly with spaces.

- Embed the query once per producible space, in parallel; search each space's index
  concurrently with BM25 and graph; spaces without a query vector go through alignment or
  are skipped, never re-embedded on the read path.
- Per-space candidate budget on the dials, adapted from attribution (step 6); the
  cross-encoder remains one top-K pass after fusion.

**Done when** recall p95 with two spaces is within 1.3× of one space at 100k memories.

### 11. Cheap indexes for non-primary spaces

**Goal.** Memory and disk do not double per space.

- PQ (`src/vector_db/pq.rs`) as the default codec for every space but the primary, with
  exact-vector reranking of the top candidates; codec recorded per space in the registry.
- Vectors stored once in the record; index files per space so migration and retirement
  touch one file.

**Done when** RSS at 100k memories with two spaces is under 1.3× one space, measured.

### 12. Measure per space, as a gate

**Goal.** Quality and cost are reported together before anything is adopted.

- Extend the storage benches (redb-equivalence task 2) with a space dimension: write
  throughput deferred vs inline, recall p50/p95 vs number of spaces at 10k and 100k, RSS
  per space with and without PQ, drain throughput per batch size per embedder host.
- LoCoMo gate under each space configuration.
- Baseline **before step 9 lands**; every later step re-runs it; results in BENCHMARKS.md
  beside the engine comparison; dials expose budget and codec.

**Done when** each step above has a before/after row and a configuration that improves
recall at a cost the dials cannot pay is recorded, not adopted.

## Milestones in spine

- `veld-recall-knowledge-orch-20261018`: steps 1, 2, 3 (absorbs task 5), 5, 6, 7, 8.
- `veld-perf-phase2-20261019`: steps 9 (with task 5), 10, 11.
- `veld-redb-equivalence-perf-20261018`: step 4 (the :3031 replica is the test bed) and
  the bench extension of step 12.

## Not decided here

- Which model becomes the second production space. nomic@768 is live; minilm@384 exists
  in older records; BGE-M3 is the open proposal (embedder-long-inputs plan).
- Whether a space can be tenant-scoped rather than corpus-wide. The registry is designed
  per corpus; per-tenant is a later refinement if a fleet needs it.
