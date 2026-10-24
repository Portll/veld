# Embedding spaces

A vector is comparable to another by cosine only when the same model produced both at the same
width. Veld calls that pair a **space** and writes it beside every stored vector, so a corpus
can say which embedder made each vector instead of assuming the one it booted with.

## The space id

A space id is `<model>@<dimension>`, for example `nomic-embed-text-v1.5@768` or
`minilm-l6-v2@384`. It is derived, never typed:

- from a live embedder: its model name and the width of the vector it returned
  (`SpaceId::for_vector` in `src/embeddings/space.rs`);
- from the corpus markers `.veld_embedder` and `.veld_embedding_dim`, for vectors written before
  spaces were recorded (`corpus_space`).

The model name is lowercased and loses LM Studio's `text-embedding-` catalogue prefix, so the
same weights served by LM Studio, llama-server or the ONNX loader are one space. A Matryoshka
width is its own space: `nomic-embed-text-v1.5@256` and `@768` do not mix.

A vector whose width the markers do not explain gets `unrecorded@<dimension>`: the width is
known, the model is not, and Veld does not guess.

## Where the space is stored

`Experience` carries `embedding_space` and `embedding_space_secondary`. On disk they are the last
two fields of the memory record, not part of `Experience`'s own encoding, so the positional
layout older records use is unchanged. The record encoding is positional bincode, so a field
appended to `MemoryFlat` must encode its default as one zero byte. A record written before a
field existed is then read by padding it with those zero bytes until it decodes exactly
(`decode_with_trailing_defaults` in `src/memory/storage.rs`). Both storage engines read records
through that path. Older binaries ignore the trailing bytes, so a mixed-version leader and
follower stay compatible.

The intent-log snapshot is the memory record, so the space travels in every journaled write and
in every replication frame without further work.

## Migrations run in the background

At engine start, `RetrievalEngine::reconcile_corpus_spaces` reads the corpus markers and decides;
it rewrites nothing except in one case. The work is a `SpaceMigration` record in the tenant's
own store, which the embedding backlog's tick drains (`src/memory/space_migration.rs`):

- **Stamp.** On a corpus never stamped, every vector with no space gets the space the markers
  name when the widths agree, else `unrecorded@<width>`. Until the drain reaches a vector, it
  reads as the index's at the index's width, as every vector did before spaces were recorded.
  The finished stamp writes `.veld_spaces_stamped`.
- **Re-embed.** When the active embedder writes another space than the markers name, every
  primary vector not in the new space is re-embedded, batched, or promoted from a secondary
  already in it. The vector it replaces is kept. The markers advance at start: the record, not
  the markers, says what is pending, so a restart resumes the migration rather than starting
  another. A change of model at the same width over a corpus never stamped is stamped at start
  first, since at one width a vector with no space would otherwise read as the new model's.

Each tick runs batches of `VELD_SPACE_MIGRATION_BATCH` memories (default 128) for up to
`VELD_SPACE_MIGRATION_TICK_SECS` (default 20). The cursor, the last memory id finished, is saved
after every batch, so a kill or a full disk costs at most one batch, which is idempotent to redo.
An embedder that does not answer stops the batch without moving the cursor. While a re-embed is
under way, the start-up repair and the backlog reconcile leave its memories alone.

Once 256 memories hold vectors in both spaces, the drain fits the map from the new space back
into the old one. The old space then gets an index from the kept vectors, and the memories not
yet migrated stay searchable through the map. Until a memory is migrated, a cosine taken
directly against its primary slot compares the old model's vector with the new one's. The
window lasts as long as the migration.

`/health` and `GET /api/spaces` report the migration's progress. A space no embedder here writes
is retired with `veld spaces retire <space>`, or `POST /api/spaces/retire`, which drops its kept
vectors, index and maps. The command is refused for a space this server writes, for the space a
running migration is leaving, and while any memory's only vector is in it.

## One index per space

The primary index (`vector_index/`) holds the primary embedder's space. Its mappings live in the
store and its chunk vectors are not stored on the record, so its layout is unchanged. Every other
space a query can reach has its own index under `space_indexes/<slug>-<hash>/`:

| File | Holds |
|---|---|
| `pq.idx` | The product-quantized codes and codebooks (codec `pq`, the default) |
| `vamana.idx` | The Vamana graph and full vectors (codec `f32`) |
| `mapping.bin` | Which memory each vector belongs to, saved with the index |
| `space` | The space id the directory was built for |
| `saved.clean` | Present only while the index file and the mapping agree; removed by the first change after a save |

An index loads only when the marker is present, the space matches and every mapped vector is
live. Otherwise it is rebuilt from every memory holding a vector in its space, in one pass over
the store. Saving the mapping replaced the old secondary index's insertion-order mapping, which
drifted whenever a vector was added after a save or soft-deleted.

### Codecs

A space other than the primary would otherwise hold its vectors twice: in full in every record,
and in full again in its index. Its index's codec, `VELD_SPACE_CODEC`, says how it holds them:

- **`pq`** (default): one byte per 8 dimensions, 48 bytes for a 384-dimensional vector instead
  of 1536 (`src/vector_db/quantized.rs`). A search scores every live code against the query and
  reranks the best two per result asked for by the cosine to the exact vector, read from the
  memory's record. Every score a search returns is exact. Until a space holds 4096 vectors they
  are kept exact and searched exactly, with no reads. At 4096 the quantizer is trained on them,
  its subspaces in parallel, and from then the index holds codes. The heavy maintenance pass
  retrains it from the records once the index has grown eightfold since training. A dimension
  that 8, 4 or 2 does not divide stays exact.
- **`f32`**: full vectors in a Vamana graph, as before step 11.

The codec of every index is in the registry and `GET /api/spaces` (`codec`), and on
`GET /api/health/index` under `spaces` with the bytes each index holds its vectors in. An index
saved under the other codec is rebuilt from the records at the next start. Retiring or migrating
a space touches only its own directory. The primary index keeps full vectors in its graph: it is
the leg the rest of recall is ranked against.

Today the indexed spaces are the primary embedder's and the secondary embedder's.
`RetrievalEngine::search_space` searches any of them. Recall searches every space a query vector
reaches (see below), fuses each space's results by RRF, and compares a stored vector with a
query only in the space both are in. That holds for the brute-force pass over recent memories and for the
cross-embedder agreement signal too.

A memory may hold vectors in more spaces than its two slots. `other_vectors`, the last field of
the record, keeps one vector per further space. When a re-embed replaces a primary vector, the
old one moves there, or into the free secondary slot when that is its space. It stays until its
space is retired. A kept vector in an indexed space is indexed there.

The intent-log Vamana projection (`/api/query`) binds its text-primary index to the tenant's
active space when it opens. It skips any vector of another width or recorded space, so a log
replayed from the start across an embedder change no longer builds the index in the old space.

## Writes do not wait on the embedder

`POST /api/remember`, `/api/remember/batch` and `/api/upsert` store the memory, index it for
keyword search and return. The memory joins its tenant's embed queue
(`src/memory/embed_queue.rs`). The request then starts a background drain, and the server's own
drainer ticks every 250 ms (`VELD_EMBED_QUEUE_INTERVAL_MS`). A drain takes up to 64 memories
(`VELD_EMBED_BATCH`, at most 256), finds their vectors in the content cache or in one
`encode_batch` call, and finishes each memory as before: vector index, interference detection,
write gate. A chunked memory's chunk vectors come from one call too. Concurrent writes coalesce:
a drain that finds the queue already taken returns at once.

An upsert that changes content drops the old vectors and leaves the vector indexes until its new
vectors arrive. Until then a queued memory is found by keyword and graph search, as during an
embedder outage. When the batch call fails, every memory it was for becomes a backlog row,
rather than being retried one at a time against a host that is down. The queue is in memory: a
memory still queued at a crash is stored but unindexed, and the backlog reconcile finds it.

`MemorySystem::remember` stays synchronous for library callers and the recall harness, which
recall what they just wrote. Bulk imports (seed, ingest, MIF) still embed as they write.

## The query once, every space at once

Recall embeds the query once in each space an embedder here writes. When neither vector is in
the query cache, the primary and secondary embeddings run at the same time. The secondary
embeds the text as a query, the side an asymmetric model such as nomic searches with. Both are
cached per space, so a repeated query embeds nothing.

Every index those vectors reach beyond the primary one is a leg (`src/memory/space_legs.rs`):

- another space's index, searched with the query vector in that space;
- the primary index through the installed secondary-to-primary map, in place of the direct
  secondary search, which runs only if the mapped one fails;
- an indexed space no embedder here writes, through a fitted map from a space with a query
  vector.

A space no query vector reaches is not searched. The read path never embeds into it. The legs
need only the query vectors, so they start on their own thread as soon as the vectors exist and
are collected at fusion. Graph traversal, the primary search and BM25 run meanwhile. The
cross-encoder stays one top-K pass after fusion.

Each leg asks its index for a candidate budget, in multiples of the results asked for. The
`space_candidates` dial sets it per space, from 1 to 32. A space without an entry learns its
budget from its own record: how deep into the leg's list the results recall returned lay. That
is a moving mean and variance, kept in memory per tenant. After 30 recalls the budget is the
mean plus three deviations, between 2 and the default of 8. A leg whose returned results all
come from its top asks for less. A learned budget never asks for more than the default.

## Fusion across spaces

Fusion is by rank (RRF), so two spaces' different cosine scales do not bias it. Two things are
per space:

- **Weights.** The `space_weights` dial (`/api/dials`) multiplies a non-primary space's RRF
  contribution, keyed by space id; a space with no entry weighs 1. A leg reached through a map
  is also weighed by the map's confidence. The primary space's vector weight stays in
  `rrf_weights`.
- **Calibration.** Where raw cosines from two spaces are compared, the max over recent memories'
  vectors in Layer 3.5, each space's similarity is first moved onto the primary space's scale by
  matching z-scores. A sharper model's near-misses would otherwise outrank the primary's hits on
  scale alone. Each space learns its scale from its own searches: a moving mean and variance of
  the best similarity each search returns, used after 30 searches and kept in the space registry
  across restarts.

Every hit's signal attribution lists the spaces whose vector search returned it, the primary
index's first, so which space won a hit is visible per result.

## Every boundary a vector crosses

A slot holds a vector in the space it is read in, or nothing. A write that brings a vector in
another space keeps it among `other_vectors` and leaves the slot empty, so the memory is embedded
in the corpus's space and every cosine against a slot compares one space. The start-up pass does
the same for a vector it cannot re-embed.

| Surface | What carries the space |
|---|---|
| Intent log and replication frames | The memory record itself, whose last fields are the spaces; each replication page also names how the leader reads an unrecorded slot ([replication](replication.md#across-embedding-spaces)) |
| Portable dump, redb backup, `veld migrate` | The record bytes, carried as stored |
| RocksDB backup | The store's files, copied as they are |
| MIF export | `embeddings.model` and `embeddings.dimensions` name the vector's space, not a fixed `minilm-l6-v2` |
| MIF import | Reads the space back; a vector in another space is kept, and the memory is embedded anew (or queued when the embedder is down) |
| Zenoh, Python, MCP | Carry no vectors: memories cross as text and are embedded on arrival |

`/health` reports `orphaned_corpora` and turns `degraded` when a cached corpus stores memories,
no index here holds any of them, and the backlog or a kept vector shows vectors waiting. That is
a corpus whose every vector is in a space this server neither writes nor reaches through a map,
or one whose embedder has been down since its first write. Serving it as healthy would hide that
recall is keyword-only. Kept vectors are counted by a census over the store, at most hourly.

## Refusals become backlog rows

An index holds one space. A stored vector in another space is refused, never inserted, because
cosine across spaces is noise the index would report as similarity. The refusal is a row in the
embedding backlog that names both spaces:

| Field | Meaning |
|---|---|
| `target_space` | The space the waiting vector must be in: the index's |
| `stored_space` | The space of the vector the memory already stores |

A row for the primary index lives at `embed_backlog:<memory id>`. A row for the secondary index
lives at `embed_backlog_space:<memory id>/<space>`; the separate prefix keeps a binary that
predates it from reading a key it cannot parse. The periodic drain embeds the memory into the
target space, stores the new vector with its space and indexes it. A displaced primary vector in
the secondary index's space moves to the secondary slot.

A vector stored with no space at the index's width is indexed as before: that is what every
vector was taken to be before spaces were recorded.

## Caches, chunking and the registry

- **Caches.** The content and query embedding caches are keyed by `(space, SHA-256 of the
  text)` (`src/memory/embedding_cache.rs`), so a vector cached for one model is never served for
  another.
- **Chunking.** The chunk size, overlap and minimum come from the space's model
  (`ChunkConfig::for_space`). A model gets a row only from a `chunk-audit --sweep` over its own
  tokenizer, because chunking decides how many vectors a memory gets and so how it ranks. Only
  MiniLM has a measured row. Every other model gets MiniLM's chunks, which fit any supported
  window, at the cost of more vectors than a wider window needs.
- **Registry.** `.veld_spaces.json` beside the corpus markers lists every space the corpus holds:
  model, width, codec, what each index holds, the backlog waiting for each, and what is stranded
  in a refused space, plus the alignment maps between pairs. The server rewrites it after a
  backlog drain when its content changed. `veld doctor` reads it without opening the store and
  names each tenant with vectors waiting or stranded.

## `/health`

`GET /health` reports `embedding_spaces`, summed over the users in cache:

```json
"embedding_spaces": [
  {"id": "nomic-embed-text-v1.5@768", "model": "nomic-embed-text-v1.5", "dimension": 768,
   "codec": "f32", "memories": 9874, "backlog": 12, "stranded": 0, "kept": 0},
  {"id": "unrecorded@384", "model": "unrecorded", "dimension": 384,
   "codec": "f32", "memories": 0, "backlog": 0, "stranded": 56, "kept": 120}
]
```

`memories` is what an index in the space holds, `backlog` the rows waiting for a vector in it,
`stranded` the memories stored in it that an index in another space refused, and `kept` the
memories keeping a vector in it that no index here holds. Space ids name
models and widths, never a tenant, so the field is safe on the unauthenticated probe.
