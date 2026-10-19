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

## The stamp pass

At engine start, `RetrievalEngine::reconcile_corpus_spaces` reads the corpus markers before
anything moves them and walks the store once:

- **Stamp.** A vector with no space gets the space the markers name, when the widths agree, and
  `unrecorded@<width>` when they do not. This runs once per corpus and records
  `.veld_spaces_stamped`.
- **Migrate.** When the active embedder is not the one the markers name, every primary vector in
  another space is re-embedded, or promoted from a secondary vector already in the new space.
  The markers advance only after a pass with no failed write, so an interrupted pass resumes.

## One index per space

The primary index (`vector_index/`) holds the primary embedder's space. Its mappings live in the
store and its chunk vectors are not stored on the record, so its layout is unchanged. Every other
space a query can reach has its own index under `space_indexes/<slug>-<hash>/`:

| File | Holds |
|---|---|
| `vamana.idx` | The Vamana graph and vectors |
| `mapping.bin` | Which memory each vector belongs to, saved with the index |
| `space` | The space id the directory was built for |
| `saved.clean` | Present only while the two files above agree; removed by the first change after a save |

An index loads only when the marker is present, the space matches and every mapped vector is
live. Otherwise it is rebuilt from every memory holding a vector in its space, in one pass over
the store. Saving the mapping replaced the old secondary index's insertion-order mapping, which
drifted whenever a vector was added after a save or soft-deleted.

Today the indexed spaces are the primary embedder's and the secondary embedder's.
`RetrievalEngine::search_space` searches any of them. Recall embeds the query in every space it
can, fuses each space's results by RRF, and compares a stored vector with a query only in the
space both are in. That holds for the brute-force pass over recent memories and for the
cross-embedder agreement signal too.

A memory may hold vectors in more spaces than its two slots. `other_vectors`, the last field of
the record, keeps one vector per further space. When a re-embed replaces a primary vector, the
old one moves there, or into the free secondary slot when that is its space. It stays until its
space is retired. A kept vector in an indexed space is indexed there.

The intent-log Vamana projection (`/api/query`) binds its text-primary index to the tenant's
active space when it opens. It skips any vector of another width or recorded space, so a log
replayed from the start across an embedder change no longer builds the index in the old space.

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
   "codec": "f32", "memories": 9874, "backlog": 12, "stranded": 0},
  {"id": "unrecorded@384", "model": "unrecorded", "dimension": 384,
   "codec": "f32", "memories": 0, "backlog": 0, "stranded": 56}
]
```

`memories` is what an index in the space holds, `backlog` the rows waiting for a vector in it, and
`stranded` the memories stored in it that an index in another space refused. Space ids name
models and widths, never a tenant, so the field is safe on the unauthenticated probe.
