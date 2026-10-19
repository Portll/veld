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
