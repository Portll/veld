# Alignment

A vector is comparable by cosine only with vectors in its own
[embedding space](embedding-spaces.md). Alignment is the bridge between two spaces: a fitted
linear map that takes a vector in one space to a vector in another, so a query embedded in a
space this server writes can search an index in a space it does not.

The stored vector is always the truth. A mapped vector is a read-side bridge: it is never written
back as a memory's vector, and what it finds is weighed by how well the map fits.

## Maps between spaces

A map is keyed by `(from, to)` space ids, for example
`nomic-embed-text-v1.5@768 → minilm-l6-v2@384`
(`src/embeddings/alignment_spaces.rs`). On disk the pair is an `AlignmentPairId` whose
`secondary` is `from` and `primary` is `to`. A corpus keeps its maps in `alignments/` beside its
store and lists them in the space registry (`.veld_spaces.json`).

At start the retrieval engine installs every map in `alignments/` whose `from` space it writes.
A map whose `to` space no index here holds gives that space an index, built from the vectors
the memories keep in it. That is how a corpus searches the space a migration left behind, or the
vectors another instance wrote.

## Fitting from the corpus

A map needs pairs: the same memory's vector in both spaces. The corpus already holds them for
every memory that has vectors in both, after a migration (the replaced vector is kept), with a
secondary embedder, or after replication from an instance with another embedder. So a map no
longer needs a curated pairs file or a second embedding pass:

```http
POST /api/spaces/align
{"user_id": "claude-code", "from": "nomic-embed-text-v1.5@768", "to": "minilm-l6-v2@384"}
```

The fit reads every memory once, holds out every fifth pair, fits whitened Procrustes (alignment
v2: centre, whiten, rotate, fold back into one affine map), saves the map, installs it, and
answers with what it measured on the held-out pairs:

| Field | Meaning |
|---|---|
| `held_out_cosine` | Mean cosine between a mapped vector and its own pair: the map's confidence |
| `held_out_recall_at_1` | Share of mapped vectors whose nearest held-out target is their own pair |
| `held_out_csls_recall_at_1` | The same with CSLS hubness correction |

At least 64 usable pairs are required. `GET /api/spaces` lists the corpus's spaces and maps.
Both routes are authenticated and tenant-bound.

## Recall

Recall searches every space it has a query vector in. For each installed map from such a space
into an indexed space it does not write, it maps the query, searches that index and fuses the
results by RRF. The weight is the secondary leg's (`secondary_rrf_ratio`) times the map's
confidence. The secondary-to-primary map, when installed, keeps its older role: the aligned
secondary leg searches the primary index through it.

## The command-line tools

`alignment-collect`, `alignment-fit` and `alignment-eval` (`src/bin/`) fit and measure a map
from a curated text corpus instead of the store, for a pair no corpus holds yet:

- `alignment-fit --method whitened` (the default) fits v2; `procrustes` and `ridge` remain.
- The pair id names the two spaces, as the server keys its maps, so a fitted file can be copied
  into a tenant's `alignments/` directory as is.
- `alignment-eval` loads a map of any method.

`VELD_ALIGNMENT_AUTOFIT=1` with a pairs file still fits the secondary-to-primary map from text
at engine start. The store-seeded fit above is the path that needs nothing prepared.

## Not measured yet

The LoCoMo gate runs one embedder in CI, so it cannot measure cross-space recall through a map.
The held-out numbers above are the measurement until a two-space gate configuration exists.

## See also

- [Embedding spaces](embedding-spaces.md) — what a space is and how indexes are kept per space
- [Retrieval pipeline](retrieval.md) — where the bridged results are fused
- `ALIGNMENT_IMPLEMENTATION.md`, `EMBEDDING_ALIGNMENT.md` in the repo root for the history
