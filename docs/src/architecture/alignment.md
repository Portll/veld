# Alignment

Veld runs dual-embedder competition — MiniLM (384d) and Nomic (768d). To
max-merge similarity scores across both spaces, the spaces must be made
comparable. That's what alignment does.

## The problem

A vector in MiniLM's 384d space and a vector in Nomic's 768d space are not
in the same coordinate system. Their cosine similarities to a given query
are not directly comparable: a 0.8 in MiniLM-space might mean something
different than a 0.8 in Nomic-space.

Without alignment, `max(sim_minilm, sim_nomic)` would consistently favour
whichever embedder happens to produce higher absolute scores — not whichever
is *more relevant*. Alignment fixes this by mapping one space into the
other so absolute scores become comparable.

## The approach

Two fitters live in [`src/bin/`](https://github.com/Portll/veld/tree/main/src/bin):

- **`alignment-fit` Procrustes:** orthogonal linear transformation rotating
  the source space to best-fit the target space in least-squares. Fast,
  preserves angles and distances.
- **`alignment-fit` Ridge:** L2-regularized linear regression. Used when
  Procrustes is insufficient — non-square dimensions (384 → 768), or when a
  bias term is needed.

The fitter learns a linear map M such that for paired (source, target) embeddings,
`source @ M ≈ target` minimises the residual.

## Pipeline

```
alignment-collect → pairs.vec → alignment-fit → fitted.npz → alignment-eval
                                                                  ↓
                                                          metrics + retrieval integration
```

1. **`alignment-collect`** — gather paired embeddings from a corpus (the
   same text embedded by both embedders).
2. **`alignment-fit`** — learn Procrustes or Ridge map. Multi-lambda sweep
   for Ridge (`feat(alignment): corpus expansion tooling + multi-lambda
   Ridge sweep` was a recent commit).
3. **`alignment-eval`** — measure retrieval quality with and without
   alignment.

Recent work (commit `22bac68`): paths in TOML specs are now CWD-relative;
pairs.vec is cached; the corpus has been expanded with MDN, cpython, Rust,
React, TypeScript documentation.

## Evaluation artifacts

Alignment evaluation outputs live in [`evaluations/alignment/`](https://github.com/Portll/veld/tree/main/evaluations/alignment).
The most recent eval pairs file is
`pairs.nomic-embed-text-v1.5__minilm-l6-v2.doc.vec`.

## Why this matters for retrieval

With alignment landed, the dual-embedder max-merge in Layer 3.5 of the
[retrieval pipeline](retrieval.md) is meaningful. Without it, dual-embedder
would have been theatre — one embedder would have dominated regardless of
relevance.

## See also

- [Retrieval pipeline](retrieval.md) — where the aligned scores are consumed
- `ALIGNMENT_IMPLEMENTATION.md`, `EMBEDDING_ALIGNMENT.md` in the repo root
  for deeper history
