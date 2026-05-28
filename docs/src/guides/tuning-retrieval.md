# Tuning retrieval

Veld's [retrieval pipeline](../architecture/retrieval.md) has many knobs. Most
default values are sensible, but specific workloads benefit from tuning.

## Most-tuned parameters

| Parameter | Default | When to change |
|---|---|---|
| Cross-encoder budget (top-K reranked) | 20 | Increase for accuracy-critical workloads; decrease for latency |
| Cross-encoder blend weight | 18% | Increase if cross-encoder consistently helps; decrease if it adds noise |
| Embedder secondary | Nomic 768d | Disable if memory-constrained — primary embedder alone works |
| Calibrated confidence gate | 0.85 | Lower if too few memories surface; raise to suppress low-confidence noise |
| Decay factor | from `decay_scales.rs` | Per-memory-type; Fourier-learned during consolidation |
| HNSW M | default | Higher M = denser graph, better recall, more memory |
| HNSW ef_search | default | Higher = better recall, slower retrieval |

## Diagnosing poor retrieval

If recall results feel wrong, the diagnostic path:

1. `/api/recall/tracked` returns `SignalAttribution` per result — which of
   the 20 signals contributed.
2. If `vector_contribution` is high but results are off-topic, the
   embedder may be the wrong choice for your domain. Consider switching
   primary embedder or adding a domain-specific secondary.
3. If `graph_contribution` is dominant and pulling unrelated memories,
   Hebbian edges may be over-strengthened. Check the consolidation report
   for excessive edge growth.
4. If `cross_encoder` consistently disagrees with the cheap layers,
   the cheap layers may be filtering out the right answers before
   reranking can save them. Increase the rerank budget.
5. If `recency` dominates and surfacing stale results, decay scales may be
   mis-tuned.

## Implicit feedback

The system learns from implicit signals — which memories an agent clicks
on, which it ignores. This drives `feedback_momentum` (signal 11). Explicit
feedback via `POST /api/reinforce`:

```json
{
  "user_id": "...",
  "memory_id": "...",
  "signal": "helpful" | "not_helpful"
}
```

A few well-placed reinforce calls per session shape the next session's
retrieval significantly.

## External Sleight dimensions

If you run [Sleight](https://github.com/Portll/sleight) alongside veld,
push topological-health scores via `POST /api/sleight/dimensions`. The
five scores (density, coherence, closure, confidence, isotropy) modulate
retrieval rank when fresh. Sleight effectively becomes the 20th retrieval
signal.

## Anchor for invariants

Use `POST /api/anchor` for memories that should never be forgotten:

- User preferences
- Project-level invariants
- Long-running decisions

Anchored memories are exempt from decay and lint-suggested archiving.

## See also

- [Retrieval pipeline](../architecture/retrieval.md)
- [Memory tiers](../architecture/memory-tiers.md)
- [Consolidation](../architecture/consolidation.md)
