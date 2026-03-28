# Pending Changes (Paused for Other Session Refactor)

## Status
The other session (PID 46664) is doing a major refactor removing ~1600 lines from mod.rs and ~380 lines from constants.rs. Compilation is broken (52 errors). Waiting for it to complete.

## Changes Already Injected (may need re-applying after refactor)

### S1: Semantic Entity Matching (Layer 4.525)
- `graph_memory.rs`: Added `find_entities_similar_to()` method — searches entity embedding cache
- `mod.rs`: Layer 4.525 entity-query overlap boost with semantic fallback
- **Bug**: References `res` variable but should be `fused`

### S4: Score Normalization (Layer 4.94)
- `mod.rs`: Min-max normalize scores to [0,1] after boost cluster
- **Bug**: Same `res` → `fused` variable issue

### BM25 Stemming
- `hybrid_search.rs`: `stem_text()` helper, `stemmed_content_field`, schema migration, stemmed search

## Changes to Re-Add After Refactor

Per user instruction: **don't remove stages, mark as inactive** so model can consider them later.

### Layer 4.52: Decision-Type Boost
- +0.15 for Decision-type memories on decision-language queries
- Should be gated with `const LAYER_4_52_ACTIVE: bool = true;`

### Layer 4.53: Specificity Penalty
- Penalize memories >1.5x mean length (0.70x floor)
- Gate: `const LAYER_4_53_ACTIVE: bool = true;`

### Layer 5.7: Confidence Gating
- Drop trailing results where score < 25% of top score
- Gate: `const LAYER_5_7_ACTIVE: bool = true;`

### Layer 5.8: Answer-Type Soft Filter
- 0.85x penalty for ExperienceType mismatch on bug/risk queries
- Gate: `const LAYER_5_8_ACTIVE: bool = true;`

### Layer 5.9 (pre-truncation): Ordinal Pinning
- "first X" → pin earliest in top-10 at rank 1
- "most recent X" → pin latest in top-10 at rank 1
- Gate: `const ORDINAL_PINNING_ACTIVE: bool = true;`

### Wider Pools
- Vector search: 8x max_results (was 3x)
- Rerank budget: 30 (was 20)

## Benchmark State Before Refactor
- Original: MRR 0.98, R@5 0.98, P@5 0.33
- LOCOMO: MRR 0.73, R@5 0.77, Composite 68.6%
- Both benchmarks deterministic (double overloop verified)
