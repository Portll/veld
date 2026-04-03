# Veld - Agentic Memory Retrieval Progress

**Current Branch**: v0.7.6-unstable | Stabilization and cleanup track toward v0.8
**Last Benchmarked Retrieval Snapshot**: v0.7.2 | MRR 0.925 | Temporal 0.900 | Single-hop 1.000 | Composite 78.4%
**Immediate Target**: clean v0.8 cut from `main`
**Post-v0.8 Target**: v0.9 redb integration and public release hardening

---

> Status note: this document mixes shipped retrieval results from the last benchmarked
> snapshot with current branch cleanup work. Treat `v0.7.6-unstable` as unreleased and
> not suitable for public release until the stabilization pass is complete.

## Shipped (retrieval track, v0.5.0 -> v0.7.2)

### Architecture
- [x] 20-signal retrieval pipeline (up from 3 effective signals)
- [x] Dual-embedder competition (MiniLM 384d + Nomic 768d via LM Studio)
- [x] HTTP embedding API backend (LM Studio / Ollama / vLLM auto-detect)
- [x] CompetitiveEmbedder with `Arc<dyn Embedder>` throughout
- [x] Build number auto-increment (build.rs + SHODH_VERSION_FULL)
- [x] ScoringSignals shared struct for recall/relevance unification
- [x] Signal attribution tracking (BM25/vector/graph/cross-encoder per memory)
- [x] Sleight dimension push API (`POST /api/sleight/dimensions`)

### Retrieval Layers
- [x] Layer 3.5: Working+Session brute-force cosine scan (root cause fix for temporal MRR)
- [x] Layer 3.5: Dual-embedder max-score merge (max of primary + secondary similarity)
- [x] Layer 4.527: BM25 specificity discount (high BM25, zero entity overlap → 5% discount)
- [x] Layer 4.92: Interference detection (pairwise semantic opposition, demote older)
- [x] Layer 5.3: CrossEncoderReranker wired (18% blend, top-20 budget)
- [x] Layer 5.85: Linguistic boost moved before ordinal pins
- [x] Layer 5.87: Calendar-aware temporal range demotion ("last week" sweet spot)
- [x] Layer 5.9: Focal-entity recency scan with tag fallback (Strategy E)
- [x] Removed redundant Layer 5.5 (bi-encoder re-embedding)

### Scoring Signals (Layer 5)
- [x] BRIDGE-1: access_count (7% log-scaled) + graph_strength (8% Hebbian)
- [x] BRIDGE-2: Cross-encoder (18% blend)
- [x] BRIDGE-3: Calibrated confidence (Bayesian gate, 0.85-1.0)
- [x] BRIDGE-4: Edge tier promotions → memory importance boost
- [x] Signal 9: Episode coherence (8%)
- [x] Signal 10: Source type multiplier on credibility
- [x] Signal 11: Emotional valence intensity (2%)
- [x] Signal 12: Sequence proximity (2%)
- [x] Signal 15: graph_contributed feedback attribution
- [x] Signal 16: Context richness (2%)
- [x] Signal 17: Activation level (3%)
- [x] Signal 18: Temporal fact density (2%)
- [x] Signal 19: Entity density (2%)
- [x] Signal 20: External (Sleight) dimension aggregate multiplier

### Tuning
- [x] WH-word gravity: ontological boost 0.08→0.15, penalty -0.08, Layer 2 softened 0.5→0.6
- [x] Source type default: Unknown instead of User
- [x] Cross-encoder blend: 12%→18%
- [x] Contradiction penalty: 0.15→0.20
- [x] Access boost: query-intent-dependent (0.02 exploratory, 0.07 factual)

### Other
- [x] Fourier-learned decay scales (decay_scales.rs, rustfft FFT)
- [x] Partition-theoretic session detection (detect_sessions_adaptive)
- [x] Causal edge extraction (extract_causal_pairs with anti-causal filtering)
- [x] Per-memory access timestamp history (autocorrelation Phase A)
- [x] 21 math formulas cataloged in Veld - Agentic Memory
- [x] Rips filtration + persistent homology in Sleight
- [x] 30+ evaluation artifacts (Bifocal, Overlook, Overloop, Supernova)

---

## Remaining Work (prioritized)

### P0 — Immediate (fixes specific benchmark failures)

| # | What | Expected Impact | Effort | Status |
|---|------|-----------------|--------|--------|
| 1 | **Secondary Vamana index for LongTerm tier** — currently dual-embedder only helps Working/Session (Layer 3.5). Need 768d Vamana index for LongTerm memories too. | +0.01-0.03 MRR | 4h | Ready to build |
| 2 | **P@5 improvement: reduce false-positive top-5 entries** — BM25 pulls in peripherally-related memories. Strengthen entity-overlap requirement for top-5 promotion. | +0.05 P@5 | 2h | Need analysis |

### P1 — This Sprint (architectural improvements)

| # | What | Expected Impact | Effort | Status |
|---|------|-----------------|--------|--------|
| 3 | **Feedback loop unification** — unify recall.rs static weights with relevance.rs learned weights using ScoringSignals struct | Enables adaptive learning | 24h | ScoringSignals struct exists |
| 4 | **Signal attribution wiring** — connect SignalAttribution to /api/reinforce so feedback trains per-signal weights | Enables per-signal learning | 6h | SignalAttribution built |
| 5 | **Nomic backfill** — when dual-embedder is active, backfill existing memories with secondary embeddings during maintenance | Enables dual-index for all memories | 3h | Infrastructure ready |
| 6 | **Benchmark expansion** — add 20+ realistic queries (messy, anaphoric, typo-laden) to calibrate real-world MRR | Diagnostic | 4h | LOCOMO-20 is saturating |

### P2 — Next Sprint (model & query improvements)

| # | What | Expected Impact | Effort | Status |
|---|------|-----------------|--------|--------|
| 7 | **Nomic as primary embedder** — once backfill complete, consider Nomic 768d as primary (replacing MiniLM) for all vector search | +0.02-0.05 MRR | 2h (config change) | Needs backfill first |
| 8 | **Query reformulation** — for abstract queries, rewrite before embedding ("strategic priorities" → "goals OKR focus areas") | +0.015-0.030 MRR | 4h | Concept expansion built (BM25 only) |
| 9 | **Multi-query embedding fusion for Vamana** — currently only in Layer 3.5, extend to main Vamana search path | +0.01-0.03 MRR | 3h | Layer 3.1 fusion built |
| 10 | **Causal direction in ontological intent** — for "why" queries, spread backward along Causes edges | +0.007 MRR | 3h | Causal edges exist |

### P3 — Deferred (diminishing returns on current benchmark)

| # | What | Expected Impact | Effort | Status |
|---|------|-----------------|--------|--------|
| 11 | **Phase-to-session mapping** — "debugging phase" → session ordinals | +0.050 temporal R@5 | 4h | |
| 12 | **Embedding model A/B testing** — compare MiniLM vs Nomic per-query to determine which model helps which query type | Diagnostic | 8h | Dual-embedder ready |
| 13 | **Session-aware tie-breaking** — when scores within epsilon, prefer same-session | +0.005 MRR | 1h | |
| 14 | **Unify coherence into MMR** — 3-way tradeoff: relevance, diversity, coherence in single formula | Architectural | 8h | |
| 15 | **EntityNode.salience enrichment** — add degree fraction for query-conditioned centrality | Low | 1h | |

### Dismissed (adversarially collapsed)

| What | Why Rejected |
|------|-------------|
| Embedding fine-tuning via Hebbian pairs | Circular training signal, catastrophic forgetting, 4 blockers |
| Standalone completeness signal | Cross-encoder IS the completeness signal (MS-MARCO trained) |
| Standalone novelty signal | Anti-correlated with access_count, mathematically can't reverse |
| Standalone centrality signal | PIPE-7 intersection boost already captures query-conditioned centrality |
| Standalone coherence layer | Contradicts MMR diversity; belongs in same formula, not separate layer |
| Standalone causal scoring layer | 65% redundant with ontological penalty; direction is the orthogonal 35% |
| Query decomposition improvements | Already strongest category (0.900 MRR) |
| Recency scale amplification (0.5→1.0) | Regresses other temporal queries |

---

## Benchmark Trajectory

| Version | MRR | Temporal | Single-hop | Multi-hop | Open | R@5 | R@10 | P@5 | Composite |
|---------|-----|----------|------------|-----------|------|-----|------|-----|-----------|
| v0.5.0 | 0.880 | 0.767 | 0.900 | 0.900 | — | — | — | — | 74.4% |
| v0.6.0 | 0.892 | 0.867 | 0.900 | 0.900 | 0.900 | 0.835 | 0.913 | 0.320 | 76.7% |
| v0.6.1 | 0.892 | 0.867 | 1.000 | 0.900 | 0.900 | 0.835 | 0.913 | 0.320 | 76.9% |
| **v0.7.2** | **0.925** | **0.900** | **1.000** | **0.900** | **0.900** | **0.852** | **0.896** | **0.330** | **78.4%** |

## Evaluation Artifacts

All stored in `evaluations/`:

| Category | Count | Key Findings |
|----------|-------|-------------|
| Bifocal+ | 15 | 5 orthogonal signals rejected, cross-encoder IS completeness, query-side is the lever |
| Supernova | 4 | Post-pin re-sort bug, Working-tier gap, 3 unsurfaced clusters, bridge plan |
| Ring 1 (orthogonal features) | 5 | WH-gravity, interference, source fix, temporal history, causal edges |
| Ring 2 (architectural) | 5 | Cross-encoder exists, decomposition done, feedback loop needs unification |
| Orthogonal V2 | 5 | None truly orthogonal — all captured by existing signals |
| Unseen dimensions | 1 | Query-side, ingestion-side, evaluation-side, data-side, temporal structure |

## Key Insight

The retrieval ALGORITHM is no longer the bottleneck. 20 signals scoring the same (query, memory) pairs have diminishing returns. The remaining levers are:
1. **Embedding quality** (Nomic 768d — now active via LM Studio)
2. **Query representation** (reformulation, multi-query fusion — partially built)
3. **Benchmark expansion** (LOCOMO-20 is saturating — need messier, real-world queries)
