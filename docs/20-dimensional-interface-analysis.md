# 20-Dimensional Interface Analysis: Claude x Veld Parameter Space

## The Interface Manifold

The coupling surface between Claude's behavioral decisions and Veld - Agentic Memory's 200+ internal constants spans exactly 20 measurable dimensions. Each dimension represents a point where Claude's behavior threads through the hook/MCP layer and modifies Veld's internal state.

```
Claude Behavior ──> Hook Parameters ──> Veld Constants ──> Retrieved Context ──> Claude Behavior
                    |                    |                                         |
                    |                    |                                         |
                    +-- editable code ---+                                         |
                                                                                  |
                    +-------------------------------------------------------------+
                    v
              (feedback loop — adaptive_weight_learning)
```

## Dimension Map

| # | Interface Parameter | Claude Side | Veld Side | Status |
|---|-------------------|-------------|------------|--------|
| 1 | `semantic_threshold` (0.6-0.65) | Query precision | Vector similarity cutoff | ACTIVE — hooks send 0.6, MCP sends 0.65 |
| 2 | `entity_match_weight` (0.3-0.4) | Entity extraction quality | Graph traversal influence | ACTIVE — but Claude can't see entity lookup failures |
| 3 | `recency_weight` (0.2) | How much Claude weights "recent" | Temporal scoring curve | ACTIVE — also 5x amplified by "recent/latest" keywords |
| 4 | `max_results` (2-5) | Context window budget | Candidate pool cutoff | ACTIVE — varies by hook event type |
| 5 | `auto_ingest` (bool) | Whether prompts become memories | Encoding trigger | ACTIVE — but segmented memories lose RichContext |
| 6 | `mode` (semantic/associative/hybrid) | Retrieval strategy choice | Which pipeline runs | ACTIVE — but semantic mode still runs BM25+vector, just skips graph |
| 7 | `content` truncation (1000 chars) | What of prompt reaches Veld | Embedding input | ACTIVE — but 95% of Claude's reasoning lost for FileAccess events |
| 8 | `type` on remember | Classification | Importance weighting (0.10-0.30) | ACTIVE — but hook auto-classifies, Claude doesn't choose |
| 9 | `credibility` (0-1) | Confidence assessment | Interference resistance | DORMANT — hooks never set this field |
| 10 | `emotional_arousal` (0-1) | Arousal tagging | HIGH_AROUSAL_THRESHOLD gating | NEWLY ACTIVE — hooks now classify per event type |
| 11 | `tags[]` | Semantic labels | Tag-based recall, LTP readiness bonus | PARTIAL — hooks set tool/file tags, not semantic intent tags |
| 12 | `episode_id` | Session threading | SESSION_GAP_THRESHOLD, ordinal boost | NEWLY ACTIVE — auto-generated per session |
| 13 | `HOOK_TIMEOUT_MS` (5000) | Latency tolerance | Whether context arrives at all | STATIC — not adaptable per-query |
| 14 | `VELD_TOKEN_BUDGET` (100k) | Context budget | When alerts fire | STATIC — env var only |
| 15 | `rrf_k` (1.0-200.0) | Rank discrimination | How equally sources weighted | NEWLY ACTIVE — exposed per-query via MCP/API |
| 16 | `rerank_count` (20) | Cross-encoder precision | How many candidates get reranked | STATIC — not exposed |
| 17 | `candidate_count` (100) | Recall ceiling | How many candidates fetched per retriever | STATIC — not exposed |
| 18 | `bm25_weight/vector_weight/graph_weight` | Retrieval signal balance | RRF fusion weights | INDIRECT — only via adaptive_weight_learning feedback |
| 19 | `ADAPTIVE_WEIGHT_LEARNING_RATE` (0.05) | How fast Veld learns from feedback | Weight convergence speed | STATIC — hardcoded |
| 20 | `SPREADING_MAX_HOPS` (6) | Graph exploration depth | How far associative traversal goes | STATIC — density-adaptive but not query-controllable |

## Orthogonal Spikes (Independent Axes)

### Spike 1: Temporal Resolution PERP Semantic Precision
```
DECAY_CROSSOVER_DAYS (3.0)     <-> independent of ->  semantic_threshold (0.65)
POWERLAW_BETA (0.5)                                    CONSOLIDATION_QUALITY_GATE (0.6)
LOG_PERIODIC_SCALES [7,30,365]                         ENTITY_CONCEPT_MERGE (0.85)
```
Decay operates on stored memories post-retrieval. Semantic precision operates on query-time similarity. Changing when memories fade has zero effect on how similar a query must be to match.

### Spike 2: Graph Depth PERP Vector Breadth
```
SPREADING_MAX_HOPS (6)         <-> independent of ->  search_list_size (100)
SPREADING_DECAY_RATE (0.5)                             VECTOR_SEARCH_CANDIDATE_MULTIPLIER (3x)
BIDIRECTIONAL_HOPS_* (2-4)                             max_degree (32) in Vamana
```
Graph traversal explores the knowledge graph through entity edges. Vector search explores the embedding space through geometric proximity. Different candidate sets fused in RRF.

### Spike 3: LTP Criteria PERP Compression Policy
```
LTP_THRESHOLD (10)                  <-> independent of ->  COMPRESSION_IMPORTANCE_HIGH (0.8)
LTP_READINESS_THRESHOLD (1.0)                              COMPRESSION_AGE_DAYS (30)
LTP_BURST_THRESHOLD (5 in 24h)                             COMPRESSION_ACCESS_THRESHOLD (10)
```
LTP determines *potentiation* (protection from decay). Compression determines *storage format* (reduced size). Independent operations.

### Spike 4: Emotional Encoding PERP Information Content Weighting
```
emotional_valence (-1 to 1)    <-> independent of ->  IC_NOUN (2.3)
emotional_arousal (0 to 1)                             IC_ADJECTIVE (1.7)
HIGH_AROUSAL_THRESHOLD (0.7)                           IC_VERB (1.0)
```
Emotional tagging affects importance and interference resistance. IC weighting affects *which words matter most* in BM25/linguistic scoring.

### Spike 5: Query Decomposition PERP Adaptive Learning
```
Claude's choice to decompose     <-> independent of ->  ADAPTIVE_WEIGHT_LEARNING_RATE (0.05)
a complex query into multiple                            ADAPTIVE_WEIGHT_MAX_BLEND (0.6)
recall() calls                                           ADAPTIVE_WEIGHT_CONFIDENCE (10)
```
Decomposition is a per-query decision. Adaptive weights learn from feedback over time.

## Fractal Patterns (Self-Similar Across Scales)

### Fractal 1: The Decay Cascade
```
Scale 0 (hours):    L1_DECAY_PER_HOUR = 0.029        exponential
Scale 1 (days):     L2_DECAY_PER_DAY = 0.031          exponential
Scale 2 (months):   L3_DECAY_PER_MONTH = 0.02          power-law transition
Scale 3 (years):    LOG_PERIODIC_SCALES = [7, 30, 365]  log-periodic oscillation

Each scale has: {rate, threshold, promotion_condition, protection_mechanism}
      L1: {0.029/h,  0.1 prune,   strength > 0.5 -> L2,     none}
      L2: {0.031/d,  0.2 prune,   strength > 0.7 -> L3,     episodic replay}
      L3: {0.02/mo,  0.3 prune,   n/a,                       LTP protection}
     LTP: {0.1x normal, 0.05 floor, n/a,                     potentiation}
```

### Fractal 2: The Trust Cascade
```
Scale 0 (edge):     EDGE_TIER_TRUST = [L1: 0.20, L2: 0.50, L3: 0.80, LTP: 0.95]
Scale 1 (memory):   MEMORY_TIER_GRAPH_MULT = [Working: 0.30, Session: 0.60, LT: 1.0, Archive: 1.2]
Scale 2 (retrieval): HYBRID weights = [linguistic: 0.15, graph: 0.35, semantic: 0.50]
Scale 3 (source):   credibility = [inferred: 0.3, ai_generated: 0.5, user: 0.8, verified: 1.0]
```
Trust builds in the same pattern at every scale: low initial, threshold-gated promotion, asymptotic approach to 1.0. The curve `trust(tier) ~ 1 - e^(-k*tier)` fits all four.

### Fractal 3: The Candidate Funnel
```
Scale 0 (index):     All memories (N)
Scale 1 (retrieval): candidate_count = 100          (100/N compression)
Scale 2 (fusion):    RRF top candidates              (~50 after dedup)
Scale 3 (reranking): rerank_count = 20               (20/50 compression)
Scale 4 (delivery):  max_results = 5                  (5/20 compression)
Scale 5 (attention):  Claude's context window focus   (~1-2 actually used)
```
Each stage compresses by ~2-5x. Same operation (score -> threshold -> top-k) repeats.

### Fractal 4: The Learning Rate Cascade
```
ADAPTIVE_WEIGHT_LEARNING_RATE = 0.05      (retrieval weights)
LTP_LEARNING_RATE = 0.10                   (edge potentiation)
HEBBIAN_BOOST_HELPFUL = 0.025              (importance adjustment)
POTENTIATION_MAINTENANCE_BOOST = 0.005     (maintenance cycle)
RECONSOLIDATION_BOOST = 0.02               (replay reinforcement)
```
Product `learning_rate x frequency_of_event` is approximately constant across all five.

## Gap Analysis: The Space Between

### Gap 1: Reasoning Intent (95% information loss)
Claude edits file to "fix critical architectural flaw preventing module composition."
Hook captures: "Modified file: X." All reasoning, intent, validation logic lost.

### Gap 2: Entity Lookup Failures (Silent)
If entity name doesn't match graph, skipped silently. No feedback to Claude.
Claude cannot disambiguate entity resolution failures.

### Gap 3: LTP/Edge Tier Visibility (Fully Hidden)
Claude cannot see:
- LTP status (None/Burst/Weekly/Full)
- Edge tier (L1/L2/L3)
- Activation timestamp history
- Strength floor proximity (how close to pruning)

### Gap 4: Interference Detection (One-Way)
New memories silently weaken old ones via proactive/retroactive interference.
Claude stores contradictory info without knowing it degrades existing memories.

### Gap 5: Dream Replay Artifacts (Indistinguishable)
Weak RelatedTo edges created by random similarity band (0.55-0.85).
Claude cannot distinguish real associations from serendipitous noise.

### Gap 6: Negative Feedback Asymmetry
Positive coactivation: 2-3 cycles to strengthen
Negative signal propagation: 10-20 cycles to suppress
Coactivated misleading memories resist unlearning for 10-15 cycles.

### Gap 7: Auto-Ingested Memory Quality
Auto-ingest (proactive_context auto_ingest=true) creates memories with:
- NO RichContext
- NO emotional signals
- NO episode threading
- 50ms timeout (abandoned if slow)
These are the MAJORITY of stored memories.
