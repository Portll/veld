# Coherence Galaxy Map: Claude Decision Trees x Shodh Memory Architecture

## How to Read This Map

This is a **family tree / galaxy cluster** showing how Claude's decision-making improves when shodh/pinky surfaces information, and what's only visible when you know to look. Each cluster is a decision domain. Lines show information flow. Dotted lines show **flows that should exist but don't yet**.

```
                              ╔═══════════════════════════════════════╗
                              ║     CLAUDE'S DECISION TREE SPACE      ║
                              ║   (what Claude decides, and when)     ║
                              ╚═══════════════╦═══════════════════════╝
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
            ┌───────▼───────┐         ┌───────▼───────┐        ┌───────▼───────┐
            │   FORMULATE   │         │    EXECUTE     │        │   EVALUATE    │
            │   (pre-action)│         │   (action)     │        │  (post-action)│
            └───────┬───────┘         └───────┬───────┘        └───────┬───────┘
                    │                         │                         │
     ┌──────────────┼──────────────┐         │          ┌──────────────┼──────┐
     │              │              │         │          │              │      │
     ▼              ▼              ▼         ▼          ▼              ▼      ▼
  ┌──────┐    ┌──────────┐   ┌────────┐  ┌──────┐  ┌───────┐   ┌─────┐  ┌─────┐
  │QUERY │    │ RETRIEVAL│   │  RRF   │  │TOOL  │  │MEMORY │   │FEED │  │DECAY│
  │CRAFT │    │  MODE    │   │  TUNE  │  │CHOICE│  │ENCODE │   │BACK │  │WATCH│
  └──┬───┘    └────┬─────┘   └───┬────┘  └──┬───┘  └───┬───┘   └──┬──┘  └──┬──┘
     │             │             │           │          │           │        │
     │             │             │           │          │           │        │
```

## CLUSTER 1: QUERY FORMULATION

```
╔════════════════════════════════════════════════════════════════════════╗
║ QUERY CRAFT — How Claude shapes the question                         ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  VISIBLE (shodh surfaces this):                                       ║
║  ● Past memories matching current context (3-5 per prompt)            ║
║  ● Relevant facts with confidence scores                              ║
║  ● Active todos matching query semantics                              ║
║  ● Context-triggered reminders                                        ║
║                                                                       ║
║  IMPROVEMENT FROM SHODH:                                              ║
║  Claude without shodh: Formulates queries from scratch each session.  ║
║  Claude with shodh: Sees "you solved a similar problem 3 days ago     ║
║  with approach X" → reformulates query to be more specific.           ║
║  Delta: 30-60% fewer exploratory queries, faster convergence.         ║
║                                                                       ║
║  ONLY VISIBLE WHEN TOLD TO LOOK:                                      ║
║  ┊ Entity resolution failures — Claude says "Alice" but graph has     ║
║  ┊ "alice_chen". Query returns nothing. Claude doesn't know WHY.      ║
║  ┊                                                                    ║
║  ┊ Query type classification — shodh internally classifies query as   ║
║  ┊ Temporal/Attribute/Exploratory and shifts weights ±0.10. Claude    ║
║  ┊ doesn't know which classification was applied.                     ║
║  ┊                                                                    ║
║  ┊ Discriminativeness score — YAKE computes keyword importance.       ║
║  ┊ High discriminativeness → BM25 weight jumps to 0.75.              ║
║  ┊ Claude doesn't know if its query was deemed discriminative.        ║
║  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ ║
║                                                                       ║
║  FIX-09 would make entity resolution visible                      │  ║
║  FIX-08 would surface recommended_mode + query classification     │  ║
║                                                               ────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## CLUSTER 2: RETRIEVAL MODE SELECTION

```
╔════════════════════════════════════════════════════════════════════════╗
║ MODE SELECT — semantic vs associative vs hybrid                       ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  DECISION TREE:                                                       ║
║                                                                       ║
║  Is query about relationships? ──YES──▶ associative                   ║
║       │                                    │                          ║
║       NO                                   │ graph density < 0.5:     ║
║       │                                    │   graph_w = 0.5 (trust)  ║
║       ▼                                    │ graph density > 2.0:     ║
║  Is query factual/specific? ──YES──▶       │   graph_w = 0.1 (distrust║
║       │                       semantic     │                          ║
║       NO                      (skips       ├──▶ PIPE-7 bidirectional  ║
║       │                        graph)      │    if 2+ entities found  ║
║       ▼                                    │                          ║
║  Default ──────────────────▶ hybrid        │                          ║
║                                            │                          ║
║  IMPROVEMENT FROM SHODH:                                              ║
║  Without mode selection: always hybrid (safe default).                ║
║  With informed mode: associative for relationship queries gives       ║
║  2-3x more relevant results when graph is sparse/mature.              ║
║  Delta: ~15% MRR improvement on relationship queries.                 ║
║                                                                       ║
║  ONLY VISIBLE WHEN TOLD TO LOOK:                                      ║
║  ┊ Graph density value — Claude doesn't see the density number.       ║
║  ┊ At density 0.4, graph weight is 0.5 (maximum trust).              ║
║  ┊ At density 2.1, graph weight is 0.1 (minimum trust).              ║
║  ┊ Claude can't tell if "associative" will help or hurt.             ║
║  ┊                                                                    ║
║  ┊ Edge tier distribution — even in associative mode, only            ║
║  ┊ LTP edges (0.95 trust) really dominate. L1 edges (0.20 trust)     ║
║  ┊ are noise. Claude doesn't know the tier distribution.              ║
║  ┊                                                                    ║
║  ┊ Bidirectional activation — when 2+ entities trigger PIPE-7,        ║
║  ┊ "bridge" entities get 1.5x boost. Claude doesn't see which        ║
║  ┊ entities were found at the intersection.                           ║
║  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ ║
║                                                                       ║
║  FIX-08 surfaces recommended_mode + density in retrieval_stats    │  ║
║  FIX-02 exposes edge tier distribution per memory                 │  ║
║                                                               ────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## CLUSTER 3: RRF TUNING

```
╔════════════════════════════════════════════════════════════════════════╗
║ RRF TUNE — k parameter and weight balance                             ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  NEW CAPABILITY (implemented this session):                           ║
║                                                                       ║
║  rrf_k: 1.0 ◄──────────────────────────────────────────── 200.0      ║
║         │                                                    │        ║
║    winner-take-all                                   democratic       ║
║    (precise factual)                             (exploratory)        ║
║                                                                       ║
║  DECISION TREE:                                                       ║
║                                                                       ║
║  Precise factual lookup ──▶ rrf_k = 5-15 (sharp discrimination)      ║
║  Exploratory/browsing   ──▶ rrf_k = 60-100 (equal weighting)         ║
║  Default (unknown)      ──▶ rrf_k = 20 (balanced)                    ║
║                                                                       ║
║  IMPROVEMENT FROM SHODH:                                              ║
║  Before: K=20 hardcoded. "What color is Alice's car?" and "tell me   ║
║  about the project" got identical rank discrimination.                ║
║  After: Precise queries can use K=5 for rank-1 dominance.            ║
║  Exploratory queries can use K=80 for diverse results.               ║
║  Delta: ~8-12% precision improvement on factual queries.             ║
║                                                                       ║
║  ONLY VISIBLE WHEN TOLD TO LOOK:                                      ║
║  ┊ Two RRF systems with different K values —                          ║
║  ┊ HybridSearchConfig.rrf_k = 45 (BM25+vector internal fusion)       ║
║  ┊ Layer 4 k = query.rrf_k (graph+hybrid+linguistic fusion)          ║
║  ┊ These are DIFFERENT fusion stages with different semantics.        ║
║  ┊ Claude setting rrf_k=5 only affects Layer 4, not BM25/vector.     ║
║  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ ║
║                                                                       ║
║  FIX-12 would unify or document the dual-K system                 │  ║
║                                                               ────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## CLUSTER 4: MEMORY ENCODING

```
╔════════════════════════════════════════════════════════════════════════╗
║ MEMORY ENCODE — What survives from Claude's intent to shodh storage   ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  INFORMATION FUNNEL:                                                  ║
║                                                                       ║
║  Claude's internal state ─────────────────────────────── 100%         ║
║       │                                                               ║
║       ▼ hook captures content + type + tags + emotional               ║
║  Hook payload ────────────────────────────────────────── ~20%         ║
║       │                                                               ║
║       ▼ content truncated, NER extracts entities                      ║
║  API processing ──────────────────────────────────────── ~10%         ║
║       │                                                               ║
║       ▼ entities lose relationships, all become co-occurrence         ║
║  Graph storage ───────────────────────────────────────── ~5%          ║
║       │                                                               ║
║       ▼ importance = type_weight + richness (NOT reasoning quality)   ║
║  Importance score ────────────────────────────────────── ~2%          ║
║                                                                       ║
║  SPECIFIC LOSSES:                                                     ║
║  ● "Refactored auth for DI" → "Modified file: auth.ts" (95% loss)   ║
║  ● "Critical security fix" → importance = 0.05 (FileAccess)          ║
║  ● "Chose A over B because X" → entities [A, B] co-occur (100% WHY) ║
║  ● Entity relationships → all flattened to "co-occurrence" edges      ║
║                                                                       ║
║  IMPROVEMENT FROM SHODH (current):                                    ║
║  Without shodh: zero persistence between sessions.                    ║
║  With shodh: 5% of reasoning survives. Better than 0%.               ║
║  Delta: infinite improvement, but low absolute value.                 ║
║                                                                       ║
║  ONLY VISIBLE WHEN TOLD TO LOOK:                                      ║
║  ┊ Auto-ingested memories (majority) have NO RichContext,             ║
║  ┊ NO emotional signals, NO episode threading. They're bare           ║
║  ┊ content + type. The segmentation engine creates these with         ║
║  ┊ a 50ms timeout (abandoned if slow).                                ║
║  ┊                                                                    ║
║  ┊ Importance formula is structurally blind to reasoning quality.     ║
║  ┊ word_count + entity_count + type_weight. A 200-word rambling       ║
║  ┊ observation scores HIGHER than a precise 10-word decision.         ║
║  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ ║
║                                                                       ║
║  FIX-01 structured reasoning capture                              │  ║
║  FIX-05 auto-ingest quality parity                                │  ║
║  FIX-07 credibility field activation                              │  ║
║  FIX-10 semantic intent tags                                      │  ║
║                                                               ────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## CLUSTER 5: FEEDBACK LOOP

```
╔════════════════════════════════════════════════════════════════════════╗
║ FEEDBACK — How Claude's behavior teaches shodh (and what's lost)      ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  THE FEEDBACK CIRCUIT (7 stages):                                     ║
║                                                                       ║
║  Stage 1: Hook captures tool actions ────────── 40% attenuation       ║
║  Stage 2: Request assembly (truncation) ─────── 60% attenuation       ║
║  Stage 3: Feedback processing (EMA smoothing) ─ 50% attenuation       ║
║  Stage 4: Hebbian learning (±0.025/cycle) ───── 60% attenuation       ║
║  Stage 5: Weight learning (±0.01/signal) ────── 99% attenuation       ║
║  Stage 6: Reconsolidation (edge-dependent) ──── 50% attenuation       ║
║  Stage 7: Context injection (120 char/memory) ─ 60% attenuation       ║
║                                                                       ║
║  END-TO-END SIGNAL SURVIVAL: ~0.2% of original signal                 ║
║                                                                       ║
║  UNLEARNING LATENCY:                                                  ║
║  ● Explicit rejection ("wrong"): 5-15 cycles                         ║
║  ● Implicit disuse: 20-50 cycles                                      ║
║  ● Coactivated misleading memory: 10-20 cycles                       ║
║  ● Adaptive weight shift: 200+ cycles (asymptotic)                    ║
║                                                                       ║
║  IMPROVEMENT FROM SHODH:                                              ║
║  Without feedback: static retrieval, no adaptation.                   ║
║  With feedback: system gradually learns which signal (BM25/vector/    ║
║  graph) works best. Misleading memories eventually suppressed.        ║
║  Delta: ~5-10% MRR improvement over 100+ cycles of feedback.         ║
║                                                                       ║
║  THE ASYMMETRY PROBLEM:                                               ║
║  ┊ Positive coactivation: 2-3 cycles to strengthen                    ║
║  ┊ Negative unlearning: 10-20 cycles to suppress                      ║
║  ┊                                                                    ║
║  ┊ This means: a helpful memory pair strengthens in 2 interactions.   ║
║  ┊ A misleading memory that was ONCE helpful resists correction       ║
║  ┊ for 10-20 interactions because its L2 edge keeps importance high.  ║
║  ┊                                                                    ║
║  ┊ Biological analogy: extinction is harder than conditioning.        ║
║  ┊ Shodh faithfully reproduces this bug from neuroscience.            ║
║  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ ║
║                                                                       ║
║  FIX-04 bidirectional feedback acceleration (strong signals bypass)│  ║
║  FIX-03 interference notification (know when damage occurs)       │  ║
║                                                               ────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## CLUSTER 6: DECAY & MEMORY HEALTH

```
╔════════════════════════════════════════════════════════════════════════╗
║ DECAY WATCH — What Claude can't see but would change its behavior     ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  THE HIDDEN DIMENSIONS:                                               ║
║                                                                       ║
║  ┌──────────────────────────────────────────────────────────────┐     ║
║  │ What shodh knows              │ What Claude sees              │     ║
║  ├──────────────────────────────────────────────────────────────┤     ║
║  │ LTP status: Full (0.1x decay) │ (nothing)                    │     ║
║  │ Edge tier: L1 (48h to die)    │ (nothing)                    │     ║
║  │ Strength: 0.12 (near prune)   │ (nothing)                    │     ║
║  │ Interference: old memory -8%  │ (nothing)                    │     ║
║  │ Dream replay edge (speculative)│ (looks like real edge)       │     ║
║  │ Fact confidence: 0.35 (weak)  │ (nothing — or shows as fact) │     ║
║  │ Log-periodic resonance at 7d  │ (nothing)                    │     ║
║  │ Activation pattern: burst 5/h │ (nothing)                    │     ║
║  └──────────────────────────────────────────────────────────────┘     ║
║                                                                       ║
║  BEHAVIORAL CHANGES IF VISIBLE:                                       ║
║                                                                       ║
║  If Claude saw L1 edge at 40h (8h from death):                        ║
║    → Would re-access that memory to prevent pruning                   ║
║    → Would know associative path through that edge is fragile         ║
║                                                                       ║
║  If Claude saw LTP status = Full:                                     ║
║    → Would trust that association is permanent (0.1x decay)           ║
║    → Would use it as a reliable foundation for reasoning              ║
║                                                                       ║
║  If Claude saw interference event:                                    ║
║    → Would resolve contradiction explicitly instead of silently       ║
║    → Would choose which memory to keep based on reasoning, not age    ║
║                                                                       ║
║  If Claude saw log-periodic scales [7, 30, 365]:                      ║
║    → Would know to check important memories at weekly intervals       ║
║    → Would understand why some memories survive week 7 but not 10     ║
║                                                                       ║
║  IMPROVEMENT FROM SHODH (current):                                    ║
║  Without shodh: no persistence, every session starts fresh.           ║
║  With shodh: memories survive across sessions with biological decay.  ║
║  Delta: persistence exists. But Claude operates it blindly.           ║
║                                                                       ║
║  THE ONE-WAY TEACHING RELATIONSHIP:                                   ║
║  ┊ The memory system learns from Claude (access patterns, feedback).  ║
║  ┊ Claude learns nothing about the memory system's state.             ║
║  ┊ This is like a student who studies with a teacher who never gives  ║
║  ┊ grades — the teacher learns the student's patterns, but the        ║
║  ┊ student doesn't know what they know or don't know.                 ║
║  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ ║
║                                                                       ║
║  FIX-02 memory health observability endpoint                      │  ║
║  FIX-06 dream replay edge tagging                                 │  ║
║                                                               ────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## THE IMPROVEMENT SURFACE: Delta Map

```
                    LOW EFFORT ◄─────────────────────────── HIGH EFFORT
                         │                                       │
    HIGH              ┌──┤                                       │
    IMPACT            │  │  FIX-07 credibility          FIX-01 reasoning
                      │  │  (trivial, dormant dim)       capture (medium,
                      │  │  +15% data integrity          95% loss fix)
                      │  │                                       │
                      │  │  FIX-04 feedback accel        FIX-02 health
                      │  │  (low, 5x unlearning)         endpoint (medium,
                      │  │                               observability)
                      │  │  FIX-05 auto-ingest parity           │
                      │  │  (low, majority of memories)  FIX-10 intent tags
                      │  │                               (medium, importance)
                      │  │  FIX-03 interference notify          │
                      │  │  (low, RPN 160→50)            FIX-09 entity
                      │  │                               feedback (medium)
                      │  │  FIX-06 dream edge tag               │
    LOW               │  │  (low)                        FIX-08 mode
    IMPACT            │  │                               recommend (medium)
                      │  │  FIX-12 K audit                      │
                      └──┤  (trivial)                    FIX-11 rerank
                         │                               expose (trivial)
                         │                                       │
```

## Score Summary

| Dimension | Current | After Fixes | Delta |
|-----------|---------|-------------|-------|
| Coherence | 4.2 | 7.0 | +2.8 |
| Data Integrity | 4.5 | 7.5 | +3.0 |
| Robustness | 5.2 | 7.0 | +1.8 |
| Usability | 5.8 | 7.5 | +1.7 |
| Completeness | 5.0 | 7.0 | +2.0 |
| **Total** | **72.9/120** | **92/120** | **+19.1** |

The two blockers (Coherence 4.2, Data Integrity 4.5) would be resolved, moving
the system from 60.75% to ~76.7%.
