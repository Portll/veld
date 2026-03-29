# Implementation Plan: Intrusive Recollections Architecture

**Source**: supernova_2026-03-29_intrusive_recollections.json
**Goal**: Transform proactive_context from pathological intrusion system (Ehlers & Clark) into adaptive involuntary memory system (Berntsen)
**Expected compound impact**: MRR +8-13%, retrieval noise -40%, memory accuracy +10-15%/30d, Claude reasoning +15-20%

---

## Implementation Order

The fixes have dependencies:

```
FIX-R2 (elaboration + fragment demotion)  ─┐
                                            ├──> can be done in parallel
FIX-R3 (MMR diversification)             ─┘
                                            │
FIX-R4 (temporal markers)                ───┤  R4 before R1 (simpler, establishes fields R1 populates)
                                            │
FIX-R1 (reconsolidation window)          ───┘  R1 last (most complex, depends on R4's activation semantics)
```

Pipeline integration order after all four:
```
Layer 5.3  (cross-encoder reranking)     [existing]
Layer 5.35 (R2: fragment demotion)       [new]
Layer 5.4  (R3: MMR diversification)     [new]
Layer 5.7  (confidence gating)           [existing]
```

---

## Phase A: FIX-R2 + FIX-R3 (P0, parallel)

### A1. Shared infrastructure — constants and types

**File**: `src/constants.rs`
```rust
// FIX-R2: Fragment demotion
pub const FRAGMENT_DEMOTION_MAX_FACTOR: f32 = 0.6;      // max demotion = 40% score reduction
pub const FRAGMENT_DEMOTION_FLOOR: f32 = 0.1;            // never fully suppress
pub const FRAGMENT_DEMOTION_SIMILARITY_GATE: f32 = 0.7;  // fact must be ≥0.7 similar to source

// FIX-R3: MMR diversification
pub const MMR_LAMBDA_EXPLORATORY: f32 = 0.6;   // strong diversity for exploratory queries
pub const MMR_LAMBDA_RELATIONSHIP: f32 = 0.7;  // moderate diversity for relationship/temporal
```

**File**: `src/memory/types.rs` — add to `MemoryMetadata`:
```rust
/// Elaboration quality score (0.0 = bare content, 1.0 = fully contextualized C-rep)
#[serde(default)]
pub elaboration_score: f32,

/// Fragment demotion factor (1.0 = no demotion, 0.1 = heavily demoted)
/// Applied when a consolidated fact exists for this source fragment
#[serde(default = "default_one")]
pub fragment_demotion: f32,

/// If this memory is a fact extracted from a source, points to the source
#[serde(default)]
pub provenance_of: Option<MemoryId>,
```

Add accessors to `Memory`:
```rust
pub fn elaboration_score(&self) -> f32 { self.metadata.lock().elaboration_score }
pub fn fragment_demotion(&self) -> f32 { self.metadata.lock().fragment_demotion }
pub fn set_fragment_demotion(&self, v: f32) { self.metadata.lock().fragment_demotion = v.clamp(FRAGMENT_DEMOTION_FLOOR, 1.0); }
```

### A2. Elaboration scoring — `src/memory/segmentation.rs`

Add `compute_elaboration_score()`:
```rust
pub fn compute_elaboration_score(memory: &Memory) -> f32 {
    let mut score = 0.0;
    let max = 7.0; // number of dimensions scored

    // 1. RichContext field population
    if memory.rich_context.conversation_context.is_some() { score += 1.0; }
    if memory.rich_context.temporal_context.is_some() { score += 1.0; }
    if memory.rich_context.semantic_context.is_some() { score += 1.0; }

    // 2. Entity diversity
    let entity_count = memory.experience.entities.len();
    score += (entity_count as f32 / 5.0).min(1.0);

    // 3. Emotional signals
    if memory.emotional_valence.is_some() || memory.emotional_arousal.is_some() { score += 1.0; }

    // 4. Temporal specificity (has episode threading)
    if memory.preceding_memory_id.is_some() { score += 1.0; }

    // 5. Content density (information per word)
    let word_count = memory.experience.content.split_whitespace().count();
    let density = if word_count > 0 { entity_count as f32 / word_count as f32 } else { 0.0 };
    score += (density * 10.0).min(1.0);

    (score / max).clamp(0.0, 1.0)
}
```

Call during `remember()` and set `memory.metadata.elaboration_score`.

### A3. Fragment demotion — `src/memory/maintenance.rs`

In `run_maintenance()`, after fact extraction (around line 867-900), add a demotion pass:

```rust
// LAYER: Fragment demotion (FIX-R2)
// When facts are extracted from source fragments, demote the sources
for fact in &extracted_facts {
    if let Some(source_ids) = &fact.source_memory_ids {
        let fact_embedding = fact.embedding.as_deref();
        for source_id in source_ids {
            if let Some(source_mem) = self.get_memory(source_id) {
                // Similarity gate: only demote if fact faithfully represents source
                let similarity = match (fact_embedding, source_mem.embedding()) {
                    (Some(fe), Some(se)) => cosine_similarity(fe, se),
                    _ => 0.0,
                };
                if similarity >= FRAGMENT_DEMOTION_SIMILARITY_GATE {
                    let fact_elab = fact.elaboration_score;
                    let demotion = 1.0 - (fact_elab * FRAGMENT_DEMOTION_MAX_FACTOR);
                    source_mem.set_fragment_demotion(demotion);
                }
            }
        }
    }
}
```

### A4. Fragment demotion at retrieval — `src/memory/recall.rs`

After Layer 5.3 (cross-encoder), before truncation:

```rust
// Layer 5.35: Fragment demotion scoring (FIX-R2)
// Temporal queries exempt — they benefit from episode-level detail
if !has_temporal_query {
    for mem in &memories {
        let demotion = mem.fragment_demotion();
        if demotion < 1.0 {
            // Apply demotion as score multiplier
            let current = mem.score.unwrap_or(0.0);
            mem.set_score(current * demotion);
        }
    }
    memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
}
```

### A5. MMR diversification — `src/memory/recall.rs`

Add helper method `apply_mmr()`:
```rust
fn apply_mmr(memories: &[SharedMemory], lambda: f32, k: usize) -> Vec<SharedMemory> {
    // Greedy MMR selection:
    // 1. Pick highest-scoring memory
    // 2. For each remaining candidate:
    //    score_mmr = lambda * relevance - (1-lambda) * max_sim_to_selected
    // 3. Pick highest MMR score, repeat until k selected
    // Complexity: O(k * n) pairwise cosine sims on 384-dim embeddings (~0.2ms for k=10, n=20)
}
```

Integrate as Layer 5.4 after fragment demotion:
```rust
// Layer 5.4: MMR diversification (FIX-R3)
// Query-type gated: skip for precise factual, apply for exploratory/relationship
let mmr_lambda = match &query_type {
    QueryType::Attribute(_) => None,
    QueryType::Exploratory => Some(MMR_LAMBDA_EXPLORATORY),
    QueryType::Temporal | QueryType::Relationship => Some(MMR_LAMBDA_RELATIONSHIP),
};
if let Some(lambda) = mmr_lambda {
    memories = Self::apply_mmr(&memories, lambda, query.max_results);
}
```

### A6. R2+R3 interaction

These are complementary, not conflicting:
- R2 pre-demotes fragments at consolidation time (persistent, zero retrieval-time cost)
- R3 handles residual redundancy at retrieval time (dynamic, ~0.2ms cost)
- If R2 already demoted a fragment, MMR has less work to do (fragment scores lower, naturally loses to diverse candidates)
- No negative interaction: R2 demotion applied before MMR runs as score multiplier

---

## Phase B: FIX-R4 (P1, after Phase A)

### B1. Extend `ProactiveSurfacedMemory` — `src/handlers/recall.rs`

Add to the struct (after `matched_entities`, line ~185):
```rust
pub hours_since_created: f32,
pub hours_since_last_accessed: f32,
pub access_count: u32,
pub activation_level: f32,
pub retrieval_trigger: String,  // "semantic" | "entity" | "co_activation" | "combined"
pub intrusion_score: f32,       // activation * (1.0 / hours_since_last_accessed.max(0.1))
```

### B2. Populate at construction site — `src/handlers/recall.rs` line ~1535

```rust
let now = Utc::now();
let hours_since_created = (now - m.created_at).num_minutes() as f32 / 60.0;
let hours_since_last_accessed = (now - m.last_accessed()).num_minutes() as f32 / 60.0;
let activation_level = m.activation();
let intrusion_score = activation_level * (1.0 / hours_since_last_accessed.max(0.1));
```

### B3. Update MCP server — `mcp-server/index.ts`

Extend TypeScript interface + add intrusion/provenance markers in formatted output:
```typescript
const intrusionTag = m.intrusion_score > 2.0 ? " [co-activated]"
                   : m.intrusion_score < 0.5 ? " [deep match]" : "";
```

---

## Phase C: FIX-R1 (P1, after Phase B)

### C1. `ReconsolidationShadow` struct — `src/memory/types.rs`

```rust
pub struct ReconsolidationShadow {
    pub memory_id: MemoryId,
    pub opened_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub new_entities: Vec<String>,
    pub updated_importance: Option<f32>,
    pub retrieval_context: String,
    pub consecutive_retrieval_count: u32,
    pub last_retrieval_at: DateTime<Utc>,
}
```

Constants in `src/constants.rs`:
```rust
pub const RECONSOLIDATION_LABILE_WINDOW_SECS: i64 = 300;       // 5 minutes
pub const RECONSOLIDATION_MAX_ACTIVE_SHADOWS: usize = 20;
pub const RECONSOLIDATION_WORKING_MEMORY_THRESHOLD: u32 = 5;
```

### C2. Shadow storage — `src/memory/mod.rs`

Add to `MemorySystem`:
```rust
reconsolidation_shadows: Arc<RwLock<HashMap<MemoryId, ReconsolidationShadow>>>,
```

### C3. Set labile on retrieval — `src/memory/recall.rs`

In `semantic_retrieve()` after `update_access_count_instrumented`:
- Set `memory.set_activation(1.0)`
- Create/extend shadow in `reconsolidation_shadows` map
- If shadow already exists: increment `consecutive_retrieval_count`, reset expiry (last-writer-wins)
- Cap at `RECONSOLIDATION_MAX_ACTIVE_SHADOWS`

### C4. Access burstiness — `src/memory/types.rs`

Wire the orphaned `access_history` field:
```rust
pub fn access_burstiness(&self) -> f32 {
    // Coefficient of variation of inter-access intervals
    // CV > 1.5 = bursty (working memory)
    // CV < 1.0 = steady (long-term retrieval)
}
```

### C5. Process shadows in maintenance — `src/memory/maintenance.rs`

New method `process_reconsolidation_shadows()`:
- Iterate expired shadows (window closed)
- Skip if `consecutive_retrieval_count >= WORKING_MEMORY_THRESHOLD` (stays labile)
- Apply `updated_importance` if set
- Decay activation from 1.0 to resting (0.3) for steady-access memories
- Keep high activation for bursty-access memories (working memory)
- Record `ConsolidationEvent::MemoryStrengthened` with `StrengtheningReason::Reconsolidated`

### C6. Connect R1 to R4

Once R1 is live, the `activation_level` in R4's response carries real signal:
- `activation = 1.0` → just retrieved, labile state
- `activation = 0.3` → recently reconsolidated
- `activation ≈ 0.0` → dormant long-term

Update `retrieval_trigger` detection:
```rust
let retrieval_trigger = if shadows.contains_key(&m.id) {
    "co_activation".to_string()
} else {
    relevance_reason.clone()
};
```

---

## Validation

### Unit tests

| Test | What it validates |
|------|-------------------|
| `test_elaboration_score_bare` | Bare memory → score near 0.0 |
| `test_elaboration_score_rich` | Rich context → score near 1.0 |
| `test_fragment_demotion_similarity_gate` | Bad fact (similarity < 0.7) → no demotion |
| `test_fragment_demotion_temporal_exempt` | Temporal queries skip demotion |
| `test_mmr_identical_embeddings` | 3 near-identical → only 1 selected + 2 diverse |
| `test_mmr_lambda_1` | lambda=1.0 → same order as relevance |
| `test_mmr_skip_attribute` | Attribute query type → MMR skipped |
| `test_reconsolidation_shadow_lifecycle` | Create → extend → expire → apply |
| `test_working_memory_detection` | 5+ consecutive retrievals → stays labile |
| `test_intrusion_score_computation` | high activation + recent = high score |

### Benchmark validation

Run LoCoMo benchmark before/after each phase:
- **Phase A**: MRR on factual (expect +5-8%), exploratory (expect +3-5%), temporal (expect unchanged)
- **Phase B**: No MRR change expected (response metadata only)
- **Phase C**: MRR after 30-day simulation (expect +10-15% accuracy on stale queries)

### Latency budget

| Component | Cost |
|-----------|------|
| Layer 5.35 (fragment demotion) | ~0.1ms (score multiplication) |
| Layer 5.4 (MMR) | ~0.2ms (45 pairwise cosine on 384-dim) |
| R4 field computation | ~0.01ms (arithmetic) |
| R1 shadow create/check | ~0.05ms (HashMap under RwLock) |
| **Total added latency** | **~0.36ms** |

---

## Risk matrix

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Over-demotion of valuable fragments | High | Similarity gate (0.7), floor (0.1), temporal exemption |
| Bad fact demotes good fragment | High | Similarity gate prevents; fragments recoverable via provenance_of |
| MMR hurts precise factual queries | Medium | Query-type gating skips MMR for Attribute queries |
| Shadow accumulation under load | Medium | Cap at 20 active shadows, evict oldest |
| Concurrent access during labile window | Low | Copy-on-write semantics, RwLock on shadow map |
| Serde backward compatibility | Low | All new fields have #[serde(default)] |
