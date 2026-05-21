# Neuroscience-Driven 5-W Memory Design

> Design proposal — 2026-05-21. Produced by a dedicated Opus research agent
> (maximum-depth reasoning) with built-in FourEyes (4-lens void analysis) and
> Bifocal (edge-walk + fractal) evaluation. Informs and partially revises the
> W3 facet refactor in [REMEDIATION_PLAN.md](../REMEDIATION_PLAN.md): it
> recommends WHAT/WHEN fold into the minimal core and WHERE/WHO/WHY/binding
> become the optional facets — with `WhereFacet` superseding the raw
> `RepositoryContext`-on-`RichContext` wiring landed in `d76c493`.

---

The five W's — WHO, WHERE, WHEN, WHY, WHAT — are not a UI convenience. They are
the canonical decomposition of an *episodic engram*. Tulving defined episodic
memory precisely as the binding of **what** happened, **where**, and **when**,
recollected from a first-person ("autonoetic") vantage. Modern work adds **who**
(source/social) and **why** (intentional/causal/schema) as separable, dissociable
streams. Veld today has fragments of all five, but only WHAT and WHEN are
first-class and queryable; WHO, WHERE, and WHY are smeared across free-text
content, string tags, `HashMap` metadata, and entity names.

## 1. Neuroscience Mapping

Episodic memory is not stored in one place. The hippocampus binds *pointers* into
a sparse, pattern-separated index; the actual content lives in distributed
neocortex. Each W is carried by a partially distinct pathway, and the
hippocampus + entorhinal cortex act as the **conjunctive binder** that
re-instates them together at recall. Design implication: each W should be a
separately encoded, separately indexed facet, *plus* a binding structure that
holds them together as one event.

- **WHAT** — ventral "what" stream (V1→V2→V4→inferotemporal) into the hippocampus
  via lateral entorhinal cortex (LEC) and perirhinal cortex. LEC carries
  non-spatial content; perirhinal cortex supports familiarity. WHAT is the
  dimension that **semanticizes** — over consolidation, repeated WHATs abstract
  into schema and lose episodic specificity (gist/verbatim, fuzzy-trace theory).
- **WHERE** — place cells (CA1/CA3), grid cells (medial EC), boundary cells,
  head-direction cells; dorsal "where/how" parietal stream. The hippocampal map
  is a **general cognitive map** — grid-like codes also organize abstract
  conceptual spaces (Constantinescu et al., 2016). For a software agent, "where"
  is multi-layered: filesystem, repo/branch, module/scope, conceptual
  neighborhood.
- **WHEN** — time cells (CA1) tile elapsed time as place cells tile space; the
  Temporal Context Model (TCM) posits a slowly-drifting context vector, so
  temporal proximity at encoding produces retrieval contiguity; lateral EC
  carries a slow temporal signal. WHEN has three facets: absolute time, ordinal
  position within an episode, and drifting context similarity.
- **WHO** — two dissociable senses: (1) **source memory** — who told me this,
  through what channel (source monitoring; Johnson, Hashtroudi & Lindsay 1993;
  prefrontal cortex + hippocampus); (2) **social/agent cognition** — persons as
  intentional agents (medial PFC, temporoparietal junction, anterior temporal
  lobe; the "social brain"; mPFC also encodes self-vs-other).
- **WHY** — the most model-based W: Event Segmentation Theory (Zacks et al.,
  2007) — the brain runs predictive **event models**; prediction-error spikes
  open a new model (an event boundary). Ventromedial PFC is the **schema** hub
  (Tse et al., 2007; schema-congruent info consolidates faster). Causal/
  intentional inference recruits mPFC + TPJ; prospective goals recruit
  frontopolar cortex.

**The binding fact that drives the design:** the hippocampus does not store the
five W's — it stores a sparse conjunctive code that points at them and lets
partial cues pattern-complete the rest. Veld's analogue: each W is its own typed
facet with its own index, and a lightweight `EngramBinding` ties them so a query
on any W can re-instate the others.

## 2. Gap Assessment

| W | Carriers today | Strength | Weakness |
|---|---|---|---|
| **WHAT** | `Experience.content`, `embeddings`/`embeddings_secondary`, `ExperienceType`, `entities`/`ner_entities`, `SemanticContext.concepts`, `tags` | **Strong** — dual-index vector search, BM25, NER, graph. Genuinely first-class. | Gist-vs-verbatim is implicit (`elaboration_score`/`fragment_demotion` gesture at it); no explicit gist abstraction tier. |
| **WHERE** | `EnvironmentContext.location` (free string), `CodeContext.*`, `RepositoryContext` (W3 scaffold), robotics `geo_location`/`heading`/`terrain_type` | **Fragmented** — coordinates exist for robotics; `RepositoryContext` is the right idea. | No unified "place" concept; physical location is schemaless `Option<String>` with no spatial index; code-place, repo-place, geo-place are three unrelated mechanisms; no conceptual place. |
| **WHEN** | `Memory.created_at`, `MemoryMetadata.last_accessed`/`access_history`, `EpisodeContext`, `TemporalContext`, `temporal_refs`, `valid_until` | **Best after WHAT** — absolute time, ordinal position, episode chaining, temporal invalidation. | No drifting temporal-context vector (TCM core) — contiguity is hard `episode_id` equality, not graded drift; encoding-time vs event-time not crisp. |
| **WHO** | `SourceContext` (source_type, credibility, verified, source_chain), `UserContext`, `agent_id`/`actor_id`, `EntityLabel::Person` | **Provenance half decent; agent/social half thin.** `SourceContext` is a real source-monitoring model. | Two senses collapsed; four disconnected identity representations; no social/agent model (no role, relationship-to-self, self-vs-other); author ≠ subject ≠ audience not distinguished. |
| **WHY** | `causal_chain: Vec<MemoryId>`, `outcomes`, `decision_context`, `predicted_outcome`, `root_cause`, `active_intents`, `UserContext.goals`, `RelationType::Causes` | **Weakest** — pieces exist, scattered, robotics-flavored. | No first-class intention/goal structure; `causal_chain` has no edge semantics; goals are bare `Vec<String>`; event boundaries detected by session heuristics not goal-change/prediction-error; WHY is essentially not queryable. |

**Summary:** WHAT and WHEN are first-class. WHERE is three half-built mechanisms
with no index. WHO is provenance-only with four disjoint identity
representations. WHY is a scatter with no goal/causal-edge model and no
queryability. The W3 facet refactor is the chance to fix all of this — the five
W's should be the *backbone* of the facet taxonomy, not an afterthought.

## 3. The Design — Five W's as First-Class Facets

**Principle.** Restructure context around the **engram**, not the data source.
The five W's become five canonical facets plus one binding facet, attached to
the core record — the hippocampal-index pattern. `RichContext` is not deleted;
its sub-structs are demoted to **detail facets** that the W-facets reference. The
W-facets are the queryable spine; `RichContext` sub-structs are the fine detail.

```rust
//! src/memory/wfacets.rs — the five-W engram facets.

/// The conjunctive binding facet — the hippocampal index.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngramBinding {
    /// Stable hash over (who_id, where_id, when_bucket, why_id) — collision =
    /// candidate duplicate / reconsolidation target.
    pub conjunctive_key: Option<String>,
    /// Recollection strength of the binding (0.0 = gist only, 1.0 = full detail).
    /// Decays faster than the W-facets it binds.
    #[serde(default)]
    pub binding_strength: f32,
    /// Which W-facets are present and reliable.
    #[serde(default)]
    pub present: WFacetMask,
}
```

**WHAT facet** — make the gist/verbatim distinction explicit so consolidation can
semanticize WHAT without destroying the engram:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhatFacet {
    pub verbatim: Option<String>,   // original surface form; may be shed with age
    pub gist: Option<String>,       // abstracted summary; survives consolidation
    pub content_kind: ContentKind,  // replaces ExperienceType
    #[serde(default)] pub content_salience: f32,
    #[serde(default)] pub abstraction_level: f32, // 0.0 raw → 1.0 semanticized
}
```

**WHERE facet** — the key structural fix: one facet, typed `Place` layers,
mirroring the cognitive-map insight that physical, organizational, and
conceptual space share a code:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhereFacet {
    pub places: Vec<Place>,                 // coarse→fine: Repo→Module→File→Scope
    pub conceptual_anchors: Vec<EntityRef>, // where in idea-space (the graph)
    pub heading: Option<f32>,               // direction of activity
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Place {
    Repo { slug: String, branch: Option<String>, commit: Option<String> },
    Module { path: String },
    File { path: String, symbol: Option<String> },
    Geo { lat: f64, lon: f64, alt: Option<f64> },
    Host { name: String, environment: Option<String> },
    Url { href: String },
    Named { label: String },
}
```

`RepositoryContext` (the W3 scaffold) is **absorbed** as `Place::Repo` + `File`/
`Module` entries. `CodeContext` stays as a detail facet (live cursor) referenced
from `WhereFacet`. Robotics `geo_location`/`terrain_type` migrate into
`Place::Geo`/`Named` — delivering the W3 "retire flat robotics fields" goal.

**WHEN facet** — gets the TCM drifting context vector + a clean event-time /
encoding-time split:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhenFacet {
    pub encoded_at: Option<DateTime<Utc>>,  // when Veld stored it
    pub event_time: Option<TimeSpan>,       // when it actually happened
    pub episode_ordinal: Option<u32>,
    #[serde(default)] pub context_drift: Vec<f32>, // ~16-d TCM drift vector
    pub recurrence: Option<RecurrencePattern>,
}
```

`context_drift` is the novelty — a slowly-drifting state captured at encoding;
cosine distance between two engrams' drift vectors = subjective-time proximity,
a *graded* contiguity signal replacing hard `episode_id` equality.

**WHO facet** — splits provenance from agent identity:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhoFacet {
    pub provenance: Provenance,   // subsumes SourceContext; structured relay chain
    pub agents: Vec<AgentRef>,    // author, subject(s), audience, self
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AgentRole { SelfAgent, Author, Subject, Audience, Mentioned }
```

Unifies the four disjoint "who" representations (`actor_id`, `source_id`, graph
`Person`, `UserContext`) and adds the role distinction + self/other marker the
social brain treats as fundamental.

**WHY facet** — the biggest new capability; built on Event Segmentation Theory +
schema:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WhyFacet {
    pub goal_stack: Vec<GoalRef>,          // immediate goal → nesting up
    pub causes: Vec<CausalLink>,           // typed, replaces causal_chain
    pub event_model: Option<String>,       // the "what's happening & why" frame
    pub boundary: Option<BoundaryCause>,   // GoalChange|PredictionError|...
    pub prediction: Option<Prediction>,    // schema-violation signal
    pub schema_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CausalRelation { Caused, Triggered, Enabled, Prevented, Motivated }
```

`GoalRef` points at a first-class `Goal`/`Plan` record (the scaffolded
`PlanFacet` is the natural anchor). `causal_chain`/`outcomes`/`predicted_outcome`/
`root_cause`/`decision_context` all migrate here as typed members.

**Indexing.** Each W gets a dedicated index — the engram-pointer pattern:

| W | Index |
|---|---|
| WHAT | Existing dual Vamana (384d+768d) + BM25 over `gist`/`verbatim` |
| WHERE | New CF `where_index`: geohash prefix keys for `Geo`, path-prefix keys for `File`/`Module` → subtree queries |
| WHEN | `created_at` ordering + `event_time_index` B-tree + a 16-d Vamana over `context_drift` (contiguity = NN in drift space) |
| WHO | Graph entities (via `AgentRef.entity_id`) + CF `provenance_index` keyed by `source_id` and `(AgentRole, entity_id)` |
| WHY | Graph causal edges + CF `goal_index` keyed by `GoalRef.id` |

`EngramBinding.conjunctive_key` gets its own CF for duplicate/reconsolidation
detection.

**Retrieval.** `Query` gains five optional W-predicate fields — *this is the W3
goal*: replace ~20 flat robotics/decision filter fields with structured
W-predicates. Enables: pattern completion (cue on WHO+WHEN → re-instate the
rest via `EngramBinding`); a 5th RRF signal from `context_drift` similarity;
source-gated ranking; WHY-driven recall ("why did we choose X"); gist fallback
when `binding_strength` has decayed.

**Migration** fits W3 step 2: every W-facet is `Option`/`Vec`/`#[serde(default)]`;
legacy memories deserialize empty; a backfill maintenance pass derives W-facets
from existing fields; flat robotics fields get `#[deprecated]` serde aliases for
one release.

## 4. FourEyes Analysis — four independent lenses

1. **Neuroscience fidelity.** Mostly faithful; two voids — (a) no **pattern
   separation** policy (the design *detects* `conjunctive_key` collisions but has
   no "separate vs merge" decision); (b) `binding_strength` decays uniformly, but
   real engrams have **asymmetric W decay** (precise WHEN + verbatim WHAT fade
   first; WHERE/WHO are schema-recoverable). Needs per-W decay rates.
2. **SE / W3 coherence.** Six new facets is *itself* kitchen-sink-shaped. W3's
   point is facets attach *only when relevant* — WHAT/WHEN are near-universal,
   WHERE/WHO/WHY are not. Resolution: **WHAT/WHEN fold into the minimal core**;
   WHERE/WHO/WHY/binding are the optional facets. Matches biology (item+time is
   the minimal trace).
3. **Data / retrieval quality.** Coordinated void: **encoding cost.** Populating
   WHO-roles, WHY-goals, `context_drift` needs inference; hooks fire on every
   tool call. Synchronous enrichment would wreck the hot path. Resolution:
   **two-phase encode** — cheap synchronous capture, then asynchronous
   consolidation-time enrichment. Matches encoding-vs-consolidation in biology.
4. **Adversarial / failure modes.** (a) WHO is a **security surface** —
   `AgentRef`/`Provenance` cross tenant boundaries; must be tenant-scoped (the
   `ba4c508` fail-closed precedent). (b) WHY is a **confabulation surface** —
   inferred causal links stored as fact = the source-misattribution error WHO
   exists to prevent; every inferred field needs `confidence` + `inferred: bool`.
   (c) The n-dimensional void all four lenses circle: the design treats the W's
   as independent, but biology binds them *conjunctively* — the failure mode is
   **false binding** (the WHO of one event recombined with the WHERE of another).
   `EngramBinding` must record binding *confidence*, so a partially-reconstructed
   engram never presents a confabulated conjunction as a real memory.

## 5. Bifocal Evaluation

**Edge-walk.**
- *Serialization.* Bincode is positional — facet fields are append-only, never
  reordered; `Option`/`Vec`/`#[serde(default)]` are safe, `skip_serializing_if`
  is not.
- *Core/facet interface.* WHERE/WHO/WHY are `Option` — every read site needs a
  None branch; accessors return neutral defaults, never `unwrap`.
- *Index/truth boundary.* W5 declares RocksDB truth, indexes rebuildable. The
  five new CFs **must be pure projections** — fully reconstructible by replay.
  This forces W-facet design to co-design with W5.
- *Tenancy.* `WhoFacet.agents` and `provenance_index` must be tenant-partitioned.
- *Encoding-time.* Synchronous W-capture must be O(1) field copies; anything
  needing a model call crosses into async consolidation.

**Fractal.** The same five-W decomposition describes a single memory (micro), an
episode (meso — "the W's held roughly constant while one varies"), and a schema
(macro — recurring (WHO-role, WHERE-type, WHY-goal) tuples). Consolidation is the
operation that moves an engram up the scales by abstracting its W's. The
abstraction is **load-bearing at every scale** — a strong signal. One crack:
`WhyFacet.goal_stack` is hierarchical within a memory but goals are also
cross-memory entities — `GoalRef` must always be a pointer to a graph/`Plan`
entity, never an inline copy (same discipline as `EntityRef`).

## 6. Synthesis

**Recommended approach** — make the five W's the organizing spine of the W3
facet refactor, not a separate workstream:

1. **WHAT and WHEN join the minimal core** (`verbatim`/`gist`/`abstraction_level`;
   `encoded_at`/`event_time`/`episode_ordinal`) — the minimal episodic trace.
2. **WHERE, WHO, WHY, EngramBinding are optional typed facets.** `WhereFacet`
   absorbs the scaffolded `RepositoryContext` + flat robotics location fields.
   `WhoFacet` unifies `SourceContext` + `actor_id` + graph `Person` +
   `UserContext`. `WhyFacet` consolidates `causal_chain`/`outcomes`/
   `decision_context`/`prediction*`/`root_cause`.
3. **Two-phase encoding** — synchronous O(1) capture, async consolidation-time
   enrichment.
4. **Five projection indexes**, all rebuildable, co-designed with W5.
5. **Every inferred W-field carries `confidence` + `inferred`**; retrieval treats
   inferred WHY/WHO as weaker evidence.

**Top risks:** facet proliferation re-creating `RichContext` (mitigate: WHAT/WHEN
in core); encoding-pipeline cost on the hot path (mitigate: two-phase encode —
the highest-likelihood real-world failure); confabulated conjunctions (mitigate:
`binding_strength` + per-field `confidence`); cross-tenant WHO leakage (mitigate:
tenant-partition); index/truth contract violation (mitigate: design the five CFs
strictly as W5 projections).

**Sequencing against W3:**
- *W3.2a (unchanged):* land `RecordKind` on `Memory`.
- *W3.2b (revised):* instead of raw `RepositoryContext` on `RichContext`, land
  `WhereFacet` with `RepositoryContext` as its `Place::Repo` variant — same
  migration surface, strictly more capable.
- *W3.2c:* land `WhatFacet`/`WhenFacet` into the minimal core during the
  core-shrink pass — they replace `ExperienceType`, `temporal_refs`,
  `EpisodeContext`.
- *W3.2d (revised):* the robotics-field migration becomes "location → `WhereFacet`,
  decision/outcome → `WhyFacet`" — retires the flat fields *and* delivers WHY.
- *W3.3:* land `WhoFacet`, `WhyFacet`, `EngramBinding` as optional facets.
- *Co-design with W5:* specify the five projection indexes as replayable
  projections before W5 formalizes the projection layer.

**Bottom line:** the five W's are not a feature layered on top of W3 — they are
the *correct taxonomy* for the facets W3 already committed to building. WHAT and
WHEN belong in the minimal core; WHERE/WHO/WHY/binding are the optional facets;
and `WhereFacet` should replace the raw `RepositoryContext` wiring. Done this
way, the neuroscience design *reduces* total W3 work while turning three
currently-unqueryable dimensions (WHERE, WHO, WHY) into first-class, indexed,
cue-completable facets.
