//! `CognitiveDials` — the unified, runtime-tunable surface for Veld's cognitive system.
//!
//! One place to toggle/tune the parameters a human (via the monitor "Dials" tab) or
//! Sleight (via the automated tuning path) can drive at runtime — instead of a second
//! tuning system beside [`crate::memory::feedback::FeedbackStore`]'s learned weights and
//! the Sleight-pushed [`crate::memory::types::ExternalDimensionScores`].
//!
//! Two groups:
//! 1. **Reconsolidation (plasticity)** — gates the FIX-R1 labile-window close + the
//!    composable re-encode layers that run **only in `run_maintenance`** (off the
//!    synchronous recall hot path).
//! 2. **Retrieval (fast↔accurate)** — optional overrides for the hybrid-search knobs;
//!    `None` means "use the engine / learned-weight default" (no behavior change).
//!
//! [`Default`] is everything OFF / `None`, so a freshly-seeded `CognitiveDials` changes
//! nothing until a dial is explicitly turned on. Persisted as JSON (per user) via the
//! KV store; every field is `#[serde(default)]` so older blobs decode forward-compatibly.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::memory::hybrid_search::RefinerMode;

/// Default labile-window length — mirrors [`crate::constants::RECONSOLIDATION_LABILE_WINDOW_SECS`].
fn default_window_secs() -> i64 {
    crate::constants::RECONSOLIDATION_LABILE_WINDOW_SECS
}

/// Default graph-strengthen boost scale (only applied when the toggle is on).
fn default_graph_boost_scale() -> f32 {
    0.5
}

/// Default embedding-nudge blend factor (only applied when the toggle is on).
fn default_embedding_blend() -> f32 {
    0.1
}

/// Default cap on simultaneous labile shadows — mirrors
/// [`crate::constants::RECONSOLIDATION_MAX_ACTIVE_SHADOWS`].
fn default_max_active_shadows() -> usize {
    crate::constants::RECONSOLIDATION_MAX_ACTIVE_SHADOWS
}

/// Default working-memory threshold — mirrors
/// [`crate::constants::RECONSOLIDATION_WORKING_MEMORY_THRESHOLD`].
fn default_working_memory_threshold() -> u32 {
    crate::constants::RECONSOLIDATION_WORKING_MEMORY_THRESHOLD
}

/// KV key under which a user's `CognitiveDials` are persisted. The store is per-user
/// (each `MemorySystem` opens that user's own database), so a bare key is sufficient.
pub const COGNITIVE_DIALS_KEY: &[u8] = b"cognitive_dials";

/// Runtime-tunable cognitive parameters for a single user's [`crate::memory::MemorySystem`].
///
/// See the module docs. All reconsolidation toggles default OFF; all retrieval overrides
/// default to `None` (engine/learned default). Clamp with [`CognitiveDials::sanitized`]
/// after deserialize and before any application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDials {
    // ---- Reconsolidation (plasticity) — all OFF by default --------------------------
    /// Master switch for the FIX-R1 reconsolidation expiry + re-encode block in
    /// `run_maintenance`. When `false`, shadows are neither expired nor re-encoded
    /// (pre-feature behavior).
    #[serde(default)]
    pub reconsolidation_enabled: bool,

    /// Labile-window length in seconds (used when creating/extending shadows on recall).
    #[serde(default = "default_window_secs")]
    pub reconsolidation_window_secs: i64,

    /// Layer 1: strengthen Hebbian graph edges between the memory and the entities found
    /// in its accumulated `retrieval_context` at window close.
    #[serde(default)]
    pub re_encode_graph_strengthen: bool,

    /// Scale applied to the Layer-1 edge boost (× a function of retrieval count).
    #[serde(default = "default_graph_boost_scale")]
    pub graph_boost_scale: f32,

    /// Layer 2: merge genuinely-new `retrieval_context` entities into the memory's
    /// `entity_refs` and persist (re-index) via `update_memory`.
    #[serde(default)]
    pub re_encode_entity_enrich: bool,

    /// Layer 3: blend the memory's embedding toward the `retrieval_context` embedding
    /// and persist (re-index). Heaviest layer; bounded by the active-shadow cap.
    #[serde(default)]
    pub re_encode_embedding_nudge: bool,

    /// Blend factor for Layer 3 (`new = normalize((1-blend)·mem + blend·ctx)`).
    #[serde(default = "default_embedding_blend")]
    pub embedding_blend: f32,

    /// Max simultaneous labile shadows before the oldest (expired-first) are evicted
    /// on the recall path. Caps memory + per-cycle re-encode work.
    #[serde(default = "default_max_active_shadows")]
    pub max_active_shadows: usize,

    /// Consecutive-retrieval count at/above which a shadow is treated as *working
    /// memory* and kept labile (NOT drained at window close). Lower ⇒ drains sooner.
    #[serde(default = "default_working_memory_threshold")]
    pub working_memory_threshold: u32,

    // ---- Retrieval (fast↔accurate) — None = engine/learned default ------------------
    /// Override the post-fusion refiner stage. `None` keeps the engine default.
    #[serde(default)]
    pub refiner_mode: Option<RefinerMode>,

    /// Override whether the cross-encoder rerank runs. `None` keeps the engine default.
    #[serde(default)]
    pub use_reranking: Option<bool>,

    /// Override whether Layer 5.4 MMR diversification runs. `None` keeps the engine default
    /// ([`crate::constants::MMR_ENABLED_DEFAULT`], off).
    #[serde(default)]
    pub use_mmr: Option<bool>,

    /// Override the initial candidate-pool size. `None` keeps the engine default.
    #[serde(default)]
    pub candidate_count: Option<usize>,

    /// Override the rerank top-k. `None` keeps the engine default.
    #[serde(default)]
    pub rerank_count: Option<usize>,

    /// Override ingest-time contextualisation of conversation turns: the embedding
    /// and BM25 document of a `Conversation` memory carry its date and the turn
    /// that preceded it (see `memory::index_text`). `None` keeps the engine
    /// default, which is on; `Some(false)` indexes the bare content.
    #[serde(default)]
    pub turn_context: Option<bool>,

    /// Whether a contextualised turn's embedding also carries its date and the prior
    /// turn, on top of the BM25 document. `None` keeps the engine default, which is
    /// off: on the LoCoMo gate the prior turn's text pulled short turns' embeddings
    /// toward the turn they answered and cost more gold than it found.
    #[serde(default)]
    pub turn_context_embedding: Option<bool>,

    /// Fused candidates kept for the cross-encoder layer (recall Layer 4's truncation).
    /// `None` keeps 30, or 50 for a decomposed query; the pool never drops below
    /// `rerank_count`. Raising `rerank_count` alone cannot reach gold that the truncation
    /// already cut.
    #[serde(default)]
    pub rerank_pool: Option<usize>,

    /// Which cross-encoder reranks, by registry id
    /// (`crate::embeddings::cross_encoder::lookup`). `None` keeps the process-wide choice
    /// (`VELD_CROSS_ENCODER`, else mxbai-rerank-xsmall-v1). Set to
    /// `ms-marco-MiniLM-L-6-v2` for the recall gate's parity arm against upstream.
    /// [`CognitiveDials::sanitized`] drops an id the registry does not know, so what
    /// `GET /api/dials` returns is what reranks.
    #[serde(default)]
    pub rerank_model: Option<String>,

    /// Override the RRF fusion weights `(bm25, vector, graph)`. `None` keeps the
    /// engine/learned weights.
    #[serde(default)]
    pub rrf_weights: Option<(f32, f32, f32)>,

    /// Whether recall adds the date window candidate leg: when a query names a
    /// calendar day or month-year, memories whose `created_at` falls in that window
    /// enter RRF fusion as their own candidate list (recall.rs Layer 4). `None` and
    /// `Some(true)` run it; `Some(false)` leaves the window to the temporal boost only.
    #[serde(default)]
    pub date_window_leg: Option<bool>,

    // ---- Recall ranking sliders (None = compile-time constant default) ---------------
    // Overlay the historical `const`s in recall.rs / hybrid_search.rs / constants.rs.
    // Rendered as sliders in the monitor "Dials" tab; see [`RECALL_SLIDER_SPECS`] for
    // each one's default / recommended / absolute range.
    /// RRF k for the outer graph+hybrid fusion (recall.rs Layer 4). Default 20.
    #[serde(default)]
    pub rrf_k_outer: Option<f32>,
    /// RRF k for the decomposed sub-query merge. Default 20.
    #[serde(default)]
    pub rrf_k_decomp: Option<f32>,
    /// RRF k for the inner BM25+vector fusion (hybrid_search). Default 45.
    #[serde(default)]
    pub rrf_k_inner: Option<f32>,
    /// Weight ratio of the secondary (768d) vector index signal. Default 0.5.
    #[serde(default)]
    pub secondary_rrf_ratio: Option<f32>,
    /// BM25 minimum-score floor for lexical candidates. Default 0.1.
    #[serde(default)]
    pub bm25_min_score: Option<f32>,
    /// Ontological rerank boost per entity-label match. Default 0.15.
    #[serde(default)]
    pub onto_rerank_boost: Option<f32>,
    /// Ontological rerank boost cap. Default 0.40.
    #[serde(default)]
    pub onto_rerank_max: Option<f32>,
    /// Ontological rerank mismatch penalty (negative). Default -0.08.
    #[serde(default)]
    pub onto_rerank_penalty: Option<f32>,
    /// Cross-encoder share of the Layer 5.3 blend for every query shape at once, except that
    /// Abstract (decomposed) never drops below its own share. `None` lets the per-shape dials
    /// and `constants::CE_BLEND_*` decide. See `ce_share`.
    #[serde(default)]
    pub ce_blend: Option<f32>,
    /// Temporal-fact source-memory boost. Default 0.20.
    #[serde(default)]
    pub temporal_recall_boost: Option<f32>,
    /// Cross-encoder share for one query shape; each overrides its `constants::CE_BLEND_*`
    /// default and is itself overridden by `ce_blend`. See `ce_share`.
    #[serde(default)]
    pub ce_blend_identifier: Option<f32>,
    #[serde(default)]
    pub ce_blend_ordinal: Option<f32>,
    #[serde(default)]
    pub ce_blend_temporal: Option<f32>,
    #[serde(default)]
    pub ce_blend_people: Option<f32>,
    #[serde(default)]
    pub ce_blend_keyword: Option<f32>,
    #[serde(default)]
    pub ce_blend_exploratory: Option<f32>,
    #[serde(default)]
    pub ce_blend_attribute: Option<f32>,
    #[serde(default)]
    pub ce_blend_relational: Option<f32>,
    #[serde(default)]
    pub ce_blend_abstract: Option<f32>,
    /// Layer 5.36 query-term coverage weight; `0` disables. Default
    /// `constants::QUERY_COVERAGE_WEIGHT`.
    #[serde(default)]
    pub query_coverage_weight: Option<f32>,
    /// Layer 5.37 person-density boost ceiling for people questions; `0` disables. Default
    /// `constants::PEOPLE_BOOST_MAX`.
    #[serde(default)]
    pub people_boost_max: Option<f32>,
    /// Pseudo-relevance feedback leg. Default `constants::PRF_ENABLED_DEFAULT`.
    #[serde(default)]
    pub use_prf: Option<bool>,
    /// Top first-pass documents PRF reads terms from. Default `constants::PRF_DOCS`.
    #[serde(default)]
    pub prf_docs: Option<usize>,
    /// Expansion terms PRF appends. Default `constants::PRF_TERMS`.
    #[serde(default)]
    pub prf_terms: Option<usize>,
    /// PRF leg weight as a fraction of the hybrid weight. Default `constants::PRF_WEIGHT`.
    #[serde(default)]
    pub prf_weight: Option<f32>,
    /// MMR diversity λ for exploratory queries. Default 0.6.
    #[serde(default)]
    pub mmr_lambda_exploratory: Option<f32>,
    /// MMR diversity λ for abstract queries. Default 0.5.
    #[serde(default)]
    pub mmr_lambda_abstract: Option<f32>,
    /// MMR diversity λ for relationship queries. Default 0.7.
    #[serde(default)]
    pub mmr_lambda_relationship: Option<f32>,
    /// Linguistic weight in the density-adaptive fusion mix. Default 0.15.
    #[serde(default)]
    pub density_linguistic_w: Option<f32>,

    /// Last time these dials were set (set by the writer; informational).
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,

    // ---- Signal disable switches (ablation / benchmarking) -------------------
    /// Set of signal IDs that are disabled at runtime. An empty set (the
    /// default) means every signal runs exactly as compiled — **no behaviour
    /// change**.  Populated from `VELD_DIALS_DISABLE_SIGNALS` (comma-separated
    /// IDs) at startup, or via the `PUT /api/dials` HTTP endpoint at runtime.
    ///
    /// Stable signal IDs:
    /// `seq_proximity`, `context_richness`, `temporal_fact_density`,
    /// `entity_confidence`, `access_burstiness`, `valence`,
    /// `bm25_specificity_discount`, `episode_tcm`, `confidence_gate`,
    /// `content_fallback`, `mmr`, `ordinal_cliff`, `arousal`,
    /// `valence_credibility`, `session_continuity`, `hebbian`, `query_coverage`,
    /// `people_density`, `prf`.
    #[serde(default)]
    pub disabled_signals: HashSet<String>,
}

/// UI/range metadata for a recall slider (consumed by `GET /api/dials/schema`).
pub struct RecallSliderSpec {
    pub id: &'static str,
    pub group: &'static str,
    pub label: &'static str,
    pub default: f64,
    pub rec_lo: f64,
    pub rec_hi: f64,
    pub abs_lo: f64,
    pub abs_hi: f64,
    pub step: f64,
}

/// Default / recommended / absolute ranges for every recall slider. Defaults mirror the
/// historical constants; recommended bands are anchored on veld's value and shodh's
/// tuned counterpart; absolute bounds are where the stage no-ops or the math breaks.
pub const RECALL_SLIDER_SPECS: &[RecallSliderSpec] = &[
    RecallSliderSpec {
        id: "rrf_k_outer",
        group: "Fusion",
        label: "RRF k (graph+hybrid)",
        default: 20.0,
        rec_lo: 20.0,
        rec_hi: 35.0,
        abs_lo: 5.0,
        abs_hi: 100.0,
        step: 1.0,
    },
    RecallSliderSpec {
        id: "rrf_k_decomp",
        group: "Fusion",
        label: "RRF k (sub-query)",
        default: 20.0,
        rec_lo: 20.0,
        rec_hi: 30.0,
        abs_lo: 5.0,
        abs_hi: 100.0,
        step: 1.0,
    },
    RecallSliderSpec {
        id: "rrf_k_inner",
        group: "Fusion",
        label: "RRF k (BM25+vector)",
        default: 45.0,
        rec_lo: 30.0,
        rec_hi: 60.0,
        abs_lo: 5.0,
        abs_hi: 100.0,
        step: 1.0,
    },
    RecallSliderSpec {
        id: "secondary_rrf_ratio",
        group: "Fusion",
        label: "Secondary index weight",
        default: 0.5,
        rec_lo: 0.3,
        rec_hi: 0.7,
        abs_lo: 0.0,
        abs_hi: 2.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "bm25_min_score",
        group: "Candidates",
        label: "BM25 min score",
        default: 0.1,
        rec_lo: 0.0,
        rec_hi: 0.2,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.01,
    },
    RecallSliderSpec {
        id: "onto_rerank_boost",
        group: "Rerank",
        label: "Ontological boost / match",
        default: 0.15,
        rec_lo: 0.05,
        rec_hi: 0.15,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.01,
    },
    RecallSliderSpec {
        id: "onto_rerank_max",
        group: "Rerank",
        label: "Ontological boost cap",
        default: 0.40,
        rec_lo: 0.20,
        rec_hi: 0.40,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.01,
    },
    RecallSliderSpec {
        id: "onto_rerank_penalty",
        group: "Rerank",
        label: "Ontological mismatch penalty",
        default: -0.08,
        rec_lo: -0.20,
        rec_hi: 0.0,
        abs_lo: -1.0,
        abs_hi: 0.0,
        step: 0.01,
    },
    RecallSliderSpec {
        id: "ce_blend",
        group: "Rerank",
        label: "Cross-encoder blend (all shapes)",
        default: 0.18,
        rec_lo: 0.0,
        rec_hi: 0.6,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_identifier",
        group: "Rerank",
        label: "CE share · identifier",
        default: 0.0,
        rec_lo: 0.0,
        rec_hi: 0.1,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_ordinal",
        group: "Rerank",
        label: "CE share · ordinal",
        default: 0.05,
        rec_lo: 0.0,
        rec_hi: 0.2,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_temporal",
        group: "Rerank",
        label: "CE share · temporal",
        default: 0.18,
        rec_lo: 0.0,
        rec_hi: 0.3,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_people",
        group: "Rerank",
        label: "CE share · people",
        default: 0.18,
        rec_lo: 0.0,
        rec_hi: 0.3,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_keyword",
        group: "Rerank",
        label: "CE share · keyword",
        default: 0.10,
        rec_lo: 0.0,
        rec_hi: 0.3,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_exploratory",
        group: "Rerank",
        label: "CE share · exploratory",
        default: 0.18,
        rec_lo: 0.1,
        rec_hi: 0.4,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_attribute",
        group: "Rerank",
        label: "CE share · attribute",
        default: 0.30,
        rec_lo: 0.2,
        rec_hi: 0.5,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_relational",
        group: "Rerank",
        label: "CE share · relational",
        default: 0.35,
        rec_lo: 0.2,
        rec_hi: 0.6,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "ce_blend_abstract",
        group: "Rerank",
        label: "CE share · abstract",
        default: 0.45,
        rec_lo: 0.3,
        rec_hi: 0.7,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "query_coverage_weight",
        group: "Weights",
        label: "Query-term coverage",
        default: 0.25,
        rec_lo: 0.1,
        rec_hi: 0.4,
        abs_lo: 0.0,
        abs_hi: 0.9,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "people_boost_max",
        group: "Weights",
        label: "People-question boost",
        default: 0.0,
        rec_lo: 0.05,
        rec_hi: 0.3,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "prf_docs",
        group: "Expansion",
        label: "PRF documents",
        default: 3.0,
        rec_lo: 2.0,
        rec_hi: 5.0,
        abs_lo: 1.0,
        abs_hi: 20.0,
        step: 1.0,
    },
    RecallSliderSpec {
        id: "prf_terms",
        group: "Expansion",
        label: "PRF terms",
        default: 5.0,
        rec_lo: 3.0,
        rec_hi: 10.0,
        abs_lo: 1.0,
        abs_hi: 30.0,
        step: 1.0,
    },
    RecallSliderSpec {
        id: "prf_weight",
        group: "Expansion",
        label: "PRF leg weight",
        default: 0.3,
        rec_lo: 0.1,
        rec_hi: 0.6,
        abs_lo: 0.0,
        abs_hi: 2.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "temporal_recall_boost",
        group: "Temporal",
        label: "Temporal source boost",
        default: 0.20,
        rec_lo: 0.10,
        rec_hi: 0.40,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.01,
    },
    RecallSliderSpec {
        id: "mmr_lambda_exploratory",
        group: "Diversity",
        label: "MMR λ exploratory",
        default: 0.6,
        rec_lo: 0.5,
        rec_hi: 0.9,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "mmr_lambda_abstract",
        group: "Diversity",
        label: "MMR λ abstract",
        default: 0.5,
        rec_lo: 0.5,
        rec_hi: 0.9,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "mmr_lambda_relationship",
        group: "Diversity",
        label: "MMR λ relationship",
        default: 0.7,
        rec_lo: 0.5,
        rec_hi: 0.9,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.05,
    },
    RecallSliderSpec {
        id: "density_linguistic_w",
        group: "Weights",
        label: "Linguistic weight",
        default: 0.15,
        rec_lo: 0.05,
        rec_hi: 0.25,
        abs_lo: 0.0,
        abs_hi: 1.0,
        step: 0.01,
    },
];

/// Outcome of one reconsolidation-expiry pass (a single `run_maintenance` cycle).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconsolidationStats {
    /// Shadows whose labile window expired and were drained this pass.
    pub expired: usize,
    /// Memories whose window was closed (activation decayed to resting) — Layer 0.
    pub applied: usize,
    /// Graph edges strengthened by Layer 1 (`re_encode_graph_strengthen`).
    pub graph_edges_strengthened: usize,
    /// Memories whose `entity_refs` were enriched by Layer 2 (`re_encode_entity_enrich`).
    pub entities_enriched: usize,
    /// Memories whose embedding was nudged by Layer 3 (`re_encode_embedding_nudge`).
    pub embeddings_nudged: usize,
}

impl Default for CognitiveDials {
    fn default() -> Self {
        Self {
            reconsolidation_enabled: false,
            reconsolidation_window_secs: default_window_secs(),
            re_encode_graph_strengthen: false,
            graph_boost_scale: default_graph_boost_scale(),
            re_encode_entity_enrich: false,
            re_encode_embedding_nudge: false,
            embedding_blend: default_embedding_blend(),
            max_active_shadows: default_max_active_shadows(),
            working_memory_threshold: default_working_memory_threshold(),
            refiner_mode: None,
            use_reranking: None,
            use_mmr: None,
            candidate_count: None,
            rerank_count: None,
            turn_context: None,
            turn_context_embedding: None,
            rerank_pool: None,
            rerank_model: None,
            rrf_weights: None,
            date_window_leg: None,
            rrf_k_outer: None,
            rrf_k_decomp: None,
            rrf_k_inner: None,
            secondary_rrf_ratio: None,
            bm25_min_score: None,
            onto_rerank_boost: None,
            onto_rerank_max: None,
            onto_rerank_penalty: None,
            ce_blend: None,
            ce_blend_identifier: None,
            ce_blend_ordinal: None,
            ce_blend_temporal: None,
            ce_blend_people: None,
            ce_blend_keyword: None,
            ce_blend_exploratory: None,
            ce_blend_attribute: None,
            ce_blend_relational: None,
            ce_blend_abstract: None,
            query_coverage_weight: None,
            people_boost_max: None,
            use_prf: None,
            prf_docs: None,
            prf_terms: None,
            prf_weight: None,
            temporal_recall_boost: None,
            mmr_lambda_exploratory: None,
            mmr_lambda_abstract: None,
            mmr_lambda_relationship: None,
            density_linguistic_w: None,
            updated_at: None,
            disabled_signals: HashSet::new(),
        }
    }
}

impl CognitiveDials {
    /// Clamp every tunable into a safe range. Call after deserialize (untrusted JSON)
    /// and before applying anywhere. Idempotent.
    pub fn sanitized(mut self) -> Self {
        self.reconsolidation_window_secs = self.reconsolidation_window_secs.clamp(30, 3600);
        self.graph_boost_scale = self.graph_boost_scale.clamp(0.0, 1.0);
        self.embedding_blend = self.embedding_blend.clamp(0.0, 0.5);
        self.max_active_shadows = self.max_active_shadows.clamp(1, 500);
        self.working_memory_threshold = self.working_memory_threshold.clamp(1, 100);
        if let Some(c) = self.candidate_count {
            self.candidate_count = Some(c.clamp(1, 1000));
        }
        if let Some(r) = self.rerank_count {
            self.rerank_count = Some(r.clamp(1, 200));
        }
        if let Some(p) = self.rerank_pool {
            self.rerank_pool = Some(p.clamp(10, 500));
        }
        self.rerank_model = self.rerank_model.take().and_then(|id| {
            let id = id.trim();
            if id.is_empty() {
                return None;
            }
            match crate::embeddings::cross_encoder::lookup(id) {
                Ok(_) => Some(id.to_string()),
                Err(e) => {
                    tracing::warn!("rerank_model dial dropped: {e:#}");
                    None
                }
            }
        });
        if let Some((b, v, g)) = self.rrf_weights {
            self.rrf_weights = Some((b.clamp(0.0, 1.0), v.clamp(0.0, 1.0), g.clamp(0.0, 1.0)));
        }
        // Recall sliders — clamp any override to its absolute range (mirrors RECALL_SLIDER_SPECS).
        fn cl(o: &mut Option<f32>, lo: f32, hi: f32) {
            if let Some(x) = o {
                *o = Some(x.clamp(lo, hi));
            }
        }
        cl(&mut self.rrf_k_outer, 5.0, 100.0);
        cl(&mut self.rrf_k_decomp, 5.0, 100.0);
        cl(&mut self.rrf_k_inner, 5.0, 100.0);
        cl(&mut self.secondary_rrf_ratio, 0.0, 2.0);
        cl(&mut self.bm25_min_score, 0.0, 1.0);
        cl(&mut self.onto_rerank_boost, 0.0, 1.0);
        cl(&mut self.onto_rerank_max, 0.0, 1.0);
        cl(&mut self.onto_rerank_penalty, -1.0, 0.0);
        cl(&mut self.ce_blend, 0.0, 1.0);
        cl(&mut self.ce_blend_identifier, 0.0, 1.0);
        cl(&mut self.ce_blend_ordinal, 0.0, 1.0);
        cl(&mut self.ce_blend_temporal, 0.0, 1.0);
        cl(&mut self.ce_blend_people, 0.0, 1.0);
        cl(&mut self.ce_blend_keyword, 0.0, 1.0);
        cl(&mut self.ce_blend_exploratory, 0.0, 1.0);
        cl(&mut self.ce_blend_attribute, 0.0, 1.0);
        cl(&mut self.ce_blend_relational, 0.0, 1.0);
        cl(&mut self.ce_blend_abstract, 0.0, 1.0);
        cl(&mut self.query_coverage_weight, 0.0, 0.9);
        cl(&mut self.people_boost_max, 0.0, 1.0);
        cl(&mut self.prf_weight, 0.0, 2.0);
        self.prf_docs = self.prf_docs.map(|v| v.clamp(1, 20));
        self.prf_terms = self.prf_terms.map(|v| v.clamp(1, 30));
        cl(&mut self.temporal_recall_boost, 0.0, 1.0);
        cl(&mut self.mmr_lambda_exploratory, 0.0, 1.0);
        cl(&mut self.mmr_lambda_abstract, 0.0, 1.0);
        cl(&mut self.mmr_lambda_relationship, 0.0, 1.0);
        cl(&mut self.density_linguistic_w, 0.0, 1.0);
        self
    }

    /// True if any reconsolidation re-encode layer is enabled (Layer 0 close always runs
    /// when `reconsolidation_enabled`, regardless of this).
    pub fn any_re_encode_layer(&self) -> bool {
        self.re_encode_graph_strengthen
            || self.re_encode_entity_enrich
            || self.re_encode_embedding_nudge
    }

    /// Returns `true` when the named recall signal is **enabled** (i.e. not in the
    /// `disabled_signals` set). An empty set (the default) means every signal runs —
    /// the `if` guards in recall.rs that call this are always `true` by default, so
    /// they add zero overhead when no signals are disabled.
    #[inline]
    pub fn signal_enabled(&self, id: &str) -> bool {
        !self.disabled_signals.contains(id)
    }

    /// Whether conversation turns are contextualised at ingest (`turn_context`;
    /// the engine default is on).
    #[inline]
    pub fn turn_context_enabled(&self) -> bool {
        self.turn_context.unwrap_or(true)
    }

    /// Whether a contextualised turn's embedding carries the context too
    /// (`turn_context_embedding`; the engine default is off).
    #[inline]
    pub fn turn_context_embedding_enabled(&self) -> bool {
        self.turn_context_enabled() && self.turn_context_embedding.unwrap_or(false)
    }

    /// Cross-encoder share of the Layer 5.3 blend for a query shape: `ce_blend` when set,
    /// else the shape's own dial, else its `constants::CE_BLEND_*` default.
    pub fn ce_share(&self, shape: crate::memory::query_parser::RerankShape) -> f32 {
        use crate::constants as c;
        use crate::memory::query_parser::RerankShape as S;
        // A global share overrides every shape except Abstract (decomposed), which keeps its
        // own share as a floor: the reranker is what reads abstract membership ("is a Redis
        // fix a performance improvement?"), so lowering every shape must not lower that one.
        // Only `ce_blend_abstract` lowers it.
        if let Some(all) = self.ce_blend {
            return match shape {
                S::Abstract => all.max(self.ce_blend_abstract.unwrap_or(c::CE_BLEND_ABSTRACT)),
                _ => all,
            };
        }
        let (dial, default) = match shape {
            S::Identifier => (self.ce_blend_identifier, c::CE_BLEND_IDENTIFIER),
            S::Ordinal => (self.ce_blend_ordinal, c::CE_BLEND_ORDINAL),
            S::Temporal => (self.ce_blend_temporal, c::CE_BLEND_TEMPORAL),
            S::People => (self.ce_blend_people, c::CE_BLEND_PEOPLE),
            S::Keyword => (self.ce_blend_keyword, c::CE_BLEND_KEYWORD),
            S::Exploratory => (self.ce_blend_exploratory, c::CE_BLEND_EXPLORATORY),
            S::Attribute => (self.ce_blend_attribute, c::CE_BLEND_ATTRIBUTE),
            S::Relational => (self.ce_blend_relational, c::CE_BLEND_RELATIONAL),
            S::Abstract => (self.ce_blend_abstract, c::CE_BLEND_ABSTRACT),
        };
        dial.unwrap_or(default)
    }

    /// Seed dials from process environment (`VELD_*`), overlaying [`Default`]. Used at
    /// `MemorySystem` construction. Unset vars keep the (inert) default. Result is
    /// [`sanitized`](Self::sanitized). Recognized vars:
    /// `VELD_RECONSOLIDATION_{ENABLED,WINDOW_SECS,GRAPH_STRENGTHEN,GRAPH_BOOST_SCALE,
    /// ENTITY_ENRICH,EMBEDDING_NUDGE,EMBEDDING_BLEND,MAX_ACTIVE_SHADOWS,
    /// WORKING_MEMORY_THRESHOLD}`,
    /// `VELD_DIALS_{USE_RERANKING,CANDIDATE_COUNT,RERANK_COUNT,RERANK_MODEL,DATE_WINDOW_LEG}`.
    pub fn from_env() -> Self {
        fn env_bool(key: &str) -> Option<bool> {
            std::env::var(key).ok().map(|v| {
                let v = v.trim();
                v.eq_ignore_ascii_case("true") || v == "1"
            })
        }
        fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
            std::env::var(key).ok().and_then(|v| v.trim().parse().ok())
        }

        let mut d = Self::default();
        if let Some(v) = env_bool("VELD_RECONSOLIDATION_ENABLED") {
            d.reconsolidation_enabled = v;
        }
        if let Some(v) = env_parse::<i64>("VELD_RECONSOLIDATION_WINDOW_SECS") {
            d.reconsolidation_window_secs = v;
        }
        if let Some(v) = env_bool("VELD_RECONSOLIDATION_GRAPH_STRENGTHEN") {
            d.re_encode_graph_strengthen = v;
        }
        if let Some(v) = env_parse::<f32>("VELD_RECONSOLIDATION_GRAPH_BOOST_SCALE") {
            d.graph_boost_scale = v;
        }
        if let Some(v) = env_bool("VELD_RECONSOLIDATION_ENTITY_ENRICH") {
            d.re_encode_entity_enrich = v;
        }
        if let Some(v) = env_bool("VELD_RECONSOLIDATION_EMBEDDING_NUDGE") {
            d.re_encode_embedding_nudge = v;
        }
        if let Some(v) = env_parse::<f32>("VELD_RECONSOLIDATION_EMBEDDING_BLEND") {
            d.embedding_blend = v;
        }
        if let Some(v) = env_parse::<usize>("VELD_RECONSOLIDATION_MAX_ACTIVE_SHADOWS") {
            d.max_active_shadows = v;
        }
        if let Some(v) = env_parse::<u32>("VELD_RECONSOLIDATION_WORKING_MEMORY_THRESHOLD") {
            d.working_memory_threshold = v;
        }
        if let Some(v) = env_bool("VELD_DIALS_USE_RERANKING") {
            d.use_reranking = Some(v);
        }
        if let Some(v) = env_bool("VELD_DIALS_USE_MMR") {
            d.use_mmr = Some(v);
        }
        if let Some(v) = env_parse::<usize>("VELD_DIALS_CANDIDATE_COUNT") {
            d.candidate_count = Some(v);
        }
        if let Some(v) = env_parse::<usize>("VELD_DIALS_RERANK_COUNT") {
            d.rerank_count = Some(v);
        }
        if let Some(v) = env_bool("VELD_DIALS_TURN_CONTEXT") {
            d.turn_context = Some(v);
        }
        if let Some(v) = env_bool("VELD_DIALS_TURN_CONTEXT_EMBEDDING") {
            d.turn_context_embedding = Some(v);
        }
        if let Some(v) = env_parse::<usize>("VELD_DIALS_RERANK_POOL") {
            d.rerank_pool = Some(v);
        }
        if let Ok(v) = std::env::var("VELD_DIALS_RERANK_MODEL") {
            d.rerank_model = Some(v);
        }
        if let Some(v) = env_bool("VELD_DIALS_DATE_WINDOW_LEG") {
            d.date_window_leg = Some(v);
        }
        // Recall sliders — VELD_TUNE_* (unset keeps the const default).
        d.rrf_k_outer = env_parse::<f32>("VELD_TUNE_RRF_K_OUTER").or(d.rrf_k_outer);
        d.rrf_k_decomp = env_parse::<f32>("VELD_TUNE_RRF_K_DECOMP").or(d.rrf_k_decomp);
        d.rrf_k_inner = env_parse::<f32>("VELD_TUNE_RRF_K_INNER").or(d.rrf_k_inner);
        d.secondary_rrf_ratio =
            env_parse::<f32>("VELD_TUNE_SECONDARY_RRF_RATIO").or(d.secondary_rrf_ratio);
        d.bm25_min_score = env_parse::<f32>("VELD_TUNE_BM25_MIN_SCORE").or(d.bm25_min_score);
        d.onto_rerank_boost =
            env_parse::<f32>("VELD_TUNE_ONTO_RERANK_BOOST").or(d.onto_rerank_boost);
        d.onto_rerank_max = env_parse::<f32>("VELD_TUNE_ONTO_RERANK_MAX").or(d.onto_rerank_max);
        d.onto_rerank_penalty =
            env_parse::<f32>("VELD_TUNE_ONTO_RERANK_PENALTY").or(d.onto_rerank_penalty);
        d.ce_blend = env_parse::<f32>("VELD_TUNE_CE_BLEND").or(d.ce_blend);
        d.ce_blend_identifier =
            env_parse::<f32>("VELD_TUNE_CE_BLEND_IDENTIFIER").or(d.ce_blend_identifier);
        d.ce_blend_ordinal = env_parse::<f32>("VELD_TUNE_CE_BLEND_ORDINAL").or(d.ce_blend_ordinal);
        d.ce_blend_temporal =
            env_parse::<f32>("VELD_TUNE_CE_BLEND_TEMPORAL").or(d.ce_blend_temporal);
        d.ce_blend_people = env_parse::<f32>("VELD_TUNE_CE_BLEND_PEOPLE").or(d.ce_blend_people);
        d.ce_blend_keyword = env_parse::<f32>("VELD_TUNE_CE_BLEND_KEYWORD").or(d.ce_blend_keyword);
        d.ce_blend_exploratory =
            env_parse::<f32>("VELD_TUNE_CE_BLEND_EXPLORATORY").or(d.ce_blend_exploratory);
        d.ce_blend_attribute =
            env_parse::<f32>("VELD_TUNE_CE_BLEND_ATTRIBUTE").or(d.ce_blend_attribute);
        d.ce_blend_relational =
            env_parse::<f32>("VELD_TUNE_CE_BLEND_RELATIONAL").or(d.ce_blend_relational);
        d.ce_blend_abstract =
            env_parse::<f32>("VELD_TUNE_CE_BLEND_ABSTRACT").or(d.ce_blend_abstract);
        d.query_coverage_weight =
            env_parse::<f32>("VELD_TUNE_QUERY_COVERAGE_WEIGHT").or(d.query_coverage_weight);
        d.people_boost_max = env_parse::<f32>("VELD_TUNE_PEOPLE_BOOST_MAX").or(d.people_boost_max);
        if let Some(v) = env_bool("VELD_DIALS_USE_PRF") {
            d.use_prf = Some(v);
        }
        d.prf_docs = env_parse::<usize>("VELD_TUNE_PRF_DOCS").or(d.prf_docs);
        d.prf_terms = env_parse::<usize>("VELD_TUNE_PRF_TERMS").or(d.prf_terms);
        d.prf_weight = env_parse::<f32>("VELD_TUNE_PRF_WEIGHT").or(d.prf_weight);
        d.temporal_recall_boost =
            env_parse::<f32>("VELD_TUNE_TEMPORAL_RECALL_BOOST").or(d.temporal_recall_boost);
        d.mmr_lambda_exploratory =
            env_parse::<f32>("VELD_TUNE_MMR_LAMBDA_EXPLORATORY").or(d.mmr_lambda_exploratory);
        d.mmr_lambda_abstract =
            env_parse::<f32>("VELD_TUNE_MMR_LAMBDA_ABSTRACT").or(d.mmr_lambda_abstract);
        d.mmr_lambda_relationship =
            env_parse::<f32>("VELD_TUNE_MMR_LAMBDA_RELATIONSHIP").or(d.mmr_lambda_relationship);
        d.density_linguistic_w =
            env_parse::<f32>("VELD_TUNE_DENSITY_LINGUISTIC_W").or(d.density_linguistic_w);
        // Disable-signals: VELD_DIALS_DISABLE_SIGNALS=id1,id2,...
        // Empty string or unset → empty set (every signal enabled, default behaviour).
        if let Ok(raw) = std::env::var("VELD_DIALS_DISABLE_SIGNALS") {
            d.disabled_signals = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
        }
        d.sanitized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_inert() {
        let d = CognitiveDials::default();
        assert!(!d.reconsolidation_enabled);
        assert!(!d.any_re_encode_layer());
        assert!(d.refiner_mode.is_none());
        assert!(d.use_reranking.is_none());
        assert!(d.turn_context.is_none());
        assert!(d.turn_context_enabled());
        assert!(d.turn_context_embedding.is_none());
        assert!(!d.turn_context_embedding_enabled());
        assert!(d.use_mmr.is_none());
        assert!(d.candidate_count.is_none());
        assert!(d.rerank_count.is_none());
        assert!(d.rerank_pool.is_none());
        assert!(d.rerank_model.is_none());
        assert!(d.rrf_weights.is_none());
        assert!(d.date_window_leg.is_none());
        assert_eq!(
            d.reconsolidation_window_secs,
            crate::constants::RECONSOLIDATION_LABILE_WINDOW_SECS
        );
        // disabled_signals must be empty by default — every signal enabled.
        assert!(d.disabled_signals.is_empty());
    }

    #[test]
    fn ce_share_resolves_global_then_shape_then_constant() {
        use crate::memory::query_parser::RerankShape as S;
        let mut d = CognitiveDials::default();
        assert_eq!(
            d.ce_share(S::Exploratory),
            crate::constants::CE_BLEND_EXPLORATORY
        );
        assert_eq!(
            d.ce_share(S::Identifier),
            crate::constants::CE_BLEND_IDENTIFIER
        );
        d.ce_blend_relational = Some(0.5);
        assert_eq!(d.ce_share(S::Relational), 0.5);
        assert_eq!(d.ce_share(S::Abstract), crate::constants::CE_BLEND_ABSTRACT);
        d.ce_blend = Some(0.2);
        assert_eq!(d.ce_share(S::Relational), 0.2);
        assert_eq!(d.ce_share(S::Identifier), 0.2);
        assert_eq!(d.ce_share(S::Abstract), crate::constants::CE_BLEND_ABSTRACT);
        d.ce_blend = Some(0.6);
        assert_eq!(d.ce_share(S::Abstract), 0.6);
        d.ce_blend = Some(0.2);
        d.ce_blend_abstract = Some(0.3);
        assert_eq!(d.ce_share(S::Abstract), 0.3);
    }

    /// SD-1: With empty disabled_signals, signal_enabled returns true for every
    /// known signal ID (default behaviour is fully unchanged).
    #[test]
    fn signal_enabled_all_on_by_default() {
        let d = CognitiveDials::default();
        let all_ids = [
            "seq_proximity",
            "context_richness",
            "temporal_fact_density",
            "entity_confidence",
            "access_burstiness",
            "valence",
            "bm25_specificity_discount",
            "episode_tcm",
            "confidence_gate",
            "content_fallback",
            "mmr",
            "ordinal_cliff",
            "arousal",
            "valence_credibility",
            "session_continuity",
            "hebbian",
            "query_coverage",
            "people_density",
            "prf",
        ];
        for id in &all_ids {
            assert!(
                d.signal_enabled(id),
                "signal '{}' must be enabled by default (empty disabled_signals)",
                id
            );
        }
    }

    /// SD-2: Constructing dials with disabled_signals containing specific IDs causes
    /// signal_enabled to return false for those IDs and true for all others.
    #[test]
    fn signal_enabled_respects_disabled_set() {
        let mut disabled = HashSet::new();
        disabled.insert("seq_proximity".to_string());
        disabled.insert("valence".to_string());

        let d = CognitiveDials {
            disabled_signals: disabled,
            ..Default::default()
        };

        // These two must be disabled.
        assert!(
            !d.signal_enabled("seq_proximity"),
            "seq_proximity should be disabled"
        );
        assert!(!d.signal_enabled("valence"), "valence should be disabled");

        // All others must remain enabled.
        for id in &[
            "context_richness",
            "temporal_fact_density",
            "entity_confidence",
            "access_burstiness",
            "bm25_specificity_discount",
            "episode_tcm",
            "confidence_gate",
            "content_fallback",
            "mmr",
            "ordinal_cliff",
            "arousal",
            "valence_credibility",
            "session_continuity",
            "hebbian",
        ] {
            assert!(
                d.signal_enabled(id),
                "signal '{}' must still be enabled",
                id
            );
        }
    }

    /// SD-3: VELD_DIALS_DISABLE_SIGNALS env var is parsed and produces the correct
    /// disabled set; all other env-reading behaviour is unchanged.
    #[test]
    fn from_env_parses_disable_signals() {
        // Temporarily set the env var for this test only. This is safe because
        // cargo test runs each #[test] in the same process, but serial tests
        // that set+unset the same var are racey only when run in parallel;
        // here we use a unique-enough var name and restore it after the call.
        std::env::set_var("VELD_DIALS_DISABLE_SIGNALS", "seq_proximity,valence");
        let d = CognitiveDials::from_env();
        std::env::remove_var("VELD_DIALS_DISABLE_SIGNALS");

        assert!(
            !d.signal_enabled("seq_proximity"),
            "seq_proximity must be disabled via env"
        );
        assert!(
            !d.signal_enabled("valence"),
            "valence must be disabled via env"
        );
        assert!(d.signal_enabled("arousal"), "arousal must remain enabled");
        assert!(d.signal_enabled("hebbian"), "hebbian must remain enabled");
        // Default reconsolidation state must be preserved.
        assert!(!d.reconsolidation_enabled);
    }

    #[test]
    fn sanitized_clamps_out_of_range() {
        let d = CognitiveDials {
            reconsolidation_window_secs: 10,
            graph_boost_scale: 5.0,
            embedding_blend: 0.9,
            candidate_count: Some(100_000),
            rerank_count: Some(0),
            rerank_pool: Some(5),
            rrf_weights: Some((2.0, -1.0, 0.5)),
            ..Default::default()
        }
        .sanitized();
        assert_eq!(d.rerank_pool, Some(10));
        assert_eq!(d.reconsolidation_window_secs, 30);
        assert_eq!(d.graph_boost_scale, 1.0);
        assert_eq!(d.embedding_blend, 0.5);
        assert_eq!(d.candidate_count, Some(1000));
        assert_eq!(d.rerank_count, Some(1));
        assert_eq!(d.rrf_weights, Some((1.0, 0.0, 0.5)));
    }

    /// The reranker dials round-trip through the JSON the `/api/dials` routes carry, and
    /// sanitising keeps a registry id, trims it, and drops an unknown or empty one.
    #[test]
    fn rerank_model_and_count_round_trip_and_sanitise() {
        let set = CognitiveDials {
            rerank_model: Some("  ms-marco-MiniLM-L-6-v2 ".to_string()),
            rerank_count: Some(30),
            ..Default::default()
        };
        let json = serde_json::to_string(&set).unwrap();
        let back: CognitiveDials = serde_json::from_str(&json).unwrap();
        let back = back.sanitized();
        assert_eq!(back.rerank_model.as_deref(), Some("ms-marco-MiniLM-L-6-v2"));
        assert_eq!(back.rerank_count, Some(30));

        let unknown = CognitiveDials {
            rerank_model: Some("no-such-reranker".to_string()),
            ..Default::default()
        }
        .sanitized();
        assert_eq!(unknown.rerank_model, None);

        let empty = CognitiveDials {
            rerank_model: Some("   ".to_string()),
            ..Default::default()
        }
        .sanitized();
        assert_eq!(empty.rerank_model, None);
    }

    #[test]
    fn json_forward_compat_missing_fields_default() {
        // An older persisted blob that only had the master toggle must still decode.
        let json = r#"{"reconsolidation_enabled": true}"#;
        let d: CognitiveDials = serde_json::from_str(json).expect("decode partial dials");
        assert!(d.reconsolidation_enabled);
        assert_eq!(
            d.reconsolidation_window_secs,
            crate::constants::RECONSOLIDATION_LABILE_WINDOW_SECS
        );
        assert!(d.refiner_mode.is_none());
    }
}
