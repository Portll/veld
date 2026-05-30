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

    // ---- Retrieval (fast↔accurate) — None = engine/learned default ------------------
    /// Override the post-fusion refiner stage. `None` keeps the engine default.
    #[serde(default)]
    pub refiner_mode: Option<RefinerMode>,

    /// Override whether the cross-encoder rerank runs. `None` keeps the engine default.
    #[serde(default)]
    pub use_reranking: Option<bool>,

    /// Override the initial candidate-pool size. `None` keeps the engine default.
    #[serde(default)]
    pub candidate_count: Option<usize>,

    /// Override the rerank top-k. `None` keeps the engine default.
    #[serde(default)]
    pub rerank_count: Option<usize>,

    /// Override the RRF fusion weights `(bm25, vector, graph)`. `None` keeps the
    /// engine/learned weights.
    #[serde(default)]
    pub rrf_weights: Option<(f32, f32, f32)>,

    /// Last time these dials were set (set by the writer; informational).
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,
}

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
            refiner_mode: None,
            use_reranking: None,
            candidate_count: None,
            rerank_count: None,
            rrf_weights: None,
            updated_at: None,
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
        if let Some(c) = self.candidate_count {
            self.candidate_count = Some(c.clamp(1, 1000));
        }
        if let Some(r) = self.rerank_count {
            self.rerank_count = Some(r.clamp(1, 200));
        }
        if let Some((b, v, g)) = self.rrf_weights {
            self.rrf_weights = Some((b.clamp(0.0, 1.0), v.clamp(0.0, 1.0), g.clamp(0.0, 1.0)));
        }
        self
    }

    /// True if any reconsolidation re-encode layer is enabled (Layer 0 close always runs
    /// when `reconsolidation_enabled`, regardless of this).
    pub fn any_re_encode_layer(&self) -> bool {
        self.re_encode_graph_strengthen
            || self.re_encode_entity_enrich
            || self.re_encode_embedding_nudge
    }

    /// Seed dials from process environment (`VELD_*`), overlaying [`Default`]. Used at
    /// `MemorySystem` construction. Unset vars keep the (inert) default. Result is
    /// [`sanitized`](Self::sanitized). Recognized vars:
    /// `VELD_RECONSOLIDATION_{ENABLED,WINDOW_SECS,GRAPH_STRENGTHEN,GRAPH_BOOST_SCALE,
    /// ENTITY_ENRICH,EMBEDDING_NUDGE,EMBEDDING_BLEND}`,
    /// `VELD_DIALS_{USE_RERANKING,CANDIDATE_COUNT,RERANK_COUNT}`.
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
        if let Some(v) = env_bool("VELD_DIALS_USE_RERANKING") {
            d.use_reranking = Some(v);
        }
        if let Some(v) = env_parse::<usize>("VELD_DIALS_CANDIDATE_COUNT") {
            d.candidate_count = Some(v);
        }
        if let Some(v) = env_parse::<usize>("VELD_DIALS_RERANK_COUNT") {
            d.rerank_count = Some(v);
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
        assert!(d.candidate_count.is_none());
        assert!(d.rrf_weights.is_none());
        assert_eq!(
            d.reconsolidation_window_secs,
            crate::constants::RECONSOLIDATION_LABILE_WINDOW_SECS
        );
    }

    #[test]
    fn sanitized_clamps_out_of_range() {
        let d = CognitiveDials {
            reconsolidation_window_secs: 10,
            graph_boost_scale: 5.0,
            embedding_blend: 0.9,
            candidate_count: Some(100_000),
            rerank_count: Some(0),
            rrf_weights: Some((2.0, -1.0, 0.5)),
            ..Default::default()
        }
        .sanitized();
        assert_eq!(d.reconsolidation_window_secs, 30);
        assert_eq!(d.graph_boost_scale, 1.0);
        assert_eq!(d.embedding_blend, 0.5);
        assert_eq!(d.candidate_count, Some(1000));
        assert_eq!(d.rerank_count, Some(1));
        assert_eq!(d.rrf_weights, Some((1.0, 0.0, 0.5)));
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
