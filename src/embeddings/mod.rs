//! Embedding infrastructure (model loading, inference, trait definitions).
//! Embedding generation module
//!
//! Provides semantic embedding generation for memory retrieval.
//! Uses ONNX Runtime with MiniLM-L6-v2 for 384-dimensional embeddings.
//!
//! # Features
//! - **Auto-download**: Model files downloaded on first use to ~/.cache/veld/
//! - **Circuit breaker**: Automatic fallback when ONNX service is degraded
//! - **Lazy loading**: Model loaded on first embed() call, not at startup
//!
//! # Configuration
//! - `VELD_OFFLINE=true` - Disable auto-download
//! - `VELD_AUTO_DOWNLOAD_MODELS=true` - Explicitly allow model/runtime downloads
//! - `VELD_NEURAL_NER=true` - Enable neural NER when local models exist
//! - `VELD_LAZY_LOAD=false` - Load model at startup
//! - `VELD_ONNX_THREADS=N` - Set ONNX intra-op thread count (default: 1 on macOS ARM64, 2 elsewhere)

pub mod alignment;
pub mod alignment_affine;
pub mod alignment_conditioning;
pub mod alignment_config;
pub mod alignment_csls;
pub mod alignment_procrustes;
pub mod alignment_ridge;
pub mod chunking;
pub mod circuit_breaker;
pub mod competitive;
pub mod cross_encoder;
pub mod downloader;
pub mod http_embedder;
pub mod keywords;
pub mod minilm;
pub mod model_tape;
pub mod ner;
pub mod nomic;
#[cfg(feature = "zenoh")]
pub mod zenoh_embedder;

// Re-export chunking types
pub use chunking::{chunk_text, ChunkConfig, ChunkResult};

use anyhow::Result;

// Re-export downloader functions for convenience
pub use downloader::{
    are_models_downloaded, are_ner_models_downloaded, are_nomic_models_downloaded,
    download_ner_models, download_nomic_models, ensure_downloaded, get_cache_dir, get_models_dir,
    get_ner_models_dir, get_nomic_models_dir, get_onnx_runtime_path, is_onnx_runtime_downloaded,
    print_status,
};

// Re-export NER types
pub use ner::{NerConfig, NerEntity, NerEntityType, NeuralNer};

// Re-export keyword types
pub use keywords::{Keyword, KeywordConfig, KeywordExtractor};

// Re-export circuit breaker types
pub use circuit_breaker::{
    CircuitBreakerConfig, CircuitBreakerMetrics, CircuitState, ResilientEmbedder,
};

// Re-export competitive embedder
pub use competitive::CompetitiveEmbedder;

// Re-export alignment scaffolding (Phase 1) + Procrustes (Phase 3)
pub use alignment::{
    read_alignment_file, resolve_alignment_path, save_alignment, unix_ts_now, Alignment,
    AlignmentHeader, AlignmentPairId, IdentityAlignment,
};
pub use alignment_procrustes::ProcrustesAlignment;
pub use alignment_ridge::RidgeAlignment;

// Alignment v2: conditioning + whitened-Procrustes affine map, CSLS, and the runtime
// performance-flag config (gated behind AlignmentConfig::v2_enabled).
pub use alignment_affine::AffineAlignment;
pub use alignment_conditioning::Conditioning;
pub use alignment_config::{
    AlignmentConfig, AlignmentMethod, AlignmentObjective, CapabilityState, EvalMetric, WhitenMode,
};
pub use alignment_csls::{csls_adjust, local_means, topk_mean};

/// Load a primary embedder by canonical identifier.
///
/// Dispatch rules:
/// - prefix `nomic` → `NomicEmbedder::new(NomicConfig::from_env())`
/// - prefix `minilm` → `MiniLMEmbedder::new(EmbeddingConfig::default())`
/// - `http://` / `https://` URL → `HttpEmbedder::new` with `base_url = id`
/// - anything else → `bail!`
pub fn load_primary_embedder(id: &str) -> Result<std::sync::Arc<dyn Embedder>> {
    load_embedder_by_id(id)
}

/// Load a secondary embedder by canonical identifier. Same dispatch rules as
/// [`load_primary_embedder`] — the distinction is purely informational at the
/// caller level (which side of the alignment pair we're constructing).
pub fn load_secondary_embedder(id: &str) -> Result<std::sync::Arc<dyn Embedder>> {
    load_embedder_by_id(id)
}

fn load_embedder_by_id(id: &str) -> Result<std::sync::Arc<dyn Embedder>> {
    use std::sync::Arc;

    if id.starts_with("nomic") {
        let cfg = nomic::NomicConfig::from_env();
        let e = nomic::NomicEmbedder::new(cfg)?;
        return Ok(Arc::new(e));
    }
    if id.starts_with("minilm") {
        let cfg = minilm::EmbeddingConfig::default();
        let e = minilm::MiniLMEmbedder::new(cfg)?;
        return Ok(Arc::new(e));
    }
    if id.starts_with("http://") || id.starts_with("https://") {
        // `<url>` or `<url>#<model>` — the form HttpEmbedder::id() emits.
        let cfg = http_embedder::HttpEmbedderConfig::from_id(id);
        let e = http_embedder::HttpEmbedder::new(cfg);
        return Ok(Arc::new(e));
    }
    anyhow::bail!(
        "unknown embedder id: {id} (supported prefixes: nomic*, minilm*, http://, https://)"
    );
}

/// The primary embedder this process chose, for `/health` and anyone else who needs to know
/// what the vectors in the store were made by. Recorded once at the first `MemorySystem`
/// construction; every user's system reads the same environment, so first-wins is exact.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PrimaryEmbedderInfo {
    /// Canonical id, the value the corpus marker `.veld_embedder` holds.
    pub id: String,
    pub model: String,
    pub dimension: usize,
    /// `nomic-onnx` | `minilm-onnx` | `http` — where the vectors come from.
    pub kind: &'static str,
    /// Why this one: the env that asked for it, or the fallback step that reached it.
    pub reason: String,
}

static PRIMARY_EMBEDDER: std::sync::OnceLock<PrimaryEmbedderInfo> = std::sync::OnceLock::new();

pub fn record_primary_embedder(info: PrimaryEmbedderInfo) {
    let _ = PRIMARY_EMBEDDER.set(info);
}

pub fn primary_embedder_info() -> Option<PrimaryEmbedderInfo> {
    PRIMARY_EMBEDDER.get().cloned()
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

pub(crate) fn offline_mode_enabled() -> bool {
    env_flag("VELD_OFFLINE", false)
}

pub(crate) fn auto_download_models_enabled() -> bool {
    env_flag("VELD_AUTO_DOWNLOAD_MODELS", false)
}

pub(crate) fn neural_ner_enabled() -> bool {
    env_flag("VELD_NEURAL_NER", false)
}

/// Opt-in to running with NON-SEMANTIC hash embeddings.
///
/// Simplified mode makes `encode()` return hash-derived vectors. Nothing errors,
/// so recall still "works", benchmarks still report numbers, and every one of
/// those numbers is computed against noise. shodh-memory traced an entire voided
/// A/B to exactly this: CI runners failed the model download, vamana recall was
/// all-zero, and the conclusion recorded was "success".
///
/// So the fallback is refused unless it is asked for. It remains genuinely
/// useful for tests and offline smoke runs where nobody is measuring quality —
/// the point is that choosing it is now a decision someone made, not something
/// that happened.
pub(crate) fn allow_simplified_embeddings() -> bool {
    env_flag("VELD_ALLOW_SIMPLIFIED_EMBEDDINGS", false)
}

/// Trait for embedding generation
pub trait Embedder: Send + Sync {
    /// Generate embedding for text.
    ///
    /// This is the *document* side of an asymmetric model — content being
    /// stored/indexed. Symmetric models (e.g. MiniLM) treat documents and
    /// queries identically; see `encode_for_query` for the query side.
    fn encode(&self, text: &str) -> Result<Vec<f32>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Which embedder this IS, as the canonical id `load_embedder_by_id` accepts.
    ///
    /// Required, not defaulted. The trait could report an embedder's SHAPE and not
    /// its identity, which is why the corpus migration below could only key on
    /// dimension — and two different models at the same dimension are the case that
    /// matters, because that is when the marker matches and the migration
    /// short-circuits over a corpus embedded by something else. A default here
    /// would let the next embedder be anonymous and reopen exactly that hole.
    fn id(&self) -> &str;

    /// Encode text as a *query* embedding (retrieval side).
    ///
    /// Asymmetric models (e.g. Nomic, which prepends `search_query: `) override
    /// this to apply the query-specific transform. The default delegates to
    /// `encode`, which is correct for symmetric models. Always use this for
    /// search queries so the document/query asymmetry is honored end-to-end.
    fn encode_for_query(&self, text: &str) -> Result<Vec<f32>> {
        self.encode(text)
    }

    /// Encode text and report whether the result is a degraded fallback.
    /// Returns (embedding, is_degraded). Default: delegates to encode(), reports healthy.
    fn encode_with_status(&self, text: &str) -> Result<(Vec<f32>, bool)> {
        self.encode(text).map(|v| (v, false))
    }

    /// Batch encode multiple texts (default: sequential)
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Canonical, version-pinned identifier for this embedder (e.g.
    /// `"nomic-embed-text-v1.5"`, `"minilm-l6-v2"`). Consulted by the
    /// alignment subsystem to construct [`AlignmentPairId`] and to refuse
    /// loading an alignment fitted on a different pair. Default `"unknown"`
    /// keeps the trait change non-breaking for external implementors; concrete
    /// embedders should override.
    fn model_id(&self) -> &str {
        "unknown"
    }
}
