//! Earth layer: embedding infrastructure (model loading, inference, trait definitions).
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
        let mut cfg = http_embedder::HttpEmbedderConfig::from_env();
        cfg.base_url = id.to_string();
        let e = http_embedder::HttpEmbedder::new(cfg);
        return Ok(Arc::new(e));
    }
    anyhow::bail!(
        "unknown embedder id: {id} (supported prefixes: nomic*, minilm*, http://, https://)"
    );
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
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
