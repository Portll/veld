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
    /// Generate embedding for text
    fn encode(&self, text: &str) -> Result<Vec<f32>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Encode text and report whether the result is a degraded fallback.
    /// Returns (embedding, is_degraded). Default: delegates to encode(), reports healthy.
    fn encode_with_status(&self, text: &str) -> Result<(Vec<f32>, bool)> {
        self.encode(text).map(|v| (v, false))
    }

    /// Batch encode multiple texts (default: sequential)
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
}
