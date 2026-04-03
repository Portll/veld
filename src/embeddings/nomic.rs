//! Nomic-embed-text-v1.5 embedding model using ONNX Runtime
//!
//! Generates 768-dimensional sentence embeddings with Matryoshka representation learning.
//! Model: nomic-ai/nomic-embed-text-v1.5
//!
//! Key differences from MiniLM:
//! - 768-dimensional embeddings (vs 384)
//! - Requires task-specific prefixes: "search_document: " for documents, "search_query: " for queries
//! - Supports up to 8192 tokens (we default to 512 for edge efficiency)
//! - Uses rotary position embeddings (RoPE)
//!
//! Edge Optimizations:
//! - Lazy model loading: Model is only loaded on first embed call
//! - Configurable thread count for power efficiency
//! - Simplified fallback for resource-constrained devices
//!
//! Configuration via environment variables:
//! - SHODH_NOMIC_MODEL_PATH: Base path to model files (default: ~/.cache/shodh-memory/models/nomic-embed-v1.5)
//! - SHODH_NOMIC_EMBED_TIMEOUT_MS: Embedding timeout in ms (default: 5000)
//! - SHODH_LAZY_LOAD: Set to "false" to load model at startup (default: true)
//! - SHODH_ONNX_THREADS: Number of ONNX threads (default: 1 on macOS ARM64, 2 elsewhere)

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

use super::Embedder;

/// Nomic task prefixes for asymmetric embedding
const SEARCH_DOCUMENT_PREFIX: &str = "search_document: ";
const SEARCH_QUERY_PREFIX: &str = "search_query: ";

/// Lazily initialized ONNX session and tokenizer for Nomic
struct LazyNomicModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl LazyNomicModel {
    fn new(config: &NomicConfig) -> Result<Self> {
        // macOS ARM64 (M1/M2/M3): default to 1 thread to avoid Eigen thread pool
        // spin-to-block deadlock on heterogeneous P/E cores.
        // See: https://github.com/microsoft/onnxruntime/issues/10270
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let default_threads = 1;
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let default_threads = 2;

        let num_threads = std::env::var("SHODH_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_threads);

        tracing::info!(
            "Loading Nomic-embed-text-v1.5 model from {:?} with {} threads",
            config.model_path,
            num_threads
        );

        let builder = Session::builder()
            .context("Failed to create session builder")?
            .with_intra_threads(num_threads)
            .context("Failed to set intra thread count")?
            .with_inter_threads(1)
            .context("Failed to set inter thread count")?;

        // Disable thread pool spinning to prevent Eigen spin-to-block deadlock
        // on macOS ARM64 heterogeneous cores (P-core/E-core architecture).
        let builder = builder
            .with_intra_op_spinning(false)
            .context("Failed to disable intra-op spinning")?
            .with_inter_op_spinning(false)
            .context("Failed to disable inter-op spinning")?;

        let session = builder
            .commit_from_file(&config.model_path)
            .context("Failed to load Nomic ONNX model")?;

        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load Nomic tokenizer: {e}"))?;

        tracing::info!("Nomic-embed-text-v1.5 model loaded successfully");

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }
}

/// Configuration for Nomic embedder
#[derive(Debug, Clone)]
pub struct NomicConfig {
    /// Path to ONNX model file
    pub model_path: PathBuf,

    /// Path to tokenizer file
    pub tokenizer_path: PathBuf,

    /// Maximum sequence length (Nomic supports 8192, default: 512 for edge efficiency)
    pub max_length: usize,

    /// Use quantized model for faster inference
    pub use_quantized: bool,

    /// Timeout for embedding generation in milliseconds
    pub embed_timeout_ms: u64,
}

impl Default for NomicConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl NomicConfig {
    /// Create configuration from environment variables with sensible defaults
    ///
    /// Search order for model files:
    /// 1. SHODH_NOMIC_MODEL_PATH environment variable
    /// 2. Bundled in Python package (VELD_PACKAGE_DIR/models/nomic-embed-v1.5)
    /// 3. ./models/nomic-embed-v1.5 (local)
    /// 4. ../models/nomic-embed-v1.5 (parent)
    /// 5. ~/.cache/shodh-memory/models/nomic-embed-v1.5 (auto-download location)
    pub fn from_env() -> Self {
        let base_path = std::env::var("SHODH_NOMIC_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let candidates = vec![
                    // Bundled in Python package (highest priority for pip install)
                    std::env::var("VELD_PACKAGE_DIR")
                        .ok()
                        .map(|p| PathBuf::from(p).join("models/nomic-embed-v1.5")),
                    Some(PathBuf::from("./models/nomic-embed-v1.5")),
                    Some(PathBuf::from("../models/nomic-embed-v1.5")),
                    // Auto-download cache location
                    Some(super::downloader::get_nomic_models_dir()),
                    dirs::data_dir().map(|p| p.join("shodh-memory/models/nomic-embed-v1.5")),
                ];

                candidates
                    .into_iter()
                    .flatten()
                    .find(|p| {
                        p.join("model_quantized.onnx").exists() || p.join("model.onnx").exists()
                    })
                    .unwrap_or_else(super::downloader::get_nomic_models_dir)
            });

        let embed_timeout_ms = std::env::var("SHODH_NOMIC_EMBED_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5000);

        let use_quantized = std::env::var("SHODH_NOMIC_USE_QUANTIZED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        let model_filename = if use_quantized {
            "model_quantized.onnx"
        } else {
            "model.onnx"
        };

        Self {
            model_path: base_path.join(model_filename),
            tokenizer_path: base_path.join("tokenizer.json"),
            max_length: 512,
            use_quantized,
            embed_timeout_ms,
        }
    }

    /// Create configuration with explicit paths (for testing or programmatic use)
    pub fn with_paths(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            model_path,
            tokenizer_path,
            max_length: 512,
            use_quantized: true,
            embed_timeout_ms: 5000,
        }
    }
}

/// Nomic-embed-text-v1.5 embedder with ONNX Runtime
///
/// Features lazy model loading for edge devices:
/// - Model is only loaded on first embed() call
/// - Reduces startup time from ~2s to <100ms
/// - Reduces idle RAM until first use
///
/// Nomic requires task-specific prefixes:
/// - `encode_document()` prepends "search_document: "
/// - `encode_query()` prepends "search_query: "
/// - `encode()` (trait method) defaults to "search_document: " prefix
pub struct NomicEmbedder {
    config: NomicConfig,
    /// Lazily initialized model (OnceLock for thread-safe init)
    lazy_model: OnceLock<Result<Arc<LazyNomicModel>, String>>,
    /// Flag for simplified mode (no ONNX)
    simplified_mode: bool,
    dimension: usize,
}

impl NomicEmbedder {
    /// Ensure ONNX Runtime is available before any ort code runs.
    /// Delegates to the shared ORT_PATH_INIT in the minilm module since
    /// both embedders share the same ONNX Runtime binary.
    fn ensure_onnx_runtime_available(offline_mode: bool) -> Result<()> {
        // Re-use the same OnceLock-based init from minilm to avoid duplicate set_var
        super::minilm::pre_init_ort_runtime(offline_mode);

        // Check if initialization succeeded
        if let Ok(existing) = std::env::var("ORT_DYLIB_PATH") {
            if std::path::Path::new(&existing).exists() {
                return Ok(());
            }
        }

        // Check cached path
        if super::downloader::get_onnx_runtime_path().is_some() {
            return Ok(());
        }

        if offline_mode {
            anyhow::bail!("ONNX Runtime not found and SHODH_OFFLINE=true");
        }

        if !super::auto_download_models_enabled() {
            anyhow::bail!(
                "ONNX Runtime not found locally and SHODH_AUTO_DOWNLOAD_MODELS is not enabled"
            );
        }

        // Download as last resort
        tracing::info!("ONNX Runtime not found locally. Downloading...");
        let onnx_path = super::downloader::download_onnx_runtime(None)?;
        std::env::set_var("ORT_DYLIB_PATH", &onnx_path);
        Ok(())
    }

    /// Create new Nomic embedder with lazy loading (default)
    ///
    /// Model is NOT loaded until first embed() call.
    /// Set SHODH_LAZY_LOAD=false to load immediately.
    /// Set SHODH_OFFLINE=true to disable auto-download.
    /// Set SHODH_AUTO_DOWNLOAD_MODELS=true to explicitly allow downloads.
    ///
    /// Auto-download behavior:
    /// - If model files not found, downloads from HuggingFace (~65MB quantized)
    /// - If ONNX Runtime not found, downloads from GitHub (~50MB)
    /// - Files cached in ~/.cache/shodh-memory/models/nomic-embed-v1.5/
    pub fn new(config: NomicConfig) -> Result<Self> {
        let lazy_load = std::env::var("SHODH_LAZY_LOAD")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        let offline_mode = super::offline_mode_enabled();

        // Ensure ORT_DYLIB_PATH is set before any ort code runs
        if let Err(e) = Self::ensure_onnx_runtime_available(offline_mode) {
            tracing::warn!(
                "Failed to set up ONNX Runtime: {}. Using simplified Nomic embeddings.",
                e
            );
            return Self::new_simplified(config);
        }

        // Check if model files exist
        let model_available = config.model_path.exists() && config.tokenizer_path.exists();

        if !model_available {
            if offline_mode {
                tracing::warn!(
                    "Nomic model files not found and SHODH_OFFLINE=true. Using simplified embeddings.",
                );
                return Self::new_simplified(config);
            }

            if !super::auto_download_models_enabled() {
                tracing::warn!(
                    "Nomic model files not found locally and SHODH_AUTO_DOWNLOAD_MODELS is not enabled. Using simplified embeddings.",
                );
                return Self::new_simplified(config);
            }

            // Try to auto-download model files
            tracing::info!(
                "Nomic model files not found locally at {:?}. Downloading...",
                config.model_path.parent().unwrap_or(&config.model_path)
            );

            match super::downloader::download_nomic_models(Some(std::sync::Arc::new(
                |downloaded, total| {
                    if total > 0 {
                        let percent = (downloaded as f64 / total as f64 * 100.0) as u32;
                        if percent.is_multiple_of(10) {
                            tracing::info!(
                                "Downloading Nomic models: {}% ({}/{})",
                                percent,
                                downloaded,
                                total
                            );
                        }
                    }
                },
            ))) {
                Ok(models_dir) => {
                    tracing::info!("Nomic models downloaded to {:?}", models_dir);

                    let model_filename = if config.use_quantized {
                        "model_quantized.onnx"
                    } else {
                        "model.onnx"
                    };
                    let updated_config = NomicConfig {
                        model_path: models_dir.join(model_filename),
                        tokenizer_path: models_dir.join("tokenizer.json"),
                        ..config
                    };

                    return Self::new(updated_config);
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to download Nomic models: {}. Using simplified embeddings.",
                        e
                    );
                    return Self::new_simplified(config);
                }
            }
        }

        let embedder = Self {
            config: config.clone(),
            lazy_model: OnceLock::new(),
            simplified_mode: false,
            dimension: 768,
        };

        if !lazy_load {
            tracing::info!("Eager loading Nomic ONNX model (SHODH_LAZY_LOAD=false)");
            embedder.ensure_model_loaded()?;
        } else {
            tracing::info!("Lazy loading enabled - Nomic model will load on first embed()");
        }

        Ok(embedder)
    }

    /// Ensure the model is loaded (thread-safe, idempotent)
    fn ensure_model_loaded(&self) -> Result<&Arc<LazyNomicModel>> {
        let result = self.lazy_model.get_or_init(|| {
            LazyNomicModel::new(&self.config)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });

        match result {
            Ok(model) => Ok(model),
            Err(e) => Err(anyhow::anyhow!("Failed to load Nomic model: {e}")),
        }
    }

    /// Check if model is currently loaded (for diagnostics)
    pub fn is_model_loaded(&self) -> bool {
        self.lazy_model.get().is_some()
    }

    /// Create simplified embedder as fallback when model files are missing
    ///
    /// Uses hash-based embeddings that are fast but less semantic.
    /// Suitable for edge devices without enough RAM for ONNX.
    pub fn new_simplified(config: NomicConfig) -> Result<Self> {
        tracing::warn!(
            "Using SIMPLIFIED Nomic embeddings (hash-based). Semantic search will be limited."
        );
        tracing::warn!(
            "    To enable full semantic search, ensure Nomic-embed-text-v1.5 model files exist at:"
        );
        tracing::warn!("    Model: {:?}", config.model_path);
        tracing::warn!("    Tokenizer: {:?}", config.tokenizer_path);

        Ok(Self {
            config,
            lazy_model: OnceLock::new(),
            simplified_mode: true,
            dimension: 768,
        })
    }

    /// Encode text as a document embedding (prepends "search_document: " prefix)
    ///
    /// Use this for content being stored/indexed.
    pub fn encode_document(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }
        let prefixed = format!("{SEARCH_DOCUMENT_PREFIX}{text}");
        self.encode_raw(&prefixed)
    }

    /// Encode text as a query embedding (prepends "search_query: " prefix)
    ///
    /// Use this for search queries at retrieval time.
    pub fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }
        let prefixed = format!("{SEARCH_QUERY_PREFIX}{text}");
        self.encode_raw(&prefixed)
    }

    /// Batch encode documents (prepends "search_document: " prefix to each)
    pub fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{SEARCH_DOCUMENT_PREFIX}{t}"))
            .collect();
        let prefixed_refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.encode_batch_raw(&prefixed_refs)
    }

    /// Batch encode queries (prepends "search_query: " prefix to each)
    pub fn encode_queries(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{SEARCH_QUERY_PREFIX}{t}"))
            .collect();
        let prefixed_refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.encode_batch_raw(&prefixed_refs)
    }

    /// Raw encode without any prefix (text must already include prefix if needed)
    fn encode_raw(&self, text: &str) -> Result<Vec<f32>> {
        if self.simplified_mode {
            let start = std::time::Instant::now();
            let result = self.generate_embedding_simplified(text);
            let duration = start.elapsed().as_secs_f64();

            if result.is_ok() {
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["nomic_simplified"])
                    .observe(duration);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["nomic_simplified", "success"])
                    .inc();
            } else {
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["nomic_simplified", "failure"])
                    .inc();
            }

            return result;
        }

        let start = std::time::Instant::now();

        match self.generate_embedding_onnx(text) {
            Ok(embedding) => {
                let duration = start.elapsed().as_secs_f64();
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["nomic_onnx"])
                    .observe(duration);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["nomic_onnx", "success"])
                    .inc();

                if duration * 1000.0 > self.config.embed_timeout_ms as f64 {
                    tracing::warn!(
                        "Nomic ONNX inference took {:.0}ms (threshold: {}ms)",
                        duration * 1000.0,
                        self.config.embed_timeout_ms
                    );
                }

                Ok(embedding)
            }
            Err(e) => {
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["nomic_onnx", "failure"])
                    .inc();
                tracing::warn!(
                    "Nomic ONNX inference failed: {}. Falling back to simplified.",
                    e
                );
                self.generate_embedding_simplified(text)
            }
        }
    }

    /// Batch encode without prefix manipulation (texts must already include prefixes)
    fn encode_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let empty_embedding = vec![0.0; self.dimension];
        if texts.iter().all(|t| t.is_empty()) {
            return Ok(vec![empty_embedding; texts.len()]);
        }

        if self.simplified_mode {
            let start = std::time::Instant::now();
            let results: Result<Vec<_>> = texts
                .iter()
                .map(|text| {
                    if text.is_empty() {
                        Ok(vec![0.0; self.dimension])
                    } else {
                        self.generate_embedding_simplified(text)
                    }
                })
                .collect();
            let duration = start.elapsed().as_secs_f64();

            crate::metrics::EMBEDDING_GENERATE_DURATION
                .with_label_values(&["nomic_simplified_batch"])
                .observe(duration);
            crate::metrics::EMBEDDING_GENERATE_TOTAL
                .with_label_values(&[
                    "nomic_simplified_batch",
                    if results.is_ok() {
                        "success"
                    } else {
                        "failure"
                    },
                ])
                .inc();

            return results;
        }

        let start = std::time::Instant::now();

        // Filter out empty strings and track their positions
        let (non_empty_texts, empty_indices): (Vec<_>, Vec<_>) =
            texts.iter().enumerate().partition(|(_, t)| !t.is_empty());

        let non_empty_texts: Vec<&str> = non_empty_texts.into_iter().map(|(_, t)| *t).collect();
        let empty_indices: Vec<usize> = empty_indices.into_iter().map(|(i, _)| i).collect();

        match self.generate_embeddings_batch_onnx(&non_empty_texts) {
            Ok(embeddings) => {
                let duration = start.elapsed().as_secs_f64();
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["nomic_onnx_batch"])
                    .observe(duration);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["nomic_onnx_batch", "success"])
                    .inc();

                // Reconstruct results with empty embeddings in correct positions
                let mut results = Vec::with_capacity(texts.len());
                let mut embedding_iter = embeddings.into_iter();

                for i in 0..texts.len() {
                    if empty_indices.contains(&i) {
                        results.push(vec![0.0; self.dimension]);
                    } else {
                        results.push(
                            embedding_iter
                                .next()
                                .unwrap_or_else(|| vec![0.0; self.dimension]),
                        );
                    }
                }

                Ok(results)
            }
            Err(e) => {
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["nomic_onnx_batch", "failure"])
                    .inc();
                tracing::warn!(
                    "Nomic batch ONNX inference failed: {}. Falling back to sequential simplified.",
                    e
                );

                texts
                    .iter()
                    .map(|text| {
                        if text.is_empty() {
                            Ok(vec![0.0; self.dimension])
                        } else {
                            self.generate_embedding_simplified(text)
                        }
                    })
                    .collect()
            }
        }
    }

    /// L2 normalize embedding
    /// Returns false if normalization failed (zero norm or NaN detected)
    fn normalize(embedding: &mut [f32]) -> bool {
        // Check for NaN values before normalization
        if embedding.iter().any(|x| x.is_nan() || x.is_infinite()) {
            for val in embedding.iter_mut() {
                if val.is_nan() || val.is_infinite() {
                    *val = 0.0;
                }
            }
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm.is_nan() || norm < f32::EPSILON {
            return false;
        }

        for val in embedding.iter_mut() {
            *val /= norm;
        }

        true
    }

    /// Generate embedding using simplified approach
    fn generate_embedding_simplified(&self, text: &str) -> Result<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0; self.dimension];

        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();

            for j in 0..self.dimension {
                let index = (i.wrapping_mul(7) + j) % self.dimension;
                if j < 64 {
                    embedding[index] += ((hash >> j) & 1) as f32 * 0.1;
                } else {
                    embedding[index] += ((hash >> (j % 64)) & 1) as f32 * 0.1;
                }
            }
        }

        // Add character bigram features
        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len().saturating_sub(1) {
            let mut hasher = DefaultHasher::new();
            let bigram = format!("{}{}", chars[i], chars[i + 1]);
            bigram.hash(&mut hasher);
            let hash = hasher.finish();

            for j in 0..32 {
                let index = ((hash as usize) + j) % self.dimension;
                embedding[index] += ((hash >> (j % 64)) & 1) as f32 * 0.05;
            }
        }

        if !Self::normalize(&mut embedding) {
            tracing::warn!(
                "Nomic embedding normalization failed (zero norm or NaN), returning zero vector"
            );
            embedding.iter_mut().for_each(|v| *v = 0.0);
        }

        Ok(embedding)
    }

    /// Generate embedding using ONNX Runtime
    ///
    /// Lazily loads the model on first call if not already loaded.
    fn generate_embedding_onnx(&self, text: &str) -> Result<Vec<f32>> {
        tracing::debug!("Nomic ONNX: ensuring model loaded...");
        let model = self.ensure_model_loaded()?;
        tracing::debug!("Nomic ONNX: model ready, acquiring session lock...");

        let lock_timeout = std::time::Duration::from_secs(30);
        let mut session = model.session.try_lock_for(lock_timeout).ok_or_else(|| {
            tracing::error!(
                "Nomic ONNX session lock acquisition timed out after {}s",
                lock_timeout.as_secs()
            );
            anyhow::anyhow!(
                "Nomic ONNX session lock timeout ({}s)",
                lock_timeout.as_secs()
            )
        })?;
        tracing::debug!("Nomic ONNX: session lock acquired, tokenizing...");

        let encoding = model
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Nomic tokenization failed: {e}"))?;

        let tokens = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let max_length = self.config.max_length;
        tracing::debug!("Nomic ONNX: tokenized {} tokens", tokens.len());

        // Truncate or pad to max_length
        let mut input_ids = vec![0i64; max_length];
        let mut attention = vec![0i64; max_length];
        let token_type_ids = vec![0i64; max_length];

        for (i, &token) in tokens.iter().take(max_length).enumerate() {
            input_ids[i] = token as i64;
        }
        for (i, &mask) in attention_mask.iter().take(max_length).enumerate() {
            attention[i] = mask as i64;
        }

        // Create input tensors
        let input_ids_value = Value::from_array((vec![1, max_length], input_ids))?;
        let attention_mask_value = Value::from_array((vec![1, max_length], attention.clone()))?;
        let token_type_ids_value = Value::from_array((vec![1, max_length], token_type_ids))?;

        // Run inference
        tracing::debug!("Nomic ONNX: running inference...");
        let outputs = session.run(ort::inputs![
            "input_ids" => &input_ids_value,
            "attention_mask" => &attention_mask_value,
            "token_type_ids" => &token_type_ids_value,
        ])?;
        tracing::debug!("Nomic ONNX: inference complete");

        // Extract embeddings
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, output_data) = output_tensor;

        // Mean pooling over sequence dimension
        let mut pooled = vec![0.0; self.dimension];
        let mut mask_sum = 0.0;

        for (seq_idx, &att) in attention.iter().enumerate() {
            if att == 1 {
                for (dim_idx, pooled_val) in pooled.iter_mut().enumerate() {
                    let idx = seq_idx * self.dimension + dim_idx;
                    *pooled_val += output_data[idx];
                }
                mask_sum += 1.0;
            }
        }

        if mask_sum > 0.0 {
            for val in &mut pooled {
                *val /= mask_sum;
            }
        }

        // Handle NaN/Inf values
        for val in pooled.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }

        // L2 normalize
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON && !norm.is_nan() {
            for val in &mut pooled {
                *val /= norm;
            }
        }

        Ok(pooled)
    }

    /// Generate embeddings for multiple texts in a single ONNX batch
    fn generate_embeddings_batch_onnx(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let model = self.ensure_model_loaded()?;
        let lock_timeout = std::time::Duration::from_secs(30);
        let mut session = model.session.try_lock_for(lock_timeout).ok_or_else(|| {
            tracing::error!(
                "Nomic ONNX session lock timed out after {}s in batch embed",
                lock_timeout.as_secs()
            );
            anyhow::anyhow!(
                "Nomic ONNX session lock timeout ({}s) in batch embed",
                lock_timeout.as_secs()
            )
        })?;

        let batch_size = texts.len();
        let max_length = self.config.max_length;

        // Tokenize all texts
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                model
                    .tokenizer
                    .encode(*text, true)
                    .map_err(|e| anyhow::anyhow!("Nomic tokenization failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;

        // Prepare batched tensors
        let total_elements = batch_size * max_length;
        let mut input_ids = vec![0i64; total_elements];
        let mut attention_masks = vec![0i64; total_elements];
        let token_type_ids = vec![0i64; total_elements];

        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let tokens = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let offset = batch_idx * max_length;

            for (i, &token) in tokens.iter().take(max_length).enumerate() {
                input_ids[offset + i] = token as i64;
            }
            for (i, &mask) in attention_mask.iter().take(max_length).enumerate() {
                attention_masks[offset + i] = mask as i64;
            }
        }

        // Create batched input tensors
        let input_ids_value = Value::from_array((vec![batch_size, max_length], input_ids))?;
        let attention_mask_value =
            Value::from_array((vec![batch_size, max_length], attention_masks.clone()))?;
        let token_type_ids_value =
            Value::from_array((vec![batch_size, max_length], token_type_ids))?;

        // Run batch inference
        let outputs = session.run(ort::inputs![
            "input_ids" => &input_ids_value,
            "attention_mask" => &attention_mask_value,
            "token_type_ids" => &token_type_ids_value,
        ])?;

        // Extract embeddings - output shape is [batch_size, seq_length, hidden_size]
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, output_data) = output_tensor;

        // Mean pooling for each item in batch
        let mut results = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let mut pooled = vec![0.0; self.dimension];
            let mut mask_sum = 0.0;

            let batch_offset = batch_idx * max_length * self.dimension;
            let attention_offset = batch_idx * max_length;

            for seq_idx in 0..max_length {
                if attention_masks[attention_offset + seq_idx] == 1 {
                    for (dim_idx, pooled_val) in pooled.iter_mut().enumerate().take(self.dimension)
                    {
                        let idx = batch_offset + seq_idx * self.dimension + dim_idx;
                        *pooled_val += output_data[idx];
                    }
                    mask_sum += 1.0;
                }
            }

            if mask_sum > 0.0 {
                for val in &mut pooled {
                    *val /= mask_sum;
                }
            }

            for val in pooled.iter_mut() {
                if val.is_nan() || val.is_infinite() {
                    *val = 0.0;
                }
            }

            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON && !norm.is_nan() {
                for val in &mut pooled {
                    *val /= norm;
                }
            }

            results.push(pooled);
        }

        Ok(results)
    }
}

impl Embedder for NomicEmbedder {
    /// Default encode uses "search_document: " prefix.
    ///
    /// For query embeddings, use `encode_query()` directly.
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_document(text)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.encode_documents(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nomic_config_default() {
        let config = NomicConfig::default();
        assert_eq!(config.max_length, 512);
        assert!(config.use_quantized);
        assert_eq!(config.embed_timeout_ms, 5000);
    }

    #[test]
    fn test_nomic_config_with_paths() {
        let config = NomicConfig::with_paths(
            PathBuf::from("/tmp/model.onnx"),
            PathBuf::from("/tmp/tokenizer.json"),
        );
        assert_eq!(config.model_path, PathBuf::from("/tmp/model.onnx"));
        assert_eq!(config.tokenizer_path, PathBuf::from("/tmp/tokenizer.json"));
        assert_eq!(config.max_length, 512);
    }

    #[test]
    fn test_nomic_dimension() {
        let config = NomicConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 512,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = NomicEmbedder::new_simplified(config).unwrap();
        assert_eq!(embedder.dimension(), 768);
    }

    #[test]
    fn test_embedding_generation_simplified() {
        let config = NomicConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 512,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = NomicEmbedder::new_simplified(config).unwrap();

        let text = "Hello world";
        let embedding = embedder.encode(text).unwrap();

        assert_eq!(embedding.len(), 768);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");
    }

    #[test]
    fn test_query_vs_document_simplified() {
        let config = NomicConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 512,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = NomicEmbedder::new_simplified(config).unwrap();

        let doc_embedding = embedder.encode_document("test text").unwrap();
        let query_embedding = embedder.encode_query("test text").unwrap();

        assert_eq!(doc_embedding.len(), 768);
        assert_eq!(query_embedding.len(), 768);

        // Document and query embeddings should differ (different prefixes)
        assert_ne!(doc_embedding, query_embedding);
    }

    #[test]
    fn test_batch_encoding_simplified() {
        let config = NomicConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 512,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = NomicEmbedder::new_simplified(config).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let embeddings = embedder.encode_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.len(), 768);
        }
    }

    #[test]
    fn test_empty_text_handling() {
        let config = NomicConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 512,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = NomicEmbedder::new_simplified(config).unwrap();

        let embedding = embedder.encode("").unwrap();
        assert_eq!(embedding.len(), 768);
        assert!(embedding.iter().all(|&v| v == 0.0));

        let query_embedding = embedder.encode_query("").unwrap();
        assert_eq!(query_embedding.len(), 768);
        assert!(query_embedding.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize() {
        let mut embedding = vec![3.0, 4.0];
        assert!(NomicEmbedder::normalize(&mut embedding));
        assert!((embedding[0] - 0.6).abs() < 1e-5);
        assert!((embedding[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut embedding = vec![0.0; 768];
        assert!(!NomicEmbedder::normalize(&mut embedding));
    }

    #[test]
    fn test_normalize_nan() {
        let mut embedding = vec![f32::NAN, 1.0, f32::INFINITY];
        // NaN and Inf should be replaced with 0.0 before normalization
        NomicEmbedder::normalize(&mut embedding);
        assert!(!embedding.iter().any(|x| x.is_nan() || x.is_infinite()));
    }
}
