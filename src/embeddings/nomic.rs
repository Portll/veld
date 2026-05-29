//! Nomic-embed-text-v1.5 embedding model using ONNX Runtime
//!
//! Generates sentence embeddings with Matryoshka representation learning (MRL).
//! Model: nomic-ai/nomic-embed-text-v1.5
//!
//! # Matryoshka dimensions
//! The model is MRL-trained at a native width of 768. Because information is
//! front-loaded, any prefix of length {64, 128, 256, 512, 768} — re-normalized —
//! is a usable embedding. The exposed dimension is set via `VELD_NOMIC_DIM`
//! (default 768). 512 retains ~99% of 768's quality at 2/3 the storage and a
//! faster search; 256 retains ~97-98%. Truncating *down* is free (one model,
//! one download); it is the runtime fidelity knob for the product.
//!
//! Key differences from MiniLM:
//! - 768-dimensional embeddings by default (vs 384), truncatable via MRL
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
//! - VELD_NOMIC_MODEL_PATH: Base path to model files (default: ~/.cache/veld/models/nomic-embed-v1.5)
//! - VELD_NOMIC_EMBED_TIMEOUT_MS: Embedding timeout in ms (default: 5000)
//! - VELD_NOMIC_DIM: Matryoshka output dimension — 64|128|256|512|768 (default: 768)
//! - VELD_LAZY_LOAD: Set to "false" to load model at startup (default: true)
//! - VELD_ONNX_THREADS: Number of ONNX threads (default: 1 on macOS ARM64, min(cores,8) elsewhere)
//! - VELD_NOMIC_POOL_SIZE: Number of ONNX sessions in the pool, each independently
//!   lockable. Trades memory (~150-200 MB per session) for concurrent throughput.
//!   Default: 2 on macOS ARM64, min(cores/2, 4) elsewhere. Set to 1 for low-memory
//!   deployments.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

use super::Embedder;

/// Thread-pool / spinning knobs handed to the ORT session builder.
#[derive(Clone, Copy)]
struct ThreadsConfig {
    intra: usize,
    inter: usize,
    intra_spin: bool,
    inter_spin: bool,
}

/// Per-platform safe defaults for ORT threading.
///
/// macOS aarch64 (M-series) hits an Eigen spin-to-block deadlock on heterogeneous
/// P/E cores (microsoft/onnxruntime#10270), so we keep a single non-spinning
/// thread there. Every other platform gets a wide thread pool and enables
/// spinning — leaving these off costs 3-5x on x86 CPUs that are not vulnerable
/// to the macOS bug.
fn default_threads_config() -> ThreadsConfig {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        ThreadsConfig {
            intra: 1,
            inter: 1,
            intra_spin: false,
            inter_spin: false,
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8);
        ThreadsConfig {
            intra: cores,
            inter: 1,
            intra_spin: true,
            inter_spin: true,
        }
    }
}

/// Default number of independently-lockable sessions in the pool.
///
/// Each session is ~150-200 MB resident; the pool multiplies throughput by N
/// at the cost of N× model memory. A modest pool (~cores/2 capped at 4) gives
/// good concurrency without ballooning RSS.
fn default_pool_size() -> usize {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        2
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(2))
            .unwrap_or(2)
            .min(4)
    }
}

/// Round a sequence length up to the next multiple of 32, capped at `max_length`.
/// Better SIMD/matmul tile alignment than feeding raw token counts to the model.
fn align_seq_len(n: usize, max_length: usize) -> usize {
    let aligned = (n + 31) & !31;
    aligned.clamp(32, max_length)
}

/// Build a single ORT session with the supplied threading config.
fn build_session(model_path: &Path, threads: ThreadsConfig) -> Result<Session> {
    Session::builder()
        .context("Failed to create session builder")?
        .with_intra_threads(threads.intra)
        .context("Failed to set intra thread count")?
        .with_inter_threads(threads.inter)
        .context("Failed to set inter thread count")?
        .with_intra_op_spinning(threads.intra_spin)
        .context("Failed to set intra-op spinning")?
        .with_inter_op_spinning(threads.inter_spin)
        .context("Failed to set inter-op spinning")?
        .commit_from_file(model_path)
        .context("Failed to load Nomic ONNX model")
}

/// Nomic task prefixes for asymmetric embedding
const SEARCH_DOCUMENT_PREFIX: &str = "search_document: ";
const SEARCH_QUERY_PREFIX: &str = "search_query: ";

/// Native hidden size of nomic-embed-text-v1.5.
///
/// This is the ONNX output width and the stride used for mean-pooling. It is
/// fixed by the model and is deliberately NOT the same value as the exposed
/// embedding dimension (`NomicEmbedder::dimension`): when Matryoshka truncation
/// is active the model still emits 768-wide rows which we pool, then truncate.
const NOMIC_NATIVE_DIM: usize = 768;

/// Valid Matryoshka output dimensions for nomic-embed-text-v1.5.
///
/// The model was trained with MRL, so any prefix of these lengths — re-normalized
/// — is a usable embedding. A `VELD_NOMIC_DIM` value outside this set is rejected
/// (with a warning) and the native dimension is used instead.
const NOMIC_VALID_DIMS: [usize; 5] = [64, 128, 256, 512, 768];

/// Resolve and validate the configured Matryoshka output dimension.
fn resolve_output_dim() -> usize {
    match std::env::var("VELD_NOMIC_DIM") {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(d) if NOMIC_VALID_DIMS.contains(&d) => d,
            Ok(d) => {
                tracing::warn!(
                    "VELD_NOMIC_DIM={} is not a valid Matryoshka dimension {:?}; using {}",
                    d,
                    NOMIC_VALID_DIMS,
                    NOMIC_NATIVE_DIM
                );
                NOMIC_NATIVE_DIM
            }
            Err(_) => {
                tracing::warn!(
                    "VELD_NOMIC_DIM={:?} is not a number; using {}",
                    raw,
                    NOMIC_NATIVE_DIM
                );
                NOMIC_NATIVE_DIM
            }
        },
        Err(_) => NOMIC_NATIVE_DIM,
    }
}

/// Pool of independently-lockable ONNX sessions sharing one tokenizer.
///
/// The `ort` v2 API requires `&mut Session` for `Session::run`, so concurrent
/// inference on one session is impossible. The pool sidesteps that by holding
/// N sessions, each guarded by its own mutex, and round-robins requests across
/// them. Real-world throughput scales close to linearly with pool size until
/// CPU saturates.
struct LazyNomicModel {
    /// One `Mutex<Session>` per pool slot. Pool size is fixed at construction.
    sessions: Vec<Mutex<Session>>,
    /// Wrapping counter used to pick the next pool slot for a request.
    next: AtomicUsize,
    tokenizer: Tokenizer,
}

impl LazyNomicModel {
    fn new(config: &NomicConfig) -> Result<Self> {
        let threads_env = std::env::var("VELD_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok());
        let mut threads = default_threads_config();
        if let Some(n) = threads_env {
            threads.intra = n;
        }

        let pool_size = std::env::var("VELD_NOMIC_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n: &usize| *n >= 1)
            .unwrap_or_else(default_pool_size);

        tracing::info!(
            "Loading Nomic-embed-text-v1.5 from {:?} (pool={}, intra={}, inter={}, spin_intra={}, spin_inter={})",
            config.model_path,
            pool_size,
            threads.intra,
            threads.inter,
            threads.intra_spin,
            threads.inter_spin
        );

        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load Nomic tokenizer: {e}"))?;

        let mut sessions = Vec::with_capacity(pool_size);
        for slot in 0..pool_size {
            let session = build_session(&config.model_path, threads)
                .with_context(|| format!("Failed to build Nomic session #{slot}"))?;
            sessions.push(Mutex::new(session));
        }

        tracing::info!(
            "Nomic-embed-text-v1.5 model loaded successfully ({} sessions)",
            pool_size
        );

        Ok(Self {
            sessions,
            next: AtomicUsize::new(0),
            tokenizer,
        })
    }

    /// Pick the next pool slot. Round-robin via a wrapping counter — cheaper than
    /// scanning for the least-contended slot and good enough for uniform loads.
    fn acquire(&self) -> &Mutex<Session> {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        &self.sessions[i]
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

    /// Matryoshka output dimension — one of {64, 128, 256, 512, 768}.
    /// The model always emits 768; embeddings are truncated to this length and
    /// re-normalized. Defaults to `NOMIC_NATIVE_DIM` (768, no truncation).
    pub output_dim: usize,
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
    /// 1. VELD_NOMIC_MODEL_PATH environment variable
    /// 2. Bundled in Python package (VELD_PACKAGE_DIR/models/nomic-embed-v1.5)
    /// 3. ./models/nomic-embed-v1.5 (local)
    /// 4. ../models/nomic-embed-v1.5 (parent)
    /// 5. ~/.cache/veld/models/nomic-embed-v1.5 (auto-download location)
    pub fn from_env() -> Self {
        let base_path = std::env::var("VELD_NOMIC_MODEL_PATH")
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
                    dirs::data_dir().map(|p| p.join("veld/models/nomic-embed-v1.5")),
                ];

                candidates
                    .into_iter()
                    .flatten()
                    .find(|p| {
                        p.join("model_quantized.onnx").exists() || p.join("model.onnx").exists()
                    })
                    .unwrap_or_else(super::downloader::get_nomic_models_dir)
            });

        let embed_timeout_ms = std::env::var("VELD_NOMIC_EMBED_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5000);

        let use_quantized = std::env::var("VELD_NOMIC_USE_QUANTIZED")
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
            output_dim: resolve_output_dim(),
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
            output_dim: resolve_output_dim(),
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
            anyhow::bail!("ONNX Runtime not found and VELD_OFFLINE=true");
        }

        if !super::auto_download_models_enabled() {
            anyhow::bail!(
                "ONNX Runtime not found locally and VELD_AUTO_DOWNLOAD_MODELS is not enabled"
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
    /// Set VELD_LAZY_LOAD=false to load immediately.
    /// Set VELD_OFFLINE=true to disable auto-download.
    /// Set VELD_AUTO_DOWNLOAD_MODELS=true to explicitly allow downloads.
    ///
    /// Auto-download behavior:
    /// - If model files not found, downloads from HuggingFace (~65MB quantized)
    /// - If ONNX Runtime not found, downloads from GitHub (~50MB)
    /// - Files cached in ~/.cache/veld/models/nomic-embed-v1.5/
    pub fn new(config: NomicConfig) -> Result<Self> {
        let lazy_load = std::env::var("VELD_LAZY_LOAD")
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
                    "Nomic model files not found and VELD_OFFLINE=true. Using simplified embeddings.",
                );
                return Self::new_simplified(config);
            }

            if !super::auto_download_models_enabled() {
                tracing::warn!(
                    "Nomic model files not found locally and VELD_AUTO_DOWNLOAD_MODELS is not enabled. Using simplified embeddings.",
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
            dimension: config.output_dim,
            config: config.clone(),
            lazy_model: OnceLock::new(),
            simplified_mode: false,
        };

        if !lazy_load {
            tracing::info!("Eager loading Nomic ONNX model (VELD_LAZY_LOAD=false)");
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
            dimension: config.output_dim,
            config,
            lazy_model: OnceLock::new(),
            simplified_mode: true,
        })
    }

    /// Whether this embedder is running in simplified (hash-based, non-semantic)
    /// mode because ONNX Runtime or the model files were unavailable.
    ///
    /// Used by embedder selection to decide whether Nomic is fit to be the
    /// primary embedder, or whether to fall back to MiniLM's real ONNX path.
    pub fn is_simplified(&self) -> bool {
        self.simplified_mode
    }

    /// Apply Matryoshka truncation: keep the first `self.dimension` components
    /// of a native-width (768) embedding and re-normalize to unit length.
    ///
    /// A no-op when `self.dimension == NOMIC_NATIVE_DIM`. MRL guarantees the
    /// truncated-and-renormalized prefix is itself a valid embedding.
    fn apply_matryoshka(&self, mut full: Vec<f32>) -> Vec<f32> {
        if full.len() <= self.dimension {
            return full;
        }
        full.truncate(self.dimension);
        if !Self::normalize(&mut full) {
            tracing::warn!(
                "Matryoshka truncation to {}d produced a zero/NaN norm; returning zero vector",
                self.dimension
            );
            full.iter_mut().for_each(|v| *v = 0.0);
        }
        full
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

        let total_start = std::time::Instant::now();

        match self.generate_embedding_onnx(text) {
            Ok((embedding, queue_ms, run_ms)) => {
                let total_duration = total_start.elapsed().as_secs_f64();
                // Total stays for back-compat dashboards; the new labels expose the
                // split that actually matters: are we waiting in line or doing work?
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["nomic_onnx"])
                    .observe(total_duration);
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["nomic_onnx_queue"])
                    .observe(queue_ms / 1000.0);
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["nomic_onnx_run"])
                    .observe(run_ms / 1000.0);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["nomic_onnx", "success"])
                    .inc();

                // Two separate warning paths so the operator can tell whether
                // throughput is bound by queue depth (add pool slots) or by
                // per-call cost (model size, sequence length, threading).
                let threshold = self.config.embed_timeout_ms as f64;
                if run_ms > threshold {
                    tracing::warn!(
                        "Nomic ONNX inference run took {:.0}ms (threshold: {:.0}ms)",
                        run_ms,
                        threshold
                    );
                }
                if queue_ms > threshold {
                    tracing::warn!(
                        "Nomic ONNX inference queued for {:.0}ms before running \
                         (threshold: {:.0}ms) — consider raising VELD_NOMIC_POOL_SIZE",
                        queue_ms,
                        threshold
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
    fn generate_embedding_onnx(&self, text: &str) -> Result<(Vec<f32>, f64, f64)> {
        tracing::debug!("Nomic ONNX: ensuring model loaded...");
        let model = self.ensure_model_loaded()?;
        tracing::debug!("Nomic ONNX: model ready, acquiring session lock...");

        // Tokenize before grabbing the lock — there's no need to hold a session
        // mutex while we're just running the tokenizer.
        let encoding = model
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Nomic tokenization failed: {e}"))?;

        let tokens = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let max_length = self.config.max_length;
        // Pad only as far as the actual input requires (rounded up for SIMD
        // alignment). A 10-token query no longer pays 512-token cost.
        let seq_len = align_seq_len(tokens.len().min(max_length), max_length);
        tracing::debug!(
            "Nomic ONNX: tokenized {} tokens, seq_len={} (max={})",
            tokens.len(),
            seq_len,
            max_length
        );

        // Truncate or pad to seq_len
        let mut input_ids = vec![0i64; seq_len];
        let mut attention = vec![0i64; seq_len];
        let token_type_ids = vec![0i64; seq_len];

        for (i, &token) in tokens.iter().take(seq_len).enumerate() {
            input_ids[i] = token as i64;
        }
        for (i, &mask) in attention_mask.iter().take(seq_len).enumerate() {
            attention[i] = mask as i64;
        }

        // Acquire a pool slot. The queue wait is measured separately from the
        // run so the operator can tell which one is hot.
        let lock_timeout = std::time::Duration::from_secs(30);
        let queue_start = std::time::Instant::now();
        let session_mutex = model.acquire();
        let mut session = session_mutex.try_lock_for(lock_timeout).ok_or_else(|| {
            tracing::error!(
                "Nomic ONNX session lock acquisition timed out after {}s",
                lock_timeout.as_secs()
            );
            anyhow::anyhow!(
                "Nomic ONNX session lock timeout ({}s)",
                lock_timeout.as_secs()
            )
        })?;
        let queue_ms = queue_start.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!(
            "Nomic ONNX: session lock acquired after {:.1}ms",
            queue_ms
        );

        // Create input tensors
        let input_ids_value = Value::from_array((vec![1, seq_len], input_ids))?;
        let attention_mask_value = Value::from_array((vec![1, seq_len], attention.clone()))?;
        let token_type_ids_value = Value::from_array((vec![1, seq_len], token_type_ids))?;

        // Run inference — this is the only span that holds the session lock.
        tracing::debug!("Nomic ONNX: running inference...");
        let run_start = std::time::Instant::now();
        let outputs = session.run(ort::inputs![
            "input_ids" => &input_ids_value,
            "attention_mask" => &attention_mask_value,
            "token_type_ids" => &token_type_ids_value,
        ])?;
        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        // Can't drop `session` here — `outputs` borrows from it. The lock
        // releases at function end. Pool size (VELD_NOMIC_POOL_SIZE) is what
        // gives concurrent throughput, not per-call drop timing.
        tracing::debug!("Nomic ONNX: inference complete in {:.1}ms", run_ms);

        // Extract embeddings
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, output_data) = output_tensor;

        // Mean pooling over sequence dimension.
        // Pooling uses NOMIC_NATIVE_DIM (768) — the model's true output width and
        // row stride — NOT self.dimension, which may be a smaller Matryoshka size.
        let mut pooled = vec![0.0; NOMIC_NATIVE_DIM];
        let mut mask_sum = 0.0;

        for (seq_idx, &att) in attention.iter().enumerate() {
            if att == 1 {
                for (dim_idx, pooled_val) in pooled.iter_mut().enumerate() {
                    let idx = seq_idx * NOMIC_NATIVE_DIM + dim_idx;
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

        // Matryoshka: truncate to the configured dimension and re-normalize.
        Ok((self.apply_matryoshka(pooled), queue_ms, run_ms))
    }

    /// Generate embeddings for multiple texts in a single ONNX batch
    fn generate_embeddings_batch_onnx(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let model = self.ensure_model_loaded()?;
        let batch_size = texts.len();
        let max_length = self.config.max_length;

        // Tokenize all texts before grabbing a session — purely CPU work, no
        // reason to serialize it behind the model lock.
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                model
                    .tokenizer
                    .encode(*text, true)
                    .map_err(|e| anyhow::anyhow!("Nomic tokenization failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;

        // The batched seq_len is the longest tokenization in the batch, aligned
        // to the next multiple of 32, capped at max_length. Empty batches and
        // short queries no longer pay full 512-token cost.
        let longest = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let seq_len = align_seq_len(longest.min(max_length), max_length);

        // Prepare batched tensors
        let total_elements = batch_size * seq_len;
        let mut input_ids = vec![0i64; total_elements];
        let mut attention_masks = vec![0i64; total_elements];
        let token_type_ids = vec![0i64; total_elements];

        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let tokens = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let offset = batch_idx * seq_len;

            for (i, &token) in tokens.iter().take(seq_len).enumerate() {
                input_ids[offset + i] = token as i64;
            }
            for (i, &mask) in attention_mask.iter().take(seq_len).enumerate() {
                attention_masks[offset + i] = mask as i64;
            }
        }

        // Acquire a pool slot for the run only.
        let lock_timeout = std::time::Duration::from_secs(30);
        let session_mutex = model.acquire();
        let mut session = session_mutex.try_lock_for(lock_timeout).ok_or_else(|| {
            tracing::error!(
                "Nomic ONNX session lock timed out after {}s in batch embed",
                lock_timeout.as_secs()
            );
            anyhow::anyhow!(
                "Nomic ONNX session lock timeout ({}s) in batch embed",
                lock_timeout.as_secs()
            )
        })?;

        // Create batched input tensors
        let input_ids_value = Value::from_array((vec![batch_size, seq_len], input_ids))?;
        let attention_mask_value =
            Value::from_array((vec![batch_size, seq_len], attention_masks.clone()))?;
        let token_type_ids_value =
            Value::from_array((vec![batch_size, seq_len], token_type_ids))?;

        // Run batch inference. `outputs` borrows from `session`, so the lock
        // stays held through pooling below; pool size (VELD_NOMIC_POOL_SIZE)
        // is what enables concurrent batches across slots.
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
            // Pool at NOMIC_NATIVE_DIM (the model's row stride), truncate after.
            let mut pooled = vec![0.0; NOMIC_NATIVE_DIM];
            let mut mask_sum = 0.0;

            let batch_offset = batch_idx * seq_len * NOMIC_NATIVE_DIM;
            let attention_offset = batch_idx * seq_len;

            for seq_idx in 0..seq_len {
                if attention_masks[attention_offset + seq_idx] == 1 {
                    for (dim_idx, pooled_val) in
                        pooled.iter_mut().enumerate().take(NOMIC_NATIVE_DIM)
                    {
                        let idx = batch_offset + seq_idx * NOMIC_NATIVE_DIM + dim_idx;
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

            // Matryoshka: truncate to the configured dimension and re-normalize.
            results.push(self.apply_matryoshka(pooled));
        }

        Ok(results)
    }
}

impl Embedder for NomicEmbedder {
    fn model_id(&self) -> &str {
        "nomic-embed-text-v1.5"
    }

    /// Default encode uses the "search_document: " prefix (content being stored).
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_document(text)
    }

    /// Query-side encode uses the "search_query: " prefix. Honoring Nomic's
    /// document/query asymmetry is worth a few MTEB points of retrieval quality.
    fn encode_for_query(&self, text: &str) -> Result<Vec<f32>> {
        // `self.encode_query` resolves to the inherent method (the trait method
        // is named `encode_for_query`), so this is not recursive.
        self.encode_query(text)
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
            output_dim: NOMIC_NATIVE_DIM,
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
            output_dim: NOMIC_NATIVE_DIM,
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
            output_dim: NOMIC_NATIVE_DIM,
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
            output_dim: NOMIC_NATIVE_DIM,
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
            output_dim: NOMIC_NATIVE_DIM,
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

    #[test]
    fn test_resolve_output_dim_validation() {
        // NOMIC_VALID_DIMS is the authoritative MRL dimension set.
        assert_eq!(NOMIC_VALID_DIMS, [64, 128, 256, 512, 768]);
        assert_eq!(NOMIC_NATIVE_DIM, 768);
    }

    #[test]
    fn test_matryoshka_truncation_and_renormalization() {
        // Build a 512d embedder and verify apply_matryoshka shrinks + renormalizes
        // a native-width (768) vector. Done via apply_matryoshka directly because
        // the ONNX path needs model files.
        for &dim in &[256usize, 512, 768] {
            let config = NomicConfig {
                model_path: PathBuf::from("dummy.onnx"),
                tokenizer_path: PathBuf::from("dummy.json"),
                max_length: 512,
                use_quantized: true,
                embed_timeout_ms: 5000,
                output_dim: dim,
            };
            let embedder = NomicEmbedder::new_simplified(config).unwrap();
            assert_eq!(embedder.dimension(), dim);

            // A non-trivial native-width vector.
            let mut full: Vec<f32> = (0..NOMIC_NATIVE_DIM).map(|i| (i as f32) * 0.013 + 1.0).collect();
            NomicEmbedder::normalize(&mut full);

            let truncated = embedder.apply_matryoshka(full.clone());
            assert_eq!(truncated.len(), dim, "truncated to configured dimension");

            // Result must be unit-normalized.
            let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "truncated embedding must be re-normalized");

            // The truncated vector is the renormalized prefix of the original.
            if dim < NOMIC_NATIVE_DIM {
                let prefix_norm: f32 =
                    full[..dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                for (t, f) in truncated.iter().zip(full[..dim].iter()) {
                    assert!((t - f / prefix_norm).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_simplified_mode_honors_output_dim() {
        // In simplified (hash) mode the embedder emits vectors directly at the
        // configured Matryoshka dimension — no truncation pass needed.
        let config = NomicConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 512,
            use_quantized: true,
            embed_timeout_ms: 5000,
            output_dim: 256,
        };
        let embedder = NomicEmbedder::new_simplified(config).unwrap();
        assert!(embedder.is_simplified());
        let emb = embedder.encode("hello matryoshka").unwrap();
        assert_eq!(emb.len(), 256);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
