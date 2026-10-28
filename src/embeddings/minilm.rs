//! MiniLM-L6-v2 embedding model using ONNX Runtime
//!
//! Generates 384-dimensional sentence embeddings optimized for semantic similarity.
//! Model: sentence-transformers/all-MiniLM-L6-v2
//!
//! Edge Optimizations:
//! - Lazy model loading: Model is only loaded on first embed call
//! - Configurable thread count for power efficiency
//! - Simplified fallback for resource-constrained devices
//!
//! Configuration via environment variables:
//! - VELD_MODEL_PATH: Base path to model files (default: ./models/minilm-l6)
//! - VELD_EMBED_TIMEOUT_MS: Embedding timeout in ms (default: 5000)
//! - VELD_LAZY_LOAD: Set to "false" to load model at startup (default: true)
//! - VELD_ONNX_THREADS: intra-op threads for every ONNX session veld builds (default: 1 on
//!   macOS ARM64, min(available cores, 8) elsewhere); see [`OrtThreading`]

use anyhow::{Context, Result};
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::value::Value;
use parking_lot::{Mutex, MutexGuard};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokenizers::Tokenizer;

use super::Embedder;

/// Threading for every ONNX session veld builds: both embedders, the cross-encoder and NER.
///
/// Intra-op threads default to `min(available_parallelism, 8)` and inter-op threads to 1,
/// because the graphs run in sequential mode and never use a second inter-op thread. macOS on
/// Apple silicon keeps one non-spinning intra-op thread: ORT's Eigen pool deadlocks there when
/// its threads move from spinning to blocking across P and E cores (microsoft/onnxruntime#10270,
/// pykeio/ort#516). Elsewhere the threads spin, since turning spinning off cost 3-5x per call on
/// x86. `VELD_ONNX_THREADS`, or the recall harness's `SHODH_ONNX_THREADS`, sets the intra-op
/// count on every platform.
///
/// The value is part of each shared model's spec, so a process that pins one thread for
/// deterministic reductions gets its own sessions rather than reusing wider ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrtThreading {
    pub intra: usize,
    pub inter: usize,
    pub spinning: bool,
}

impl OrtThreading {
    /// The platform default with the environment's intra-op pin applied.
    pub fn resolve() -> Self {
        let pinned = std::env::var("VELD_ONNX_THREADS")
            // The recall harness pins the upstream spelling.
            .or_else(|_| std::env::var("SHODH_ONNX_THREADS"))
            .ok()
            .and_then(|s| s.trim().parse().ok());
        Self::with_pin(Self::platform_default(), pinned)
    }

    fn with_pin(mut base: Self, pinned: Option<usize>) -> Self {
        if let Some(intra) = pinned {
            base.intra = intra;
        }
        base
    }

    fn platform_default() -> Self {
        static DEFAULT: OnceLock<OrtThreading> = OnceLock::new();
        *DEFAULT.get_or_init(|| {
            Self::default_for(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            )
        })
    }

    fn default_for(parallelism: usize) -> Self {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Self {
                intra: 1,
                inter: 1,
                spinning: false,
            }
        } else {
            Self {
                intra: parallelism.clamp(1, 8),
                inter: 1,
                spinning: true,
            }
        }
    }

    /// Apply to a session builder before the model is committed.
    pub(crate) fn apply(self, builder: SessionBuilder) -> Result<SessionBuilder> {
        builder
            .with_intra_threads(self.intra)
            .context("Failed to set intra thread count")?
            .with_inter_threads(self.inter)
            .context("Failed to set inter thread count")?
            .with_intra_op_spinning(self.spinning)
            .context("Failed to set intra-op spinning")?
            .with_inter_op_spinning(self.spinning)
            .context("Failed to set inter-op spinning")
    }
}

/// Default number of sessions in a model's [`SessionPool`].
///
/// A session is the model's weights plus its arena, so the pool multiplies resident memory by
/// its size. It is shared by every tenant on the model, so it bounds that model's concurrent
/// inferences for the whole process; past half the cores, more sessions only oversubscribe.
pub(crate) fn default_session_pool_size() -> usize {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        2
    } else {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(2))
            .unwrap_or(2)
            .min(4)
    }
}

/// Independently lockable sessions for one model.
///
/// `Session::run` takes `&mut self`, so a session runs one inference at a time; N sessions run N.
pub(crate) struct SessionPool {
    sessions: Vec<Mutex<Session>>,
    next: AtomicUsize,
}

impl SessionPool {
    /// `size` sessions (at least one), each from `build(slot)`.
    pub(crate) fn build(
        size: usize,
        mut build: impl FnMut(usize) -> Result<Session>,
    ) -> Result<Self> {
        let sessions = (0..size.max(1))
            .map(|slot| build(slot).map(Mutex::new))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            sessions,
            next: AtomicUsize::new(0),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.sessions.len()
    }

    /// A free session if one is free, else the next in turn once it frees within `timeout`.
    pub(crate) fn acquire(&self, timeout: Duration) -> Option<MutexGuard<'_, Session>> {
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        let n = self.sessions.len();
        (0..n)
            .find_map(|i| self.sessions[(start + i) % n].try_lock())
            .or_else(|| self.sessions[start % n].try_lock_for(timeout))
    }
}

/// Thread-safe guard for ORT_DYLIB_PATH initialization.
/// Using OnceLock ensures set_var is called exactly once.
static ORT_PATH_INIT: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Pre-initialize the ONNX Runtime path before any async work begins.
///
/// # Safety
/// This function calls `std::env::set_var` which is unsound in multi-threaded
/// contexts (Rust 1.66+). It MUST be called before `tokio::main` spawns worker
/// threads — i.e., very early in `async fn main()` before any `.await` or
/// `tokio::spawn` calls. The OnceLock ensures it only runs once.
pub fn pre_init_ort_runtime(offline_mode: bool) {
    let _ = ORT_PATH_INIT.get_or_init(|| MiniLMEmbedder::init_ort_path_inner(offline_mode));
}

/// Locate (or, when allowed, download) the shared ONNX Runtime before any `ort` call.
pub(crate) fn ensure_ort_runtime(offline_mode: bool) -> Result<()> {
    MiniLMEmbedder::ensure_onnx_runtime_available(offline_mode)
}

/// `VELD_LAZY_LOAD=false` loads a model when its embedder is built instead of on first use.
pub(crate) fn lazy_load_enabled() -> bool {
    std::env::var("VELD_LAZY_LOAD")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true)
}

/// Inference time past which a single embed is logged as slow (`VELD_EMBED_TIMEOUT_MS`).
pub(crate) fn embed_timeout_ms() -> u64 {
    std::env::var("VELD_EMBED_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000)
}

/// Lazily initialized ONNX sessions and tokenizer
struct LazyModel {
    sessions: SessionPool,
    tokenizer: Tokenizer,
}

impl LazyModel {
    fn new(config: &EmbeddingConfig, threading: OrtThreading, pool_size: usize) -> Result<Self> {
        tracing::info!(
            "Loading MiniLM-L6-v2 model from {:?} (pool={}, intra={}, inter={}, spinning={})",
            config.model_path,
            pool_size,
            threading.intra,
            threading.inter,
            threading.spinning
        );

        let sessions = SessionPool::build(pool_size, |slot| {
            threading
                .apply(Session::builder().context("Failed to create session builder")?)?
                .commit_from_file(&config.model_path)
                .with_context(|| format!("Failed to load ONNX model (session #{slot})"))
        })?;

        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // all-MiniLM-L6-v2 ships tokenizer.json with `truncation: {max_length: 128}`. Left in
        // place it becomes the REAL ceiling — below `config.max_length` of 256, so `max_length`
        // never binds and the `.take(max_length)` below is dead code. Measured 2026-08-27 over 300
        // memories: the chunker sizes chunks for ~200 tokens, so 90.2% of chunks exceeded 128 and
        // 82,300 tokens were dropped, silently, from 86% of records — by the very component built
        // to stop content being silently dropped.
        //
        // veld owns its own window. A downloaded JSON does not get to impose a different one
        // behind it, and the difference between "the ceiling I configured" and "the ceiling that
        // applies" is not something a caller can see.
        let shipped = tokenizer.get_truncation().map(|t| t.max_length);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to clear tokenizer truncation: {e}"))?;
        if let Some(limit) = shipped {
            tracing::info!(
                shipped_truncation = limit,
                window = config.max_length,
                "cleared tokenizer.json's own truncation; the configured window is now the only one"
            );
        }

        tracing::info!(
            "MiniLM-L6-v2 model loaded successfully ({} sessions)",
            sessions.len()
        );

        Ok(Self {
            sessions,
            tokenizer,
        })
    }
}

/// Configuration for MiniLM embedder
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmbeddingConfig {
    /// Path to ONNX model file
    pub model_path: PathBuf,

    /// Path to tokenizer file
    pub tokenizer_path: PathBuf,

    /// Maximum sequence length (MiniLM default: 256)
    pub max_length: usize,

    /// Use quantized model for faster inference
    pub use_quantized: bool,

    /// Timeout for embedding generation in milliseconds
    pub embed_timeout_ms: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl EmbeddingConfig {
    /// Create configuration from environment variables with sensible defaults
    ///
    /// Search order for model files:
    /// 1. VELD_MODEL_PATH environment variable
    /// 2. Bundled in Python package (VELD_PACKAGE_DIR/models/minilm-l6)
    /// 3. ./models/minilm-l6 (local)
    /// 4. ../models/minilm-l6 (parent)
    /// 5. ~/.cache/veld/models/minilm-l6 (auto-download location)
    pub fn from_env() -> Self {
        let base_path = std::env::var("VELD_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                // Try common locations in order (bundled first for 1-click install)
                let candidates = vec![
                    // Bundled in Python package (highest priority for pip install)
                    std::env::var("VELD_PACKAGE_DIR")
                        .ok()
                        .map(|p| PathBuf::from(p).join("models/minilm-l6")),
                    Some(PathBuf::from("./models/minilm-l6")),
                    Some(PathBuf::from("../models/minilm-l6")),
                    // Auto-download cache location
                    Some(super::downloader::get_models_dir()),
                    dirs::data_dir().map(|p| p.join("veld/models/minilm-l6")),
                ];

                candidates
                    .into_iter()
                    .flatten()
                    .find(|p| {
                        p.join("model_quantized.onnx").exists() || p.join("model.onnx").exists()
                    })
                    .unwrap_or_else(super::downloader::get_models_dir) // Default to cache dir
            });

        let embed_timeout_ms = embed_timeout_ms();

        let use_quantized = std::env::var("VELD_USE_QUANTIZED_MODEL")
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
            max_length: 256,
            use_quantized,
            embed_timeout_ms,
        }
    }

    /// Create configuration with explicit paths (for testing or programmatic use)
    pub fn with_paths(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            model_path,
            tokenizer_path,
            max_length: 256,
            use_quantized: true,
            embed_timeout_ms: 5000,
        }
    }
}

/// MiniLM-L6-v2 embedder with ONNX Runtime
///
/// Features lazy model loading for edge devices:
/// - Model is only loaded on first embed() call
/// - Reduces startup time from ~2s to <100ms
/// - Reduces idle RAM by ~200MB until first use
pub struct MiniLMEmbedder {
    config: EmbeddingConfig,
    threading: OrtThreading,
    pool_size: usize,
    /// Lazily initialized model (OnceLock for thread-safe init)
    lazy_model: OnceLock<Result<Arc<LazyModel>, String>>,
    /// Flag for simplified mode (no ONNX)
    simplified_mode: bool,
    dimension: usize,
}

impl MiniLMEmbedder {
    /// Ensure ONNX Runtime is available before any ort code runs.
    /// This MUST be called before creating any ONNX sessions.
    /// Sets ORT_DYLIB_PATH if needed from cache or download.
    ///
    /// SAFETY: Uses OnceLock to ensure set_var is called at most once,
    /// mitigating the thread-safety issue with std::env::set_var.
    fn ensure_onnx_runtime_available(offline_mode: bool) -> Result<()> {
        // Use OnceLock to ensure we only initialize once (thread-safe)
        let result = ORT_PATH_INIT.get_or_init(|| Self::init_ort_path_inner(offline_mode));

        match result {
            Ok(_) => Ok(()),
            Err(e) => anyhow::bail!("{e}"),
        }
    }

    /// Inner initialization logic - called exactly once via OnceLock.
    /// SAFETY: set_var is only called once due to OnceLock guard.
    fn init_ort_path_inner(offline_mode: bool) -> Result<PathBuf, String> {
        // If ORT_DYLIB_PATH is already set to a valid path, we're good
        if let Ok(existing_path) = std::env::var("ORT_DYLIB_PATH") {
            let path = std::path::PathBuf::from(&existing_path);
            if path.exists() {
                tracing::debug!(
                    "Using existing ONNX Runtime from ORT_DYLIB_PATH: {:?}",
                    path
                );
                return Ok(path);
            }
        }

        // Check for bundled ONNX Runtime in Python package (1-click install)
        if let Some(bundled_path) = Self::find_bundled_onnx_runtime() {
            tracing::info!(
                "Using bundled ONNX Runtime from package: {:?}",
                bundled_path
            );
            // SAFETY: This is called once via OnceLock, before other threads start
            std::env::set_var("ORT_DYLIB_PATH", &bundled_path);
            return Ok(bundled_path);
        }

        // Check if we have ONNX Runtime in our cache
        if let Some(cached_path) = super::downloader::get_onnx_runtime_path() {
            tracing::info!(
                "Setting ORT_DYLIB_PATH to cached runtime: {:?}",
                cached_path
            );
            // SAFETY: This is called once via OnceLock, before other threads start
            std::env::set_var("ORT_DYLIB_PATH", &cached_path);
            return Ok(cached_path);
        }

        // Need to download ONNX Runtime
        if offline_mode {
            return Err("ONNX Runtime not found and VELD_OFFLINE=true".to_string());
        }

        if !super::auto_download_models_enabled() {
            return Err(
                "ONNX Runtime not found locally and VELD_AUTO_DOWNLOAD_MODELS is not enabled"
                    .to_string(),
            );
        }

        tracing::info!("ONNX Runtime not found locally. Downloading...");
        let onnx_path =
            super::downloader::download_onnx_runtime(None).map_err(|e| e.to_string())?;
        tracing::info!(
            "Setting ORT_DYLIB_PATH to downloaded runtime: {:?}",
            onnx_path
        );
        // SAFETY: This is called once via OnceLock, before other threads start
        std::env::set_var("ORT_DYLIB_PATH", &onnx_path);
        Ok(onnx_path)
    }

    /// Find bundled ONNX Runtime in the Python package's lib/ directory
    fn find_bundled_onnx_runtime() -> Option<PathBuf> {
        // Try to find ONNX Runtime bundled with the Python package
        // The lib/ folder is adjacent to the veld.pyd file

        #[cfg(target_os = "windows")]
        let dll_name = "onnxruntime.dll";
        #[cfg(target_os = "macos")]
        let dll_name = "libonnxruntime.dylib";
        #[cfg(target_os = "linux")]
        let dll_name = "libonnxruntime.so";

        // Common locations to check for bundled library
        let candidates = [
            // Adjacent to current executable (common deployment layout)
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.join(dll_name))),
            // In lib/ subdir relative to executable
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.join("lib").join(dll_name))),
            // Relative to working directory
            Some(PathBuf::from("lib").join(dll_name)),
            // In working directory directly
            Some(PathBuf::from(dll_name)),
            // Python site-packages layout: veld/lib/onnxruntime.dll
            dirs::data_dir().map(|p| {
                p.join("Python")
                    .join("site-packages")
                    .join("veld")
                    .join("lib")
                    .join(dll_name)
            }),
            // Check relative to this module (for pip-installed packages)
            // This uses the fact that Python modules are in site-packages/veld/
            std::env::var("VELD_PACKAGE_DIR")
                .ok()
                .map(|p| PathBuf::from(p).join("lib").join(dll_name)),
        ];

        for candidate in candidates.into_iter().flatten() {
            if candidate.exists() {
                tracing::debug!("Found bundled ONNX Runtime at: {:?}", candidate);
                return Some(candidate);
            }
        }

        None
    }

    /// Create new MiniLM embedder with lazy loading (default)
    ///
    /// Model is NOT loaded until first embed() call.
    /// Set VELD_LAZY_LOAD=false to load immediately.
    /// Set VELD_OFFLINE=true to disable auto-download.
    /// Set VELD_AUTO_DOWNLOAD_MODELS=true to explicitly allow downloads.
    ///
    /// Auto-download behavior:
    /// - If model files not found, downloads from HuggingFace (~22MB)
    /// - If ONNX Runtime not found, downloads from GitHub (~50MB)
    /// - Files cached in ~/.cache/veld/
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        Self::with_runtime(config, OrtThreading::resolve(), default_session_pool_size())
    }

    /// [`Self::new`] with the session threading and pool size given rather than resolved, so a
    /// shared instance is built from exactly the values its spec was keyed on.
    pub fn with_runtime(
        config: EmbeddingConfig,
        threading: OrtThreading,
        pool_size: usize,
    ) -> Result<Self> {
        let lazy_load = lazy_load_enabled();

        let offline_mode = super::offline_mode_enabled();

        // CRITICAL: Ensure ORT_DYLIB_PATH is set BEFORE any ort code runs
        // This prevents ort from picking up system DLLs with wrong versions
        if let Err(e) = Self::ensure_onnx_runtime_available(offline_mode) {
            return Self::simplified_or_fail(config, &format!("ONNX Runtime setup failed: {e}"));
        }

        // Check if model files exist
        let model_available = config.model_path.exists() && config.tokenizer_path.exists();

        if !model_available {
            if offline_mode {
                return Self::simplified_or_fail(
                    config,
                    "model files not found and VELD_OFFLINE=true",
                );
            }

            if !super::auto_download_models_enabled() {
                return Self::simplified_or_fail(
                    config,
                    "model files not found locally and VELD_AUTO_DOWNLOAD_MODELS is not enabled",
                );
            }

            // Try to auto-download model files
            tracing::info!(
                "Model files not found locally at {:?}. Downloading...",
                config.model_path.parent().unwrap_or(&config.model_path)
            );

            match super::downloader::download_models(Some(std::sync::Arc::new(
                |downloaded, total| {
                    if total > 0 {
                        let percent = (downloaded as f64 / total as f64 * 100.0) as u32;
                        if percent.is_multiple_of(10) {
                            tracing::info!(
                                "Downloading models: {}% ({}/{})",
                                percent,
                                downloaded,
                                total
                            );
                        }
                    }
                },
            ))) {
                Ok(models_dir) => {
                    tracing::info!("Models downloaded to {:?}", models_dir);

                    // Update config with downloaded paths
                    let model_filename = if config.use_quantized {
                        "model_quantized.onnx"
                    } else {
                        "model.onnx"
                    };
                    let updated_config = EmbeddingConfig {
                        model_path: models_dir.join(model_filename),
                        tokenizer_path: models_dir.join("tokenizer.json"),
                        ..config
                    };

                    // Recursively create with updated config (ORT_DYLIB_PATH already set)
                    return Self::with_runtime(updated_config, threading, pool_size);
                }
                Err(e) => {
                    return Self::simplified_or_fail(
                        config,
                        &format!("model download failed: {e}"),
                    );
                }
            }
        }

        super::model_tape::register_model(
            super::model_tape::Model::Embedder,
            &[&config.model_path, &config.tokenizer_path],
        );
        let embedder = Self {
            config: config.clone(),
            threading,
            pool_size,
            lazy_model: OnceLock::new(),
            simplified_mode: false,
            dimension: 384,
        };

        // If not lazy loading, initialize now
        if !lazy_load {
            tracing::info!("Eager loading ONNX model (VELD_LAZY_LOAD=false)");
            embedder.ensure_model_loaded()?;
        } else {
            tracing::info!("Lazy loading enabled - model will load on first embed()");
        }

        Ok(embedder)
    }

    /// Ensure the model is loaded (thread-safe, idempotent)
    fn ensure_model_loaded(&self) -> Result<&Arc<LazyModel>> {
        let result = self.lazy_model.get_or_init(|| {
            LazyModel::new(&self.config, self.threading, self.pool_size)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });

        match result {
            Ok(model) => Ok(model),
            Err(e) => Err(anyhow::anyhow!("Failed to load model: {e}")),
        }
    }

    /// Check if model is currently loaded (for diagnostics)
    pub fn is_model_loaded(&self) -> bool {
        self.lazy_model.get().is_some()
    }

    /// True once a load was attempted and failed. The failure is kept for this instance's
    /// lifetime, so a shared registry replaces such an instance rather than handing it out.
    pub fn model_load_failed(&self) -> bool {
        matches!(self.lazy_model.get(), Some(Err(_)))
    }

    /// Whether this embedder produces non-semantic hash vectors instead of running the model.
    pub fn is_simplified(&self) -> bool {
        self.simplified_mode
    }

    /// Degrade to hash embeddings, but only if someone asked for that.
    ///
    /// Every caller of this used to be a silent `return Self::new_simplified(..)`
    /// on a failure path. The result is an embedder whose `encode()` returns
    /// non-semantic vectors while nothing reports an error — so recall still
    /// answers, benchmarks still produce numbers, and all of it is measured
    /// against noise. That is not a degraded search, it is an invalidated
    /// measurement that looks like a working one.
    ///
    /// Refuse by default. `VELD_ALLOW_SIMPLIFIED_EMBEDDINGS=1` opts in, for
    /// tests and offline smoke runs where nobody is reading the quality numbers.
    fn simplified_or_fail(config: EmbeddingConfig, why: &str) -> Result<Self> {
        if super::allow_simplified_embeddings() {
            tracing::warn!(
                reason = why,
                "DEGRADED: using non-semantic hash embeddings because \
                 VELD_ALLOW_SIMPLIFIED_EMBEDDINGS is set. Any recall quality or \
                 benchmark figure from this process is meaningless.",
            );
            return Self::new_simplified(config);
        }
        anyhow::bail!(
            "refusing to run on non-semantic hash embeddings: {why}. Every recall \
             and benchmark number would be computed against noise while still \
             reporting success. Fix the model setup, or set \
             VELD_ALLOW_SIMPLIFIED_EMBEDDINGS=1 to accept a degraded embedder \
             deliberately."
        )
    }

    /// Create simplified embedder as fallback when model files are missing
    ///
    /// Uses hash-based embeddings that are fast but less semantic.
    /// Suitable for edge devices without enough RAM for ONNX.
    ///
    /// Prefer [`Self::simplified_or_fail`] on failure paths — calling this
    /// directly is how the degrade became silent in the first place. Tests that
    /// deliberately want hash mode are the legitimate direct callers.
    pub fn new_simplified(config: EmbeddingConfig) -> Result<Self> {
        tracing::warn!(
            "Using SIMPLIFIED embeddings (hash-based). Semantic search will be limited."
        );
        tracing::warn!(
            "    To enable full semantic search, ensure MiniLM-L6-v2 model files exist at:"
        );
        tracing::warn!("    Model: {:?}", config.model_path);
        tracing::warn!("    Tokenizer: {:?}", config.tokenizer_path);

        Ok(Self {
            config,
            threading: OrtThreading::resolve(),
            pool_size: 1,
            lazy_model: OnceLock::new(),
            simplified_mode: true,
            dimension: 384,
        })
    }

    /// L2 normalize embedding
    /// Returns false if normalization failed (zero norm or NaN detected)
    fn normalize(&self, embedding: &mut [f32]) -> bool {
        // Check for NaN values before normalization
        if embedding.iter().any(|x| x.is_nan() || x.is_infinite()) {
            // Replace invalid values with zero
            for val in embedding.iter_mut() {
                if val.is_nan() || val.is_infinite() {
                    *val = 0.0;
                }
            }
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Handle zero norm (all zeros) or NaN norm
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
        // Production fallback: Hash-based embeddings for resilience
        // Used when: (1) ONNX models unavailable, (2) ONNX inference fails, (3) Timeout exceeded
        // Provides basic semantic similarity via word + character n-gram hashing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0; self.dimension];

        // Use words and character n-grams for better quality
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();

            // Distribute hash bits across embedding dimensions with positional offset
            for j in 0..self.dimension {
                let index = (i.wrapping_mul(7) + j) % self.dimension;
                if j < 64 {
                    embedding[index] += ((hash >> j) & 1) as f32 * 0.1;
                } else {
                    embedding[index] += ((hash >> (j % 64)) & 1) as f32 * 0.1;
                }
            }
        }

        // Add character bigram features for better semantic representation
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

        // Normalize - if normalization fails (empty text / NaN), return zero vector
        if !self.normalize(&mut embedding) {
            tracing::warn!(
                "Embedding normalization failed (zero norm or NaN), returning zero vector"
            );
            embedding.iter_mut().for_each(|v| *v = 0.0);
        }

        Ok(embedding)
    }

    /// Generate embedding using ONNX Runtime (production)
    ///
    /// Lazily loads the model on first call if not already loaded.
    #[allow(clippy::needless_range_loop)]
    fn generate_embedding_onnx(&self, text: &str) -> Result<Vec<f32>> {
        // Lazy load model on first use
        tracing::debug!("ONNX: ensuring model loaded...");
        let model = self.ensure_model_loaded()?;
        tracing::debug!("ONNX: model ready, acquiring session lock...");

        let lock_timeout = std::time::Duration::from_secs(30);
        let mut session = model.sessions.acquire(lock_timeout).ok_or_else(|| {
            tracing::error!(
                "ONNX session lock acquisition timed out after {}s — a previous inference \
                     call is likely stuck. Falling back to simplified embeddings.",
                lock_timeout.as_secs()
            );
            anyhow::anyhow!("ONNX session lock timeout ({}s)", lock_timeout.as_secs())
        })?;
        tracing::debug!("ONNX: session lock acquired, tokenizing...");

        // Tokenize input text
        let encoding = model
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let tokens = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let max_length = self.config.max_length;
        tracing::debug!("ONNX: tokenized {} tokens", tokens.len());

        // WINDOWED MEAN POOLING — the whole input is embedded, never its first `max_length` tokens.
        //
        // This used to be `tokens.iter().take(max_length)`: everything past the window was dropped
        // with no marker, no warning and no field, so a vector computed from 12% of a document was
        // indistinguishable from one computed over all of it. Measured 2026-08-27 over 300
        // memories, the layer above lost 82,300 tokens from 86% of records that way.
        //
        // Mean pooling is a sum over attended positions divided by their count, and both halves of
        // that decompose across windows. So the loop below accumulates the numerator and the
        // denominator over every window and divides once at the end, which for a text that fits in
        // one window is arithmetically the same computation as before — short content embeds
        // BIT-IDENTICALLY, and only content that was previously being truncated changes.
        //
        // It is not equivalent to a model with a longer context: windows past the first begin
        // without a real [CLS] and their position embeddings restart, so cross-window attention
        // does not happen. It is an honest average of the whole text rather than an exact reading
        // of it, which is the difference between approximate and absent.
        let window_count = window_spans(tokens.len(), max_length).len();

        // A bound, because an unbounded loop over attacker-supplied text is a denial of service.
        // It refuses rather than truncating: past this point we cannot embed the whole input, and
        // the one thing this function must not do is silently return an embedding of part of it.
        const MAX_WINDOWS: usize = 64;
        anyhow::ensure!(
            window_count <= MAX_WINDOWS,
            "input is {} tokens ({} windows of {}), over the {} window limit. Refusing rather \
             than embedding a prefix and reporting it as the whole — chunk it upstream.",
            tokens.len(),
            window_count,
            max_length,
            MAX_WINDOWS
        );
        if window_count > 1 {
            tracing::debug!(
                tokens = tokens.len(),
                windows = window_count,
                "ONNX: input exceeds one window; pooling across all of it"
            );
        }

        let mut pooled = vec![0.0; self.dimension];
        let mut mask_sum = 0.0;

        for (window, (start, end)) in window_spans(tokens.len(), max_length)
            .into_iter()
            .enumerate()
        {
            let mut input_ids = vec![0i64; max_length];
            let mut attention = vec![0i64; max_length];
            let token_type_ids = vec![0i64; max_length];

            for (i, &token) in tokens[start..end].iter().enumerate() {
                input_ids[i] = token as i64;
            }
            for (i, &mask) in attention_mask[start..end].iter().enumerate() {
                attention[i] = mask as i64;
            }

            // Create input tensors
            let input_ids_value = Value::from_array((vec![1, max_length], input_ids))?;
            let attention_mask_value = Value::from_array((vec![1, max_length], attention.clone()))?;
            let token_type_ids_value = Value::from_array((vec![1, max_length], token_type_ids))?;

            // Run inference
            tracing::debug!("ONNX: running inference on window {}...", window);
            let outputs = session.run(ort::inputs![
                "input_ids" => &input_ids_value,
                "attention_mask" => &attention_mask_value,
                "token_type_ids" => &token_type_ids_value,
            ])?;

            // Extract embeddings
            let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
            let (_shape, output_data) = output_tensor;

            // Accumulate the mean-pooling numerator and denominator across windows.
            for (seq_idx, &att) in attention.iter().enumerate() {
                if att == 1 {
                    for (dim_idx, pooled_val) in pooled.iter_mut().enumerate() {
                        let idx = seq_idx * self.dimension + dim_idx;
                        *pooled_val += output_data[idx];
                    }
                    mask_sum += 1.0;
                }
            }
        }
        tracing::debug!("ONNX: inference complete ({} window(s))", window_count);

        // Average and L2 normalize
        if mask_sum > 0.0 {
            for val in &mut pooled {
                *val /= mask_sum;
            }
        }

        // Handle NaN/Inf values that may come from model output
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

        Ok(pooled)
    }
}

/// The model name this embedder reports, and the one a corpus marker of `minilm` maps to.
pub const MODEL_ID: &str = "minilm-l6-v2";

impl Embedder for MiniLMEmbedder {
    fn id(&self) -> &str {
        "minilm"
    }

    fn model_id(&self) -> &str {
        MODEL_ID
    }

    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }

        // A model tape answers before either path below: replay must never reach the model
        // or the simplified fallback, and recording keeps only a real model output.
        if let Some(taped) = super::model_tape::embed(text, || self.generate_embedding_onnx(text)) {
            return taped;
        }

        // Use simplified mode if in that mode
        if self.simplified_mode {
            let start = std::time::Instant::now();
            let result = self.generate_embedding_simplified(text);
            let duration = start.elapsed().as_secs_f64();

            if result.is_ok() {
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["simplified"])
                    .observe(duration);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["simplified", "success"])
                    .inc();
            } else {
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["simplified", "failure"])
                    .inc();
            }

            return result;
        }

        // Try ONNX inference (lazy loads model on first call)
        let start = std::time::Instant::now();

        match self.generate_embedding_onnx(text) {
            Ok(embedding) => {
                let duration = start.elapsed().as_secs_f64();
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["onnx"])
                    .observe(duration);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["onnx", "success"])
                    .inc();

                // Warn if inference is slow
                if duration * 1000.0 > self.config.embed_timeout_ms as f64 {
                    tracing::warn!(
                        "ONNX inference took {:.0}ms (threshold: {}ms)",
                        duration * 1000.0,
                        self.config.embed_timeout_ms
                    );
                }

                Ok(embedding)
            }
            Err(e) => {
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["onnx", "failure"])
                    .inc();
                tracing::warn!("ONNX inference failed: {}. Falling back to simplified.", e);

                // Fallback to simplified
                self.generate_embedding_simplified(text)
            }
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threading_is_the_cores_up_to_eight_with_one_inter_op_thread() {
        let wide = OrtThreading::default_for(32);
        assert_eq!(wide.inter, 1);
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert_eq!(wide.intra, 1, "the Apple-silicon deadlock guard holds");
            assert!(!wide.spinning);
        } else {
            assert_eq!(wide.intra, 8);
            assert_eq!(OrtThreading::default_for(3).intra, 3);
            assert!(wide.spinning);
        }
        assert_eq!(OrtThreading::with_pin(wide, Some(1)).intra, 1);
        assert_eq!(OrtThreading::with_pin(wide, None), wide);
    }

    #[test]
    fn a_session_pool_always_builds_at_least_one_session() {
        let mut attempts = 0;
        let built = SessionPool::build(0, |_| {
            attempts += 1;
            anyhow::bail!("no runtime in this test")
        });
        assert!(built.is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn test_minilm_creation() {
        // Test with default config
        let config = EmbeddingConfig::default();

        // Check dimension
        assert_eq!(config.max_length, 256);
    }

    #[test]
    fn test_embedding_generation_simplified() {
        // Create embedder in simplified mode (no ONNX model needed)
        let config = EmbeddingConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 256,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = MiniLMEmbedder::new_simplified(config).unwrap();

        let text = "Hello world";
        let embedding = embedder.encode(text).unwrap();

        assert_eq!(embedding.len(), 384);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");
    }

    #[test]
    fn test_batch_encoding_simplified() {
        // Create embedder in simplified mode
        let config = EmbeddingConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 256,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = MiniLMEmbedder::new_simplified(config).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let embeddings = embedder.encode_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    /// The quantized model scales its activations over a whole batch, so a batched run gave a
    /// text a different vector from the one it got alone; stored chunk vectors moved with their
    /// sibling chunks (conv-42_q84).
    #[test]
    fn a_batched_text_gets_the_vector_it_gets_alone() {
        let embedder = crate::embeddings::shared::minilm_embedder(
            &EmbeddingConfig::from_env(),
            OrtThreading::resolve(),
            default_session_pool_size(),
        )
        .expect("minilm");
        assert!(
            !embedder.is_simplified(),
            "needs the ONNX model: cargo run --bin model-fetch -- --onnxruntime --minilm"
        );
        let long = "A longer note about backups and their windows. ".repeat(60);
        let within_one_window = [
            "The tide tables were reprinted.",
            "",
            "Melanie painted a sunrise over the lake after the storm had passed.",
            "Backups ran.",
        ];
        // One over-length text sends a whole batch down another path, so it gets its own batch.
        for texts in [&within_one_window[..], &[long.as_str(), "Backups ran."][..]] {
            let batch = embedder.encode_batch(texts).unwrap();
            for (text, vector) in texts.iter().zip(&batch) {
                assert_eq!(&embedder.encode(text).unwrap(), vector, "{text:.40}");
            }
        }
    }
}

/// The windows a token sequence is embedded in, as `[start, end)` byte-index pairs.
///
/// Extracted from the inference loop so the claim it exists to make — **every token lands in
/// exactly one window** — can be checked without a model, a tokenizer or an ONNX runtime. The
/// defect this replaces was `tokens.iter().take(max_length)`, which dropped the rest in silence;
/// a fix for that should not itself rest on a loop nobody can test on a machine with no model
/// files, which is the machine this was written on.
///
/// An empty input yields one window, because the encoder still wants a padded tensor and a
/// zero-window loop would return an unnormalised zero vector by a different route.
pub(crate) fn window_spans(token_count: usize, max_length: usize) -> Vec<(usize, usize)> {
    if max_length == 0 || token_count == 0 {
        return vec![(0, 0)];
    }
    (0..token_count.div_ceil(max_length))
        .map(|w| {
            let start = w * max_length;
            (start, (start + max_length).min(token_count))
        })
        .collect()
}

#[cfg(test)]
mod window_spans_tests {
    use super::window_spans;

    /// The one property that makes "keep the whole length" true rather than aspirational: the
    /// windows must TILE the token sequence — every index in exactly one window, nothing dropped
    /// off the end, nothing counted twice into the mean.
    ///
    /// It is checked by reconstruction rather than by arithmetic about the spans, because the
    /// defect being fixed (`take(max_length)`) was itself arithmetic that looked right.
    #[test]
    fn windows_tile_the_sequence_exactly_leaving_nothing_out() {
        for max_length in [1usize, 2, 7, 128, 256, 512] {
            for token_count in [
                0usize, 1, 2, 6, 7, 8, 127, 128, 129, 255, 256, 257, 999, 4096,
            ] {
                let spans = window_spans(token_count, max_length);
                assert!(
                    !spans.is_empty(),
                    "len={token_count} w={max_length}: no windows"
                );

                let mut visited = vec![0u32; token_count];
                for (start, end) in &spans {
                    assert!(
                        end >= start && *end <= token_count,
                        "len={token_count} w={max_length}: span {start}..{end} out of range"
                    );
                    assert!(
                        end - start <= max_length,
                        "len={token_count} w={max_length}: span {start}..{end} exceeds the window"
                    );
                    for i in *start..*end {
                        visited[i] += 1;
                    }
                }

                // Both directions asserted separately: a 0 is a dropped token (the original bug)
                // and a 2 is a token double-weighted in the mean (the bug a naive overlap fix
                // would introduce). Only one of those is the direction that lies to you, so they
                // do not get to share an assertion.
                let dropped = visited.iter().filter(|&&v| v == 0).count();
                let duplicated = visited.iter().filter(|&&v| v > 1).count();
                assert_eq!(
                    dropped, 0,
                    "len={token_count} w={max_length}: {dropped} token(s) in NO window — this is \
                     the silent truncation the windowing exists to remove"
                );
                assert_eq!(
                    duplicated, 0,
                    "len={token_count} w={max_length}: {duplicated} token(s) in more than one \
                     window — they would be weighted twice in the pooled mean"
                );
            }
        }
    }

    /// A text that fits in one window must produce exactly one window, so its embedding is
    /// bit-identical to what the pre-windowing code produced. Without this the change would
    /// silently re-embed the entire existing corpus into a slightly different space.
    #[test]
    fn text_within_the_window_still_takes_exactly_one_pass() {
        for token_count in [0usize, 1, 100, 255, 256] {
            assert_eq!(
                window_spans(token_count, 256).len(),
                1,
                "{token_count} tokens should need one window, not more"
            );
        }
        assert_eq!(
            window_spans(257, 256).len(),
            2,
            "257 tokens needs a second window"
        );
    }

    /// A degenerate window must not produce an infinite or empty plan.
    #[test]
    fn a_zero_window_yields_one_span_rather_than_looping_forever() {
        assert_eq!(window_spans(100, 0), vec![(0, 0)]);
    }
}
