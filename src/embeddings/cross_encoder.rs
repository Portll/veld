//! Pluggable cross-encoder rerankers for query-document relevance scoring.
//!
//! Unlike bi-encoders that encode query and document separately, cross-encoders
//! encode the (query, document) pair jointly, allowing token-level attention
//! between them. This produces better relevance scores at the cost of higher
//! latency (no precomputed document embeddings).
//!
//! # Pluggable architecture
//!
//! [`CrossEncoderModel`] is a trait so callers can hold an
//! `Arc<dyn CrossEncoderModel>` and not care whether it's a single ONNX
//! model or an ensemble.
//!
//! [`OnnxCrossEncoder`] is the concrete ONNX-runtime-backed implementation,
//! parameterised by [`CrossEncoderConfig`]. Built-in configs:
//!
//! - `ms-marco-MiniLM-L-6-v2` — 22M params, MS MARCO web-search training (legacy default).
//! - `bge-reranker-v2-m3` — 568M params, multilingual, multi-task QA/conversational.
//! - `mxbai-rerank-large-v1` — 435M params, QA + retrieval.
//! - `jina-reranker-v2-base-multilingual` — 278M params, multilingual late-interaction.
//!
//! [`EnsembleCrossEncoder`] runs multiple models in sequence and combines
//! their sigmoid-normalised scores via weighted mean. The two-model case
//! (e.g. ms-marco + bge) gives the lexical signal of one and the
//! conversational signal of the other.
//!
//! # Selection
//!
//! [`load_cross_encoder_from_env`] reads the environment:
//!
//! - `VELD_CROSS_ENCODER=<id>` — single model.
//! - `VELD_CROSS_ENCODERS=<id1>:<weight1>,<id2>:<weight2>,...` — weighted
//!   ensemble. Weights default to 1.0 if omitted (`<id>` is allowed).
//! - Default (neither set): ms-marco-MiniLM-L-6-v2.
//!
//! The result is shared process-wide by [`CrossEncoderSpec`]: every tenant's reranker on the
//! same models holds the same instance and so the same ONNX session.
//!
//! Files cache under `~/.cache/veld/cross-encoder/<sanitised-id>/`.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock, OnceLock, Weak};
use tokenizers::Tokenizer;

use super::minilm::OrtThreading;

/// Maximum input sequence length (query + document tokens combined).
const MAX_PAIR_LENGTH: usize = 512;

/// Default cross-encoder id when no env var is set. mxbai-rerank-xsmall-v1
/// is the smallest 2024-vintage reranker that meaningfully beats the
/// legacy ms-marco-MiniLM — ~280 MB download, 70 M params, modern QA +
/// retrieval training. Picks the size/quality sweet spot for the typical
/// veld install: fast to download, fast at query time, well above the
/// ms-marco floor. Swap up to `bge-reranker-v2-m3` (568 M) for the best
/// quality, or stack via `VELD_CROSS_ENCODERS=…`.
const DEFAULT_CROSS_ENCODER_ID: &str = "mxbai-rerank-xsmall-v1";

/// The query and the candidate the background pre-warm scores.
pub const WARMUP_TEXT: &str = "warmup";

// -----------------------------------------------------------------------------
// Trait
// -----------------------------------------------------------------------------

/// Cross-encoder reranking model. Implemented by [`OnnxCrossEncoder`] and
/// [`EnsembleCrossEncoder`].
pub trait CrossEncoderModel: Send + Sync {
    /// Stable identifier (matches the config id; e.g. `ms-marco-MiniLM-L-6-v2`).
    fn id(&self) -> &str;

    /// `true` once the underlying model(s) have been downloaded + loaded
    /// successfully. `false` if any step failed; callers fall back.
    fn is_available(&self) -> bool;

    /// Score (query, document) pairs. Returns raw logits per document; higher
    /// = more relevant. Length matches `documents`.
    fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;
}

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Configuration for an ONNX-backed cross-encoder. Owned (`String` URLs) so
/// custom entries can be added at runtime via [`register`].
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Short identifier used in logs, cache paths, and env vars.
    pub id: String,
    /// HuggingFace URL (or any HTTPS URL) for the ONNX model file.
    pub model_url: String,
    /// URL for the tokenizer JSON.
    pub tokenizer_url: String,
    /// Hint indicating whether the model's ONNX graph accepts
    /// `token_type_ids`. BERT-family models do; XLM-RoBERTa-family models
    /// (BGE, Jina) do not. This field is **informational only at runtime**
    /// — the loader probes `Session.inputs()` and uses the detected value,
    /// so a wrong hint will not produce a runtime crash, just a debug-log
    /// diagnostic. The field is retained for external registrations and
    /// diagnostics.
    pub has_token_type_ids: bool,
    /// Recommended candidate count for reranking with this model. Heavier
    /// models score fewer candidates to stay under the HTTP request budget;
    /// `VELD_RERANK_DEPTH` overrides per-process.
    pub rerank_top_k: usize,
    /// Parameter count in millions (for the picker UI and registry table).
    pub params_millions: u32,
    /// Approximate download size in MB.
    pub download_mb: u32,
    /// Year released — gives a rough freshness signal.
    pub year: u16,
    /// One-line summary of training data / domain.
    pub notes: &'static str,
}

impl CrossEncoderConfig {
    /// Sanitise the id for filesystem use (so e.g. `BAAI/bge-…` doesn't
    /// embed a directory separator in the cache path).
    fn cache_key(&self) -> String {
        self.id.replace(['/', '\\'], "_")
    }
}

// -----------------------------------------------------------------------------
// Registry — built-in catalogue + runtime registration
// -----------------------------------------------------------------------------

/// Built-in cross-encoder catalogue. Edit the table inside `built_ins()` to
/// add a new known model; consumers using a private model at runtime should
/// call [`register`] instead.
fn built_ins() -> Vec<CrossEncoderConfig> {
    vec![
        CrossEncoderConfig {
            id: "mxbai-rerank-xsmall-v1".into(),
            model_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: true,
            params_millions: 70,
            download_mb: 280,
            year: 2024,
            rerank_top_k: 20,
            notes:
                "BERT; smallest current reranker. ~3x MiniLM but trained on modern QA + retrieval; \
                 the cheap upgrade path off MS-MARCO-MiniLM.",
        },
        CrossEncoderConfig {
            id: "mxbai-rerank-base-v1".into(),
            model_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: true,
            params_millions: 184,
            download_mb: 740,
            year: 2024,
            rerank_top_k: 12,
            notes:
                "BERT; mid-size mxbai variant. Quality close to bge-reranker-v2-m3 at a third the cost.",
        },
        CrossEncoderConfig {
            id: "jina-reranker-v1-tiny-en".into(),
            model_url:
                "https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: false,
            params_millions: 33,
            download_mb: 130,
            year: 2024,
            rerank_top_k: 20,
            notes:
                "JinaBERT (English only); modern drop-in replacement for ms-marco-MiniLM-L-6-v2 \
                 in the same size class. ~50% larger than MiniLM, 3 years newer, broader training mix.",
        },
        CrossEncoderConfig {
            id: "jina-reranker-v1-turbo-en".into(),
            model_url:
                "https://huggingface.co/jinaai/jina-reranker-v1-turbo-en/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/jinaai/jina-reranker-v1-turbo-en/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: false,
            params_millions: 37,
            download_mb: 150,
            year: 2024,
            rerank_top_k: 20,
            notes:
                "JinaBERT (English only); slightly larger sibling of jina-reranker-v1-tiny-en. \
                 The mini-class option for English-only deployments wanting a small accuracy bump.",
        },
        CrossEncoderConfig {
            id: "ms-marco-MiniLM-L-6-v2".into(),
            model_url:
                "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: true,
            params_millions: 22,
            download_mb: 80,
            year: 2021,
            rerank_top_k: 20,
            notes: "BERT-style; trained on MS MARCO web-search passages. Fast, narrow domain.",
        },
        CrossEncoderConfig {
            id: "bge-reranker-v2-m3".into(),
            // BAAI's repo only ships safetensors. hooman650/bge-reranker-v2-m3-onnx-o4
            // is an Olive O4 CPU-optimised ONNX export explicitly tagged
            // apache-2.0 (derivative of BAAI's apache-2.0 base), so it's
            // safely redistributable. We previously pointed at
            // EmbeddedLLM/bge-reranker-v2-m3-onnx-o3-cpu, which has no
            // declared license and is therefore a gray-area dependency.
            // External data file is fetched automatically by
            // `download_external_data_if_present()`.
            model_url:
                "https://huggingface.co/hooman650/bge-reranker-v2-m3-onnx-o4/resolve/main/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/hooman650/bge-reranker-v2-m3-onnx-o4/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: false,
            params_millions: 568,
            download_mb: 2270,
            year: 2024,
            rerank_top_k: 8,
            notes:
                "XLM-RoBERTa-style; multilingual + multi-task (QA, retrieval, conversational). \
                 Best quality but biggest download (2.3 GB across model.onnx + model.onnx_data). \
                 Source: hooman650/bge-reranker-v2-m3-onnx-o4 (Olive O4 CPU-optimised, apache-2.0).",
        },
        CrossEncoderConfig {
            id: "mxbai-rerank-large-v1".into(),
            model_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: true,
            params_millions: 435,
            download_mb: 1740,
            year: 2024,
            rerank_top_k: 8,
            notes: "BERT-style; trained on QA + retrieval pairs. Strong on factoid queries.",
        },
        CrossEncoderConfig {
            id: "jina-reranker-v2-base-multilingual".into(),
            model_url:
                "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: false,
            params_millions: 278,
            download_mb: 1110,
            year: 2024,
            rerank_top_k: 10,
            notes:
                "XLM-RoBERTa-style; multilingual; late-interaction-influenced training. Mid-size sweet spot.",
        },
    ]
}

/// Editable registry. Keyed by id; populated from `built_ins()` on first
/// access. Callers can [`register`] custom entries at runtime (e.g. a
/// private fine-tune hosted on a corporate model server).
static REGISTRY: LazyLock<RwLock<HashMap<String, CrossEncoderConfig>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    for cfg in built_ins() {
        map.insert(cfg.id.clone(), cfg);
    }
    RwLock::new(map)
});

/// Look up a cross-encoder config by id. Built-ins are seeded from
/// [`built_ins`]; additional entries can be added with [`register`].
pub fn lookup(id: &str) -> Result<CrossEncoderConfig> {
    REGISTRY.read().get(id).cloned().with_context(|| {
        let mut ids: Vec<String> = REGISTRY.read().keys().cloned().collect();
        ids.sort();
        format!(
            "unknown cross-encoder id `{id}` (known: {})",
            ids.join(", ")
        )
    })
}

/// Register (or override) a custom cross-encoder config. Useful for
/// privately-hosted models or fine-tunes that shouldn't be hard-coded.
pub fn register(config: CrossEncoderConfig) {
    REGISTRY.write().insert(config.id.clone(), config);
}

/// Recommended rerank candidate count for the active cross-encoder.
///
/// Resolution order:
/// 1. `VELD_RERANK_DEPTH=<usize>` — operator override, applied globally.
/// 2. Per-model `rerank_top_k` from the registry entry pointed at by
///    `VELD_CROSS_ENCODER` (or default).
/// 3. Fallback constant (20) if neither is resolvable.
pub fn current_rerank_depth() -> usize {
    if let Ok(raw) = std::env::var("VELD_RERANK_DEPTH") {
        if let Ok(n) = raw.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
            tracing::warn!("VELD_RERANK_DEPTH=`{raw}` invalid (must be > 0); using model default");
        } else {
            tracing::warn!("VELD_RERANK_DEPTH=`{raw}` is not a usize; using model default");
        }
    }

    let id = std::env::var("VELD_CROSS_ENCODER")
        .ok()
        .unwrap_or_else(|| DEFAULT_CROSS_ENCODER_ID.to_string());

    lookup(&id).map(|c| c.rerank_top_k).unwrap_or(20)
}

/// Snapshot of registered models — `(id, params_M, download_MB, year, notes)`.
/// For UI / docs / `--list-rerankers` style introspection.
pub fn list_registered() -> Vec<(String, u32, u32, u16, &'static str)> {
    let r = REGISTRY.read();
    let mut entries: Vec<_> = r
        .values()
        .map(|c| {
            (
                c.id.clone(),
                c.params_millions,
                c.download_mb,
                c.year,
                c.notes,
            )
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

// -----------------------------------------------------------------------------
// ONNX cross-encoder
// -----------------------------------------------------------------------------

/// What the model's primary output head looks like — detected by inspecting
/// the loaded ONNX graph, not the registry's hint, so a wrong (or missing)
/// `rerank_top_k`/`has_token_type_ids` config never lets a runtime bug
/// through.
#[derive(Copy, Clone, Debug, PartialEq)]
enum OutputHead {
    /// `[…, 1]` — a single relevance logit per pair. Used by
    /// `ms-marco-MiniLM-L-6-v2`, mxbai-rerank-* (BERT family). Score is
    /// the logit itself.
    Regression,
    /// `[…, 2]` — `[not_relevant, relevant]` logits. Used by
    /// `bge-reranker-v2-m3` and most XLM-RoBERTa-based rerankers. Score
    /// is `softmax([neg, pos])[1]` so it's a probability in `[0, 1]` and
    /// comparable to regression logits after sigmoid.
    BinaryClassifier,
}

impl OutputHead {
    /// Probe the session's first output and decide which head it is.
    fn detect(session: &Session, ce_id: &str) -> Self {
        let last_dim = session.outputs().first().and_then(|o| match o.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.last().copied(),
            _ => None,
        });
        match last_dim {
            Some(1) => Self::Regression,
            Some(2) => Self::BinaryClassifier,
            Some(other) => {
                tracing::warn!(
                    "cross-encoder `{ce_id}` has unexpected output last-dim {other}; \
                     defaulting to regression-head interpretation"
                );
                Self::Regression
            }
            None => {
                tracing::warn!(
                    "cross-encoder `{ce_id}` exposes no output shape; defaulting to regression"
                );
                Self::Regression
            }
        }
    }

    /// Reduce the raw output tensor's flat `f32` slice to a single relevance
    /// score per pair.
    fn extract_score(&self, data: &[f32]) -> f32 {
        match self {
            Self::Regression => data.first().copied().unwrap_or(0.0),
            Self::BinaryClassifier => {
                let neg = data.first().copied().unwrap_or(0.0);
                let pos = data.get(1).copied().unwrap_or(0.0);
                // Numerically-stable softmax over 2 logits, return P(relevant).
                let m = neg.max(pos);
                let e_neg = (neg - m).exp();
                let e_pos = (pos - m).exp();
                e_pos / (e_neg + e_pos)
            }
        }
    }
}

struct LazyOnnx {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// Whether the loaded ONNX graph accepts `token_type_ids`. Detected
    /// from `session.inputs` at load time — the registry's
    /// `has_token_type_ids` config field is now only an unused hint.
    has_token_type_ids: bool,
    /// Output head strategy, detected from `session.outputs[0]`'s shape.
    output_head: OutputHead,
}

/// Cross-encoder backed by a single ONNX model.
pub struct OnnxCrossEncoder {
    config: CrossEncoderConfig,
    threading: OrtThreading,
    model: OnceLock<Result<LazyOnnx, String>>,
    available: AtomicBool,
}

impl OnnxCrossEncoder {
    pub fn new(config: CrossEncoderConfig) -> Self {
        Self::with_threading(config, OrtThreading::resolve())
    }

    /// [`Self::new`] with the session threading given rather than resolved.
    pub fn with_threading(config: CrossEncoderConfig, threading: OrtThreading) -> Self {
        Self {
            config,
            threading,
            model: OnceLock::new(),
            available: AtomicBool::new(true),
        }
    }

    /// True once a download or load was attempted and failed. The failure is kept for this
    /// instance's lifetime, so the shared registry replaces such an instance.
    fn load_failed(&self) -> bool {
        matches!(self.model.get(), Some(Err(_)))
    }

    fn cache_dir(&self) -> PathBuf {
        super::downloader::get_cache_dir()
            .join("cross-encoder")
            .join(self.config.cache_key())
    }

    /// Eagerly download + load on a background thread. Idempotent.
    /// Skipped when `VELD_LAZY_CROSS_ENCODER=1`.
    pub fn prewarm(self_arc: Arc<Self>) {
        let lazy = std::env::var("VELD_LAZY_CROSS_ENCODER")
            .ok()
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        if lazy {
            tracing::debug!(
                "cross-encoder pre-warm skipped for {} (VELD_LAZY_CROSS_ENCODER set)",
                self_arc.config.id
            );
            return;
        }
        // The thread below is detached: `.spawn(…).ok()` drops the JoinHandle and
        // nothing joins it. A test binary exits seconds after start, so teardown can
        // race an in-flight ONNX CreateSession; a server runs for hours and the
        // pre-warm is long done.
        //
        // Measured 2026-09-02: `cargo test --lib memory::` segfaulted on ~half of
        // parallel runs, always AFTER `test result: ok`; the crash report named this
        // thread, EXC_BAD_ACCESS at 0x17 in InferenceSession::Initialize. Three
        // single-threaded runs were clean.
        //
        // Skipped rather than joined — a pre-warm you wait for is not a pre-warm.
        // A caller needing the model loaded calls `ensure_loaded()` directly.
        if cfg!(test) {
            return;
        }
        std::thread::Builder::new()
            .name(format!("cross-encoder-prewarm-{}", self_arc.config.id))
            .spawn(move || match self_arc.ensure_loaded() {
                Ok(_) => tracing::info!("Cross-encoder pre-warmed: {}", self_arc.config.id),
                Err(e) => tracing::warn!(
                    "Cross-encoder pre-warm failed for {} (lazy fallback engaged): {e}",
                    self_arc.config.id
                ),
            })
            .ok();
    }

    fn ensure_downloaded(&self) -> Result<(PathBuf, PathBuf)> {
        let dir = self.cache_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create cross-encoder cache dir: {dir:?}"))?;

        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        if model_path.exists() && tokenizer_path.exists() {
            return Ok((model_path, tokenizer_path));
        }

        if std::env::var("VELD_OFFLINE").is_ok_and(|v| v == "true" || v == "1") {
            anyhow::bail!(
                "Cross-encoder {} not cached and VELD_OFFLINE=true. \
                 Download manually to {:?}",
                self.config.id,
                dir
            );
        }

        tracing::info!("Downloading cross-encoder model ({})...", self.config.id);
        if !model_path.exists() {
            download_file(&self.config.model_url, &model_path)
                .with_context(|| format!("Failed to download model for {}", self.config.id))?;
            // ONNX exports of models >2 GB split into `model.onnx` + a companion
            // external-data file. The exact filename varies between
            // `model.onnx_data` and `model.onnx.data` depending on the export
            // tool. Try both, best-effort; small models that don't need it
            // simply 404 silently. ORT loads whichever sits next to model.onnx.
            for ext in ["onnx_data", "onnx.data"] {
                let data_url = derive_data_url(&self.config.model_url, ext);
                let data_path = dir.join(format!("model.{ext}"));
                if let Err(e) = download_file(&data_url, &data_path) {
                    tracing::debug!(
                        "Optional external data file not present for {} ({ext}): {e}",
                        self.config.id
                    );
                    let _ = std::fs::remove_file(&data_path);
                }
            }
        }
        if !tokenizer_path.exists() {
            download_file(&self.config.tokenizer_url, &tokenizer_path)
                .with_context(|| format!("Failed to download tokenizer for {}", self.config.id))?;
        }
        tracing::info!("Cross-encoder downloaded: {}", self.config.id);
        Ok((model_path, tokenizer_path))
    }

    fn ensure_loaded(&self) -> Result<&LazyOnnx> {
        let result = self.model.get_or_init(|| {
            let (model_path, tokenizer_path) = match self.ensure_downloaded() {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("Cross-encoder unavailable ({}): {e}", self.config.id);
                    self.available
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    return Err(e.to_string());
                }
            };
            super::model_tape::register_model(
                super::model_tape::Model::CrossEncoder,
                &[&model_path, &tokenizer_path],
            );

            let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
                Ok(t) => t,
                Err(e) => {
                    self.available
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    return Err(format!(
                        "Failed to load tokenizer for {}: {e}",
                        self.config.id
                    ));
                }
            };

            // ort's `load-dynamic` PANICS (not Errs) when libonnxruntime cannot be
            // dlopen'd, which ends the calling thread instead of engaging the lazy
            // fallback below (and the whole process in the abort-built fortress
            // profile). Resolve + verify the runtime path before touching any ort
            // API, and fail soft (cross-encoder disabled) if it is unavailable.
            // Cross-platform: checks the resolved .dylib/.dll/.so path.
            super::minilm::pre_init_ort_runtime(
                std::env::var("VELD_OFFLINE").is_ok_and(|v| v == "true" || v == "1"),
            );
            let ort_ready = std::env::var("ORT_DYLIB_PATH")
                .ok()
                .map(|p| std::path::Path::new(&p).exists())
                .unwrap_or(false);
            if !ort_ready {
                self.available
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                return Err(format!(
                    "ONNX Runtime unavailable for {} (ORT_DYLIB_PATH unset or missing); \
                     cross-encoder disabled, lazy fallback engaged",
                    self.config.id
                ));
            }

            let session = match (|| -> Result<Session> {
                self.threading
                    .apply(Session::builder().context("Failed to create session builder")?)?
                    .commit_from_file(&model_path)
                    .context("Failed to load ONNX model")
            })() {
                Ok(s) => s,
                Err(e) => {
                    self.available
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    return Err(format!(
                        "Failed to load ONNX model for {}: {e}",
                        self.config.id
                    ));
                }
            };

            // Probe the loaded graph instead of trusting the registry — this
            // closes the [1,1] vs [1,2] output-head bug class and the
            // input-layout mismatch class at the same time.
            let has_token_type_ids = session
                .inputs()
                .iter()
                .any(|i| i.name() == "token_type_ids");
            let output_head = OutputHead::detect(&session, &self.config.id);
            if has_token_type_ids != self.config.has_token_type_ids {
                tracing::debug!(
                    "cross-encoder `{}`: registry has_token_type_ids={} but graph says {}; \
                     using graph value",
                    self.config.id,
                    self.config.has_token_type_ids,
                    has_token_type_ids
                );
            }
            tracing::info!(
                "cross-encoder `{}` loaded: head={:?} token_type_ids={}",
                self.config.id,
                output_head,
                has_token_type_ids
            );

            Ok(LazyOnnx {
                session: Mutex::new(session),
                tokenizer,
                has_token_type_ids,
                output_head,
            })
        });

        match result {
            Ok(model) => Ok(model),
            Err(msg) => Err(anyhow::anyhow!("{}", msg)),
        }
    }
}

impl CrossEncoderModel for OnnxCrossEncoder {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn is_available(&self) -> bool {
        self.available.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Scores N (query, document) pairs, each in its own forward pass at its own token length,
    /// capped at `MAX_PAIR_LENGTH`. One batch would pad every pair to the longest, letting one long
    /// memory set the cost of all of them; on live 20-candidate sets, pair by pair takes half the
    /// time for the same scores. The session lock is held across the N passes.
    fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Loaded first even under a replay, so the tape can check which model files it was
        // recorded from. Each pair is keyed on its own.
        let model = self.ensure_loaded()?;
        if let Some(taped) = super::model_tape::cross_encode(query, documents, || {
            self.score_pairs_live(model, query, documents)
        }) {
            return taped;
        }
        self.score_pairs_live(model, query, documents)
    }
}

impl OnnxCrossEncoder {
    /// One forward pass per pair; `documents` is non-empty.
    fn score_pairs_live(
        &self,
        model: &LazyOnnx,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<f32>> {
        let mut session = model
            .session
            .try_lock_for(std::time::Duration::from_secs(30))
            .ok_or_else(|| {
                anyhow::anyhow!("Cross-encoder session lock timeout ({})", self.config.id)
            })?;
        let head_size = match model.output_head {
            OutputHead::Regression => 1_usize,
            OutputHead::BinaryClassifier => 2_usize,
        };

        let mut scores = Vec::with_capacity(documents.len());
        for &doc in documents {
            let encoding = model
                .tokenizer
                .encode((query, doc), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed ({}): {e}", self.config.id))?;
            let len = encoding
                .get_attention_mask()
                .iter()
                .rposition(|&m| m != 0)
                .map_or(0, |p| p + 1)
                .clamp(1, MAX_PAIR_LENGTH);
            let row = |values: &[u32]| -> Vec<i64> {
                let mut out: Vec<i64> = values.iter().take(len).map(|&v| i64::from(v)).collect();
                out.resize(len, 0);
                out
            };
            let input_ids = Value::from_array((vec![1, len], row(encoding.get_ids())))?;
            let attention = Value::from_array((vec![1, len], row(encoding.get_attention_mask())))?;
            let outputs = if model.has_token_type_ids {
                let token_types = Value::from_array((vec![1, len], row(encoding.get_type_ids())))?;
                session.run(ort::inputs![
                    "input_ids"      => &input_ids,
                    "attention_mask" => &attention,
                    "token_type_ids" => &token_types,
                ])?
            } else {
                session.run(ort::inputs![
                    "input_ids"      => &input_ids,
                    "attention_mask" => &attention,
                ])?
            };
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            scores.push(model.output_head.extract_score(&data[..head_size]));
        }

        Ok(scores)
    }
}

// -----------------------------------------------------------------------------
// Ensemble
// -----------------------------------------------------------------------------

/// Weighted ensemble of multiple cross-encoders. Each member's raw logits
/// are sigmoid-normalised to `[0,1]` before averaging so models with
/// different score scales blend correctly.
pub struct EnsembleCrossEncoder {
    id: String,
    members: Vec<(Arc<dyn CrossEncoderModel>, f32)>,
}

impl EnsembleCrossEncoder {
    pub fn new(members: Vec<(Arc<dyn CrossEncoderModel>, f32)>) -> Self {
        let id = members
            .iter()
            .map(|(m, w)| format!("{}:{:.2}", m.id(), w))
            .collect::<Vec<_>>()
            .join("+");
        Self { id, members }
    }
}

impl CrossEncoderModel for EnsembleCrossEncoder {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_available(&self) -> bool {
        // The ensemble is usable as soon as any member is available.
        self.members.iter().any(|(m, _)| m.is_available())
    }

    fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut combined = vec![0.0_f32; documents.len()];
        let mut total_weight = 0.0_f32;
        for (model, weight) in &self.members {
            if !model.is_available() {
                tracing::debug!("ensemble: skipping unavailable member {}", model.id());
                continue;
            }
            let scores = match model.score_pairs(query, documents) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("ensemble member {} failed: {e}", model.id());
                    continue;
                }
            };
            for (i, raw) in scores.iter().enumerate() {
                combined[i] += weight * sigmoid(*raw);
            }
            total_weight += weight;
        }
        if total_weight <= 0.0 {
            anyhow::bail!("ensemble: no members produced scores");
        }
        for c in combined.iter_mut() {
            *c /= total_weight;
        }
        Ok(combined)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// -----------------------------------------------------------------------------
// Env-var-driven factory
// -----------------------------------------------------------------------------

/// The env-configured cross-encoder, shared with every other caller whose environment
/// resolves to the same models. Default is [`DEFAULT_CROSS_ENCODER_ID`] when neither
/// `VELD_CROSS_ENCODER` nor `VELD_CROSS_ENCODERS` is set.
///
/// `VELD_CROSS_ENCODERS` takes precedence when both are set.
pub fn load_cross_encoder_from_env() -> Arc<dyn CrossEncoderModel> {
    cross_encoder_for(&CrossEncoderSpec::from_env())
}

/// What selects a cross-encoder, as values: the member models, their ensemble weights, and the
/// session threading. Equal specs share one instance.
#[derive(Clone, Debug)]
pub struct CrossEncoderSpec {
    /// One entry is a single model; more are a weighted ensemble.
    members: Vec<(CrossEncoderConfig, f32)>,
    threading: OrtThreading,
}

impl CrossEncoderSpec {
    /// Resolve from `VELD_CROSS_ENCODERS`, then `VELD_CROSS_ENCODER`, then the default; an
    /// unparseable value is warned about and the next source is used.
    pub fn from_env() -> Self {
        if let Ok(spec) = std::env::var("VELD_CROSS_ENCODERS") {
            match parse_ensemble_members(&spec) {
                Ok(members) => return Self::of(members),
                Err(e) => {
                    tracing::warn!(
                        "VELD_CROSS_ENCODERS=`{spec}` invalid: {e}; falling back to default"
                    );
                }
            }
        }

        if let Ok(id) = std::env::var("VELD_CROSS_ENCODER") {
            match lookup(id.trim()) {
                Ok(cfg) => return Self::of(vec![(cfg, 1.0)]),
                Err(e) => {
                    tracing::warn!(
                        "VELD_CROSS_ENCODER=`{id}` invalid: {e}; falling back to default"
                    );
                }
            }
        }

        let cfg = lookup(DEFAULT_CROSS_ENCODER_ID)
            .expect("default cross-encoder id must be registered as a built-in");
        Self::of(vec![(cfg, 1.0)])
    }

    /// `members` (at least one) with the process's session threading.
    fn of(members: Vec<(CrossEncoderConfig, f32)>) -> Self {
        Self {
            members,
            threading: OrtThreading::resolve(),
        }
    }
}

/// What one [`OnnxCrossEncoder`] is keyed by: the files it loads and how its session threads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ModelKey {
    id: String,
    model_url: String,
    tokenizer_url: String,
    threading: OrtThreading,
}

impl ModelKey {
    fn new(config: &CrossEncoderConfig, threading: OrtThreading) -> Self {
        Self {
            id: config.id.clone(),
            model_url: config.model_url.clone(),
            tokenizer_url: config.tokenizer_url.clone(),
            threading,
        }
    }
}

/// A shared ensemble and the shared models it is made of.
struct SharedEnsemble {
    ensemble: Arc<EnsembleCrossEncoder>,
    members: Vec<Arc<OnnxCrossEncoder>>,
}

static SHARED_MODELS: LazyLock<Mutex<HashMap<ModelKey, Arc<OnnxCrossEncoder>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Each member's [`ModelKey`] and the bits of its weight.
type EnsembleKey = Vec<(ModelKey, u32)>;

static SHARED_ENSEMBLES: LazyLock<Mutex<HashMap<EnsembleKey, SharedEnsemble>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The model `config` loads with `threading`, one instance per process. An instance whose
/// download or load failed is replaced, so the next caller retries instead of inheriting it.
fn shared_model(config: &CrossEncoderConfig, threading: OrtThreading) -> Arc<OnnxCrossEncoder> {
    let key = ModelKey::new(config, threading);
    let mut models = SHARED_MODELS.lock();
    if let Some(model) = models.get(&key).filter(|m| !m.load_failed()) {
        return Arc::clone(model);
    }
    let model = Arc::new(OnnxCrossEncoder::with_threading(config.clone(), threading));
    models.insert(key, Arc::clone(&model));
    model
}

/// The cross-encoder `spec` selects, shared with every caller whose spec is equal. A
/// single-member spec is that model itself; an ensemble's members are the same instances a
/// single-model spec on their files gets, so a model is loaded once however it is used.
pub fn cross_encoder_for(spec: &CrossEncoderSpec) -> Arc<dyn CrossEncoderModel> {
    if let [(config, _)] = spec.members.as_slice() {
        return shared_model(config, spec.threading);
    }
    let key: EnsembleKey = spec
        .members
        .iter()
        .map(|(config, weight)| (ModelKey::new(config, spec.threading), weight.to_bits()))
        .collect();
    let mut ensembles = SHARED_ENSEMBLES.lock();
    if let Some(shared) = ensembles.get(&key) {
        if !shared.members.iter().any(|m| m.load_failed()) {
            return Arc::clone(&shared.ensemble) as Arc<dyn CrossEncoderModel>;
        }
    }
    let members: Vec<Arc<OnnxCrossEncoder>> = spec
        .members
        .iter()
        .map(|(config, _)| shared_model(config, spec.threading))
        .collect();
    let ensemble = Arc::new(EnsembleCrossEncoder::new(
        members
            .iter()
            .zip(&spec.members)
            .map(|(model, (_, weight))| (Arc::clone(model) as Arc<dyn CrossEncoderModel>, *weight))
            .collect(),
    ));
    ensembles.insert(
        key,
        SharedEnsemble {
            ensemble: Arc::clone(&ensemble),
            members,
        },
    );
    ensemble
}

/// `<id>[:<weight>],…` as configs and weights; at least one member.
fn parse_ensemble_members(spec: &str) -> Result<Vec<(CrossEncoderConfig, f32)>> {
    let mut members = Vec::new();
    for raw in spec.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let (id, weight) = match raw.split_once(':') {
            Some((id, w)) => (
                id.trim(),
                w.trim()
                    .parse::<f32>()
                    .with_context(|| format!("invalid weight in ensemble entry `{raw}`"))?,
            ),
            None => (raw, 1.0),
        };
        members.push((lookup(id)?, weight));
    }
    if members.is_empty() {
        anyhow::bail!("VELD_CROSS_ENCODERS produced no members");
    }
    Ok(members)
}

/// Whether `model` is yet to be pre-warmed, recording that it now has been. Every tenant's
/// reranker passes the shared model to [`prewarm`]; one warm-up per instance is enough, and
/// each further one would hold the session lock against a live rerank.
fn claim_prewarm(model: &Arc<dyn CrossEncoderModel>) -> bool {
    static PREWARMED: LazyLock<Mutex<Vec<Weak<dyn CrossEncoderModel>>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
    let mut prewarmed = PREWARMED.lock();
    prewarmed.retain(|seen| seen.strong_count() > 0);
    if prewarmed
        .iter()
        .any(|seen| std::ptr::addr_eq(seen.as_ptr(), Arc::as_ptr(model)))
    {
        return false;
    }
    prewarmed.push(Arc::downgrade(model));
    true
}

/// Pre-warm a cross-encoder (single model or ensemble) by issuing a dummy
/// `score_pairs` call on a background thread. For [`OnnxCrossEncoder`] this
/// triggers the ~80 MB download and ONNX session build; for
/// [`EnsembleCrossEncoder`] it cascades into every member. Idempotent: an
/// instance is warmed once however many rerankers share it.
///
/// Skipped when `VELD_LAZY_CROSS_ENCODER=1` is set.
pub fn prewarm(model: Arc<dyn CrossEncoderModel>) {
    let lazy = std::env::var("VELD_LAZY_CROSS_ENCODER")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    if lazy {
        tracing::debug!("cross-encoder pre-warm skipped ({})", model.id());
        return;
    }
    let id = model.id().to_string();
    // Same detached-thread hazard as the method above — see its comment.
    if cfg!(test) {
        return;
    }
    if !claim_prewarm(&model) {
        tracing::debug!("cross-encoder pre-warm already issued for {id}");
        return;
    }
    std::thread::Builder::new()
        .name(format!("cross-encoder-prewarm-{id}"))
        .spawn(
            move || match model.score_pairs(WARMUP_TEXT, &[WARMUP_TEXT]) {
                Ok(_) => tracing::info!("Cross-encoder pre-warmed: {id}"),
                Err(e) => tracing::warn!(
                    "Cross-encoder pre-warm failed for {id} (lazy fallback engaged): {e}"
                ),
            },
        )
        .ok();
}

// -----------------------------------------------------------------------------
// File download helper
// -----------------------------------------------------------------------------

fn download_file(url: &str, path: &PathBuf) -> Result<()> {
    tracing::info!("Downloading cross-encoder file: {url}");

    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP request failed: {url}"))?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        anyhow::bail!("HTTP {status} for {url}");
    }

    let mut reader = resp.into_body().into_reader();
    let mut file =
        std::fs::File::create(path).with_context(|| format!("Failed to create file: {path:?}"))?;

    std::io::copy(&mut reader, &mut file)?;
    Ok(())
}

/// Given a model.onnx URL, return the companion external-data URL with the
/// given extension (e.g. `onnx_data` or `onnx.data`).
fn derive_data_url(model_url: &str, ext: &str) -> String {
    // Replace the trailing `model.onnx` with `model.<ext>`.
    if let Some(stripped) = model_url.strip_suffix("model.onnx") {
        format!("{stripped}model.{ext}")
    } else {
        // Fallback for non-standard URLs: just append.
        format!("{model_url}.{ext}")
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_keys_are_filesystem_safe() {
        let cfg = lookup("bge-reranker-v2-m3").unwrap();
        let key = cfg.cache_key();
        assert!(!key.contains('/'));
        assert!(!key.contains('\\'));
    }

    #[test]
    fn lookup_resolves_built_ins() {
        assert!(lookup("ms-marco-MiniLM-L-6-v2").is_ok());
        assert!(lookup("bge-reranker-v2-m3").is_ok());
        assert!(lookup("does-not-exist").is_err());
    }

    #[test]
    fn registry_default_is_resolvable() {
        // Default id is registered as a built-in. The specific id is whatever
        // the project currently picks as the sweet spot — verify it resolves.
        assert!(lookup(DEFAULT_CROSS_ENCODER_ID).is_ok());
    }

    #[test]
    fn list_registered_returns_all_built_ins() {
        let entries = list_registered();
        assert!(entries.len() >= 4);
        assert!(entries.iter().any(|(id, ..)| id == "bge-reranker-v2-m3"));
    }

    #[test]
    fn register_overrides_existing() {
        register(CrossEncoderConfig {
            id: "test-custom-model".into(),
            model_url: "https://example.com/m.onnx".into(),
            tokenizer_url: "https://example.com/t.json".into(),
            has_token_type_ids: false,
            params_millions: 100,
            download_mb: 400,
            year: 2026,
            rerank_top_k: 15,
            notes: "test fixture",
        });
        let cfg = lookup("test-custom-model").unwrap();
        assert_eq!(cfg.params_millions, 100);
    }

    fn spec(members: &str) -> CrossEncoderSpec {
        CrossEncoderSpec::of(parse_ensemble_members(members).unwrap())
    }

    fn from_spec(members: &str) -> Arc<dyn CrossEncoderModel> {
        cross_encoder_for(&spec(members))
    }

    #[test]
    fn parse_ensemble_spec_handles_weights() {
        let res = from_spec("ms-marco-MiniLM-L-6-v2:0.5,bge-reranker-v2-m3:1.5");
        assert!(res.id().contains("0.50"));
        assert!(res.id().contains("1.50"));
    }

    #[test]
    fn parse_ensemble_spec_collapses_singleton() {
        let res = from_spec("ms-marco-MiniLM-L-6-v2");
        assert_eq!(res.id(), "ms-marco-MiniLM-L-6-v2");
    }

    #[test]
    fn parse_ensemble_spec_refuses_an_empty_list() {
        assert!(parse_ensemble_members(" , ").is_err());
    }

    /// Every tenant's reranker on the same models holds one instance, so one ONNX session.
    /// Each spec is resolved once: the recall harness may pin the thread count mid-run.
    #[test]
    fn equal_specs_share_one_instance_and_different_specs_do_not() {
        let ms_marco = spec("ms-marco-MiniLM-L-6-v2");
        let single = cross_encoder_for(&ms_marco);
        assert!(Arc::ptr_eq(&single, &cross_encoder_for(&ms_marco.clone())));
        assert!(!Arc::ptr_eq(&single, &from_spec("bge-reranker-v2-m3")));

        let pair = CrossEncoderSpec {
            threading: ms_marco.threading,
            ..spec("ms-marco-MiniLM-L-6-v2:0.5,bge-reranker-v2-m3:1.5")
        };
        let ensemble = cross_encoder_for(&pair);
        assert!(Arc::ptr_eq(&ensemble, &cross_encoder_for(&pair.clone())));
        let key: EnsembleKey = pair
            .members
            .iter()
            .map(|(config, weight)| (ModelKey::new(config, pair.threading), weight.to_bits()))
            .collect();
        let first_member: Arc<OnnxCrossEncoder> =
            Arc::clone(&SHARED_ENSEMBLES.lock()[&key].members[0]);
        let first_member: Arc<dyn CrossEncoderModel> = first_member;
        assert!(
            Arc::ptr_eq(&first_member, &single),
            "an ensemble member is the single model's instance, so its session loads once"
        );
        let mut reweighted = pair.clone();
        reweighted.members[0].1 = 1.5;
        assert!(
            !Arc::ptr_eq(&ensemble, &cross_encoder_for(&reweighted)),
            "the weights are part of the spec"
        );

        let mut other_threads = ms_marco;
        other_threads.threading.intra += 1;
        assert!(
            !Arc::ptr_eq(&single, &cross_encoder_for(&other_threads)),
            "a session threaded differently is a different instance"
        );
    }

    #[test]
    fn an_instance_is_prewarmed_once() {
        let cfg = lookup("ms-marco-MiniLM-L-6-v2").unwrap();
        let a: Arc<dyn CrossEncoderModel> = Arc::new(OnnxCrossEncoder::new(cfg.clone()));
        let b: Arc<dyn CrossEncoderModel> = Arc::new(OnnxCrossEncoder::new(cfg.clone()));
        assert!(claim_prewarm(&a));
        assert!(!claim_prewarm(&a), "a second reranker on the same instance");
        assert!(claim_prewarm(&b), "another instance is warmed on its own");
        drop(a);
        let c: Arc<dyn CrossEncoderModel> = Arc::new(OnnxCrossEncoder::new(cfg));
        assert!(
            claim_prewarm(&c),
            "a new instance is never mistaken for a dropped one at the same address"
        );
    }

    // -- OutputHead detection --------------------------------------------

    #[test]
    fn output_head_regression_extracts_first_value() {
        // Regression-head models emit a single relevance logit per pair —
        // ms-marco-MiniLM, mxbai-rerank-*. Score is the raw value.
        let data = [1.7_f32, 9.9, -3.4];
        assert_eq!(OutputHead::Regression.extract_score(&data), 1.7);
    }

    #[test]
    fn output_head_binary_classifier_softmaxes_pair() {
        // P(relevant) softmax of [neg, pos]; logits equal => 0.5.
        let equal = OutputHead::BinaryClassifier.extract_score(&[2.0, 2.0]);
        assert!((equal - 0.5).abs() < 1e-6);

        // Pos >> neg => approaches 1.0.
        let high = OutputHead::BinaryClassifier.extract_score(&[-10.0, 10.0]);
        assert!(high > 0.999);

        // Neg >> pos => approaches 0.0.
        let low = OutputHead::BinaryClassifier.extract_score(&[10.0, -10.0]);
        assert!(low < 0.001);
    }

    #[test]
    fn output_head_binary_classifier_is_numerically_stable() {
        // Naive `exp(x)/sum(exp)` overflows around x ≈ 88 in f32; the
        // stabilised form must not.
        let big = OutputHead::BinaryClassifier.extract_score(&[1000.0, 1001.0]);
        assert!(big.is_finite());
        assert!(big > 0.7);
    }

    #[test]
    fn binary_classifier_inverts_when_misread_as_regression() {
        // Regression interpretation of a binary-head output reads `neg` as
        // the score → highly relevant pairs (large positive `pos`) end up
        // *demoted* relative to less relevant ones. This is the exact bug
        // class that `OutputHead::detect` exists to close. The assertion
        // documents why the dispatch matters.
        let very_relevant = [-10.0_f32, 10.0];
        let barely_relevant = [-1.0_f32, 1.0];
        let reg_score_relevant = OutputHead::Regression.extract_score(&very_relevant);
        let reg_score_barely = OutputHead::Regression.extract_score(&barely_relevant);
        assert!(reg_score_relevant < reg_score_barely); // inverted!

        let bc_score_relevant = OutputHead::BinaryClassifier.extract_score(&very_relevant);
        let bc_score_barely = OutputHead::BinaryClassifier.extract_score(&barely_relevant);
        assert!(bc_score_relevant > bc_score_barely); // correct ordering.
    }
}
