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
//! Files cache under `~/.cache/veld/cross-encoder/<sanitised-id>/`.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock, OnceLock};
use tokenizers::Tokenizer;

/// Maximum input sequence length (query + document tokens combined).
const MAX_PAIR_LENGTH: usize = 512;

/// Default cross-encoder id used when no env var is set. BGE-reranker-v2-m3
/// outperformed ms-marco-MiniLM-L-6-v2 on every LoCoMo category in our
/// internal bench because it was trained on diverse multi-task data
/// (QA + retrieval + conversational) instead of MS MARCO web-search.
const DEFAULT_CROSS_ENCODER_ID: &str = "bge-reranker-v2-m3";

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
    /// Whether the model's ONNX graph accepts `token_type_ids`. BERT-family
    /// models do; XLM-RoBERTa-family models (BGE, Jina) do not.
    pub has_token_type_ids: bool,
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
            notes: "BERT-style; trained on MS MARCO web-search passages. Fast, narrow domain.",
        },
        CrossEncoderConfig {
            id: "bge-reranker-v2-m3".into(),
            model_url:
                "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/onnx/model.onnx".into(),
            tokenizer_url:
                "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/tokenizer.json".into(),
            has_token_type_ids: false,
            params_millions: 568,
            download_mb: 2270,
            year: 2024,
            notes:
                "XLM-RoBERTa-style; multilingual + multi-task (QA, retrieval, conversational). Default.",
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
    REGISTRY
        .read()
        .get(id)
        .cloned()
        .with_context(|| {
            let mut ids: Vec<String> = REGISTRY.read().keys().cloned().collect();
            ids.sort();
            format!("unknown cross-encoder id `{id}` (known: {})", ids.join(", "))
        })
}

/// Register (or override) a custom cross-encoder config. Useful for
/// privately-hosted models or fine-tunes that shouldn't be hard-coded.
pub fn register(config: CrossEncoderConfig) {
    REGISTRY.write().insert(config.id.clone(), config);
}

/// Snapshot of registered models — `(id, params_M, download_MB, year, notes)`.
/// For UI / docs / `--list-rerankers` style introspection.
pub fn list_registered() -> Vec<(String, u32, u32, u16, &'static str)> {
    let r = REGISTRY.read();
    let mut entries: Vec<_> = r
        .values()
        .map(|c| (c.id.clone(), c.params_millions, c.download_mb, c.year, c.notes))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

// -----------------------------------------------------------------------------
// ONNX cross-encoder
// -----------------------------------------------------------------------------

struct LazyOnnx {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    has_token_type_ids: bool,
}

/// Cross-encoder backed by a single ONNX model.
pub struct OnnxCrossEncoder {
    config: CrossEncoderConfig,
    model: OnceLock<Result<LazyOnnx, String>>,
    available: AtomicBool,
}

impl OnnxCrossEncoder {
    pub fn new(config: CrossEncoderConfig) -> Self {
        Self {
            config,
            model: OnceLock::new(),
            available: AtomicBool::new(true),
        }
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
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        if lazy {
            tracing::debug!(
                "cross-encoder pre-warm skipped for {} (VELD_LAZY_CROSS_ENCODER set)",
                self_arc.config.id
            );
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

            let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
                Ok(t) => t,
                Err(e) => {
                    self.available
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    return Err(format!("Failed to load tokenizer for {}: {e}", self.config.id));
                }
            };

            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let num_threads = 1;
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let num_threads = 2;

            let session = match (|| -> Result<Session> {
                let builder = Session::builder()
                    .context("Failed to create session builder")?
                    .with_intra_threads(num_threads)
                    .context("Failed to set thread count")?;
                builder
                    .commit_from_file(&model_path)
                    .context("Failed to load ONNX model")
            })() {
                Ok(s) => s,
                Err(e) => {
                    self.available
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    return Err(format!("Failed to load ONNX model for {}: {e}", self.config.id));
                }
            };

            Ok(LazyOnnx {
                session: Mutex::new(session),
                tokenizer,
                has_token_type_ids: self.config.has_token_type_ids,
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

    fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let model = self.ensure_loaded()?;
        let mut session = model
            .session
            .try_lock_for(std::time::Duration::from_secs(30))
            .ok_or_else(|| anyhow::anyhow!("Cross-encoder session lock timeout ({})", self.config.id))?;

        let mut scores = Vec::with_capacity(documents.len());

        for &doc in documents {
            let encoding = model
                .tokenizer
                .encode((query, doc), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed ({}): {e}", self.config.id))?;

            let tokens = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            let mut input_ids = vec![0i64; MAX_PAIR_LENGTH];
            let mut attention = vec![0i64; MAX_PAIR_LENGTH];
            let mut token_types = vec![0i64; MAX_PAIR_LENGTH];

            for (i, &token) in tokens.iter().take(MAX_PAIR_LENGTH).enumerate() {
                input_ids[i] = token as i64;
            }
            for (i, &mask) in attention_mask.iter().take(MAX_PAIR_LENGTH).enumerate() {
                attention[i] = mask as i64;
            }
            for (i, &tid) in type_ids.iter().take(MAX_PAIR_LENGTH).enumerate() {
                token_types[i] = tid as i64;
            }

            let input_ids_val = Value::from_array((vec![1, MAX_PAIR_LENGTH], input_ids))?;
            let attention_val = Value::from_array((vec![1, MAX_PAIR_LENGTH], attention))?;

            let outputs = if model.has_token_type_ids {
                let token_type_val =
                    Value::from_array((vec![1, MAX_PAIR_LENGTH], token_types))?;
                session.run(ort::inputs![
                    "input_ids" => &input_ids_val,
                    "attention_mask" => &attention_val,
                    "token_type_ids" => &token_type_val,
                ])?
            } else {
                session.run(ort::inputs![
                    "input_ids" => &input_ids_val,
                    "attention_mask" => &attention_val,
                ])?
            };

            let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
            let (_shape, data) = output_tensor;
            let score = data.first().copied().unwrap_or(0.0);
            scores.push(score);
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

/// Build the cross-encoder from environment variables. Default is the
/// legacy single-model ms-marco when neither `VELD_CROSS_ENCODER` nor
/// `VELD_CROSS_ENCODERS` is set.
///
/// `VELD_CROSS_ENCODERS` takes precedence when both are set.
pub fn load_cross_encoder_from_env() -> Arc<dyn CrossEncoderModel> {
    if let Ok(spec) = std::env::var("VELD_CROSS_ENCODERS") {
        match parse_ensemble_spec(&spec) {
            Ok(arc) => return arc,
            Err(e) => {
                tracing::warn!(
                    "VELD_CROSS_ENCODERS=`{spec}` invalid: {e}; falling back to default"
                );
            }
        }
    }

    if let Ok(id) = std::env::var("VELD_CROSS_ENCODER") {
        match lookup(id.trim()) {
            Ok(cfg) => return Arc::new(OnnxCrossEncoder::new(cfg)),
            Err(e) => {
                tracing::warn!(
                    "VELD_CROSS_ENCODER=`{id}` invalid: {e}; falling back to default"
                );
            }
        }
    }

    // Default is BGE-reranker-v2-m3 (set above as DEFAULT_CROSS_ENCODER_ID).
    let cfg = lookup(DEFAULT_CROSS_ENCODER_ID)
        .expect("default cross-encoder id must be registered as a built-in");
    Arc::new(OnnxCrossEncoder::new(cfg))
}

fn parse_ensemble_spec(spec: &str) -> Result<Arc<dyn CrossEncoderModel>> {
    let mut members: Vec<(Arc<dyn CrossEncoderModel>, f32)> = Vec::new();
    for raw in spec.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let (id, weight) = match raw.split_once(':') {
            Some((id, w)) => (id.trim(), w.trim().parse::<f32>().with_context(|| {
                format!("invalid weight in ensemble entry `{raw}`")
            })?),
            None => (raw, 1.0),
        };
        let cfg = lookup(id)?;
        members.push((Arc::new(OnnxCrossEncoder::new(cfg)), weight));
    }
    if members.is_empty() {
        anyhow::bail!("VELD_CROSS_ENCODERS produced no members");
    }
    if members.len() == 1 {
        // Single-member ensemble is just the underlying model.
        return Ok(members.into_iter().next().unwrap().0);
    }
    Ok(Arc::new(EnsembleCrossEncoder::new(members)))
}

/// Pre-warm a cross-encoder (single model or ensemble) by issuing a dummy
/// `score_pairs` call on a background thread. For [`OnnxCrossEncoder`] this
/// triggers the ~80 MB download and ONNX session build; for
/// [`EnsembleCrossEncoder`] it cascades into every member. Idempotent — the
/// underlying `OnceLock` guards against duplicate work.
///
/// Skipped when `VELD_LAZY_CROSS_ENCODER=1` is set.
pub fn prewarm(model: Arc<dyn CrossEncoderModel>) {
    let lazy = std::env::var("VELD_LAZY_CROSS_ENCODER")
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    if lazy {
        tracing::debug!("cross-encoder pre-warm skipped ({})", model.id());
        return;
    }
    let id = model.id().to_string();
    std::thread::Builder::new()
        .name(format!("cross-encoder-prewarm-{id}"))
        .spawn(move || match model.score_pairs("warmup", &["warmup"]) {
            Ok(_) => tracing::info!("Cross-encoder pre-warmed: {id}"),
            Err(e) => tracing::warn!(
                "Cross-encoder pre-warm failed for {id} (lazy fallback engaged): {e}"
            ),
        })
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

    let mut reader = resp.into_body().into_reader();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create file: {path:?}"))?;

    std::io::copy(&mut reader, &mut file)?;
    Ok(())
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
    fn registry_default_is_bge() {
        assert_eq!(DEFAULT_CROSS_ENCODER_ID, "bge-reranker-v2-m3");
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
            notes: "test fixture",
        });
        let cfg = lookup("test-custom-model").unwrap();
        assert_eq!(cfg.params_millions, 100);
    }

    #[test]
    fn parse_ensemble_spec_handles_weights() {
        let res = parse_ensemble_spec("ms-marco-MiniLM-L-6-v2:0.5,bge-reranker-v2-m3:1.5").unwrap();
        assert!(res.id().contains("0.50"));
        assert!(res.id().contains("1.50"));
    }

    #[test]
    fn parse_ensemble_spec_collapses_singleton() {
        let res = parse_ensemble_spec("ms-marco-MiniLM-L-6-v2").unwrap();
        assert_eq!(res.id(), "ms-marco-MiniLM-L-6-v2");
    }
}
