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
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

/// Maximum input sequence length (query + document tokens combined).
const MAX_PAIR_LENGTH: usize = 512;

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

/// Configuration for an ONNX-backed cross-encoder.
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Short identifier used in logs, cache paths, and env vars.
    pub id: String,
    /// HuggingFace URL for the ONNX model.
    pub model_url: String,
    /// HuggingFace URL for the tokenizer.
    pub tokenizer_url: String,
    /// Whether the model's ONNX graph accepts `token_type_ids`. BERT-style
    /// models do; XLM-RoBERTa-style models (BGE, Jina) do not.
    pub has_token_type_ids: bool,
}

impl CrossEncoderConfig {
    /// MS MARCO MiniLM-L-6-v2 (legacy default, 22M params, BERT-style).
    pub fn ms_marco_minilm_l6_v2() -> Self {
        Self {
            id: "ms-marco-MiniLM-L-6-v2".into(),
            model_url:
                "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: true,
        }
    }

    /// BGE reranker v2 m3 (568M params, multilingual, XLM-RoBERTa-style).
    pub fn bge_reranker_v2_m3() -> Self {
        Self {
            id: "bge-reranker-v2-m3".into(),
            model_url:
                "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/onnx/model.onnx".into(),
            tokenizer_url:
                "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/tokenizer.json".into(),
            has_token_type_ids: false,
        }
    }

    /// Mixedbread mxbai-rerank-large v1 (435M params, QA + retrieval, BERT-style).
    pub fn mxbai_rerank_large_v1() -> Self {
        Self {
            id: "mxbai-rerank-large-v1".into(),
            model_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: true,
        }
    }

    /// Jina Reranker v2 base multilingual (278M, XLM-RoBERTa-style).
    pub fn jina_reranker_v2_base() -> Self {
        Self {
            id: "jina-reranker-v2-base-multilingual".into(),
            model_url:
                "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/tokenizer.json"
                    .into(),
            has_token_type_ids: false,
        }
    }

    /// Resolve a built-in config by id. Used by env-var parsing.
    pub fn from_id(id: &str) -> Result<Self> {
        match id {
            "ms-marco-MiniLM-L-6-v2" => Ok(Self::ms_marco_minilm_l6_v2()),
            "bge-reranker-v2-m3" => Ok(Self::bge_reranker_v2_m3()),
            "mxbai-rerank-large-v1" => Ok(Self::mxbai_rerank_large_v1()),
            "jina-reranker-v2-base-multilingual" => Ok(Self::jina_reranker_v2_base()),
            other => anyhow::bail!(
                "unknown cross-encoder id: {other} (built-ins: ms-marco-MiniLM-L-6-v2, \
                 bge-reranker-v2-m3, mxbai-rerank-large-v1, jina-reranker-v2-base-multilingual)"
            ),
        }
    }

    /// Sanitise the id for filesystem use (so e.g. `BAAI/bge-…` doesn't
    /// embed a directory separator in the cache path).
    fn cache_key(&self) -> String {
        self.id.replace(['/', '\\'], "_")
    }
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
        match CrossEncoderConfig::from_id(id.trim()) {
            Ok(cfg) => return Arc::new(OnnxCrossEncoder::new(cfg)),
            Err(e) => {
                tracing::warn!(
                    "VELD_CROSS_ENCODER=`{id}` invalid: {e}; falling back to default"
                );
            }
        }
    }

    Arc::new(OnnxCrossEncoder::new(
        CrossEncoderConfig::ms_marco_minilm_l6_v2(),
    ))
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
        let cfg = CrossEncoderConfig::from_id(id)?;
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
        let cfg = CrossEncoderConfig::bge_reranker_v2_m3();
        let key = cfg.cache_key();
        assert!(!key.contains('/'));
        assert!(!key.contains('\\'));
    }

    #[test]
    fn from_id_resolves_built_ins() {
        assert!(CrossEncoderConfig::from_id("ms-marco-MiniLM-L-6-v2").is_ok());
        assert!(CrossEncoderConfig::from_id("bge-reranker-v2-m3").is_ok());
        assert!(CrossEncoderConfig::from_id("does-not-exist").is_err());
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
