//! Cross-encoder reranker using ms-marco-MiniLM-L-6-v2
//!
//! Unlike bi-encoders that encode query and document separately, cross-encoders
//! encode the (query, document) pair jointly, allowing token-level attention
//! between them. This produces significantly better relevance scores at the cost
//! of higher latency (cannot pre-compute document embeddings).
//!
//! Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB ONNX)
//! Input: [CLS] query [SEP] document [SEP]
//! Output: single relevance logit per pair
//!
//! Auto-downloads to ~/.cache/veld/cross-encoder/ on first use.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

/// HuggingFace model URLs — tracking `main`. The previous pin to commit
/// `5b0d1d1b7c8a21c04e5e0168e09ef62faebdcca0` started returning "Entry not
/// found" after an upstream restructure, silently breaking download +
/// rerank. `main` always resolves to the current head of the repo.
const CROSS_ENCODER_MODEL_URL: &str =
    "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx";
const CROSS_ENCODER_TOKENIZER_URL: &str =
    "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json";

/// Maximum input sequence length (query + document tokens)
const MAX_PAIR_LENGTH: usize = 512;

struct LazyModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

/// Cross-encoder model for query-document relevance scoring
pub struct CrossEncoder {
    model: OnceLock<Result<LazyModel, String>>,
    available: std::sync::atomic::AtomicBool,
}

impl Default for CrossEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEncoder {
    pub fn new() -> Self {
        Self {
            model: OnceLock::new(),
            available: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Check if the cross-encoder model is available (downloaded and loadable)
    pub fn is_available(&self) -> bool {
        self.available.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the model directory
    fn model_dir() -> PathBuf {
        super::downloader::get_cache_dir().join("cross-encoder")
    }

    /// Ensure model files are downloaded
    fn ensure_downloaded() -> Result<(PathBuf, PathBuf)> {
        let dir = Self::model_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create cross-encoder model dir: {dir:?}"))?;

        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        if model_path.exists() && tokenizer_path.exists() {
            return Ok((model_path, tokenizer_path));
        }

        // Check offline mode
        if std::env::var("VELD_OFFLINE").is_ok_and(|v| v == "true" || v == "1") {
            anyhow::bail!(
                "Cross-encoder model not found and VELD_OFFLINE=true. \
                 Download manually to {:?}",
                dir
            );
        }

        tracing::info!("Downloading cross-encoder model (ms-marco-MiniLM-L-6-v2)...");

        if !model_path.exists() {
            download_file(CROSS_ENCODER_MODEL_URL, &model_path)
                .context("Failed to download cross-encoder model")?;
        }
        if !tokenizer_path.exists() {
            download_file(CROSS_ENCODER_TOKENIZER_URL, &tokenizer_path)
                .context("Failed to download cross-encoder tokenizer")?;
        }

        tracing::info!("Cross-encoder model downloaded successfully");
        Ok((model_path, tokenizer_path))
    }

    /// Load the model lazily on first use
    fn ensure_loaded(&self) -> Result<&LazyModel> {
        let result = self.model.get_or_init(|| {
            let (model_path, tokenizer_path) = match Self::ensure_downloaded() {
                Ok(paths) => paths,
                Err(e) => {
                    tracing::warn!("Cross-encoder unavailable: {e}");
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
                    return Err(format!("Failed to load cross-encoder tokenizer: {e}"));
                }
            };

            // Use same thread config as MiniLM
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
                    return Err(format!("Failed to load cross-encoder ONNX model: {e}"));
                }
            };

            Ok(LazyModel {
                session: Mutex::new(session),
                tokenizer,
            })
        });

        match result {
            Ok(model) => Ok(model),
            Err(msg) => Err(anyhow::anyhow!("{}", msg)),
        }
    }

    /// Score a batch of (query, document) pairs
    ///
    /// Returns relevance scores (higher = more relevant). Scores are raw logits,
    /// not normalized to [0,1] — use sigmoid if probability is needed.
    pub fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let model = self.ensure_loaded()?;
        let mut session = model
            .session
            .try_lock_for(std::time::Duration::from_secs(30))
            .ok_or_else(|| anyhow::anyhow!("Cross-encoder session lock timeout"))?;

        let mut scores = Vec::with_capacity(documents.len());

        for &doc in documents {
            // Encode (query, document) pair — tokenizer handles [CLS] q [SEP] d [SEP]
            let encoding = model
                .tokenizer
                .encode((query, doc), true)
                .map_err(|e| anyhow::anyhow!("Cross-encoder tokenization failed: {e}"))?;

            let tokens = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            // Truncate/pad to MAX_PAIR_LENGTH
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

            // Create input tensors
            let input_ids_val = Value::from_array((vec![1, MAX_PAIR_LENGTH], input_ids))?;
            let attention_val = Value::from_array((vec![1, MAX_PAIR_LENGTH], attention))?;
            let token_type_val = Value::from_array((vec![1, MAX_PAIR_LENGTH], token_types))?;

            // Run inference
            let outputs = session.run(ort::inputs![
                "input_ids" => &input_ids_val,
                "attention_mask" => &attention_val,
                "token_type_ids" => &token_type_val,
            ])?;

            // Extract relevance logit — output shape is (1, 1)
            let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
            let (_shape, data) = output_tensor;
            let score = data.first().copied().unwrap_or(0.0);

            scores.push(score);
        }

        Ok(scores)
    }
}

/// Simple file download (no checksum verification for initial release)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dir_creation() {
        let dir = CrossEncoder::model_dir();
        assert!(dir.to_string_lossy().contains("cross-encoder"));
    }
}
