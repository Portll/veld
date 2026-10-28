//! snowflake-arctic-embed-xs through ONNX Runtime: 384-dimensional retrieval embeddings.
//!
//! Model: Snowflake/snowflake-arctic-embed-xs, Apache-2.0, pinned in `downloader.rs`. It is a
//! retrieval fine-tune of all-MiniLM-L6-v2: the same 6-layer, 384-wide BERT with the same uncased
//! WordPiece vocabulary and the same ONNX inputs, so a token costs what it costs MiniLM. Three
//! things differ, each read from the model's own files at the pinned revision:
//!
//! - **CLS pooling.** `1_Pooling/config.json` sets `pooling_mode_cls_token` (mean pooling off) and
//!   `modules.json` ends in `Normalize`: the sentence vector is the final hidden state of `[CLS]`,
//!   L2-normalised. MiniLM mean-pools.
//! - **A query instruction.** `config_sentence_transformers.json` declares one prompt, `query`,
//!   which is [`QUERY_PREFIX`]. The model card applies it to queries only; documents are bare.
//! - **A 512-token window** (`sentence_bert_config.json` `max_seq_length`, and
//!   `max_position_embeddings`), against the 256 veld runs MiniLM at.
//!
//! Selected with `VELD_EMBEDDER=arctic-embed-xs`. Configuration:
//! - `VELD_ARCTIC_MODEL_PATH`: directory holding `model_quantized.onnx` / `model.onnx` and
//!   `tokenizer.json` (default `~/.cache/veld/models/arctic-embed-xs`, which
//!   `model-fetch --arctic-xs` fills)
//! - `VELD_ARCTIC_USE_QUANTIZED`: `false` loads the fp32 `model.onnx` (90MB) instead of the int8
//!   `model_quantized.onnx` (23MB), the precision MiniLM runs at by default
//! - `VELD_EMBED_TIMEOUT_MS`, `VELD_LAZY_LOAD` and `VELD_ONNX_THREADS`, shared with MiniLM
//!
//! There is no hash-embedding fallback: a missing model or a failed inference is an error, so
//! nothing this embedder returns is a vector of the right shape and no meaning.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::{Mutex, MutexGuard};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

use super::minilm::OrtThreading;
use super::Embedder;

/// The id `VELD_EMBEDDER` selects this embedder by, the corpus marker records, and the recall
/// report names it with.
pub const ARCTIC_XS_ID: &str = "arctic-embed-xs";

/// The instruction the model was trained to see before a query: `prompts.query` in the pinned
/// `config_sentence_transformers.json`, trailing space included.
pub const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// `hidden_size` in `config.json`, and so the embedding dimension: CLS pooling has no projection.
const DIMENSION: usize = 384;

/// The model's window in tokens, `[CLS]` and `[SEP]` included.
const MAX_LENGTH: usize = 512;

/// Content tokens one window holds once `[CLS]` and `[SEP]` frame it.
const WINDOW_CONTENT: usize = MAX_LENGTH - 2;

/// Windows one text may span before it is refused, as for MiniLM: an unbounded loop over
/// caller-supplied text is a denial of service.
const MAX_WINDOWS: usize = 64;

const SESSION_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// What a query looks like to the model.
pub fn query_text(query: &str) -> String {
    format!("{QUERY_PREFIX}{query}")
}

/// Configuration for the arctic-embed-xs embedder
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArcticConfig {
    /// Path to the ONNX model file
    pub model_path: PathBuf,
    /// Path to tokenizer.json
    pub tokenizer_path: PathBuf,
    /// Whether `model_path` names the int8 quantised export
    pub use_quantized: bool,
    /// Inference time in milliseconds past which one embed is logged as slow
    pub embed_timeout_ms: u64,
}

impl Default for ArcticConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl ArcticConfig {
    /// Configuration from the environment.
    ///
    /// Model directory, in order: `VELD_ARCTIC_MODEL_PATH`; `./models/arctic-embed-xs`,
    /// `../models/arctic-embed-xs`; the download cache.
    pub fn from_env() -> Self {
        let use_quantized = std::env::var("VELD_ARCTIC_USE_QUANTIZED")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let model_filename = model_filename(use_quantized);

        let base_path = std::env::var("VELD_ARCTIC_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                [
                    PathBuf::from("./models/arctic-embed-xs"),
                    PathBuf::from("../models/arctic-embed-xs"),
                ]
                .into_iter()
                .find(|p| p.join(model_filename).exists())
                .unwrap_or_else(super::downloader::get_arctic_xs_models_dir)
            });

        Self {
            model_path: base_path.join(model_filename),
            tokenizer_path: base_path.join("tokenizer.json"),
            use_quantized,
            embed_timeout_ms: super::minilm::embed_timeout_ms(),
        }
    }

    /// Configuration with explicit paths (for testing or programmatic use)
    pub fn with_paths(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            model_path,
            tokenizer_path,
            use_quantized: true,
            embed_timeout_ms: 5000,
        }
    }
}

fn model_filename(use_quantized: bool) -> &'static str {
    if use_quantized {
        "model_quantized.onnx"
    } else {
        "model.onnx"
    }
}

/// The loaded session and the tokenizer that feeds it.
struct LazyArcticModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    cls_id: i64,
    sep_id: i64,
}

impl LazyArcticModel {
    fn new(config: &ArcticConfig, threading: OrtThreading) -> Result<Self> {
        tracing::info!(
            "Loading snowflake-arctic-embed-xs from {:?} (intra={}, inter={}, spinning={})",
            config.model_path,
            threading.intra,
            threading.inter,
            threading.spinning
        );

        let session = threading
            .apply(Session::builder().context("Failed to create session builder")?)?
            .commit_from_file(&config.model_path)
            .context("Failed to load arctic-embed-xs ONNX model")?;

        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load arctic-embed-xs tokenizer: {e}"))?;
        // tokenizer.json ships `truncation: 512` and batch padding. The windowing below is the only
        // place a long text is split, so the tokenizer must hand over every token, unpadded.
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to clear tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None);

        let special = |token: &str| -> Result<i64> {
            tokenizer
                .token_to_id(token)
                .map(i64::from)
                .with_context(|| format!("arctic-embed-xs tokenizer has no {token} token"))
        };
        let cls_id = special("[CLS]")?;
        let sep_id = special("[SEP]")?;

        tracing::info!("snowflake-arctic-embed-xs loaded");
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            cls_id,
            sep_id,
        })
    }

    /// Token ids of `text` without `[CLS]`/`[SEP]`; [`Self::frame`] adds them per window.
    fn content_ids(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("arctic-embed-xs tokenization failed: {e}"))?;
        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    /// `[CLS] content [SEP]`, the single-sequence template in tokenizer.json, so a text that fits
    /// one window is tokenized exactly as the model card's `tokenizer(text)` call does.
    fn frame(&self, content: &[i64]) -> Vec<i64> {
        let mut ids = Vec::with_capacity(content.len() + 2);
        ids.push(self.cls_id);
        ids.extend_from_slice(content);
        ids.push(self.sep_id);
        ids
    }

    fn lock(&self) -> Result<MutexGuard<'_, Session>> {
        self.session
            .try_lock_for(SESSION_LOCK_TIMEOUT)
            .with_context(|| {
                format!(
                    "arctic-embed-xs ONNX session lock timeout ({}s): a previous inference is stuck",
                    SESSION_LOCK_TIMEOUT.as_secs()
                )
            })
    }
}

/// Run framed sequences through the model in one batch and return each one's `[CLS]` row.
///
/// Sequences are right-padded to the longest with attention 0, so a padded position is never
/// attended to; a lone sequence is not padded at all.
fn run_cls(session: &mut Session, sequences: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
    let batch = sequences.len();
    let seq_len = sequences.iter().map(Vec::len).max().unwrap_or(0);
    anyhow::ensure!(batch > 0 && seq_len > 0, "empty batch");

    let mut input_ids = vec![0i64; batch * seq_len];
    let mut attention = vec![0i64; batch * seq_len];
    for (row, ids) in sequences.iter().enumerate() {
        let at = row * seq_len;
        input_ids[at..at + ids.len()].copy_from_slice(ids);
        attention[at..at + ids.len()].fill(1);
    }
    let token_type_ids = vec![0i64; batch * seq_len];

    let input_ids = Value::from_array((vec![batch, seq_len], input_ids))?;
    let attention = Value::from_array((vec![batch, seq_len], attention))?;
    let token_type_ids = Value::from_array((vec![batch, seq_len], token_type_ids))?;
    let outputs = session.run(ort::inputs![
        "input_ids" => &input_ids,
        "attention_mask" => &attention,
        "token_type_ids" => &token_type_ids,
    ])?;

    // last_hidden_state, [batch, seq_len, 384]: the graph's only output.
    let (_shape, hidden) = outputs[0].try_extract_tensor::<f32>()?;
    anyhow::ensure!(
        hidden.len() == batch * seq_len * DIMENSION,
        "arctic-embed-xs returned {} values for a [{batch}, {seq_len}, {DIMENSION}] output",
        hidden.len()
    );
    Ok((0..batch)
        .map(|row| {
            let at = row * seq_len * DIMENSION;
            hidden[at..at + DIMENSION].to_vec()
        })
        .collect())
}

/// Embed one text's content tokens, window by window.
fn embed_windowed(
    model: &LazyArcticModel,
    session: &mut Session,
    content: &[i64],
) -> Result<Vec<f32>> {
    let spans = super::minilm::window_spans(content.len(), WINDOW_CONTENT);
    anyhow::ensure!(
        spans.len() <= MAX_WINDOWS,
        "input is {} tokens ({} windows of {WINDOW_CONTENT}), over the {MAX_WINDOWS} window limit. \
         Refusing rather than embedding a prefix and reporting it as the whole — chunk it upstream.",
        content.len(),
        spans.len()
    );
    let mut windows = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        let cls = run_cls(session, &[model.frame(&content[start..end])])?
            .into_iter()
            .next()
            .context("arctic-embed-xs returned no row for a window")?;
        windows.push((cls, end - start));
    }
    Ok(pool_windows(windows))
}

/// One vector for a text from its windows' `[CLS]` vectors.
///
/// A CLS vector reads one window, so unlike a mean over tokens it does not decompose across
/// windows. A text past the window gets the mean of its windows' unit vectors weighted by the
/// content tokens each holds, every token counting once, then renormalised. Each window is framed
/// by its own `[CLS]`/`[SEP]` and positions restart, so it is an average of readings, not one
/// reading of the whole. A text that fits one window takes its CLS vector, normalised, unchanged.
fn pool_windows(windows: Vec<(Vec<f32>, usize)>) -> Vec<f32> {
    if windows.len() == 1 {
        return windows
            .into_iter()
            .next()
            .map(|(v, _)| normalized(v))
            .unwrap_or_else(|| vec![0.0; DIMENSION]);
    }
    let mut pooled = vec![0.0f32; DIMENSION];
    for (v, tokens) in windows {
        let weight = tokens.max(1) as f32;
        for (p, x) in pooled.iter_mut().zip(normalized(v)) {
            *p += weight * x;
        }
    }
    normalized(pooled)
}

/// L2-normalise, with NaN/Inf zeroed first; a zero vector stays zero.
fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    for x in v.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// snowflake-arctic-embed-xs embedder with ONNX Runtime and lazy model loading.
pub struct ArcticEmbedder {
    config: ArcticConfig,
    threading: OrtThreading,
    lazy_model: OnceLock<Result<Arc<LazyArcticModel>, String>>,
}

impl ArcticEmbedder {
    /// Create the embedder; the model loads on first use unless `VELD_LAZY_LOAD=false`.
    ///
    /// Missing model files are fetched only with `VELD_AUTO_DOWNLOAD_MODELS=true` and without
    /// `VELD_OFFLINE`; otherwise this is an error naming `model-fetch --arctic-xs`.
    pub fn new(config: ArcticConfig) -> Result<Self> {
        Self::with_threading(config, OrtThreading::resolve())
    }

    /// [`Self::new`] with the session threading given rather than resolved, so a shared
    /// instance is built from exactly the values its spec was keyed on.
    pub fn with_threading(config: ArcticConfig, threading: OrtThreading) -> Result<Self> {
        let offline = super::offline_mode_enabled();
        super::minilm::ensure_ort_runtime(offline)
            .context("arctic-embed-xs needs the ONNX Runtime")?;

        let present = |c: &ArcticConfig| c.model_path.exists() && c.tokenizer_path.exists();
        let missing = |c: &ArcticConfig| {
            format!(
                "arctic-embed-xs model files not found ({:?}, {:?})",
                c.model_path, c.tokenizer_path
            )
        };

        let mut config = config;
        if !present(&config) {
            if offline {
                anyhow::bail!("{} and VELD_OFFLINE=true", missing(&config));
            }
            if !super::auto_download_models_enabled() {
                anyhow::bail!(
                    "{}. Fetch them with `model-fetch --arctic-xs`, or set \
                     VELD_AUTO_DOWNLOAD_MODELS=true",
                    missing(&config)
                );
            }
            let last_decile = std::sync::atomic::AtomicU64::new(u64::MAX);
            let progress: super::downloader::ProgressCallback =
                Arc::new(move |downloaded: u64, total: u64| {
                    if total == 0 {
                        return;
                    }
                    let decile = downloaded * 10 / total;
                    if last_decile.swap(decile, std::sync::atomic::Ordering::Relaxed) != decile {
                        tracing::info!(
                            "Downloading arctic-embed-xs: {}% ({downloaded}/{total})",
                            decile * 10
                        );
                    }
                });
            let dir = super::downloader::download_arctic_xs_models_internal(
                Some(progress),
                config.use_quantized,
            )
            .context("arctic-embed-xs model download failed")?;
            config = ArcticConfig {
                model_path: dir.join(model_filename(config.use_quantized)),
                tokenizer_path: dir.join("tokenizer.json"),
                ..config
            };
            anyhow::ensure!(
                present(&config),
                "{} after a download that reported success",
                missing(&config)
            );
        }

        super::model_tape::register_model(
            super::model_tape::Model::Embedder,
            &[&config.model_path, &config.tokenizer_path],
        );
        let embedder = Self {
            config,
            threading,
            lazy_model: OnceLock::new(),
        };
        if !super::minilm::lazy_load_enabled() {
            embedder.ensure_model_loaded()?;
        }
        Ok(embedder)
    }

    /// True once a load was attempted and failed; the shared registry then builds a new
    /// instance instead of handing this one out.
    pub fn model_load_failed(&self) -> bool {
        matches!(self.lazy_model.get(), Some(Err(_)))
    }

    fn ensure_model_loaded(&self) -> Result<&Arc<LazyArcticModel>> {
        let loaded = self.lazy_model.get_or_init(|| {
            LazyArcticModel::new(&self.config, self.threading)
                .map(Arc::new)
                .map_err(|e| format!("{e:#}"))
        });
        loaded
            .as_ref()
            .map_err(|e| anyhow::anyhow!("Failed to load arctic-embed-xs: {e}"))
    }

    /// Embed `seen`, the exact text the model reads (a query already carries its prefix).
    fn embed_seen(&self, seen: &str) -> Result<Vec<f32>> {
        if let Some(taped) = super::model_tape::embed(seen, || self.embed_live(seen)) {
            return taped;
        }
        let start = std::time::Instant::now();
        let result = self.embed_live(seen);
        let elapsed = start.elapsed().as_secs_f64();
        record_outcome("arctic_onnx", elapsed, result.is_ok());
        if result.is_ok() && elapsed * 1000.0 > self.config.embed_timeout_ms as f64 {
            tracing::warn!(
                "arctic-embed-xs inference took {:.0}ms (threshold: {}ms)",
                elapsed * 1000.0,
                self.config.embed_timeout_ms
            );
        }
        result
    }

    fn embed_live(&self, seen: &str) -> Result<Vec<f32>> {
        let model = self.ensure_model_loaded()?;
        let content = model.content_ids(seen)?;
        let mut session = model.lock()?;
        embed_windowed(model, &mut session, &content)
    }
}

fn record_outcome(mode: &str, seconds: f64, ok: bool) {
    if ok {
        crate::metrics::EMBEDDING_GENERATE_DURATION
            .with_label_values(&[mode])
            .observe(seconds);
    }
    crate::metrics::EMBEDDING_GENERATE_TOTAL
        .with_label_values(&[mode, if ok { "success" } else { "failure" }])
        .inc();
}

impl Embedder for ArcticEmbedder {
    fn id(&self) -> &str {
        ARCTIC_XS_ID
    }

    fn model_id(&self) -> &str {
        ARCTIC_XS_ID
    }

    fn dimension(&self) -> usize {
        DIMENSION
    }

    /// Document side: the text as it is, no prefix.
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; DIMENSION]);
        }
        self.embed_seen(text)
    }

    /// Query side: [`QUERY_PREFIX`] first, as the model was trained.
    fn encode_for_query(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; DIMENSION]);
        }
        self.embed_seen(&query_text(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_query_prefix_is_the_pinned_prompt_and_only_queries_carry_it() {
        assert_eq!(
            QUERY_PREFIX,
            "Represent this sentence for searching relevant passages: "
        );
        assert_eq!(
            query_text("what is snowflake?"),
            "Represent this sentence for searching relevant passages: what is snowflake?"
        );
    }

    #[test]
    fn one_window_is_its_cls_vector_normalised() {
        let cls: Vec<f32> = (0..DIMENSION).map(|i| i as f32 - 100.0).collect();
        let pooled = pool_windows(vec![(cls.clone(), 7)]);
        assert_eq!(pooled, normalized(cls));
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn windows_are_weighted_by_the_tokens_they_hold() {
        let mut a = vec![0.0f32; DIMENSION];
        a[0] = 5.0;
        let mut b = vec![0.0f32; DIMENSION];
        b[1] = 0.5;
        // Unit vectors along two axes, 3 tokens against 1: the result leans 3:1 towards `a`
        // whatever the raw norms were.
        let pooled = pool_windows(vec![(a, 3), (b, 1)]);
        assert!(
            (pooled[0] / pooled[1] - 3.0).abs() < 1e-5,
            "{:?}",
            &pooled[..2]
        );
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn non_finite_values_never_reach_an_embedding() {
        let mut v = vec![0.0f32; DIMENSION];
        v[0] = f32::NAN;
        v[1] = f32::INFINITY;
        v[2] = 2.0;
        let n = normalized(v);
        assert!(n.iter().all(|x| x.is_finite()));
        assert_eq!(n[2], 1.0);
        assert!(normalized(vec![0.0; DIMENSION]).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn a_window_holds_the_model_window_less_its_two_special_tokens() {
        assert_eq!(WINDOW_CONTENT, 510);
        let spans = super::super::minilm::window_spans(1021, WINDOW_CONTENT);
        assert_eq!(spans, vec![(0, 510), (510, 1020), (1020, 1021)]);
    }

    /// Missing model files refuse, rather than degrade to hash vectors as MiniLM can.
    #[test]
    fn missing_model_files_are_an_error_not_a_degraded_embedder() {
        if super::super::auto_download_models_enabled() {
            eprintln!("skipped: VELD_AUTO_DOWNLOAD_MODELS would fetch the model");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let config = ArcticConfig::with_paths(
            dir.path().join("model_quantized.onnx"),
            dir.path().join("tokenizer.json"),
        );
        assert!(ArcticEmbedder::new(config).is_err());
    }

    /// LIVE: `model-fetch --onnxruntime --arctic-xs`, then
    /// `VELD_LIVE_ARCTIC=1 VELD_ARCTIC_USE_QUANTIZED=false cargo test -- --ignored arctic_matches`.
    /// The model card's sentence-transformers scores for its own example, reproduced through this
    /// module's tokenizer framing, CLS pooling, normalisation and query prefix on the fp32 export.
    #[test]
    #[ignore]
    fn arctic_matches_the_model_card_scores() {
        if std::env::var("VELD_LIVE_ARCTIC").as_deref() != Ok("1") {
            eprintln!("skipped: VELD_LIVE_ARCTIC!=1");
            return;
        }
        let embedder = ArcticEmbedder::new(ArcticConfig::from_env()).expect("model files");
        let docs = embedder
            .encode_batch(&["The Data Cloud!", "Mexico City of Course!"])
            .unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        for (query, want) in [
            ("what is snowflake?", [0.57515126f32, 0.45798576]),
            ("Where can I get the best tacos?", [0.5044898, 0.5636022]),
        ] {
            let q = embedder.encode_for_query(query).unwrap();
            for (doc, w) in docs.iter().zip(want) {
                let got = dot(&q, doc);
                assert!((got - w).abs() < 2e-3, "{query}: {got} vs card {w}");
            }
        }
        // A text's batched vector is the one it gets alone.
        assert_eq!(embedder.encode("The Data Cloud!").unwrap(), docs[0]);
    }
}
