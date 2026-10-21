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
//! - Default (neither set): mxbai-rerank-xsmall-v1.
//!
//! The env vars are the process-wide choice. On top of it, the per-user `rerank_model`
//! dial (`crate::memory::dials::CognitiveDials`) selects one registry id at runtime;
//! [`load_cross_encoder`] resolves a dial value over the env. The dial is `None` by
//! default, so the shipped pipeline is unchanged until someone turns it.
//!
//! Files cache under `~/.cache/veld/cross-encoder/<sanitised-id>/`.
//!
//! # Speed
//!
//! - `VELD_ONNX_THREADS`: intra-op threads per session (default: the core count, at most 4).
//! - `VELD_CROSS_ENCODER_BATCH=pair|bucket`: run each pair alone, or batch pairs of similar
//!   token length (default `bucket`).
//! - `VELD_CROSS_ENCODER_SESSIONS`: sessions one model may open for concurrent reranks
//!   (default 2, 1..=8). Each holds its own copy of the weights; extras open on contention.
//! - `VELD_CROSS_ENCODER_CACHE`: pair scores one reranker remembers (default 4096, 0 disables).

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

/// Most pairs one forward pass carries.
const MAX_BATCH: usize = 16;

/// How much longer than the shortest pair of a batch another pair may be and still join it:
/// a quarter of that length, and never less than [`BATCH_SLACK_MIN_TOKENS`]. Every pair is
/// padded to the batch's longest, so a wider band trades fewer passes for more padding.
const BATCH_SLACK_FRACTION: usize = 4;
const BATCH_SLACK_MIN_TOKENS: usize = 8;

/// Sessions a cross-encoder may open (`VELD_CROSS_ENCODER_SESSIONS`, 1..=8). Each holds its
/// own copy of the weights, 284 MB for mxbai-rerank-xsmall-v1, so a second one opens only when
/// two recalls rerank at the same moment.
const DEFAULT_SESSIONS: usize = 2;

/// Pair scores a cross-encoder remembers (`VELD_CROSS_ENCODER_CACHE`, 0 disables).
/// `proactive_context` reranks overlapping candidates for similar queries on every prompt.
const DEFAULT_PAIR_CACHE: usize = 4096;

/// Intra-op threads per session. `VELD_ONNX_THREADS`, or `SHODH_ONNX_THREADS`, which the recall
/// harness pins to 1 so repeat passes score identically; otherwise the core count, at most 4.
/// Apple Silicon used to get 1 here to leave room for Metal or Neural Engine providers, but
/// none is registered, so the CPU provider ran single-threaded.
fn intra_threads() -> usize {
    std::env::var("VELD_ONNX_THREADS")
        .or_else(|_| std::env::var("SHODH_ONNX_THREADS"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .min(4)
        })
}

fn env_count(name: &str, default: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
        .min(max)
}

/// How pairs reach the model: `bucket` (the default) batches pairs of similar token length,
/// `pair` runs each pair alone (`VELD_CROSS_ENCODER_BATCH`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BatchMode {
    Pair,
    Bucket,
}

fn batch_mode() -> BatchMode {
    match std::env::var("VELD_CROSS_ENCODER_BATCH").as_deref() {
        Ok("pair") => BatchMode::Pair,
        _ => BatchMode::Bucket,
    }
}

fn build_session(model_path: &std::path::Path, threads: usize) -> Result<Session> {
    Session::builder()
        .context("Failed to create session builder")?
        .with_intra_threads(threads)
        .context("Failed to set thread count")?
        .commit_from_file(model_path)
        .context("Failed to load ONNX model")
}

/// Default cross-encoder id when no env var is set. mxbai-rerank-xsmall-v1
/// is the smallest 2024-vintage reranker that meaningfully beats the
/// legacy ms-marco-MiniLM — ~280 MB download, 70 M params, modern QA +
/// retrieval training. Picks the size/quality sweet spot for the typical
/// veld install: fast to download, fast at query time, well above the
/// ms-marco floor. Swap up to `bge-reranker-v2-m3` (568 M) for the best
/// quality, or stack via `VELD_CROSS_ENCODERS=…`.
const DEFAULT_CROSS_ENCODER_ID: &str = "mxbai-rerank-xsmall-v1";

/// Registry id of the legacy reranker, pinned so the recall gate has a parity arm against
/// upstream shodh-memory (which measured under this model). Select it per user with the
/// `rerank_model` dial.
pub const MS_MARCO_MINILM_L6_ID: &str = "ms-marco-MiniLM-L-6-v2";

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

    /// Fetch whatever this model needs from the network, without building an
    /// ONNX session. This is the only part of loading that [`prewarm`] does off
    /// the caller's thread — see the pre-warm worker's notes for why the session
    /// build may not run there. Default: nothing to fetch.
    fn prefetch(&self) -> Result<()> {
        Ok(())
    }
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
    /// SHA-256 of the model file. With a checksum the URL names a fixed revision, a
    /// download is verified before it lands, and a cached file is verified before it
    /// loads. `None` accepts whatever the URL serves.
    pub model_sha256: Option<String>,
    /// SHA-256 of the tokenizer file; as `model_sha256`.
    pub tokenizer_sha256: Option<String>,
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
    /// SPDX identifier of the licence the model's weights are distributed under, as the
    /// model card declares it at the pinned revision. Recorded with the pin so a
    /// deployment can tell at a glance which rerankers it may ship.
    pub license: &'static str,
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
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/resolve/b5c6e9da73abc3711f593f705371cdbe9e0fe422/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1/resolve/b5c6e9da73abc3711f593f705371cdbe9e0fe422/tokenizer.json"
                    .into(),
            model_sha256: Some(
                "536b7183f8964dd212c4e2f4029fcc595cfc03ac233aba03d293546aaa5a875b".into(),
            ),
            tokenizer_sha256: Some(
                "305674b4d785287feecfb5f73f24aa75e9b57c87c579cfe24fbd207987d4b4c4".into(),
            ),
            has_token_type_ids: true,
            params_millions: 70,
            download_mb: 280,
            year: 2024,
            rerank_top_k: 20,
            notes:
                "BERT; smallest current reranker. ~3x MiniLM but trained on modern QA + retrieval; \
                 the cheap upgrade path off MS-MARCO-MiniLM.",
            license: "apache-2.0",
        },
        CrossEncoderConfig {
            id: "mxbai-rerank-base-v1".into(),
            model_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1/resolve/main/tokenizer.json"
                    .into(),
            model_sha256: None,
            tokenizer_sha256: None,
            has_token_type_ids: true,
            params_millions: 184,
            download_mb: 740,
            year: 2024,
            rerank_top_k: 12,
            notes:
                "BERT; mid-size mxbai variant. Quality close to bge-reranker-v2-m3 at a third the cost.",
            license: "apache-2.0",
        },
        CrossEncoderConfig {
            id: "jina-reranker-v1-tiny-en".into(),
            model_url:
                "https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/resolve/main/tokenizer.json"
                    .into(),
            model_sha256: None,
            tokenizer_sha256: None,
            has_token_type_ids: false,
            params_millions: 33,
            download_mb: 130,
            year: 2024,
            rerank_top_k: 20,
            notes:
                "JinaBERT (English only); modern drop-in replacement for ms-marco-MiniLM-L-6-v2 \
                 in the same size class. ~50% larger than MiniLM, 3 years newer, broader training mix.",
            license: "apache-2.0",
        },
        CrossEncoderConfig {
            id: "jina-reranker-v1-turbo-en".into(),
            model_url:
                "https://huggingface.co/jinaai/jina-reranker-v1-turbo-en/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/jinaai/jina-reranker-v1-turbo-en/resolve/main/tokenizer.json"
                    .into(),
            model_sha256: None,
            tokenizer_sha256: None,
            has_token_type_ids: false,
            params_millions: 37,
            download_mb: 150,
            year: 2024,
            rerank_top_k: 20,
            notes:
                "JinaBERT (English only); slightly larger sibling of jina-reranker-v1-tiny-en. \
                 The mini-class option for English-only deployments wanting a small accuracy bump.",
            license: "apache-2.0",
        },
        // Pinned like the default: upstream shodh-memory measured its LoCoMo numbers under
        // this model, so it is the parity arm of the recall gate (`rerank_model` dial), and
        // a parity arm whose files can move is not one.
        CrossEncoderConfig {
            id: MS_MARCO_MINILM_L6_ID.into(),
            model_url:
                "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/233902d25c440f23af6f7d6e94d2946bac0bee0a/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/233902d25c440f23af6f7d6e94d2946bac0bee0a/tokenizer.json"
                    .into(),
            model_sha256: Some(
                "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a".into(),
            ),
            tokenizer_sha256: Some(
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66".into(),
            ),
            has_token_type_ids: true,
            params_millions: 22,
            download_mb: 87,
            year: 2021,
            rerank_top_k: 20,
            notes: "BERT-style; trained on MS MARCO web-search passages. Fast, narrow domain.",
            license: "apache-2.0",
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
            model_sha256: None,
            tokenizer_sha256: None,
            has_token_type_ids: false,
            params_millions: 568,
            download_mb: 2270,
            year: 2024,
            rerank_top_k: 8,
            notes:
                "XLM-RoBERTa-style; multilingual + multi-task (QA, retrieval, conversational). \
                 Best quality but biggest download (2.3 GB across model.onnx + model.onnx_data). \
                 Source: hooman650/bge-reranker-v2-m3-onnx-o4 (Olive O4 CPU-optimised, apache-2.0).",
            license: "apache-2.0",
        },
        CrossEncoderConfig {
            id: "mxbai-rerank-large-v1".into(),
            model_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/resolve/main/tokenizer.json"
                    .into(),
            model_sha256: None,
            tokenizer_sha256: None,
            has_token_type_ids: true,
            params_millions: 435,
            download_mb: 1740,
            year: 2024,
            rerank_top_k: 8,
            notes: "BERT-style; trained on QA + retrieval pairs. Strong on factoid queries.",
            license: "apache-2.0",
        },
        CrossEncoderConfig {
            id: "jina-reranker-v2-base-multilingual".into(),
            model_url:
                "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/onnx/model.onnx"
                    .into(),
            tokenizer_url:
                "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/resolve/main/tokenizer.json"
                    .into(),
            model_sha256: None,
            tokenizer_sha256: None,
            has_token_type_ids: false,
            params_millions: 278,
            download_mb: 1110,
            year: 2024,
            rerank_top_k: 10,
            notes:
                "XLM-RoBERTa-style; multilingual; late-interaction-influenced training. Mid-size sweet spot.",
            // Non-commercial: selectable, never a default.
            license: "cc-by-nc-4.0",
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

/// One download lock per cache directory, shared by every [`OnnxCrossEncoder`] that
/// resolves to it: the registry can hand out two instances of one id (the env-selected
/// model and a dial-selected one, or the harness's tape filler), and they must not fetch
/// the same files at once.
static DOWNLOAD_LOCKS: LazyLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn download_lock(dir: &std::path::Path) -> Arc<Mutex<()>> {
    Arc::clone(
        DOWNLOAD_LOCKS
            .lock()
            .entry(dir.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(()))),
    )
}

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
    /// Built at load; every call can fall back to waiting for it.
    primary: Mutex<Session>,
    /// Opened one at a time, the first time every open session is busy.
    extra: Vec<OnceLock<Mutex<Session>>>,
    /// Held while an extra session is built, so two callers never build the same slot.
    growing: Mutex<()>,
    /// Set when building an extra session failed, so contended calls stop retrying it.
    grow_failed: AtomicBool,
    model_path: PathBuf,
    threads: usize,
    tokenizer: Tokenizer,
    /// Whether the loaded ONNX graph accepts `token_type_ids`. Detected
    /// from `session.inputs` at load time — the registry's
    /// `has_token_type_ids` config field is now only an unused hint.
    has_token_type_ids: bool,
    /// Output head strategy, detected from `session.outputs[0]`'s shape.
    output_head: OutputHead,
}

impl LazyOnnx {
    /// A session to run on: a free open one, else a newly opened one while the pool has
    /// room, else the primary once it frees.
    fn checkout(&self, id: &str) -> Result<parking_lot::MutexGuard<'_, Session>> {
        if let Some(guard) = self.primary.try_lock() {
            return Ok(guard);
        }
        for slot in &self.extra {
            if let Some(guard) = slot.get().and_then(|s| s.try_lock()) {
                return Ok(guard);
            }
        }
        if let Some(guard) = self.open_extra(id) {
            return Ok(guard);
        }
        self.primary
            .try_lock_for(std::time::Duration::from_secs(30))
            .ok_or_else(|| anyhow::anyhow!("Cross-encoder session lock timeout ({id})"))
    }

    fn open_extra(&self, id: &str) -> Option<parking_lot::MutexGuard<'_, Session>> {
        if self.grow_failed.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let _growing = self.growing.try_lock()?;
        let slot = self.extra.iter().find(|s| s.get().is_none())?;
        match build_session(&self.model_path, self.threads) {
            Ok(session) => {
                let _ = slot.set(Mutex::new(session));
                let open = 1 + self.extra.iter().filter(|s| s.get().is_some()).count();
                crate::metrics::CROSS_ENCODER_SESSIONS
                    .with_label_values(&[id])
                    .set(open as i64);
                tracing::info!(
                    "cross-encoder `{id}`: opened session {open} for concurrent reranks"
                );
                slot.get().and_then(|s| s.try_lock())
            }
            Err(e) => {
                self.grow_failed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    "cross-encoder `{id}`: could not open another session, concurrent reranks \
                     will wait for the first: {e:#}"
                );
                None
            }
        }
    }
}

/// One (query, document) pair, tokenized and cut to its own length.
struct EncodedPair {
    ids: Vec<i64>,
    mask: Vec<i64>,
    types: Vec<i64>,
}

impl EncodedPair {
    fn len(&self) -> usize {
        self.ids.len()
    }
}

/// Group pair indices, already sorted by token length, into batches whose longest pair is
/// within the slack of its shortest.
fn length_buckets(order: &[usize], lens: &[usize]) -> Vec<Vec<usize>> {
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    for &i in order {
        let joins = buckets.last().is_some_and(|b| {
            let shortest = lens[b[0]];
            let slack = (shortest / BATCH_SLACK_FRACTION).max(BATCH_SLACK_MIN_TOKENS);
            b.len() < MAX_BATCH && lens[i] <= shortest + slack
        });
        match buckets.last_mut() {
            Some(b) if joins => b.push(i),
            _ => buckets.push(vec![i]),
        }
    }
    buckets
}

/// Cache key for a pair. The query's byte length leads, so no two distinct pairs share a key
/// whatever characters either text contains.
fn pair_key(query: &str, document: &str) -> String {
    format!("{}:{query}{document}", query.len())
}

/// Cross-encoder backed by a single ONNX model.
pub struct OnnxCrossEncoder {
    config: CrossEncoderConfig,
    model: OnceLock<Result<LazyOnnx, String>>,
    available: AtomicBool,
    /// Scores this reranker has already computed. Per instance, so a fresh system (each
    /// recall-gate repeat builds one) starts cold and its latency is the model's.
    pair_cache: Option<Mutex<lru::LruCache<String, f32>>>,
}

impl OnnxCrossEncoder {
    pub fn new(config: CrossEncoderConfig) -> Self {
        let capacity = env_count("VELD_CROSS_ENCODER_CACHE", DEFAULT_PAIR_CACHE, 1 << 20);
        Self {
            config,
            model: OnceLock::new(),
            available: AtomicBool::new(true),
            pair_cache: std::num::NonZeroUsize::new(capacity)
                .map(|n| Mutex::new(lru::LruCache::new(n))),
        }
    }

    pub fn cache_dir(&self) -> PathBuf {
        super::downloader::get_cache_dir()
            .join("cross-encoder")
            .join(self.config.cache_key())
    }

    /// Fetch the model files on the shared pre-warm worker so the first query
    /// pays only for the ONNX session build. Idempotent. Skipped when
    /// `VELD_LAZY_CROSS_ENCODER=1`.
    pub fn prewarm(self_arc: Arc<Self>) {
        if prewarm_disabled() {
            tracing::debug!(
                "cross-encoder pre-warm skipped for {} (VELD_LAZY_CROSS_ENCODER set)",
                self_arc.config.id
            );
            return;
        }
        submit_prewarm(Box::new(move || match self_arc.prefetch() {
            Ok(_) => tracing::info!("Cross-encoder files ready: {}", self_arc.config.id),
            Err(e) => tracing::warn!(
                "Cross-encoder pre-fetch failed for {} (lazy fallback engaged): {e}",
                self_arc.config.id
            ),
        }));
    }

    /// The files on disk, downloading whichever is missing. Downloads of one cache
    /// directory are serialised process-wide: the background pre-warm and the first
    /// scorer (the recall harness's probe, a first query) both arrive here within
    /// milliseconds of each other on a cold cache, and two downloads of one file share one
    /// `.part` path, so the second to finish found its partial file renamed away by the
    /// first and failed as "Failed to download model". Under the lock the second finds
    /// the files present and returns.
    fn ensure_downloaded(&self) -> Result<(PathBuf, PathBuf)> {
        let dir = self.cache_dir();
        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        if model_path.exists() && tokenizer_path.exists() {
            return Ok((model_path, tokenizer_path));
        }

        let lock = download_lock(&dir);
        let _downloading = lock.lock();
        if model_path.exists() && tokenizer_path.exists() {
            return Ok((model_path, tokenizer_path));
        }

        if std::env::var("VELD_OFFLINE").is_ok_and(|v| v == "true" || v == "1") {
            anyhow::bail!(
                "Cross-encoder {} not cached and VELD_OFFLINE=true. \
                 Fetch it with `model-fetch --cross-encoder`, or download it manually to {:?}",
                self.config.id,
                dir
            );
        }

        self.download(&model_path, &tokenizer_path, None)?;
        Ok((model_path, tokenizer_path))
    }

    /// Download whichever of the two files is missing, each verified against its
    /// checksum when the config pins one.
    fn download(
        &self,
        model_path: &PathBuf,
        tokenizer_path: &PathBuf,
        progress: Option<&(dyn Fn(u64, u64) + Send + Sync)>,
    ) -> Result<()> {
        let dir = self.cache_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create cross-encoder cache dir: {dir:?}"))?;
        tracing::info!("Downloading cross-encoder model ({})...", self.config.id);
        if !model_path.exists() {
            super::downloader::download_file_with_checksum(
                &self.config.model_url,
                model_path,
                progress,
                self.config.model_sha256.as_deref(),
            )
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
            super::downloader::download_file_with_checksum(
                &self.config.tokenizer_url,
                tokenizer_path,
                progress,
                self.config.tokenizer_sha256.as_deref(),
            )
            .with_context(|| format!("Failed to download tokenizer for {}", self.config.id))?;
        }
        tracing::info!("Cross-encoder downloaded: {}", self.config.id);
        Ok(())
    }

    /// Make the cache hold this model's verified files: a cached file that does not
    /// match its pinned checksum is removed and fetched again. Returns the cache dir.
    pub fn fetch(&self, progress: Option<&(dyn Fn(u64, u64) + Send + Sync)>) -> Result<PathBuf> {
        let dir = self.cache_dir();
        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");
        for (path, sha) in [
            (&model_path, self.config.model_sha256.as_deref()),
            (&tokenizer_path, self.config.tokenizer_sha256.as_deref()),
        ] {
            if let (true, Some(sha)) = (path.exists(), sha) {
                if !super::downloader::verify_checksum(path, sha)? {
                    std::fs::remove_file(path)
                        .with_context(|| format!("Failed to remove {path:?}"))?;
                }
            }
        }
        self.download(&model_path, &tokenizer_path, progress)?;
        Ok(dir)
    }

    /// Whether both files are in the cache, without loading them.
    pub fn is_cached(&self) -> bool {
        let dir = self.cache_dir();
        dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists()
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

            let num_threads = intra_threads();

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

            let session = match build_session(&model_path, num_threads) {
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

            let sessions = env_count("VELD_CROSS_ENCODER_SESSIONS", DEFAULT_SESSIONS, 8).max(1);
            crate::metrics::CROSS_ENCODER_SESSIONS
                .with_label_values(&[self.config.id.as_str()])
                .set(1);
            Ok(LazyOnnx {
                primary: Mutex::new(session),
                extra: (1..sessions).map(|_| OnceLock::new()).collect(),
                growing: Mutex::new(()),
                grow_failed: AtomicBool::new(false),
                model_path,
                threads: num_threads,
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

    fn prefetch(&self) -> Result<()> {
        self.ensure_downloaded().map(|_| ())
    }

    /// Scores N (query, document) pairs, each cut to its own token length (at most
    /// `MAX_PAIR_LENGTH`). Pairs of similar length share a forward pass; one batch of all N would
    /// pad every pair to the longest and let one long memory set the cost of the rest. A pair this
    /// reranker scored before is answered from its cache.
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
    /// Scores from the cache where it has them and from the model for the rest;
    /// `documents` is non-empty.
    fn score_pairs_live(
        &self,
        model: &LazyOnnx,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<f32>> {
        let mut scores: Vec<Option<f32>> = vec![None; documents.len()];
        let mut pending: Vec<usize> = Vec::with_capacity(documents.len());
        match &self.pair_cache {
            Some(cache) => {
                let mut cache = cache.lock();
                for (i, doc) in documents.iter().enumerate() {
                    match cache.get(&pair_key(query, doc)) {
                        Some(&score) => scores[i] = Some(score),
                        None => pending.push(i),
                    }
                }
            }
            None => pending.extend(0..documents.len()),
        }
        let hits = documents.len() - pending.len();
        if hits > 0 {
            crate::metrics::CROSS_ENCODER_PAIRS_TOTAL
                .with_label_values(&["cache"])
                .inc_by(hits as u64);
        }

        if !pending.is_empty() {
            let encoded = pending
                .iter()
                .map(|&i| self.encode_pair(model, query, documents[i]))
                .collect::<Result<Vec<_>>>()?;
            let fresh = self.run_pairs(model, &encoded)?;
            crate::metrics::CROSS_ENCODER_PAIRS_TOTAL
                .with_label_values(&["model"])
                .inc_by(fresh.len() as u64);
            let mut cache = self.pair_cache.as_ref().map(|c| c.lock());
            for (&i, score) in pending.iter().zip(fresh) {
                if let Some(cache) = cache.as_mut() {
                    cache.put(pair_key(query, documents[i]), score);
                }
                scores[i] = Some(score);
            }
        }

        scores
            .into_iter()
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| anyhow::anyhow!("cross-encoder {} left a pair unscored", self.config.id))
    }

    /// Tokenize a pair and cut it to its last attended token, at most `MAX_PAIR_LENGTH`.
    fn encode_pair(&self, model: &LazyOnnx, query: &str, document: &str) -> Result<EncodedPair> {
        let encoding = model
            .tokenizer
            .encode((query, document), true)
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
        Ok(EncodedPair {
            ids: row(encoding.get_ids()),
            mask: row(encoding.get_attention_mask()),
            types: row(encoding.get_type_ids()),
        })
    }

    /// Score encoded pairs, in their order, on one checked-out session.
    fn run_pairs(&self, model: &LazyOnnx, pairs: &[EncodedPair]) -> Result<Vec<f32>> {
        let batches: Vec<Vec<usize>> = match batch_mode() {
            BatchMode::Pair => (0..pairs.len()).map(|i| vec![i]).collect(),
            BatchMode::Bucket => {
                let lens: Vec<usize> = pairs.iter().map(EncodedPair::len).collect();
                let mut order: Vec<usize> = (0..pairs.len()).collect();
                order.sort_by_key(|&i| lens[i]);
                length_buckets(&order, &lens)
            }
        };
        let mut session = model.checkout(&self.config.id)?;
        let mut scores = vec![0.0_f32; pairs.len()];
        for batch in &batches {
            let rows: Vec<&EncodedPair> = batch.iter().map(|&i| &pairs[i]).collect();
            for (&i, score) in batch
                .iter()
                .zip(self.run_batch(&mut session, model, &rows)?)
            {
                scores[i] = score;
            }
        }
        Ok(scores)
    }

    /// One forward pass over `rows`, each padded with zeros (attention 0) to the longest.
    /// A batch of one is exactly the pair at its own length.
    fn run_batch(
        &self,
        session: &mut Session,
        model: &LazyOnnx,
        rows: &[&EncodedPair],
    ) -> Result<Vec<f32>> {
        let width = rows.iter().map(|r| r.len()).max().unwrap_or(1);
        let padded = |pick: fn(&EncodedPair) -> &[i64]| -> Vec<i64> {
            let mut out = Vec::with_capacity(rows.len() * width);
            for row in rows {
                out.extend_from_slice(pick(row));
                out.resize(out.len() + width - row.len(), 0);
            }
            out
        };
        let shape = vec![rows.len(), width];
        let input_ids = Value::from_array((shape.clone(), padded(|r| &r.ids)))?;
        let attention = Value::from_array((shape.clone(), padded(|r| &r.mask)))?;
        let outputs = if model.has_token_type_ids {
            let token_types = Value::from_array((shape, padded(|r| &r.types)))?;
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
        let head_size = match model.output_head {
            OutputHead::Regression => 1_usize,
            OutputHead::BinaryClassifier => 2_usize,
        };
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        if data.len() < rows.len() * head_size {
            anyhow::bail!(
                "cross-encoder {} returned {} values for {} pairs",
                self.config.id,
                data.len(),
                rows.len()
            );
        }
        Ok(data
            .chunks(head_size)
            .take(rows.len())
            .map(|chunk| model.output_head.extract_score(chunk))
            .collect())
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

    fn prefetch(&self) -> Result<()> {
        // Best-effort per member: one member that cannot be fetched should not
        // stop the others from being ready.
        for (model, _) in &self.members {
            if let Err(e) = model.prefetch() {
                tracing::warn!("ensemble member {} prefetch failed: {e}", model.id());
            }
        }
        Ok(())
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

/// The built-in default cross-encoder, the one used when no env var selects another.
pub fn default_cross_encoder() -> OnnxCrossEncoder {
    OnnxCrossEncoder::new(
        lookup(DEFAULT_CROSS_ENCODER_ID)
            .expect("default cross-encoder id must be registered as a built-in"),
    )
}

/// The cross-encoder a `rerank_model` dial value selects. `Some(id)` is a single registry
/// model, whatever the environment says; `None` defers to
/// [`load_cross_encoder_from_env`]. An id the registry does not know is an error, not a
/// fallback: the dial is sanitized before it reaches here, so an unknown id means the
/// registry changed under a persisted dial, and reranking silently with another model
/// would mislabel every measurement made under it.
pub fn load_cross_encoder(selected: Option<&str>) -> Result<Arc<dyn CrossEncoderModel>> {
    match selected {
        Some(id) => Ok(Arc::new(OnnxCrossEncoder::new(lookup(id)?))),
        None => Ok(load_cross_encoder_from_env()),
    }
}

/// Build the cross-encoder from environment variables. Default is
/// [`DEFAULT_CROSS_ENCODER_ID`] when neither `VELD_CROSS_ENCODER` nor
/// `VELD_CROSS_ENCODERS` is set. `VELD_CROSS_ENCODERS` takes precedence when both are set.
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
                tracing::warn!("VELD_CROSS_ENCODER=`{id}` invalid: {e}; falling back to default");
            }
        }
    }

    // Default is whatever DEFAULT_CROSS_ENCODER_ID points at (currently
    // mxbai-rerank-xsmall-v1). Override via VELD_CROSS_ENCODER=<id> or
    // VELD_CROSS_ENCODERS=<id>:<w>,…
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
            Some((id, w)) => (
                id.trim(),
                w.trim()
                    .parse::<f32>()
                    .with_context(|| format!("invalid weight in ensemble entry `{raw}`"))?,
            ),
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

/// Pre-fetch a cross-encoder's files (single model or ensemble) on the shared
/// pre-warm worker, so the first query pays for the ONNX session build and not
/// for an 80-280 MB download. Idempotent — a model already on disk returns
/// immediately.
///
/// Skipped when `VELD_LAZY_CROSS_ENCODER=1` is set.
pub fn prewarm(model: Arc<dyn CrossEncoderModel>) {
    if prewarm_disabled() {
        tracing::debug!("cross-encoder pre-warm skipped ({})", model.id());
        return;
    }
    let id = model.id().to_string();
    submit_prewarm(Box::new(move || match model.prefetch() {
        Ok(_) => tracing::info!("Cross-encoder files ready: {id}"),
        Err(e) => {
            tracing::warn!("Cross-encoder pre-fetch failed for {id} (lazy fallback engaged): {e}")
        }
    }));
}

// -----------------------------------------------------------------------------
// Pre-warm worker
// -----------------------------------------------------------------------------
//
// THE PRE-WARM WORKER NEVER BUILDS AN ONNX SESSION. It downloads, and that is
// all. An ONNX session may only be built on a thread the process waits for —
// a test thread, a request thread, a blocking task inside a runtime that is
// dropped — never on one that can still be running when the process exits.
//
// Why, measured on handler_tests at 314c070. The pre-warm used to run
// `ensure_loaded()` on a detached thread, one per `CrossEncoderReranker`:
// SIGSEGV after `test result: ok` in 13 of 20 direct runs. EXC_BAD_ACCESS at
// 0x17 on a thread named `cross-encoder-prewarm-mxbai-rerank-xsmall-v1`, frames
// `OpSchema::FormalParameter::FormalParameter(const&)` <- `InferAndVerifyTypeMatch`
// <- `ConstantFolding::ApplyImpl` <- `InferenceSession::Initialize`, while the
// main thread sat in `rocksdb::PosixEnv::JoinThreadsOnExit::~JoinThreadsOnExit()`.
// `exit()` was already destroying C++ statics, ONNX Runtime's operator-schema
// registry among them, and the session build read them after they were gone.
// 35 such threads were live in that one crash report.
//
// The guard that stood here, `if cfg!(test) { return }`, could not cover it:
// `test` is not set when the lib is compiled as a dependency of an integration
// test binary, so it protected `cargo test --lib` and nothing under `tests/`.
// It also said nothing about a production process that exits mid-pre-warm,
// which is the same bug.
//
// Joining the worker from a `libc::atexit` handler was tried and does not work,
// and the reason generalises to any exit hook. Handlers run newest-registered
// first, and ONNX Runtime constructs the schema registry lazily inside the first
// `CreateSession` — so its destructor is always registered after any hook we
// could arm beforehand, and always runs first. Measured: the handler did run
// (`shutdown_prewarm` <- `__cxa_finalize_ranges` <- `exit` on the main thread)
// and the worker still faulted inside `OpSchemaRegistry::Schema`. Loading the
// dylib early with `ort::api()` does not change it; the registry is not built
// at dlopen.
//
// So the fix is ownership, not ordering: the work that cannot survive teardown
// does not run where teardown can reach it.

type PrewarmJob = Box<dyn FnOnce() + Send + 'static>;

/// Sender for the single background worker. One thread, not one per reranker:
/// the 35 concurrent downloads the old code could start were 35 copies of the
/// same fetch.
static PREWARM_TX: LazyLock<Mutex<Option<std::sync::mpsc::Sender<PrewarmJob>>>> =
    LazyLock::new(|| Mutex::new(None));

fn prewarm_disabled() -> bool {
    std::env::var("VELD_LAZY_CROSS_ENCODER")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn submit_prewarm(job: PrewarmJob) {
    let mut slot = PREWARM_TX.lock();
    if slot.is_none() {
        let (tx, rx) = std::sync::mpsc::channel::<PrewarmJob>();
        let spawned = std::thread::Builder::new()
            .name("cross-encoder-prewarm".to_string())
            .spawn(move || {
                for job in rx {
                    job();
                }
            })
            .is_ok();
        if !spawned {
            tracing::warn!(
                "cross-encoder pre-warm worker could not be spawned; loading stays lazy"
            );
            return;
        }
        *slot = Some(tx);
    }
    if let Some(tx) = slot.as_ref() {
        let _ = tx.send(job);
    }
}

// -----------------------------------------------------------------------------
// File download helper
// -----------------------------------------------------------------------------

fn download_file(url: &str, path: &PathBuf) -> Result<()> {
    tracing::info!("Downloading cross-encoder file: {url}");

    let resp = super::downloader::download_agent()
        .get(url)
        .call()
        .with_context(|| format!("HTTP request failed: {url}"))?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        anyhow::bail!("HTTP {status} for {url}");
    }

    let mut reader = resp.into_body().into_reader();
    // Write beside the target and rename: `ensure_downloaded` treats an existing
    // file as a complete one, so a download cut short — the pre-warm worker is
    // not joined at exit — must not leave a truncated model under that name.
    let partial = path.with_extension(format!(
        "{}.part",
        path.extension().and_then(|e| e.to_str()).unwrap_or("")
    ));
    let mut file = std::fs::File::create(&partial)
        .with_context(|| format!("Failed to create file: {partial:?}"))?;

    std::io::copy(&mut reader, &mut file)?;
    drop(file);
    std::fs::rename(&partial, path)
        .with_context(|| format!("Failed to publish downloaded file: {path:?}"))?;
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
    fn the_default_reranker_is_pinned_to_a_revision_and_checksums() {
        let cfg = lookup(DEFAULT_CROSS_ENCODER_ID).unwrap();
        for url in [&cfg.model_url, &cfg.tokenizer_url] {
            assert!(
                !url.contains("/resolve/main/"),
                "{url} follows a moving branch"
            );
        }
        for sha in [&cfg.model_sha256, &cfg.tokenizer_sha256] {
            let sha = sha.as_deref().expect("a pinned checksum");
            assert!(sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()));
        }
    }

    /// The parity arm of the recall gate reranks with this model; a moving revision or an
    /// unverified file would make two runs under the same dial two pipelines.
    #[test]
    fn the_ms_marco_parity_reranker_is_pinned_to_a_revision_and_checksums() {
        let cfg = lookup(MS_MARCO_MINILM_L6_ID).unwrap();
        for url in [&cfg.model_url, &cfg.tokenizer_url] {
            assert!(
                url.contains("/resolve/233902d25c440f23af6f7d6e94d2946bac0bee0a/"),
                "{url} does not name the pinned revision"
            );
        }
        for sha in [&cfg.model_sha256, &cfg.tokenizer_sha256] {
            let sha = sha.as_deref().expect("a pinned checksum");
            assert!(sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        assert_eq!(cfg.license, "apache-2.0");
    }

    #[test]
    fn every_built_in_records_a_licence() {
        for cfg in built_ins() {
            assert!(!cfg.license.is_empty(), "{} has no licence", cfg.id);
        }
        assert_eq!(
            lookup(DEFAULT_CROSS_ENCODER_ID).unwrap().license,
            "apache-2.0"
        );
    }

    #[test]
    fn a_selected_id_loads_that_model_and_an_unknown_one_is_refused() {
        let selected = load_cross_encoder(Some(MS_MARCO_MINILM_L6_ID)).unwrap();
        assert_eq!(selected.id(), MS_MARCO_MINILM_L6_ID);
        assert!(load_cross_encoder(Some("no-such-reranker")).is_err());
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
            model_sha256: None,
            tokenizer_sha256: None,
            has_token_type_ids: false,
            params_millions: 100,
            download_mb: 400,
            year: 2026,
            rerank_top_k: 15,
            notes: "test fixture",
            license: "apache-2.0",
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

    fn sorted_by_len(lens: &[usize]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..lens.len()).collect();
        order.sort_by_key(|&i| lens[i]);
        order
    }

    #[test]
    fn pairs_of_similar_length_share_a_batch_and_a_long_one_runs_apart() {
        let lens = [40, 42, 300, 38, 45];
        let buckets = length_buckets(&sorted_by_len(&lens), &lens);
        assert_eq!(buckets, vec![vec![3, 0, 1, 4], vec![2]]);
    }

    #[test]
    fn short_pairs_within_the_minimum_slack_batch_together() {
        let lens = [10, 18, 19];
        let buckets = length_buckets(&sorted_by_len(&lens), &lens);
        assert_eq!(buckets, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn a_batch_never_exceeds_the_cap() {
        let lens = vec![50; MAX_BATCH * 2 + 1];
        let buckets = length_buckets(&sorted_by_len(&lens), &lens);
        assert_eq!(buckets.len(), 3);
        assert!(buckets.iter().all(|b| b.len() <= MAX_BATCH));
        let mut all: Vec<usize> = buckets.concat();
        all.sort_unstable();
        assert_eq!(all, (0..lens.len()).collect::<Vec<_>>());
    }

    #[test]
    fn pair_keys_differ_wherever_the_query_ends() {
        assert_ne!(pair_key("ab", "c"), pair_key("a", "bc"));
        assert_ne!(pair_key("a:1", "x"), pair_key("a", ":1x"));
        assert_eq!(pair_key("q", "d"), pair_key("q", "d"));
    }
}
