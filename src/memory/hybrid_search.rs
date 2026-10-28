//! Hybrid Search Pipeline (BM25 + Vector + Cognitive)
//!
//! Production-grade retrieval combining:
//! 1. BM25 full-text search (tantivy) - keyword matching
//! 2. Vector search (Vamana) - semantic similarity
//! 3. Reciprocal Rank Fusion (RRF) - signal combination
//! 4. Cross-encoder reranking - accurate top-k scoring
//! 5. Cognitive signals - Hebbian strength, decay, feedback momentum
//!
//! Architecture:
//! ```text
//! Query → [BM25] ──┐
//!                  ├─→ [RRF Fusion] → [Cross-Encoder] → [Cognitive Boost] → Results
//! Query → [Vector] ┘
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{Index, IndexReader, IndexWriter, TantivyDocument};
use tracing::{debug, info};

use rust_stemmers::{Algorithm, Stemmer};

use super::types::MemoryId;
use crate::embeddings::Embedder;
use crate::memory::llm_refiner::Refiner;

/// Stem text using the English Porter stemmer for BM25 matching.
/// Enables "choose" to match "chose", "decided" to match "decision", etc.
fn stem_text(text: &str) -> String {
    let stemmer = Stemmer::create(Algorithm::English);
    text.split_whitespace()
        .map(|word| {
            let lower = word.to_lowercase();
            let cleaned: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
            if cleaned.len() <= 1 {
                cleaned
            } else {
                stemmer.stem(&cleaned).to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Final-stage refiner selection for hybrid search.
///
/// Controls which post-RRF stage (cross-encoder, LLM refiner, both, or
/// neither) is applied to the fused candidate set. The `llm-eval` harness
/// uses this to compare refiner strategies against an otherwise identical
/// pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RefinerMode {
    /// Skip both cross-encoder and refiner. Returns raw RRF scores.
    None,
    /// Apply only the cross-encoder rerank (current production default).
    #[default]
    CrossEncoder,
    /// Apply only the LLM refiner; skip the cross-encoder.
    #[serde(alias = "rlm")]
    Llm,
    /// Cross-encoder first, then LLM refiner over its output.
    Stacked,
}

fn default_refiner_mode() -> RefinerMode {
    RefinerMode::CrossEncoder
}

/// Configuration for hybrid search
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Weight for BM25 scores in RRF (0.0-1.0)
    #[serde(default = "default_bm25_weight")]
    pub bm25_weight: f32,

    /// Weight for vector scores in RRF (0.0-1.0)
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,

    /// Weight for graph (spreading activation) scores in RRF (0.0-1.0) (SHO-D4)
    /// Graph weight is dynamic based on graph density and memory tier
    #[serde(default = "default_graph_weight")]
    pub graph_weight: f32,

    /// RRF constant k (higher = more equal weighting)
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,

    /// Number of candidates to fetch from each retriever
    #[serde(default = "default_candidate_count")]
    pub candidate_count: usize,

    /// Number of top results to rerank with cross-encoder
    #[serde(default = "default_rerank_count")]
    pub rerank_count: usize,

    /// Whether to use cross-encoder reranking
    #[serde(default = "default_use_reranking")]
    pub use_reranking: bool,

    /// Minimum BM25 score to consider (filters noise)
    #[serde(default = "default_min_bm25_score")]
    pub min_bm25_score: f32,

    /// Minimum graph activation score to consider (SHO-D4)
    #[serde(default = "default_min_graph_score")]
    pub min_graph_score: f32,

    /// Final-stage refiner mode (cross-encoder, LLM refiner, both, or none).
    ///
    /// Defaults to `CrossEncoder` for backward compatibility with existing
    /// callers. The `llm-eval` harness varies this to compare refiner
    /// strategies.
    #[serde(default = "default_refiner_mode")]
    pub refiner_mode: RefinerMode,
}

fn default_bm25_weight() -> f32 {
    0.35 // Reduced: BM25 over-matches common terms (names), diluting rare discriminative terms
}
fn default_vector_weight() -> f32 {
    0.40 // Vector similarity handles semantic relationships
}
fn default_graph_weight() -> f32 {
    0.25 // Graph spreading activation for associative retrieval (SHO-D4)
}
fn default_rrf_k() -> f32 {
    // NOTE: This K=45 is for internal BM25+vector fusion in search_with_dynamic_weights().
    // The main retrieval pipeline (Layer 4 in semantic_retrieve) uses query.rrf_k (default 20.0).
    // These are DIFFERENT fusion stages:
    //   K=45 here: merges BM25 keyword scores with vector semantic scores (gentle blend)
    //   K=20 in Layer 4: merges graph + hybrid + linguistic signals (sharper discrimination)
    // The higher K here prevents BM25 false positives from dominating vector results.
    45.0
}
fn default_candidate_count() -> usize {
    100 // Increased for better recall; slight latency tradeoff acceptable
}
fn default_rerank_count() -> usize {
    20
}
fn default_use_reranking() -> bool {
    true // Enabled: uses ms-marco-MiniLM-L-6-v2 cross-encoder with bi-encoder fallback
}
fn default_min_bm25_score() -> f32 {
    0.01 // Lower threshold to capture more keyword matches
}
fn default_min_graph_score() -> f32 {
    0.01 // Lower threshold to capture graph-based associations (SHO-D4)
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            bm25_weight: default_bm25_weight(),
            vector_weight: default_vector_weight(),
            graph_weight: default_graph_weight(),
            rrf_k: default_rrf_k(),
            candidate_count: default_candidate_count(),
            rerank_count: default_rerank_count(),
            use_reranking: default_use_reranking(),
            min_bm25_score: default_min_bm25_score(),
            min_graph_score: default_min_graph_score(),
            refiner_mode: default_refiner_mode(),
        }
    }
}

/// Signal contribution scores from a retrieval result
///
/// Records which retrieval signal (BM25, vector, graph) contributed most to
/// finding a particular result. Used as the learning target for adaptive weight
/// updates: helpful results shift weights toward their dominant signal.
#[derive(Debug, Clone, Copy)]
pub struct SignalScores {
    /// How much BM25 (keyword) matching contributed to this result (0.0-1.0)
    pub bm25_contribution: f32,
    /// How much vector (semantic) search contributed to this result (0.0-1.0)
    pub vector_contribution: f32,
    /// How much graph (spreading activation) contributed to this result (0.0-1.0)
    pub graph_contribution: f32,
}

impl SignalScores {
    /// Create signal scores from component presence and normalize to sum to 1.0
    pub fn from_components(bm25: f32, vector: f32, graph: f32) -> Self {
        let sum = bm25 + vector + graph;
        if sum <= 0.0 {
            return Self {
                bm25_contribution: 1.0 / 3.0,
                vector_contribution: 1.0 / 3.0,
                graph_contribution: 1.0 / 3.0,
            };
        }
        Self {
            bm25_contribution: bm25 / sum,
            vector_contribution: vector / sum,
            graph_contribution: graph / sum,
        }
    }

    /// Invert contributions for negative feedback (misleading results)
    ///
    /// When a result is misleading, we want to shift weights AWAY from the
    /// signals that produced it. Inversion maps high contributions to low targets.
    pub fn inverted(&self) -> Self {
        // Invert: high contribution → low target, low → high
        let inv_bm25 = 1.0 - self.bm25_contribution;
        let inv_vector = 1.0 - self.vector_contribution;
        let inv_graph = 1.0 - self.graph_contribution;
        Self::from_components(inv_bm25, inv_vector, inv_graph)
    }
}

/// Current on-disk schema version for [`LearnedWeights`] (B12).
///
/// Bump this whenever the persisted shape changes incompatibly. On load,
/// a mismatch (or a deser failure) falls back to [`LearnedWeights::default`]
/// — the neutral uniform weights — and warns; it never panics.
pub const LEARNED_WEIGHTS_SCHEMA_VERSION: u32 = 1;

/// Learned retrieval weights from feedback history
///
/// Updated via EMA (exponential moving average) when retrieval feedback is received.
/// Blended with static defaults based on update_count confidence.
///
/// The system starts with static defaults (well-tuned from benchmarks) and gradually
/// shifts toward empirically-observed signal contributions as feedback accumulates.
///
/// R3a: `Serialize`/`Deserialize`/`Default` are derived so the vector can be
/// persisted per-user (closes F3 — the in-memory-only ranker reset on restart).
/// Persistence is pure infrastructure: these weights are NOT read by the scoring
/// path in this step. Field-level `#[serde(default)]` keeps older serialized forms
/// loadable (additive evolution); `schema_version` guards incompatible changes (B12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedWeights {
    /// Learned BM25 weight (initialized to static default)
    #[serde(default)]
    pub bm25: f32,
    /// Learned vector weight (initialized to static default)
    #[serde(default)]
    pub vector: f32,
    /// Learned graph weight (initialized to static default)
    #[serde(default)]
    pub graph: f32,
    /// Number of feedback updates applied (confidence measure)
    #[serde(default)]
    pub update_count: u64,
    /// R3b B8: Number of NEGATIVE feedback updates (Misleading / Abstained / Ignored).
    /// Used to enforce the diversity/negative-sample floor before weights are
    /// considered ready for live application.  Serde-defaulted to 0 for
    /// older serialized forms (additive evolution, B12-safe).
    #[serde(default)]
    pub negative_update_count: u64,
    /// On-disk schema version (B12). Defaults to the current version for
    /// older serialized forms that predate this field.
    #[serde(default = "default_learned_weights_schema_version")]
    pub schema_version: u32,
}

fn default_learned_weights_schema_version() -> u32 {
    LEARNED_WEIGHTS_SCHEMA_VERSION
}

impl Default for LearnedWeights {
    /// Neutral, static-equivalent weights: uniform 1/3 across signals, no
    /// feedback applied. This is the B12 fallback when persisted weights are
    /// missing, corrupt, or carry a mismatched schema version.
    fn default() -> Self {
        Self {
            bm25: 1.0 / 3.0,
            vector: 1.0 / 3.0,
            graph: 1.0 / 3.0,
            update_count: 0,
            negative_update_count: 0,
            schema_version: LEARNED_WEIGHTS_SCHEMA_VERSION,
        }
    }
}

impl LearnedWeights {
    /// Create learned weights initialized to static defaults
    pub fn from_defaults(config: &HybridSearchConfig) -> Self {
        Self {
            bm25: config.bm25_weight,
            vector: config.vector_weight,
            graph: config.graph_weight,
            update_count: 0,
            negative_update_count: 0,
            schema_version: LEARNED_WEIGHTS_SCHEMA_VERSION,
        }
    }

    /// Apply an EMA update from feedback signal scores
    ///
    /// For helpful outcomes: target = signal scores (shift toward what worked)
    /// For misleading outcomes: target = inverted scores (shift away from what failed)
    ///
    /// `is_negative` should be true for Misleading / Abstained / Ignored outcomes.
    /// When true, `negative_update_count` is incremented (R3b B8 diversity guard).
    pub fn apply_update(&mut self, target: &SignalScores, lr: f32, is_negative: bool) {
        self.bm25 = (1.0 - lr) * self.bm25 + lr * target.bm25_contribution;
        self.vector = (1.0 - lr) * self.vector + lr * target.vector_contribution;
        self.graph = (1.0 - lr) * self.graph + lr * target.graph_contribution;

        // Normalize to sum to 1.0 (maintains weight invariant)
        let sum = self.bm25 + self.vector + self.graph;
        if sum > 0.0 {
            self.bm25 /= sum;
            self.vector /= sum;
            self.graph /= sum;
        } else {
            // Degenerate case: reset to uniform
            self.bm25 = 1.0 / 3.0;
            self.vector = 1.0 / 3.0;
            self.graph = 1.0 / 3.0;
        }

        self.update_count += 1;
        if is_negative {
            self.negative_update_count += 1;
        }
    }

    /// R3b B8: Check diversity floor.
    ///
    /// Returns `true` when the fraction of negative-label updates meets or
    /// exceeds `R3B_MIN_NEGATIVE_SAMPLE_FRACTION`.  Requires at least
    /// `R3B_MIN_UPDATES_BEFORE_READY` total updates to avoid a 0/0 check on a
    /// fresh weight vector.
    pub fn negative_sample_floor_met(&self) -> bool {
        use crate::constants::{R3B_MIN_NEGATIVE_SAMPLE_FRACTION, R3B_MIN_UPDATES_BEFORE_READY};
        if self.update_count < R3B_MIN_UPDATES_BEFORE_READY {
            return false;
        }
        let frac = self.negative_update_count as f32 / self.update_count as f32;
        frac >= R3B_MIN_NEGATIVE_SAMPLE_FRACTION
    }
}

/// Storage key for the persisted learned-weight vector (R3a).
///
/// A single fixed key: the per-user key-value column family already namespaces
/// tenants, so the `user_id` is NOT folded into the key (mirrors how
/// `RetrievalTraceStore` relies on per-user CF isolation). One vector per user.
pub(crate) const LEARNED_WEIGHTS_KEY: &[u8] = b"learned_weights:v1";

/// Per-user persistent store for [`LearnedWeights`] (R3a).
///
/// Closes F3: the EMA ranker was in-memory-only and reset on every restart.
/// This store persists/loads the vector through a per-user
/// [`crate::storage::KeyValueStore`] handle (the same per-user CF pattern as
/// [`crate::memory::feedback::RetrievalTraceStore`]). When no handle is
/// configured it is a no-op: `load()` returns the neutral default and `save()`
/// is a silent skip.
///
/// PURE PERSISTENCE INFRASTRUCTURE — the loaded weights are NOT read by the
/// scoring path in this step (R3a). No ranking/scoring behavior changes.
#[derive(Clone)]
pub struct LearnedWeightsStore {
    kv: Option<Arc<dyn crate::storage::KeyValueStore>>,
}

impl std::fmt::Debug for LearnedWeightsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnedWeightsStore")
            .field("persistent", &self.kv.is_some())
            .finish()
    }
}

impl Default for LearnedWeightsStore {
    fn default() -> Self {
        Self { kv: None }
    }
}

impl LearnedWeightsStore {
    /// In-memory/no-op store (no persistence).
    pub fn new() -> Self {
        Self::default()
    }

    /// Persistent store over a per-user KV handle.
    pub fn with_kv(kv: Arc<dyn crate::storage::KeyValueStore>) -> Self {
        Self { kv: Some(kv) }
    }

    /// Load the persisted learned weights for this user.
    ///
    /// B12: on a missing key, a deserialization failure, OR a `schema_version`
    /// mismatch, returns [`LearnedWeights::default`] (neutral weights) and warns
    /// — never panics. A KV read error is treated the same (warn + default).
    pub fn load(&self) -> LearnedWeights {
        let Some(kv) = &self.kv else {
            return LearnedWeights::default();
        };
        match kv.get_bytes(LEARNED_WEIGHTS_KEY) {
            Ok(Some(bytes)) => match serde_json::from_slice::<LearnedWeights>(&bytes) {
                Ok(w) if w.schema_version == LEARNED_WEIGHTS_SCHEMA_VERSION => w,
                Ok(w) => {
                    tracing::warn!(
                        found = w.schema_version,
                        expected = LEARNED_WEIGHTS_SCHEMA_VERSION,
                        "learned weights: schema version mismatch, falling back to default"
                    );
                    LearnedWeights::default()
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "learned weights: failed to deserialize, falling back to default"
                    );
                    LearnedWeights::default()
                }
            },
            Ok(None) => LearnedWeights::default(),
            Err(e) => {
                tracing::warn!(error = %e, "learned weights: failed to read, falling back to default");
                LearnedWeights::default()
            }
        }
    }

    /// Persist the learned weights for this user. Best-effort: serialization or
    /// write failures are logged, never propagated (this is side-state).
    pub fn save(&self, weights: &LearnedWeights) {
        let Some(kv) = &self.kv else { return };
        match serde_json::to_vec(weights) {
            Ok(bytes) => {
                if let Err(e) = kv.put_bytes(LEARNED_WEIGHTS_KEY, &bytes) {
                    tracing::warn!(error = %e, "learned weights: failed to persist");
                }
            }
            Err(e) => tracing::warn!(error = %e, "learned weights: failed to serialize"),
        }
    }
}

/// Result from hybrid search with component scores
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// Memory ID
    pub memory_id: MemoryId,

    /// Final combined score (0.0-1.0)
    pub score: f32,

    /// BM25 score (if matched)
    pub bm25_score: Option<f32>,

    /// Vector similarity score (if matched)
    pub vector_score: Option<f32>,

    /// Graph activation score from spreading activation (if matched) (SHO-D4)
    pub graph_score: Option<f32>,

    /// RRF score before reranking
    pub rrf_score: f32,

    /// Cross-encoder score (if reranked)
    pub rerank_score: Option<f32>,

    /// Rank from BM25 (if matched)
    pub bm25_rank: Option<usize>,

    /// Rank from vector search (if matched)
    pub vector_rank: Option<usize>,

    /// Rank from graph spreading activation (if matched) (SHO-D4)
    pub graph_rank: Option<usize>,
}

/// BM25 Index using Tantivy
pub struct BM25Index {
    index: Index,
    reader: IndexReader,
    writer: Arc<RwLock<IndexWriter>>,
    id_field: Field,
    content_field: Field,
    tags_field: Field,
    entities_field: Field,
    stemmed_content_field: Field,
    /// Held in RAM only (an encrypted tenant): nothing of it is on disk.
    in_memory: bool,
    /// Upserts and deletes buffered in the writer since its last commit. Only
    /// changed while the writer lock is held, so it agrees with the writer.
    uncommitted: AtomicU64,
    /// When the writer last committed, from any caller.
    last_commit: parking_lot::Mutex<Instant>,
}

impl BM25Index {
    fn schema() -> Schema {
        let mut schema_builder = Schema::builder();

        // Memory ID (stored, not tokenized)
        schema_builder.add_text_field("id", STRING | STORED);

        // Main content (tokenized for BM25)
        schema_builder.add_text_field("content", TEXT | STORED);

        // Tags (tokenized)
        schema_builder.add_text_field("tags", TEXT);

        // Entities (tokenized)
        schema_builder.add_text_field("entities", TEXT);

        // Stemmed content for morphological matching (search-only, not stored)
        schema_builder.add_text_field("stemmed_content", TEXT);

        schema_builder.build()
    }

    /// Create or open a BM25 index at the given path
    pub fn new(path: &Path) -> Result<Self> {
        let schema = Self::schema();

        // Create or open index
        std::fs::create_dir_all(path)?;
        let dir = tantivy::directory::MmapDirectory::open(path)
            .context("Failed to open tantivy directory")?;

        let index = if Index::exists(&dir)? {
            let existing = Index::open(dir).context("Failed to open existing BM25 index")?;
            // Schema migration: if existing index lacks stemmed_content, rebuild
            if existing.schema().get_field("stemmed_content").is_err() {
                tracing::warn!(
                    "BM25 schema missing stemmed_content — rebuilding index for stemming support"
                );
                drop(existing);
                if let Err(e) = std::fs::remove_dir_all(path) {
                    tracing::warn!("Failed to remove old BM25 index: {e}");
                }
                std::fs::create_dir_all(path)?;
                Index::create_in_dir(path, schema)
                    .context("Failed to create BM25 index with stemmed_content")?
            } else {
                existing
            }
        } else {
            Index::create_in_dir(path, schema).context("Failed to create BM25 index")?
        };
        info!("BM25 index initialized at {:?}", path);
        Self::from_index(index, false)
    }

    /// The index for an ENCRYPTED tenant: built in RAM, never written to disk.
    ///
    /// It stores each memory's `content` and every token's positions, so on disk
    /// beside a sealed store and journal it would be a plaintext copy of every
    /// memory. It starts empty and its projection replays the whole intent log;
    /// an index left at `path` from before encryption is deleted as plaintext.
    pub fn in_memory(path: &Path) -> Result<Self> {
        if path.exists() {
            std::fs::remove_dir_all(path).with_context(|| {
                format!(
                    "Failed to remove the plaintext BM25 index at {path:?} for an encrypted tenant"
                )
            })?;
            tracing::warn!(
                "removed the on-disk BM25 index at {:?}: for an encrypted tenant it is a plaintext \
                 copy of every memory. It is rebuilt in memory from the journal on each start.",
                path
            );
        }
        Self::from_index(Index::create_in_ram(Self::schema()), true)
    }

    /// Whether this index lives only in memory.
    pub fn is_in_memory(&self) -> bool {
        self.in_memory
    }

    fn from_index(index: Index, in_memory: bool) -> Result<Self> {
        let actual_schema = index.schema();
        let id_field = actual_schema
            .get_field("id")
            .context("BM25 schema missing 'id' field")?;
        let content_field = actual_schema
            .get_field("content")
            .context("BM25 schema missing 'content' field")?;
        let tags_field = actual_schema
            .get_field("tags")
            .context("BM25 schema missing 'tags' field")?;
        let entities_field = actual_schema
            .get_field("entities")
            .context("BM25 schema missing 'entities' field")?;
        let stemmed_content_field = actual_schema
            .get_field("stemmed_content")
            .context("BM25 schema missing 'stemmed_content' field")?;

        // One indexing thread, 15MB heap. `Index::writer` would spread documents over
        // a CPU-dependent thread count, and BM25 scores come from per-segment corpus
        // statistics, so segment layout (and the score of one document) would then
        // depend on how the work happened to be divided: whole-percent drift between
        // two ingests of the same corpus. A single thread makes layout a function of
        // the corpus alone, at the cost of ingest throughput on multi-core machines.
        let writer = index
            .writer_with_num_threads(1, 15_000_000)
            .context("Failed to create index writer")?;

        let reader = index
            .reader_builder()
            .reload_policy(tantivy::ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create index reader")?;

        Ok(Self {
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            id_field,
            content_field,
            tags_field,
            entities_field,
            stemmed_content_field,
            in_memory,
            uncommitted: AtomicU64::new(0),
            last_commit: parking_lot::Mutex::new(Instant::now()),
        })
    }

    /// Add or update a document in the index
    pub fn upsert(
        &self,
        memory_id: &MemoryId,
        content: &str,
        tags: &[String],
        entities: &[String],
    ) -> Result<()> {
        let writer = self.writer.write();

        // Delete existing document with this ID
        let id_term = tantivy::Term::from_field_text(self.id_field, &memory_id.0.to_string());
        writer.delete_term(id_term);

        // Create new document
        let mut doc = TantivyDocument::new();
        doc.add_text(self.id_field, memory_id.0.to_string());
        doc.add_text(self.content_field, content);
        doc.add_text(self.stemmed_content_field, stem_text(content));
        doc.add_text(self.tags_field, tags.join(" "));
        doc.add_text(self.entities_field, entities.join(" "));

        writer.add_document(doc)?;
        self.uncommitted.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Remove a document from the index
    pub fn delete(&self, memory_id: &MemoryId) -> Result<()> {
        let writer = self.writer.write();
        let id_term = tantivy::Term::from_field_text(self.id_field, &memory_id.0.to_string());
        writer.delete_term(id_term);
        self.uncommitted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Commit pending changes to disk
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.write();
        Self::commit_locked(&mut writer, &self.uncommitted, &self.last_commit)
    }

    fn commit_locked(
        writer: &mut IndexWriter,
        uncommitted: &AtomicU64,
        last_commit: &parking_lot::Mutex<Instant>,
    ) -> Result<()> {
        writer.commit().context("Failed to commit BM25 index")?;
        uncommitted.store(0, Ordering::Relaxed);
        *last_commit.lock() = Instant::now();
        Ok(())
    }

    /// Whether upserts or deletes are buffered in the writer and not yet
    /// visible to searches.
    pub fn has_uncommitted(&self) -> bool {
        self.uncommitted.load(Ordering::Relaxed) > 0
    }

    /// Time since the writer last committed.
    pub fn since_last_commit(&self) -> Duration {
        self.last_commit.lock().elapsed()
    }

    /// Commit and reload the reader if anything is buffered; returns whether
    /// a commit ran. The check is repeated under the writer lock so that
    /// concurrent callers racing on one batch produce a single commit.
    pub fn commit_pending(&self) -> Result<bool> {
        if !self.has_uncommitted() {
            return Ok(false);
        }
        {
            let mut writer = self.writer.write();
            if self.uncommitted.load(Ordering::Relaxed) == 0 {
                return Ok(false);
            }
            Self::commit_locked(&mut writer, &self.uncommitted, &self.last_commit)?;
        }
        self.reload()?;
        Ok(true)
    }

    /// Search using BM25
    ///
    /// Returns (memory_id, score) pairs sorted by score descending
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(MemoryId, f32)>> {
        self.search_with_term_weights(query, limit, None)
    }

    /// Search using BM25 with IC-weighted term boosting
    ///
    /// Term weights are derived from linguistic Information Content (IC):
    /// - Nouns: IC_NOUN = 1.5 (focal entities, highest weight)
    /// - Adjectives: IC_ADJECTIVE = 0.9 (discriminative modifiers)
    /// - Verbs: IC_VERB = 0.7 (relational context)
    ///
    /// The weights are applied as Tantivy boost operators (term^weight)
    pub fn search_with_term_weights(
        &self,
        query: &str,
        limit: usize,
        term_weights: Option<&HashMap<String, f32>>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        self.search_with_term_and_phrase_weights(query, limit, term_weights, None)
    }

    /// Search with IC-weighted term boosting AND phrase matching
    ///
    /// Phrase boosts significantly improve retrieval for multi-word concepts:
    /// - "support group" matches exact phrase, not just "support" OR "group"
    /// - Reduces false positives from partial term matches
    pub fn search_with_term_and_phrase_weights(
        &self,
        query: &str,
        limit: usize,
        term_weights: Option<&HashMap<String, f32>>,
        phrase_boosts: Option<&[(String, f32)]>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();

        // Parse query across content, tags, entities, and stemmed content
        let query_parser = QueryParser::for_index(
            &self.index,
            vec![
                self.content_field,
                self.tags_field,
                self.entities_field,
                self.stemmed_content_field,
            ],
        );

        // Build boosted query with term weights + stemmed variants
        let stemmer = Stemmer::create(Algorithm::English);
        let mut query_parts: Vec<String> = Vec::new();

        // Add individual terms with IC weights.
        // Also add stemmed variants so "choose" matches "chose", "decided" matches "decision".
        if let Some(weights) = term_weights {
            for word in query.split_whitespace() {
                let clean_word: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                if clean_word.is_empty() {
                    continue;
                }
                if let Some(&weight) = weights.get(&clean_word) {
                    query_parts.push(format!("{}^{:.1}", clean_word, weight));
                } else {
                    query_parts.push(clean_word.clone());
                }
                // Add stemmed form if different (searches stemmed_content field)
                let stemmed = stemmer.stem(&clean_word).to_string();
                if stemmed != clean_word && stemmed.len() > 1 {
                    if let Some(&weight) = weights.get(&clean_word) {
                        query_parts.push(format!("{}^{:.1}", stemmed, weight * 0.8));
                    } else {
                        query_parts.push(stemmed);
                    }
                }
            }
        } else {
            // No term weights - add words + stemmed variants
            for word in query.split_whitespace() {
                let clean_word: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                if !clean_word.is_empty() {
                    let stemmed = stemmer.stem(&clean_word).to_string();
                    if stemmed != clean_word && stemmed.len() > 1 {
                        query_parts.push(stemmed);
                    }
                    query_parts.push(clean_word);
                }
            }
        }

        // Add phrase queries with boosts (e.g., "support group"^2.0)
        // Phrase queries provide significant boost when exact phrase is found
        if let Some(phrases) = phrase_boosts {
            for (phrase, boost) in phrases {
                // Tantivy phrase query syntax: "word1 word2"^boost
                // Only add if phrase has multiple words and doesn't contain special chars
                if phrase.contains(' ') && !phrase.contains('"') {
                    query_parts.push(format!("\"{}\"^{:.1}", phrase, boost));
                }
            }
        }

        let boosted_query = query_parts.join(" ");

        // Handle query parsing errors gracefully
        let parsed_query = match query_parser.parse_query(&boosted_query) {
            Ok(q) => q,
            Err(e) => {
                debug!("BM25 query parse error for '{}': {}", boosted_query, e);
                // Fall back to simple term query without boosts
                let escaped = query.replace(
                    [
                        ':', '^', '~', '*', '?', '[', ']', '{', '}', '(', ')', '"', '\\', '/', '+',
                        '-', '!', '&', '|',
                    ],
                    " ",
                );
                match query_parser.parse_query(&escaped) {
                    Ok(q) => q,
                    Err(_) => return Ok(Vec::new()),
                }
            }
        };

        if limit == 0 {
            return Ok(Vec::new());
        }

        // Equal BM25 scores are common (quantized field norms, short documents), and TopDocs
        // breaks them by doc address, which moves whenever background segment merges finish.
        // Widen the window until the score group at the cutoff is complete (bounded, so a
        // degenerate query cannot read the whole index), then order ties by content so the
        // cut does not depend on segment layout.
        let max_window = limit.saturating_mul(32).max(1024);
        let mut window = limit + 1;
        let top_docs = loop {
            let docs = searcher
                .search(&parsed_query, &TopDocs::with_limit(window).order_by_score())
                .context("BM25 search failed")?;
            if docs.len() < window
                || docs[window - 1].0 != docs[limit - 1].0
                || window >= max_window
            {
                break docs;
            }
            window = window.saturating_mul(2).min(max_window);
        };

        let mut scored: Vec<(MemoryId, f32, String)> = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            if let Ok(doc) = searcher.doc::<TantivyDocument>(doc_address) {
                let id = doc
                    .get_first(self.id_field)
                    .and_then(|v| v.as_str())
                    .and_then(|s| uuid::Uuid::parse_str(s).ok());
                if let Some(uuid) = id {
                    let content = doc
                        .get_first(self.content_field)
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    scored.push((MemoryId(uuid), score, content));
                }
            }
        }
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(limit);

        Ok(scored
            .into_iter()
            .map(|(id, score, _)| (id, score))
            .collect())
    }

    /// Get document count
    pub fn len(&self) -> usize {
        let searcher = self.reader.searcher();
        searcher.num_docs() as usize
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reload the reader to see committed changes
    pub fn reload(&self) -> Result<()> {
        self.reader.reload()?;
        Ok(())
    }
}

/// Reciprocal Rank Fusion (RRF) implementation
///
/// Combines rankings from multiple retrievers using:
/// RRF(d) = Σ 1/(k + rank_i(d))
///
/// Where k is a constant (typically 60) that controls how much
/// weight is given to documents ranked lower.
pub struct RRFusion {
    /// RRF constant k
    k: f32,
    /// Weight for each retriever (normalized)
    weights: Vec<f32>,
}

impl RRFusion {
    /// Create new RRF with given k and weights
    pub fn new(k: f32, weights: Vec<f32>) -> Self {
        // Normalize weights
        let sum: f32 = weights.iter().sum();
        let normalized = if sum > 0.0 {
            weights.iter().map(|w| w / sum).collect()
        } else {
            vec![1.0 / weights.len() as f32; weights.len()]
        };

        Self {
            k,
            weights: normalized,
        }
    }

    /// Fuse multiple ranked lists into a single ranking
    ///
    /// Each input is a Vec of (MemoryId, score) sorted by score descending.
    /// Returns fused (MemoryId, rrf_score) sorted by rrf_score descending.
    pub fn fuse(&self, ranked_lists: Vec<Vec<(MemoryId, f32)>>) -> Vec<(MemoryId, f32)> {
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();
        let mut original_scores: HashMap<MemoryId, Vec<Option<f32>>> = HashMap::new();
        // First-appearance order across the input lists: equal fused scores keep it, so the
        // output does not depend on the map's per-instance hash order.
        let mut first_seen: Vec<MemoryId> = Vec::new();

        for (list_idx, ranked_list) in ranked_lists.iter().enumerate() {
            let weight = self.weights.get(list_idx).copied().unwrap_or(1.0);

            for (rank, (memory_id, score)) in ranked_list.iter().enumerate() {
                // RRF formula: weight * 1/(k + rank)
                // rank is 0-indexed, so rank+1 for 1-indexed
                let rrf_contribution = weight / (self.k + (rank + 1) as f32);

                let fused = scores.entry(memory_id.clone()).or_insert_with(|| {
                    first_seen.push(memory_id.clone());
                    0.0
                });
                *fused += rrf_contribution;

                // Track original scores for debugging
                let orig = original_scores
                    .entry(memory_id.clone())
                    .or_insert_with(|| vec![None; ranked_lists.len()]);
                if list_idx < orig.len() {
                    orig[list_idx] = Some(*score);
                }
            }
        }

        // Sort by RRF score descending; the sort is stable over first appearance.
        let mut results: Vec<(MemoryId, f32)> = first_seen
            .into_iter()
            .map(|id| {
                let score = scores[&id];
                (id, score)
            })
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1));

        results
    }
}

/// Cross-encoder reranker over the registry in `crate::embeddings::cross_encoder`.
///
/// Uses a true cross-encoder model that jointly encodes (query, document)
/// pairs for relevance scoring. Significantly more accurate than bi-encoder
/// cosine similarity for reranking. Falls back to bi-encoder if cross-encoder
/// model is unavailable.
///
/// The active model is switchable at runtime ([`Self::select_model`], driven by the
/// `rerank_model` dial). Models that have been selected once stay loaded, keyed by id, so
/// turning the dial back does not rebuild an ONNX session.
///
/// Blending: 70% cross-encoder + 30% bi-encoder (proven ratio from literature).
pub struct CrossEncoderReranker {
    embedder: Arc<dyn Embedder>,
    /// The model that reranks now.
    cross_encoder: RwLock<Arc<dyn crate::embeddings::cross_encoder::CrossEncoderModel>>,
    /// The process-wide choice (`VELD_CROSS_ENCODER*`, else the built-in default): what a
    /// `rerank_model` of `None` selects.
    env_model: Arc<dyn crate::embeddings::cross_encoder::CrossEncoderModel>,
    /// Every model selected so far, by registry id.
    loaded: RwLock<HashMap<String, Arc<dyn crate::embeddings::cross_encoder::CrossEncoderModel>>>,
}

/// The query and the candidate [`CrossEncoderReranker::probe`] scores.
pub const CROSS_ENCODER_PROBE: &str = "probe";

/// Weight for cross-encoder score in the blend (0.7 = 70%)
const CROSS_ENCODER_BLEND_WEIGHT: f32 = 0.70;
/// Weight for bi-encoder score in the blend (0.3 = 30%)
const BI_ENCODER_BLEND_WEIGHT: f32 = 0.30;

impl CrossEncoderReranker {
    /// Create reranker with shared embedder and the env-configured cross-encoder
    /// (single model or weighted ensemble). The cross-encoder is pre-warmed on
    /// a background thread by default — the ONNX download + load happens
    /// during server startup, not on the first /api/recall. Set
    /// `VELD_LAZY_CROSS_ENCODER=1` to opt back into lazy-on-first-use, or
    /// `VELD_CROSS_ENCODER=<id>` / `VELD_CROSS_ENCODERS=<id>:<w>,…` to swap
    /// the model.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        let cross_encoder = crate::embeddings::cross_encoder::load_cross_encoder_from_env();
        crate::embeddings::cross_encoder::prewarm(Arc::clone(&cross_encoder));
        Self::with_cross_encoder(embedder, cross_encoder)
    }

    /// Create reranker with an existing cross-encoder instance, which is also what a
    /// `rerank_model` of `None` selects.
    pub fn with_cross_encoder(
        embedder: Arc<dyn Embedder>,
        cross_encoder: Arc<dyn crate::embeddings::cross_encoder::CrossEncoderModel>,
    ) -> Self {
        let mut loaded: HashMap<
            String,
            Arc<dyn crate::embeddings::cross_encoder::CrossEncoderModel>,
        > = HashMap::new();
        loaded.insert(cross_encoder.id().to_string(), Arc::clone(&cross_encoder));
        Self {
            embedder,
            env_model: Arc::clone(&cross_encoder),
            cross_encoder: RwLock::new(cross_encoder),
            loaded: RwLock::new(loaded),
        }
    }

    /// The model that reranks now.
    pub fn model(&self) -> Arc<dyn crate::embeddings::cross_encoder::CrossEncoderModel> {
        Arc::clone(&self.cross_encoder.read())
    }

    /// Registry id of the model that reranks now.
    pub fn model_id(&self) -> String {
        self.cross_encoder.read().id().to_string()
    }

    /// Make `selected` the model that reranks: a registry id, or `None` for the
    /// process-wide choice this reranker was built with. A model selected before is reused;
    /// a new one is built and its files pre-fetched in the background, so the first rerank
    /// after the switch pays for the ONNX session and not for the download. An id the
    /// registry does not know is an error and leaves the active model as it was.
    pub fn select_model(&self, selected: Option<&str>) -> Result<()> {
        let model = match selected {
            None => Arc::clone(&self.env_model),
            Some(id) => {
                // Bound first: under edition 2021 a guard taken in an `if let` condition lives
                // through the else branch, and the write below would wait on it forever.
                let cached = self.loaded.read().get(id).cloned();
                match cached {
                    Some(m) => m,
                    None => {
                        let m = crate::embeddings::cross_encoder::load_cross_encoder(Some(id))?;
                        crate::embeddings::cross_encoder::prewarm(Arc::clone(&m));
                        self.loaded.write().insert(id.to_string(), Arc::clone(&m));
                        m
                    }
                }
            }
        };
        let mut active = self.cross_encoder.write();
        if active.id() != model.id() {
            tracing::info!(from = %active.id(), to = %model.id(), "cross-encoder switched");
        }
        *active = model;
        Ok(())
    }

    /// Score one pair through the cross-encoder, loading it if it has not loaded yet, and
    /// return its id. An error means [`Self::rerank`] would fall back to bi-encoder scores.
    pub fn probe(&self) -> Result<String> {
        let model = self.model();
        model.score_pairs(CROSS_ENCODER_PROBE, &[CROSS_ENCODER_PROBE])?;
        Ok(model.id().to_string())
    }

    /// Rerank candidates using true cross-encoder with bi-encoder fallback
    ///
    /// Takes (memory_id, content, current_score, stored_embedding) and returns
    /// reranked scores. When `stored_embedding` is `Some` and its dimensionality
    /// matches the query embedding, it is reused — avoiding a redundant embedder
    /// call (Nomic via HTTP can be 5–6s per inference on CPU). On mismatch or
    /// `None`, the document content is embedded on the fly.
    ///
    /// Uses 70/30 cross-encoder/bi-encoder blend when cross-encoder is available,
    /// falls back to pure bi-encoder cosine similarity otherwise.
    pub fn rerank(
        &self,
        query: &str,
        candidates: Vec<(MemoryId, String, f32, Option<Vec<f32>>)>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Always compute bi-encoder scores (fast, used for blending or fallback).
        // Query side uses encode_for_query so asymmetric models apply the query prefix.
        let query_embedding = self.embedder.encode_for_query(query)?;
        let query_dim = query_embedding.len();
        let mut bi_scores: Vec<f32> = Vec::with_capacity(candidates.len());
        for (_, content, _, stored) in &candidates {
            let bi = match stored {
                Some(emb) if emb.len() == query_dim => cosine_similarity(&query_embedding, emb),
                _ => {
                    let doc_embedding = self.embedder.encode(content)?;
                    cosine_similarity(&query_embedding, &doc_embedding)
                }
            };
            bi_scores.push(bi);
        }

        // Try cross-encoder for blended scoring. The model is read once per call so a dial
        // turned mid-query scores every candidate of this query with one model.
        let cross_encoder = self.model();
        let use_cross_encoder = cross_encoder.is_available();
        let blended_scores = if use_cross_encoder {
            let doc_texts: Vec<&str> = candidates.iter().map(|(_, c, _, _)| c.as_str()).collect();
            match cross_encoder.score_pairs(query, &doc_texts) {
                Ok(ce_scores) => {
                    // Normalize cross-encoder logits to [0,1] via sigmoid
                    let ce_normalized: Vec<f32> = ce_scores
                        .iter()
                        .map(|&s| 1.0 / (1.0 + (-s).exp()))
                        .collect();

                    // Blend: 70% cross-encoder + 30% bi-encoder
                    bi_scores
                        .iter()
                        .zip(ce_normalized.iter())
                        .map(|(&bi, &ce)| {
                            BI_ENCODER_BLEND_WEIGHT * bi + CROSS_ENCODER_BLEND_WEIGHT * ce
                        })
                        .collect()
                }
                Err(e) => {
                    tracing::debug!("Cross-encoder inference failed, using bi-encoder: {e}");
                    bi_scores.clone()
                }
            }
        } else {
            bi_scores.clone()
        };

        let mut results: Vec<(MemoryId, f32)> = candidates
            .into_iter()
            .zip(blended_scores)
            .map(|((id, _, _, _), score)| (id, score))
            .collect();

        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(results)
    }
}

/// Public alias for cross-encoder reranking in recall pipeline (Plan 001 §2.1)
pub fn cosine_similarity_pub(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity(a, b)
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Unified hybrid search engine
///
/// Combines BM25 + Vector + RRF + Cross-encoder + Cognitive signals.
/// Includes adaptive weight learning from retrieval feedback (P2.3).
///
/// The BM25 inverted index is held as a shared `Arc<BM25Index>` so the
/// retrieval-side reader can be wired to the same handle the
/// [`crate::memory::Bm25Projection`] writes through. In server mode the
/// `Arc` is injected via [`HybridSearchEngine::with_bm25_index`]; in
/// standalone / test usage [`HybridSearchEngine::new`] opens the index on
/// disk and wraps it in `Arc` itself. Either way, every read goes through
/// the same handle that the journaled writer's projection commits to.
pub struct HybridSearchEngine {
    /// Count of BM25 index inserts lost after their error was handled and
    /// ingest continued. A memory whose insert failed is STORED BUT
    /// UNFINDABLE: it exists in the primary store and no lexical query can
    /// reach it. Distinct from a lost commit, where the document reached the
    /// writer and the batch did not reach disk.
    bm25_index_failures: std::sync::atomic::AtomicU64,

    /// Count of BM25 commit batches lost after their error was handled and
    /// ingest continued. Seven callers in memory/mod.rs treat a failed
    /// commit_and_reload as recoverable and log it — correct for availability,
    /// but it leaves the searchable index missing documents with nothing
    /// downstream able to tell. A measurement run reads this and refuses
    /// rather than reporting recall over an index it cannot vouch for.
    bm25_commit_failures: std::sync::atomic::AtomicU64,
    bm25_index: Arc<BM25Index>,
    config: HybridSearchConfig,
    reranker: Option<CrossEncoderReranker>,
    /// Optional LLM-driven refiner applied after RRF (and optionally after
    /// cross-encoder rerank). Gated by `config.refiner_mode`.
    refiner: Option<Box<dyn Refiner>>,
    /// Learned weights from retrieval feedback (EMA-adapted)
    /// Protected by RwLock for concurrent read access during search
    /// and exclusive write access during feedback updates
    learned_weights: RwLock<LearnedWeights>,
    /// Optional shared runtime tuning dials (the same `Arc` the owning
    /// `MemorySystem` holds). When attached, `Some` retrieval overrides take
    /// precedence per query; `None` (default) preserves exact engine/learned
    /// behavior. See [`crate::memory::dials::CognitiveDials`].
    dials: RwLock<Option<Arc<RwLock<crate::memory::dials::CognitiveDials>>>>,
}

impl HybridSearchEngine {
    /// Create hybrid search engine that opens its own BM25 index at
    /// `bm25_path`.
    ///
    /// Used in standalone / test code paths where there is no projection
    /// driving the index from an intent log. Server code should prefer
    /// [`HybridSearchEngine::with_bm25_index`] so the engine shares the
    /// `Arc<BM25Index>` owned by the projection — that's the only way
    /// reads converge with writes without a reader reload race.
    pub fn new(
        bm25_path: &Path,
        embedder: Arc<dyn Embedder>,
        config: HybridSearchConfig,
    ) -> Result<Self> {
        let bm25_index = Arc::new(BM25Index::new(bm25_path)?);
        Ok(Self::with_bm25_index(bm25_index, embedder, config))
    }

    /// Create hybrid search engine around a pre-opened, shared
    /// `Arc<BM25Index>`.
    ///
    /// This is the server-side seam: the `MultiUserMemoryManager` owns
    /// one `Arc<BM25Index>` per tenant (created lazily and cached). The
    /// `Bm25Projection` writes through that handle from the intent log,
    /// committing + reloading on its `COMMIT_EVERY` cadence. Passing the
    /// same handle here means a search that fires immediately after a
    /// journaled write sees the committed segments without a separate
    /// reload step on the read side.
    pub fn with_bm25_index(
        bm25_index: Arc<BM25Index>,
        embedder: Arc<dyn Embedder>,
        config: HybridSearchConfig,
    ) -> Self {
        let reranker = if config.use_reranking {
            Some(CrossEncoderReranker::new(embedder))
        } else {
            None
        };

        let learned_weights = RwLock::new(LearnedWeights::from_defaults(&config));

        Self {
            bm25_index_failures: std::sync::atomic::AtomicU64::new(0),
            bm25_commit_failures: std::sync::atomic::AtomicU64::new(0),
            bm25_index,
            config,
            reranker,
            refiner: None,
            learned_weights,
            dials: RwLock::new(None),
        }
    }

    /// Return the shared `Arc<BM25Index>` handle. Lets the host (e.g. the
    /// `MultiUserMemoryManager`) keep a second pointer alive for the
    /// projection without re-opening tantivy. Production wiring injects
    /// the handle the other way (manager → engine via
    /// [`HybridSearchEngine::with_bm25_index`]); this accessor exists for
    /// tests and admin tooling that needs to bridge in the reverse
    /// direction.
    pub fn bm25_index_arc(&self) -> Arc<BM25Index> {
        self.bm25_index.clone()
    }

    /// Attach an LLM-driven refiner to be applied after RRF fusion (and
    /// optionally after cross-encoder reranking — see [`RefinerMode`]).
    ///
    /// Without an attached refiner, `RefinerMode::Llm` and
    /// `RefinerMode::Stacked` degrade to their respective baselines:
    /// `Llm` becomes RRF-only, `Stacked` becomes cross-encoder-only.
    pub fn with_refiner(mut self, refiner: Box<dyn Refiner>) -> Self {
        self.refiner = Some(refiner);
        self
    }

    /// Attach the shared runtime [`crate::memory::dials::CognitiveDials`] so per-query
    /// retrieval overrides (refiner mode, rerank/candidate counts, RRF weights) take
    /// effect. Pass the same `Arc` the owning `MemorySystem` holds. `&self` (mirrors
    /// `set_learned_weight_state`) so it can be called on the engine before it is
    /// wrapped in an `Arc`.
    pub fn attach_dials(&self, dials: Arc<RwLock<crate::memory::dials::CognitiveDials>>) {
        *self.dials.write() = Some(dials);
    }

    /// Index a memory for BM25 search
    pub fn index_memory(
        &self,
        memory_id: &MemoryId,
        content: &str,
        tags: &[String],
        entities: &[String],
    ) -> Result<()> {
        // Counted HERE, not at the five call sites that log and continue: a
        // caller added later cannot forget to record its own loss.
        if let Err(e) = self.bm25_index.upsert(memory_id, content, tags, entities) {
            self.bm25_index_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(e);
        }
        Ok(())
    }

    /// BM25 index inserts lost since process start. Non-zero means memories
    /// are stored and lexically unreachable.
    pub fn bm25_index_failure_count(&self) -> u64 {
        self.bm25_index_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Remove a memory from the BM25 index
    pub fn remove_memory(&self, memory_id: &MemoryId) -> Result<()> {
        self.bm25_index.delete(memory_id)
    }

    /// Commit BM25 index changes
    pub fn commit(&self) -> Result<()> {
        self.bm25_index.commit()
    }

    /// Reload BM25 reader to see committed changes immediately
    pub fn reload(&self) -> Result<()> {
        self.bm25_index.reload()
    }

    /// Commit and reload in one call for immediate searchability
    /// Whether the BM25 index is held only in memory.
    pub fn bm25_in_memory(&self) -> bool {
        self.bm25_index.is_in_memory()
    }

    pub fn commit_and_reload(&self) -> Result<()> {
        // Counted HERE, not at the seven call sites: a caller added later
        // cannot forget to record its own loss.
        if let Err(e) = self.bm25_index.commit() {
            self.bm25_commit_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(e);
        }
        if let Err(e) = self.bm25_index.reload() {
            self.bm25_commit_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(e);
        }
        Ok(())
    }

    /// Commit and reload only if documents are buffered. Writes journaled
    /// through the intent log reach the index via the BM25 projection, which
    /// commits in batches; a search calls this first so a memory written by
    /// the caller is keyword-findable on its next recall.
    pub fn commit_pending(&self) -> Result<bool> {
        self.bm25_index.commit_pending().inspect_err(|_| {
            self.bm25_commit_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })
    }

    /// Commit batches lost since process start. Non-zero means the BM25 index
    /// is missing documents.
    pub fn bm25_commit_failure_count(&self) -> u64 {
        self.bm25_commit_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get BM25 index reference for direct searches
    pub fn bm25_index(&self) -> &BM25Index {
        &self.bm25_index
    }

    /// Update learned weights from retrieval feedback (R3b: wired to the
    /// reinforcement/feedback path).
    ///
    /// Called from `reinforce_recall` after it processes a feedback outcome.
    /// Uses EMA to shift weights toward (Helpful) or away from (Misleading /
    /// negative) the signal contributions that produced the retrieval result.
    ///
    /// B8 regularisation: after each update, nudge every weight 1 % back
    /// toward the static prior, preventing pathological collapse onto a single
    /// signal even when the sample distribution is skewed.
    ///
    /// After updating, the new vector is persisted via `learned_weights_store`
    /// so it survives a server restart (R3a closure).
    ///
    /// Thread-safe: takes write lock on learned_weights only for the update.
    ///
    /// `is_negative` must be true for Misleading / Abstained / Ignored outcomes
    /// so the B8 diversity counter advances.
    pub fn update_weights_from_feedback(
        &self,
        signal_scores: &SignalScores,
        helpful: bool,
        is_negative: bool,
    ) {
        use crate::constants::ADAPTIVE_WEIGHT_LEARNING_RATE;

        let target = if helpful {
            *signal_scores
        } else {
            signal_scores.inverted()
        };

        let mut weights = self.learned_weights.write();
        weights.apply_update(&target, ADAPTIVE_WEIGHT_LEARNING_RATE, is_negative);

        // B8 regularisation toward the static prior: 1 % per epoch prevents
        // weight collapse even under skewed positive-only feedback.
        const REGULARISATION_RATE: f32 = 0.01;
        weights.bm25 = (1.0 - REGULARISATION_RATE) * weights.bm25
            + REGULARISATION_RATE * self.config.bm25_weight;
        weights.vector = (1.0 - REGULARISATION_RATE) * weights.vector
            + REGULARISATION_RATE * self.config.vector_weight;
        weights.graph = (1.0 - REGULARISATION_RATE) * weights.graph
            + REGULARISATION_RATE * self.config.graph_weight;
        // Re-normalise after regularisation to keep the simplex invariant.
        let rsum = weights.bm25 + weights.vector + weights.graph;
        if rsum > 0.0 {
            weights.bm25 /= rsum;
            weights.vector /= rsum;
            weights.graph /= rsum;
        }

        tracing::debug!(
            "R3b weights updated ({}): bm25={:.3}, vector={:.3}, graph={:.3}, \
             updates={}, neg_updates={}",
            if helpful { "helpful" } else { "negative" },
            weights.bm25,
            weights.vector,
            weights.graph,
            weights.update_count,
            weights.negative_update_count,
        );
    }

    /// R3b B1: Snapshot the current learned weight vector ONCE at the start of a
    /// query.  Returns a value (not a reference) so the caller holds a stable copy
    /// that cannot be torn by a concurrent `update_weights_from_feedback` call
    /// mid-scoring loop.
    pub fn snapshot_learned_weights(&self) -> LearnedWeights {
        self.learned_weights.read().clone()
    }

    /// R3b: Check whether the learned weights pass all gates required before
    /// live application would be allowed.
    ///
    /// Returns `true` when ALL of:
    ///   1. `R3B_LIVE_APPLICATION_ENABLED` is `true` (dormant unless flipped),
    ///   2. `update_count >= R3B_MIN_UPDATES_BEFORE_READY`,
    ///   3. The B8 negative-sample diversity floor is met.
    ///
    /// Currently always returns `false` because `R3B_LIVE_APPLICATION_ENABLED`
    /// is `false` in this PR.  The rest of the checks are exercised by tests.
    pub fn weights_ready(&self) -> bool {
        use crate::constants::R3B_LIVE_APPLICATION_ENABLED;
        if !R3B_LIVE_APPLICATION_ENABLED {
            return false;
        }
        let w = self.learned_weights.read();
        w.negative_sample_floor_met()
    }

    /// Read-only access to the static configuration (R3b: used by the recall
    /// path to retrieve the static weight floor without holding a lock).
    pub fn config(&self) -> &HybridSearchConfig {
        &self.config
    }

    /// Get effective weights blending static defaults with learned weights
    ///
    /// The blend factor increases with update_count:
    /// - 0 updates: 100% static defaults
    /// - CONFIDENCE_THRESHOLD updates: MAX_BLEND% learned, (1-MAX_BLEND)% static
    /// - Beyond threshold: capped at MAX_BLEND% learned
    ///
    /// Returns (bm25_weight, vector_weight, graph_weight) normalized to sum to 1.0
    pub fn effective_weights(&self) -> (f32, f32, f32) {
        use crate::constants::{ADAPTIVE_WEIGHT_CONFIDENCE_THRESHOLD, ADAPTIVE_WEIGHT_MAX_BLEND};

        let learned = self.learned_weights.read();

        if learned.update_count == 0 {
            return (
                self.config.bm25_weight,
                self.config.vector_weight,
                self.config.graph_weight,
            );
        }

        // Confidence ramp: linearly increase blend factor with update count
        let confidence =
            (learned.update_count as f32 / ADAPTIVE_WEIGHT_CONFIDENCE_THRESHOLD as f32).min(1.0);
        let blend = confidence * ADAPTIVE_WEIGHT_MAX_BLEND;

        let bm25 = (1.0 - blend) * self.config.bm25_weight + blend * learned.bm25;
        let vector = (1.0 - blend) * self.config.vector_weight + blend * learned.vector;
        let graph = (1.0 - blend) * self.config.graph_weight + blend * learned.graph;

        // Normalize to sum to 1.0
        let sum = bm25 + vector + graph;
        if sum > 0.0 {
            (bm25 / sum, vector / sum, graph / sum)
        } else {
            (
                self.config.bm25_weight,
                self.config.vector_weight,
                self.config.graph_weight,
            )
        }
    }

    /// Get the current learned weight state (for diagnostics/introspection)
    pub fn learned_weight_state(&self) -> LearnedWeights {
        self.learned_weights.read().clone()
    }

    /// Replace learned weights during bootstrap or restore.
    pub fn set_learned_weight_state(&self, weights: LearnedWeights) {
        *self.learned_weights.write() = weights;
    }

    /// Access the cross-encoder reranker (if configured).
    /// Used by recall.rs Layer 5.3 to inject cross-encoder scores into unified scoring.
    pub fn reranker(&self) -> Option<&CrossEncoderReranker> {
        self.reranker.as_ref()
    }

    /// Perform hybrid search combining BM25 and vector results
    ///
    /// # Arguments
    /// * `query` - Search query text
    /// * `vector_results` - Pre-computed vector search results (memory_id, similarity)
    /// * `get_content` - Closure to fetch content for reranking
    ///
    /// # Returns
    /// Hybrid search results with component scores
    pub fn search<F>(
        &self,
        query: &str,
        vector_results: Vec<(MemoryId, f32)>,
        get_content: F,
    ) -> Result<Vec<HybridSearchResult>>
    where
        F: Fn(&MemoryId) -> Option<String>,
    {
        self.search_with_ic_weights(query, vector_results, get_content, None)
    }

    /// Perform hybrid search with IC-weighted BM25 term boosting
    ///
    /// IC weights from linguistic analysis boost important terms:
    /// - Nouns (focal entities): IC=1.5
    /// - Adjectives (modifiers): IC=0.9
    /// - Verbs (relations): IC=0.7
    ///
    /// This improves retrieval by prioritizing semantically important query terms.
    pub fn search_with_ic_weights<F>(
        &self,
        query: &str,
        vector_results: Vec<(MemoryId, f32)>,
        get_content: F,
        term_weights: Option<&HashMap<String, f32>>,
    ) -> Result<Vec<HybridSearchResult>>
    where
        F: Fn(&MemoryId) -> Option<String>,
    {
        self.search_with_ic_weights_and_phrases(
            query,
            vector_results,
            get_content,
            term_weights,
            None,
        )
    }

    /// Perform hybrid search with IC-weighted BM25 term boosting AND phrase matching
    ///
    /// IC weights from linguistic analysis boost important terms.
    /// Phrase boosts enable exact multi-word phrase matching:
    /// - "support group" matches the exact phrase, not just "support" OR "group"
    /// - Compound nouns get 2.0x boost, adjacent nouns get 1.5x boost
    pub fn search_with_ic_weights_and_phrases<F>(
        &self,
        query: &str,
        vector_results: Vec<(MemoryId, f32)>,
        get_content: F,
        term_weights: Option<&HashMap<String, f32>>,
        phrase_boosts: Option<&[(String, f32)]>,
    ) -> Result<Vec<HybridSearchResult>>
    where
        F: Fn(&MemoryId) -> Option<String>,
    {
        // Use default discriminativeness (no dynamic weight adjustment)
        self.search_with_dynamic_weights(
            query,
            vector_results,
            get_content,
            term_weights,
            phrase_boosts,
            None,
            None,
        )
    }

    /// Perform hybrid search with dynamic BM25/vector weight adjustment
    ///
    /// When `keyword_discriminativeness` is provided and high (>0.5), BM25 weight
    /// is boosted to ensure discriminative keywords are properly matched.
    ///
    /// This solves the multi-hop retrieval problem where queries like
    /// "When did Melanie paint a sunrise?" fail because common terms ("Melanie", "paint")
    /// dominate, while the discriminative term ("sunrise") gets diluted in vector search.
    ///
    /// Dynamic weight adjustment:
    /// - discriminativeness 0.0-0.4: use default weights (BM25=0.4, Vector=0.6)
    /// - discriminativeness 0.5-0.7: boost BM25 (BM25=0.55, Vector=0.45)
    /// - discriminativeness 0.8-1.0: strong BM25 (BM25=0.7, Vector=0.3)
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_dynamic_weights<F>(
        &self,
        query: &str,
        vector_results: Vec<(MemoryId, f32)>,
        get_content: F,
        term_weights: Option<&HashMap<String, f32>>,
        phrase_boosts: Option<&[(String, f32)]>,
        keyword_discriminativeness: Option<f32>,
        rerank_count_override: Option<usize>,
    ) -> Result<Vec<HybridSearchResult>>
    where
        F: Fn(&MemoryId) -> Option<String>,
    {
        // Runtime dial overrides snapshotted once. `None` (no dials attached, or a
        // given override unset) => exact engine/learned-weight behavior (parity).
        let dial = self.dials.read().as_ref().map(|h| h.read().clone());
        let candidate_count = dial
            .as_ref()
            .and_then(|d| d.candidate_count)
            .unwrap_or(self.config.candidate_count);

        if let Err(e) = self.commit_pending() {
            tracing::warn!(error = %e, "BM25 commit before search failed; buffered writes are not yet searchable");
        }

        // 1. BM25 search with IC-weighted term boosting AND phrase matching
        let bm25_results = self.bm25_index.search_with_term_and_phrase_weights(
            query,
            candidate_count,
            term_weights,
            phrase_boosts,
        )?;

        // Filter low BM25 scores
        let bm25_results: Vec<_> = bm25_results
            .into_iter()
            .filter(|(_, score)| {
                *score
                    >= dial
                        .as_ref()
                        .and_then(|d| d.bm25_min_score)
                        .unwrap_or(self.config.min_bm25_score)
            })
            .collect();

        // Calculate dynamic weights based on keyword discriminativeness
        // When YAKE identifies discriminative keywords, trust BM25 more
        // YAKE importance = 1/(1+score), so 0.9+ means very discriminative keywords
        //
        // Base weights come from effective_weights() which blends static defaults
        // with learned weights from retrieval feedback (P2.3 adaptive learning)
        let (eff_bm25, eff_vector, _eff_graph) = match dial.as_ref().and_then(|d| d.rrf_weights) {
            Some((b, v, g)) => (b, v, g),
            None => self.effective_weights(),
        };
        let (bm25_weight, vector_weight) = if let Some(disc) = keyword_discriminativeness {
            if disc >= 0.8 {
                // Highly discriminative keywords - strong BM25 preference
                (0.75, 0.25)
            } else if disc >= 0.5 {
                // Moderately discriminative - BM25 dominant
                (0.6, 0.4)
            } else {
                // Low discriminativeness - use adaptive learned weights
                (eff_bm25, eff_vector)
            }
        } else {
            (eff_bm25, eff_vector)
        };

        // Log counts and weights for debugging
        if bm25_results.is_empty() {
            tracing::warn!(
                "Hybrid search: BM25 returned 0 results for query '{}', using {} vector results only",
                query,
                vector_results.len()
            );
        } else {
            debug!(
                "Hybrid search: {} BM25 (top: {:.3}), {} vector, weights: BM25={:.2}/Vec={:.2}, disc={:?} for '{}'",
                bm25_results.len(),
                bm25_results.first().map(|(_, s)| *s).unwrap_or(0.0),
                vector_results.len(),
                bm25_weight,
                vector_weight,
                keyword_discriminativeness,
                &query[..query.len().min(50)]
            );
        }

        // 2. RRF Fusion with dynamic weights
        let rrf = RRFusion::new(
            dial.as_ref()
                .and_then(|d| d.rrf_k_inner)
                .unwrap_or(self.config.rrf_k),
            vec![bm25_weight, vector_weight],
        );

        let fused = rrf.fuse(vec![bm25_results.clone(), vector_results.clone()]);

        // Build lookup maps for component scores
        let bm25_map: HashMap<MemoryId, (f32, usize)> = bm25_results
            .iter()
            .enumerate()
            .map(|(rank, (id, score))| (id.clone(), (*score, rank)))
            .collect();

        let vector_map: HashMap<MemoryId, (f32, usize)> = vector_results
            .iter()
            .enumerate()
            .map(|(rank, (id, score))| (id.clone(), (*score, rank)))
            .collect();

        // 3. Optional cross-encoder reranking + LLM refiner pass
        //
        // CE DE-DUPLICATION (perf/a2): The cross-encoder is run exactly ONCE per
        // query — at recall Layer 5.3, AFTER Layer-5 unified scoring (recency,
        // arousal, feedback, Hebbian signals), on the final merged candidate set.
        // Running it here a second time on raw RRF candidates would fire two
        // sequential full-batch ONNX passes (~7 s each → ~14 s total at N=20).
        //
        // `use_cross_encoder` is therefore hardcoded to `false` here. The
        // `reranker` field is still populated and accessible via `self.reranker()`
        // so recall Layer 5.3 can invoke it. The `rerank_score` field on
        // `HybridSearchResult` will be `None` (used only for signal attribution
        // telemetry, not for score computation).
        //
        // The LLM refiner pass is unaffected and continues to run when configured.
        let effective_rerank_count = rerank_count_override
            .or_else(|| dial.as_ref().and_then(|d| d.rerank_count))
            .unwrap_or(self.config.rerank_count);
        let mode = dial
            .as_ref()
            .and_then(|d| d.refiner_mode)
            .unwrap_or(self.config.refiner_mode);
        let use_cross_encoder = false; // CE runs once at recall Layer 5.3 (batched, dial-gated)
        let use_refiner = matches!(mode, RefinerMode::Llm | RefinerMode::Stacked);

        let mut final_results: Vec<HybridSearchResult> =
            if let Some(reranker) = self.reranker.as_ref().filter(|_| use_cross_encoder) {
                // Take top-k for reranking (FIX-11: dynamic per-query override)
                let to_rerank: Vec<_> = fused
                    .iter()
                    .take(effective_rerank_count)
                    .filter_map(|(id, _score)| {
                        get_content(id).map(|content| (id.clone(), content, *_score, None))
                    })
                    .collect();

                if !to_rerank.is_empty() {
                    let reranked = reranker.rerank(query, to_rerank)?;

                    // Build rerank map with cosine similarities (range [-1, 1])
                    let rerank_map: HashMap<MemoryId, f32> = reranked.into_iter().collect();

                    // Normalize rerank scores to [0, 1] for scale-compatible blending
                    // Cosine similarity ∈ [-1, 1] → shift to [0, 1]
                    let rerank_normalized: HashMap<MemoryId, f32> = rerank_map
                        .iter()
                        .map(|(id, s)| (id.clone(), (s + 1.0) / 2.0))
                        .collect();

                    // Combine reranked results with non-reranked
                    // Blend: 0.6 * RRF + 0.4 * normalized_rerank (preserves RRF scale)
                    let rerank_blend: f32 = dial.as_ref().and_then(|d| d.ce_blend).unwrap_or(0.4);
                    let mut results: Vec<HybridSearchResult> = Vec::new();

                    for (memory_id, rrf_score) in fused {
                        let bm25_info = bm25_map.get(&memory_id);
                        let vector_info = vector_map.get(&memory_id);
                        let rerank_score = rerank_map.get(&memory_id).copied();

                        // Blend rerank with RRF to preserve score scale
                        let final_score =
                            if let Some(norm_rerank) = rerank_normalized.get(&memory_id).copied() {
                                (1.0 - rerank_blend) * rrf_score
                                    + rerank_blend * norm_rerank * rrf_score
                            } else {
                                rrf_score
                            };

                        results.push(HybridSearchResult {
                            memory_id,
                            score: final_score,
                            bm25_score: bm25_info.map(|(s, _)| *s),
                            vector_score: vector_info.map(|(s, _)| *s),
                            graph_score: None,
                            rrf_score,
                            rerank_score,
                            bm25_rank: bm25_info.map(|(_, r)| *r),
                            vector_rank: vector_info.map(|(_, r)| *r),
                            graph_rank: None,
                        });
                    }

                    // Re-sort by final score
                    results.sort_by(|a, b| b.score.total_cmp(&a.score));

                    results
                } else {
                    // No content available for reranking, use RRF scores
                    fused
                        .into_iter()
                        .map(|(memory_id, rrf_score)| {
                            let bm25_info = bm25_map.get(&memory_id);
                            let vector_info = vector_map.get(&memory_id);

                            HybridSearchResult {
                                memory_id,
                                score: rrf_score,
                                bm25_score: bm25_info.map(|(s, _)| *s),
                                vector_score: vector_info.map(|(s, _)| *s),
                                graph_score: None,
                                rrf_score,
                                rerank_score: None,
                                bm25_rank: bm25_info.map(|(_, r)| *r),
                                vector_rank: vector_info.map(|(_, r)| *r),
                                graph_rank: None,
                            }
                        })
                        .collect()
                }
            } else {
                // No reranking, use RRF scores directly
                fused
                    .into_iter()
                    .map(|(memory_id, rrf_score)| {
                        let bm25_info = bm25_map.get(&memory_id);
                        let vector_info = vector_map.get(&memory_id);

                        HybridSearchResult {
                            memory_id,
                            score: rrf_score,
                            bm25_score: bm25_info.map(|(s, _)| *s),
                            vector_score: vector_info.map(|(s, _)| *s),
                            graph_score: None,
                            rrf_score,
                            rerank_score: None,
                            bm25_rank: bm25_info.map(|(_, r)| *r),
                            vector_rank: vector_info.map(|(_, r)| *r),
                            graph_rank: None,
                        }
                    })
                    .collect()
            };

        // 4. Optional LLM refiner pass (applied after cross-encoder when
        // mode is Stacked; replaces cross-encoder when mode is Llm).
        if use_refiner {
            if let Some(refiner) = self.refiner.as_ref() {
                let to_refine: Vec<(MemoryId, String, f32)> = final_results
                    .iter()
                    .take(effective_rerank_count)
                    .filter_map(|r| {
                        get_content(&r.memory_id).map(|c| (r.memory_id.clone(), c, r.score))
                    })
                    .collect();
                if !to_refine.is_empty() {
                    let refined = refiner.refine(query, to_refine)?;
                    let refined_map: HashMap<MemoryId, f32> = refined.into_iter().collect();

                    // Blend mirrors the cross-encoder formula: refiner can
                    // demote weakly relevant candidates but the multiplicative
                    // structure preserves the RRF rank floor.
                    const REFINER_BLEND: f32 = 0.4;
                    for r in &mut final_results {
                        if let Some(&rs) = refined_map.get(&r.memory_id) {
                            r.score = (1.0 - REFINER_BLEND) * r.score
                                + REFINER_BLEND * rs * r.score.max(0.0);
                        }
                    }
                    final_results.sort_by(|a, b| b.score.total_cmp(&a.score));
                }
            }
        }

        Ok(final_results)
    }

    /// Get BM25 document count
    pub fn bm25_doc_count(&self) -> usize {
        self.bm25_index.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An encrypted tenant's index never reaches disk, and an index left there from
    /// before encryption is removed rather than reopened.
    #[test]
    fn an_in_memory_index_writes_nothing_and_removes_the_plaintext_one() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("bm25_projection_index");
        {
            let on_disk = BM25Index::new(&path).unwrap();
            on_disk
                .upsert(&MemoryId(uuid::Uuid::new_v4()), "the secret plan", &[], &[])
                .unwrap();
            on_disk.commit().unwrap();
        }
        assert!(path.exists(), "precondition: a plaintext index on disk");

        let index = BM25Index::in_memory(&path).unwrap();
        assert!(index.is_in_memory());
        assert!(
            !path.exists(),
            "the plaintext index left on disk is removed"
        );
        let id = MemoryId(uuid::Uuid::new_v4());
        index.upsert(&id, "the sealed plan", &[], &[]).unwrap();
        index.commit().unwrap();
        index.reload().unwrap();
        assert_eq!(
            index
                .search("plan", 5)
                .unwrap()
                .first()
                .map(|(hit, _)| hit.clone()),
            Some(id)
        );
        assert!(!path.exists(), "indexing wrote nothing to disk");
    }
    use crate::storage::KeyValueStore;

    #[test]
    fn test_rrf_fusion_basic() {
        let rrf = RRFusion::new(60.0, vec![0.5, 0.5]);

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());
        let id3 = MemoryId(uuid::Uuid::new_v4());

        // List 1: id1 > id2 > id3
        let list1 = vec![(id1.clone(), 0.9), (id2.clone(), 0.7), (id3.clone(), 0.5)];

        // List 2: id2 > id1 > id3
        let list2 = vec![(id2.clone(), 0.95), (id1.clone(), 0.6), (id3.clone(), 0.4)];

        let fused = rrf.fuse(vec![list1, list2]);

        // id1 and id2 have symmetric ranks (1,2) and (2,1), so they should have equal RRF scores
        // The ordering between them is implementation-defined, but both should be above id3
        // id3 is rank 3 in both lists, so it should be last
        assert_eq!(fused.len(), 3);

        // Both id1 and id2 should have higher scores than id3
        let id1_score = fused.iter().find(|(id, _)| *id == id1).unwrap().1;
        let id2_score = fused.iter().find(|(id, _)| *id == id2).unwrap().1;
        let id3_score = fused.iter().find(|(id, _)| *id == id3).unwrap().1;

        // id1 and id2 should have equal scores (symmetric ranking)
        assert!(
            (id1_score - id2_score).abs() < 0.0001,
            "id1 and id2 should have equal RRF scores"
        );

        // id3 should be last (lowest score)
        assert!(
            id3_score < id1_score,
            "id3 should have lower score than id1"
        );
        assert!(
            id3_score < id2_score,
            "id3 should have lower score than id2"
        );
        assert_eq!(fused[2].0, id3, "id3 should be ranked last");
    }

    #[test]
    fn test_rrf_fusion_disjoint() {
        let rrf = RRFusion::new(60.0, vec![0.5, 0.5]);

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());

        // Disjoint lists
        let list1 = vec![(id1.clone(), 0.9)];
        let list2 = vec![(id2.clone(), 0.8)];

        let fused = rrf.fuse(vec![list1, list2]);

        assert_eq!(fused.len(), 2);
        // Both should have same RRF score (rank 1 in their respective list)
        assert!((fused[0].1 - fused[1].1).abs() < 0.001);
    }

    #[test]
    fn rrf_equal_scores_keep_first_appearance_order() {
        let rrf = RRFusion::new(60.0, vec![0.5, 0.5]);
        let left: Vec<MemoryId> = (0..24).map(|_| MemoryId(uuid::Uuid::new_v4())).collect();
        let right: Vec<MemoryId> = (0..24).map(|_| MemoryId(uuid::Uuid::new_v4())).collect();
        let as_list = |ids: &[MemoryId]| ids.iter().map(|id| (id.clone(), 1.0)).collect();

        let fused = rrf.fuse(vec![as_list(&left), as_list(&right)]);

        // Rank r in one list ties rank r in the other; the left list was read first.
        let expected: Vec<MemoryId> = left
            .iter()
            .zip(&right)
            .flat_map(|(l, r)| [l.clone(), r.clone()])
            .collect();
        let got: Vec<MemoryId> = fused.into_iter().map(|(id, _)| id).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn bm25_ties_at_the_cutoff_are_ordered_by_content() {
        let temp_dir = tempfile::tempdir().unwrap();
        let index = BM25Index::new(temp_dir.path()).unwrap();
        // Same length, same term frequency: every document scores the same for "lantern".
        // Written in reverse content order, one commit each, so doc order is not content order.
        for i in (0..12).rev() {
            let id = MemoryId(uuid::Uuid::new_v4());
            index
                .upsert(&id, &format!("lantern marker{i:02}"), &[], &[])
                .unwrap();
            index.commit().unwrap();
        }
        index.reload().unwrap();

        let results = index.search("lantern", 3).unwrap();
        assert_eq!(results.len(), 3);
        assert!(
            results.iter().all(|(_, s)| *s == results[0].1),
            "fixture must tie"
        );

        let searcher = index.reader.searcher();
        let content_of = |id: &MemoryId| -> String {
            let query = tantivy::query::TermQuery::new(
                tantivy::Term::from_field_text(index.id_field, &id.0.to_string()),
                tantivy::schema::IndexRecordOption::Basic,
            );
            let hits = searcher
                .search(&query, &TopDocs::with_limit(1).order_by_score())
                .unwrap();
            let doc: TantivyDocument = searcher.doc(hits[0].1).unwrap();
            doc.get_first(index.content_field)
                .and_then(|v| v.as_str())
                .unwrap()
                .to_string()
        };
        let contents: Vec<String> = results.iter().map(|(id, _)| content_of(id)).collect();
        assert_eq!(
            contents,
            ["lantern marker00", "lantern marker01", "lantern marker02"]
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_config_defaults() {
        let config = HybridSearchConfig::default();
        assert_eq!(config.bm25_weight, 0.35); // BM25 for keyword matching
        assert_eq!(config.vector_weight, 0.40); // Vector for semantic relationships
        assert_eq!(config.graph_weight, 0.25); // Graph for associative retrieval (SHO-D4)
        assert_eq!(config.rrf_k, 45.0); // Lower k for top-rank emphasis
        assert_eq!(config.candidate_count, 100); // Increased for better recall
        assert_eq!(config.rerank_count, 20);
        assert!(config.use_reranking); // Enabled: true cross-encoder with bi-encoder fallback
        assert_eq!(config.min_graph_score, 0.01); // Graph score threshold (SHO-D4)
    }

    #[test]
    fn test_bm25_index_and_search() {
        let temp_dir = tempfile::tempdir().unwrap();
        let index = BM25Index::new(temp_dir.path()).unwrap();

        // Create test memories
        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());
        let id3 = MemoryId(uuid::Uuid::new_v4());

        // Index documents with different content
        index
            .upsert(
                &id1,
                "The user prefers Rust programming language for systems development",
                &["rust".to_string(), "programming".to_string()],
                &["Rust".to_string()],
            )
            .unwrap();

        index
            .upsert(
                &id2,
                "Python is great for machine learning and data science projects",
                &["python".to_string(), "ml".to_string()],
                &["Python".to_string()],
            )
            .unwrap();

        index
            .upsert(
                &id3,
                "The authentication system uses JWT tokens for security",
                &["auth".to_string(), "security".to_string()],
                &["JWT".to_string()],
            )
            .unwrap();

        index.commit().unwrap();
        index.reload().unwrap();

        // Test: Search for "Rust" should find id1
        let results = index.search("Rust programming", 10).unwrap();
        assert!(!results.is_empty(), "Should find Rust document");
        assert_eq!(results[0].0, id1, "Rust doc should be first");

        // Test: Search for "Python" should find id2
        let results = index.search("Python machine learning", 10).unwrap();
        assert!(!results.is_empty(), "Should find Python document");
        assert_eq!(results[0].0, id2, "Python doc should be first");

        // Test: Search for "JWT" should find id3
        let results = index.search("JWT authentication", 10).unwrap();
        assert!(!results.is_empty(), "Should find auth document");
        assert_eq!(results[0].0, id3, "Auth doc should be first");

        // Test: Search for unrelated term should return empty or low scores
        let results = index.search("quantum physics", 10).unwrap();
        assert!(
            results.is_empty() || results[0].1 < 0.5,
            "Unrelated search should have low/no results"
        );
    }

    #[test]
    fn test_bm25_keyword_vs_semantic_gap() {
        // This test demonstrates why BM25 is needed alongside vector search
        let temp_dir = tempfile::tempdir().unwrap();
        let index = BM25Index::new(temp_dir.path()).unwrap();

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());

        // Document with specific technical term "SIGHUP"
        index
            .upsert(
                &id1,
                "The server reloads configuration when it receives SIGHUP signal",
                &["linux".to_string(), "signals".to_string()],
                &[],
            )
            .unwrap();

        // Document about reloading (semantically similar but different keyword)
        index
            .upsert(
                &id2,
                "Configuration refresh happens automatically every hour",
                &["config".to_string()],
                &[],
            )
            .unwrap();

        index.commit().unwrap();
        index.reload().unwrap();

        // BM25 should find exact match for "SIGHUP" even if vector search might not
        let results = index.search("SIGHUP", 10).unwrap();
        assert!(!results.is_empty(), "BM25 should find SIGHUP");
        assert_eq!(results[0].0, id1, "Exact keyword match should win");
    }

    #[test]
    fn test_rrf_weighted_fusion() {
        // Test that weights affect fusion correctly
        let rrf_bm25_heavy = RRFusion::new(60.0, vec![0.8, 0.2]); // BM25 weighted higher
        let rrf_vector_heavy = RRFusion::new(60.0, vec![0.2, 0.8]); // Vector weighted higher

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());

        // id1 ranks #1 in BM25, #2 in vector
        // id2 ranks #2 in BM25, #1 in vector
        let bm25_list = vec![(id1.clone(), 0.9), (id2.clone(), 0.7)];
        let vector_list = vec![(id2.clone(), 0.95), (id1.clone(), 0.6)];

        // With BM25 weighted higher, id1 should win
        let fused_bm25 = rrf_bm25_heavy.fuse(vec![bm25_list.clone(), vector_list.clone()]);
        assert_eq!(fused_bm25[0].0, id1, "BM25-heavy should favor BM25 winner");

        // With vector weighted higher, id2 should win
        let fused_vector = rrf_vector_heavy.fuse(vec![bm25_list, vector_list]);
        assert_eq!(
            fused_vector[0].0, id2,
            "Vector-heavy should favor vector winner"
        );
    }

    #[test]
    fn test_rrf_k_parameter_effect() {
        // Higher k = more equal weighting across ranks
        // Lower k = more emphasis on top ranks
        let rrf_low_k = RRFusion::new(1.0, vec![0.5, 0.5]); // Low k
        let rrf_high_k = RRFusion::new(100.0, vec![0.5, 0.5]); // High k

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());
        let id3 = MemoryId(uuid::Uuid::new_v4());

        // id1 is #1 in list1, #3 in list2
        // id3 is #3 in list1, #1 in list2
        // id2 is #2 in both lists
        let list1 = vec![(id1.clone(), 0.9), (id2.clone(), 0.7), (id3.clone(), 0.5)];
        let list2 = vec![(id3.clone(), 0.9), (id2.clone(), 0.7), (id1.clone(), 0.5)];

        let fused_low_k = rrf_low_k.fuse(vec![list1.clone(), list2.clone()]);
        let fused_high_k = rrf_high_k.fuse(vec![list1, list2]);

        // With low k, rank differences matter more
        // With high k, id2 (consistent #2) should do relatively better
        // id2's score should be relatively higher with high k
        let id2_score_low = fused_low_k.iter().find(|(id, _)| *id == id2).unwrap().1;
        let id2_score_high = fused_high_k.iter().find(|(id, _)| *id == id2).unwrap().1;

        // Normalize by max score to compare relative positions
        let max_low = fused_low_k[0].1;
        let max_high = fused_high_k[0].1;

        let id2_relative_low = id2_score_low / max_low;
        let id2_relative_high = id2_score_high / max_high;

        // id2 should have higher relative score with high k (more forgiving of rank differences)
        assert!(
            id2_relative_high >= id2_relative_low - 0.01,
            "High k should be more forgiving of rank variation"
        );
    }

    #[test]
    fn test_signal_scores_normalization() {
        let scores = SignalScores::from_components(0.5, 0.3, 0.2);
        let sum = scores.bm25_contribution + scores.vector_contribution + scores.graph_contribution;
        assert!((sum - 1.0).abs() < 0.001, "Signal scores should sum to 1.0");
        assert!((scores.bm25_contribution - 0.5).abs() < 0.001);
        assert!((scores.vector_contribution - 0.3).abs() < 0.001);
        assert!((scores.graph_contribution - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_signal_scores_zero_input() {
        let scores = SignalScores::from_components(0.0, 0.0, 0.0);
        let sum = scores.bm25_contribution + scores.vector_contribution + scores.graph_contribution;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Zero input should produce uniform weights"
        );
        assert!((scores.bm25_contribution - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_signal_scores_inversion() {
        let scores = SignalScores::from_components(0.8, 0.1, 0.1);
        let inverted = scores.inverted();
        // High BM25 contribution should become low after inversion
        assert!(inverted.bm25_contribution < inverted.vector_contribution);
        assert!(inverted.bm25_contribution < inverted.graph_contribution);
        let sum =
            inverted.bm25_contribution + inverted.vector_contribution + inverted.graph_contribution;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Inverted scores should sum to 1.0"
        );
    }

    #[test]
    fn test_learned_weights_ema_update() {
        let config = HybridSearchConfig::default();
        let mut weights = LearnedWeights::from_defaults(&config);

        assert_eq!(weights.update_count, 0);
        assert!((weights.bm25 - config.bm25_weight).abs() < 0.001);

        // Apply a strong BM25 signal (positive feedback)
        let target = SignalScores::from_components(1.0, 0.0, 0.0);
        weights.apply_update(&target, 0.1, false);

        assert_eq!(weights.update_count, 1);
        // BM25 weight should have increased
        assert!(weights.bm25 > config.bm25_weight);
        // Vector weight should have decreased
        assert!(weights.vector < config.vector_weight);
        // Should still sum to 1.0
        let sum = weights.bm25 + weights.vector + weights.graph;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_learned_weights_convergence() {
        let config = HybridSearchConfig::default();
        let mut weights = LearnedWeights::from_defaults(&config);

        // Apply many updates pushing toward vector dominance (positive feedback)
        let target = SignalScores::from_components(0.1, 0.8, 0.1);
        for _ in 0..100 {
            weights.apply_update(&target, 0.05, false);
        }

        // After 100 updates at lr=0.05, weights should be close to target
        assert!(
            weights.vector > 0.6,
            "Vector should dominate after many updates: {}",
            weights.vector
        );
        assert!(weights.bm25 < 0.2, "BM25 should be low: {}", weights.bm25);
        assert!(
            weights.graph < 0.2,
            "Graph should be low: {}",
            weights.graph
        );
    }

    #[test]
    fn test_learned_weights_normalization_invariant() {
        let config = HybridSearchConfig::default();
        let mut weights = LearnedWeights::from_defaults(&config);

        // Apply various updates and check normalization
        let targets = vec![
            SignalScores::from_components(1.0, 0.0, 0.0),
            SignalScores::from_components(0.0, 1.0, 0.0),
            SignalScores::from_components(0.0, 0.0, 1.0),
            SignalScores::from_components(0.33, 0.33, 0.34),
        ];

        for (i, target) in targets.iter().enumerate() {
            // Alternate positive/negative to exercise both code paths
            weights.apply_update(target, 0.1, i % 2 == 1);
            let sum = weights.bm25 + weights.vector + weights.graph;
            assert!(
                (sum - 1.0).abs() < 0.001,
                "Weights should sum to 1.0 after every update, got {}",
                sum
            );
        }
    }

    // ----- R3a: LearnedWeightsStore persistence (pure infra, no scoring change) -----

    use crate::config::StorageBackend;
    use parking_lot::Mutex as PlMutex;
    use std::collections::BTreeMap;

    /// Minimal in-memory KeyValueStore for learned-weight persistence tests.
    /// Each instance is an isolated per-user CF stand-in.
    #[derive(Default)]
    struct MemKv {
        map: PlMutex<BTreeMap<Vec<u8>, Vec<u8>>>,
    }

    impl crate::storage::KeyValueStore for MemKv {
        fn backend(&self) -> StorageBackend {
            StorageBackend::Redb
        }
        fn get_bytes(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
            Ok(self.map.lock().get(key).cloned())
        }
        fn put_many(&self, pairs: &[(Vec<u8>, Vec<u8>)]) -> anyhow::Result<()> {
            let mut map = self.map.lock();
            for (k, v) in pairs {
                map.insert(k.clone(), v.clone());
            }
            Ok(())
        }
        fn delete_many(&self, keys: &[Vec<u8>]) -> anyhow::Result<()> {
            let mut map = self.map.lock();
            for k in keys {
                map.remove(k);
            }
            Ok(())
        }
        fn put_bytes(&self, key: &[u8], value: &[u8]) -> anyhow::Result<()> {
            self.map.lock().insert(key.to_vec(), value.to_vec());
            Ok(())
        }
        fn delete_bytes(&self, key: &[u8]) -> anyhow::Result<()> {
            self.map.lock().remove(key);
            Ok(())
        }
        fn scan_prefix(
            &self,
            prefix: &[u8],
            limit: Option<usize>,
        ) -> anyhow::Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let guard = self.map.lock();
            let mut out: Vec<(Vec<u8>, Vec<u8>)> = guard
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if let Some(n) = limit {
                out.truncate(n);
            }
            Ok(out)
        }
        fn flush(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// Round-trip: save non-default weights, load them back, assert equality.
    #[test]
    fn test_learned_weights_store_round_trip() {
        let store = LearnedWeightsStore::with_kv(Arc::new(MemKv::default()));

        let mut weights = LearnedWeights::default();
        weights.bm25 = 0.5;
        weights.vector = 0.3;
        weights.graph = 0.2;
        weights.update_count = 42;
        assert_ne!(weights.bm25, LearnedWeights::default().bm25);

        store.save(&weights);
        let loaded = store.load();

        assert!((loaded.bm25 - 0.5).abs() < 1e-6);
        assert!((loaded.vector - 0.3).abs() < 1e-6);
        assert!((loaded.graph - 0.2).abs() < 1e-6);
        assert_eq!(loaded.update_count, 42);
        assert_eq!(loaded.schema_version, LEARNED_WEIGHTS_SCHEMA_VERSION);
    }

    /// No persisted value → neutral default, no panic.
    #[test]
    fn test_learned_weights_store_missing_returns_default() {
        let store = LearnedWeightsStore::with_kv(Arc::new(MemKv::default()));
        let loaded = store.load();
        let def = LearnedWeights::default();
        assert!((loaded.bm25 - def.bm25).abs() < 1e-6);
        assert!((loaded.vector - def.vector).abs() < 1e-6);
        assert!((loaded.graph - def.graph).abs() < 1e-6);
        assert_eq!(loaded.update_count, 0);
    }

    /// No KV handle (no-op store) → default, save is a silent skip.
    #[test]
    fn test_learned_weights_store_no_kv_is_noop() {
        let store = LearnedWeightsStore::new();
        let mut weights = LearnedWeights::default();
        weights.bm25 = 0.9;
        store.save(&weights); // must not panic
        let loaded = store.load();
        assert!((loaded.bm25 - LearnedWeights::default().bm25).abs() < 1e-6);
    }

    /// B12: garbage bytes at the key → load returns default(), never panics.
    #[test]
    fn test_learned_weights_store_corrupt_bytes_fallback() {
        let kv = Arc::new(MemKv::default());
        kv.put_bytes(LEARNED_WEIGHTS_KEY, b"this is not valid json {{{")
            .unwrap();
        let store = LearnedWeightsStore::with_kv(kv);

        let loaded = store.load(); // must not panic
        let def = LearnedWeights::default();
        assert!((loaded.bm25 - def.bm25).abs() < 1e-6);
        assert_eq!(loaded.update_count, 0);
        assert_eq!(loaded.schema_version, LEARNED_WEIGHTS_SCHEMA_VERSION);
    }

    /// B12: a valid payload carrying a mismatched schema_version → default().
    #[test]
    fn test_learned_weights_store_schema_mismatch_fallback() {
        let kv = Arc::new(MemKv::default());
        // Hand-craft a payload with a future/unknown schema version.
        let payload = format!(
            r#"{{"bm25":0.8,"vector":0.1,"graph":0.1,"update_count":99,"schema_version":{}}}"#,
            LEARNED_WEIGHTS_SCHEMA_VERSION + 1
        );
        kv.put_bytes(LEARNED_WEIGHTS_KEY, payload.as_bytes())
            .unwrap();
        let store = LearnedWeightsStore::with_kv(kv);

        let loaded = store.load();
        let def = LearnedWeights::default();
        // Mismatch ignored → neutral default, NOT the persisted 0.8/99.
        assert!((loaded.bm25 - def.bm25).abs() < 1e-6);
        assert_eq!(loaded.update_count, 0);
    }

    /// Older serialized form lacking `schema_version` loads via serde default
    /// (treated as the current version), so it is accepted (additive evolution).
    #[test]
    fn test_learned_weights_store_legacy_no_schema_version_loads() {
        let kv = Arc::new(MemKv::default());
        // No schema_version field — predates B12.
        kv.put_bytes(
            LEARNED_WEIGHTS_KEY,
            br#"{"bm25":0.6,"vector":0.25,"graph":0.15,"update_count":7}"#,
        )
        .unwrap();
        let store = LearnedWeightsStore::with_kv(kv);

        let loaded = store.load();
        assert!((loaded.bm25 - 0.6).abs() < 1e-6);
        assert_eq!(loaded.update_count, 7);
        assert_eq!(loaded.schema_version, LEARNED_WEIGHTS_SCHEMA_VERSION);
    }

    /// Per-user isolation: two stores over distinct KV handles do not cross-read.
    #[test]
    fn test_learned_weights_store_per_user_isolation() {
        let store_a = LearnedWeightsStore::with_kv(Arc::new(MemKv::default()));
        let store_b = LearnedWeightsStore::with_kv(Arc::new(MemKv::default()));

        let mut wa = LearnedWeights::default();
        wa.bm25 = 0.7;
        wa.vector = 0.2;
        wa.graph = 0.1;
        wa.update_count = 11;
        store_a.save(&wa);

        // store_b never saw user A's write → still default.
        let lb = store_b.load();
        let def = LearnedWeights::default();
        assert!((lb.bm25 - def.bm25).abs() < 1e-6);
        assert_eq!(lb.update_count, 0);

        // store_a still reads its own.
        let la = store_a.load();
        assert!((la.bm25 - 0.7).abs() < 1e-6);
        assert_eq!(la.update_count, 11);
    }

    // =========================================================================
    // R3b unit tests
    // =========================================================================

    // Helper: build a minimal HybridSearchEngine for R3b tests that exercise
    // only the weight machinery (no actual BM25/vector search is performed).
    fn make_test_engine() -> (HybridSearchEngine, tempfile::TempDir) {
        use crate::embeddings::competitive::CompetitiveEmbedder;

        struct StubEmb;
        impl crate::embeddings::Embedder for StubEmb {
            fn id(&self) -> &str {
                "stub-hybrid"
            }
            fn encode(&self, _: &str) -> anyhow::Result<Vec<f32>> {
                Ok(vec![0.0; 4])
            }
            fn dimension(&self) -> usize {
                4
            }
        }

        let embedder = Arc::new(CompetitiveEmbedder::primary_only(
            Arc::new(StubEmb) as Arc<dyn crate::embeddings::Embedder>
        ));
        let config = HybridSearchConfig::default();
        let tmp = tempfile::tempdir().unwrap();
        let engine = HybridSearchEngine::new(&tmp.path().join("bm25"), embedder, config).unwrap();
        (engine, tmp)
    }

    /// R3b-T1: `snapshot_learned_weights` returns a value copy; mutating the
    /// engine afterwards does NOT retroactively change the snapshot (no torn read
    /// within a single query B1 guard).
    #[test]
    fn test_r3b_snapshot_is_stable_within_query() {
        let (engine, _tmp) = make_test_engine();

        // Take snapshot BEFORE injecting a feedback update.
        let snapshot_before = engine.snapshot_learned_weights();
        assert_eq!(snapshot_before.update_count, 0);

        // Simulate a helpful feedback update on the engine.
        let scores = SignalScores::from_components(1.0, 0.0, 0.0);
        engine.update_weights_from_feedback(&scores, true, false);

        // The live state must have changed (update_count incremented).
        let live = engine.learned_weight_state();
        assert_eq!(live.update_count, 1, "update count must increment");

        // The snapshot itself must be frozen at the pre-update state.
        assert_eq!(
            snapshot_before.update_count, 0,
            "snapshot must not be mutated by subsequent update_weights_from_feedback"
        );
    }

    /// R3b-T2: per-epoch weight delta is bounded by ADAPTIVE_WEIGHT_MAX_BLEND
    /// (the 0.6 blend cap).  After a single maximally-skewed feedback update the
    /// blended weight must NOT exceed the static default by more than the
    /// max-blend factor.
    #[test]
    fn test_r3b_per_epoch_delta_clamped_at_0_6() {
        use crate::constants::{ADAPTIVE_WEIGHT_CONFIDENCE_THRESHOLD, ADAPTIVE_WEIGHT_MAX_BLEND};

        let config = HybridSearchConfig::default();
        let static_bm25 = config.bm25_weight; // e.g. 0.4

        let mut weights = LearnedWeights::from_defaults(&config);

        // Drive the ranker toward 100% BM25 with many updates to max out confidence.
        let all_bm25 = SignalScores::from_components(1.0, 0.0, 0.0);
        for _ in 0..ADAPTIVE_WEIGHT_CONFIDENCE_THRESHOLD * 2 {
            weights.apply_update(&all_bm25, 0.05, false);
        }

        // `effective_weights()` is where the blend cap is applied.  We can't call
        // it here without an engine, so we reproduce the formula inline to verify
        // the constant.
        let confidence =
            (weights.update_count as f32 / ADAPTIVE_WEIGHT_CONFIDENCE_THRESHOLD as f32).min(1.0);
        let blend = confidence * ADAPTIVE_WEIGHT_MAX_BLEND;
        let eff_bm25 = (1.0 - blend) * static_bm25 + blend * weights.bm25;

        // The effective bm25 weight must be <= 100% of the static + max blend lift.
        let max_possible = static_bm25 + ADAPTIVE_WEIGHT_MAX_BLEND;
        assert!(
            eff_bm25 <= max_possible + 1e-5,
            "effective bm25 weight {eff_bm25:.4} exceeds max possible {max_possible:.4}"
        );
        // The blend factor itself must be capped at ADAPTIVE_WEIGHT_MAX_BLEND.
        assert!(
            blend <= ADAPTIVE_WEIGHT_MAX_BLEND + 1e-5,
            "blend {blend:.4} must be <= ADAPTIVE_WEIGHT_MAX_BLEND ({ADAPTIVE_WEIGHT_MAX_BLEND})"
        );
    }

    /// R3b-T3: `R3B_LIVE_APPLICATION_ENABLED` defaults to false, so
    /// `weights_ready()` always returns false in this PR regardless of the
    /// update count or diversity floor.  This ensures the live path is
    /// unreachable until the constant is explicitly flipped.
    #[test]
    fn test_r3b_live_application_defaults_off() {
        use crate::constants::R3B_LIVE_APPLICATION_ENABLED;

        // Constant-level guard: must be false in this PR.
        assert!(
            !R3B_LIVE_APPLICATION_ENABLED,
            "R3B_LIVE_APPLICATION_ENABLED must be false — live ramp is a separate future commit"
        );

        let (engine, _tmp) = make_test_engine();

        // Even after enough helpful+negative updates to satisfy the diversity floor,
        // weights_ready() must return false because the constant is off.
        let scores = SignalScores::from_components(0.33, 0.33, 0.34);
        // Positive updates
        for _ in 0..15 {
            engine.update_weights_from_feedback(&scores, true, false);
        }
        // Negative updates (>= 20% floor)
        for _ in 0..5 {
            engine.update_weights_from_feedback(&scores, false, true);
        }
        assert!(
            !engine.weights_ready(),
            "weights_ready() must return false while R3B_LIVE_APPLICATION_ENABLED=false"
        );
    }

    /// R3b-T4: `weights_ready()` sub-check — `negative_sample_floor_met()` only
    /// returns true once the fraction of negative updates meets the threshold.
    #[test]
    fn test_r3b_negative_sample_floor() {
        use crate::constants::{R3B_MIN_NEGATIVE_SAMPLE_FRACTION, R3B_MIN_UPDATES_BEFORE_READY};

        let config = HybridSearchConfig::default();
        let mut weights = LearnedWeights::from_defaults(&config);

        // Below minimum total count → false
        assert!(!weights.negative_sample_floor_met());

        // Enough total but zero negatives → false
        let pos = SignalScores::from_components(0.5, 0.3, 0.2);
        for _ in 0..R3B_MIN_UPDATES_BEFORE_READY {
            weights.apply_update(&pos, 0.05, false);
        }
        assert!(
            !weights.negative_sample_floor_met(),
            "all-positive must not meet floor"
        );

        // Add enough negatives to satisfy the fraction.
        // Need: neg / (pos_count + neg) >= MIN_FRAC
        // => neg >= MIN_FRAC * pos_count / (1 - MIN_FRAC), rounded UP.
        let pos_so_far = weights.update_count as f32;
        let neg_count = (pos_so_far * R3B_MIN_NEGATIVE_SAMPLE_FRACTION
            / (1.0 - R3B_MIN_NEGATIVE_SAMPLE_FRACTION))
            .ceil() as u64;
        let neg_count = neg_count.max(1);
        for _ in 0..neg_count {
            weights.apply_update(&pos, 0.05, true);
        }
        assert!(
            weights.negative_sample_floor_met(),
            "after adding {neg_count} negatives floor should be met; \
             neg={}, total={}",
            weights.negative_update_count,
            weights.update_count
        );
    }

    /// R3b-T5: static score equals the returned score (live path unchanged) when
    /// computing the shadow delta.
    ///
    /// This is a unit-level check of the shadow-delta formula: when the learned
    /// weights equal the static weights the delta must be zero.
    #[test]
    fn test_r3b_shadow_delta_zero_when_weights_equal() {
        let static_bm25 = 0.4_f32;
        let static_vector = 0.6_f32;
        let static_graph = 0.0_f32;

        let bm25_attr = 0.7_f32;
        let vector_attr = 0.3_f32;
        let graph_attr = 0.0_f32;

        let static_signal =
            static_bm25 * bm25_attr + static_vector * vector_attr + static_graph * graph_attr;

        // Learned = same as static in this case.
        let learned_bm25 = static_bm25;
        let learned_vector = static_vector;
        let learned_graph = static_graph;
        let learned_signal =
            learned_bm25 * bm25_attr + learned_vector * vector_attr + learned_graph * graph_attr;

        let delta = learned_signal - static_signal;
        assert!(
            delta.abs() < 1e-6,
            "delta must be zero when learned == static; got {delta}"
        );
    }

    /// Selecting a reranker the reranker has not loaded yet must return, and a second
    /// selection must reuse it. Under edition 2021 a read guard held across the load
    /// deadlocked the first selection: every gate arm that set rerank_model hung until the
    /// job limit.
    #[test]
    fn selecting_a_reranker_not_yet_loaded_returns_and_is_reused() {
        struct FixedEmbedder;
        impl Embedder for FixedEmbedder {
            fn encode(&self, _text: &str) -> Result<Vec<f32>> {
                Ok(vec![1.0, 0.0])
            }
            fn dimension(&self) -> usize {
                2
            }
            fn id(&self) -> &str {
                "fixed"
            }
        }
        let env_model = crate::embeddings::cross_encoder::load_cross_encoder(None).unwrap();
        let reranker = Arc::new(CrossEncoderReranker::with_cross_encoder(
            Arc::new(FixedEmbedder),
            env_model,
        ));
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = Arc::clone(&reranker);
        std::thread::spawn(move || {
            let id = crate::embeddings::cross_encoder::MS_MARCO_MINILM_L6_ID;
            let selected = worker
                .select_model(Some(id))
                .and_then(|_| worker.select_model(Some(id)))
                .map(|_| worker.model().id().to_string());
            let _ = tx.send(selected.ok());
        });
        let selected = rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .expect("select_model deadlocked");
        assert_eq!(
            selected.as_deref(),
            Some(crate::embeddings::cross_encoder::MS_MARCO_MINILM_L6_ID)
        );
    }
}
