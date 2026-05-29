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
use std::sync::Arc;

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
use crate::memory::llm_reranker::Refiner;

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
/// neither) is applied to the fused candidate set. The `rlm-eval` harness
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
    Rlm,
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
    /// callers. The `rlm-eval` harness varies this to compare refiner
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

/// Learned retrieval weights from feedback history
///
/// Updated via EMA (exponential moving average) when retrieval feedback is received.
/// Blended with static defaults based on update_count confidence.
///
/// The system starts with static defaults (well-tuned from benchmarks) and gradually
/// shifts toward empirically-observed signal contributions as feedback accumulates.
#[derive(Debug, Clone)]
pub struct LearnedWeights {
    /// Learned BM25 weight (initialized to static default)
    pub bm25: f32,
    /// Learned vector weight (initialized to static default)
    pub vector: f32,
    /// Learned graph weight (initialized to static default)
    pub graph: f32,
    /// Number of feedback updates applied (confidence measure)
    pub update_count: u64,
}

impl LearnedWeights {
    /// Create learned weights initialized to static defaults
    pub fn from_defaults(config: &HybridSearchConfig) -> Self {
        Self {
            bm25: config.bm25_weight,
            vector: config.vector_weight,
            graph: config.graph_weight,
            update_count: 0,
        }
    }

    /// Apply an EMA update from feedback signal scores
    ///
    /// For helpful outcomes: target = signal scores (shift toward what worked)
    /// For misleading outcomes: target = inverted scores (shift away from what failed)
    pub fn apply_update(&mut self, target: &SignalScores, lr: f32) {
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
}

impl BM25Index {
    /// Create or open a BM25 index at the given path
    pub fn new(path: &Path) -> Result<Self> {
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

        let schema = schema_builder.build();

        // Create or open index
        std::fs::create_dir_all(path)?;
        let dir = tantivy::directory::MmapDirectory::open(path)
            .context("Failed to open tantivy directory")?;

        let index = if Index::exists(&dir)? {
            let existing = Index::open(dir).context("Failed to open existing BM25 index")?;
            // Schema migration: if existing index lacks stemmed_content, rebuild
            if existing.schema().get_field("stemmed_content").is_err() {
                tracing::warn!("BM25 schema missing stemmed_content — rebuilding index for stemming support");
                drop(existing);
                if let Err(e) = std::fs::remove_dir_all(path) {
                    tracing::warn!("Failed to remove old BM25 index: {e}");
                }
                std::fs::create_dir_all(path)?;
                Index::create_in_dir(path, schema).context("Failed to create BM25 index with stemmed_content")?
            } else {
                existing
            }
        } else {
            Index::create_in_dir(path, schema).context("Failed to create BM25 index")?
        };

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

        // 15MB writer heap — sufficient for edge workloads
        let writer = index
            .writer(15_000_000)
            .context("Failed to create index writer")?;

        let reader = index
            .reader_builder()
            .reload_policy(tantivy::ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create index reader")?;

        info!("BM25 index initialized at {:?}", path);

        Ok(Self {
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            id_field,
            content_field,
            tags_field,
            entities_field,
            stemmed_content_field,
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

        Ok(())
    }

    /// Remove a document from the index
    pub fn delete(&self, memory_id: &MemoryId) -> Result<()> {
        let writer = self.writer.write();
        let id_term = tantivy::Term::from_field_text(self.id_field, &memory_id.0.to_string());
        writer.delete_term(id_term);
        Ok(())
    }

    /// Commit pending changes to disk
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.write();
        writer.commit().context("Failed to commit BM25 index")?;
        Ok(())
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

        let top_docs = searcher
            .search(&parsed_query, &TopDocs::with_limit(limit))
            .context("BM25 search failed")?;

        let mut results = Vec::with_capacity(top_docs.len());

        for (score, doc_address) in top_docs {
            if let Ok(doc) = searcher.doc::<TantivyDocument>(doc_address) {
                if let Some(id_value) = doc.get_first(self.id_field) {
                    if let Some(id_str) = id_value.as_str() {
                        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                            results.push((MemoryId(uuid), score));
                        }
                    }
                }
            }
        }

        Ok(results)
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

        for (list_idx, ranked_list) in ranked_lists.iter().enumerate() {
            let weight = self.weights.get(list_idx).copied().unwrap_or(1.0);

            for (rank, (memory_id, score)) in ranked_list.iter().enumerate() {
                // RRF formula: weight * 1/(k + rank)
                // rank is 0-indexed, so rank+1 for 1-indexed
                let rrf_contribution = weight / (self.k + (rank + 1) as f32);

                *scores.entry(memory_id.clone()).or_insert(0.0) += rrf_contribution;

                // Track original scores for debugging
                let orig = original_scores
                    .entry(memory_id.clone())
                    .or_insert_with(|| vec![None; ranked_lists.len()]);
                if list_idx < orig.len() {
                    orig[list_idx] = Some(*score);
                }
            }
        }

        // Sort by RRF score descending
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1));

        results
    }
}

/// Cross-encoder reranker using ms-marco-MiniLM-L-6-v2
///
/// Uses a true cross-encoder model that jointly encodes (query, document)
/// pairs for relevance scoring. Significantly more accurate than bi-encoder
/// cosine similarity for reranking. Falls back to bi-encoder if cross-encoder
/// model is unavailable.
///
/// Blending: 70% cross-encoder + 30% bi-encoder (proven ratio from literature).
pub struct CrossEncoderReranker {
    embedder: Arc<dyn Embedder>,
    cross_encoder: Arc<crate::embeddings::cross_encoder::CrossEncoder>,
}

/// Weight for cross-encoder score in the blend (0.7 = 70%)
const CROSS_ENCODER_BLEND_WEIGHT: f32 = 0.70;
/// Weight for bi-encoder score in the blend (0.3 = 30%)
const BI_ENCODER_BLEND_WEIGHT: f32 = 0.30;

impl CrossEncoderReranker {
    /// Create reranker with shared embedder and cross-encoder model
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            cross_encoder: Arc::new(
                crate::embeddings::cross_encoder::CrossEncoder::new(),
            ),
        }
    }

    /// Create reranker with an existing cross-encoder instance
    pub fn with_cross_encoder(
        embedder: Arc<dyn Embedder>,
        cross_encoder: Arc<crate::embeddings::cross_encoder::CrossEncoder>,
    ) -> Self {
        Self {
            embedder,
            cross_encoder,
        }
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
                Some(emb) if emb.len() == query_dim => {
                    cosine_similarity(&query_embedding, emb)
                }
                _ => {
                    let doc_embedding = self.embedder.encode(content)?;
                    cosine_similarity(&query_embedding, &doc_embedding)
                }
            };
            bi_scores.push(bi);
        }

        // Try cross-encoder for blended scoring
        let use_cross_encoder = self.cross_encoder.is_available();
        let blended_scores = if use_cross_encoder {
            let doc_texts: Vec<&str> = candidates.iter().map(|(_, c, _, _)| c.as_str()).collect();
            match self.cross_encoder.score_pairs(query, &doc_texts) {
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
            bm25_index,
            config,
            reranker,
            refiner: None,
            learned_weights,
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
    /// Without an attached refiner, `RefinerMode::Rlm` and
    /// `RefinerMode::Stacked` degrade to their respective baselines:
    /// `Rlm` becomes RRF-only, `Stacked` becomes cross-encoder-only.
    pub fn with_refiner(mut self, refiner: Box<dyn Refiner>) -> Self {
        self.refiner = Some(refiner);
        self
    }

    /// Attach an LLM-driven refiner to be applied after RRF fusion (and
    /// optionally after cross-encoder reranking — see [`RefinerMode`]).
    ///
    /// Without an attached refiner, `RefinerMode::Rlm` and
    /// `RefinerMode::Stacked` degrade to their respective baselines:
    /// `Rlm` becomes RRF-only, `Stacked` becomes cross-encoder-only.
    pub fn with_refiner(mut self, refiner: Box<dyn Refiner>) -> Self {
        self.refiner = Some(refiner);
        self
    }

    /// Index a memory for BM25 search
    pub fn index_memory(
        &self,
        memory_id: &MemoryId,
        content: &str,
        tags: &[String],
        entities: &[String],
    ) -> Result<()> {
        self.bm25_index.upsert(memory_id, content, tags, entities)
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
    pub fn commit_and_reload(&self) -> Result<()> {
        self.bm25_index.commit()?;
        self.bm25_index.reload()
    }

    /// Get BM25 index reference for direct searches
    pub fn bm25_index(&self) -> &BM25Index {
        &self.bm25_index
    }

    /// Update learned weights from retrieval feedback
    ///
    /// Called after `reinforce_recall` processes a Helpful or Misleading outcome.
    /// Uses EMA to shift weights toward (helpful) or away from (misleading) the
    /// signal contributions that produced the retrieval result.
    ///
    /// Thread-safe: takes write lock on learned_weights only for the update.
    pub fn update_weights_from_feedback(&self, signal_scores: &SignalScores, helpful: bool) {
        use crate::constants::ADAPTIVE_WEIGHT_LEARNING_RATE;

        let target = if helpful {
            *signal_scores
        } else {
            signal_scores.inverted()
        };

        let mut weights = self.learned_weights.write();
        weights.apply_update(&target, ADAPTIVE_WEIGHT_LEARNING_RATE);

        tracing::debug!(
            "Adaptive weights updated ({}): bm25={:.3}, vector={:.3}, graph={:.3}, updates={}",
            if helpful { "helpful" } else { "misleading" },
            weights.bm25,
            weights.vector,
            weights.graph,
            weights.update_count,
        );
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
        use crate::constants::{
            ADAPTIVE_WEIGHT_CONFIDENCE_THRESHOLD, ADAPTIVE_WEIGHT_MAX_BLEND,
        };

        let learned = self.learned_weights.read();

        if learned.update_count == 0 {
            return (
                self.config.bm25_weight,
                self.config.vector_weight,
                self.config.graph_weight,
            );
        }

        // Confidence ramp: linearly increase blend factor with update count
        let confidence = (learned.update_count as f32
            / ADAPTIVE_WEIGHT_CONFIDENCE_THRESHOLD as f32)
            .min(1.0);
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
        // 1. BM25 search with IC-weighted term boosting AND phrase matching
        let bm25_results = self.bm25_index.search_with_term_and_phrase_weights(
            query,
            self.config.candidate_count,
            term_weights,
            phrase_boosts,
        )?;

        // Filter low BM25 scores
        let bm25_results: Vec<_> = bm25_results
            .into_iter()
            .filter(|(_, score)| *score >= self.config.min_bm25_score)
            .collect();

        // Calculate dynamic weights based on keyword discriminativeness
        // When YAKE identifies discriminative keywords, trust BM25 more
        // YAKE importance = 1/(1+score), so 0.9+ means very discriminative keywords
        //
        // Base weights come from effective_weights() which blends static defaults
        // with learned weights from retrieval feedback (P2.3 adaptive learning)
        let (eff_bm25, eff_vector, _eff_graph) = self.effective_weights();
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
        let rrf = RRFusion::new(self.config.rrf_k, vec![bm25_weight, vector_weight]);

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
        let effective_rerank_count = rerank_count_override.unwrap_or(self.config.rerank_count);
        let mode = self.config.refiner_mode;
        let use_cross_encoder =
            matches!(mode, RefinerMode::CrossEncoder | RefinerMode::Stacked);
        let use_refiner = matches!(mode, RefinerMode::Rlm | RefinerMode::Stacked);

        let mut final_results: Vec<HybridSearchResult> = if let Some(reranker) =
            self.reranker.as_ref().filter(|_| use_cross_encoder)
        {
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
                const RERANK_BLEND: f32 = 0.4;
                let mut results: Vec<HybridSearchResult> = Vec::new();

                for (memory_id, rrf_score) in fused {
                    let bm25_info = bm25_map.get(&memory_id);
                    let vector_info = vector_map.get(&memory_id);
                    let rerank_score = rerank_map.get(&memory_id).copied();

                    // Blend rerank with RRF to preserve score scale
                    let final_score = if let Some(norm_rerank) =
                        rerank_normalized.get(&memory_id).copied()
                    {
                        (1.0 - RERANK_BLEND) * rrf_score + RERANK_BLEND * norm_rerank * rrf_score
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
        // mode is Stacked; replaces cross-encoder when mode is Rlm).
        if use_refiner {
            if let Some(refiner) = self.refiner.as_ref() {
                let to_refine: Vec<(MemoryId, String, f32)> = final_results
                    .iter()
                    .take(effective_rerank_count)
                    .filter_map(|r| {
                        get_content(&r.memory_id)
                            .map(|c| (r.memory_id.clone(), c, r.score))
                    })
                    .collect();
                if !to_refine.is_empty() {
                    let refined = refiner.refine(query, to_refine)?;
                    let refined_map: HashMap<MemoryId, f32> =
                        refined.into_iter().collect();

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
        assert!((sum - 1.0).abs() < 0.001, "Zero input should produce uniform weights");
        assert!((scores.bm25_contribution - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_signal_scores_inversion() {
        let scores = SignalScores::from_components(0.8, 0.1, 0.1);
        let inverted = scores.inverted();
        // High BM25 contribution should become low after inversion
        assert!(inverted.bm25_contribution < inverted.vector_contribution);
        assert!(inverted.bm25_contribution < inverted.graph_contribution);
        let sum = inverted.bm25_contribution + inverted.vector_contribution + inverted.graph_contribution;
        assert!((sum - 1.0).abs() < 0.001, "Inverted scores should sum to 1.0");
    }

    #[test]
    fn test_learned_weights_ema_update() {
        let config = HybridSearchConfig::default();
        let mut weights = LearnedWeights::from_defaults(&config);

        assert_eq!(weights.update_count, 0);
        assert!((weights.bm25 - config.bm25_weight).abs() < 0.001);

        // Apply a strong BM25 signal
        let target = SignalScores::from_components(1.0, 0.0, 0.0);
        weights.apply_update(&target, 0.1);

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

        // Apply many updates pushing toward vector dominance
        let target = SignalScores::from_components(0.1, 0.8, 0.1);
        for _ in 0..100 {
            weights.apply_update(&target, 0.05);
        }

        // After 100 updates at lr=0.05, weights should be close to target
        assert!(weights.vector > 0.6, "Vector should dominate after many updates: {}", weights.vector);
        assert!(weights.bm25 < 0.2, "BM25 should be low: {}", weights.bm25);
        assert!(weights.graph < 0.2, "Graph should be low: {}", weights.graph);
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

        for target in &targets {
            weights.apply_update(target, 0.1);
            let sum = weights.bm25 + weights.vector + weights.graph;
            assert!(
                (sum - 1.0).abs() < 0.001,
                "Weights should sum to 1.0 after every update, got {}",
                sum
            );
        }
    }
}
