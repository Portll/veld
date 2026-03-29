//! Memory System for LLM Context Management
//!
//! A medium-complexity memory system that provides:
//! - Hierarchical memory storage (working → session → long-term)
//! - Smart compression based on age and importance
//! - Multi-modal retrieval (similarity, temporal, causal)
//! - Automatic memory consolidation

pub mod compression;
pub mod context;
pub mod context_blocks;
pub mod facts;
pub mod feedback;
pub mod files;
pub mod gap_topology;
pub mod graph_retrieval;
pub mod hybrid_search;
pub mod injection;
pub mod introspection;
pub mod learning_history;
pub mod lineage;
pub mod mapper;
pub mod pattern_detection;
pub mod persistence;
pub mod prospective;
pub mod query_parser;
mod recall;
pub mod replay;
pub mod retrieval;
pub mod segmentation;
pub mod sessions;
pub mod slow_store;
pub mod storage;
pub mod wavelet_sessions;
pub mod temporal_facts;
pub mod todo_formatter;
pub mod voronoi;
pub mod todos;
pub mod types;
pub mod visualization;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::debug;
use uuid::Uuid;

use crate::metrics::{EMBEDDING_CACHE_CONTENT, EMBEDDING_CACHE_CONTENT_SIZE};

use crate::constants::{
    DEFAULT_COMPRESSION_AGE_DAYS, DEFAULT_IMPORTANCE_THRESHOLD, DEFAULT_MAX_HEAP_PER_USER_MB,
    DEFAULT_SESSION_MEMORY_SIZE_MB, DEFAULT_WORKING_MEMORY_SIZE, EDGE_SEMANTIC_WEIGHT_FLOOR,
    POTENTIATION_ACCESS_THRESHOLD,
    POTENTIATION_MAINTENANCE_BOOST, TIER_PROMOTION_SESSION_AGE_SECS,
    TIER_PROMOTION_SESSION_IMPORTANCE, TIER_PROMOTION_WORKING_AGE_SECS,
    TIER_PROMOTION_WORKING_IMPORTANCE,
};

use crate::memory::storage::MemoryStorage;
pub use crate::memory::types::*;
// pub use crate::memory::vector_storage::{VectorIndexedMemoryStorage, StorageStats};  // Disabled
use crate::embeddings::Embedder;
use crate::memory::compression::CompressionPipeline;
pub use crate::memory::compression::{
    ConsolidationResult, FactType, SemanticConsolidator, SemanticFact,
};
pub use crate::memory::facts::{FactQueryResponse, FactStats, SemanticFactStore};
pub use crate::memory::feedback::{
    apply_context_pattern_signals, calculate_entity_flow, calculate_entity_overlap,
    detect_negative_keywords, extract_entities_simple, process_implicit_feedback,
    process_implicit_feedback_with_semantics, signal_from_entity_flow, ContextFingerprint,
    FeedbackMomentum, FeedbackStore, FeedbackStoreStats, PendingFeedback, PreviousContext,
    SignalRecord, SignalTrigger, SurfacedMemoryInfo, Trend,
};
pub use crate::memory::files::{FileMemoryStats, FileMemoryStore, IndexingResult};
pub use crate::memory::graph_retrieval::{
    calculate_density_weights, spreading_activation_retrieve, ActivatedMemory,
};
pub use crate::memory::hybrid_search::{
    BM25Index, CrossEncoderReranker, HybridSearchConfig, HybridSearchEngine, HybridSearchResult,
    LearnedWeights, RRFusion, SignalScores,
};
pub use crate::memory::introspection::{
    AssociationChange, ConsolidationEvent, ConsolidationEventBuffer, ConsolidationReport,
    ConsolidationStats, EdgeFormationReason, FactChange, InterferenceEvent, InterferenceType,
    MemoryChange, PruningReason, ReplayEvent, ReportPeriod, StrengtheningReason,
};
pub use crate::memory::learning_history::{
    LearningEventType, LearningHistoryStore, LearningStats, LearningVelocity, StoredLearningEvent,
};
pub use crate::memory::lineage::{
    CausalRelation, InferenceConfig, LineageBranch, LineageEdge, LineageGraph, LineageSource,
    LineageStats, LineageTrace, PostMortem, TraceDirection,
};
pub use crate::memory::prospective::ProspectiveStore;
pub use crate::memory::replay::{
    InterferenceCheckResult, InterferenceDetector, InterferenceRecord, ReplayCandidate,
    ReplayCycleResult, ReplayManager,
};
use crate::memory::retrieval::RetrievalEngine;
pub use crate::memory::retrieval::{
    AnticipatoryPrefetch, IndexHealth, MemoryGraphStats, PrefetchContext, PrefetchReason,
    PrefetchResult, ReinforcementStats, RetrievalFeedback, RetrievalOutcome, TrackedRetrieval,
};
pub use crate::memory::segmentation::{
    AtomicMemory, ConversationMemory, ConversationTurn, DeduplicationEngine, DeduplicationResult, InputSource, SegmentationEngine, SpeakerEntity,
};
pub use crate::memory::sessions::{
    Session, SessionEvent, SessionId, SessionStats, SessionStatus, SessionStore, SessionStoreStats,
    SessionSummary, TemporalContext, TimeOfDay,
};
pub use crate::memory::context_blocks::{ContextBlock, ContextBlockStore};
pub use crate::memory::temporal_facts::{
    ContradictionType, EventType, ResolvedTime, TemporalFact, TemporalFactStore,
};
pub use crate::memory::todos::{ProjectStats, TodoStore, UserTodoStats};
pub use crate::memory::visualization::{GraphStats, MemoryLogger};

/// Configuration for the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Base directory for memory storage
    pub storage_path: PathBuf,

    /// Maximum size of working memory (in entries)
    pub working_memory_size: usize,

    /// Maximum size of session memory (in MB)
    pub session_memory_size_mb: usize,

    /// Maximum heap memory per user (in MB) - prevents OOM from single user
    pub max_heap_per_user_mb: usize,

    /// Enable auto-compression of old memories
    pub auto_compress: bool,

    /// Compression threshold (days)
    pub compression_age_days: u32,

    /// Importance threshold for long-term storage
    pub importance_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./memory_store"),
            working_memory_size: DEFAULT_WORKING_MEMORY_SIZE,
            session_memory_size_mb: DEFAULT_SESSION_MEMORY_SIZE_MB,
            max_heap_per_user_mb: DEFAULT_MAX_HEAP_PER_USER_MB,
            auto_compress: true,
            compression_age_days: DEFAULT_COMPRESSION_AGE_DAYS,
            importance_threshold: DEFAULT_IMPORTANCE_THRESHOLD,
        }
    }
}

/// Main memory system
pub struct MemorySystem {
    config: MemoryConfig,

    /// Three-tier memory hierarchy
    working_memory: Arc<RwLock<WorkingMemory>>,
    session_memory: Arc<RwLock<SessionMemory>>,
    long_term_memory: Arc<MemoryStorage>,

    /// Compression pipeline
    compressor: CompressionPipeline,

    /// Retrieval engine
    retriever: RetrievalEngine,

    /// Embedder for semantic search
    embedder: Arc<crate::embeddings::minilm::MiniLMEmbedder>,

    /// Query embedding cache - SHA256(query_text) → embedding
    /// Uses SHA256 for stable hashing across restarts (unlike DefaultHasher)
    /// MASSIVE PERF WIN: 80ms → <1ms for cached queries
    /// LRU eviction: max 10,000 entries (~15MB for 384-dim embeddings)
    query_cache: moka::sync::Cache<[u8; 32], Vec<f32>>,

    /// Content embedding cache - SHA256(content) → embedding
    /// Uses SHA256 for stable hashing across restarts (unlike DefaultHasher)
    /// MASSIVE PERF WIN: 80ms → <1ms for repeated content
    /// LRU eviction: max 10,000 entries (~15MB for 384-dim embeddings)
    content_cache: moka::sync::Cache<[u8; 32], Vec<f32>>,

    /// Memory statistics
    stats: Arc<RwLock<MemoryStats>>,

    /// Visualization logger
    logger: Arc<RwLock<MemoryLogger>>,

    /// Consolidation event buffer for introspection
    /// Tracks what the memory system is learning (strengthening, decay, edges, facts)
    consolidation_events: Arc<RwLock<ConsolidationEventBuffer>>,

    /// Memory replay manager (SHO-105)
    /// Implements sleep-like consolidation through replay of high-value memories
    replay_manager: Arc<RwLock<replay::ReplayManager>>,

    /// Interference detector (SHO-106)
    /// Detects and handles memory interference (retroactive/proactive)
    interference_detector: Arc<RwLock<replay::InterferenceDetector>>,

    /// Pattern detector for intelligent replay triggers (PIPE-2)
    /// Replaces fixed 1-hour intervals with pattern-based consolidation
    pattern_detector: Arc<RwLock<pattern_detection::PatternDetector>>,

    /// Semantic fact store (SHO-f0e7)
    /// Stores distilled knowledge extracted from episodic memories
    /// Separate from episodic storage: facts persist, episodes flow
    fact_store: Arc<facts::SemanticFactStore>,

    /// Decision lineage graph (SHO-118)
    /// Tracks causal relationships between memories for "why" reasoning
    /// Enables: audit trails, project branching, automatic post-mortems
    lineage_graph: Arc<lineage::LineageGraph>,

    /// Hybrid search engine (BM25 + Vector + RRF + Reranking)
    /// Combines keyword matching with semantic similarity for better retrieval
    hybrid_search: Arc<hybrid_search::HybridSearchEngine>,

    /// Optional graph memory for entity relationships and spreading activation
    /// When set, entities are extracted and added to the knowledge graph on remember()
    /// This enables spreading activation retrieval and Hebbian co-activation learning
    graph_memory: Option<Arc<parking_lot::RwLock<crate::graph_memory::GraphMemory>>>,

    /// Optional feedback store for momentum-based scoring (PIPE-9)
    /// When set, retrieval applies feedback momentum to boost proven-helpful memories
    /// and suppress frequently-ignored memories (up to 20% penalty for negative momentum)
    feedback_store: Option<Arc<parking_lot::RwLock<FeedbackStore>>>,

    /// Pinky dimension scores: topological health of the knowledge graph.
    /// Pushed by Pinky via `/api/pinky/dimensions`, read during Layer 5 scoring.
    /// When available, memories from high-quality graph regions rank higher.
    pinky_scores: Arc<parking_lot::RwLock<Option<types::PinkyDimensionScores>>>,

    /// Persistent learning history for significant events
    /// Enables recency-weighted retrieval and learning velocity tracking
    learning_history: Arc<learning_history::LearningHistoryStore>,

    /// Temporal fact store for multi-hop temporal reasoning
    /// Extracts and indexes facts like "Melanie is planning camping next month"
    /// Resolves relative dates ("next month" → June 2023) for accurate retrieval
    temporal_fact_store: Arc<temporal_facts::TemporalFactStore>,

    /// Flag: new memories stored since last fact extraction cycle.
    /// When false, fact extraction is skipped entirely (no RocksDB scan, no clones).
    /// Set to true in remember(), cleared by maintenance after extraction runs.
    fact_extraction_needed: std::sync::atomic::AtomicBool,

    /// Watermark: only memories with created_at > this timestamp (unix millis) are
    /// processed for fact extraction. Persisted to RocksDB so server restarts don't
    /// re-process the entire memory store. Initialized from the latest fact's
    /// created_at or 0 if no facts exist.
    fact_extraction_watermark: std::sync::atomic::AtomicI64,

    /// Cached wavelet-detected session map.
    /// Invalidated when fact_extraction_needed is set (any new remember call).
    session_map_cache: parking_lot::Mutex<Option<wavelet_sessions::SessionMap>>,

    /// Active reconsolidation shadows (FIX-R1): memories currently in the labile window.
    /// Key: MemoryId, Value: ReconsolidationShadow
    /// When a memory is retrieved, it becomes labile (activation=1.0) and a shadow
    /// is created. The shadow accumulates retrieval context until the window expires,
    /// at which point maintenance atomically applies updates.
    /// Reference: Nader et al. (2000) — reconsolidation theory.
    reconsolidation_shadows:
        Arc<parking_lot::RwLock<std::collections::HashMap<MemoryId, types::ReconsolidationShadow>>>,

    /// Signal attribution from the most recent semantic retrieval.
    /// Records which scoring signals (BM25, vector, graph, cross-encoder, recency,
    /// temporal match, entity overlap) contributed to each memory's ranking.
    /// Populated during Layer 4 (RRF fusion) and Layer 5 (unified scoring).
    /// Used for adaptive weight learning and retrieval diagnostics.
    last_signal_attributions:
        Arc<parking_lot::RwLock<std::collections::HashMap<MemoryId, types::SignalAttribution>>>,
}

/// Extract causal entity pairs from memory content.
///
/// Detects causal language patterns (because, therefore, led to, etc.) and
/// identifies which entities are the cause and which are the effect. Uses
/// anti-causal filtering to avoid false positives from temporal coincidence.
///
/// Returns Vec<(cause_entity, effect_entity, RelationType)>.
fn extract_causal_pairs(
    content: &str,
    entities: &[String],
) -> Vec<(String, String, crate::graph_memory::RelationType)> {
    if entities.len() < 2 {
        return Vec::new();
    }

    let lower = content.to_lowercase();

    // Causal forward patterns: cause → effect
    // The entity BEFORE the marker is the cause, AFTER is the effect
    let forward_markers = [
        " caused ",
        " led to ",
        " resulted in ",
        " triggered ",
        " enabled ",
        " produced ",
        " created ",
        " which meant ",
        " so that ",
        " therefore ",
    ];

    // Causal backward patterns: effect ← cause
    // The entity BEFORE the marker is the effect, AFTER is the cause
    let backward_markers = [
        " because of ",
        " due to ",
        " as a result of ",
        " thanks to ",
        " caused by ",
        " driven by ",
        " enabled by ",
    ];

    // Anti-causal markers: these look causal but aren't
    // (temporal coincidence, not causation)
    let anti_causal = [
        " and then ",
        " followed by ",
        " after which ",
        " next ",
        " meanwhile ",
    ];

    // If any anti-causal marker is present, reduce confidence
    let has_anti_causal = anti_causal.iter().any(|m| lower.contains(m));
    if has_anti_causal {
        return Vec::new();
    }

    let mut pairs = Vec::new();

    // For each causal marker, find which entities appear before/after it
    for marker in &forward_markers {
        if let Some(pos) = lower.find(marker) {
            let before = &lower[..pos];
            let after = &lower[pos + marker.len()..];

            // Find the last entity mentioned before the marker (cause)
            // and the first entity mentioned after (effect)
            let cause = entities
                .iter()
                .rfind(|e| before.contains(&e.to_lowercase()));
            let effect = entities
                .iter()
                .find(|e| after.contains(&e.to_lowercase()));

            if let (Some(c), Some(e)) = (cause, effect) {
                if c != e {
                    pairs.push((
                        c.clone(),
                        e.clone(),
                        crate::graph_memory::RelationType::Causes,
                    ));
                }
            }
        }
    }

    for marker in &backward_markers {
        if let Some(pos) = lower.find(marker) {
            let before = &lower[..pos];
            let after = &lower[pos + marker.len()..];

            // Before marker = effect, after marker = cause (reversed)
            let effect = entities
                .iter()
                .rfind(|e| before.contains(&e.to_lowercase()));
            let cause = entities
                .iter()
                .find(|e| after.contains(&e.to_lowercase()));

            if let (Some(c), Some(e)) = (cause, effect) {
                if c != e {
                    pairs.push((
                        c.clone(),
                        e.clone(),
                        crate::graph_memory::RelationType::Causes,
                    ));
                }
            }
        }
    }

    // Deduplicate: keep first occurrence of each (cause, effect) pair
    pairs.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    pairs
}

/// Resolve an entity name to a graph label and salience using pre-extracted NER data.
///
/// Returns (EntityLabel, salience) based on NER type mapping, defaulting to (Concept, 0.5).
fn resolve_entity_label(
    entity_name: &str,
    ner_lookup: &std::collections::HashMap<String, (String, f32)>,
) -> (crate::graph_memory::EntityLabel, f32) {
    if let Some((ner_type, confidence)) = ner_lookup.get(&entity_name.to_lowercase()) {
        let label = match ner_type.as_str() {
            "PER" => crate::graph_memory::EntityLabel::Person,
            "ORG" => crate::graph_memory::EntityLabel::Organization,
            "LOC" => crate::graph_memory::EntityLabel::Location,
            _ => crate::graph_memory::EntityLabel::Concept,
        };
        (label, *confidence)
    } else {
        (crate::graph_memory::EntityLabel::Concept, 0.5)
    }
}

/// Build a lookup table from NER entity records for label resolution.
fn build_ner_lookup(
    ner_entities: &[NerEntityRecord],
) -> std::collections::HashMap<String, (String, f32)> {
    ner_entities
        .iter()
        .map(|r| (r.text.to_lowercase(), (r.entity_type.clone(), r.confidence)))
        .collect()
}

impl MemorySystem {
    /// Create a new memory system.
    ///
    /// If `shared_cache` is provided, all per-user RocksDB instances share the
    /// same LRU block cache (multi-tenant server mode). Pass `None` for
    /// standalone / test use — each DB gets a small local cache.
    pub fn new(config: MemoryConfig, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        let storage_path = config.storage_path.clone();
        let storage = Arc::new(
            MemoryStorage::new(&storage_path, shared_cache)
                .with_context(|| format!("Failed to open storage at {:?}", storage_path))?,
        );

        // CRITICAL: Initialize embedder ONCE and share between MemorySystem and RetrievalEngine
        // This prevents loading the ONNX model multiple times (50-200ms overhead per load)
        let embedding_config = crate::embeddings::minilm::EmbeddingConfig::default();
        let embedder = Arc::new(
            crate::embeddings::minilm::MiniLMEmbedder::new(embedding_config)
                .context("Failed to initialize MiniLM embedder (ONNX model)")?,
        );

        // Create consolidation event buffer first so we can share it with retriever
        let consolidation_events = Arc::new(RwLock::new(ConsolidationEventBuffer::new()));

        // Pass shared embedder and event buffer to retrieval engine (no duplicate model load)
        // Event buffer allows retriever to record Hebbian edge events for introspection
        let retriever = RetrievalEngine::with_event_buffer(
            storage.clone(),
            embedder.clone(),
            Some(consolidation_events.clone()),
        )
        .context("Failed to initialize retrieval engine")?;

        // STARTUP RECOVERY: Check for orphaned memories and auto-repair
        // This fixes memories that were stored but not indexed (crash, embedding failure, etc.)
        let storage_count = storage.get_stats().map(|s| s.total_count).unwrap_or(0);
        let indexed_count = retriever.len();
        let orphaned_count = storage_count.saturating_sub(indexed_count);

        if orphaned_count > 0 {
            tracing::warn!(
                storage_count = storage_count,
                indexed_count = indexed_count,
                orphaned_count = orphaned_count,
                "Detected orphaned memories at startup - initiating auto-repair"
            );

            // Get all memories from storage
            if let Ok(all_memories) = storage.get_all() {
                let indexed_ids = retriever.get_indexed_memory_ids();
                let mut repaired = 0;
                let mut failed = 0;

                for memory in all_memories {
                    if indexed_ids.contains(&memory.id) {
                        continue; // Already indexed
                    }

                    // Skip absurdly large memories (>1MB) - likely binary data or log dumps
                    // MiniLM only uses first ~512 tokens anyway, so this protects ONNX from hanging
                    const MAX_REPAIR_CONTENT_LEN: usize = 1_000_000;
                    if memory.experience.content.len() > MAX_REPAIR_CONTENT_LEN {
                        tracing::warn!(
                            memory_id = %memory.id.0,
                            content_len = memory.experience.content.len(),
                            "Skipping oversized memory during auto-repair (>1MB)"
                        );
                        failed += 1;
                        continue;
                    }

                    // Orphaned memory - try to index it
                    tracing::info!(memory_id = %memory.id.0, content_len = memory.experience.content.len(), "Attempting to repair orphaned memory...");
                    match retriever.index_memory(&memory) {
                        Ok(_) => {
                            repaired += 1;
                            if repaired <= 10 || repaired % 100 == 0 {
                                tracing::info!(
                                    memory_id = %memory.id.0,
                                    progress = format!("{}/{}", repaired, orphaned_count),
                                    "Repaired orphaned memory"
                                );
                            }
                        }
                        Err(e) => {
                            failed += 1;
                            tracing::error!(
                                memory_id = %memory.id.0,
                                error = %e,
                                "Failed to repair orphaned memory"
                            );
                        }
                    }
                }

                // Persist the repaired index
                if repaired > 0 {
                    if let Err(e) = retriever.save() {
                        tracing::error!("Failed to persist repaired index: {}", e);
                    } else {
                        tracing::info!(
                            repaired = repaired,
                            failed = failed,
                            final_indexed = retriever.len(),
                            "Startup repair complete - index persisted"
                        );
                    }
                }
            }
        } else if storage_count > 0 {
            tracing::info!(
                storage_count = storage_count,
                indexed_count = indexed_count,
                "All memories indexed - no repair needed"
            );
        }

        // Disable visualization logging for production performance
        let logger = Arc::new(RwLock::new(MemoryLogger::new(false)));

        // Load stats from storage to recover state after restart
        let initial_stats = {
            let storage_stats = storage.get_stats().unwrap_or_default();
            let vector_count = retriever.len();
            MemoryStats {
                total_memories: storage_stats.total_count,
                working_memory_count: 0, // Working memory is in-memory only, starts empty
                session_memory_count: 0, // Session memory is in-memory only, starts empty
                long_term_memory_count: storage_stats.total_count,
                vector_index_count: vector_count,
                compressed_count: storage_stats.compressed_count,
                promotions_to_session: 0,  // Runtime counter, not persisted
                promotions_to_longterm: 0, // Runtime counter, not persisted
                total_retrievals: storage_stats.total_retrievals,
                average_importance: storage_stats.average_importance,
                graph_nodes: 0, // Loaded separately from GraphMemory
                graph_edges: 0, // Loaded separately from GraphMemory
            }
        };

        // SHO-f0e7: Create semantic fact store using the same DB as long-term memory
        // Facts use "facts:" prefix to avoid key collisions with episodic memories
        let fact_store = Arc::new(facts::SemanticFactStore::new(storage.db()));

        // SHO-118: Create lineage graph for causal memory tracking
        // Lineage uses "lineage:" prefix for edges and branches
        let lineage_graph = Arc::new(lineage::LineageGraph::new(storage.db()));

        // Initialize hybrid search engine (BM25 + Vector + RRF + Reranking)
        let bm25_path = storage_path.join("bm25_index");
        let hybrid_search_config = hybrid_search::HybridSearchConfig::default();
        let hybrid_search_engine = hybrid_search::HybridSearchEngine::new(
            &bm25_path,
            embedder.clone(),
            hybrid_search_config,
        )
        .context("Failed to initialize hybrid search engine")?;

        // Backfill BM25 index if empty but memories exist
        if hybrid_search_engine.needs_backfill() {
            let existing_memories = storage.get_all()?;
            let memory_count = existing_memories.len();

            if memory_count > 0 {
                tracing::info!(
                    "BM25 index empty, backfilling {} existing memories...",
                    memory_count
                );

                let memories_iter = existing_memories.into_iter().map(|mem| {
                    (
                        mem.id,
                        mem.experience.content,
                        mem.experience.tags,
                        mem.experience.entities,
                    )
                });

                match hybrid_search_engine.backfill(memories_iter) {
                    Ok(indexed) => {
                        tracing::info!("BM25 backfill complete: {} memories indexed", indexed);
                    }
                    Err(e) => {
                        tracing::warn!("BM25 backfill failed (non-fatal): {}", e);
                    }
                }
            }
        }

        // Initialize learning history store for persistent significant events
        // Uses the same DB as long-term memory with "learning:" prefix
        let learning_history = Arc::new(learning_history::LearningHistoryStore::new(storage.db()));

        // Initialize temporal fact store for multi-hop temporal reasoning
        // Uses the same DB with "temporal_facts:", "temporal_by_entity:", "temporal_by_event:" prefixes
        let temporal_fact_store = Arc::new(temporal_facts::TemporalFactStore::new(storage.db()));

        // SHO-106: Load persisted interference history from RocksDB
        let interference_detector = {
            let mut detector = replay::InterferenceDetector::new();
            match storage.load_all_interference_records() {
                Ok((history, total_events)) => {
                    if !history.is_empty() {
                        detector.load_history(history, total_events);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to load interference history, starting fresh"
                    );
                }
            }
            Arc::new(RwLock::new(detector))
        };

        Ok(Self {
            config: config.clone(),
            working_memory: Arc::new(RwLock::new(WorkingMemory::new(config.working_memory_size))),
            session_memory: Arc::new(RwLock::new(SessionMemory::new(
                config.session_memory_size_mb,
            ))),
            long_term_memory: storage,
            compressor: CompressionPipeline::new(),
            retriever,
            embedder,
            // LRU embedding caches: max 2,000 entries each (~3MB for 384-dim embeddings)
            query_cache: moka::sync::Cache::builder().max_capacity(2_000).build(),
            content_cache: moka::sync::Cache::builder().max_capacity(2_000).build(),
            stats: Arc::new(RwLock::new(initial_stats)),
            logger,
            consolidation_events, // Use the shared buffer created earlier
            // SHO-105: Memory replay manager
            replay_manager: Arc::new(RwLock::new(replay::ReplayManager::new())),
            // SHO-106: Interference detector (loaded from RocksDB)
            interference_detector,
            // PIPE-2: Pattern detector for intelligent replay triggers
            pattern_detector: Arc::new(RwLock::new(pattern_detection::PatternDetector::new())),
            // SHO-f0e7: Semantic fact store
            fact_store,
            // SHO-118: Decision lineage graph
            lineage_graph,
            // Hybrid search engine (always enabled)
            hybrid_search: Arc::new(hybrid_search_engine),
            // Graph memory is optional - wire up with set_graph_memory() for entity relationships
            graph_memory: None,
            // Feedback store is optional - wire up with set_feedback_store() for momentum scoring (PIPE-9)
            feedback_store: None,
            // Pinky dimension scores: initialized empty, populated via API push
            pinky_scores: Arc::new(parking_lot::RwLock::new(None)),
            // Persistent learning history for retrieval boosting
            learning_history,
            // Temporal fact store for multi-hop temporal reasoning
            temporal_fact_store,
            // Dirty flag for fact extraction: run on first cycle, then only when new memories stored
            fact_extraction_needed: std::sync::atomic::AtomicBool::new(true),
            // Watermark for incremental fact extraction — initialized to 0 (sentinel).
            // On first maintenance call, loaded from RocksDB or derived from latest fact timestamp.
            fact_extraction_watermark: std::sync::atomic::AtomicI64::new(0),
            // Wavelet session detection cache — recomputed when new memories arrive
            session_map_cache: parking_lot::Mutex::new(None),
            // FIX-R1: Reconsolidation shadow map — tracks labile memories
            reconsolidation_shadows: Arc::new(parking_lot::RwLock::new(
                std::collections::HashMap::new(),
            )),
            // Signal attribution from most recent semantic retrieval
            last_signal_attributions: Arc::new(parking_lot::RwLock::new(
                std::collections::HashMap::new(),
            )),
        })
    }

    /// Wire up GraphMemory for entity relationships and spreading activation
    ///
    /// When GraphMemory is set, the remember() method will:
    /// 1. Extract entities from memory content
    /// 2. Add them to the knowledge graph
    /// 3. Create edges between co-occurring entities
    ///
    /// This enables spreading activation retrieval and Hebbian learning
    pub fn set_graph_memory(
        &mut self,
        graph: Arc<parking_lot::RwLock<crate::graph_memory::GraphMemory>>,
    ) {
        self.graph_memory = Some(graph);
    }

    /// Get reference to the optional graph memory
    pub fn graph_memory(
        &self,
    ) -> Option<&Arc<parking_lot::RwLock<crate::graph_memory::GraphMemory>>> {
        self.graph_memory.as_ref()
    }

    /// Set the feedback store for momentum-based scoring (PIPE-9)
    ///
    /// When set, retrieval automatically applies feedback momentum:
    /// - Positive momentum (frequently helpful) → boost score
    /// - Negative momentum (frequently ignored) → suppress score (up to 20%)
    ///
    /// This provides consistent feedback integration across all retrieval paths.
    pub fn set_feedback_store(&mut self, feedback: Arc<parking_lot::RwLock<FeedbackStore>>) {
        self.feedback_store = Some(feedback);
    }

    /// Get reference to the optional feedback store
    pub fn feedback_store(&self) -> Option<&Arc<parking_lot::RwLock<FeedbackStore>>> {
        self.feedback_store.as_ref()
    }

    /// Update Pinky dimension scores (called via API push from Pinky).
    pub fn set_pinky_scores(&self, scores: types::PinkyDimensionScores) {
        *self.pinky_scores.write() = Some(scores);
    }

    /// Get current Pinky dimension scores (read during Layer 5 scoring).
    /// Returns None if Pinky hasn't pushed scores or they're stale.
    pub fn pinky_aggregate_score(&self) -> Option<f32> {
        let guard = self.pinky_scores.read();
        guard.as_ref().and_then(|s| {
            if s.is_stale() {
                None
            } else {
                let agg = s.aggregate();
                if (agg - 1.0).abs() < f32::EPSILON {
                    None // neutral = no Pinky data
                } else {
                    Some(agg)
                }
            }
        })
    }

    /// Get signal attributions from the most recent semantic retrieval.
    ///
    /// Returns a snapshot of the attribution map. Each entry maps a MemoryId to
    /// the scoring signals that contributed to its ranking. Useful for retrieval
    /// diagnostics and adaptive weight learning.
    pub fn last_signal_attributions(
        &self,
    ) -> std::collections::HashMap<MemoryId, types::SignalAttribution> {
        self.last_signal_attributions.read().clone()
    }

    /// Get signal attribution for a specific memory from the most recent retrieval.
    pub fn signal_attribution_for(&self, id: &MemoryId) -> Option<types::SignalAttribution> {
        self.last_signal_attributions.read().get(id).cloned()
    }

    /// Store a new memory with an explicit ID.
    ///
    /// Used by MIF import to preserve original UUIDs. Stores the memory with
    /// embedding generation and vector indexing, but skips graph entity extraction
    /// (imported memories already have their entity relationships established).
    pub fn remember_with_id(
        &self,
        memory_id: MemoryId,
        mut experience: Experience,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<MemoryId> {
        let importance = self.calculate_importance(&experience);

        // Generate embedding if not provided
        if experience.embeddings.is_none() {
            let content_hash = Self::sha256_hash(&experience.content);
            if let Some(cached) = self.content_cache.get(&content_hash) {
                experience.embeddings = Some(cached.clone());
            } else if let Ok(embedding) = self.embedder.encode(&experience.content) {
                self.content_cache.insert(content_hash, embedding.clone());
                experience.embeddings = Some(embedding);
            }
        }

        let memory = Arc::new(Memory::new(
            memory_id.clone(),
            experience,
            importance,
            None,
            None,
            None,
            created_at,
        ));

        self.long_term_memory.store(&memory)?;
        self.logger.write().log_created(&memory, "import");

        self.working_memory
            .write()
            .add_shared(Arc::clone(&memory))?;

        if let Err(e) = self.retriever.index_memory(&memory) {
            tracing::warn!("Failed to index imported memory {}: {}", memory.id.0, e);
        }

        Ok(memory_id)
    }

    /// Store a new memory (full synchronous path: embedding + vector indexing + interference).
    /// Thread-safe: uses interior mutability for all internal state.
    /// If `created_at` is None, uses current time (Utc::now()).
    pub fn remember(
        &self,
        experience: Experience,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<MemoryId> {
        self.remember_impl(experience, created_at, false)
    }

    /// Store a new memory WITHOUT embedding generation or vector indexing (fast path).
    ///
    /// Returns in ~10ms instead of 150-250ms. The memory is immediately searchable
    /// via BM25 keyword search. Call `embed_and_index()` afterward (e.g. from a
    /// background task) to generate the embedding, update the vector index, and run
    /// interference detection.
    pub fn remember_deferred(
        &self,
        experience: Experience,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<MemoryId> {
        self.remember_impl(experience, created_at, true)
    }

    /// Internal implementation shared by `remember()` and `remember_deferred()`.
    ///
    /// When `defer_embedding` is true, skips:
    /// - ONNX embedding generation (80-150ms)
    /// - Vector index insertion (depends on embedding + additional ONNX for chunks)
    /// - Interference detection (depends on embedding)
    fn remember_impl(
        &self,
        mut experience: Experience,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
        defer_embedding: bool,
    ) -> Result<MemoryId> {
        // IDEMPOTENCY (issue #109): Check content hash index before creating a new memory.
        // If identical content already exists, return the existing MemoryId instead of
        // creating a duplicate. Catches all duplication paths: timeout retries, auto_ingest,
        // and manual re-remembers. O(1) RocksDB index lookup.
        if let Some(existing_id) = self
            .long_term_memory
            .get_by_content_hash(&experience.content)
        {
            tracing::debug!(
                existing_id = %existing_id.0,
                "Content dedup: returning existing memory (identical content already stored)"
            );
            return Ok(existing_id);
        }

        let memory_id = MemoryId(Uuid::new_v4());

        // Calculate importance
        let importance = self.calculate_importance(&experience);

        // PERFORMANCE: Content embedding cache (80ms → <1μs for repeated content)
        // If experience doesn't have embeddings, check cache or generate.
        // Skipped when defer_embedding=true — handled by embed_and_index() later.
        if !defer_embedding && experience.embeddings.is_none() {
            // SHA256 hash for stable cache keys (survives restarts, unlike DefaultHasher)
            let content_hash = Self::sha256_hash(&experience.content);

            // Check cache first
            if let Some(cached_embedding) = self.content_cache.get(&content_hash) {
                experience.embeddings = Some(cached_embedding.clone());
                EMBEDDING_CACHE_CONTENT.with_label_values(&["hit"]).inc();
                tracing::debug!("Content embedding cache HIT");
            } else {
                // Cache miss - generate embedding
                EMBEDDING_CACHE_CONTENT.with_label_values(&["miss"]).inc();
                match self.embedder.encode(&experience.content) {
                    Ok(embedding) => {
                        // Store in cache for future use
                        self.content_cache.insert(content_hash, embedding.clone());
                        EMBEDDING_CACHE_CONTENT_SIZE.set(self.content_cache.entry_count() as i64);
                        experience.embeddings = Some(embedding);
                        tracing::debug!("Content embedding cache MISS - generated and cached");
                    }
                    Err(e) => {
                        tracing::warn!("Failed to generate embedding: {}", e);
                        // Continue without embedding - will be generated on-demand if needed
                    }
                }
            }
        }

        // TEMPORAL EXTRACTION: Extract dates from content for temporal filtering
        // Based on TEMPR approach (Hindsight paper achieving 89.6% on LoCoMo)
        if experience.temporal_refs.is_empty() {
            let temporal = crate::memory::query_parser::extract_temporal_refs(&experience.content);
            for temp_ref in temporal.refs {
                experience.temporal_refs.push(temp_ref.date.to_string());
            }
        }

        // Create memory entry (zero-copy with Arc)
        // CRITICAL: Move experience instead of clone to avoid 2-10KB allocation
        let memory = Arc::new(Memory::new(
            memory_id.clone(),
            experience, // Move ownership (zero-cost)
            importance,
            None,       // agent_id
            None,       // run_id
            None,       // actor_id
            created_at, // Use provided timestamp or Utc::now() if None
        ));

        // FIX-R2: Compute elaboration score at encoding time.
        // Measures how contextualized this memory is (S-rep=0.0 vs C-rep=1.0).
        // Reference: Ehlers & Clark (2000) — poor elaboration → pathological intrusions.
        memory.set_elaboration_score(Self::compute_elaboration_score(&memory));

        // CRITICAL: Persist to RocksDB storage FIRST (before indexing/in-memory tiers)
        // This ensures retrieval can always fetch the memory from persistent storage
        self.long_term_memory.store(&memory)?;

        // Log creation
        self.logger.write().log_created(&memory, "working");

        // Add to working memory (cheap Arc clone, not full Memory clone)
        self.working_memory
            .write()
            .add_shared(Arc::clone(&memory))?;

        // Index memory for semantic search (vector DB).
        // When defer_embedding=true, this is skipped — embed_and_index() handles it later.
        // BM25 keyword search (below) provides immediate searchability regardless.
        let indexed = if !defer_embedding {
            if let Err(e) = self.retriever.index_memory(&memory) {
                tracing::warn!("Failed to index memory {} in vector DB: {}", memory.id.0, e);
                false
            } else {
                true
            }
        } else {
            false
        };

        // NOTE: Graph processing (entities + co-occurrence edges) is handled by
        // process_experience_into_graph() at the handler layer (remember.rs, recall.rs).
        // That path creates richer EpisodicNodes with temporal context and does proper
        // entity embedding for concept-level dedup. Doing it here too would cause
        // double entity inserts and inflated mention_counts.

        // Index in BM25 for hybrid search (keyword + semantic)
        if let Err(e) = self.hybrid_search.index_memory(
            &memory.id,
            &memory.experience.content,
            &memory.experience.tags,
            &memory.experience.entities,
        ) {
            tracing::warn!("Failed to index memory {} in BM25: {}", memory.id.0, e);
        }

        // PIPE-2: Register memory for pattern-triggered replay
        // Tracks entity co-occurrence, salience spikes, and temporal clusters
        {
            let arousal = memory
                .experience
                .context
                .as_ref()
                .map(|c| c.emotional.arousal)
                .unwrap_or(0.3);

            let pattern_memory = pattern_detection::PatternMemory {
                id: memory.id.0.to_string(),
                content_preview: memory.experience.content.chars().take(100).collect(),
                entities: memory.experience.entities.clone(),
                importance,
                arousal,
                created_at: memory.created_at,
                embedding_hash: memory.experience.embeddings.as_ref().map(|e| {
                    e.iter()
                        .fold(0u64, |acc, &x| acc.wrapping_add(x.to_bits() as u64))
                }),
                session_id: memory
                    .experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.episode_id.clone()),
                memory_type: format!("{:?}", memory.experience.experience_type),
            };

            let mut detector = self.pattern_detector.write();
            detector.register_memory(pattern_memory.clone());

            // Check for immediate salience spike (high-importance memory)
            if let Some(trigger) = detector.check_salience_spike(&pattern_memory) {
                tracing::debug!(
                    "Salience spike detected for memory {}: {}",
                    memory.id.0,
                    trigger.description()
                );
                // Record event for introspection
                self.record_consolidation_event(
                    introspection::ConsolidationEvent::PatternDetected {
                        trigger_type: trigger.trigger_type_name().to_string(),
                        description: trigger.description(),
                        memory_ids: trigger.memory_ids(),
                        timestamp: chrono::Utc::now(),
                    },
                );
            }
        }

        // TEMPORAL FACT EXTRACTION: Extract and index temporal facts for multi-hop reasoning
        // Key insight: Multi-hop temporal queries like "When is X planning Y?" require:
        // 1. Finding the FIRST/PLANNING mention, not any mention
        // 2. Resolving relative dates ("next month", "last Saturday") to absolute dates
        // This enables accurate answers to temporal questions in LoCoMo benchmark
        if !memory.experience.entities.is_empty() {
            let facts = temporal_facts::extract_temporal_facts(
                &memory.experience.content,
                &memory.id,
                memory.created_at,
                &memory.experience.entities,
            );
            if !facts.is_empty() {
                // Note: We don't have user_id in remember(), will need to pass it
                // For now, extract facts but don't store - storage happens at handler level
                // or we can use a placeholder user_id
                tracing::debug!(
                    "Extracted {} temporal facts from memory {}",
                    facts.len(),
                    memory.id.0
                );
            }
        }

        // SHO-106: Check for interference with existing memories
        // Find similar memories and apply retroactive/proactive interference.
        // Skipped when defer_embedding=true — embed_and_index() runs this later.
        if !defer_embedding {
        if let Some(embedding) = &memory.experience.embeddings {
            // Search for similar memories (excluding the new one)
            if let Ok(similar_ids) =
                self.retriever
                    .search_by_embedding(embedding, 5, Some(&memory.id))
            {
                if !similar_ids.is_empty() {
                    // Collect similar memory data for interference check
                    let similar_memories: Vec<_> = similar_ids
                        .iter()
                        .filter_map(|(id, similarity)| {
                            self.retriever.get_from_storage(id).ok().map(|m| {
                                (
                                    id.0.to_string(),
                                    *similarity,
                                    m.importance(),
                                    m.created_at,
                                    m.experience.content.chars().take(50).collect::<String>(),
                                )
                            })
                        })
                        .collect();

                    if !similar_memories.is_empty() {
                        let interference_result =
                            self.interference_detector.write().check_interference(
                                &memory.id.0.to_string(),
                                importance,
                                memory.created_at,
                                &similar_memories,
                            );

                        // Apply retroactive interference (weaken old memories)
                        for (old_id, _similarity, decay_amount) in
                            &interference_result.retroactive_targets
                        {
                            if let Ok(old_uuid) = uuid::Uuid::parse_str(old_id) {
                                if let Ok(old_memory) =
                                    self.long_term_memory.get(&MemoryId(old_uuid))
                                {
                                    old_memory.decay_importance(*decay_amount);
                                    if let Err(e) = self.long_term_memory.update(&old_memory) {
                                        tracing::debug!("Failed to persist retroactive decay: {e}");
                                    }
                                }
                            } else {
                                tracing::warn!(
                                    "Skipping retroactive decay: malformed UUID '{old_id}'"
                                );
                            }
                        }

                        // Apply proactive interference (reduce new memory importance)
                        if interference_result.proactive_decay > 0.0 {
                            memory.decay_importance(interference_result.proactive_decay);
                            if let Err(e) = self.long_term_memory.update(&memory) {
                                tracing::debug!("Failed to persist proactive decay: {e}");
                            }
                        }

                        // Record interference events
                        for event in &interference_result.events {
                            self.record_consolidation_event(event.clone());
                        }

                        // Handle duplicates: suppress the near-duplicate to near-zero importance
                        // so it decays naturally. We can't delete it here because callers
                        // expect the returned MemoryId to be retrievable.
                        if interference_result.is_duplicate {
                            tracing::info!(
                                memory_id = %memory.id.0,
                                "Near-duplicate detected (≥0.95 cosine), suppressing importance"
                            );
                            // Heavy decay: drop to ~1% importance so natural decay removes it
                            memory.decay_importance(0.99);
                            if let Err(e) = self.long_term_memory.update(&memory) {
                                tracing::debug!("Failed to suppress duplicate importance: {e}");
                            }
                        }

                        // Persist affected interference records to RocksDB
                        {
                            let detector = self.interference_detector.read();
                            let affected_ids = detector.get_affected_ids_from_check(
                                &memory.id.0.to_string(),
                                &interference_result,
                            );
                            for (id, records) in detector.get_records_for_ids(&affected_ids) {
                                if let Err(e) =
                                    self.long_term_memory.save_interference_records(id, records)
                                {
                                    tracing::debug!("Failed to persist interference records: {e}");
                                }
                            }
                            let (total_events, _) = detector.stats();
                            if let Err(e) = self
                                .long_term_memory
                                .save_interference_event_count(total_events)
                            {
                                tracing::debug!("Failed to persist interference event count: {e}");
                            }
                        }
                    }
                }
            }
        }
        } // end if !defer_embedding (interference)

        // If important enough, prepare for session storage
        let added_to_session = if importance > self.config.importance_threshold {
            self.session_memory
                .write()
                .add_shared(Arc::clone(&memory))?;
            self.logger.write().log_created(&memory, "session");
            true
        } else {
            false
        };

        // Update stats - track all tier counts accurately
        {
            let mut stats = self.stats.write();
            stats.total_memories += 1;
            stats.long_term_memory_count += 1; // Always stored to long-term first
            stats.working_memory_count += 1;
            if added_to_session {
                stats.session_memory_count += 1;
            }
            if indexed {
                stats.vector_index_count += 1;
            }
        }

        // Trigger background consolidation if needed
        self.consolidate_if_needed()?;

        // Commit and reload BM25 index changes (makes documents searchable immediately)
        // Note: This is done per-memory for immediate searchability.
        // For high-throughput scenarios, consider batching commits.
        if let Err(e) = self.hybrid_search.commit_and_reload() {
            tracing::warn!("Failed to commit/reload BM25 index: {}", e);
        }

        // Signal that fact extraction should run on next maintenance cycle
        self.fact_extraction_needed
            .store(true, std::sync::atomic::Ordering::Relaxed);

        Ok(memory_id)
    }

    /// Complete the deferred embedding + vector indexing for a previously stored memory.
    ///
    /// Call this after `remember_deferred()` to generate the ONNX embedding, persist it,
    /// index the memory in the vector DB, and run interference detection.
    ///
    /// Safe to call from a background thread (e.g. `tokio::task::spawn_blocking`).
    /// All errors are non-fatal — the memory remains searchable via BM25 regardless.
    pub fn embed_and_index(&self, memory_id: &MemoryId) -> Result<()> {
        let _span = tracing::info_span!("embed_and_index", memory_id = %memory_id.0).entered();
        let start = std::time::Instant::now();

        // Load fresh from storage (owned Memory, so we can mutate experience.embeddings)
        let mut memory = self
            .long_term_memory
            .get(memory_id)
            .context("Failed to load memory for background indexing")?;

        let importance = memory.importance();

        // Generate embedding if not already present
        if memory.experience.embeddings.is_none() {
            let content_hash = Self::sha256_hash(&memory.experience.content);

            if let Some(cached) = self.content_cache.get(&content_hash) {
                memory.experience.embeddings = Some(cached.clone());
                EMBEDDING_CACHE_CONTENT.with_label_values(&["hit"]).inc();
            } else {
                EMBEDDING_CACHE_CONTENT.with_label_values(&["miss"]).inc();
                match self.embedder.encode(&memory.experience.content) {
                    Ok(embedding) => {
                        self.content_cache.insert(content_hash, embedding.clone());
                        EMBEDDING_CACHE_CONTENT_SIZE
                            .set(self.content_cache.entry_count() as i64);
                        memory.experience.embeddings = Some(embedding);
                    }
                    Err(e) => {
                        tracing::warn!("Background embedding generation failed: {e}");
                        // Non-fatal: memory is still searchable via BM25
                        return Ok(());
                    }
                }
            }

            // Persist embedding to storage so restarts don't re-embed
            if let Err(e) = self.long_term_memory.update(&memory) {
                tracing::warn!("Failed to persist background embedding: {e}");
            }
        }

        // Vector index (uses pre-computed embedding for short content, re-embeds chunks)
        let memory_arc = Arc::new(memory);
        if let Err(e) = self.retriever.index_memory(&memory_arc) {
            tracing::warn!(
                "Background vector indexing failed for {}: {e}",
                memory_id.0
            );
        } else {
            self.stats.write().vector_index_count += 1;
        }

        // Write gate + interference detection (retroactive + proactive)
        if let Some(embedding) = &memory_arc.experience.embeddings {
            if let Ok(similar_ids) =
                self.retriever
                    .search_by_embedding(embedding, 5, Some(&memory_arc.id))
            {
                if !similar_ids.is_empty() {
                    let similar_memories: Vec<_> = similar_ids
                        .iter()
                        .filter_map(|(id, similarity)| {
                            self.retriever.get_from_storage(id).ok().map(|m| {
                                (
                                    id.0.to_string(),
                                    *similarity,
                                    m.importance(),
                                    m.created_at,
                                    m.experience.content.chars().take(50).collect::<String>(),
                                )
                            })
                        })
                        .collect();

                    // ── Write gate: absorption ──────────────────────────────
                    // If nearest neighbor exceeds absorption threshold AND we
                    // have enough memories to trust the index, absorb this
                    // memory into the existing one rather than keeping both.
                    let total_memories = self.stats.read().total_memories;
                    if total_memories >= crate::constants::WRITE_GATE_COLD_START_BYPASS {
                        if let Some((best_id, best_sim, _, _, _)) = similar_memories
                            .iter()
                            .max_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            })
                        {
                            if *best_sim >= crate::constants::WRITE_GATE_ABSORPTION_THRESHOLD {
                                // Boost existing memory's importance
                                if let Ok(best_uuid) = uuid::Uuid::parse_str(best_id) {
                                    if let Ok(existing) =
                                        self.long_term_memory.get(&MemoryId(best_uuid))
                                    {
                                        existing.boost_importance(
                                            crate::constants::WRITE_GATE_ABSORPTION_BOOST,
                                        );
                                        let _ = self.long_term_memory.update(&existing);
                                    }
                                }
                                // Remove the redundant new memory
                                let _ = self.long_term_memory.delete(memory_id);
                                self.retriever.remove_memory(memory_id);
                                self.record_consolidation_event(
                                    ConsolidationEvent::MemoryWeakened {
                                        memory_id: memory_id.0.to_string(),
                                        content_preview: memory_arc
                                            .experience
                                            .content
                                            .chars()
                                            .take(50)
                                            .collect(),
                                        activation_before: importance,
                                        activation_after: 0.0,
                                        interfering_memory_id: best_id.clone(),
                                        interference_type: InterferenceType::Retroactive,
                                        timestamp: chrono::Utc::now(),
                                    },
                                );
                                crate::metrics::WRITE_GATE_ABSORBED.inc();
                                tracing::info!(
                                    memory_id = %memory_id.0,
                                    absorbed_by = %best_id,
                                    similarity = %best_sim,
                                    "Write gate: absorbed redundant memory"
                                );
                                let elapsed = start.elapsed();
                                crate::metrics::EMBED_BACKGROUND_DURATION
                                    .observe(elapsed.as_secs_f64());
                                return Ok(());
                            }
                        }
                    }

                    // ── Standard interference detection ─────────────────────
                    if !similar_memories.is_empty() {
                        let interference_result =
                            self.interference_detector.write().check_interference(
                                &memory_arc.id.0.to_string(),
                                importance,
                                memory_arc.created_at,
                                &similar_memories,
                            );

                        for (old_id, _similarity, decay_amount) in
                            &interference_result.retroactive_targets
                        {
                            if let Ok(old_uuid) = uuid::Uuid::parse_str(old_id) {
                                if let Ok(old_memory) =
                                    self.long_term_memory.get(&MemoryId(old_uuid))
                                {
                                    old_memory.decay_importance(*decay_amount);
                                    let _ = self.long_term_memory.update(&old_memory);
                                }
                            }
                        }

                        if interference_result.proactive_decay > 0.0 {
                            memory_arc.decay_importance(interference_result.proactive_decay);
                            let _ = self.long_term_memory.update(&memory_arc);
                        }

                        for event in &interference_result.events {
                            self.record_consolidation_event(event.clone());
                        }

                        if interference_result.is_duplicate {
                            tracing::info!(
                                memory_id = %memory_arc.id.0,
                                "Near-duplicate detected (>=0.95 cosine), suppressing importance"
                            );
                            memory_arc.decay_importance(0.99);
                            let _ = self.long_term_memory.update(&memory_arc);
                        }

                        {
                            let detector = self.interference_detector.read();
                            let affected_ids = detector.get_affected_ids_from_check(
                                &memory_arc.id.0.to_string(),
                                &interference_result,
                            );
                            for (id, records) in detector.get_records_for_ids(&affected_ids) {
                                let _ =
                                    self.long_term_memory.save_interference_records(id, records);
                            }
                            let (total_events, _) = detector.stats();
                            let _ = self
                                .long_term_memory
                                .save_interference_event_count(total_events);
                        }
                    }
                }
            }
        }

        let elapsed = start.elapsed();
        tracing::debug!(
            memory_id = %memory_id.0,
            elapsed_ms = elapsed.as_millis(),
            "Background embed_and_index complete"
        );
        crate::metrics::EMBED_BACKGROUND_DURATION.observe(elapsed.as_secs_f64());

        Ok(())
    }

    /// Remember with agent context for multi-agent systems
    ///
    /// Same as `remember` but tracks which agent created the memory,
    /// enabling agent-specific retrieval and hierarchical memory tracking.
    pub fn remember_with_agent(
        &self,
        experience: Experience,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> Result<MemoryId> {
        self.remember_with_agent_impl(experience, created_at, agent_id, run_id, false)
    }

    /// Fast-path variant of `remember_with_agent` — skips content embedding,
    /// entity batch encoding, and vector indexing. Call `embed_and_index()` afterward.
    pub fn remember_with_agent_deferred(
        &self,
        experience: Experience,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> Result<MemoryId> {
        self.remember_with_agent_impl(experience, created_at, agent_id, run_id, true)
    }

    fn remember_with_agent_impl(
        &self,
        mut experience: Experience,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
        agent_id: Option<String>,
        run_id: Option<String>,
        defer_embedding: bool,
    ) -> Result<MemoryId> {
        // IDEMPOTENCY (issue #109): Content hash dedup (same as remember())
        if let Some(existing_id) = self
            .long_term_memory
            .get_by_content_hash(&experience.content)
        {
            tracing::debug!(
                existing_id = %existing_id.0,
                "Content dedup: returning existing memory (identical content already stored)"
            );
            return Ok(existing_id);
        }

        let memory_id = MemoryId(Uuid::new_v4());

        // Calculate importance
        let importance = self.calculate_importance(&experience);

        // PERFORMANCE: Content embedding cache
        // Skipped when defer_embedding=true — handled by embed_and_index() later.
        if !defer_embedding && experience.embeddings.is_none() {
            let content_hash = Self::sha256_hash(&experience.content);
            if let Some(cached_embedding) = self.content_cache.get(&content_hash) {
                experience.embeddings = Some(cached_embedding.clone());
                EMBEDDING_CACHE_CONTENT.with_label_values(&["hit"]).inc();
            } else {
                EMBEDDING_CACHE_CONTENT.with_label_values(&["miss"]).inc();
                if let Ok(embedding) = self.embedder.encode(&experience.content) {
                    self.content_cache.insert(content_hash, embedding.clone());
                    EMBEDDING_CACHE_CONTENT_SIZE.set(self.content_cache.entry_count() as i64);
                    experience.embeddings = Some(embedding);
                }
            }
        }

        // TEMPORAL EXTRACTION: Extract dates from content for temporal filtering
        if experience.temporal_refs.is_empty() {
            let temporal = crate::memory::query_parser::extract_temporal_refs(&experience.content);
            for temp_ref in temporal.refs {
                experience.temporal_refs.push(temp_ref.date.to_string());
            }
        }

        // Create memory with agent context
        let memory = Arc::new(Memory::new(
            memory_id.clone(),
            experience,
            importance,
            agent_id,
            run_id,
            None, // actor_id
            created_at,
        ));

        // FIX-R2: Compute elaboration score at encoding time
        memory.set_elaboration_score(Self::compute_elaboration_score(&memory));

        // Persist to RocksDB storage
        self.long_term_memory.store(&memory)?;
        self.logger.write().log_created(&memory, "working");

        // Add to working memory
        self.working_memory
            .write()
            .add_shared(Arc::clone(&memory))?;

        // Index for semantic search (skipped when deferred — embed_and_index handles it)
        if !defer_embedding {
            if let Err(e) = self.retriever.index_memory(&memory) {
                tracing::warn!("Failed to index memory {} in vector DB: {}", memory.id.0, e);
            }
        }

        // Add entities to knowledge graph with co-occurrence edges
        // PERF: Build entity structs and extract co-occurrences OUTSIDE the lock
        // GraphMemory is internally thread-safe; read lock allows concurrent graph access
        if let Some(graph) = &self.graph_memory {
            let now = chrono::Utc::now();

            // Phase 1: Build entity structs with proper labels from NER
            let ner_lookup = build_ner_lookup(&memory.experience.ner_entities);

            // Batch-encode entity names for concept-level dedup
            // When defer_embedding=true, skip ONNX batch encoding — graph still works,
            // entities are created without embeddings (semantic edge weighting falls back to 1.0)
            let entity_names: Vec<&str> = memory
                .experience
                .entities
                .iter()
                .map(|s| s.as_str())
                .collect();
            let entity_embeddings: Vec<Option<Vec<f32>>> = if entity_names.is_empty() {
                Vec::new()
            } else if defer_embedding {
                vec![None; entity_names.len()]
            } else {
                match self.embedder.encode_batch(&entity_names) {
                    Ok(embs) => embs.into_iter().map(Some).collect(),
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            "Entity name embedding failed, concept merge disabled for this batch"
                        );
                        vec![None; entity_names.len()]
                    }
                }
            };

            let entities_to_add: Vec<crate::graph_memory::EntityNode> = memory
                .experience
                .entities
                .iter()
                .zip(entity_embeddings)
                .map(|(entity_name, embedding)| {
                    let (label, salience) = resolve_entity_label(entity_name, &ner_lookup);
                    crate::graph_memory::EntityNode {
                        uuid: Uuid::new_v4(),
                        name: entity_name.clone(),
                        labels: vec![label],
                        created_at: now,
                        last_seen_at: now,
                        mention_count: 1,
                        summary: String::new(),
                        attributes: std::collections::HashMap::new(),
                        name_embedding: embedding,
                        salience,
                        is_proper_noun: entity_name
                            .chars()
                            .next()
                            .map(|c| c.is_uppercase())
                            .unwrap_or(false),
                    }
                })
                .collect();

            // Phase 2: Use pre-extracted co-occurrence pairs or extract fresh
            let cooccurrence_pairs = if !memory.experience.cooccurrence_pairs.is_empty() {
                memory.experience.cooccurrence_pairs.clone()
            } else {
                let entity_extractor = crate::graph_memory::EntityExtractor::new();
                entity_extractor.extract_cooccurrence_pairs(&memory.experience.content)
            };

            let edge_context = format!("Co-occurred in memory {}", memory.id.0);

            // Phase 3: Acquire read lock for graph insertions (GraphMemory is internally thread-safe)
            let graph_guard = graph.read();

            for entity in entities_to_add {
                if let Err(e) = graph_guard.add_entity(entity.clone()) {
                    tracing::debug!("Failed to add entity '{}' to graph: {}", entity.name, e);
                }
            }

            // Semantic edge weighting
            let l1_base_weight = crate::graph_memory::EdgeTier::L1Working.initial_weight();
            for (entity1, entity2) in cooccurrence_pairs {
                if let (Ok(Some(e1)), Ok(Some(e2))) = (
                    graph_guard.find_entity_by_name(&entity1),
                    graph_guard.find_entity_by_name(&entity2),
                ) {
                    let entity_confidence = Some((e1.salience + e2.salience) / 2.0);

                    let semantic_weight = match (&e1.name_embedding, &e2.name_embedding) {
                        (Some(emb1), Some(emb2)) => {
                            let sim = crate::similarity::cosine_similarity(emb1, emb2).max(0.0);
                            EDGE_SEMANTIC_WEIGHT_FLOOR + (1.0 - EDGE_SEMANTIC_WEIGHT_FLOOR) * sim
                        }
                        _ => 1.0,
                    };

                    let edge = crate::graph_memory::RelationshipEdge {
                        uuid: Uuid::new_v4(),
                        from_entity: e1.uuid,
                        to_entity: e2.uuid,
                        relation_type: crate::graph_memory::RelationType::CoOccurs,
                        strength: l1_base_weight * semantic_weight,
                        created_at: now,
                        valid_at: now,
                        invalidated_at: None,
                        source_episode_id: Some(memory.id.0),
                        context: edge_context.clone(),
                        last_activated: now,
                        activation_count: 1,
                        ltp_status: crate::graph_memory::LtpStatus::None,
                        tier: crate::graph_memory::EdgeTier::L1Working,
                        activation_timestamps: None,
                        entity_confidence,
                        created_by: crate::graph_memory::EdgeSource::CoOccurrence,
                    };

                    if let Err(e) = graph_guard.add_relationship(edge) {
                        tracing::trace!(
                            "Failed to add co-occurrence edge {}<->{}: {}",
                            entity1,
                            entity2,
                            e
                        );
                    }
                }
            }

            // Phase 4: Extract causal edges from content
            // Detect causal language patterns (because, therefore, as a result, etc.)
            // and create directed Causes/ResultsIn edges between co-occurring entities.
            let causal_pairs = extract_causal_pairs(
                &memory.experience.content,
                &memory.experience.entities,
            );
            if !causal_pairs.is_empty() {
                let causal_context = format!("Causal relation in memory {}", memory.id.0);
                for (cause_entity, effect_entity, relation) in &causal_pairs {
                    if let (Ok(Some(e1)), Ok(Some(e2))) = (
                        graph_guard.find_entity_by_name(cause_entity),
                        graph_guard.find_entity_by_name(effect_entity),
                    ) {
                        let edge = crate::graph_memory::RelationshipEdge {
                            uuid: Uuid::new_v4(),
                            from_entity: e1.uuid,
                            to_entity: e2.uuid,
                            relation_type: relation.clone(),
                            strength: l1_base_weight * 0.8, // slightly below CoOccurs until confirmed
                            created_at: now,
                            valid_at: now,
                            invalidated_at: None,
                            source_episode_id: Some(memory.id.0),
                            context: causal_context.clone(),
                            last_activated: now,
                            activation_count: 1,
                            ltp_status: crate::graph_memory::LtpStatus::None,
                            tier: crate::graph_memory::EdgeTier::L1Working,
                            activation_timestamps: None,
                            entity_confidence: None,
                            created_by: crate::graph_memory::EdgeSource::CoOccurrence,
                        };
                        if let Err(e) = graph_guard.add_relationship(edge) {
                            tracing::trace!(
                                "Failed to add causal edge {}→{}: {}",
                                cause_entity,
                                effect_entity,
                                e
                            );
                        }
                    }
                }
                tracing::debug!(
                    "Extracted {} causal edges from memory {}",
                    causal_pairs.len(),
                    memory.id.0
                );
            }
        }

        // Index in BM25 for hybrid search
        if let Err(e) = self.hybrid_search.index_memory(
            &memory.id,
            &memory.experience.content,
            &memory.experience.tags,
            &memory.experience.entities,
        ) {
            tracing::warn!("Failed to index memory {} in BM25: {}", memory.id.0, e);
        }

        // If important enough, add to session memory
        if importance > self.config.importance_threshold {
            self.session_memory
                .write()
                .add_shared(Arc::clone(&memory))?;
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories += 1;
            stats.long_term_memory_count += 1;
            stats.working_memory_count += 1;
        }

        self.consolidate_if_needed()?;

        // Commit and reload BM25 index changes
        if let Err(e) = self.hybrid_search.commit_and_reload() {
            tracing::warn!("Failed to commit/reload BM25 index: {}", e);
        }

        // Signal that fact extraction should run on next maintenance cycle
        self.fact_extraction_needed
            .store(true, std::sync::atomic::Ordering::Relaxed);

        Ok(memory_id)
    }


    // ==========================================================================
    // TEMPORAL FACT EXTRACTION (for multi-hop temporal queries)
    // ==========================================================================

    /// Extract and store temporal facts from a memory
    ///
    /// Call this after remember() when you have access to user_id.
    /// Extracts facts like "Melanie is planning camping next month" and stores them
    /// with resolved absolute dates for accurate multi-hop retrieval.
    pub fn store_temporal_facts_for_memory(
        &self,
        user_id: &str,
        memory_id: &MemoryId,
        content: &str,
        entities: &[String],
        created_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        let facts =
            temporal_facts::extract_temporal_facts(content, memory_id, created_at, entities);
        if facts.is_empty() {
            return Ok(0);
        }

        // Run contradiction detection before storing each new fact.
        // This invalidates old facts that the new fact supersedes.
        let mut total_invalidated = 0usize;
        for fact in &facts {
            match self
                .temporal_fact_store
                .detect_and_resolve_contradictions(user_id, fact)
            {
                Ok(ids) => total_invalidated += ids.len(),
                Err(e) => {
                    tracing::warn!(
                        user_id = user_id,
                        fact_id = %fact.id,
                        error = %e,
                        "Contradiction detection failed for temporal fact"
                    );
                }
            }
        }

        let stored = self.temporal_fact_store.store_batch(user_id, &facts)?;
        if stored > 0 {
            tracing::debug!(
                user_id = user_id,
                memory_id = %memory_id.0,
                facts_stored = stored,
                facts_invalidated = total_invalidated,
                "Stored temporal facts for memory"
            );
        }
        Ok(stored)
    }

    /// Find temporal facts by entity and event keywords
    ///
    /// Used for multi-hop queries like "When did Melanie paint a sunrise?"
    /// Returns facts sorted by conversation date (earliest first for planning queries).
    pub fn find_temporal_facts(
        &self,
        user_id: &str,
        entity: &str,
        event_keywords: &[&str],
        event_type: Option<temporal_facts::EventType>,
    ) -> Result<Vec<temporal_facts::TemporalFact>> {
        self.temporal_fact_store.find_by_entity_and_event(
            user_id,
            entity,
            event_keywords,
            event_type,
        )
    }

    /// Find temporal facts by entity and event keywords, optionally including expired facts.
    pub fn find_temporal_facts_filtered(
        &self,
        user_id: &str,
        entity: &str,
        event_keywords: &[&str],
        event_type: Option<temporal_facts::EventType>,
        include_expired: bool,
    ) -> Result<Vec<temporal_facts::TemporalFact>> {
        self.temporal_fact_store.find_by_entity_and_event_filtered(
            user_id,
            entity,
            event_keywords,
            event_type,
            include_expired,
        )
    }

    /// List all temporal facts for a user
    pub fn list_temporal_facts(
        &self,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<temporal_facts::TemporalFact>> {
        self.temporal_fact_store.list(user_id, limit)
    }

    /// List all temporal facts for a user, optionally including expired facts.
    pub fn list_temporal_facts_filtered(
        &self,
        user_id: &str,
        limit: usize,
        include_expired: bool,
    ) -> Result<Vec<temporal_facts::TemporalFact>> {
        self.temporal_fact_store
            .list_filtered(user_id, limit, include_expired)
    }

    /// Find temporal facts by entity name only
    pub fn find_temporal_facts_by_entity(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
    ) -> Result<Vec<temporal_facts::TemporalFact>> {
        self.temporal_fact_store.find_by_entity(user_id, entity, limit)
    }

    /// Find temporal facts by entity name, optionally including expired facts.
    pub fn find_temporal_facts_by_entity_filtered(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
        include_expired: bool,
    ) -> Result<Vec<temporal_facts::TemporalFact>> {
        self.temporal_fact_store
            .find_by_entity_filtered(user_id, entity, limit, include_expired)
    }

    /// Find temporal facts by event keyword only
    pub fn find_temporal_facts_by_event(
        &self,
        user_id: &str,
        event: &str,
        limit: usize,
    ) -> Result<Vec<temporal_facts::TemporalFact>> {
        self.temporal_fact_store.find_by_event(user_id, event, limit)
    }

    /// Find temporal facts by event keyword, optionally including expired facts.
    pub fn find_temporal_facts_by_event_filtered(
        &self,
        user_id: &str,
        event: &str,
        limit: usize,
        include_expired: bool,
    ) -> Result<Vec<temporal_facts::TemporalFact>> {
        self.temporal_fact_store
            .find_by_event_filtered(user_id, event, limit, include_expired)
    }

    /// Filter temporal facts to those whose conversation_date falls within
    /// the parsed query time window (±7 days around the date range).
    ///
    /// Returns the subset of `facts` that overlap the window. If
    /// `query_temporal` has no parsed refs, returns an empty vec so callers
    /// can fall through to the unfiltered set.
    fn filter_facts_by_time_window<'a>(
        facts: &'a [temporal_facts::TemporalFact],
        query_temporal: &query_parser::TemporalExtraction,
    ) -> Vec<&'a temporal_facts::TemporalFact> {
        let (earliest, latest) = match query_temporal.date_range() {
            Some(range) => range,
            None => return Vec::new(),
        };

        // Widen the window by 7 days on each side to tolerate fuzzy dates
        let window_start = earliest - chrono::Duration::days(7);
        let window_end = latest + chrono::Duration::days(7);

        facts
            .iter()
            .filter(|f| {
                let fact_date = f.conversation_date.date_naive();
                fact_date >= window_start && fact_date <= window_end
            })
            .collect()
    }


    /// Compute SHA256 hash of text for stable cache keys
    ///
    /// Unlike std::hash::DefaultHasher, SHA256 produces deterministic hashes
    /// across process restarts and Rust versions. This is critical for:
    /// - Embedding cache persistence (future feature)
    /// - Consistent behavior across restarts
    /// - Avoiding cache key collisions
    #[inline]
    fn sha256_hash(text: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hasher.finalize().into()
    }

    /// Forget memories based on criteria
    /// Thread-safe: uses interior mutability for all internal state
    pub fn forget(&self, criteria: ForgetCriteria) -> Result<usize> {
        let forgotten_count = match criteria {
            ForgetCriteria::ById(memory_id) => {
                // Delete a single memory by ID from all tiers, tracking which tiers had it
                let mut deleted_from_any = false;
                let mut was_in_working = false;
                let mut was_in_session = false;
                let mut was_in_longterm = false;

                // Remove from working memory
                if self.working_memory.write().remove(&memory_id).is_ok() {
                    deleted_from_any = true;
                    was_in_working = true;
                }

                // Remove from session memory
                if self.session_memory.write().remove(&memory_id).is_ok() {
                    deleted_from_any = true;
                    was_in_session = true;
                }

                // Remove from long-term storage
                if self.long_term_memory.delete(&memory_id).is_ok() {
                    deleted_from_any = true;
                    was_in_longterm = true;
                }

                // Remove from vector index (soft delete) - CRITICAL for semantic search
                // This marks the vector as deleted so it won't appear in search results
                let was_indexed = self.retriever.remove_memory(&memory_id);

                // Clean up knowledge graph episode and sourced edges
                if let Some(graph) = &self.graph_memory {
                    if let Err(e) = graph.read().delete_episode(&memory_id.0) {
                        tracing::warn!(
                            memory_id = %memory_id.0,
                            error = %e,
                            "Failed to clean up graph episode for deleted memory"
                        );
                    }
                }

                // Clean up BM25 keyword index
                if let Err(e) = self.hybrid_search.remove_memory(&memory_id) {
                    tracing::warn!(
                        memory_id = %memory_id.0,
                        error = %e,
                        "Failed to clean BM25 index for deleted memory"
                    );
                }

                // Clean up interference records
                self.cleanup_interference_for_ids(std::slice::from_ref(&memory_id));

                // Update stats - decrement each tier count that had this memory
                if deleted_from_any {
                    let mut stats = self.stats.write();
                    stats.total_memories = stats.total_memories.saturating_sub(1);
                    if was_in_working {
                        stats.working_memory_count = stats.working_memory_count.saturating_sub(1);
                    }
                    if was_in_session {
                        stats.session_memory_count = stats.session_memory_count.saturating_sub(1);
                    }
                    if was_in_longterm {
                        stats.long_term_memory_count =
                            stats.long_term_memory_count.saturating_sub(1);
                    }
                    if was_indexed {
                        stats.vector_index_count = stats.vector_index_count.saturating_sub(1);
                    }
                    1
                } else {
                    0
                }
            }
            ForgetCriteria::OlderThan(days) => {
                let cutoff = chrono::Utc::now() - chrono::Duration::days(days as i64);

                // Remove from working memory
                let working_removed = self.working_memory.write().remove_older_than(cutoff)?;

                // Remove from session memory
                let session_removed = self.session_memory.write().remove_older_than(cutoff)?;

                // Mark as forgotten in long-term (don't delete, just flag)
                let flagged_ids = self.long_term_memory.mark_forgotten_by_age(cutoff)?;
                let lt_flagged = flagged_ids.len();

                // Clean up secondary indices for soft-forgotten memories
                for id in &flagged_ids {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                }
                self.cleanup_graph_for_ids(&flagged_ids);
                self.cleanup_interference_for_ids(&flagged_ids);

                // Update stats for hard-deleted and soft-deleted tiers
                {
                    let removed = working_removed + session_removed + lt_flagged;
                    if removed > 0 {
                        let mut stats = self.stats.write();
                        stats.working_memory_count =
                            stats.working_memory_count.saturating_sub(working_removed);
                        stats.session_memory_count =
                            stats.session_memory_count.saturating_sub(session_removed);
                        stats.long_term_memory_count =
                            stats.long_term_memory_count.saturating_sub(lt_flagged);
                        stats.total_memories = stats.total_memories.saturating_sub(removed);
                    }
                }

                lt_flagged
            }
            ForgetCriteria::LowImportance(threshold) => {
                let working_removed = self
                    .working_memory
                    .write()
                    .remove_below_importance(threshold)?;
                let session_removed = self
                    .session_memory
                    .write()
                    .remove_below_importance(threshold)?;
                let flagged_ids = self
                    .long_term_memory
                    .mark_forgotten_by_importance(threshold)?;
                let lt_flagged = flagged_ids.len();

                // Clean up secondary indices for soft-forgotten memories
                for id in &flagged_ids {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                }
                self.cleanup_graph_for_ids(&flagged_ids);
                self.cleanup_interference_for_ids(&flagged_ids);

                // Update stats for hard-deleted and soft-deleted tiers
                {
                    let removed = working_removed + session_removed + lt_flagged;
                    if removed > 0 {
                        let mut stats = self.stats.write();
                        stats.working_memory_count =
                            stats.working_memory_count.saturating_sub(working_removed);
                        stats.session_memory_count =
                            stats.session_memory_count.saturating_sub(session_removed);
                        stats.long_term_memory_count =
                            stats.long_term_memory_count.saturating_sub(lt_flagged);
                        stats.total_memories = stats.total_memories.saturating_sub(removed);
                    }
                }

                lt_flagged
            }
            ForgetCriteria::Pattern(pattern) => {
                // Remove memories matching pattern
                self.forget_by_pattern(&pattern)?
            }
            ForgetCriteria::ByTags(tags) => {
                // Remove memories matching ANY of the specified tags
                self.forget_by_tags(&tags)?
            }
            ForgetCriteria::ByDateRange { start, end } => {
                // Remove memories within the date range
                self.forget_by_date_range(start, end)?
            }
            ForgetCriteria::ByType(exp_type) => {
                // Remove memories of a specific type
                self.forget_by_type(exp_type)?
            }
            ForgetCriteria::All => {
                // GDPR: Clear ALL memories for the user
                self.forget_all()?
            }
        };

        // Commit BM25 changes after any deletion to make removals visible
        if forgotten_count > 0 {
            if let Err(e) = self.hybrid_search.commit_and_reload() {
                tracing::warn!(error = %e, "Failed to commit BM25 after forget");
            }
        }

        Ok(forgotten_count)
    }

    /// Get memory statistics
    ///
    /// Returns current stats with fresh average_importance calculated from storage.
    /// Most counters are cached in-memory for performance, but importance is
    /// recalculated to ensure accuracy after memory modifications.
    pub fn stats(&self) -> MemoryStats {
        let mut stats = self.stats.read().clone();

        // Recalculate average_importance from storage for accuracy
        // This ensures importance reflects current memory state after adds/deletes
        if let Ok(storage_stats) = self.long_term_memory.get_stats() {
            stats.average_importance = storage_stats.average_importance;
        }

        stats
    }

    /// Export visualization graph as DOT format for Graphviz
    pub fn export_visualization_dot(&self) -> String {
        self.logger.read().graph.to_dot()
    }

    /// Build visualization graph from current memory state
    /// Call this to populate the visualization graph with all current memories
    pub fn build_visualization_graph(&self) -> Result<visualization::GraphStats> {
        let mut logger = self.logger.write();

        // Add working memory entries directly to the graph (bypasses enabled check)
        for memory in self.working_memory.read().all_memories() {
            logger.graph.add_memory(&memory, "working");
        }

        // Add session memory entries
        for memory in self.session_memory.read().all_memories() {
            logger.graph.add_memory(&memory, "session");
        }

        // Add long-term memory entries
        for memory in self.long_term_memory.get_all()? {
            logger.graph.add_memory(&memory, "longterm");
        }

        Ok(logger.get_stats())
    }

    /// Get reference to embedder for graph-aware retrieval
    pub fn get_embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    /// Compute embedding for arbitrary text (for external use like prospective memory)
    pub fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder.encode(text)
    }

    /// Get all memories across all tiers for graph-aware retrieval
    /// Deduplicates by memory ID, preferring working > session > long-term
    /// Check if a memory has an active reconsolidation shadow (FIX-R1).
    /// Returns true if the memory was recently retrieved and is currently labile.
    /// Used to set retrieval_trigger = "co_activation" in proactive context.
    pub fn has_active_shadow(&self, memory_id: &MemoryId) -> bool {
        self.reconsolidation_shadows.read().contains_key(memory_id)
    }

    pub fn get_all_memories(&self) -> Result<Vec<SharedMemory>> {
        use std::collections::HashSet;
        let mut seen_ids: HashSet<MemoryId> = HashSet::new();
        let mut all_memories = Vec::new();

        // Collect from working memory (highest priority - most recent/active)
        {
            let working = self.working_memory.read();
            for mem in working.all_memories() {
                if seen_ids.insert(mem.id.clone()) {
                    all_memories.push(mem);
                }
            }
        }

        // Collect from session memory (medium priority)
        {
            let session = self.session_memory.read();
            for mem in session.all_memories() {
                if seen_ids.insert(mem.id.clone()) {
                    all_memories.push(mem);
                }
            }
        }

        // Collect from long-term memory (lowest priority - wrap in Arc)
        {
            let longterm_mems = self.long_term_memory.get_all()?;
            for mem in longterm_mems {
                if seen_ids.insert(mem.id.clone()) {
                    all_memories.push(Arc::new(mem));
                }
            }
        }

        Ok(all_memories)
    }

    /// Find a memory by UUID prefix across all tiers.
    ///
    /// Accepts both full UUIDs and 8+ char hex prefixes (as displayed by MCP tools).
    /// Searches working → session → long-term memory with deduplication.
    /// Returns `Err` for ambiguous prefixes (multiple matches).
    pub fn find_memory_by_prefix(&self, id_prefix: &str) -> Result<Option<SharedMemory>> {
        // Fast path: try full UUID first via direct lookup
        if let Ok(uuid) = uuid::Uuid::parse_str(id_prefix) {
            let target_id = MemoryId(uuid);
            let all = self.get_all_memories()?;
            return Ok(all.into_iter().find(|m| m.id == target_id));
        }

        // Prefix search across all tiers
        let prefix_lower = id_prefix.to_lowercase();
        let all_memories = self.get_all_memories()?;
        let matches: Vec<SharedMemory> = all_memories
            .into_iter()
            .filter(|m| {
                m.id.0
                    .to_string()
                    .replace('-', "")
                    .to_lowercase()
                    .starts_with(&prefix_lower)
            })
            .collect();

        match matches.len() {
            0 => Ok(None),
            1 => Ok(Some(matches.into_iter().next().unwrap())),
            n => Err(anyhow::anyhow!(
                "Ambiguous memory ID prefix '{}': matches {} memories",
                id_prefix,
                n
            )),
        }
    }

    /// Get memories from working memory tier (highest activation, most recent)
    pub fn get_working_memories(&self) -> Vec<SharedMemory> {
        let working = self.working_memory.read();
        working.all_memories()
    }

    /// Get memories from session memory tier (medium-term, consolidated)
    pub fn get_session_memories(&self) -> Vec<SharedMemory> {
        let session = self.session_memory.read();
        session.all_memories()
    }

    /// Get memories from long-term memory tier (persistent, lower activation)
    /// Returns up to `limit` memories to avoid overwhelming responses
    pub fn get_longterm_memories(&self, limit: usize) -> Result<Vec<Memory>> {
        let all = self.long_term_memory.get_all()?;
        Ok(all.into_iter().take(limit).collect())
    }

    /// Compute elaboration quality score for a memory (FIX-R2).
    ///
    /// Measures how well-contextualized a memory is on a 0.0-1.0 scale:
    /// - 0.0 = bare S-rep (raw content, no context, no entities, no threading)
    /// - 1.0 = fully elaborated C-rep (rich context, diverse entities, temporal binding)
    ///
    /// Reference: Ehlers & Clark (2000) — poor elaboration/contextualization at encoding
    /// produces fragmented memories that intrude without temporal grounding.
    fn compute_elaboration_score(memory: &Memory) -> f32 {
        let max_dimensions = 7.0_f32;
        let mut score = 0.0_f32;

        // 1. Context richness (0 or 1): reuse existing context_richness() / 10
        let ctx_richness = memory.context_richness() as f32 / 10.0;
        score += ctx_richness.min(1.0);

        // 2. Entity diversity (0 or 1): more unique entities = better elaboration
        let entity_count = memory.experience.entities.len();
        score += (entity_count as f32 / 5.0).min(1.0);

        // 3. Emotional signals present (0 or 1)
        let has_emotion = memory.experience.context.as_ref().is_some_and(|ctx| {
            ctx.emotional.valence != 0.0 || ctx.emotional.arousal != 0.0
        });
        if has_emotion {
            score += 1.0;
        }

        // 4. Episode threading (0 or 1): linked to preceding memory
        let has_threading = memory.experience.context.as_ref().is_some_and(|ctx| {
            ctx.episode.preceding_memory_id.is_some()
        });
        if has_threading {
            score += 1.0;
        }

        // 5. Temporal references (0 or 1): has extractable dates/times
        if !memory.experience.temporal_refs.is_empty() {
            score += 1.0;
        }

        // 6. Tags present (0 or 1): intentional classification
        if !memory.experience.tags.is_empty() {
            score += 1.0;
        }

        // 7. Content density (0 or 1): information per word (entities/words ratio)
        let word_count = memory.experience.content.split_whitespace().count();
        let density = if word_count > 0 {
            entity_count as f32 / word_count as f32
        } else {
            0.0
        };
        score += (density * 10.0).min(1.0);

        (score / max_dimensions).clamp(0.0, 1.0)
    }

    /// Calculate importance of an experience using multi-factor analysis
    fn calculate_importance(&self, experience: &Experience) -> f32 {
        let mut factors = Vec::new();

        // Factor 1: Experience type base score (0.0 - 0.3)
        let type_score = match experience.experience_type {
            ExperienceType::Decision => 0.3,
            ExperienceType::Error => 0.25,
            ExperienceType::Learning => 0.25,
            ExperienceType::Discovery => 0.2,
            ExperienceType::Pattern => 0.2,
            ExperienceType::Task => 0.15,
            ExperienceType::Conversation => 0.1,
            ExperienceType::Context => 0.1,
            _ => 0.05,
        };
        factors.push(("type", type_score));

        // Factor 2: Content richness (0.0 - 0.25)
        let _content_length = experience.content.len();
        let word_count = experience.content.split_whitespace().count();
        let richness_score = if word_count > 50 {
            0.25
        } else if word_count > 20 {
            0.15
        } else if word_count > 5 {
            0.08
        } else {
            0.02
        };
        factors.push(("richness", richness_score));

        // Factor 3: Entity density (0.0 - 0.2)
        let entity_score = if experience.entities.len() > 10 {
            0.2
        } else if experience.entities.len() > 5 {
            0.15
        } else if experience.entities.len() > 2 {
            0.1
        } else if !experience.entities.is_empty() {
            0.05
        } else {
            0.0
        };
        factors.push(("entities", entity_score));

        // Factor 4: Context depth (0.0 - 0.2)
        let context_score = if let Some(ctx) = &experience.context {
            let mut score: f32 = 0.0;

            // Rich semantic context
            if !ctx.semantic.concepts.is_empty() {
                score += 0.05;
            }
            if !ctx.semantic.tags.is_empty() {
                score += 0.03;
            }
            if !ctx.semantic.related_concepts.is_empty() {
                score += 0.04;
            }

            // Project/workspace context
            if ctx.project.project_id.is_some() {
                score += 0.03;
            }

            // Code context
            if ctx.code.current_file.is_some() {
                score += 0.03;
            }

            // Document citations
            if !ctx.document.citations.is_empty() {
                score += 0.02;
            }

            score.min(0.2)
        } else {
            0.0
        };
        factors.push(("context", context_score));

        // Factor 5: Metadata signals (0.0 - 0.15)
        let mut metadata_score: f32 = 0.0;

        if experience.metadata.contains_key("priority") {
            if let Some(priority) = experience.metadata.get("priority") {
                metadata_score += match priority.as_str() {
                    "critical" => 0.15,
                    "high" => 0.10,
                    "medium" => 0.05,
                    _ => 0.0,
                };
            }
        }

        if experience.metadata.contains_key("unexpected") {
            metadata_score += 0.08;
        }

        if experience.metadata.contains_key("breakthrough") {
            metadata_score += 0.12;
        }

        if experience.metadata.get("role") == Some(&"user".to_string()) {
            metadata_score += 0.02; // User messages slightly more important
        }

        factors.push(("metadata", metadata_score.min(0.15)));

        // Factor 6: Embeddings quality (0.0 - 0.1)
        let embedding_score = if let Some(emb) = &experience.embeddings {
            if emb.len() >= 384 {
                // Full embedding vector
                0.1
            } else {
                0.05
            }
        } else {
            0.0
        };
        factors.push(("embeddings", embedding_score));

        // Factor 7: Content quality indicators (0.0 - 0.1)
        let content_lower = experience.content.to_lowercase();
        let mut quality_score: f32 = 0.0;

        // Technical terms indicate higher quality
        let technical_terms = [
            "algorithm",
            "architecture",
            "implementation",
            "optimization",
            "performance",
            "security",
            "database",
            "api",
            "framework",
        ];
        for term in &technical_terms {
            if content_lower.contains(term) {
                quality_score += 0.015;
            }
        }

        // Questions indicate learning/discovery
        if content_lower.contains('?') {
            quality_score += 0.02;
        }

        // Code snippets indicate actionable content
        if experience.content.contains("```")
            || experience.content.contains("fn ")
            || experience.content.contains("function ")
            || experience.content.contains("class ")
        {
            quality_score += 0.03;
        }

        factors.push(("quality", quality_score.min(0.1)));

        // Aggregate all factors
        let importance: f32 = factors.iter().map(|(_, score)| score).sum();

        // Ensure importance is in valid range [0.0, 1.0]
        let importance = importance.clamp(0.0, 1.0);

        // Log importance calculation for transparency
        if importance > 0.7 {
            debug!("High importance memory: {:.2} (type={:.2}, richness={:.2}, entities={:.2}, context={:.2})",
                importance,
                factors.iter().find(|(k, _)| *k == "type").map(|(_, v)| v).unwrap_or(&0.0),
                factors.iter().find(|(k, _)| *k == "richness").map(|(_, v)| v).unwrap_or(&0.0),
                factors.iter().find(|(k, _)| *k == "entities").map(|(_, v)| v).unwrap_or(&0.0),
                factors.iter().find(|(k, _)| *k == "context").map(|(_, v)| v).unwrap_or(&0.0)
            );
        }

        importance
    }

    /// Consolidate memories based on Cowan's model (importance + time, not size)
    ///
    /// Tier promotion criteria:
    /// - Working → Session: importance >= 0.4 AND age >= 5 minutes
    /// - Session → LongTerm: importance >= 0.6 AND age >= 1 hour
    fn consolidate_if_needed(&self) -> Result<()> {
        // Promote eligible memories from working to session (importance + time based)
        self.promote_working_to_session()?;

        // Promote eligible memories from session to long-term (importance + time based)
        self.promote_session_to_longterm()?;

        // Compress old memories if auto-compress is enabled
        if self.config.auto_compress {
            self.compress_old_memories()?;
        }

        Ok(())
    }

    /// Move memories from working to session memory (Cowan's model)
    ///
    /// Promotion criteria: importance >= TIER_PROMOTION_WORKING_IMPORTANCE
    /// AND age >= TIER_PROMOTION_WORKING_AGE_SECS
    fn promote_working_to_session(&self) -> Result<()> {
        let now = chrono::Utc::now();
        let min_age = chrono::Duration::seconds(TIER_PROMOTION_WORKING_AGE_SECS);

        // Find eligible memories (importance + time threshold, with graph-adjusted threshold)
        let to_promote: Vec<SharedMemory> = {
            let working = self.working_memory.read();
            working
                .all_memories()
                .into_iter()
                .filter(|m| {
                    let age = now - m.created_at;
                    let importance = m.importance();
                    let threshold =
                        self.graph_adjusted_threshold(m, TIER_PROMOTION_WORKING_IMPORTANCE);
                    importance >= threshold && age >= min_age
                })
                .collect()
        };

        if to_promote.is_empty() {
            return Ok(());
        }

        let count = to_promote.len();
        let mut working = self.working_memory.write();
        let mut session = self.session_memory.write();

        for memory in &to_promote {
            // Log promotion
            self.logger
                .write()
                .log_promoted(&memory.id, "working", "session", count);

            // Clone out of Arc and update tier before session storage
            let mut promoted_memory = (**memory).clone();
            promoted_memory.promote(); // Working -> Session
            session.add(promoted_memory)?;
            working.remove(&memory.id)?;
        }

        if count > 0 {
            let mut stats = self.stats.write();
            stats.promotions_to_session += count;
            stats.working_memory_count = stats.working_memory_count.saturating_sub(count);
            stats.session_memory_count += count;
            tracing::debug!(
                "Promoted {} memories from working to session (importance >= {}, age >= {}s)",
                count,
                TIER_PROMOTION_WORKING_IMPORTANCE,
                TIER_PROMOTION_WORKING_AGE_SECS
            );
        }
        Ok(())
    }

    /// Move memories from session to long-term storage (Cowan's model)
    ///
    /// Promotion criteria: importance >= TIER_PROMOTION_SESSION_IMPORTANCE
    /// AND age >= TIER_PROMOTION_SESSION_AGE_SECS
    fn promote_session_to_longterm(&self) -> Result<()> {
        let now = chrono::Utc::now();
        let min_age = chrono::Duration::seconds(TIER_PROMOTION_SESSION_AGE_SECS);

        // Find eligible memories (importance + time threshold, with graph-adjusted threshold)
        let to_promote: Vec<SharedMemory> = {
            let session = self.session_memory.read();
            session
                .all_memories()
                .into_iter()
                .filter(|m| {
                    let age = now - m.created_at;
                    let importance = m.importance();
                    let threshold =
                        self.graph_adjusted_threshold(m, TIER_PROMOTION_SESSION_IMPORTANCE);
                    importance >= threshold && age >= min_age
                })
                .collect()
        };

        if to_promote.is_empty() {
            return Ok(());
        }

        let count = to_promote.len();
        let mut session = self.session_memory.write();

        for memory in &to_promote {
            // Log promotion
            self.logger
                .write()
                .log_promoted(&memory.id, "session", "longterm", count);

            // Clone out of Arc and update tier before long-term storage
            let mut owned_memory = (**memory).clone();
            owned_memory.promote(); // Session -> LongTerm

            // Compress if old enough
            let compressed_memory = if self.should_compress(&owned_memory) {
                self.compressor.compress(&owned_memory)?
            } else {
                owned_memory
            };

            // Store in long-term
            self.long_term_memory.store(&compressed_memory)?;

            // PRODUCTION: Index memory in Vamana vector DB for semantic search
            if let Err(e) = self.retriever.index_memory(&compressed_memory) {
                tracing::warn!(
                    "Failed to index memory {} in vector DB: {}",
                    compressed_memory.id.0,
                    e
                );
                // Don't fail promotion if indexing fails - memory is still stored
            }

            // Remove from session
            session.remove(&memory.id)?;
        }

        if count > 0 {
            let mut stats = self.stats.write();
            stats.promotions_to_longterm += count;
            stats.session_memory_count = stats.session_memory_count.saturating_sub(count);
            stats.long_term_memory_count += count;
            tracing::debug!(
                "Promoted {} memories from session to long-term (importance >= {}, age >= {}s)",
                count,
                TIER_PROMOTION_SESSION_IMPORTANCE,
                TIER_PROMOTION_SESSION_AGE_SECS
            );
        }
        Ok(())
    }

    // =========================================================================
    // Memory-Edge Tier Coupling Methods
    // =========================================================================

    /// Calculate graph-adjusted importance threshold for tier promotion (Direction 3).
    ///
    /// Well-connected memories (many L2+ edges) get a discount on the promotion threshold.
    /// Isolated memories (entities but no edges) get a penalty.
    /// Memories with no entities are unaffected (no graph context to evaluate).
    fn graph_adjusted_threshold(&self, memory: &Memory, base_threshold: f32) -> f32 {
        use crate::constants::*;

        let graph = match &self.graph_memory {
            Some(g) => g,
            None => return base_threshold,
        };

        if memory.entity_refs.is_empty() {
            return base_threshold;
        }

        let graph_guard = graph.read();
        let mut l2_plus_count = 0usize;

        for entity_ref in &memory.entity_refs {
            if let Ok(edges) = graph_guard.get_entity_relationships(&entity_ref.entity_id) {
                for edge in &edges {
                    if matches!(
                        edge.tier,
                        crate::graph_memory::EdgeTier::L2Episodic
                            | crate::graph_memory::EdgeTier::L3Semantic
                    ) {
                        l2_plus_count += 1;
                    }
                }
            }
        }

        if l2_plus_count == 0 {
            // Memory has entities but no strong edges — penalize
            base_threshold * (1.0 + GRAPH_HEALTH_NO_EDGES_PENALTY as f32)
        } else {
            // Discount proportional to edge count, capped at saturation
            let ratio = (l2_plus_count as f64 / GRAPH_HEALTH_EDGE_SATURATION).min(1.0);
            base_threshold * (1.0 - (GRAPH_HEALTH_PROMOTION_DISCOUNT * ratio) as f32)
        }
    }

    /// Apply importance boosts to memories whose edges were promoted (Direction 1).
    ///
    /// When an edge promotes from L1→L2 or L2→L3, the memories involved get
    /// a small importance boost, reflecting that they participate in a consolidating
    /// relationship. Uses interior mutability — `set_importance` works through Arc.
    pub fn apply_edge_promotion_boosts(
        &self,
        boosts: &[crate::memory::types::EdgePromotionBoost],
    ) -> Result<usize> {
        let mut applied = 0;

        for boost in boosts {
            let memory_id = match uuid::Uuid::parse_str(&boost.memory_id) {
                Ok(uuid) => MemoryId(uuid),
                Err(_) => continue,
            };

            // Search across tiers: working → session → long-term
            let found = self
                .working_memory
                .read()
                .get(&memory_id)
                .or_else(|| self.session_memory.read().get(&memory_id));

            if let Some(memory) = found {
                let new_importance = (memory.importance() + boost.boost as f32).min(1.0);
                memory.set_importance(new_importance);
                self.record_consolidation_event(ConsolidationEvent::EdgePromotionBoostApplied {
                    memory_id: boost.memory_id.clone(),
                    entity_name: boost.entity_name.clone(),
                    old_tier: boost.old_tier.clone(),
                    new_tier: boost.new_tier.clone(),
                    importance_boost: boost.boost,
                    new_importance: new_importance as f64,
                    timestamp: chrono::Utc::now(),
                });
                applied += 1;
            } else if let Ok(memory) = self.long_term_memory.get(&memory_id) {
                let new_importance = (memory.importance() + boost.boost as f32).min(1.0);
                memory.set_importance(new_importance);
                if let Err(e) = self.long_term_memory.store(&memory) {
                    tracing::debug!(
                        "Failed to persist edge promotion boost for {}: {}",
                        boost.memory_id,
                        e
                    );
                    continue;
                }
                self.record_consolidation_event(ConsolidationEvent::EdgePromotionBoostApplied {
                    memory_id: boost.memory_id.clone(),
                    entity_name: boost.entity_name.clone(),
                    old_tier: boost.old_tier.clone(),
                    new_tier: boost.new_tier.clone(),
                    importance_boost: boost.boost,
                    new_importance: new_importance as f64,
                    timestamp: chrono::Utc::now(),
                });
                applied += 1;
            }
        }

        if applied > 0 {
            tracing::debug!(
                "Applied {} edge promotion boosts to memory importance",
                applied
            );
        }

        Ok(applied)
    }

    /// Apply compensatory boost to memories that lost all graph edges (Direction 2).
    ///
    /// When graph decay prunes edges and leaves entities orphaned, the memories
    /// referencing those entities get a small importance boost to prevent immediate
    /// decay death. This gives them one more maintenance cycle to prove value.
    pub fn compensate_orphaned_memories(&self, orphaned_entity_ids: &[String]) -> Result<usize> {
        use crate::constants::ORPHAN_COMPENSATORY_BOOST;

        if orphaned_entity_ids.is_empty() {
            return Ok(0);
        }

        let orphaned_set: std::collections::HashSet<&str> =
            orphaned_entity_ids.iter().map(|s| s.as_str()).collect();

        let mut compensated = 0;

        // Scan working + session memories for references to orphaned entities
        let tiers: Vec<Vec<SharedMemory>> = vec![
            self.working_memory.read().all_memories(),
            self.session_memory.read().all_memories(),
        ];

        for memories in &tiers {
            for memory in memories {
                let entity_count = memory
                    .entity_refs
                    .iter()
                    .filter(|e| orphaned_set.contains(e.entity_id.to_string().as_str()))
                    .count();
                if entity_count > 0 {
                    let new_importance =
                        (memory.importance() + ORPHAN_COMPENSATORY_BOOST as f32).min(1.0);
                    memory.set_importance(new_importance);
                    self.record_consolidation_event(ConsolidationEvent::GraphOrphanDetected {
                        memory_id: memory.id.0.to_string(),
                        entity_count,
                        compensatory_boost: ORPHAN_COMPENSATORY_BOOST,
                        timestamp: chrono::Utc::now(),
                    });
                    compensated += 1;
                }
            }
        }

        if compensated > 0 {
            tracing::debug!(
                "Compensated {} orphaned memories (from {} orphaned entities)",
                compensated,
                orphaned_entity_ids.len()
            );
        }

        Ok(compensated)
    }

    /// Compress old memories to save space
    fn compress_old_memories(&self) -> Result<()> {
        let cutoff =
            chrono::Utc::now() - chrono::Duration::days(self.config.compression_age_days as i64);

        // Get uncompressed old memories
        let to_compress = self.long_term_memory.get_uncompressed_older_than(cutoff)?;

        for memory in to_compress {
            let compressed = self.compressor.compress(&memory)?;
            self.long_term_memory.update(&compressed)?;
            self.stats.write().compressed_count += 1;
        }

        Ok(())
    }

    /// Check if a memory should be compressed
    fn should_compress(&self, memory: &Memory) -> bool {
        let age = chrono::Utc::now() - memory.created_at;
        age.num_days() > self.config.compression_age_days as i64 && !memory.compressed
    }

    /// Update access count for a memory (handles concurrency properly)
    /// Note: Prefer update_access_count_instrumented() for consolidation tracking
    #[allow(dead_code)]
    fn update_access_count(&self, memory_id: &MemoryId) -> Result<()> {
        // Try updating in working memory first (most common case)
        // Use write lock directly to avoid TOCTOU race condition
        {
            let mut wm = self.working_memory.write();

            if wm.contains(memory_id) {
                // Memory found in working memory - update and return
                return wm
                    .update_access(memory_id)
                    .map_err(|e| anyhow::anyhow!("Failed to update working memory access: {e}"));
            }
        } // Release write lock

        // Try session memory
        {
            let mut sm = self.session_memory.write();

            if sm.contains(memory_id) {
                return sm
                    .update_access(memory_id)
                    .map_err(|e| anyhow::anyhow!("Failed to update session memory access: {e}"));
            }
        } // Release write lock

        // Try long-term memory (has its own internal locking)
        self.long_term_memory
            .update_access(memory_id)
            .map_err(|e| anyhow::anyhow!("Failed to update long-term memory access: {e}"))
    }

    /// Update access count with instrumentation for consolidation events
    ///
    /// Records MemoryStrengthened events when memories are accessed during retrieval,
    /// capturing activation changes for introspection.
    fn update_access_count_instrumented(&self, memory: &SharedMemory, reason: StrengtheningReason) {
        // Capture activation before update
        let activation_before = memory.importance();

        // Perform the actual access update
        memory.update_access();

        // Capture activation after update
        let activation_after = memory.importance();

        // Only record event if activation actually changed
        if (activation_after - activation_before).abs() > f32::EPSILON {
            let content_preview = if memory.experience.content.chars().count() > 50 {
                let truncated: String = memory.experience.content.chars().take(50).collect();
                format!("{}...", truncated)
            } else {
                memory.experience.content.clone()
            };

            let event = ConsolidationEvent::MemoryStrengthened {
                memory_id: memory.id.0.to_string(),
                content_preview,
                activation_before,
                activation_after,
                reason,
                timestamp: chrono::Utc::now(),
            };

            self.consolidation_events.write().push(event);
        }
    }

    /// Clean up graph episodes for a batch of deleted memory IDs (best-effort)
    fn cleanup_graph_for_ids(&self, ids: &[MemoryId]) {
        if ids.is_empty() {
            return;
        }
        if let Some(graph) = &self.graph_memory {
            let graph_guard = graph.read();
            for id in ids {
                if let Err(e) = graph_guard.delete_episode(&id.0) {
                    tracing::debug!("Graph cleanup failed for {}: {}", &id.0.to_string()[..8], e);
                }
            }
        }
    }

    /// Clean up interference records for a batch of deleted memory IDs (best-effort)
    fn cleanup_interference_for_ids(&self, ids: &[MemoryId]) {
        if ids.is_empty() {
            return;
        }
        let mut detector = self.interference_detector.write();
        for id in ids {
            let id_str = id.0.to_string();
            detector.clear_memory(&id_str);
            if let Err(e) = self.long_term_memory.delete_interference_records(&id_str) {
                tracing::debug!(
                    "Interference cleanup failed for {}: {e}",
                    &id_str[..8.min(id_str.len())]
                );
            }
        }
    }

    /// Forget memories matching a pattern
    ///
    /// Uses validated regex compilation with ReDoS protection
    fn forget_by_pattern(&self, pattern: &str) -> Result<usize> {
        // Use validated pattern compilation with ReDoS protection
        let regex = crate::validation::validate_and_compile_pattern(pattern)
            .map_err(|e| anyhow::anyhow!("Invalid pattern: {e}"))?;
        let mut count = 0;
        let mut working_removed = 0;
        let mut session_removed = 0;
        let mut long_term_removed = 0;

        // Collect IDs from working memory that match
        let working_ids: Vec<MemoryId> = {
            let working = self.working_memory.read();
            working
                .all_memories()
                .iter()
                .filter(|m| regex.is_match(&m.experience.content))
                .map(|m| m.id.clone())
                .collect()
        };
        // Remove from working memory and vector/BM25 index
        {
            let mut working = self.working_memory.write();
            for id in &working_ids {
                if working.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                    working_removed += 1;
                    count += 1;
                }
            }
        }

        // Collect IDs from session memory that match
        let session_ids: Vec<MemoryId> = {
            let session = self.session_memory.read();
            session
                .all_memories()
                .iter()
                .filter(|m| regex.is_match(&m.experience.content))
                .map(|m| m.id.clone())
                .collect()
        };
        // Remove from session memory and vector/BM25 index
        {
            let mut session = self.session_memory.write();
            for id in &session_ids {
                if session.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                    session_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from long-term memory
        let all_lt = self.long_term_memory.get_all()?;
        let mut lt_ids = Vec::new();
        for memory in all_lt {
            if regex.is_match(&memory.experience.content) {
                lt_ids.push(memory.id.clone());
                self.retriever.remove_memory(&memory.id);
                let _ = self.hybrid_search.remove_memory(&memory.id);
                self.long_term_memory.delete(&memory.id)?;
                long_term_removed += 1;
                count += 1;
            }
        }

        // Clean up graph episodes and interference records for all deleted memories
        let all_ids: Vec<MemoryId> = working_ids
            .into_iter()
            .chain(session_ids)
            .chain(lt_ids)
            .collect();
        self.cleanup_graph_for_ids(&all_ids);
        self.cleanup_interference_for_ids(&all_ids);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories = stats.total_memories.saturating_sub(count);
            stats.working_memory_count = stats.working_memory_count.saturating_sub(working_removed);
            stats.session_memory_count = stats.session_memory_count.saturating_sub(session_removed);
            stats.long_term_memory_count = stats
                .long_term_memory_count
                .saturating_sub(long_term_removed);
            stats.vector_index_count = stats.vector_index_count.saturating_sub(count);
        }

        Ok(count)
    }

    /// Forget memories matching ANY of the specified tags
    pub fn forget_by_tags(&self, tags: &[String]) -> Result<usize> {
        let mut count = 0;
        let mut working_removed = 0;
        let mut session_removed = 0;
        let mut long_term_removed = 0;
        let mut all_deleted_ids = Vec::new();

        // Remove from working memory
        {
            let mut working = self.working_memory.write();
            let ids_to_remove: Vec<MemoryId> = working
                .all_memories()
                .iter()
                .filter(|m| m.experience.tags.iter().any(|t| tags.contains(t)))
                .map(|m| m.id.clone())
                .collect();
            for id in &ids_to_remove {
                if working.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                    working_removed += 1;
                    count += 1;
                }
            }
            all_deleted_ids.extend(ids_to_remove);
        }

        // Remove from session memory
        {
            let mut session = self.session_memory.write();
            let ids_to_remove: Vec<MemoryId> = session
                .all_memories()
                .iter()
                .filter(|m| m.experience.tags.iter().any(|t| tags.contains(t)))
                .map(|m| m.id.clone())
                .collect();
            for id in &ids_to_remove {
                if session.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                    session_removed += 1;
                    count += 1;
                }
            }
            all_deleted_ids.extend(ids_to_remove);
        }

        // Remove from long-term memory (hard delete for tag-based)
        let all_lt = self.long_term_memory.get_all()?;
        for memory in all_lt {
            if memory.experience.tags.iter().any(|t| tags.contains(t)) {
                all_deleted_ids.push(memory.id.clone());
                self.retriever.remove_memory(&memory.id);
                let _ = self.hybrid_search.remove_memory(&memory.id);
                self.long_term_memory.delete(&memory.id)?;
                long_term_removed += 1;
                count += 1;
            }
        }

        // Clean up graph episodes and interference records for all deleted memories
        self.cleanup_graph_for_ids(&all_deleted_ids);
        self.cleanup_interference_for_ids(&all_deleted_ids);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories = stats.total_memories.saturating_sub(count);
            stats.working_memory_count = stats.working_memory_count.saturating_sub(working_removed);
            stats.session_memory_count = stats.session_memory_count.saturating_sub(session_removed);
            stats.long_term_memory_count = stats
                .long_term_memory_count
                .saturating_sub(long_term_removed);
            stats.vector_index_count = stats.vector_index_count.saturating_sub(count);
        }

        Ok(count)
    }

    /// Forget memories within a date range (inclusive)
    fn forget_by_date_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        let mut count = 0;
        let mut working_removed = 0;
        let mut session_removed = 0;
        let mut long_term_removed = 0;

        // Collect IDs from working memory that match date range
        let working_ids: Vec<MemoryId> = {
            let working = self.working_memory.read();
            working
                .all_memories()
                .iter()
                .filter(|m| m.created_at >= start && m.created_at <= end)
                .map(|m| m.id.clone())
                .collect()
        };
        // Remove from working memory and vector/BM25 index
        {
            let mut working = self.working_memory.write();
            for id in &working_ids {
                if working.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                    working_removed += 1;
                    count += 1;
                }
            }
        }

        // Collect IDs from session memory that match date range
        let session_ids: Vec<MemoryId> = {
            let session = self.session_memory.read();
            session
                .all_memories()
                .iter()
                .filter(|m| m.created_at >= start && m.created_at <= end)
                .map(|m| m.id.clone())
                .collect()
        };
        // Remove from session memory and vector/BM25 index
        {
            let mut session = self.session_memory.write();
            for id in &session_ids {
                if session.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                    session_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from long-term memory using storage search
        let memories = self
            .long_term_memory
            .search(storage::SearchCriteria::ByDate { start, end })?;
        let mut lt_ids = Vec::new();
        for memory in memories {
            lt_ids.push(memory.id.clone());
            self.retriever.remove_memory(&memory.id);
            let _ = self.hybrid_search.remove_memory(&memory.id);
            self.long_term_memory.delete(&memory.id)?;
            long_term_removed += 1;
            count += 1;
        }

        // Clean up graph episodes and interference records for all deleted memories
        let all_ids: Vec<MemoryId> = working_ids
            .into_iter()
            .chain(session_ids)
            .chain(lt_ids)
            .collect();
        self.cleanup_graph_for_ids(&all_ids);
        self.cleanup_interference_for_ids(&all_ids);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories = stats.total_memories.saturating_sub(count);
            stats.working_memory_count = stats.working_memory_count.saturating_sub(working_removed);
            stats.session_memory_count = stats.session_memory_count.saturating_sub(session_removed);
            stats.long_term_memory_count = stats
                .long_term_memory_count
                .saturating_sub(long_term_removed);
            stats.vector_index_count = stats.vector_index_count.saturating_sub(count);
        }

        Ok(count)
    }

    /// Forget memories of a specific type
    fn forget_by_type(&self, exp_type: ExperienceType) -> Result<usize> {
        let mut count = 0;
        let mut working_removed = 0;
        let mut session_removed = 0;
        let mut long_term_removed = 0;

        // Collect IDs from working memory that match type
        let working_ids: Vec<MemoryId> = {
            let working = self.working_memory.read();
            working
                .all_memories()
                .iter()
                .filter(|m| m.experience.experience_type == exp_type)
                .map(|m| m.id.clone())
                .collect()
        };
        // Remove from working memory and vector/BM25 index
        {
            let mut working = self.working_memory.write();
            for id in &working_ids {
                if working.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                    working_removed += 1;
                    count += 1;
                }
            }
        }

        // Collect IDs from session memory that match type
        let session_ids: Vec<MemoryId> = {
            let session = self.session_memory.read();
            session
                .all_memories()
                .iter()
                .filter(|m| m.experience.experience_type == exp_type)
                .map(|m| m.id.clone())
                .collect()
        };
        // Remove from session memory and vector/BM25 index
        {
            let mut session = self.session_memory.write();
            for id in &session_ids {
                if session.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    let _ = self.hybrid_search.remove_memory(id);
                    session_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from long-term memory using storage search
        let memories = self
            .long_term_memory
            .search(storage::SearchCriteria::ByType(exp_type))?;
        let mut lt_ids = Vec::new();
        for memory in memories {
            lt_ids.push(memory.id.clone());
            self.retriever.remove_memory(&memory.id);
            let _ = self.hybrid_search.remove_memory(&memory.id);
            self.long_term_memory.delete(&memory.id)?;
            long_term_removed += 1;
            count += 1;
        }

        // Clean up graph episodes and interference records for all deleted memories
        let all_ids: Vec<MemoryId> = working_ids
            .into_iter()
            .chain(session_ids)
            .chain(lt_ids)
            .collect();
        self.cleanup_graph_for_ids(&all_ids);
        self.cleanup_interference_for_ids(&all_ids);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories = stats.total_memories.saturating_sub(count);
            stats.working_memory_count = stats.working_memory_count.saturating_sub(working_removed);
            stats.session_memory_count = stats.session_memory_count.saturating_sub(session_removed);
            stats.long_term_memory_count = stats
                .long_term_memory_count
                .saturating_sub(long_term_removed);
            stats.vector_index_count = stats.vector_index_count.saturating_sub(count);
        }

        Ok(count)
    }

    /// Forget ALL memories for a user (GDPR compliance - right to erasure)
    ///
    /// WARNING: This is a destructive operation. All memories across all tiers
    /// will be permanently deleted. This cannot be undone.
    fn forget_all(&self) -> Result<usize> {
        // Deletion order: graph → long-term → session → working → stats
        // This is fail-safe: if we crash mid-way, the most durable data
        // (graph/long-term) is already deleted. Working/session memory is
        // ephemeral and will be empty on restart anyway.

        let mut count = 0;

        // Step 1: Clear knowledge graph first (GDPR - complete erasure)
        // Graph references memories, so clean references before deleting memories
        if let Some(graph) = &self.graph_memory {
            match graph.read().clear_all() {
                Ok((entities, relationships, episodes)) => {
                    tracing::info!(
                        entities,
                        relationships,
                        episodes,
                        "Graph cleared during forget_all"
                    );
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to clear knowledge graph during forget_all");
                    // Continue — graph cleanup is best-effort for GDPR
                }
            }
        }

        // Step 2: Clear long-term memory (persistent, most important to delete)
        let all_lt = self.long_term_memory.get_all()?;
        let long_term_count = all_lt.len();
        for memory in all_lt {
            self.retriever.remove_memory(&memory.id);
            let _ = self.hybrid_search.remove_memory(&memory.id);
            self.long_term_memory.delete(&memory.id)?;
        }
        count += long_term_count;

        // Step 3: Clear session memory (ephemeral — lost on restart anyway)
        let session_ids: Vec<MemoryId> = {
            let session = self.session_memory.read();
            session
                .all_memories()
                .iter()
                .map(|m| m.id.clone())
                .collect()
        };
        let session_count = session_ids.len();
        for id in &session_ids {
            self.retriever.remove_memory(id);
            let _ = self.hybrid_search.remove_memory(id);
        }
        {
            let mut session = self.session_memory.write();
            session.clear();
        }
        count += session_count;

        // Step 4: Clear working memory (ephemeral — lost on restart anyway)
        let working_ids: Vec<MemoryId> = {
            let working = self.working_memory.read();
            working
                .all_memories()
                .iter()
                .map(|m| m.id.clone())
                .collect()
        };
        let working_count = working_ids.len();
        for id in &working_ids {
            self.retriever.remove_memory(id);
            let _ = self.hybrid_search.remove_memory(id);
        }
        {
            let mut working = self.working_memory.write();
            working.clear();
        }
        count += working_count;

        // Step 5: Commit BM25 deletions
        if let Err(e) = self.hybrid_search.commit_and_reload() {
            tracing::warn!(error = %e, "BM25 commit failed during forget_all");
        }

        // Step 6: Clear semantic facts (GDPR — knowledge derived from memories)
        {
            let db = self.long_term_memory.db();
            let mut batch = rocksdb::WriteBatch::default();
            let mut facts_deleted = 0usize;
            for prefix in &[
                "facts:",
                "facts_by_entity:",
                "facts_by_type:",
                "facts_embedding:",
            ] {
                let iter = db.prefix_iterator(prefix.as_bytes());
                for (key, _) in iter.flatten() {
                    if !key.starts_with(prefix.as_bytes()) {
                        break;
                    }
                    batch.delete(&key);
                    if *prefix == "facts:" {
                        facts_deleted += 1;
                    }
                }
            }
            // Clear temporal facts
            let iter = db.prefix_iterator(b"temporal_facts:");
            for (key, _) in iter.flatten() {
                if !key.starts_with(b"temporal_facts:") {
                    break;
                }
                batch.delete(&key);
            }
            if facts_deleted > 0 || !batch.is_empty() {
                if let Err(e) = db.write(batch) {
                    tracing::warn!(error = %e, "Failed to clear facts during forget_all");
                } else {
                    tracing::info!(facts_deleted, "Semantic facts cleared during forget_all");
                }
            }
        }

        // Step 7: Clear interference history (in-memory + persisted)
        {
            let mut detector = self.interference_detector.write();
            *detector = replay::InterferenceDetector::new();
        }
        if let Err(e) = self.long_term_memory.clear_all_interference_records() {
            tracing::warn!(error = %e, "Failed to clear interference records during forget_all");
        }

        // Step 8: Reset stats last (reflects final state)
        {
            let mut stats = self.stats.write();
            stats.total_memories = 0;
            stats.working_memory_count = 0;
            stats.session_memory_count = 0;
            stats.long_term_memory_count = 0;
            stats.vector_index_count = 0;
        }

        Ok(count)
    }

    /// Show memory visualization (ASCII art of memory graph)
    pub fn show_visualization(&self) {
        self.logger.read().show_visualization();
    }

    /// Export memory graph as DOT file for Graphviz
    pub fn export_graph(&self, path: &std::path::Path) -> Result<()> {
        self.logger.read().export_dot(path)
    }

    /// Get visualization statistics
    pub fn get_visualization_stats(&self) -> GraphStats {
        self.logger.read().get_stats()
    }

    /// Flush long-term storage to ensure data persistence (critical for graceful shutdown)
    pub fn flush_storage(&self) -> Result<()> {
        // Flush RocksDB storage
        self.long_term_memory.flush()?;

        // Persist vector index and ID mapping for restart recovery
        self.retriever.save()?;

        Ok(())
    }

    /// Get the underlying RocksDB database handle for backup operations
    ///
    /// # Warning
    /// This provides direct access to the database. Use with caution.
    /// Primarily intended for backup/restore operations.
    pub fn get_db(&self) -> std::sync::Arc<rocksdb::DB> {
        self.long_term_memory.db()
    }

    /// Advanced search using storage criteria
    pub fn advanced_search(&self, criteria: storage::SearchCriteria) -> Result<Vec<Memory>> {
        self.long_term_memory.search(criteria)
    }

    /// Get memory by ID from long-term storage
    pub fn get_memory(&self, id: &MemoryId) -> Result<Memory> {
        self.long_term_memory.get(id)
    }

    /// Get all memory IDs from long-term storage.
    ///
    /// Used by dream replay and other bulk operations that need to sample
    /// random memories without loading their full content.
    pub fn get_long_term_ids(&self) -> Result<Vec<MemoryId>> {
        self.long_term_memory.get_all_ids()
    }

    /// Update a memory in storage with full re-indexing
    ///
    /// This properly updates the memory by:
    /// 1. Removing stale secondary indices and re-storing in RocksDB
    /// 2. Re-indexing in vector DB (semantic search) if embeddings changed
    /// 3. Re-indexing in BM25 (keyword/hybrid search)
    /// 4. Updating working/session memory caches if the memory is cached
    pub fn update_memory(&self, memory: &Memory) -> Result<()> {
        let memory_id = memory.id.clone();

        // Update in storage (removes old indices, re-stores with fresh indices)
        self.long_term_memory.update(memory)?;

        // Re-index in vector DB with updated embeddings
        if let Err(e) = self.retriever.reindex_memory(memory) {
            tracing::warn!(
                "Failed to reindex memory {} in vector DB: {}",
                memory_id.0,
                e
            );
        }

        // Re-index in BM25 with updated content
        if let Err(e) = self.hybrid_search.index_memory(
            &memory_id,
            &memory.experience.content,
            &memory.experience.tags,
            &memory.experience.entities,
        ) {
            tracing::warn!("Failed to reindex memory {} in BM25: {}", memory_id.0, e);
        }
        if let Err(e) = self.hybrid_search.commit_and_reload() {
            tracing::warn!("Failed to commit/reload BM25 index: {}", e);
        }

        // Update in working/session memory caches if present
        {
            let mut working = self.working_memory.write();
            if working.contains(&memory_id) {
                let _ = working.remove(&memory_id);
                let _ = working.add_shared(std::sync::Arc::new(memory.clone()));
            }
        }
        {
            let mut session = self.session_memory.write();
            if session.contains(&memory_id) {
                let _ = session.remove(&memory_id);
                let _ = session.add_shared(std::sync::Arc::new(memory.clone()));
            }
        }

        Ok(())
    }

    /// Set or update the parent of a memory for hierarchical organization
    ///
    /// This enables memory trees where memories can have parent-child relationships.
    /// Example: "71-research" -> "algebraic" -> "21×27≡-1"
    ///
    /// Pass `None` as parent_id to remove the parent (make it a root memory).
    pub fn set_memory_parent(
        &self,
        memory_id: &MemoryId,
        parent_id: Option<MemoryId>,
    ) -> Result<()> {
        // Update the persistent copy in long-term storage
        let mut memory = self.long_term_memory.get(memory_id)?;
        memory.set_parent(parent_id.clone());
        self.long_term_memory.update(&memory)?;

        // Also update the in-memory tier copy (working or session) so reads
        // reflect the parent_id immediately without waiting for tier promotion
        let updated = Arc::new(memory);
        {
            let mut wm = self.working_memory.write();
            if wm.contains(memory_id) {
                let _ = wm.remove(memory_id);
                let _ = wm.add_shared(Arc::clone(&updated));
            }
        }
        {
            let mut sm = self.session_memory.write();
            if sm.contains(memory_id) {
                let _ = sm.remove(memory_id);
                let _ = sm.add_shared(Arc::clone(&updated));
            }
        }

        Ok(())
    }

    /// Get children of a memory
    pub fn get_memory_children(&self, parent_id: &MemoryId) -> Result<Vec<Memory>> {
        self.long_term_memory.get_children(parent_id)
    }

    /// Get ancestors (parent chain) of a memory
    pub fn get_memory_ancestors(&self, memory_id: &MemoryId) -> Result<Vec<Memory>> {
        self.long_term_memory.get_ancestors(memory_id)
    }

    /// Get full hierarchy context (ancestors, memory, children)
    pub fn get_memory_hierarchy(
        &self,
        memory_id: &MemoryId,
    ) -> Result<(Vec<Memory>, Memory, Vec<Memory>)> {
        self.long_term_memory.get_hierarchy_context(memory_id)
    }

    /// Decompress a memory
    pub fn decompress_memory(&self, memory: &Memory) -> Result<Memory> {
        self.compressor.decompress(memory)
    }

    /// Get storage statistics
    pub fn get_storage_stats(&self) -> Result<storage::StorageStats> {
        self.long_term_memory.get_stats()
    }

    /// Get uncompressed old memories
    pub fn get_uncompressed_older_than(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<Memory>> {
        self.long_term_memory.get_uncompressed_older_than(cutoff)
    }

    /// Rebuild vector index from all existing long-term memories (startup initialization)
    pub fn rebuild_vector_index(&self) -> Result<()> {
        self.retriever.rebuild_index()
    }

    /// Repair vector index by finding and re-indexing orphaned memories
    ///
    /// Orphaned memories are those stored in RocksDB but missing from the vector index.
    /// This can happen if embedding generation fails during record().
    ///
    /// Returns: (total_storage, indexed, repaired, failed)
    pub fn repair_vector_index(&self) -> Result<(usize, usize, usize, usize)> {
        let all_memories = self.long_term_memory.get_all()?;
        let total_storage = all_memories.len();
        let indexed_before = self.retriever.len();

        let mut repaired = 0;
        let mut failed = 0;

        // Get set of indexed memory IDs
        let indexed_ids = self.retriever.get_indexed_memory_ids();

        for memory in all_memories {
            // Check if memory is already indexed
            if indexed_ids.contains(&memory.id) {
                continue;
            }

            // Memory is orphaned - try to index it
            tracing::info!(
                memory_id = %memory.id.0,
                content_preview = %memory.experience.content.chars().take(50).collect::<String>(),
                "Repairing orphaned memory"
            );

            match self.retriever.index_memory(&memory) {
                Ok(_) => {
                    repaired += 1;
                    tracing::info!(memory_id = %memory.id.0, "Successfully repaired orphaned memory");
                }
                Err(e) => {
                    failed += 1;
                    tracing::error!(
                        memory_id = %memory.id.0,
                        error = %e,
                        "Failed to repair orphaned memory - embedding generation failed"
                    );
                }
            }
        }

        let indexed_after = self.retriever.len();

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.vector_index_count = indexed_after;
        }

        tracing::info!(
            total_storage = total_storage,
            indexed_before = indexed_before,
            indexed_after = indexed_after,
            repaired = repaired,
            failed = failed,
            "Vector index repair completed"
        );

        Ok((total_storage, indexed_after, repaired, failed))
    }

    /// Verify index integrity and return diagnostic information
    ///
    /// Returns a struct with:
    /// - total_storage: memories in RocksDB
    /// - total_indexed: memories in vector index
    /// - orphaned_count: memories missing from index
    /// - orphaned_ids: list of orphaned memory IDs (first 100)
    pub fn verify_index_integrity(&self) -> Result<IndexIntegrityReport> {
        let all_memories = self.long_term_memory.get_all()?;
        let total_storage = all_memories.len();
        let indexed_ids = self.retriever.get_indexed_memory_ids();
        let total_indexed = indexed_ids.len();

        let mut orphaned_ids = Vec::new();
        for memory in &all_memories {
            if !indexed_ids.contains(&memory.id)
                && orphaned_ids.len() < 100 {
                    orphaned_ids.push(memory.id.clone());
                }
        }

        let orphaned_count = total_storage.saturating_sub(total_indexed);

        let is_healthy = orphaned_count == 0;
        Ok(IndexIntegrityReport {
            total_storage,
            total_indexed,
            orphaned_count,
            orphaned_ids,
            is_healthy,
            healthy: is_healthy,
        })
    }

    /// Cleanup corrupted memories that fail to deserialize
    /// Returns the number of entries deleted
    pub fn cleanup_corrupted(&self) -> Result<usize> {
        self.long_term_memory.cleanup_corrupted()
    }

    /// Migrate legacy memories to current format for improved performance
    /// Returns (migrated_count, already_current_count, failed_count)
    pub fn migrate_legacy(&self) -> Result<(usize, usize, usize)> {
        self.long_term_memory.migrate_legacy()
    }

    /// Rebuild vector index from scratch using only valid memories in storage
    /// This removes orphaned index entries and rebuilds with proper ID mappings
    /// Returns (total_memories, total_indexed)
    pub fn rebuild_index(&self) -> Result<(usize, usize)> {
        tracing::info!("Starting full index rebuild from storage");
        self.retriever.rebuild_index()?;
        let indexed = self.retriever.len();
        let storage_count = self.long_term_memory.get_stats()?.total_count;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.vector_index_count = indexed;
        }

        tracing::info!(
            storage_count = storage_count,
            indexed = indexed,
            "Index rebuild complete"
        );

        Ok((storage_count, indexed))
    }

    /// Save vector index to disk (shutdown persistence)
    /// Uses Vamana persistence format for instant startup on restart
    pub fn save_vector_index(&self, _path: &Path) -> Result<()> {
        self.retriever.save()
    }
    /// Get vector index health information
    ///
    /// Returns metrics about the Vamana index including total vectors,
    /// incremental inserts since last build, and whether rebuild is recommended.
    pub fn index_health(&self) -> retrieval::IndexHealth {
        self.retriever.index_health()
    }

    /// Auto-rebuild vector index if degradation threshold is exceeded
    ///
    /// Returns `Ok(true)` if rebuild was performed, `Ok(false)` if not needed.
    /// Thread-safe: concurrent calls are no-ops while rebuild is in progress.
    pub fn auto_rebuild_index_if_needed(&self) -> Result<bool> {
        self.retriever.auto_rebuild_index_if_needed()
    }

    /// Auto-repair index integrity and compact if needed
    ///
    /// Called during maintenance to ensure storage↔index consistency:
    /// 1. Checks index health (fast O(1) operation)
    /// 2. If needs compaction (>30% deleted), triggers auto-rebuild
    /// 3. If orphaned memories detected, repairs them
    ///
    /// This provides eventual consistency between storage and index.
    fn auto_repair_and_compact(&self) {
        // Check index health first (fast operation)
        let health = self.index_health();

        // Auto-compact if deletion ratio exceeds threshold
        if health.needs_compaction {
            tracing::info!(
                "Index compaction triggered: {:.1}% deleted ({} of {} vectors)",
                health.deletion_ratio * 100.0,
                health.deleted_count,
                health.total_vectors
            );
            if let Err(e) = self.auto_rebuild_index_if_needed() {
                tracing::warn!("Index compaction failed: {}", e);
            }
        }

        // Check for orphaned memories (stored but not indexed)
        // Only do full scan if we suspect issues (cheap heuristic: counts differ)
        let storage_count = self
            .long_term_memory
            .get_stats()
            .map(|s| s.total_count)
            .unwrap_or(0);
        let index_count = health.total_vectors.saturating_sub(health.deleted_count);

        if storage_count > index_count {
            // Potential orphans detected - run repair
            let orphan_estimate = storage_count - index_count;
            if orphan_estimate > 0 {
                tracing::info!(
                    "Potential orphaned memories detected: ~{} (storage={}, indexed={})",
                    orphan_estimate,
                    storage_count,
                    index_count
                );
                match self.repair_vector_index() {
                    Ok((_, _, repaired, failed)) => {
                        if repaired > 0 || failed > 0 {
                            tracing::info!(
                                "Index repair complete: {} repaired, {} failed",
                                repaired,
                                failed
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Index repair failed: {}", e);
                    }
                }
            }
        }
    }

    /// Perform graph maintenance (decay old edges, prune weak ones)
    ///
    /// Call this periodically (e.g., every hour or on user logout)
    /// to let unused associations naturally fade.
    pub fn graph_maintenance(&self) {
        if let Some(graph) = &self.graph_memory {
            if let Err(e) = graph.read().apply_decay() {
                tracing::debug!("Graph decay maintenance failed: {e}");
            }
        }
    }

    /// Connect extracted facts to the knowledge graph.
    ///
    /// For each fact, ensures all related entities exist as EntityNodes and creates
    /// RelationshipEdges between all pairs. Uses L2Episodic tier because facts are
    /// consolidated knowledge (survived 7-day aging + 2+ supporting memories).
    /// `add_entity` upserts (increments mention_count if existing), and
    /// `add_relationship` strengthens via Hebbian learning if the edge already exists.
    fn connect_facts_to_graph(&self, facts: &[SemanticFact]) {
        let graph = match &self.graph_memory {
            Some(g) => g,
            None => return,
        };
        let graph_guard = graph.read();
        let now = chrono::Utc::now();
        let mut entities_added = 0;
        let mut edges_added = 0;

        // Collect all unique entity names across all facts for batch encoding
        let mut all_entity_names: Vec<String> = Vec::new();
        for fact in facts {
            for name in &fact.related_entities {
                if !all_entity_names.contains(name) {
                    all_entity_names.push(name.clone());
                }
            }
        }

        // Batch-encode entity names for concept-level dedup
        let embedding_map: std::collections::HashMap<String, Vec<f32>> =
            if all_entity_names.is_empty() {
                std::collections::HashMap::new()
            } else {
                let name_refs: Vec<&str> = all_entity_names.iter().map(|s| s.as_str()).collect();
                match self.embedder.encode_batch(&name_refs) {
                    Ok(embs) => all_entity_names.into_iter().zip(embs).collect(),
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            "Fact entity name embedding failed, concept merge disabled"
                        );
                        std::collections::HashMap::new()
                    }
                }
            };

        for fact in facts {
            // Ensure all related entities exist as graph nodes
            for entity_name in &fact.related_entities {
                let entity = crate::graph_memory::EntityNode {
                    uuid: Uuid::new_v4(),
                    name: entity_name.clone(),
                    labels: vec![crate::graph_memory::EntityLabel::Concept],
                    created_at: now,
                    last_seen_at: now,
                    mention_count: 1,
                    summary: String::new(),
                    attributes: std::collections::HashMap::new(),
                    name_embedding: embedding_map.get(entity_name).cloned(),
                    salience: fact.confidence * 0.5,
                    is_proper_noun: entity_name
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false),
                };
                if graph_guard.add_entity(entity).is_ok() {
                    entities_added += 1;
                }
            }

            // Create edges between all pairs of related entities with semantic weighting
            let entities = &fact.related_entities;
            let l2_base_weight = crate::graph_memory::EdgeTier::L2Episodic.initial_weight();
            for i in 0..entities.len() {
                for j in (i + 1)..entities.len() {
                    if let (Ok(Some(e1)), Ok(Some(e2))) = (
                        graph_guard.find_entity_by_name(&entities[i]),
                        graph_guard.find_entity_by_name(&entities[j]),
                    ) {
                        let semantic_weight = match (&e1.name_embedding, &e2.name_embedding) {
                            (Some(emb1), Some(emb2)) => {
                                let sim = crate::similarity::cosine_similarity(emb1, emb2).max(0.0);
                                EDGE_SEMANTIC_WEIGHT_FLOOR
                                    + (1.0 - EDGE_SEMANTIC_WEIGHT_FLOOR) * sim
                            }
                            _ => 1.0,
                        };

                        let edge = crate::graph_memory::RelationshipEdge {
                            uuid: Uuid::new_v4(),
                            from_entity: e1.uuid,
                            to_entity: e2.uuid,
                            relation_type: crate::graph_memory::RelationType::RelatedTo,
                            strength: l2_base_weight * semantic_weight,
                            created_at: now,
                            valid_at: now,
                            invalidated_at: None,
                            source_episode_id: None,
                            context: fact.fact.chars().take(100).collect(),
                            last_activated: now,
                            activation_count: 1,
                            ltp_status: crate::graph_memory::LtpStatus::None,
                            tier: crate::graph_memory::EdgeTier::L2Episodic,
                            activation_timestamps: None,
                            entity_confidence: Some(fact.confidence),
                            created_by: crate::graph_memory::EdgeSource::Coactivation,
                        };
                        if graph_guard.add_relationship(edge).is_ok() {
                            edges_added += 1;
                        }
                    }
                }
            }
        }

        if entities_added > 0 || edges_added > 0 {
            tracing::debug!(
                entities_added,
                edges_added,
                facts = facts.len(),
                "Connected facts to knowledge graph"
            );
        }
    }

    /// Get memory graph statistics
    pub fn graph_stats(&self) -> MemoryGraphStats {
        if let Some(graph) = &self.graph_memory {
            let g = graph.read();
            if let Ok(stats) = g.get_stats() {
                // Calculate avg_strength and potentiated_count from relationships
                let (avg_strength, potentiated_count) = if let Ok(relationships) =
                    g.get_all_relationships()
                {
                    if relationships.is_empty() {
                        (0.0, 0)
                    } else {
                        let total_strength: f32 = relationships.iter().map(|r| r.strength).sum();
                        let potentiated =
                            relationships.iter().filter(|r| r.is_potentiated()).count();
                        (total_strength / relationships.len() as f32, potentiated)
                    }
                } else {
                    (0.0, 0)
                };

                return MemoryGraphStats {
                    node_count: stats.entity_count,
                    edge_count: stats.relationship_count,
                    avg_strength,
                    potentiated_count,
                };
            }
        }
        // Return empty stats if no graph
        MemoryGraphStats {
            node_count: 0,
            edge_count: 0,
            avg_strength: 0.0,
            potentiated_count: 0,
        }
    }

    // =========================================================================
    // UPSERT: Mutable memories with external linking and audit history
    // =========================================================================

    /// Upsert a memory: create if new, update with history tracking if exists
    ///
    /// When a memory with the same external_id exists:
    /// 1. Old content is pushed to history (audit trail)
    /// 2. Content is updated with new content
    /// 3. Version is incremented
    /// 4. Embeddings are regenerated for new content
    /// 5. Vector index is updated
    ///
    /// # Arguments
    /// * `external_id` - External system identifier (e.g., "linear:SHO-39", "github:pr-123")
    /// * `experience` - The experience data to store
    /// * `change_type` - Type of change (ContentUpdated, StatusChanged, etc.)
    /// * `changed_by` - Optional: who/what triggered the change
    /// * `change_reason` - Optional: description of why this changed
    ///
    /// # Returns
    /// * `(MemoryId, bool)` - Memory ID and whether it was an update (true) or create (false)
    pub fn upsert(
        &self,
        external_id: String,
        mut experience: Experience,
        change_type: ChangeType,
        changed_by: Option<String>,
        change_reason: Option<String>,
    ) -> Result<(MemoryId, bool)> {
        // Try to find existing memory with this external_id
        if let Some(mut existing) = self.long_term_memory.find_by_external_id(&external_id)? {
            // === UPDATE PATH ===
            let memory_id = existing.id.clone();

            // Push old content to history and update
            existing.update_content(
                experience.content.clone(),
                change_type,
                changed_by,
                change_reason,
            );

            // Update entities if provided
            if !experience.entities.is_empty() {
                existing.experience.entities = experience.entities;
            }

            // Update tags if provided
            if !experience.tags.is_empty() {
                existing.experience.tags = experience.tags;
            }

            // Regenerate embeddings for new content
            let content_hash = Self::sha256_hash(&existing.experience.content);
            if let Some(cached_embedding) = self.content_cache.get(&content_hash) {
                existing.experience.embeddings = Some(cached_embedding.clone());
            } else {
                match self.embedder.encode(&existing.experience.content) {
                    Ok(embedding) => {
                        self.content_cache.insert(content_hash, embedding.clone());
                        existing.experience.embeddings = Some(embedding);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to regenerate embedding during upsert: {}", e);
                    }
                }
            }

            // TEMPORAL EXTRACTION: Re-extract dates when content changes
            let temporal =
                crate::memory::query_parser::extract_temporal_refs(&existing.experience.content);
            existing.experience.temporal_refs.clear();
            for temp_ref in temporal.refs {
                existing
                    .experience
                    .temporal_refs
                    .push(temp_ref.date.to_string());
            }

            // Persist updated memory
            self.long_term_memory.update(&existing)?;

            // Re-index in vector DB with new embeddings
            if let Err(e) = self.retriever.reindex_memory(&existing) {
                tracing::warn!(
                    "Failed to reindex memory {} in vector DB: {}",
                    memory_id.0,
                    e
                );
            }

            // Re-index in BM25 with updated content
            if let Err(e) = self.hybrid_search.index_memory(
                &memory_id,
                &existing.experience.content,
                &existing.experience.tags,
                &existing.experience.entities,
            ) {
                tracing::warn!("Failed to reindex memory {} in BM25: {}", memory_id.0, e);
            }
            if let Err(e) = self.hybrid_search.commit_and_reload() {
                tracing::warn!("Failed to commit/reload BM25 index: {}", e);
            }

            // Update in working/session memory if cached
            {
                let mut working = self.working_memory.write();
                if working.contains(&memory_id) {
                    working.remove(&memory_id)?;
                    working.add_shared(Arc::new(existing.clone()))?;
                }
            }
            {
                let mut session = self.session_memory.write();
                if session.contains(&memory_id) {
                    session.remove(&memory_id)?;
                    session.add_shared(Arc::new(existing.clone()))?;
                }
            }

            tracing::info!(
                external_id = %external_id,
                memory_id = %memory_id.0,
                version = existing.version,
                "Memory upserted (update)"
            );

            Ok((memory_id, true))
        } else {
            // === CREATE PATH ===
            let memory_id = MemoryId(Uuid::new_v4());
            let importance = self.calculate_importance(&experience);

            // Generate embeddings if not provided
            if experience.embeddings.is_none() {
                let content_hash = Self::sha256_hash(&experience.content);
                if let Some(cached_embedding) = self.content_cache.get(&content_hash) {
                    experience.embeddings = Some(cached_embedding.clone());
                } else {
                    match self.embedder.encode(&experience.content) {
                        Ok(embedding) => {
                            self.content_cache.insert(content_hash, embedding.clone());
                            experience.embeddings = Some(embedding);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to generate embedding: {}", e);
                        }
                    }
                }
            }

            // TEMPORAL EXTRACTION: Extract dates from content for temporal filtering
            if experience.temporal_refs.is_empty() {
                let temporal =
                    crate::memory::query_parser::extract_temporal_refs(&experience.content);
                for temp_ref in temporal.refs {
                    experience.temporal_refs.push(temp_ref.date.to_string());
                }
            }

            // Create memory with external_id
            let memory = Arc::new(Memory::new_with_external_id(
                memory_id.clone(),
                experience,
                importance,
                external_id.clone(),
                None, // agent_id
                None, // run_id
                None, // actor_id
                None, // created_at
            ));

            // Persist to storage
            self.long_term_memory.store(&memory)?;

            // Log creation
            self.logger.write().log_created(&memory, "working");

            // Add to working memory
            self.working_memory
                .write()
                .add_shared(Arc::clone(&memory))?;

            // Index in vector DB
            if let Err(e) = self.retriever.index_memory(&memory) {
                tracing::warn!("Failed to index memory {} in vector DB: {}", memory.id.0, e);
            }

            // Add entities to knowledge graph with co-occurrence edges
            // PERF: Build entity structs and extract co-occurrences OUTSIDE the lock
            if let Some(graph) = &self.graph_memory {
                let now = chrono::Utc::now();

                // Phase 1: Build entity structs with proper labels from NER
                let ner_lookup = build_ner_lookup(&memory.experience.ner_entities);

                // Batch-encode entity names for concept-level dedup
                let entity_names: Vec<&str> = memory
                    .experience
                    .entities
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                let entity_embeddings: Vec<Option<Vec<f32>>> = if entity_names.is_empty() {
                    Vec::new()
                } else {
                    match self.embedder.encode_batch(&entity_names) {
                        Ok(embs) => embs.into_iter().map(Some).collect(),
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                "Entity name embedding failed, concept merge disabled for this batch"
                            );
                            vec![None; entity_names.len()]
                        }
                    }
                };

                let entities_to_add: Vec<crate::graph_memory::EntityNode> = memory
                    .experience
                    .entities
                    .iter()
                    .zip(entity_embeddings)
                    .map(|(entity_name, embedding)| {
                        let (label, salience) = resolve_entity_label(entity_name, &ner_lookup);
                        crate::graph_memory::EntityNode {
                            uuid: Uuid::new_v4(),
                            name: entity_name.clone(),
                            labels: vec![label],
                            created_at: now,
                            last_seen_at: now,
                            mention_count: 1,
                            summary: String::new(),
                            attributes: std::collections::HashMap::new(),
                            name_embedding: embedding,
                            salience,
                            is_proper_noun: entity_name
                                .chars()
                                .next()
                                .map(|c| c.is_uppercase())
                                .unwrap_or(false),
                        }
                    })
                    .collect();

                // Phase 2: Use pre-extracted co-occurrence pairs or extract fresh
                let cooccurrence_pairs = if !memory.experience.cooccurrence_pairs.is_empty() {
                    memory.experience.cooccurrence_pairs.clone()
                } else {
                    let entity_extractor = crate::graph_memory::EntityExtractor::new();
                    entity_extractor.extract_cooccurrence_pairs(&memory.experience.content)
                };

                let edge_context = format!("Co-occurred in memory {}", memory.id.0);

                // Phase 3: Acquire read lock for graph insertions (GraphMemory is internally thread-safe)
                let graph_guard = graph.read();

                for entity in entities_to_add {
                    if let Err(e) = graph_guard.add_entity(entity.clone()) {
                        tracing::debug!("Failed to add entity '{}' to graph: {}", entity.name, e);
                    }
                }

                // Semantic edge weighting
                let l1_base_weight = crate::graph_memory::EdgeTier::L1Working.initial_weight();
                for (entity1, entity2) in cooccurrence_pairs {
                    if let (Ok(Some(e1)), Ok(Some(e2))) = (
                        graph_guard.find_entity_by_name(&entity1),
                        graph_guard.find_entity_by_name(&entity2),
                    ) {
                        let entity_confidence = Some((e1.salience + e2.salience) / 2.0);

                        let semantic_weight = match (&e1.name_embedding, &e2.name_embedding) {
                            (Some(emb1), Some(emb2)) => {
                                let sim = crate::similarity::cosine_similarity(emb1, emb2).max(0.0);
                                EDGE_SEMANTIC_WEIGHT_FLOOR
                                    + (1.0 - EDGE_SEMANTIC_WEIGHT_FLOOR) * sim
                            }
                            _ => 1.0,
                        };

                        let edge = crate::graph_memory::RelationshipEdge {
                            uuid: Uuid::new_v4(),
                            from_entity: e1.uuid,
                            to_entity: e2.uuid,
                            relation_type: crate::graph_memory::RelationType::CoOccurs,
                            strength: l1_base_weight * semantic_weight,
                            created_at: now,
                            valid_at: now,
                            invalidated_at: None,
                            source_episode_id: Some(memory.id.0),
                            context: edge_context.clone(),
                            last_activated: now,
                            activation_count: 1,
                            ltp_status: crate::graph_memory::LtpStatus::None,
                            tier: crate::graph_memory::EdgeTier::L1Working,
                            activation_timestamps: None,
                            entity_confidence,
                            created_by: crate::graph_memory::EdgeSource::CoOccurrence,
                        };

                        if let Err(e) = graph_guard.add_relationship(edge) {
                            tracing::trace!(
                                "Failed to add co-occurrence edge {}<->{}: {}",
                                entity1,
                                entity2,
                                e
                            );
                        }
                    }
                }
            }

            // Index in BM25 for hybrid search
            if let Err(e) = self.hybrid_search.index_memory(
                &memory.id,
                &memory.experience.content,
                &memory.experience.tags,
                &memory.experience.entities,
            ) {
                tracing::warn!("Failed to index memory {} in BM25: {}", memory.id.0, e);
            }
            if let Err(e) = self.hybrid_search.commit_and_reload() {
                tracing::warn!("Failed to commit/reload BM25 index: {}", e);
            }

            // Add to session if important
            if importance > self.config.importance_threshold {
                self.session_memory
                    .write()
                    .add_shared(Arc::clone(&memory))?;
            }

            // Update stats
            self.stats.write().total_memories += 1;

            tracing::info!(
                external_id = %external_id,
                memory_id = %memory_id.0,
                "Memory upserted (create)"
            );

            Ok((memory_id, false))
        }
    }

    /// Get the history of a memory (audit trail of changes)
    ///
    /// Returns the full revision history for memories with external linking.
    /// Returns empty vec for regular (non-mutable) memories.
    pub fn get_memory_history(&self, memory_id: &MemoryId) -> Result<Vec<MemoryRevision>> {
        let memory = self.long_term_memory.get(memory_id)?;
        Ok(memory.history.clone())
    }

    /// Find a memory by external ID
    ///
    /// Used to check if a memory already exists for an external entity
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<Memory>> {
        self.long_term_memory.find_by_external_id(external_id)
    }

    /// Run periodic maintenance (consolidation, activation decay, graph maintenance)
    ///
    /// Call this periodically (e.g., every 5 minutes) to:
    /// 1. Promote memories between tiers based on thresholds
    /// 2. Decay activation levels on all memories
    /// 3. Run graph maintenance (prune weak edges)
    ///
    /// `is_heavy`: when true, runs expensive operations (fact extraction, auto-repair)
    /// that require full RocksDB scans. Light cycles only touch in-memory data.
    ///
    /// Returns the number of memories processed for activation decay.
    /// Also records consolidation events for introspection.
    pub fn run_maintenance(
        &self,
        decay_factor: f32,
        user_id: &str,
        is_heavy: bool,
    ) -> Result<MaintenanceResult> {
        let start_time = std::time::Instant::now();
        let now = chrono::Utc::now();

        // 1. Consolidation: promote memories between tiers
        self.consolidate_if_needed()?;

        // 2. Decay activation on all in-memory memories (working + session)
        let mut decayed_count = 0;
        let mut at_risk_count = 0;
        const AT_RISK_THRESHOLD: f32 = 0.2; // Memories below this are at risk of being forgotten

        // Decay working memory activations with event tracking
        {
            let working = self.working_memory.read();
            for memory in working.all_memories() {
                let activation_before = memory.activation();
                memory.decay_activation(decay_factor);
                let activation_after = memory.activation();
                decayed_count += 1;

                // Only record event if there was actual decay
                if activation_before != activation_after {
                    let at_risk = activation_after < AT_RISK_THRESHOLD;
                    if at_risk {
                        at_risk_count += 1;
                    }

                    // Record decay event
                    self.record_consolidation_event(ConsolidationEvent::MemoryDecayed {
                        memory_id: memory.id.0.to_string(),
                        content_preview: memory.experience.content.chars().take(50).collect(),
                        activation_before,
                        activation_after,
                        at_risk,
                        timestamp: now,
                    });
                }
            }
        }

        // Decay session memory activations with event tracking
        {
            let session = self.session_memory.read();
            for memory in session.all_memories() {
                let activation_before = memory.activation();
                memory.decay_activation(decay_factor);
                let activation_after = memory.activation();
                decayed_count += 1;

                // Only record event if there was actual decay
                if activation_before != activation_after {
                    let at_risk = activation_after < AT_RISK_THRESHOLD;
                    if at_risk {
                        at_risk_count += 1;
                    }

                    // Record decay event
                    self.record_consolidation_event(ConsolidationEvent::MemoryDecayed {
                        memory_id: memory.id.0.to_string(),
                        content_preview: memory.experience.content.chars().take(50).collect(),
                        activation_before,
                        activation_after,
                        at_risk,
                        timestamp: now,
                    });
                }
            }
        }

        // 2.5 Potentiation: boost ALL memories based on access count (Hebbian LTP)
        // This implements "neurons that fire together wire together" - memories
        // that are accessed frequently get importance boosts during maintenance
        let mut potentiated_count = 0;
        {
            // Potentiate working memory
            let working = self.working_memory.read();
            for memory in working.all_memories() {
                // Only boost if below saturation threshold (0.95) to prevent
                // all frequently-accessed memories converging to importance=1.0
                if memory.access_count() >= POTENTIATION_ACCESS_THRESHOLD
                    && memory.importance() < 0.95
                {
                    let activation_before = memory.importance();
                    memory.boost_importance(POTENTIATION_MAINTENANCE_BOOST);
                    potentiated_count += 1;

                    self.record_consolidation_event(ConsolidationEvent::MemoryStrengthened {
                        memory_id: memory.id.0.to_string(),
                        content_preview: memory.experience.content.chars().take(50).collect(),
                        activation_before,
                        activation_after: memory.importance(),
                        reason: StrengtheningReason::MaintenancePotentiation,
                        timestamp: now,
                    });
                }
            }
        }
        {
            // Potentiate session memory
            let session = self.session_memory.read();
            for memory in session.all_memories() {
                if memory.access_count() >= POTENTIATION_ACCESS_THRESHOLD
                    && memory.importance() < 0.95
                {
                    let activation_before = memory.importance();
                    memory.boost_importance(POTENTIATION_MAINTENANCE_BOOST);
                    potentiated_count += 1;

                    self.record_consolidation_event(ConsolidationEvent::MemoryStrengthened {
                        memory_id: memory.id.0.to_string(),
                        content_preview: memory.experience.content.chars().take(50).collect(),
                        activation_before,
                        activation_after: memory.importance(),
                        reason: StrengtheningReason::MaintenancePotentiation,
                        timestamp: now,
                    });
                }
            }
        }

        if potentiated_count > 0 {
            tracing::debug!(
                "Potentiated {} memories during maintenance (access >= {})",
                potentiated_count,
                POTENTIATION_ACCESS_THRESHOLD
            );
        }

        // 3. Graph maintenance moved to state.rs run_maintenance_all_users()
        // This fixes the double-decay bug: apply_decay() was called both here
        // (via graph_maintenance()) and in state.rs. Now only state.rs calls it,
        // and the result is used for orphan detection (Direction 2 coupling).

        // 3.5. Temporal fact decay: decay/delete stale facts (heavy only)
        // Scans all facts via RocksDB iterator — deferred to heavy cycles
        let (facts_decayed, facts_deleted) = if is_heavy {
            self.decay_facts_for_all_users().unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        if facts_decayed > 0 || facts_deleted > 0 {
            tracing::debug!(
                "Temporal fact maintenance: {} decayed, {} deleted",
                facts_decayed,
                facts_deleted
            );
        }

        // 3.7. Heavy cycle: load all memories once for both fact extraction and replay.
        // This avoids two separate RocksDB full scans on the same cycle.
        let all_memories_for_heavy: Vec<SharedMemory> = if is_heavy {
            self.get_all_memories().unwrap_or_default()
        } else {
            Vec::new()
        };

        // 3.8. Fact extraction: consolidate episodic memories into semantic facts
        // HEAVY ONLY: requires ONNX inference for embedding new facts.
        // The dirty flag (fact_extraction_needed) is only checked on heavy cycles;
        // it stays set across light cycles until the next heavy cycle processes it.
        let mut facts_extracted_count = 0;
        let mut facts_reinforced_count = 0;
        if is_heavy
            && self
                .fact_extraction_needed
                .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            let all_memories = &all_memories_for_heavy;
            if !all_memories.is_empty() {
                // Incremental: only process memories created since last extraction watermark.
                // First run (watermark=0) processes everything; subsequent runs only new memories.
                // Lazy init: if watermark is 0 (startup sentinel), load persisted value
                // or derive from the latest fact's created_at timestamp.
                let mut watermark_millis = self
                    .fact_extraction_watermark
                    .load(std::sync::atomic::Ordering::Relaxed);
                if watermark_millis == 0 {
                    watermark_millis = self
                        .long_term_memory
                        .get_fact_watermark(user_id)
                        .or_else(|| self.fact_store.latest_fact_created_at(user_id))
                        .unwrap_or(0);
                    if watermark_millis > 0 {
                        self.fact_extraction_watermark
                            .store(watermark_millis, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                let watermark_dt = chrono::DateTime::from_timestamp_millis(watermark_millis)
                    .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC);

                let memories: Vec<Memory> = all_memories
                    .iter()
                    .filter(|m| m.created_at > watermark_dt)
                    .map(|arc_mem| arc_mem.as_ref().clone())
                    .collect();

                tracing::info!(
                    total_memories = all_memories.len(),
                    new_since_watermark = memories.len(),
                    watermark = %watermark_dt.format("%Y-%m-%dT%H:%M:%S"),
                    "Incremental fact extraction"
                );

                let consolidator = compression::SemanticConsolidator::new();
                let consolidation_result = consolidator.consolidate(&memories);

                if !consolidation_result.new_facts.is_empty() {
                    // Batch-encode all new fact texts for hybrid dedup
                    let fact_texts: Vec<&str> = consolidation_result
                        .new_facts
                        .iter()
                        .map(|f| f.fact.as_str())
                        .collect();
                    let fact_embeddings: Vec<Option<Vec<f32>>> = match self
                        .embedder
                        .encode_batch(&fact_texts)
                    {
                        Ok(embs) => embs.into_iter().map(Some).collect(),
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                "Fact embedding batch failed, falling back to Jaccard-only dedup"
                            );
                            vec![None; fact_texts.len()]
                        }
                    };

                    // Quality gate: reject facts whose embeddings diverge from
                    // source cluster centroid. Prevents hallucinated consolidations.
                    let quality_threshold =
                        crate::constants::CONSOLIDATION_QUALITY_GATE_THRESHOLD;
                    let max_sources =
                        crate::constants::CONSOLIDATION_QUALITY_GATE_MAX_SOURCES;

                    let gated: Vec<(SemanticFact, Option<Vec<f32>>)> = consolidation_result
                        .new_facts
                        .into_iter()
                        .zip(fact_embeddings)
                        .filter(|(fact, fact_emb)| {
                            let Some(ref fv) = fact_emb else { return true };
                            let source_texts: Vec<&str> = fact
                                .source_memories
                                .iter()
                                .filter_map(|id| {
                                    memories.iter().find(|m| &m.id == id)
                                })
                                .take(max_sources)
                                .map(|m| m.experience.content.as_str())
                                .collect();
                            if source_texts.is_empty() {
                                return true;
                            }
                            match self.embedder.encode_batch(&source_texts) {
                                Ok(src_embs) if !src_embs.is_empty() => {
                                    let dim = src_embs[0].len();
                                    let mut centroid = vec![0.0f32; dim];
                                    for emb in &src_embs {
                                        for (i, &v) in emb.iter().enumerate() {
                                            if i < dim {
                                                centroid[i] += v;
                                            }
                                        }
                                    }
                                    let n = src_embs.len() as f32;
                                    for v in &mut centroid {
                                        *v /= n;
                                    }
                                    let sim = crate::similarity::cosine_similarity(fv, &centroid);
                                    if sim < quality_threshold {
                                        tracing::debug!(
                                            fact = %fact.fact,
                                            similarity = %format!("{sim:.3}"),
                                            threshold = quality_threshold,
                                            "Quality gate: rejected fact diverging from source centroid"
                                        );
                                        return false;
                                    }
                                    true
                                }
                                _ => true,
                            }
                        })
                        .collect();

                    let mut truly_new: Vec<(SemanticFact, Option<Vec<f32>>)> = Vec::new();

                    for (fact, embedding) in gated.into_iter()
                    {
                        // Hybrid dedup: embedding cosine + entity gate + polarity + Jaccard floor
                        match self.fact_store.find_similar(
                            user_id,
                            &fact.fact,
                            &fact.related_entities,
                            embedding.as_deref(),
                        ) {
                            Ok(Some(mut existing)) => {
                                // Reinforce the existing fact
                                existing.support_count += 1;
                                existing.last_reinforced = now;
                                let confidence_before = existing.confidence;
                                let boost = 0.1 * (1.0 - existing.confidence);
                                existing.confidence = (existing.confidence + boost).min(1.0);

                                // Extend source memories and related entities
                                for src in &fact.source_memories {
                                    if !existing.source_memories.contains(src) {
                                        existing.source_memories.push(src.clone());
                                    }
                                }
                                for entity in &fact.related_entities {
                                    if !existing.related_entities.contains(entity) {
                                        existing.related_entities.push(entity.clone());
                                    }
                                }

                                if let Err(e) = self.fact_store.update(user_id, &existing) {
                                    tracing::debug!("Failed to reinforce fact: {e}");
                                } else {
                                    // Update existing fact's embedding with latest encoding
                                    if let Some(ref emb) = embedding {
                                        let _ = self.fact_store.store_embedding(
                                            user_id,
                                            &existing.id,
                                            emb,
                                        );
                                    }
                                    facts_reinforced_count += 1;
                                    self.record_consolidation_event_for_user(
                                        user_id,
                                        ConsolidationEvent::FactReinforced {
                                            fact_id: existing.id.clone(),
                                            fact_content: existing.fact.clone(),
                                            confidence_before,
                                            confidence_after: existing.confidence,
                                            new_support_count: existing.support_count,
                                            timestamp: now,
                                        },
                                    );
                                }
                            }
                            _ => {
                                truly_new.push((fact.clone(), embedding));
                            }
                        }
                    }

                    // Store new facts
                    if !truly_new.is_empty() {
                        let facts_only: Vec<SemanticFact> =
                            truly_new.iter().map(|(f, _)| f.clone()).collect();
                        match self.fact_store.store_batch(user_id, &facts_only) {
                            Ok(stored) => {
                                facts_extracted_count = stored;
                                // Store embeddings for newly persisted facts
                                for (fact, embedding) in &truly_new {
                                    if let Some(emb) = embedding {
                                        let _ =
                                            self.fact_store.store_embedding(user_id, &fact.id, emb);
                                    }
                                    self.record_consolidation_event_for_user(
                                        user_id,
                                        ConsolidationEvent::FactExtracted {
                                            fact_id: fact.id.clone(),
                                            fact_content: fact.fact.clone(),
                                            confidence: fact.confidence,
                                            fact_type: format!("{:?}", fact.fact_type),
                                            source_memory_count: fact.source_memories.len(),
                                            timestamp: now,
                                        },
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to store extracted facts: {e}");
                            }
                        }

                        // Connect newly extracted facts to the knowledge graph
                        self.connect_facts_to_graph(&facts_only);
                    }

                    if facts_extracted_count > 0 || facts_reinforced_count > 0 {
                        tracing::debug!(
                            extracted = facts_extracted_count,
                            reinforced = facts_reinforced_count,
                            "Fact consolidation during maintenance"
                        );
                    }
                }

                // Advance watermark to the LAST memory's created_at timestamp,
                // NOT to now(). Using now() would skip memories created during the
                // (potentially slow) fact extraction cycle — they'd have created_at
                // < now() and never be processed for facts.
                if !memories.is_empty() {
                    let new_watermark = memories
                        .iter()
                        .map(|m| m.created_at.timestamp_millis())
                        .max()
                        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
                    self.fact_extraction_watermark
                        .store(new_watermark, std::sync::atomic::Ordering::Relaxed);
                    self.long_term_memory
                        .set_fact_watermark(user_id, new_watermark);
                }
            }
        } else {
            tracing::debug!("Fact extraction skipped: no new memories since last cycle");
        }

        // 4. SHO-105 + PIPE-2: Memory replay cycle (consolidation during heavy cycles)
        //
        // HEAVY ONLY: Replay draws candidates from ALL memory tiers (including long-term
        // via the shared all_memories_for_heavy loaded above). Light cycles skip replay
        // entirely — analogous to "consolidation during deep sleep, not during waking."
        //
        // Pattern detection still runs to record triggers, but actual replay execution
        // only happens on heavy cycles where we have the full memory corpus.
        let mut replay_result = replay::ReplayCycleResult::default();
        {
            // PIPE-2: Check for pattern-triggered replay first
            let pattern_result = self.pattern_detector.write().detect_patterns();
            let has_pattern_triggers = !pattern_result.triggers.is_empty();

            // Log pattern detection results
            if has_pattern_triggers {
                tracing::debug!(
                    "Pattern detection: {} entity, {} semantic, {} temporal, {} salience, {} behavioral triggers",
                    pattern_result.entity_patterns_found,
                    pattern_result.semantic_clusters_found,
                    pattern_result.temporal_clusters_found,
                    pattern_result.salience_spikes_found,
                    pattern_result.behavioral_changes_found
                );

                // Record pattern-triggered replay events
                for trigger in &pattern_result.triggers {
                    self.record_consolidation_event(
                        introspection::ConsolidationEvent::PatternTriggeredReplay {
                            trigger_type: trigger.trigger_type_name().to_string(),
                            memory_ids: trigger.memory_ids(),
                            pattern_confidence: match trigger {
                                pattern_detection::ReplayTrigger::EntityCoOccurrence {
                                    confidence,
                                    ..
                                } => *confidence,
                                pattern_detection::ReplayTrigger::SemanticCluster {
                                    avg_similarity,
                                    ..
                                } => *avg_similarity,
                                pattern_detection::ReplayTrigger::SalienceSpike {
                                    importance,
                                    ..
                                } => *importance,
                                _ => 0.7, // Default confidence for other triggers
                            },
                            trigger_description: trigger.description(),
                            timestamp: now,
                        },
                    );
                }
            }

            // Replay only on heavy cycles — uses shared all_memories_for_heavy
            let timer_should_replay = self.replay_manager.read().should_replay();
            let should_replay = is_heavy && (has_pattern_triggers || timer_should_replay);

            if should_replay && !all_memories_for_heavy.is_empty() {
                // Build replay candidates from ALL memory tiers (not just working+session)
                let graph_ref = self.graph_memory.clone();
                let candidates_data: Vec<_> = all_memories_for_heavy
                    .iter()
                    .map(|m| {
                        // Fetch actual connections from GraphMemory
                        let connections: Vec<String> = if let Some(ref graph) = graph_ref {
                            let graph_guard = graph.read();
                            graph_guard
                                .find_memory_associations(&m.id.0, 10)
                                .unwrap_or_default()
                                .into_iter()
                                .map(|(uuid, _)| uuid.to_string())
                                .collect()
                        } else {
                            Vec::new()
                        };
                        let arousal = m
                            .experience
                            .context
                            .as_ref()
                            .map(|c| c.emotional.arousal)
                            .unwrap_or(0.3);
                        (
                            m.id.0.to_string(),
                            m.importance(),
                            arousal,
                            m.created_at,
                            connections,
                            m.experience.content.chars().take(50).collect::<String>(),
                        )
                    })
                    .collect();

                // Identify and execute replay
                let candidates = self
                    .replay_manager
                    .read()
                    .identify_replay_candidates(&candidates_data);

                if !candidates.is_empty() {
                    let (memory_boosts, edge_boosts, events) =
                        self.replay_manager.write().execute_replay(&candidates);

                    replay_result.memories_replayed = candidates.len();
                    replay_result.edges_strengthened = edge_boosts.len();
                    replay_result.total_priority_score =
                        candidates.iter().map(|c| c.priority_score).sum();

                    // Collect replayed memory IDs for entity-entity edge strengthening
                    replay_result.replay_memory_ids =
                        candidates.iter().map(|c| c.memory_id.clone()).collect();

                    // Apply memory boosts
                    for (mem_id_str, boost) in &memory_boosts {
                        if let Ok(mem_id) = uuid::Uuid::parse_str(mem_id_str) {
                            if let Ok(memory) = self.long_term_memory.get(&MemoryId(mem_id)) {
                                memory.boost_importance(*boost);
                                if let Err(e) = self.long_term_memory.update(&memory) {
                                    tracing::debug!("Failed to persist replay boost: {e}");
                                }
                            }
                        }
                    }

                    // Collect edge boosts to return - will be applied via GraphMemory at API layer
                    replay_result.edge_boosts = edge_boosts;
                    if !replay_result.edge_boosts.is_empty() {
                        tracing::debug!(
                            "Replay produced {} edge boosts (to be applied via GraphMemory)",
                            replay_result.edge_boosts.len()
                        );
                    }

                    // Record events
                    for event in events {
                        self.record_consolidation_event(event);
                    }

                    // Record replay cycle completion
                    self.record_consolidation_event(ConsolidationEvent::ReplayCycleCompleted {
                        memories_replayed: replay_result.memories_replayed,
                        edges_strengthened: replay_result.edges_strengthened,
                        total_priority_score: replay_result.total_priority_score,
                        duration_ms: start_time.elapsed().as_millis() as u64,
                        timestamp: now,
                    });

                    tracing::debug!(
                        "Replay cycle complete: {} memories replayed, {} edges strengthened",
                        replay_result.memories_replayed,
                        replay_result.edges_strengthened
                    );
                }
            }
        }

        // 4.5. PIPE-2: Cleanup old patterns to prevent unbounded memory growth
        // Removes patterns older than 24 hours
        self.pattern_detector.write().cleanup();

        // 5. Auto-repair index integrity and compact if needed (heavy only)
        // repair_vector_index() does a full RocksDB scan + ONNX inference per orphan
        if is_heavy {
            self.auto_repair_and_compact();
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        // Record maintenance cycle completion event
        self.record_consolidation_event(ConsolidationEvent::MaintenanceCycleCompleted {
            memories_processed: decayed_count,
            memories_decayed: decayed_count, // All memories get decay applied
            edges_pruned: 0,                 // Graph maintenance doesn't report this yet
            duration_ms,
            timestamp: now,
        });

        tracing::debug!(
            "Maintenance complete: {} memories decayed (factor={}), {} at risk, {} replayed, took {}ms",
            decayed_count,
            decay_factor,
            at_risk_count,
            replay_result.memories_replayed,
            duration_ms
        );

        Ok(MaintenanceResult {
            decayed_count,
            edge_boosts: replay_result.edge_boosts,
            replay_memory_ids: replay_result.replay_memory_ids,
            memories_replayed: replay_result.memories_replayed,
            total_priority_score: replay_result.total_priority_score,
            facts_extracted: facts_extracted_count,
            facts_reinforced: facts_reinforced_count,
        })
    }

    // =========================================================================
    // CONSOLIDATION INTROSPECTION API
    // =========================================================================

    /// Get a consolidation report for a time period
    ///
    /// Shows what the memory system has been learning:
    /// - Which memories strengthened or decayed
    /// - What associations formed or were pruned
    /// - What facts were extracted or reinforced
    ///
    /// # Arguments
    /// * `since` - Start of the time period
    /// * `until` - End of the time period (default: now)
    pub fn get_consolidation_report(
        &self,
        since: chrono::DateTime<chrono::Utc>,
        until: Option<chrono::DateTime<chrono::Utc>>,
    ) -> ConsolidationReport {
        let until = until.unwrap_or_else(chrono::Utc::now);
        let events = self.consolidation_events.read();
        events.generate_report(since, until)
    }

    /// Get a consolidation report for a user using persisted history
    ///
    /// Unlike `get_consolidation_report`, this method uses persisted learning history
    /// and can generate reports spanning across restarts. It combines:
    /// - Persisted significant events from learning_history (survives restarts)
    /// - Ephemeral events from the event buffer (current session)
    ///
    /// Use this when you need historical reports beyond the current session.
    pub fn get_consolidation_report_for_user(
        &self,
        user_id: &str,
        since: chrono::DateTime<chrono::Utc>,
        until: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<ConsolidationReport> {
        let until = until.unwrap_or_else(chrono::Utc::now);

        // Get persisted significant events from learning history
        let persisted_events = self
            .learning_history
            .events_in_range(user_id, since, until)?;

        // Get ephemeral events from the buffer
        let ephemeral_events = {
            let events = self.consolidation_events.read();
            events.events_since(since)
        };

        // Combine events, deduplicating by (timestamp, event_type) to avoid
        // dropping distinct events that share a nanosecond timestamp.
        let mut all_events: Vec<ConsolidationEvent> = Vec::new();
        let mut seen_keys: std::collections::HashSet<(
            i64,
            std::mem::Discriminant<ConsolidationEvent>,
        )> = std::collections::HashSet::new();

        // Add persisted events first (these are significant events that survived restart)
        for stored in &persisted_events {
            let ts = stored.event.timestamp().timestamp_nanos_opt().unwrap_or(0);
            let key = (ts, std::mem::discriminant(&stored.event));
            if seen_keys.insert(key) {
                all_events.push(stored.event.clone());
            }
        }

        // Add ephemeral events that aren't already included
        let until_nanos = until.timestamp_nanos_opt().unwrap_or(i64::MAX);
        for event in ephemeral_events {
            let ts = event.timestamp().timestamp_nanos_opt().unwrap_or(0);
            let key = (ts, std::mem::discriminant(&event));
            if ts <= until_nanos && seen_keys.insert(key) {
                all_events.push(event);
            }
        }

        // Sort by timestamp
        all_events.sort_by_key(|a| a.timestamp());

        // Generate report from combined events
        let report =
            ConsolidationEventBuffer::generate_report_from_events(&all_events, since, until);

        Ok(report)
    }

    /// Get all consolidation events since a timestamp
    ///
    /// Returns raw events for detailed analysis
    pub fn get_consolidation_events_since(
        &self,
        since: chrono::DateTime<chrono::Utc>,
    ) -> Vec<ConsolidationEvent> {
        let events = self.consolidation_events.read();
        events.events_since(since)
    }

    /// Get all consolidation events in the buffer
    pub fn get_all_consolidation_events(&self) -> Vec<ConsolidationEvent> {
        let events = self.consolidation_events.read();
        events.all_events()
    }

    /// Record a consolidation event
    ///
    /// Used internally by the memory system to log learning events.
    /// Also available for external callers that want to track custom events.
    pub fn record_consolidation_event(&self, event: ConsolidationEvent) {
        let mut events = self.consolidation_events.write();
        events.push(event);
    }

    /// Record a consolidation event for a specific user
    ///
    /// This method both:
    /// 1. Pushes to the ephemeral event buffer (for real-time introspection)
    /// 2. Persists significant events to learning_history (for retrieval boosting)
    ///
    /// Use this instead of `record_consolidation_event` when you have a user_id.
    pub fn record_consolidation_event_for_user(&self, user_id: &str, event: ConsolidationEvent) {
        // Always push to ephemeral buffer
        {
            let mut events = self.consolidation_events.write();
            events.push(event.clone());
        }

        // Persist significant events to learning history
        if event.is_significant() {
            if let Err(e) = self.learning_history.record(user_id, &event) {
                tracing::warn!(
                    user_id = %user_id,
                    event_type = ?std::mem::discriminant(&event),
                    error = %e,
                    "Failed to persist learning event"
                );
            }
        }
    }

    /// Clear all consolidation events
    pub fn clear_consolidation_events(&self) {
        let mut events = self.consolidation_events.write();
        events.clear();
    }

    /// Get the number of consolidation events in the buffer
    pub fn consolidation_event_count(&self) -> usize {
        let events = self.consolidation_events.read();
        events.len()
    }

    // =========================================================================
    // SEMANTIC FACT OPERATIONS (SHO-f0e7)
    // Distilled knowledge extracted from episodic memories
    // =========================================================================

    /// Distill semantic facts from episodic memories
    ///
    /// Runs the consolidation process to extract durable knowledge:
    /// 1. Find patterns appearing in multiple memories
    /// 2. Create or reinforce semantic facts
    /// 3. Store facts in the fact store
    ///
    /// # Arguments
    /// * `user_id` - User whose memories to consolidate
    /// * `min_support` - Minimum memories needed to form a fact (default: 3)
    /// * `min_age_days` - Minimum age of memories to consider (default: 7)
    ///
    /// # Returns
    /// ConsolidationResult with stats and newly extracted facts
    pub fn distill_facts(
        &self,
        user_id: &str,
        min_support: usize,
        min_age_days: i64,
    ) -> Result<ConsolidationResult> {
        // Get all memories for consolidation
        let all_memories = self.get_all_memories()?;

        // Incremental: only process memories created since last extraction watermark
        let mut watermark_millis = self
            .fact_extraction_watermark
            .load(std::sync::atomic::Ordering::Relaxed);
        if watermark_millis == 0 {
            watermark_millis = self
                .long_term_memory
                .get_fact_watermark(user_id)
                .or_else(|| self.fact_store.latest_fact_created_at(user_id))
                .unwrap_or(0);
            if watermark_millis > 0 {
                self.fact_extraction_watermark
                    .store(watermark_millis, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let watermark_dt = chrono::DateTime::from_timestamp_millis(watermark_millis)
            .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC);

        let memories: Vec<Memory> = all_memories
            .iter()
            .filter(|m| m.created_at > watermark_dt)
            .map(|arc_mem| arc_mem.as_ref().clone())
            .collect();

        tracing::info!(
            total_memories = all_memories.len(),
            new_since_watermark = memories.len(),
            watermark = %watermark_dt.format("%Y-%m-%dT%H:%M:%S"),
            "Incremental fact extraction (on-demand)"
        );

        // Create consolidator with custom thresholds
        let consolidator =
            compression::SemanticConsolidator::with_thresholds(min_support, min_age_days);

        // Run consolidation
        let mut result = consolidator.consolidate(&memories);

        // Store extracted facts with quality gating
        if !result.new_facts.is_empty() {
            // Quality gate: reject facts whose embeddings diverge from
            // source cluster centroid. Prevents hallucinated consolidations.
            let quality_threshold =
                crate::constants::CONSOLIDATION_QUALITY_GATE_THRESHOLD;
            let max_sources =
                crate::constants::CONSOLIDATION_QUALITY_GATE_MAX_SOURCES;

            // Encode all fact texts
            let fact_texts: Vec<&str> = result.new_facts.iter().map(|f| f.fact.as_str()).collect();
            let fact_embeddings: Vec<Option<Vec<f32>>> = match self.embedder.encode_batch(&fact_texts) {
                Ok(embs) => embs.into_iter().map(Some).collect(),
                Err(e) => {
                    tracing::warn!("Failed to encode facts for quality gate: {e}");
                    vec![None; fact_texts.len()]
                }
            };

            let gated_facts: Vec<(SemanticFact, Option<Vec<f32>>)> = std::mem::take(&mut result.new_facts)
                .into_iter()
                .zip(fact_embeddings)
                .filter(|(fact, fact_emb)| {
                    let Some(ref fv) = fact_emb else { return true };
                    let source_texts: Vec<&str> = fact
                        .source_memories
                        .iter()
                        .filter_map(|id| {
                            memories.iter().find(|m| &m.id == id)
                        })
                        .take(max_sources)
                        .map(|m| m.experience.content.as_str())
                        .collect();
                    if source_texts.is_empty() {
                        return true;
                    }
                    match self.embedder.encode_batch(&source_texts) {
                        Ok(src_embs) if !src_embs.is_empty() => {
                            let dim = src_embs[0].len();
                            let mut centroid = vec![0.0f32; dim];
                            for emb in &src_embs {
                                for (i, &v) in emb.iter().enumerate() {
                                    if i < dim {
                                        centroid[i] += v;
                                    }
                                }
                            }
                            let n = src_embs.len() as f32;
                            for v in &mut centroid {
                                *v /= n;
                            }
                            let sim = crate::similarity::cosine_similarity(fv, &centroid);
                            if sim < quality_threshold {
                                tracing::debug!(
                                    fact = %fact.fact,
                                    similarity = %format!("{sim:.3}"),
                                    threshold = quality_threshold,
                                    "Quality gate: rejected fact diverging from source centroid"
                                );
                                return false;
                            }
                            true
                        }
                        _ => true,
                    }
                })
                .collect();
            let facts_gated = result.facts_extracted.saturating_sub(gated_facts.len());

            let gated_fact_refs: Vec<SemanticFact> = gated_facts.iter().map(|(f, _)| f.clone()).collect();
            let stored = self.fact_store.store_batch(user_id, &gated_fact_refs)?;
            tracing::info!(
                user_id = %user_id,
                facts_extracted = result.facts_extracted,
                facts_gated,
                facts_stored = stored,
                "Semantic distillation complete"
            );

            // Store embeddings for distilled facts
            for (fact, emb) in &gated_facts {
                if let Some(emb) = emb {
                    let _ = self.fact_store.store_embedding(user_id, &fact.id, emb);
                }
            }

            // Record consolidation event for each fact (persists significant events)
            for (fact, _) in &gated_facts {
                self.record_consolidation_event_for_user(
                    user_id,
                    ConsolidationEvent::FactExtracted {
                        fact_id: fact.id.clone(),
                        fact_content: fact.fact.clone(),
                        confidence: fact.confidence,
                        fact_type: format!("{:?}", fact.fact_type),
                        source_memory_count: fact.source_memories.len(),
                        timestamp: chrono::Utc::now(),
                    },
                );
            }

            // Update result with gated facts for return value
            result.new_facts = gated_fact_refs;
        }

        // Advance watermark after successful extraction
        if !memories.is_empty() {
            let new_watermark = chrono::Utc::now().timestamp_millis();
            self.fact_extraction_watermark
                .store(new_watermark, std::sync::atomic::Ordering::Relaxed);
            self.long_term_memory
                .set_fact_watermark(user_id, new_watermark);
        }

        Ok(result)
    }

    /// Get semantic facts for a user
    ///
    /// # Arguments
    /// * `user_id` - User whose facts to retrieve
    /// * `limit` - Maximum number of facts to return
    pub fn get_facts(&self, user_id: &str, limit: usize) -> Result<Vec<SemanticFact>> {
        self.fact_store.list(user_id, limit)
    }

    /// Get facts related to a specific entity
    ///
    /// # Arguments
    /// * `user_id` - User whose facts to search
    /// * `entity` - Entity to search for (e.g., "authentication", "JWT")
    /// * `limit` - Maximum number of facts to return
    pub fn get_facts_by_entity(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        self.fact_store.find_by_entity(user_id, entity, limit)
    }

    /// Get facts of a specific type
    ///
    /// # Arguments
    /// * `user_id` - User whose facts to search
    /// * `fact_type` - Type of fact (Preference, Procedure, Definition, etc.)
    /// * `limit` - Maximum number of facts to return
    pub fn get_facts_by_type(
        &self,
        user_id: &str,
        fact_type: FactType,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        self.fact_store.find_by_type(user_id, fact_type, limit)
    }

    /// Search facts by keyword
    ///
    /// # Arguments
    /// * `user_id` - User whose facts to search
    /// * `query` - Search query
    /// * `limit` - Maximum number of facts to return
    pub fn search_facts(
        &self,
        user_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        self.fact_store.search(user_id, query, limit)
    }

    /// Get statistics about stored facts
    pub fn get_fact_stats(&self, user_id: &str) -> Result<facts::FactStats> {
        self.fact_store.stats(user_id)
    }

    /// Get facts associated with graph entity names.
    ///
    /// Bridges graph traversal → fact retrieval: when spreading activation discovers
    /// entity nodes, this method returns the semantic facts linked to those entities.
    /// Results are deduplicated and sorted by confidence (highest first).
    pub fn get_facts_for_graph_entities(
        &self,
        user_id: &str,
        entity_names: &[String],
        limit_per_entity: usize,
    ) -> Result<Vec<SemanticFact>> {
        let mut seen_ids = std::collections::HashSet::new();
        let mut results = Vec::new();

        for name in entity_names {
            let facts = self
                .fact_store
                .find_by_entity(user_id, name, limit_per_entity)?;
            for fact in facts {
                if seen_ids.insert(fact.id.clone()) {
                    results.push(fact);
                }
            }
        }

        results.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        Ok(results)
    }

    /// Reinforce a fact with new supporting evidence
    ///
    /// Called when a new memory supports an existing fact.
    /// Increments support_count and boosts confidence.
    pub fn reinforce_fact(
        &self,
        user_id: &str,
        fact_id: &str,
        memory_id: &MemoryId,
    ) -> Result<bool> {
        if let Some(mut fact) = self.fact_store.get(user_id, fact_id)? {
            // Track confidence before change for event
            let confidence_before = fact.confidence;

            // Increment support
            fact.support_count += 1;
            fact.last_reinforced = chrono::Utc::now();

            // Boost confidence with diminishing returns
            let boost = 0.1 * (1.0 - fact.confidence);
            fact.confidence = (fact.confidence + boost).min(1.0);

            // Add source if not already present
            if !fact.source_memories.contains(memory_id) {
                fact.source_memories.push(memory_id.clone());
            }

            // Update in store
            self.fact_store.update(user_id, &fact)?;

            // Record reinforcement event (persists significant events)
            self.record_consolidation_event_for_user(
                user_id,
                ConsolidationEvent::FactReinforced {
                    fact_id: fact.id.clone(),
                    fact_content: fact.fact.clone(),
                    confidence_before,
                    confidence_after: fact.confidence,
                    new_support_count: fact.support_count,
                    timestamp: chrono::Utc::now(),
                },
            );

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete a fact (soft delete or hard delete)
    pub fn delete_fact(&self, user_id: &str, fact_id: &str) -> Result<bool> {
        self.fact_store.delete(user_id, fact_id)
    }

    /// Get the fact store for direct access
    pub fn fact_store(&self) -> &Arc<facts::SemanticFactStore> {
        &self.fact_store
    }

    // =========================================================================
    // SHO-118: DECISION LINEAGE GRAPH METHODS
    // =========================================================================

    /// Get the lineage graph for direct access
    pub fn lineage_graph(&self) -> &Arc<lineage::LineageGraph> {
        &self.lineage_graph
    }

    /// Infer and store lineage between a new memory and existing memories
    ///
    /// Called after storing a new memory to automatically detect causal relationships.
    /// Uses entity overlap, temporal proximity, and memory type patterns.
    pub fn infer_lineage_for_memory(
        &self,
        user_id: &str,
        new_memory: &Memory,
        candidate_memories: &[Memory],
    ) -> Result<Vec<LineageEdge>> {
        let mut inferred_edges = Vec::new();

        for candidate in candidate_memories {
            // Try inferring from candidate to new memory (candidate caused new)
            if let Some((relation, confidence)) =
                self.lineage_graph.infer_relation(candidate, new_memory)
            {
                // Check if edge already exists
                if !self
                    .lineage_graph
                    .edge_exists(user_id, &candidate.id, &new_memory.id)?
                {
                    let edge = LineageEdge::inferred(
                        candidate.id.clone(),
                        new_memory.id.clone(),
                        relation,
                        confidence,
                    );
                    self.lineage_graph.store_edge(user_id, &edge)?;
                    inferred_edges.push(edge);
                }
            }
        }

        // Check for branch signal in memory content
        if lineage::LineageGraph::detect_branch_signal(&new_memory.experience.content) {
            // Ensure main branch exists
            self.lineage_graph.ensure_main_branch(user_id)?;
        }

        Ok(inferred_edges)
    }

    /// Trace lineage from a memory
    pub fn trace_lineage(
        &self,
        user_id: &str,
        memory_id: &MemoryId,
        direction: TraceDirection,
        max_depth: usize,
    ) -> Result<LineageTrace> {
        self.lineage_graph
            .trace(user_id, memory_id, direction, max_depth)
    }

    /// Find the root cause of a memory
    pub fn find_root_cause(&self, user_id: &str, memory_id: &MemoryId) -> Result<Option<MemoryId>> {
        self.lineage_graph.find_root_cause(user_id, memory_id)
    }

    /// Get lineage statistics
    pub fn lineage_stats(&self, user_id: &str) -> Result<LineageStats> {
        self.lineage_graph.stats(user_id)
    }

    /// Decay facts for all users during maintenance
    ///
    /// Facts decay based on lack of reinforcement. The decay rate is modulated by support_count:
    /// - Higher support = slower decay (fact is well-established)
    /// - Lower support = faster decay (fact is tentative)
    ///
    /// Returns (facts_decayed, facts_deleted)
    fn decay_facts_for_all_users(&self) -> Result<(usize, usize)> {
        use crate::constants::{
            FACT_DECAY_GRACE_DAYS, FACT_DECAY_HALF_LIFE_BASE_DAYS,
            FACT_DECAY_HALF_LIFE_PER_SUPPORT_DAYS,
        };
        const DELETE_CONFIDENCE: f32 = 0.1;

        let now = chrono::Utc::now();
        let mut total_decayed = 0;
        let mut total_deleted = 0;

        let user_ids = self.fact_store.list_users(100)?;

        for user_id in &user_ids {
            let facts = self.fact_store.list(user_id, 10000)?;

            for mut fact in facts {
                let days_since_reinforcement = (now - fact.last_reinforced).num_days();

                // Grace period: no decay at all
                if days_since_reinforcement <= FACT_DECAY_GRACE_DAYS {
                    continue;
                }

                let confidence_before = fact.confidence;

                // Exponential half-life decay: confidence × 0.5^(elapsed / half_life)
                // Half-life grows linearly with support_count — each corroborating source
                // is genuine evidence that the fact is stable knowledge.
                let elapsed = (days_since_reinforcement - FACT_DECAY_GRACE_DAYS) as f64;
                let half_life = FACT_DECAY_HALF_LIFE_BASE_DAYS
                    + (fact.support_count as f64 * FACT_DECAY_HALF_LIFE_PER_SUPPORT_DAYS);
                let decay_factor = (0.5_f64).powf(elapsed / half_life) as f32;
                fact.confidence = (confidence_before * decay_factor).max(0.0);

                // Delete if below threshold
                if fact.confidence < DELETE_CONFIDENCE {
                    // Record deletion event (persists significant events)
                    self.record_consolidation_event_for_user(
                        user_id,
                        ConsolidationEvent::FactDeleted {
                            fact_id: fact.id.clone(),
                            fact_content: fact.fact.clone(),
                            final_confidence: fact.confidence,
                            support_count: fact.support_count,
                            reason: format!("confidence_below_{}", DELETE_CONFIDENCE),
                            timestamp: now,
                        },
                    );

                    self.fact_store.delete(user_id, &fact.id)?;
                    total_deleted += 1;
                } else if (confidence_before - fact.confidence) > 0.001 {
                    // Record decay event (not significant - routine maintenance)
                    self.record_consolidation_event(ConsolidationEvent::FactDecayed {
                        fact_id: fact.id.clone(),
                        fact_content: fact.fact.clone(),
                        confidence_before,
                        confidence_after: fact.confidence,
                        days_since_reinforcement,
                        timestamp: now,
                    });

                    self.fact_store.update(user_id, &fact)?;
                    total_decayed += 1;
                }
            }
        }

        if total_decayed > 0 || total_deleted > 0 {
            tracing::info!(
                facts_decayed = total_decayed,
                facts_deleted = total_deleted,
                "Fact maintenance complete"
            );
        }

        Ok((total_decayed, total_deleted))
    }
}

/// Automatic persistence on drop - ensures vector index and ID mappings survive restarts
///
/// This is CRITICAL for local memory: when the system shuts down (gracefully or via drop),
/// all in-memory state (vector index, ID mappings) must be persisted to disk.
impl Drop for MemorySystem {
    fn drop(&mut self) {
        // Vector index saved via explicit shutdown (save_all_vector_indices)
        // Do NOT save here - Drop fires for temporary instances, overwriting valid saves

        // Flush RocksDB WAL to ensure all writes are durable
        if let Err(e) = self.long_term_memory.flush() {
            tracing::error!("Failed to flush storage on shutdown: {}", e);
        }
    }
}
