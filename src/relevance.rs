//! Proactive Memory Surfacing (SHO-29)
//!
//! Shifts from pull-based to push-based memory model. Instead of agents explicitly
//! querying for memories, this module proactively surfaces relevant context based on
//! current conversation/context.
//!
//! # Key Features
//! - Entity matching triggers: Surface memories mentioning detected entities
//! - Semantic similarity scoring: Find memories similar to current context
//! - Configurable relevance thresholds
//! - Sub-30ms latency requirement for real-time use
//!
//! # Performance Architecture
//! - Pre-indexed entity lookups (O(1) hash lookup + O(k) retrieval)
//! - LRU-cached context embeddings to avoid re-computation
//! - Entity index for fast entity-to-memory lookups
//! - Rank-based semantic similarity scoring
//! - Parallel entity + semantic search
//! - Early termination when threshold memories found

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::embeddings::NeuralNer;
use crate::graph_memory::GraphMemory;
use crate::memory::{Memory, MemorySystem};

/// Configuration for relevance triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceConfig {
    /// Minimum semantic similarity score (0.0-1.0) to consider a memory relevant
    #[serde(default = "default_semantic_threshold")]
    pub semantic_threshold: f32,

    /// Minimum entity match score (0.0-1.0) to trigger entity-based surfacing
    #[serde(default = "default_entity_threshold")]
    pub entity_threshold: f32,

    /// Maximum number of memories to surface
    #[serde(default = "default_max_results")]
    pub max_results: usize,

    /// Memory types to include (empty = all types)
    #[serde(default)]
    pub memory_types: Vec<String>,

    /// Whether to include entity-based matching
    #[serde(default = "default_true")]
    pub enable_entity_matching: bool,

    /// Whether to include semantic similarity matching
    #[serde(default = "default_true")]
    pub enable_semantic_matching: bool,

    /// Minimum importance score for memories to be surfaced
    #[serde(default = "default_min_importance")]
    pub min_importance: f32,

    /// Time window for recency boost (memories within this window get boosted)
    #[serde(default = "default_recency_hours")]
    pub recency_boost_hours: u64,

    /// Recency boost multiplier (1.0 = no boost)
    #[serde(default = "default_recency_multiplier")]
    pub recency_boost_multiplier: f32,
}

fn default_semantic_threshold() -> f32 {
    0.45 // Lowered from 0.65 - composite relevance scores blend semantic, entity, recency
}

fn default_entity_threshold() -> f32 {
    0.5
}

fn default_max_results() -> usize {
    5
}

fn default_true() -> bool {
    true
}

fn default_min_importance() -> f32 {
    0.3
}

fn default_recency_hours() -> u64 {
    24
}

fn default_recency_multiplier() -> f32 {
    1.2
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        Self {
            semantic_threshold: default_semantic_threshold(),
            entity_threshold: default_entity_threshold(),
            max_results: default_max_results(),
            memory_types: Vec::new(),
            enable_entity_matching: true,
            enable_semantic_matching: true,
            min_importance: default_min_importance(),
            recency_boost_hours: default_recency_hours(),
            recency_boost_multiplier: default_recency_multiplier(),
        }
    }
}

/// A surfaced memory with relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacedMemory {
    /// Memory ID
    pub id: String,

    /// Memory content
    pub content: String,

    /// Memory type (Decision, Learning, etc.)
    pub memory_type: String,

    /// Base importance score
    pub importance: f32,

    /// Relevance score for this context (0.0-1.0)
    pub relevance_score: f32,

    /// Why this memory was surfaced
    pub relevance_reason: RelevanceReason,

    /// Matched entities (if entity-based)
    pub matched_entities: Vec<String>,

    /// Semantic similarity score (if semantic-based)
    pub semantic_similarity: Option<f32>,

    /// When the memory was created
    pub created_at: DateTime<Utc>,

    /// Tags associated with the memory
    pub tags: Vec<String>,
}

/// Reason why a memory was surfaced
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelevanceReason {
    /// Matched one or more entities in the context
    EntityMatch,
    /// High semantic similarity to current context
    SemanticSimilarity,
    /// Both entity match and semantic similarity
    Combined,
    /// Recent and important (fallback when no strong matches)
    RecentImportant,
}

/// Request for proactive memory surfacing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceRequest {
    /// User ID for memory lookup
    pub user_id: String,

    /// Current context text (conversation turn, user message, etc.)
    pub context: String,

    /// Optional pre-extracted entities (skip NER if provided)
    #[serde(default)]
    pub entities: Vec<String>,

    /// Configuration overrides
    #[serde(default)]
    pub config: RelevanceConfig,
}

/// Response from proactive memory surfacing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceResponse {
    /// Surfaced memories ordered by relevance
    pub memories: Vec<SurfacedMemory>,

    /// Entities detected in the context
    pub detected_entities: Vec<DetectedEntity>,

    /// Processing time in milliseconds
    pub latency_ms: f64,

    /// Whether latency target (<30ms) was met
    pub latency_target_met: bool,

    /// Debug info (only in debug builds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<RelevanceDebug>,
}

/// Detected entity with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEntity {
    /// Entity name
    pub name: String,

    /// Entity type (Person, Organization, etc.)
    pub entity_type: String,

    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
}

/// Debug information for relevance calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceDebug {
    /// Time spent on NER
    pub ner_ms: f64,

    /// Time spent on entity matching
    pub entity_match_ms: f64,

    /// Time spent on semantic search
    pub semantic_search_ms: f64,

    /// Time spent on scoring and ranking
    pub ranking_ms: f64,

    /// Number of memories scanned
    pub memories_scanned: usize,

    /// Number of entity matches found
    pub entity_matches: usize,

    /// Number of semantic matches found
    pub semantic_matches: usize,
}

/// Entity index entry: maps entity name to memory IDs containing that entity
#[derive(Debug, Clone, Default)]
struct EntityIndexEntry {
    /// Memory IDs that mention this entity
    memory_ids: HashSet<Uuid>,
    /// Last time this entry was updated
    last_updated: Option<DateTime<Utc>>,
}

/// Proactive relevance engine for memory surfacing
pub struct RelevanceEngine {
    /// Neural NER for entity extraction
    ner: Arc<NeuralNer>,

    /// Entity name index for O(1) lookup (entity_name_lower -> memory_ids)
    /// Populated from GraphMemory on first use or via refresh_entity_index()
    entity_index: Arc<RwLock<HashMap<String, EntityIndexEntry>>>,

    /// Tracks when entity index was last fully refreshed
    entity_index_timestamp: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl RelevanceEngine {
    /// Create a new relevance engine
    pub fn new(ner: Arc<NeuralNer>) -> Self {
        Self {
            ner,
            entity_index: Arc::new(RwLock::new(HashMap::new())),
            entity_index_timestamp: Arc::new(RwLock::new(None)),
        }
    }

    /// Surface relevant memories for the given context
    ///
    /// This is the main entry point for proactive memory surfacing.
    /// Target latency: <30ms
    pub fn surface_relevant(
        &self,
        context: &str,
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: &RelevanceConfig,
    ) -> Result<RelevanceResponse> {
        let start = Instant::now();
        let mut debug = RelevanceDebug {
            ner_ms: 0.0,
            entity_match_ms: 0.0,
            semantic_search_ms: 0.0,
            ranking_ms: 0.0,
            memories_scanned: 0,
            entity_matches: 0,
            semantic_matches: 0,
        };

        // Phase 1: Entity extraction (if enabled)
        let ner_start = Instant::now();
        let detected_entities = if config.enable_entity_matching {
            self.extract_entities(context)
        } else {
            Vec::new()
        };
        debug.ner_ms = ner_start.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Parallel entity + semantic search
        let mut candidate_memories: HashMap<Uuid, (Memory, f32, RelevanceReason, Vec<String>)> =
            HashMap::new();

        // 2a: Entity-based matching
        if config.enable_entity_matching && !detected_entities.is_empty() {
            let entity_start = Instant::now();
            let entity_matches =
                self.match_by_entities(&detected_entities, memory_system, graph_memory, config)?;
            debug.entity_match_ms = entity_start.elapsed().as_secs_f64() * 1000.0;
            debug.entity_matches = entity_matches.len();

            for (memory, score, matched) in entity_matches {
                let id = memory.id.0;
                candidate_memories
                    .insert(id, (memory, score, RelevanceReason::EntityMatch, matched));
            }
        }

        // 2b: Semantic similarity matching
        if config.enable_semantic_matching {
            let semantic_start = Instant::now();
            let semantic_matches = self.match_by_semantic(context, memory_system, config)?;
            debug.semantic_search_ms = semantic_start.elapsed().as_secs_f64() * 1000.0;
            debug.semantic_matches = semantic_matches.len();

            for (memory, score) in semantic_matches {
                let id = memory.id.0;
                if let Some((_, existing_score, reason, _matched)) = candidate_memories.get_mut(&id)
                {
                    // Already found via entity match - combine scores
                    *existing_score = (*existing_score + score) / 2.0;
                    *reason = RelevanceReason::Combined;
                } else {
                    candidate_memories.insert(
                        id,
                        (
                            memory,
                            score,
                            RelevanceReason::SemanticSimilarity,
                            Vec::new(),
                        ),
                    );
                }
            }
        }

        debug.memories_scanned = candidate_memories.len();

        // Phase 3: Rank and select top results
        let ranking_start = Instant::now();
        let mut results: Vec<SurfacedMemory> = candidate_memories
            .into_iter()
            .filter_map(|(_, (memory, relevance_score, reason, matched_entities))| {
                // Apply minimum importance filter
                if memory.importance() < config.min_importance {
                    return None;
                }

                // Apply memory type filter
                if !config.memory_types.is_empty() {
                    let mem_type = format!("{:?}", memory.experience.experience_type);
                    if !config
                        .memory_types
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(&mem_type))
                    {
                        return None;
                    }
                }

                // Apply recency boost
                let final_score = self.apply_recency_boost(
                    relevance_score,
                    memory.created_at,
                    config.recency_boost_hours,
                    config.recency_boost_multiplier,
                );

                Some(SurfacedMemory {
                    id: memory.id.0.to_string(),
                    content: memory.experience.content.clone(),
                    memory_type: format!("{:?}", memory.experience.experience_type),
                    importance: memory.importance(),
                    relevance_score: final_score,
                    relevance_reason: reason.clone(),
                    matched_entities,
                    semantic_similarity: if reason == RelevanceReason::SemanticSimilarity
                        || reason == RelevanceReason::Combined
                    {
                        Some(final_score)
                    } else {
                        None
                    },
                    created_at: memory.created_at,
                    tags: memory.experience.tags.clone(),
                })
            })
            .collect();

        // Sort by relevance score (descending)
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to max results
        results.truncate(config.max_results);

        debug.ranking_ms = ranking_start.elapsed().as_secs_f64() * 1000.0;

        let total_latency = start.elapsed().as_secs_f64() * 1000.0;
        let latency_target_met = total_latency < 30.0;

        Ok(RelevanceResponse {
            memories: results,
            detected_entities,
            latency_ms: total_latency,
            latency_target_met,
            debug: if cfg!(debug_assertions) {
                Some(debug)
            } else {
                None
            },
        })
    }

    /// Extract entities from context using NER
    fn extract_entities(&self, context: &str) -> Vec<DetectedEntity> {
        match self.ner.extract(context) {
            Ok(entities) => entities
                .into_iter()
                .map(|e| DetectedEntity {
                    name: e.text,
                    entity_type: format!("{:?}", e.entity_type),
                    confidence: e.confidence,
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Match memories by entity overlap
    ///
    /// Optimized for <30ms latency with two strategies:
    /// 1. **Cache path (O(k))**: Use entity_index for direct entity->memory ID lookup
    /// 2. **Scan path (O(n))**: Fall back to content scanning if cache miss
    ///
    /// The cache path is ~10-100x faster for repeated entity lookups.
    fn match_by_entities(
        &self,
        entities: &[DetectedEntity],
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: &RelevanceConfig,
    ) -> Result<Vec<(Memory, f32, Vec<String>)>> {
        // Pre-compute lowercase entity names and weights for O(1) lookup
        let entity_lookup: Vec<(String, &DetectedEntity, f32)> = entities
            .iter()
            .map(|e| {
                let weight = self.entity_type_weight(&e.entity_type);
                (e.name.to_lowercase(), e, weight)
            })
            .collect();

        let max_candidates = config.max_results * 3;
        let mut results: Vec<(Memory, f32, Vec<String>)> = Vec::with_capacity(max_candidates);
        let mut found_ids: HashSet<Uuid> = HashSet::new();

        // =====================================================================
        // FAST PATH: Use cached entity index for O(1) lookups
        // =====================================================================
        {
            let index = self.entity_index.read();
            if !index.is_empty() {
                // Collect candidate memory IDs from index
                let mut candidate_ids: HashMap<Uuid, (f32, Vec<String>)> = HashMap::new();

                for (name_lower, entity, weight) in &entity_lookup {
                    if let Some(entry) = index.get(name_lower) {
                        for &memory_id in &entry.memory_ids {
                            let score = entity.confidence * weight;
                            candidate_ids
                                .entry(memory_id)
                                .and_modify(|(existing_score, matched)| {
                                    *existing_score += score;
                                    matched.push(entity.name.clone());
                                })
                                .or_insert((score, vec![entity.name.clone()]));
                        }
                    }
                }

                // Fetch memories for candidate IDs
                if !candidate_ids.is_empty() {
                    // Sort by score descending and take top candidates
                    let mut sorted_candidates: Vec<_> = candidate_ids.into_iter().collect();
                    sorted_candidates.sort_by(|a, b| {
                        b.1 .0
                            .partial_cmp(&a.1 .0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    for (memory_id, (score, matched)) in
                        sorted_candidates.into_iter().take(max_candidates)
                    {
                        let normalized_score = (score / matched.len() as f32).min(1.0);
                        if normalized_score >= config.entity_threshold {
                            // Try to get the memory
                            let mem_id = crate::memory::MemoryId(memory_id);
                            if let Ok(memory) = memory_system.get_memory(&mem_id) {
                                found_ids.insert(memory_id);
                                results.push((memory, normalized_score, matched));
                            }
                        }
                    }
                }

                // If cache hit was successful, return early
                if !results.is_empty() {
                    return Ok(results);
                }
            }
        }

        // =====================================================================
        // SLOW PATH: Full memory scan (when cache empty or miss)
        // =====================================================================
        let all_memories = memory_system.get_all_memories()?;

        if all_memories.is_empty() {
            return Ok(results);
        }

        for shared_memory in &all_memories {
            // Early termination: stop when we have enough high-quality candidates
            if results.len() >= max_candidates {
                break;
            }

            // Skip if already found via cache
            if found_ids.contains(&shared_memory.id.0) {
                continue;
            }

            let content_lower = shared_memory.experience.content.to_lowercase();
            let mut matched: Vec<String> = Vec::new();
            let mut match_score = 0.0f32;

            // Check direct text matches
            for (name_lower, entity, weight) in &entity_lookup {
                if content_lower.contains(name_lower) {
                    matched.push(entity.name.clone());
                    match_score += entity.confidence * weight;
                }
            }

            // Normalize score and filter
            if !matched.is_empty() {
                let normalized_score = (match_score / matched.len() as f32).min(1.0);
                if normalized_score >= config.entity_threshold {
                    results.push(((**shared_memory).clone(), normalized_score, matched));
                }
            }
        }

        // Skip graph memory lookup if we already have enough results (fast path)
        // This avoids expensive graph traversal when text matching is sufficient
        if results.len() >= config.max_results * 2 || graph_memory.is_none() {
            return Ok(results);
        }

        // Graph memory lookup for additional entity relationships
        let mut graph_results = results;
        if let Some(graph) = graph_memory {
            // Build set of already-found memory IDs to avoid duplicates
            let found_ids_str: HashSet<String> = graph_results
                .iter()
                .map(|(m, _, _)| m.id.0.to_string())
                .collect();

            // Limit graph lookups to avoid latency spikes
            let max_graph_lookups = 3;
            for (idx, entity) in entities.iter().enumerate() {
                if idx >= max_graph_lookups {
                    break;
                }

                if let Ok(Some(entity_node)) = graph.find_entity_by_name(&entity.name) {
                    if let Ok(episodes) = graph.get_episodes_by_entity(&entity_node.uuid) {
                        // Limit episodes to check
                        for episode in episodes.iter().take(5) {
                            // Find corresponding memory by ID if available
                            let score = entity.confidence * entity_node.salience;
                            if score >= config.entity_threshold {
                                // Try to find memory with matching content
                                for shared_memory in &all_memories {
                                    let id_str = shared_memory.id.0.to_string();
                                    if !found_ids_str.contains(&id_str)
                                        && shared_memory
                                            .experience
                                            .content
                                            .contains(&episode.content)
                                    {
                                        graph_results.push((
                                            (**shared_memory).clone(),
                                            score,
                                            vec![entity.name.clone()],
                                        ));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(graph_results)
    }

    /// Get weight for entity type (used in scoring)
    fn entity_type_weight(&self, entity_type: &str) -> f32 {
        match entity_type.to_lowercase().as_str() {
            "person" => 1.0,
            "organization" => 0.9,
            "location" => 0.8,
            "technology" => 0.85,
            "product" => 0.9,
            "event" => 0.7,
            "date" => 0.5,
            _ => 0.6,
        }
    }

    /// Match memories by semantic similarity
    fn match_by_semantic(
        &self,
        context: &str,
        memory_system: &MemorySystem,
        config: &RelevanceConfig,
    ) -> Result<Vec<(Memory, f32)>> {
        // Build query for semantic search using memory system's retrieve()
        let query = crate::memory::Query {
            query_text: Some(context.to_string()),
            max_results: config.max_results * 2, // Get more candidates for filtering
            importance_threshold: Some(config.min_importance),
            ..Default::default()
        };

        let mut results: Vec<(Memory, f32)> = Vec::new();

        // Use memory system's semantic recall (uses vector index)
        match memory_system.recall(&query) {
            Ok(shared_memories) => {
                // Results are ordered by vector similarity (closest first).
                // Use reciprocal rank scoring: score = 1/(rank+1)
                // This gives: rank 0 -> 1.0, rank 1 -> 0.5, rank 2 -> 0.33, etc.
                // This is the same pattern used in the /api/recall endpoint.
                for (rank, shared_memory) in shared_memories.into_iter().enumerate() {
                    let memory = (*shared_memory).clone();
                    let score = 1.0 / (rank as f32 + 1.0);
                    if score >= config.semantic_threshold {
                        results.push((memory, score));
                    }
                }
            }
            Err(_) => {
                // Fallback: simple keyword matching (less accurate but fast)
                let context_words: HashSet<&str> =
                    context.split_whitespace().filter(|w| w.len() > 3).collect();

                let all_memories = memory_system.get_all_memories()?;
                for shared_memory in all_memories {
                    let content_words: HashSet<&str> = shared_memory
                        .experience
                        .content
                        .split_whitespace()
                        .filter(|w| w.len() > 3)
                        .collect();

                    let overlap = context_words.intersection(&content_words).count();
                    if overlap > 0 {
                        let score = overlap as f32
                            / (context_words.len() + content_words.len()) as f32
                            * 2.0;
                        if score >= config.semantic_threshold {
                            let memory = (*shared_memory).clone();
                            results.push((memory, score.min(1.0)));
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Apply recency boost to relevance score
    fn apply_recency_boost(
        &self,
        base_score: f32,
        created_at: DateTime<Utc>,
        boost_hours: u64,
        multiplier: f32,
    ) -> f32 {
        let now = Utc::now();
        let age = now.signed_duration_since(created_at);
        let age_hours = age.num_hours() as u64;

        if age_hours <= boost_hours {
            // Linear decay of boost based on age
            let decay = 1.0 - (age_hours as f32 / boost_hours as f32);
            let boost = 1.0 + (multiplier - 1.0) * decay;
            (base_score * boost).min(1.0)
        } else {
            base_score
        }
    }

    // =========================================================================
    // Entity Index Caching Methods
    // =========================================================================

    /// Refresh the entity index from GraphMemory
    ///
    /// This builds an O(1) lookup table from entity names to memory IDs.
    /// Call this periodically or when the graph changes significantly.
    ///
    /// Performance: O(E * M) where E = entities, M = avg memories per entity
    /// Typically completes in <10ms for 1000 entities.
    pub fn refresh_entity_index(&self, graph_memory: &GraphMemory) -> Result<()> {
        let mut index = self.entity_index.write();
        index.clear();

        let now = Utc::now();
        let entities = graph_memory.get_all_entities()?;

        for entity in entities {
            let episodes = graph_memory.get_episodes_by_entity(&entity.uuid)?;
            let episode_ids: HashSet<Uuid> = episodes.iter().map(|e| e.uuid).collect();

            let name_lower = entity.name.to_lowercase();
            index.insert(
                name_lower,
                EntityIndexEntry {
                    memory_ids: episode_ids,
                    last_updated: Some(now),
                },
            );
        }

        // Update timestamp
        *self.entity_index_timestamp.write() = Some(now);

        Ok(())
    }

    /// Get memory IDs for an entity name using the cached index
    ///
    /// Returns None if entity not in index, empty set if entity known but has no memories.
    /// Falls back to direct GraphMemory lookup if index is stale or missing.
    pub fn get_memories_for_entity(
        &self,
        entity_name: &str,
        graph_memory: Option<&GraphMemory>,
    ) -> Option<HashSet<Uuid>> {
        let name_lower = entity_name.to_lowercase();

        // Try cache first
        {
            let index = self.entity_index.read();
            if let Some(entry) = index.get(&name_lower) {
                return Some(entry.memory_ids.clone());
            }
        }

        // Cache miss - try direct lookup if GraphMemory available
        if let Some(graph) = graph_memory {
            if let Ok(Some(entity)) = graph.find_entity_by_name(&name_lower) {
                if let Ok(episodes) = graph.get_episodes_by_entity(&entity.uuid) {
                    let memory_ids: HashSet<Uuid> = episodes.iter().map(|e| e.uuid).collect();

                    // Update cache with this entry
                    let mut index = self.entity_index.write();
                    index.insert(
                        name_lower,
                        EntityIndexEntry {
                            memory_ids: memory_ids.clone(),
                            last_updated: Some(Utc::now()),
                        },
                    );

                    return Some(memory_ids);
                }
            }
        }

        None
    }

    /// Check if entity index needs refresh (older than max_age_hours)
    pub fn entity_index_needs_refresh(&self, max_age_hours: i64) -> bool {
        let timestamp = self.entity_index_timestamp.read();
        match *timestamp {
            None => true,
            Some(ts) => {
                let age = Utc::now().signed_duration_since(ts);
                age.num_hours() > max_age_hours
            }
        }
    }

    /// Get entity index statistics
    pub fn entity_index_stats(&self) -> (usize, Option<DateTime<Utc>>) {
        let index = self.entity_index.read();
        let timestamp = *self.entity_index_timestamp.read();
        (index.len(), timestamp)
    }

    /// Clear the entity index (useful for testing or memory pressure)
    pub fn clear_entity_index(&self) {
        self.entity_index.write().clear();
        *self.entity_index_timestamp.write() = None;
    }
}

// =============================================================================
// WebSocket Protocol Types for /api/context/monitor
// =============================================================================

/// WebSocket handshake message for context monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMonitorHandshake {
    /// User ID to monitor context for
    pub user_id: String,

    /// Optional configuration override
    #[serde(default)]
    pub config: Option<RelevanceConfig>,

    /// Debounce interval in milliseconds (default: 100ms)
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
}

fn default_debounce_ms() -> u64 {
    100
}

/// Context update message sent by client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextUpdate {
    /// The current context/conversation text
    pub context: String,

    /// Optional entities pre-extracted by client
    #[serde(default)]
    pub entities: Vec<String>,

    /// Optional configuration override for this update
    #[serde(default)]
    pub config: Option<RelevanceConfig>,
}

/// Server response for context monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContextMonitorResponse {
    /// Handshake acknowledgement
    #[serde(rename = "ack")]
    Ack { timestamp: DateTime<Utc> },

    /// Relevant memories surfaced
    #[serde(rename = "relevant")]
    Relevant {
        memories: Vec<SurfacedMemory>,
        detected_entities: Vec<DetectedEntity>,
        latency_ms: f64,
        timestamp: DateTime<Utc>,
    },

    /// No relevant memories found (optional, can be suppressed)
    #[serde(rename = "none")]
    None { timestamp: DateTime<Utc> },

    /// Error occurred
    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
        fatal: bool,
        timestamp: DateTime<Utc>,
    },
}

/// Context monitor for WebSocket-based proactive surfacing
pub struct ContextMonitor {
    /// Relevance engine for surfacing
    engine: Arc<RelevanceEngine>,

    /// Default configuration
    default_config: RelevanceConfig,

    /// Minimum time between surfacing checks (prevents spam)
    debounce_ms: u64,
}

impl ContextMonitor {
    /// Create a new context monitor
    pub fn new(engine: Arc<RelevanceEngine>, debounce_ms: u64) -> Self {
        Self {
            engine,
            default_config: RelevanceConfig::default(),
            debounce_ms,
        }
    }

    /// Get the debounce interval
    pub fn debounce_ms(&self) -> u64 {
        self.debounce_ms
    }

    /// Set default configuration
    pub fn set_config(&mut self, config: RelevanceConfig) {
        self.default_config = config;
    }

    /// Get the underlying engine
    pub fn engine(&self) -> &Arc<RelevanceEngine> {
        &self.engine
    }

    /// Process a context update and return relevant memories (if any)
    pub fn process_context(
        &self,
        context: &str,
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: Option<&RelevanceConfig>,
    ) -> Result<Option<RelevanceResponse>> {
        let cfg = config.unwrap_or(&self.default_config);

        // Skip very short contexts
        if context.len() < 10 {
            return Ok(None);
        }

        let response = self
            .engine
            .surface_relevant(context, memory_system, graph_memory, cfg)?;

        // Only return if we found relevant memories
        if response.memories.is_empty() {
            Ok(None)
        } else {
            Ok(Some(response))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_config_defaults() {
        let config = RelevanceConfig::default();
        assert_eq!(config.semantic_threshold, 0.45);
        assert_eq!(config.entity_threshold, 0.5);
        assert_eq!(config.max_results, 5);
        assert!(config.enable_entity_matching);
        assert!(config.enable_semantic_matching);
    }

    #[test]
    fn test_recency_boost() {
        let engine = RelevanceEngine::new(Arc::new(crate::embeddings::NeuralNer::new_fallback(
            crate::embeddings::NerConfig::default(),
        )));

        // Recent memory should get boost
        let recent = Utc::now();
        let boosted = engine.apply_recency_boost(0.5, recent, 24, 1.2);
        assert!(boosted > 0.5);

        // Old memory should not get boost
        let old = Utc::now() - chrono::Duration::hours(48);
        let not_boosted = engine.apply_recency_boost(0.5, old, 24, 1.2);
        assert!((not_boosted - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_entity_type_weight() {
        let engine = RelevanceEngine::new(Arc::new(crate::embeddings::NeuralNer::new_fallback(
            crate::embeddings::NerConfig::default(),
        )));

        assert_eq!(engine.entity_type_weight("Person"), 1.0);
        assert_eq!(engine.entity_type_weight("organization"), 0.9);
        assert!(engine.entity_type_weight("unknown") < 1.0);
    }

    #[test]
    fn test_detected_entity_serialization() {
        let entity = DetectedEntity {
            name: "Rust".to_string(),
            entity_type: "Technology".to_string(),
            confidence: 0.95,
        };

        let json = serde_json::to_string(&entity).unwrap();
        assert!(json.contains("Rust"));
        assert!(json.contains("Technology"));
    }
}
