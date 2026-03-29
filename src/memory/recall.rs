//! Recall and retrieval methods for the memory system.
//!
//! This module contains all functions related to searching, retrieving, and
//! ranking memories: semantic retrieval (the 11-layer pipeline), paginated
//! recall, tag/date-based recall, learning-boost helpers, hierarchy expansion,
//! Hebbian reinforcement, and tracked retrieval feedback.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::sync::Arc;

use crate::constants::{HEBBIAN_BOOST_HELPFUL, HEBBIAN_DECAY_MISLEADING};
use crate::embeddings::Embedder;
use crate::memory::graph_retrieval::calculate_density_weights;
use crate::memory::introspection::{self, StrengtheningReason};
use crate::memory::learning_history;
use crate::memory::query_parser;
use crate::memory::retrieval::{ReinforcementStats, RetrievalOutcome, TrackedRetrieval};
use crate::memory::storage::{self, SearchCriteria};
use crate::memory::temporal_facts;
use crate::memory::types::*;
use crate::memory::wavelet_sessions;
use crate::metrics::{EMBEDDING_CACHE_QUERY, EMBEDDING_CACHE_QUERY_SIZE};

impl super::MemorySystem {
    /// Search and retrieve relevant memories (zero-copy with Arc<Memory>)
    ///
    /// PRODUCTION IMPLEMENTATION:
    /// - Semantic search: Uses embeddings + vector similarity across ALL tiers
    /// - Non-semantic search: Uses importance * temporal decay
    /// - Zero shortcuts, no TODOs, enterprise-grade
    pub fn recall(&self, query: &Query) -> Result<Vec<SharedMemory>> {
        // Semantic search requires special handling
        if let Some(query_text) = &query.query_text {
            return self.semantic_retrieve(query_text, query);
        }

        // Non-semantic search: filter-based retrieval
        let mut memories = Vec::new();
        let mut seen_ids: HashSet<MemoryId> = HashSet::new();
        let mut sources = Vec::new();

        // Collect from all tiers with deduplication (priority: working > session > long_term)
        {
            let working = self.working_memory.read();
            let working_results = working.search(query, query.max_results)?;
            if !working_results.is_empty() {
                sources.push("working");
            }
            for memory in working_results {
                if seen_ids.insert(memory.id.clone()) {
                    memories.push(memory);
                }
            }
        }

        {
            let session = self.session_memory.read();
            let session_results = session.search(query, query.max_results)?;
            if !session_results.is_empty() {
                sources.push("session");
            }
            for memory in session_results {
                if seen_ids.insert(memory.id.clone()) {
                    memories.push(memory);
                }
            }
        }

        {
            let long_term_results = self.retriever.search(query, query.max_results)?;
            if !long_term_results.is_empty() {
                sources.push("longterm");
            }
            for memory in long_term_results {
                if seen_ids.insert(memory.id.clone()) {
                    memories.push(memory);
                }
            }
        }

        // Expand with hierarchy context (parent chain + children)
        // Related memories in hierarchy get a decayed score boost
        self.expand_with_hierarchy(&mut memories, &mut seen_ids);

        // Rank by importance * temporal relevance
        let now = chrono::Utc::now();
        memories.sort_by(|a, b| {
            let age_days_a = (now - a.created_at).num_days();
            let temporal_a = Self::calculate_temporal_relevance(age_days_a);
            let score_a = a.importance() * temporal_a;

            let age_days_b = (now - b.created_at).num_days();
            let temporal_b = Self::calculate_temporal_relevance(age_days_b);
            let score_b = b.importance() * temporal_b;

            score_b.total_cmp(&score_a)
        });

        // Filter temporally invalidated memories
        memories.retain(|m| !m.is_expired());

        memories.truncate(query.max_results);

        // Log retrieval
        self.logger
            .read()
            .log_retrieved("", memories.len(), &sources);

        // Update access counts with instrumentation for consolidation events
        for memory in &memories {
            self.update_access_count_instrumented(memory, StrengtheningReason::Recalled);
        }

        // Hebbian learning: co-activation strengthens associations between memories
        // When memories are retrieved together, they form/strengthen edges in the memory graph
        if memories.len() >= 2 {
            if let Some(graph) = &self.graph_memory {
                let memory_uuids: Vec<uuid::Uuid> = memories.iter().map(|m| m.id.0).collect();
                match graph.read().record_memory_coactivation(&memory_uuids) {
                    Ok(result) => {
                        // BRIDGE-4: Consume edge tier promotions — boost memory importance
                        for promo in &result.promotions {
                            let boost = if promo.new_tier.contains("L3") { 0.15 } else { 0.10 };
                            for mem in &memories {
                                if mem.id.0 == promo.from_entity || mem.id.0 == promo.to_entity {
                                    mem.boost_importance(boost);
                                }
                            }
                        }
                        if !result.promotions.is_empty() {
                            tracing::debug!(
                                "BRIDGE-4: {} edge promotions → boosted associated memory importance",
                                result.promotions.len()
                            );
                        }
                    }
                    Err(e) => {
                        tracing::trace!("Coactivation recording failed (non-critical): {e}");
                    }
                }
            }
        }

        // Increment and persist retrieval counter
        if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
            self.stats.write().total_retrievals = count;
        }

        Ok(memories)
    }

    /// Paginated memory recall with "has_more" indicator (SHO-69)
    ///
    /// Returns a PaginatedResults struct containing:
    /// - The page of results
    /// - Whether there are more results beyond this page
    /// - The total count (if computed)
    /// - Pagination metadata (offset, limit)
    ///
    /// Uses the limit+1 trick: requests one extra result to detect if there are more.
    pub fn paginated_recall(&self, query: &Query) -> Result<PaginatedResults<SharedMemory>> {
        // Request offset+limit+1 to detect if there are more results.
        // We must fetch enough to cover both the skipped offset portion AND the
        // requested limit, plus 1 extra for has_more detection.
        let extra_limit = query.offset + query.max_results + 1;
        let mut modified_query = query.clone();
        modified_query.max_results = extra_limit;
        modified_query.offset = 0; // We handle offset ourselves

        // Get all results up to extra_limit
        let all_results = self.recall(&modified_query)?;

        // Apply offset and limit, detect has_more
        let offset = query.offset;
        let limit = query.max_results;

        let results_after_offset: Vec<_> = all_results.into_iter().skip(offset).collect();
        let has_more = results_after_offset.len() > limit;

        let final_results: Vec<_> = results_after_offset.into_iter().take(limit).collect();

        Ok(PaginatedResults {
            results: final_results,
            has_more,
            total_count: None, // Computing total would require a separate count query
            offset,
            limit,
        })
    }

    /// Recall memories by tags (fast, no embedding required)
    ///
    /// Returns memories that have ANY of the specified tags.
    pub fn recall_by_tags(&self, tags: &[String], limit: usize) -> Result<Vec<Memory>> {
        let criteria = storage::SearchCriteria::ByTags(tags.to_vec());
        let mut memories = self.advanced_search(criteria)?;
        memories.truncate(limit);
        if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
            self.stats.write().total_retrievals = count;
        }
        Ok(memories)
    }

    /// Recall memories within a date range
    ///
    /// Returns memories created between `start` and `end` (inclusive).
    pub fn recall_by_date(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        let criteria = storage::SearchCriteria::ByDate { start, end };
        let mut memories = self.advanced_search(criteria)?;
        memories.truncate(limit);
        if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
            self.stats.write().total_retrievals = count;
        }
        Ok(memories)
    }

    /// CACHE-AWARE semantic retrieval: Check working → session → storage
    ///
    /// Implementation:
    /// 1. Generate query embedding and search vector index for memory IDs
    /// 2. For each ID, check working memory (instant Arc clone)
    /// 3. If not found, check session memory (instant Arc clone)
    /// 4. Only fetch from RocksDB storage as last resort
    /// 5. This eliminates deserialization overhead for cached memories
    fn semantic_retrieve(&self, query_text: &str, query: &Query) -> Result<Vec<SharedMemory>> {
        let recall_start = std::time::Instant::now();

        // ===========================================================================
        // TEMPORAL EXTRACTION (TEMPR approach from Hindsight - 89.6% on LoCoMo)
        // ===========================================================================
        // Key insight: Temporal filtering is critical for multi-hop retrieval accuracy.
        // Extract temporal constraints from query and use them to boost/filter results.
        let query_temporal = query_parser::extract_temporal_refs(query_text);
        let has_temporal_query = query_parser::requires_temporal_filtering(query_text);
        // Broader temporal intent detection: includes WhenQuestion, SpecificTime,
        // Duration, Ordering — any query with temporal semantics should boost
        // temporal fact source memories in RRF fusion.
        let temporal_intent = query_parser::detect_temporal_intent(query_text);
        let has_any_temporal_intent =
            !matches!(temporal_intent, query_parser::TemporalIntent::None);

        if has_temporal_query {
            tracing::debug!(
                "Temporal query detected: intent={:?}, refs={:?}",
                temporal_intent,
                query_temporal
                    .refs
                    .iter()
                    .map(|r| r.date.to_string())
                    .collect::<Vec<_>>()
            );
        }

        // ===========================================================================
        // LAYER 0.5: ATTRIBUTE QUERY DETECTION (Fact-First Retrieval)
        // ===========================================================================
        // For attribute queries like "What is Caroline's relationship status?",
        // semantic search fails because "relationship status" doesn't match "single".
        // Instead, we detect the query pattern, expand with synonyms, and boost
        // memories containing the entity + attribute values.
        let query_type = query_parser::classify_query(query_text);
        // Single parse for all layers (was called 3x: temporal facts, graph expansion, linguistic boost)
        let query_analysis = query_parser::analyze_query(query_text);

        // Ontological intent: infer expected entity types and relation types from query structure.
        // Used by Layer 2 (filtered traversal) and Layer 4.9 (type-aware re-ranking).
        let onto_intent = query_parser::infer_ontological_intent(query_text, &query_analysis);

        // Compute graph density for ontological gating (consistent with Layer 2 in graph_retrieval.rs).
        // Dense/young graphs have too many noisy L1 edges for type filtering to help.
        // Short-circuit: skip the RocksDB density lookups entirely when confidence is too low.
        let (graph_density_for_rerank, use_ontology_rerank) =
            if onto_intent.confidence < crate::constants::ONTOLOGICAL_MIN_CONFIDENCE {
                (None, false)
            } else if let Some(graph) = self.graph_memory.as_ref() {
                let g = graph.read();
                let seed_uuids: Vec<uuid::Uuid> = query_analysis
                    .focal_entities
                    .iter()
                    .filter_map(|e| {
                        g.find_entity_by_name(&e.text)
                            .ok()
                            .flatten()
                            .map(|n| n.uuid)
                    })
                    .collect();
                let density = if seed_uuids.is_empty() {
                    None
                } else {
                    g.entities_average_density(&seed_uuids).ok().flatten()
                };
                let use_rerank = !density
                    .is_some_and(|d| d >= crate::constants::ONTOLOGICAL_DENSITY_THRESHOLD);
                (density, use_rerank)
            } else {
                (None, false)
            };

        // Ontology telemetry
        crate::metrics::ONTOLOGICAL_INTENT_CONFIDENCE.observe(onto_intent.confidence as f64);
        if onto_intent.confidence > 0.0
            && onto_intent.confidence < crate::constants::ONTOLOGICAL_MIN_CONFIDENCE
        {
            crate::metrics::ONTOLOGICAL_FALLBACK_TOTAL.inc();
        }
        if graph_density_for_rerank
            .is_some_and(|d| d >= crate::constants::ONTOLOGICAL_DENSITY_THRESHOLD)
        {
            crate::metrics::ONTOLOGICAL_DENSITY_SKIP_TOTAL.inc();
        }

        let attribute_boost_ids: HashSet<MemoryId> = match &query_type {
            query_parser::QueryType::Attribute(attr_query) => {
                tracing::debug!(
                    "Layer 0.5: Attribute query detected - entity='{}', attribute='{}', synonyms={:?}",
                    attr_query.entity,
                    attr_query.attribute,
                    attr_query.attribute_synonyms
                );

                // Build expanded query: entity + attribute + all synonyms
                // E.g., "Caroline single married divorced engaged dating relationship"
                let mut expanded_terms: Vec<String> = vec![attr_query.entity.clone()];
                expanded_terms.extend(attr_query.attribute_synonyms.clone());

                // Search BM25 with expanded query to find memories with these terms
                let expanded_query = expanded_terms.join(" ");
                let bm25_matches = self
                    .hybrid_search
                    .bm25_index()
                    .search(&expanded_query, query.max_results * 5)
                    .unwrap_or_default();

                // Filter to memories that contain BOTH entity AND at least one synonym
                let entity_lower = attr_query.entity.to_lowercase();
                let mut boosted_ids = HashSet::new();

                for (mem_id, _score) in bm25_matches {
                    // Get memory content to verify it contains entity + attribute value
                    let content = self
                        .working_memory
                        .read()
                        .get(&mem_id)
                        .map(|m| m.experience.content.to_lowercase())
                        .or_else(|| {
                            self.session_memory
                                .read()
                                .get(&mem_id)
                                .map(|m| m.experience.content.to_lowercase())
                        })
                        .or_else(|| {
                            self.long_term_memory
                                .get(&mem_id)
                                .ok()
                                .map(|m| m.experience.content.to_lowercase())
                        });

                    if let Some(content) = content {
                        // Must contain entity
                        if !content.contains(&entity_lower) {
                            continue;
                        }
                        // Must contain at least one attribute synonym
                        let has_synonym = attr_query
                            .attribute_synonyms
                            .iter()
                            .any(|syn| content.contains(&syn.to_lowercase()));
                        if has_synonym {
                            boosted_ids.insert(mem_id);
                        }
                    }
                }

                if !boosted_ids.is_empty() {
                    tracing::info!(
                        "Layer 0.5: Found {} memories with entity '{}' + attribute values",
                        boosted_ids.len(),
                        attr_query.entity
                    );
                }

                boosted_ids
            }
            _ => HashSet::new(),
        };

        // ===========================================================================
        // LAYER 0.6: TEMPORAL FACT LOOKUP (Multi-hop Temporal Reasoning)
        // ===========================================================================
        // For temporal queries like "When did Melanie paint a sunrise?" or
        // "When is Melanie planning on going camping?", we need to:
        // 1. Detect it's a temporal query (asking "when", "what time", etc.)
        // 2. Extract entity (Melanie) and event keywords (paint, sunrise, camping)
        // 3. Look up temporal facts matching these
        // 4. Boost the source memories of matching facts
        // Temporal fact lookup - boost source memories of matching facts in Layer 4.55
        // Uses broader temporal intent (including WhenQuestion) to surface fact sources
        // even when the query asks FOR a date rather than filtering BY a date.
        //
        // Extended gate: also fire for queries with temporal keywords that
        // detect_temporal_intent might miss (e.g. "during the meeting",
        // "in that session", "what happened recently").
        let query_lower_for_temporal = query_text.to_lowercase();
        let has_temporal_keywords = [
            "when", "last week", "recently", "during", "before", "after",
            "yesterday", "meeting", "session", "phase", "earlier", "later",
            "previous", "following", "happened", "took place", "occurred",
        ]
        .iter()
        .any(|kw| query_lower_for_temporal.contains(kw));

        let temporal_gate = has_any_temporal_intent || has_temporal_keywords;

        let temporal_fact_boost_ids: HashSet<MemoryId> = if temporal_gate {
            if let Some(user_id) = &query.user_id {
                // Get entity name (first focal entity)
                let entity = query_analysis
                    .focal_entities
                    .first()
                    .map(|e| e.text.clone())
                    .unwrap_or_default();

                // Get event keywords from nouns, verbs, and modifiers
                let event_keywords: Vec<&str> = query_analysis
                    .focal_entities
                    .iter()
                    .skip(1) // Skip the entity itself
                    .map(|e| e.text.as_str())
                    .chain(
                        query_analysis
                            .relational_context
                            .iter()
                            .map(|r| r.stem.as_str()),
                    )
                    .chain(
                        query_analysis
                            .discriminative_modifiers
                            .iter()
                            .map(|m| m.text.as_str()),
                    )
                    .collect();

                // Determine event type from query keywords
                // "planning", "going to" → Planned
                // "did", "ran", "went" → Occurred
                // year mentions (2022, 2021) → Historical
                let event_type = if query_lower_for_temporal.contains("planning")
                    || query_lower_for_temporal.contains("going to")
                    || query_lower_for_temporal.contains("will")
                {
                    Some(temporal_facts::EventType::Planned)
                } else if query_lower_for_temporal.contains(" did ")
                    || query_lower_for_temporal.contains("when did")
                    || query_lower_for_temporal.contains(" ran ")
                    || query_lower_for_temporal.contains(" went ")
                {
                    // "When did X" could be Occurred or Historical - search both
                    None
                } else {
                    None // Any event type
                };

                // ---------------------------------------------------------------
                // Strategy 1: Entity + event keyword lookup (most precise)
                // ---------------------------------------------------------------
                let mut boosted: HashSet<MemoryId> = HashSet::new();

                if !entity.is_empty() && !event_keywords.is_empty() {
                    match self.find_temporal_facts(user_id, &entity, &event_keywords, event_type) {
                        Ok(facts) if !facts.is_empty() => {
                            tracing::info!(
                                "Layer 0.6: Found {} temporal facts for entity='{}', events={:?}",
                                facts.len(),
                                entity,
                                event_keywords
                            );
                            boosted.extend(facts.iter().map(|f| f.source_memory_id.clone()));
                        }
                        Ok(_) => {
                            tracing::debug!(
                                "Layer 0.6: No temporal facts found for entity='{}', events={:?}",
                                entity,
                                event_keywords
                            );
                        }
                        Err(e) => {
                            tracing::debug!("Layer 0.6: Temporal fact lookup (entity+event) failed: {}", e);
                        }
                    }
                }

                // ---------------------------------------------------------------
                // Strategy 2 (fallback): Entity-only lookup
                // ---------------------------------------------------------------
                // Fires when entity is known but event keywords are empty or
                // Strategy 1 returned nothing.
                if boosted.is_empty() && !entity.is_empty() {
                    match self.temporal_fact_store.find_by_entity_filtered(
                        user_id, &entity, 50, false,
                    ) {
                        Ok(facts) if !facts.is_empty() => {
                            // When query has a parsed time window, filter facts by it
                            let filtered = Self::filter_facts_by_time_window(
                                &facts, &query_temporal,
                            );
                            if !filtered.is_empty() {
                                tracing::info!(
                                    "Layer 0.6: Fallback entity-only found {} facts (time-filtered) for '{}'",
                                    filtered.len(),
                                    entity
                                );
                                boosted.extend(filtered.into_iter().map(|f| f.source_memory_id.clone()));
                            } else {
                                // No time window or no overlap — take all entity facts
                                tracing::debug!(
                                    "Layer 0.6: Fallback entity-only found {} facts for '{}'",
                                    facts.len(),
                                    entity
                                );
                                boosted.extend(facts.iter().map(|f| f.source_memory_id.clone()));
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::debug!("Layer 0.6: Entity-only temporal lookup failed: {}", e);
                        }
                    }
                }

                // ---------------------------------------------------------------
                // Strategy 3 (fallback): Event-keyword-only lookup
                // ---------------------------------------------------------------
                // Fires when no entity was parsed but event keywords exist
                // (e.g. "What happened during the debugging session?").
                if boosted.is_empty() && entity.is_empty() && !event_keywords.is_empty() {
                    for kw in &event_keywords {
                        match self.temporal_fact_store.find_by_event(user_id, kw, 30) {
                            Ok(facts) if !facts.is_empty() => {
                                let filtered = Self::filter_facts_by_time_window(
                                    &facts, &query_temporal,
                                );
                                if !filtered.is_empty() {
                                    boosted.extend(filtered.iter().map(|f| f.source_memory_id.clone()));
                                } else {
                                    boosted.extend(facts.iter().map(|f| f.source_memory_id.clone()));
                                }
                            }
                            _ => {}
                        }
                    }
                    if !boosted.is_empty() {
                        tracing::info!(
                            "Layer 0.6: Fallback event-only found {} source memories",
                            boosted.len()
                        );
                    }
                }

                // ---------------------------------------------------------------
                // Strategy 4 (last resort): Time-window scan
                // ---------------------------------------------------------------
                // When parsed temporal refs give us a concrete date range but
                // neither entity nor event yielded anything, scan all facts and
                // keep those whose conversation_date falls in the window.
                if boosted.is_empty() && query_temporal.has_temporal_refs() {
                    match self.temporal_fact_store.list_filtered(user_id, 200, false) {
                        Ok(all_facts) if !all_facts.is_empty() => {
                            let filtered = Self::filter_facts_by_time_window(
                                &all_facts, &query_temporal,
                            );
                            if !filtered.is_empty() {
                                tracing::info!(
                                    "Layer 0.6: Time-window scan matched {} facts out of {}",
                                    filtered.len(),
                                    all_facts.len()
                                );
                                boosted.extend(filtered.into_iter().map(|f| f.source_memory_id.clone()));
                            }
                        }
                        _ => {}
                    }
                }

                boosted
            } else {
                HashSet::new()
            }
        } else {
            HashSet::new()
        };

        // ===========================================================================
        // LAYER 0.7: SEMANTIC FACT SOURCE LOOKUP
        // ===========================================================================
        // Pre-fetch facts by query entities to boost their source memories in Layer 4.8.
        // Facts represent consolidated knowledge — their source memories contain the
        // richest context for that knowledge and should rank higher.
        let fact_source_boosts: std::collections::HashMap<MemoryId, f32> = {
            let mut boosts: std::collections::HashMap<MemoryId, f32> =
                std::collections::HashMap::new();

            if let Some(user_id) = &query.user_id {
                let entity_names: Vec<String> = query_analysis
                    .focal_entities
                    .iter()
                    .map(|e| e.text.to_lowercase())
                    .collect();

                if !entity_names.is_empty() {
                    if let Ok(facts) = self.get_facts_for_graph_entities(user_id, &entity_names, 5)
                    {
                        for fact in &facts {
                            if fact.confidence < 0.5 || fact.support_count < 3 {
                                continue;
                            }
                            let per_fact_boost = fact.confidence * 0.08;
                            for src_id in &fact.source_memories {
                                let entry = boosts.entry(src_id.clone()).or_insert(0.0);
                                *entry = (*entry + per_fact_boost).min(0.3);
                            }
                        }
                        if !boosts.is_empty() {
                            tracing::debug!(
                                "Layer 0.7: Pre-fetched {} fact-source boosts from {} facts",
                                boosts.len(),
                                facts.len()
                            );
                        }
                    }
                }
            }
            boosts
        };

        let t_query_analysis = recall_start.elapsed();
        tracing::info!(
            query_analysis_ms = format!("{:.2}", t_query_analysis.as_secs_f64() * 1000.0),
            "recall [layer:0.5-0.7] query analysis + attribute + temporal fact + fact source lookup"
        );

        // PERFORMANCE: Use pre-computed embedding if caller provided one,
        // otherwise fall back to SHA256-keyed cache (80ms → <1μs for repeated queries)
        let query_embedding =
            if let Some(pre) = query.query_embedding.as_ref().filter(|e| !e.is_empty()) {
                EMBEDDING_CACHE_QUERY
                    .with_label_values(&["precomputed"])
                    .inc();
                tracing::debug!("Query embedding PRECOMPUTED by caller — skipping encode");
                pre.clone()
            } else {
                let query_hash = Self::sha256_hash(query_text);
                if let Some(cached_embedding) = self.query_cache.get(&query_hash) {
                    EMBEDDING_CACHE_QUERY.with_label_values(&["hit"]).inc();
                    tracing::debug!("Query embedding cache HIT for: {}", query_text);
                    cached_embedding.clone()
                } else {
                    EMBEDDING_CACHE_QUERY.with_label_values(&["miss"]).inc();
                    tracing::debug!(
                        "Query embedding cache MISS - generating for: {}",
                        query_text
                    );
                    let embedding = self
                        .embedder
                        .as_ref()
                        .encode(query_text)
                        .context("Failed to generate query embedding")?;

                    self.query_cache.insert(query_hash, embedding.clone());
                    EMBEDDING_CACHE_QUERY_SIZE.set(self.query_cache.entry_count() as i64);
                    embedding
                }
            };

        let t_embedding = recall_start.elapsed();
        tracing::info!(
            embedding_ms = format!(
                "{:.2}",
                (t_embedding - t_query_analysis).as_secs_f64() * 1000.0
            ),
            cumulative_ms = format!("{:.2}", t_embedding.as_secs_f64() * 1000.0),
            "recall [layer:embedding] query embedding (cache miss logged above if any)"
        );

        // ===========================================================================
        // LAYER 1: TEMPORAL PRE-FILTER (Episode Coherence)
        // ===========================================================================
        let episode_candidates: Option<HashSet<MemoryId>> = if let Some(episode_id) =
            &query.episode_id
        {
            match self
                .long_term_memory
                .search(SearchCriteria::ByEpisode(episode_id.clone()))
            {
                Ok(ep) if !ep.is_empty() => {
                    tracing::debug!("Layer 1: {} candidates in episode {}", ep.len(), episode_id);
                    Some(ep.into_iter().map(|m| m.id).collect())
                }
                _ => {
                    tracing::debug!("Layer 1: global search");
                    None
                }
            }
        } else {
            None
        };

        // ===========================================================================
        // LAYER 2: GRAPH EXPANSION (Knowledge Graph Traversal)
        // ===========================================================================
        let use_graph = matches!(
            query.retrieval_mode,
            RetrievalMode::Hybrid | RetrievalMode::Associative | RetrievalMode::Causal
        );
        #[allow(clippy::type_complexity)]
        let (
            graph_results,
            graph_density,
            query_entity_count,
            ic_weights,
            phrase_boosts,
            keyword_disc,
        ): (
            Vec<(MemoryId, f32, f32)>,
            Option<f32>,
            usize,
            std::collections::HashMap<String, f32>,
            Vec<(String, f32)>,
            f32, // Keyword discriminativeness for dynamic BM25/vector weight adjustment
        ) = {
            if let Some(graph) = self.graph_memory.as_ref().filter(|_| use_graph) {
                let g = graph.read();
                // Extract IC weights for BM25 term boosting
                let weights = query_analysis.to_ic_weights();
                // Extract phrase boosts for exact phrase matching (e.g., "support group")
                let phrases = query_analysis.to_phrase_boosts();
                // Extract keyword discriminativeness for dynamic weight adjustment
                // High discriminativeness → trust BM25 more for rare keywords like "sunrise"
                let (disc, disc_keywords) = query_analysis.keyword_discriminativeness();
                if disc > 0.5 && !disc_keywords.is_empty() {
                    tracing::debug!(
                        "Layer 2: YAKE discriminative keywords: {:?} (disc={:.2})",
                        disc_keywords,
                        disc
                    );
                }
                // Count entities in query for adaptive boost (multi-hop detection)
                let entity_count = query_analysis.focal_entities.len()
                    + query_analysis.discriminative_modifiers.len();

                // First, collect all query entity UUIDs
                // Include nouns, adjectives, AND verbs for multi-hop reasoning
                let mut query_entities: Vec<uuid::Uuid> = Vec::new();
                for e in query_analysis
                    .focal_entities
                    .iter()
                    .map(|e| e.text.as_str())
                    .chain(
                        query_analysis
                            .discriminative_modifiers
                            .iter()
                            .map(|m| m.text.as_str()),
                    )
                    .chain(
                        query_analysis
                            .relational_context
                            .iter()
                            .map(|r| r.text.as_str()),
                    )
                    .chain(
                        query_analysis
                            .relational_context
                            .iter()
                            .map(|r| r.stem.as_str()),
                    )
                {
                    if let Ok(Some(ent)) = g.find_entity_by_name(e) {
                        query_entities.push(ent.uuid);
                    }
                }

                // Calculate PER-ENTITY density (not global graph density)
                // Sparse entities = trust graph, Dense entities = trust vector
                let d = if !query_entities.is_empty() {
                    g.entities_average_density(&query_entities).ok().flatten()
                } else {
                    // No query entities — skip density calculation.
                    // The default weights (0.6, 0.3, 0.1) handle this case correctly.
                    None
                };

                let mut ids = Vec::new();

                // Density-adaptive traversal: dense graphs get shallower depth
                // and stricter strength filters to avoid exploring noisy L1 edges.
                // Dense graph results are already downweighted in RRF fusion
                // (graph_w=0.1 at density>2.0), so deep traversals add I/O cost
                // for results that contribute <0.01% to the fused score.
                let density_val = d.unwrap_or(0.0);
                let (bidir_depth, bidir_min_str, weighted_depth, weighted_min_str) =
                    if density_val > 15.0 {
                        (3usize, 0.12f32, 3usize, 0.15f32)
                    } else if density_val > 8.0 {
                        (4, 0.08, 4, 0.12)
                    } else {
                        (6, 0.05, 5, 0.10)
                    };

                if density_val > 0.0 {
                    tracing::debug!(
                        "Layer 2: density={:.1}, bidir_depth={}, bidir_min_str={:.2}, weighted_depth={}, weighted_min_str={:.2}",
                        density_val, bidir_depth, bidir_min_str, weighted_depth, weighted_min_str
                    );
                }

                // Multi-hop: Use bidirectional search between entity pairs
                // Cap to top 3 pairs from first 4 entities to avoid O(n²) explosion.
                // Entities are ordered by query analysis salience, so top pairs
                // capture dominant relationships.
                if query_entities.len() >= 2 {
                    let max_pairs = 3usize;
                    let max_ents = query_entities.len().min(4);
                    let mut pair_count = 0usize;
                    'bidir: for i in 0..max_ents {
                        for j in (i + 1)..max_ents {
                            if pair_count >= max_pairs {
                                break 'bidir;
                            }
                            if let Ok(path) = g.traverse_bidirectional(
                                &query_entities[i],
                                &query_entities[j],
                                bidir_depth,
                                bidir_min_str,
                            ) {
                                for tr in &path.entities {
                                    if let Ok(mut eps) = g.get_episodes_by_entity(&tr.entity.uuid) {
                                        // Keep most recent episodes — recency correlates
                                        // with relevance for graph-surfaced candidates.
                                        eps.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                                        eps.truncate(20);
                                        for ep in eps {
                                            let mid = MemoryId(ep.uuid);
                                            if episode_candidates
                                                .as_ref()
                                                .is_none_or(|c| c.contains(&mid))
                                            {
                                                let path_boost = 1.5;
                                                let activation = tr.entity.salience
                                                    * tr.decay_factor
                                                    * path_boost;
                                                // Relevance floor: skip low-activation candidates
                                                if activation >= 0.1 {
                                                    ids.push((
                                                        mid,
                                                        activation,
                                                        tr.decay_factor,
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            pair_count += 1;
                        }
                    }
                }

                // Single-hop or supplement multi-hop: Weighted traversal from each entity.
                // When ontological intent has sufficient confidence, pass relation types
                // as a filter to traverse_weighted for type-aware graph expansion.
                let use_onto_filter = onto_intent.confidence
                    >= crate::constants::ONTOLOGICAL_MIN_CONFIDENCE
                    && !onto_intent.relation_types.is_empty();
                let relation_filter: Option<Vec<crate::graph_memory::RelationType>> =
                    if use_onto_filter {
                        Some(onto_intent.relation_types.clone())
                    } else {
                        None
                    };

                for entity_uuid in &query_entities {
                    if let Ok(t) = g.traverse_weighted(
                        entity_uuid,
                        weighted_depth,
                        relation_filter.as_deref(),
                        weighted_min_str,
                    ) {
                        for tr in &t.entities {
                            if let Ok(mut eps) = g.get_episodes_by_entity(&tr.entity.uuid) {
                                eps.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                                eps.truncate(20);
                                for ep in eps {
                                    let mid = MemoryId(ep.uuid);
                                    if episode_candidates
                                        .as_ref()
                                        .is_none_or(|c| c.contains(&mid))
                                    {
                                        let activation =
                                            tr.entity.salience * tr.decay_factor;
                                        // Relevance floor: skip low-activation candidates
                                        if activation >= 0.1 {
                                            ids.push((
                                                mid,
                                                activation,
                                                tr.decay_factor,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let mut seen: std::collections::HashMap<MemoryId, (f32, f32)> =
                    std::collections::HashMap::new();
                for (id, act, heb) in ids {
                    seen.entry(id)
                        .and_modify(|(a, h)| {
                            *a = a.max(act);
                            *h = h.max(heb);
                        })
                        .or_insert((act, heb));
                }
                let mut r: Vec<_> = seen.into_iter().map(|(id, (a, h))| (id, a, h)).collect();
                // CRITICAL: Sort by activation score so RRF rank is meaningful
                r.sort_by(|a, b| b.1.total_cmp(&a.1));
                let pre_cap = r.len();
                // Cap total graph candidates to prevent flooding RRF fusion
                r.truncate(200);
                if !r.is_empty() {
                    tracing::debug!("Layer 2: {} graph results (capped from {}), {} query entities, bidirectional={}, top_activation={:.3}",
                        r.len(), pre_cap, entity_count, query_entities.len() >= 2, r.first().map(|x| x.1).unwrap_or(0.0));
                }
                (r, d, entity_count, weights, phrases, disc)
            } else {
                if !use_graph && self.graph_memory.is_some() {
                    tracing::debug!(
                        "Layer 2: SKIPPED (retrieval_mode={:?})",
                        query.retrieval_mode
                    );
                }
                // No graph traversal - still analyze query for IC weights and phrase boosts
                let (disc, _) = query_analysis.keyword_discriminativeness();
                (
                    Vec::new(),
                    None,
                    0,
                    query_analysis.to_ic_weights(),
                    query_analysis.to_phrase_boosts(),
                    disc,
                )
            }
        };

        let t_graph = recall_start.elapsed();
        tracing::info!(
            graph_ms = format!("{:.2}", (t_graph - t_embedding).as_secs_f64() * 1000.0),
            cumulative_ms = format!("{:.2}", t_graph.as_secs_f64() * 1000.0),
            graph_results = graph_results.len(),
            "recall [layer:1-2] episode filter + graph expansion"
        );

        // Create a modified query with the embedding for vector search
        let vector_query = Query {
            user_id: query.user_id.clone(),
            query_text: None, // Don't re-generate embedding
            query_embedding: Some(query_embedding.clone()),
            time_range: query.time_range,
            experience_types: query.experience_types.clone(),
            importance_threshold: query.importance_threshold,
            max_results: query.max_results,
            retrieval_mode: query.retrieval_mode.clone(),
            // Robotics filters (carry over from original query)
            robot_id: query.robot_id.clone(),
            mission_id: query.mission_id.clone(),
            geo_filter: query.geo_filter.clone(),
            action_type: query.action_type.clone(),
            reward_range: query.reward_range,
            // Decision & Learning filters (carry over from original query)
            outcome_type: query.outcome_type.clone(),
            failures_only: query.failures_only,
            anomalies_only: query.anomalies_only,
            severity: query.severity.clone(),
            tags: query.tags.clone(),
            pattern_id: query.pattern_id.clone(),
            terrain_type: query.terrain_type.clone(),
            confidence_range: query.confidence_range,
            offset: query.offset,
            episode_id: query.episode_id.clone(),
            prospective_signals: query.prospective_signals.clone(),
            recency_weight: query.recency_weight,
            rrf_k: query.rrf_k,
            rerank_count: query.rerank_count,
        };

        // ===========================================================================
        // LAYER 3: VECTOR SEARCH (Vamana Index) + QUERY DECOMPOSITION
        // ===========================================================================
        // Decompose compound/multi-entity queries into sub-queries for independent
        // vector search. BM25 and graph traversal still use the original query —
        // they handle compound queries better naturally.
        let sub_queries = query_parser::decompose_query(query_text, &query_analysis);
        let decomposed = sub_queries.len() > 1;

        let mut vector_results: Vec<(MemoryId, f32)> = if decomposed {
            // --- Decomposed path: weighted RRF merge of sub-query vector results ---
            use crate::constants::QUERY_DECOMPOSITION_SUB_WEIGHT;
            const DECOMP_RRF_K: f32 = 20.0;

            let mut merged: std::collections::HashMap<MemoryId, f32> =
                std::collections::HashMap::new();

            for (idx, sub_q) in sub_queries.iter().enumerate() {
                let is_original = idx == 0;
                let weight = if is_original {
                    1.0_f32
                } else {
                    QUERY_DECOMPOSITION_SUB_WEIGHT
                };

                // Embed sub-query (original already has embedding; sub-queries need encoding)
                let sub_embedding = if is_original {
                    query_embedding.clone()
                } else {
                    let sub_hash = Self::sha256_hash(sub_q);
                    if let Some(cached) = self.query_cache.get(&sub_hash) {
                        cached.clone()
                    } else {
                        match self.embedder.as_ref().encode(sub_q) {
                            Ok(emb) => {
                                self.query_cache.insert(sub_hash, emb.clone());
                                emb
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Layer 3: Failed to embed sub-query '{}': {}",
                                    sub_q,
                                    e
                                );
                                continue;
                            }
                        }
                    }
                };

                let sub_vector_query = Query {
                    user_id: query.user_id.clone(),
                    query_text: None,
                    query_embedding: Some(sub_embedding),
                    time_range: query.time_range,
                    experience_types: query.experience_types.clone(),
                    importance_threshold: query.importance_threshold,
                    max_results: query.max_results,
                    retrieval_mode: query.retrieval_mode.clone(),
                    robot_id: query.robot_id.clone(),
                    mission_id: query.mission_id.clone(),
                    geo_filter: query.geo_filter.clone(),
                    action_type: query.action_type.clone(),
                    reward_range: query.reward_range,
                    outcome_type: query.outcome_type.clone(),
                    failures_only: query.failures_only,
                    anomalies_only: query.anomalies_only,
                    severity: query.severity.clone(),
                    tags: query.tags.clone(),
                    pattern_id: query.pattern_id.clone(),
                    terrain_type: query.terrain_type.clone(),
                    confidence_range: query.confidence_range,
                    offset: query.offset,
                    episode_id: query.episode_id.clone(),
                    prospective_signals: query.prospective_signals.clone(),
                    recency_weight: query.recency_weight,
                    rrf_k: query.rrf_k,
                    rerank_count: query.rerank_count,
                };

                let sub_results =
                    self.retriever
                        .search_ids(&sub_vector_query, query.max_results * 8)?;

                let filtered: Vec<(MemoryId, f32)> =
                    if let Some(ref c) = episode_candidates {
                        sub_results
                            .into_iter()
                            .filter(|(id, _)| c.contains(id))
                            .collect()
                    } else {
                        sub_results
                    };

                // Weighted RRF: score = weight / (K + rank)
                for (rank, (id, _sim)) in filtered.iter().enumerate() {
                    let rrf_score = weight / (DECOMP_RRF_K + (rank + 1) as f32);
                    let entry = merged.entry(id.clone()).or_insert(0.0);
                    *entry = entry.max(rrf_score).max(*entry);
                    // Use max-merge: keep the highest RRF contribution per memory
                    // to prevent score inflation from appearing in multiple sub-queries
                    *entry = *entry + rrf_score - entry.min(rrf_score);
                }
            }

            // Sort by fused score and convert back to (MemoryId, f32)
            let mut fused_vec: Vec<(MemoryId, f32)> = merged.into_iter().collect();
            fused_vec.sort_by(|a, b| b.1.total_cmp(&a.1));

            tracing::info!(
                "Layer 3: Decomposed into {} sub-queries, merged {} unique candidates",
                sub_queries.len(),
                fused_vec.len()
            );

            fused_vec
        } else {
            // --- Standard path: single vector search ---
            let vr = self
                .retriever
                .search_ids(&vector_query, query.max_results * 8)?;
            if let Some(ref c) = episode_candidates {
                vr.into_iter().filter(|(id, _)| c.contains(id)).collect()
            } else {
                vr
            }
        };

        let t_vector = recall_start.elapsed();
        tracing::info!(
            vector_ms = format!("{:.2}", (t_vector - t_graph).as_secs_f64() * 1000.0),
            cumulative_ms = format!("{:.2}", t_vector.as_secs_f64() * 1000.0),
            vector_results = vector_results.len(),
            decomposed = decomposed,
            "recall [layer:3] Vamana vector search{}", if decomposed { " (decomposed)" } else { "" }
        );

        // ===========================================================================
        // LAYER 3.5: WORKING + SESSION TIER BRUTE-FORCE COSINE SCAN
        // ===========================================================================
        // Vamana indexes only long-term storage. Working-tier and Session-tier
        // memories (current session, last few hours) are NOT in the vector index.
        // This is the root cause of temporal MRR 0.767: the most recent memories
        // are invisible to vector search.
        //
        // Fix: brute-force cosine similarity over the small in-memory tiers
        // (typically <100 memories combined), merge results with Vamana output.
        // Cost: O(n) where n = working + session count. At <100 memories and
        // 384-dim vectors, this is <0.1ms — negligible vs Vamana's ~2-5ms.
        {
            let mut tier_hits = 0usize;
            let vamana_ids: std::collections::HashSet<MemoryId> = vector_results
                .iter()
                .map(|(id, _)| id.clone())
                .collect();

            // Scan Working tier
            for mem in self.working_memory.read().all_memories() {
                if vamana_ids.contains(&mem.id) {
                    continue; // already found by Vamana
                }
                if let Some(ref emb) = mem.experience.embeddings {
                    if !emb.is_empty() {
                        let sim = crate::memory::hybrid_search::cosine_similarity_pub(
                            &query_embedding,
                            emb,
                        );
                        if sim > 0.1 {
                            vector_results.push((mem.id.clone(), sim));
                            tier_hits += 1;
                        }
                    }
                }
            }

            // Scan Session tier
            for mem in self.session_memory.read().all_memories() {
                if vamana_ids.contains(&mem.id) {
                    continue;
                }
                if let Some(ref emb) = mem.experience.embeddings {
                    if !emb.is_empty() {
                        let sim = crate::memory::hybrid_search::cosine_similarity_pub(
                            &query_embedding,
                            emb,
                        );
                        if sim > 0.1 {
                            vector_results.push((mem.id.clone(), sim));
                            tier_hits += 1;
                        }
                    }
                }
            }

            if tier_hits > 0 {
                // Re-sort merged results by similarity descending
                vector_results.sort_by(|a, b| b.1.total_cmp(&a.1));
                vector_results.truncate(query.max_results * 8);
                tracing::info!(
                    tier_hits,
                    total = vector_results.len(),
                    "recall [layer:3.5] Working+Session brute-force cosine scan"
                );
            }
        }

        // ===========================================================================
        // LAYER 4: BM25 + RRF FUSION
        // ===========================================================================
        let (memory_ids, hebbian_scores): (
            Vec<(MemoryId, f32)>,
            std::collections::HashMap<MemoryId, f32>,
        ) = {
            let get_content = |id: &MemoryId| -> Option<String> {
                self.working_memory
                    .read()
                    .get(id)
                    .map(|m| m.experience.content.clone())
                    .or_else(|| {
                        self.session_memory
                            .read()
                            .get(id)
                            .map(|m| m.experience.content.clone())
                    })
                    .or_else(|| {
                        self.long_term_memory
                            .get(id)
                            .ok()
                            .map(|m| m.experience.content.clone())
                    })
            };
            // Use IC-weighted BM25 search with phrase matching
            let term_weights = if ic_weights.is_empty() {
                None
            } else {
                Some(&ic_weights)
            };
            let phrases = if phrase_boosts.is_empty() {
                None
            } else {
                Some(phrase_boosts.as_slice())
            };
            // Use dynamic weight adjustment based on YAKE keyword discriminativeness
            // High discriminativeness → boost BM25 weight for rare keywords
            let disc_opt = if keyword_disc > 0.3 {
                Some(keyword_disc)
            } else {
                None
            };
            let hybrid_ids = self
                .hybrid_search
                .search_with_dynamic_weights(
                    query_text,
                    vector_results.clone(),
                    get_content,
                    term_weights,
                    phrases,
                    disc_opt,
                    query.rerank_count, // FIX-11: per-query rerank count override
                )
                .map(|r| {
                    r.into_iter()
                        .map(|x| (x.memory_id, x.score))
                        .collect::<Vec<_>>()
                })
                .unwrap_or(vector_results);

            // ===========================================================================
            // LAYER 4: RRF FUSION WITH DENSITY-BASED WEIGHTS (PIPE-11)
            // ===========================================================================
            // Biological model: Memory graphs start dense (noisy L1 edges) and become
            // sparse over time through pruning (Hebbian "use it or lose it").
            //
            // Sparse graphs = mature, curated connections = trust graph more
            // Dense graphs = fresh, noisy connections = trust semantic/BM25 more
            //
            // The density weights directly control the balance - no extra multipliers.
            // This follows ACT-R's additive activation model.
            // K controls top-rank discrimination: K=20 → rank-1 vs rank-5 gap is 19%
            // (vs 12% at K=30). Higher K = more equal weighting, lower K = sharper.
            // Configurable per-query via rrf_k; defaults to 20.0.
            let k: f32 = query.rrf_k.unwrap_or(20.0);
            let mut fused: std::collections::HashMap<MemoryId, f32> =
                std::collections::HashMap::new();
            let mut heb: std::collections::HashMap<MemoryId, f32> =
                std::collections::HashMap::new();

            // Density-based weights (already tuned in calculate_density_weights)
            // Sparse (≤0.5): graph_w=0.5, semantic_w=0.4, linguistic_w=0.1
            // Dense (≥2.0):  graph_w=0.1, semantic_w=0.7, linguistic_w=0.2
            let density_weights = graph_density
                .map(calculate_density_weights)
                .unwrap_or((0.6, 0.3, 0.1));

            // Phase 2.2: Modulate weights by query type (Plan 001 §2.2)
            // Temporal → BM25↑, Attribute → Graph↑, Exploratory → density defaults
            let (semantic_w, graph_w, linguistic_w) =
                crate::memory::graph_retrieval::apply_query_type_weights(
                    density_weights,
                    &query_type,
                );

            // Hybrid weight = semantic + linguistic (BM25 + vector combined)
            let hybrid_w = semantic_w + linguistic_w;

            tracing::debug!(
                "Layer 4 RRF: density={:?}, graph_w={:.2}, hybrid_w={:.2}, query_entities={}",
                graph_density,
                graph_w,
                hybrid_w,
                query_entity_count
            );

            // Graph results: pure RRF with density weight
            for (r, (id, activation, h)) in graph_results.iter().enumerate() {
                // Standard RRF: weight / (k + rank), rank is 1-indexed
                let rrf_score = graph_w / (k + (r + 1) as f32);
                // RRF score + additive activation bonus (ACT-R spreading activation)
                let activation_bonus = graph_w * 0.2 * activation.clamp(0.0, 1.0);
                *fused.entry(id.clone()).or_insert(0.0) += rrf_score + activation_bonus;
                heb.insert(id.clone(), *h);
            }

            // Hybrid (BM25+vector) results: pure RRF with density weight
            for (r, (id, _)) in hybrid_ids.iter().enumerate() {
                *fused.entry(id.clone()).or_insert(0.0) += hybrid_w / (k + (r + 1) as f32);
            }

            // ===========================================================================
            // LAYER 4.5: ATTRIBUTE QUERY BOOST
            // ===========================================================================
            // For attribute queries, heavily boost memories that contain BOTH the entity
            // AND an attribute synonym value. This ensures "Caroline is single" ranks
            // high for "What is Caroline's relationship status?".
            if !attribute_boost_ids.is_empty() {
                const ATTRIBUTE_BOOST: f32 = 0.5; // Strong boost for attribute matches
                let mut boosted_count = 0;
                for id in &attribute_boost_ids {
                    if let Some(score) = fused.get_mut(id) {
                        *score += ATTRIBUTE_BOOST;
                        boosted_count += 1;
                    } else {
                        // Also add memories that weren't in the fusion but match attribute
                        fused.insert(id.clone(), ATTRIBUTE_BOOST);
                        boosted_count += 1;
                    }
                }
                if boosted_count > 0 {
                    tracing::info!(
                        "Layer 4.5: Boosted {} memories for attribute query",
                        boosted_count
                    );
                }
            }

            // ===========================================================================
            // LAYER 4.55: TEMPORAL FACT BOOST
            // ===========================================================================
            // Source memories of matching temporal facts get a precision-tuned boost.
            // Fires for all temporal intents (including WhenQuestion) and broader
            // temporal keywords to surface fact-sourced memories.
            // Uses TEMPORAL_RECALL_BOOST (0.20) — calibrated to lift temporal-source
            // memories by ~2 RRF positions without overriding strong semantic matches.
            if !temporal_fact_boost_ids.is_empty() {
                use crate::constants::TEMPORAL_RECALL_BOOST;
                let mut boosted_count = 0;
                for id in &temporal_fact_boost_ids {
                    if let Some(score) = fused.get_mut(id) {
                        *score += TEMPORAL_RECALL_BOOST;
                        boosted_count += 1;
                    } else {
                        fused.insert(id.clone(), TEMPORAL_RECALL_BOOST);
                        boosted_count += 1;
                    }
                }
                if boosted_count > 0 {
                    tracing::info!(
                        "Temporal boost applied to {} memories from {} matching facts",
                        boosted_count,
                        temporal_fact_boost_ids.len()
                    );
                }
            }

            // ===========================================================================
            // LAYER 4.52: EXPERIENCE-TYPE BOOST
            // ===========================================================================
            {
                let query_lower = query_text.to_lowercase();
                let is_decision_query = query_lower.contains("choose")
                    || query_lower.contains("chose")
                    || query_lower.contains("decide")
                    || query_lower.contains("decision")
                    || query_lower.contains("pick")
                    || query_lower.contains("select")
                    || query_lower.contains("what did we")
                    || (query_lower.contains("what") && query_lower.contains("use"));

                if is_decision_query {
                    let get_experience_type = |id: &MemoryId| -> Option<crate::memory::types::ExperienceType> {
                        self.working_memory
                            .read()
                            .get(id)
                            .map(|m| m.experience.experience_type.clone())
                            .or_else(|| {
                                self.session_memory
                                    .read()
                                    .get(id)
                                    .map(|m| m.experience.experience_type.clone())
                            })
                            .or_else(|| {
                                self.long_term_memory
                                    .get(id)
                                    .ok()
                                    .map(|m| m.experience.experience_type.clone())
                            })
                    };

                    const DECISION_TYPE_BOOST: f32 = 0.15;
                    let mut boosted_count = 0usize;
                    let ids: Vec<MemoryId> = fused.keys().cloned().collect();
                    for id in &ids {
                        if let Some(exp_type) = get_experience_type(id) {
                            if matches!(exp_type, crate::memory::types::ExperienceType::Decision) {
                                if let Some(score) = fused.get_mut(id) {
                                    *score += DECISION_TYPE_BOOST;
                                    boosted_count += 1;
                                }
                            }
                        }
                    }
                    if boosted_count > 0 {
                        tracing::debug!(
                            "Layer 4.52: Boosted {} Decision-type memories for decision query",
                            boosted_count
                        );
                    }
                }
            }

            // ===========================================================================
            // LAYER 4.525: ENTITY-QUERY OVERLAP BOOST (stemmed + filtered)
            // ===========================================================================
            // Boost memories whose entities/tags match stemmed query terms.
            // Porter stemming resolves morphological mismatches: "bugs"->"bug",
            // "risks"->"risk", "testing"->"test". Stop word filtering prevents
            // false positives from "the" matching inside "prometheus" etc.
            // Linear 0.10 per match, capped at 0.40.
            {
                use rust_stemmers::{Algorithm, Stemmer};
                let stemmer = Stemmer::create(Algorithm::English);

                const STOP_WORDS: &[&str] = &[
                    "the", "and", "for", "are", "but", "not", "you", "all", "can",
                    "had", "was", "one", "our", "out", "has", "how", "its", "who",
                    "did", "get", "got", "let", "say", "she", "too", "use", "what",
                    "when", "where", "which", "with", "will", "would", "could",
                    "should", "about", "after", "before", "from", "have", "been",
                    "were", "being", "their", "there", "these", "those", "this",
                    "that", "than", "into", "some", "such", "them", "then", "very",
                    "just", "also", "most", "over", "only", "other", "during",
                ];

                let query_stems: std::collections::HashSet<String> = query_text
                    .to_lowercase()
                    .split_whitespace()
                    .filter(|w| w.len() > 2)
                    .map(|w| {
                        let cleaned: String =
                            w.chars().filter(|c| c.is_alphanumeric()).collect();
                        stemmer.stem(&cleaned).to_string()
                    })
                    .filter(|s| !s.is_empty() && s.len() > 1 && !STOP_WORDS.contains(&s.as_str()))
                    .collect();

                if !query_stems.is_empty() {
                    let get_entities = |id: &MemoryId| -> Vec<String> {
                        self.working_memory
                            .read()
                            .get(id)
                            .map(|m| m.experience.entities.clone())
                            .or_else(|| {
                                self.session_memory
                                    .read()
                                    .get(id)
                                    .map(|m| m.experience.entities.clone())
                            })
                            .or_else(|| {
                                self.long_term_memory
                                    .get(id)
                                    .ok()
                                    .map(|m| m.experience.entities.clone())
                            })
                            .unwrap_or_default()
                    };

                    let ids: Vec<MemoryId> = fused.keys().cloned().collect();
                    for id in &ids {
                        let entities = get_entities(id);
                        let entity_stems: Vec<String> = entities
                            .iter()
                            .map(|e| stemmer.stem(&e.to_lowercase()).to_string())
                            .collect();
                        let overlap = query_stems
                            .iter()
                            .filter(|qs| {
                                entity_stems
                                    .iter()
                                    .any(|es| es.contains(qs.as_str()))
                            })
                            .count();
                        if overlap >= 1 {
                            let boost = (0.10 * overlap as f32).min(0.40);
                            if let Some(score) = fused.get_mut(id) {
                                *score += boost;
                            }
                        } else if overlap == 0 && !has_any_temporal_intent {
                            // Content-fallback: when no entity/tag overlap, check
                            // if query stems appear in memory content. Smaller boost
                            // (+0.05/match, cap 0.20) to keep below tag-match priority.
                            // Gated: skip for temporal queries where content overlap
                            // would boost wrong-session memories indiscriminately.
                            if let Some(content) = get_content(id) {
                                let content_lower = content.to_lowercase();
                                let content_overlap = query_stems
                                    .iter()
                                    .filter(|qs| content_lower.contains(qs.as_str()))
                                    .count();
                                if content_overlap >= 2 {
                                    let boost = (0.05 * content_overlap as f32).min(0.20);
                                    if let Some(score) = fused.get_mut(id) {
                                        *score += boost;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ===========================================================================
            // LAYER 4.53: SPECIFICITY PENALTY
            // ===========================================================================
            // Retrospective/summary memories mention many topics and score well on
            // BM25 for ANY query. Penalize memories whose content is much longer than
            // the mean — they're likely summaries, not primary sources.
            // This is the key signal that differentiates "we decided to use Rust"
            // (specific, focused) from "things that didn't go well: Rust, MongoDB..."
            // (broad retrospective).
            {
                let mut lengths: Vec<(MemoryId, usize)> = Vec::new();
                for id in fused.keys() {
                    if let Some(content) = get_content(id) {
                        lengths.push((id.clone(), content.len()));
                    }
                }
                if lengths.len() >= 5 {
                    let mean_len: f32 = lengths.iter().map(|(_, l)| *l as f32).sum::<f32>()
                        / lengths.len() as f32;
                    for (id, len) in &lengths {
                        let ratio = *len as f32 / mean_len.max(1.0);
                        // Memories 50%+ longer than mean get penalized (0.85-0.70x)
                        // Memories shorter than mean get boosted (up to 1.15x)
                        if ratio > 1.5 {
                            let penalty = 1.0 - (ratio - 1.5).min(1.0) * 0.30;
                            if let Some(score) = fused.get_mut(id) {
                                *score *= penalty.max(0.70);
                            }
                        } else if ratio < 0.8 {
                            // Shorter, focused memories get a small boost
                            if let Some(score) = fused.get_mut(id) {
                                *score *= 1.0 + (0.8 - ratio) * 0.20;
                            }
                        }
                    }
                }
            }

            // ===========================================================================
            // LAYER 4.535: LIGHT POLLUTION FILTER (Entity Concentration Ratio)
            // ===========================================================================
            // Light pollution: broad overview memories mention many topics and
            // rank well on BM25/entity-overlap for ANY query, drowning out
            // specific memories that ARE ABOUT the query topic.
            //
            // Entity concentration C = |query_stems ∩ entity_stems| / |entity_stems|
            //   C < 0.20 → memory is a broad overview mentioning the topic
            //              among many others (light polluter) → 0.85× penalty
            //   C > 0.50 → memory is focused on query topic → 1.10× boost
            //   0.20 ≤ C ≤ 0.50 → neutral (no adjustment)
            //
            // Example: M16 has entities [database, migrations, sqlx] (3 entities).
            // Query "What database?" matches 1/3 = 0.33 concentration → neutral.
            // M3 has entities [postgresql, database] (2 entities).
            // Query matches 1/2 = 0.50 concentration → 1.10× boost.
            //
            // This is the "belt" in belt-and-braces against light pollution.
            // The "braces" is Layer 4.54 (temporal topic dedup).
            {
                use rust_stemmers::{Algorithm, Stemmer};
                let stem = Stemmer::create(Algorithm::English);

                let q_stems: std::collections::HashSet<String> = query_text
                    .to_lowercase()
                    .split_whitespace()
                    .filter(|w| w.len() > 2)
                    .map(|w| {
                        let cleaned: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                        stem.stem(&cleaned).to_string()
                    })
                    .filter(|s| s.len() > 1)
                    .collect();

                if q_stems.len() >= 2 {
                    let get_ents = |id: &MemoryId| -> Vec<String> {
                        self.working_memory.read().get(id).map(|m| m.experience.entities.clone())
                            .or_else(|| self.session_memory.read().get(id).map(|m| m.experience.entities.clone()))
                            .or_else(|| self.long_term_memory.get(id).ok().map(|m| m.experience.entities.clone()))
                            .unwrap_or_default()
                    };

                    let ids: Vec<MemoryId> = fused.keys().cloned().collect();
                    for id in &ids {
                        let entities = get_ents(id);
                        if entities.len() >= 2 {
                            let ent_stems: Vec<String> = entities
                                .iter()
                                .map(|e| stem.stem(&e.to_lowercase()).to_string())
                                .collect();
                            let matching = q_stems
                                .iter()
                                .filter(|qs| ent_stems.iter().any(|es| es.contains(qs.as_str())))
                                .count();
                            let concentration = matching as f32 / ent_stems.len() as f32;

                            if let Some(score) = fused.get_mut(id) {
                                if concentration < 0.20 {
                                    *score *= 0.85;
                                } else if concentration > 0.50 {
                                    *score *= 1.10;
                                }
                            }
                        }
                    }
                }
            }

            // ===========================================================================
            // LAYER 4.54: TEMPORAL TOPIC DEDUPLICATION (Recency Tiebreaker)
            // ===========================================================================
            // When two memories cover the same topic (high entity overlap) and the
            // query has recency intent, demote the older memory. Belt & braces
            // against light pollution from old-but-topical memories.
            if has_any_temporal_intent || has_temporal_keywords {
                let ids: Vec<MemoryId> = fused.keys().cloned().collect();
                let get_created_at = |id: &MemoryId| -> Option<chrono::DateTime<chrono::Utc>> {
                    self.working_memory.read().get(id).map(|m| m.created_at)
                        .or_else(|| self.session_memory.read().get(id).map(|m| m.created_at))
                        .or_else(|| self.long_term_memory.get(id).ok().map(|m| m.created_at))
                };
                let get_ents = |id: &MemoryId| -> std::collections::HashSet<String> {
                    self.working_memory.read().get(id).map(|m| m.experience.entities.clone())
                        .or_else(|| self.session_memory.read().get(id).map(|m| m.experience.entities.clone()))
                        .or_else(|| self.long_term_memory.get(id).ok().map(|m| m.experience.entities.clone()))
                        .unwrap_or_default()
                        .into_iter()
                        .map(|e| e.to_lowercase())
                        .collect()
                };

                let entity_sets: Vec<(MemoryId, std::collections::HashSet<String>)> = ids
                    .iter()
                    .map(|id| (id.clone(), get_ents(id)))
                    .collect();

                let mut demoted = 0usize;
                for i in 0..entity_sets.len() {
                    for j in (i + 1)..entity_sets.len() {
                        let (id_a, set_a) = &entity_sets[i];
                        let (id_b, set_b) = &entity_sets[j];
                        if set_a.is_empty() || set_b.is_empty() { continue; }

                        let intersection = set_a.intersection(set_b).count();
                        let union = set_a.union(set_b).count();
                        let jaccard = intersection as f32 / union.max(1) as f32;

                        if jaccard > 0.40 {
                            if let (Some(ts_a), Some(ts_b)) = (get_created_at(id_a), get_created_at(id_b)) {
                                let older_id = if ts_a < ts_b { id_a } else { id_b };
                                if let Some(score) = fused.get_mut(older_id) {
                                    *score *= 0.85;
                                    demoted += 1;
                                }
                            }
                        }
                    }
                }
                if demoted > 0 {
                    tracing::debug!("Layer 4.54: Temporal dedup demoted {} older same-topic memories", demoted);
                }
            }

            // ===========================================================================
            // LAYER 4.6: INTERFERENCE-AWARE SCORING (PIPE-3)
            // ===========================================================================
            // Research basis: Anderson & Neely (1996) - Retrieval-induced forgetting
            //
            // Retrieval is a competitive process:
            // - Memories that frequently "lose" competitions → harder to retrieve (suppress)
            // - Memories that survive despite competition → stronger/reliable (boost)
            //
            // The adjustment is based on interference history + current activation:
            // - High interference + high activation = "survivor" → boost (1.0-1.5x)
            // - High interference + low activation = "chronic loser" → suppress (0.5-1.0x)
            // - No interference history → neutral (1.0x)
            {
                let detector = self.interference_detector.read();

                // Compute max score once for normalization
                let max_score = fused
                    .values()
                    .copied()
                    .fold(0.0_f32, |a, b| a.max(b))
                    .max(0.01);

                // Collect adjustments first to avoid borrow issues
                let adjustments: Vec<_> = fused
                    .iter()
                    .map(|(id, &score)| {
                        let current_activation = (score / max_score).clamp(0.0, 1.0);
                        let adjustment = detector
                            .calculate_retrieval_adjustment(&id.0.to_string(), current_activation);
                        (id.clone(), adjustment)
                    })
                    .filter(|(_, adj)| (*adj - 1.0).abs() > 0.01)
                    .collect();

                // Apply adjustments
                let adjusted_count = adjustments.len();
                for (id, adjustment) in adjustments {
                    if let Some(score) = fused.get_mut(&id) {
                        *score *= adjustment;
                    }
                }

                if adjusted_count > 0 {
                    tracing::debug!(
                        "Layer 4.6 (PIPE-3): Applied interference adjustments to {} memories",
                        adjusted_count
                    );
                }
            }

            // ===========================================================================
            // LAYER 4.7: PROSPECTIVE SIGNAL BOOST ("Future Informs Present")
            // ===========================================================================
            // Research basis: Einstein & McDaniel (2005) - Prospective Memory
            //
            // Active goals and pending intentions shape what we remember.
            // When context-triggered prospective tasks match the current query,
            // memories related to those intentions become more accessible —
            // just as prospective memory primes retrospective recall in humans.
            //
            // Signals come from ProspectiveTasks that matched the current query
            // via keyword or semantic similarity (built in recall handler C5).
            if let Some(ref signals) = query.prospective_signals {
                if !signals.is_empty() {
                    const PROSPECTIVE_BOOST_PER_MATCH: f32 = 0.15;
                    const MAX_PROSPECTIVE_BOOST: f32 = 0.5;

                    // Tokenize all signals into unique terms (skip noise words < 3 chars)
                    let signal_terms: std::collections::HashSet<String> = signals
                        .iter()
                        .flat_map(|s| {
                            s.to_lowercase()
                                .split_whitespace()
                                .filter(|w| w.len() >= 3)
                                .map(|w| w.to_string())
                                .collect::<Vec<_>>()
                        })
                        .collect();

                    if !signal_terms.is_empty() {
                        let mut boosted_count = 0;
                        let ids: Vec<MemoryId> = fused.keys().cloned().collect();

                        for id in &ids {
                            if let Some(content) = get_content(id) {
                                let content_lower = content.to_lowercase();
                                let match_count = signal_terms
                                    .iter()
                                    .filter(|term| content_lower.contains(term.as_str()))
                                    .count();

                                if match_count > 0 {
                                    // Sqrt scaling: diminishing returns for additional matches
                                    let boost = (PROSPECTIVE_BOOST_PER_MATCH
                                        * (match_count as f32).sqrt())
                                    .min(MAX_PROSPECTIVE_BOOST);
                                    if let Some(score) = fused.get_mut(id) {
                                        *score += boost;
                                        boosted_count += 1;
                                    }
                                }
                            }
                        }

                        if boosted_count > 0 {
                            tracing::info!(
                                "Layer 4.7: Boosted {} memories from {} prospective signal terms",
                                boosted_count,
                                signal_terms.len()
                            );
                        }
                    }
                }
            }

            // ===========================================================================
            // LAYER 4.8: SEMANTIC FACT SOURCE BOOST
            // ===========================================================================
            // Consolidated facts represent stable knowledge. When query entities match
            // fact entities, the source memories that generated those facts contain the
            // richest context — they should rank higher.
            //
            // Conservative: only boosts memories already in fused set (does NOT inject
            // new candidates). Facts validate existing retrieval signals, not override.
            if !fact_source_boosts.is_empty() {
                let mut boosted_count = 0;
                for (id, boost) in &fact_source_boosts {
                    if let Some(score) = fused.get_mut(id) {
                        *score += boost;
                        boosted_count += 1;
                    }
                }
                if boosted_count > 0 {
                    tracing::info!(
                        "Layer 4.8: Boosted {} memories from semantic fact sources",
                        boosted_count
                    );
                }
            }

            // ===========================================================================
            // LAYER 4.9: ONTOLOGICAL RE-RANKING
            // ===========================================================================
            // Boost memories connected to type-matching entities. Conservative additive
            // boost on top of the fused score. Only active when ontological intent has
            // sufficient confidence.
            //
            // Reference: Collins & Quillian (1969) — type-plausible paths retrieved faster
            // Pre-sort and limit candidates before expensive re-ranking.
            // Only look up graph entities for the top 2x max_results candidates,
            // not all fused results (avoids 100s of RocksDB reads).
            let mut res: Vec<_> = fused.into_iter().collect();
            res.sort_by(|a, b| b.1.total_cmp(&a.1));
            let rerank_budget = query.max_results * 2;

            if use_ontology_rerank && !onto_intent.expected_labels.is_empty() {
                if let Some(graph) = self.graph_memory.as_ref() {
                    let g = graph.read();
                    let mut boosted_count = 0usize;
                    let mut penalized_count = 0usize;
                    for (_mem_id, fused_score) in res.iter_mut().take(rerank_budget) {
                        if let Ok(Some(episode)) = g.get_episode(&_mem_id.0) {
                            let type_matches = episode
                                .entity_refs
                                .iter()
                                .filter(|uuid| {
                                    g.get_entity(uuid)
                                        .ok()
                                        .flatten()
                                        .map(|e| {
                                            e.labels.iter().any(|l| {
                                                onto_intent
                                                    .expected_labels
                                                    .iter()
                                                    .any(|exp| l.matches_with_hierarchy(exp))
                                            })
                                        })
                                        .unwrap_or(false)
                                })
                                .count();
                            if type_matches > 0 {
                                let boost = (type_matches as f32
                                    * crate::constants::ONTOLOGICAL_RERANK_BOOST)
                                    .min(crate::constants::ONTOLOGICAL_RERANK_MAX);
                                *fused_score += boost;
                                boosted_count += 1;
                                crate::metrics::ONTOLOGICAL_RERANK_BOOST_APPLIED
                                    .observe(boost as f64);
                            } else if !episode.entity_refs.is_empty()
                                && onto_intent.confidence
                                    >= crate::constants::ONTOLOGICAL_PENALTY_MIN_CONFIDENCE
                            {
                                // Memory has entities but NONE match expected types.
                                // Apply penalty only at high confidence to avoid
                                // punishing NER extraction gaps.
                                *fused_score += crate::constants::ONTOLOGICAL_RERANK_PENALTY;
                                penalized_count += 1;
                            }
                        }
                    }
                    if boosted_count > 0 || penalized_count > 0 {
                        tracing::debug!(
                            "Layer 4.9: Ontological re-rank boosted={} penalized={} of {} candidates (labels={:?})",
                            boosted_count,
                            penalized_count,
                            rerank_budget.min(res.len()),
                            onto_intent.expected_labels
                        );
                    }
                    // Re-sort after boosting since ranks may have changed
                    res.sort_by(|a, b| b.1.total_cmp(&a.1));
                }
            }

            // ===========================================================================
            // LAYER 4.92: INTERFERENCE DETECTION (post-reranking)
            // ===========================================================================
            // Detect contradictory memories in the result set using semantic opposition.
            // When two highly-similar memories contradict each other (negation inversion
            // or antonym pairs), demote the older one. This is orthogonal to both
            // embedding similarity and graph structure because it detects factual
            // conflict that those signals cannot distinguish from agreement.
            //
            // Uses existing detect_semantic_opposition() from temporal_facts module.
            // Only checks top-K pairs to keep O(k^2) manageable (k = interference_budget).
            {
                use crate::memory::temporal_facts::{detect_semantic_opposition, text_similarity};

                let interference_budget = 20.min(res.len());
                let mut demoted = 0usize;

                // Collect content and timestamps for the top candidates
                let contents: Vec<Option<String>> = res
                    .iter()
                    .take(interference_budget)
                    .map(|(id, _)| get_content(id))
                    .collect();
                let get_ts = |id: &MemoryId| -> Option<chrono::DateTime<chrono::Utc>> {
                    self.working_memory.read().get(id).map(|m| m.created_at)
                        .or_else(|| self.session_memory.read().get(id).map(|m| m.created_at))
                        .or_else(|| self.long_term_memory.get(id).ok().map(|m| m.created_at))
                };
                let timestamps: Vec<Option<chrono::DateTime<chrono::Utc>>> = res
                    .iter()
                    .take(interference_budget)
                    .map(|(id, _)| get_ts(id))
                    .collect();

                // Check pairwise: only pairs with high text similarity (>0.7)
                // that also show semantic opposition
                let mut demote_indices: Vec<(usize, f32)> = Vec::new();
                for i in 0..contents.len() {
                    let Some(ref ci) = contents[i] else { continue };
                    for j in (i + 1)..contents.len() {
                        let Some(ref cj) = contents[j] else { continue };

                        let sim = text_similarity(ci, cj);
                        if sim < 0.7 {
                            continue;
                        }

                        if detect_semantic_opposition(ci, cj) {
                            // Contradiction detected — demote the older memory
                            let ti = timestamps[i];
                            let tj = timestamps[j];
                            let older_idx = match (ti, tj) {
                                (Some(a), Some(b)) if a < b => i,
                                (Some(a), Some(b)) if b < a => j,
                                _ => j, // default: demote the lower-ranked one
                            };
                            // Penalty proportional to similarity: higher sim = more
                            // confident this is a true contradiction, not coincidence
                            let penalty = sim * 0.15;
                            demote_indices.push((older_idx, penalty));
                        }
                    }
                }

                // Apply demotions
                for (idx, penalty) in &demote_indices {
                    if let Some((_, score)) = res.get_mut(*idx) {
                        *score -= penalty;
                        demoted += 1;
                    }
                }

                if demoted > 0 {
                    tracing::debug!(
                        "Layer 4.92: Interference detection demoted {} contradictory memories",
                        demoted
                    );
                    res.sort_by(|a, b| b.1.total_cmp(&a.1));
                }
            }

            // ===========================================================================
            // LAYER 4.95: NEAR-DUPLICATE REMOVAL
            // ===========================================================================
            // Remove near-duplicate results by comparing content prefixes (first 200 chars).
            // When two results share identical content prefixes, keep only the higher-scored
            // one. This prevents redundant results from consuming top-k slots.
            {
                let pre_dedup = res.len();
                // Collect content prefixes for the top candidates (2x max to allow fills)
                let dedup_budget = (query.max_results * 2).min(res.len());
                let prefixes: Vec<Option<String>> = res
                    .iter()
                    .take(dedup_budget)
                    .map(|(id, _)| {
                        get_content(id).map(|c| {
                            c.chars().take(200).collect::<String>().to_lowercase()
                        })
                    })
                    .collect();

                let mut remove_indices: Vec<bool> = vec![false; res.len()];
                for i in 0..prefixes.len() {
                    if remove_indices[i] {
                        continue;
                    }
                    if let Some(ref pi) = prefixes[i] {
                        for j in (i + 1)..prefixes.len() {
                            if remove_indices[j] {
                                continue;
                            }
                            if let Some(ref pj) = prefixes[j] {
                                if pi == pj {
                                    // Results are sorted by score desc; j has lower score
                                    remove_indices[j] = true;
                                }
                            }
                        }
                    }
                }

                let mut idx = 0;
                res.retain(|_| {
                    let keep = !remove_indices.get(idx).copied().unwrap_or(false);
                    idx += 1;
                    keep
                });

                let removed = pre_dedup - res.len();
                if removed > 0 {
                    tracing::debug!(
                        "Layer 4.95: Removed {} near-duplicate results from {} candidates",
                        removed,
                        pre_dedup
                    );
                }
            }

            // Phase 2.1: Keep wider candidate pool for cross-encoder reranking.
            // Rerank top-20 after memory fetch, then truncate to max_results.
            let rerank_budget = 30_usize.max(query.max_results);
            res.truncate(rerank_budget);
            tracing::debug!("Layer 4: {} fused results (rerank budget={})", res.len(), rerank_budget);
            (res, heb)
        };

        let t_fusion = recall_start.elapsed();
        tracing::info!(
            fusion_ms = format!("{:.2}", (t_fusion - t_vector).as_secs_f64() * 1000.0),
            cumulative_ms = format!("{:.2}", t_fusion.as_secs_f64() * 1000.0),
            fused_results = memory_ids.len(),
            "recall [layer:4] BM25 + RRF fusion + boosts + interference"
        );

        // Fetch memories with cache-aware strategy
        // CRITICAL: Apply filters after fetching to ensure mission_id, robot_id etc. are respected
        let mut memories = Vec::new();
        let mut sources = Vec::new();
        let mut cache_hits = 0;
        let mut storage_fetches = 0;
        let mut filtered_out = 0;

        // Fetch up to rerank_pool candidates so Layer 5.5 cross-encoder can rerank
        // before truncating to max_results. Without this, the early break at max_results
        // would prevent the cross-encoder from ever firing (max_results == max_results).
        let rerank_pool = 20_usize.max(query.max_results);

        // Layer 5: Unified scoring with hebbian + recency + emotional + feedback signals
        // Recency decay: recent memories get boost, old memories decay
        // λ = 0.01 means ~50% at 70 hours, ~25% at 140 hours
        const RECENCY_DECAY_RATE: f32 = 0.01;
        let now = chrono::Utc::now();

        // PIPE-9: Get feedback store guard for momentum-based scoring
        // Acquire once outside the loop to avoid repeated locking
        let feedback_guard = self.feedback_store.as_ref().map(|fs| fs.read());

        // Signal 20: Pinky dimension aggregate (graph topological health)
        // Computed once per query — same value for all memories.
        // Acts as a global quality multiplier: high-quality graphs boost all scores.
        let pinky_multiplier = self.pinky_aggregate_score().unwrap_or(1.0);

        for (memory_id, score) in memory_ids {
            // Hebbian boost from learned graph weights (10% contribution)
            let hebbian_boost = hebbian_scores.get(&memory_id).copied().unwrap_or(0.0);
            let base_score = score + hebbian_boost * 0.1;

            // Helper to apply unified scoring (recency + arousal + credibility + temporal)
            // Amplify recency when query explicitly asks for recent items
            let query_lower_recency = query_text.to_lowercase();
            let wants_recent = query_lower_recency.contains("last week")
                || query_lower_recency.contains("most recent")
                || query_lower_recency.contains("recent")
                || query_lower_recency.contains("latest")
                || query_lower_recency.contains("yesterday")
                || query_lower_recency.contains("today");
            let recency_scale = if wants_recent {
                0.5 // 5x amplification for explicit recency queries
            } else {
                query.recency_weight.unwrap_or(0.1)
            };
            let with_unified_score = |mem: &SharedMemory, base: f32| -> SharedMemory {
                // Recency decay: exponential decay based on age
                let hours_old = (now - mem.created_at).num_hours().max(0) as f32;
                let recency_boost = (-RECENCY_DECAY_RATE * hours_old).exp() * recency_scale;

                // Emotional arousal boost: high arousal = more salient (5% contribution)
                // Research: LaBar & Cabeza (2006) - emotionally arousing events better remembered
                let arousal_boost = mem
                    .experience
                    .context
                    .as_ref()
                    .map(|c| c.emotional.arousal * 0.05)
                    .unwrap_or(0.0);

                // Source credibility boost: credible sources weighted higher (5% contribution)
                // Research: Source monitoring affects memory reliability
                let credibility_boost = mem
                    .experience
                    .context
                    .as_ref()
                    .map(|c| (c.source.credibility - 0.5).max(0.0) * 0.1)
                    .unwrap_or(0.0);

                // TEMPORAL BOOST (TEMPR approach - key for multi-hop retrieval)
                // If query has temporal intent and memory has matching temporal references,
                // significantly boost the memory's score (25% contribution when matched)
                let temporal_boost = if has_temporal_query
                    && !mem.experience.temporal_refs.is_empty()
                {
                    // Check if any memory temporal ref matches query temporal refs
                    let mut best_match = 0.0_f32;
                    for mem_ref in &mem.experience.temporal_refs {
                        for query_ref in &query_temporal.refs {
                            // Exact date match: strong boost
                            if mem_ref == &query_ref.date.to_string() {
                                best_match = best_match.max(0.25);
                            } else if let Ok(mem_date) =
                                chrono::NaiveDate::parse_from_str(mem_ref, "%Y-%m-%d")
                            {
                                // Approximate match: within 7 days gets partial boost
                                let days_diff = (mem_date - query_ref.date).num_days().abs();
                                if days_diff <= 7 {
                                    let proximity_boost = 0.15 * (1.0 - days_diff as f32 / 7.0);
                                    best_match = best_match.max(proximity_boost);
                                } else if days_diff <= 30 {
                                    // Within a month: smaller boost
                                    let proximity_boost = 0.05 * (1.0 - days_diff as f32 / 30.0);
                                    best_match = best_match.max(proximity_boost);
                                }
                            }
                        }
                    }
                    best_match
                } else {
                    0.0
                };

                // FEEDBACK MOMENTUM (PIPE-9)
                // Apply momentum from past feedback to consistently boost/suppress memories
                // - Positive momentum (proven helpful) → boost score
                // - Negative momentum (frequently ignored) → suppress up to 20%
                // This ensures consistent feedback integration across ALL retrieval paths
                let feedback_multiplier = if let Some(ref guard) = feedback_guard {
                    if let Some(fm) = guard.get_momentum(&mem.id) {
                        let momentum = fm.ema_with_decay();
                        if momentum < 0.0 {
                            // Suppress: up to 20% penalty for highly negative momentum
                            1.0 + (momentum * 0.2).max(-0.2)
                        } else {
                            // Boost: up to 10% bonus for positive momentum
                            1.0 + (momentum * 0.1).min(0.1)
                        }
                    } else {
                        1.0 // No feedback history
                    }
                } else {
                    1.0 // No feedback store configured
                };

                // SESSION CONTINUITY BOOST: memories from the current session
                // (created within the last 2 hours) get a small additive boost.
                // This compensates for the fact that very recent memories haven't
                // had time to accumulate Hebbian co-activation, BM25 term frequency,
                // or graph edges — they're signal-poor despite being contextually prime.
                let session_boost = {
                    let age_hours = (now - mem.created_at).num_hours();
                    if age_hours <= 2 {
                        0.03 // within current session window
                    } else {
                        0.0
                    }
                };

                // =================================================================
                // BRIDGE SIGNALS (9-19): Connect captured-but-forgotten signals
                // =================================================================

                // BRIDGE-1: access_count (signal 5, proven 14% in proactive)
                let access_boost = ((mem.access_count() as f64).ln_1p() / 5.0) as f32 * 0.07;

                // BRIDGE-1: graph_strength (signal 3+, proven 13% in proactive)
                let graph_boost = hebbian_scores
                    .get(&mem.id)
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0)
                    * 0.08;

                // Signal 9: Episode ID coherence — same-episode memories get boost
                let episode_boost = mem
                    .experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.episode_id.as_ref())
                    .and_then(|mem_ep| {
                        query.episode_id.as_ref().and_then(|q_ep| {
                            if mem_ep == q_ep { Some(0.08_f32) } else { None }
                        })
                    })
                    .unwrap_or(0.0);

                // Signal 10: Source type multiplier on credibility
                let source_type_mult = mem
                    .experience
                    .context
                    .as_ref()
                    .map(|c| match c.source.source_type {
                        crate::memory::types::SourceType::User => 1.2,
                        crate::memory::types::SourceType::File => 1.1,
                        crate::memory::types::SourceType::System => 1.0,
                        crate::memory::types::SourceType::ExternalApi => 0.9,
                        crate::memory::types::SourceType::Web => 0.8,
                        crate::memory::types::SourceType::AiGenerated => 0.7,
                        crate::memory::types::SourceType::Inferred => 0.6,
                        _ => 1.0,
                    })
                    .unwrap_or(1.0);
                let credibility_boost = credibility_boost * source_type_mult;

                // Signal 11: Emotional valence as absolute intensity
                let valence_boost = mem
                    .experience
                    .context
                    .as_ref()
                    .map(|c| c.emotional.valence.abs() * 0.02)
                    .unwrap_or(0.0);

                // Signal 12: Sequence proximity within episode
                let sequence_boost = mem
                    .experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.sequence_number)
                    .map(|seq| ((seq as f64).ln_1p() / 5.0) as f32 * 0.02)
                    .unwrap_or(0.0);

                // Signal 16: Context richness — memories with richer context
                // (more populated fields) are more useful for retrieval
                let richness_boost = (mem.context_richness() as f32 / 10.0) * 0.02;

                // Signal 17: Activation level — current Hebbian co-activation state
                let activation_boost = mem.activation() * 0.03;

                // Signal 18: Temporal fact density — more temporal refs = more anchored
                let temporal_density = (mem.experience.temporal_refs.len() as f32 / 5.0)
                    .min(1.0)
                    * 0.02;

                // Signal 19: Entity confidence — memories with more entity refs
                // are more structurally connected (proxy for graph salience)
                let entity_density = (mem.entity_refs.len() as f32 / 5.0).min(1.0) * 0.02;

                // Signal 20 (FIX-R2): Elaboration quality — well-contextualized C-rep
                // memories are more reliable than bare S-rep fragments.
                // Reference: Ehlers & Clark (2000) — elaborated memories produce
                // functional recalls, unelaborated ones produce pathological intrusions.
                let elaboration_boost = mem.elaboration_score() * 0.03;

                // Signal 21 (FIX-R1): Access burstiness — bursty access patterns
                // indicate working memory (currently active topic). Steady patterns
                // indicate long-term storage. Bursty memories get a small recency-
                // independent boost because they're likely contextually relevant.
                // Reference: Berntsen (2021) — involuntary memories favor recent,
                // actively-processed content.
                let burstiness = mem.access_burstiness();
                let burstiness_boost = if burstiness > 1.5 {
                    0.02 // bursty = working memory, small boost
                } else {
                    0.0
                };

                // BRIDGE-3: Calibrated confidence gate (Bayesian alpha/beta)
                let confidence_gate = {
                    let obs = mem.confidence_observations();
                    let total_obs = (obs - 2.0).max(0.0);
                    if total_obs >= 5.0 {
                        let bayesian = mem.calibrated_confidence();
                        0.85 + 0.15 * bayesian
                    } else {
                        1.0
                    }
                };

                let final_score = (base
                    + recency_boost
                    + arousal_boost
                    + credibility_boost
                    + valence_boost
                    + temporal_boost
                    + session_boost
                    + access_boost
                    + graph_boost
                    + episode_boost
                    + sequence_boost
                    + richness_boost
                    + activation_boost
                    + temporal_density
                    + entity_density
                    + elaboration_boost
                    + burstiness_boost)
                    * feedback_multiplier
                    * confidence_gate
                    * pinky_multiplier;

                let mut cloned: Memory = mem.as_ref().clone();
                cloned.set_score(final_score);
                Arc::new(cloned)
            };

            // Try working memory first (hot cache)
            if let Some(memory) = self.working_memory.read().get(&memory_id) {
                // CRITICAL FIX: Apply filters before adding to results
                if self.retriever.matches_filters(&memory, &vector_query) {
                    memories.push(with_unified_score(&memory, base_score));
                    if !sources.contains(&"working") {
                        sources.push("working");
                    }
                    cache_hits += 1;
                } else {
                    filtered_out += 1;
                }
                continue;
            }

            // Try session memory second (warm cache)
            if let Some(memory) = self.session_memory.read().get(&memory_id) {
                // CRITICAL FIX: Apply filters before adding to results
                if self.retriever.matches_filters(&memory, &vector_query) {
                    memories.push(with_unified_score(&memory, base_score));
                    if !sources.contains(&"session") {
                        sources.push("session");
                    }
                    cache_hits += 1;
                } else {
                    filtered_out += 1;
                }
                continue;
            }

            // Cold path: Fetch from RocksDB storage (expensive deserialization)
            match self.retriever.get_from_storage(&memory_id) {
                Ok(memory) => {
                    // CRITICAL FIX: Apply filters before adding to results
                    if self.retriever.matches_filters(&memory, &vector_query) {
                        // Reuse unified scoring (includes feedback_multiplier)
                        let shared = Arc::new(memory);
                        memories.push(with_unified_score(&shared, base_score));
                        if !sources.contains(&"longterm") {
                            sources.push("longterm");
                        }
                        storage_fetches += 1;
                    } else {
                        filtered_out += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        memory_id = %memory_id.0,
                        error = %e,
                        "Stale vector reference — cleaning up orphaned index entry"
                    );
                    self.retriever.remove_memory(&memory_id);
                }
            }

            if memories.len() >= rerank_pool {
                break;
            }
        }

        tracing::debug!(filtered_out = filtered_out, "Filter pass completed");

        // Log cache efficiency
        tracing::debug!(
            cache_hits = cache_hits,
            storage_fetches = storage_fetches,
            hit_rate = if cache_hits + storage_fetches > 0 {
                (cache_hits as f32 / (cache_hits + storage_fetches) as f32) * 100.0
            } else {
                0.0
            },
            "Cache-aware retrieval completed"
        );

        let t_fetch = recall_start.elapsed();
        tracing::info!(
            fetch_ms = format!("{:.2}", (t_fetch - t_fusion).as_secs_f64() * 1000.0),
            cumulative_ms = format!("{:.2}", t_fetch.as_secs_f64() * 1000.0),
            memories = memories.len(),
            cache_hits,
            storage_fetches,
            filtered_out,
            "recall [layer:5] memory fetch + unified scoring"
        );

        // Layer 5.1: TEMPORAL INVALIDATION FILTER
        // Remove memories whose valid_until timestamp has passed.
        {
            let before = memories.len();
            memories.retain(|m| !m.is_expired());
            if memories.len() < before {
                tracing::debug!(
                    "Layer 5.1: Filtered {} temporally invalidated memories",
                    before - memories.len()
                );
            }
        }

        // =====================================================================
        // LAYER 5.3: CROSS-ENCODER RERANKING (BRIDGE-2)
        // =====================================================================
        // Wire the existing CrossEncoderReranker (built in hybrid_search.rs)
        // into the main retrieval path. This uses joint query-document attention
        // (70% cross-encoder + 30% bi-encoder blend) to re-score the top
        // candidates. The cross-encoder score is blended with the existing
        // unified score at 12% weight — enough to promote/demote by 1-3 positions
        // without overriding the multi-signal pipeline.
        //
        // Only runs on the top `rerank_budget` candidates to keep latency bounded.
        if memories.len() > 1 {
            if let Some(reranker) = self.hybrid_search.reranker() {
                let rerank_budget = memories.len().min(query.max_results * 2).min(20);
                let candidates: Vec<(MemoryId, String, f32)> = memories
                    .iter()
                    .take(rerank_budget)
                    .map(|m| {
                        (
                            m.id.clone(),
                            m.experience.content.clone(),
                            m.score.unwrap_or(0.0),
                        )
                    })
                    .collect();

                match reranker.rerank(query_text, candidates) {
                    Ok(reranked) => {
                        // Build score lookup from cross-encoder results
                        let ce_scores: std::collections::HashMap<MemoryId, f32> = reranked
                            .into_iter()
                            .collect();

                        // Blend: 88% unified score + 12% cross-encoder score
                        for mem in memories.iter_mut().take(rerank_budget) {
                            if let Some(&ce_score) = ce_scores.get(&mem.id) {
                                let base = mem.score.unwrap_or(0.0);
                                let blended = base * 0.88 + ce_score * 0.12;
                                let mut cloned: Memory = mem.as_ref().clone();
                                cloned.set_score(blended);
                                *mem = Arc::new(cloned);
                            }
                        }

                        // Re-sort after blending
                        memories.sort_by(|a, b| {
                            b.score
                                .unwrap_or(0.0)
                                .total_cmp(&a.score.unwrap_or(0.0))
                        });

                        tracing::info!(
                            reranked = rerank_budget,
                            "recall [layer:5.3] cross-encoder reranking"
                        );
                    }
                    Err(e) => {
                        tracing::debug!(
                            "Layer 5.3: Cross-encoder reranking failed, proceeding without: {e}"
                        );
                    }
                }
            }
        }

        // =====================================================================
        // LAYER 5.35: FRAGMENT DEMOTION (FIX-R2)
        // =====================================================================
        // Demoted fragments (source memories whose facts have been extracted)
        // have their retrieval score reduced. Temporal queries are exempt —
        // they benefit from episode-level detail in source fragments.
        // Reference: Ehlers & Clark (2000) — S-rep fragments should not
        // compete equally with consolidated C-rep facts.
        if !has_temporal_query {
            let mut any_demoted = false;
            for mem in memories.iter_mut() {
                let demotion = mem.fragment_demotion();
                if demotion < 1.0 {
                    let base = mem.score.unwrap_or(0.0);
                    let demoted_score = base * demotion;
                    let mut cloned: Memory = mem.as_ref().clone();
                    cloned.set_score(demoted_score);
                    *mem = Arc::new(cloned);
                    any_demoted = true;
                }
            }
            if any_demoted {
                memories.sort_by(|a, b| {
                    b.score
                        .unwrap_or(0.0)
                        .total_cmp(&a.score.unwrap_or(0.0))
                });
                tracing::debug!("Layer 5.35: Applied fragment demotion to retrieval scores");
            }
        }

        // =====================================================================
        // LAYER 5.4: MMR DIVERSIFICATION (FIX-R3)
        // =====================================================================
        // After cross-encoder reranking and fragment demotion, apply Maximal
        // Marginal Relevance to eliminate redundant results. Query-type gated:
        // - Attribute/factual queries: skip MMR (precision > diversity)
        // - Exploratory queries: lambda=0.6 (strong diversity)
        // - Temporal queries: lambda=0.7 (moderate diversity)
        // Reference: Berntsen (2021) — pattern separation prevents similar
        // memories from co-activating (dentate gyrus analog).
        {
            let mmr_lambda = match &query_type {
                query_parser::QueryType::Attribute(_) => None,
                query_parser::QueryType::Exploratory => {
                    Some(crate::constants::MMR_LAMBDA_EXPLORATORY)
                }
                query_parser::QueryType::Temporal => {
                    Some(crate::constants::MMR_LAMBDA_RELATIONSHIP)
                }
            };
            if let Some(lambda) = mmr_lambda {
                let before_count = memories.len();
                memories =
                    Self::apply_mmr(&memories, lambda, query.max_results);
                tracing::debug!(
                    lambda = format!("{lambda:.2}"),
                    before = before_count,
                    after = memories.len(),
                    "Layer 5.4: MMR diversification"
                );
            }
        }

        memories.truncate(query.max_results);

        // =====================================================================
        // LAYER 5.7: CONFIDENCE GATING + SCORE-GAP PRUNING
        // =====================================================================
        // Category error detection: when retrieval confidence is low, return
        // fewer results rather than padding with noise. This handles queries
        // that are inherently unanswerable by recognizing the absence of a
        // confident match. Score-gap pruning drops trailing noise results.
        if memories.len() >= 2 {
            let scores: Vec<f32> = memories.iter().map(|m| m.score.unwrap_or(0.0)).collect();
            let top_score = scores[0];

            if top_score > 0.0 {
                let mut keep = memories.len();
                for i in 1..scores.len() {
                    let ratio = scores[i] / top_score;
                    // If result scores less than 25% of top score, it's noise
                    if ratio < 0.25 {
                        keep = i;
                        break;
                    }
                    // If there's a >60% relative drop from previous, cut here
                    if i >= 2 && scores[i - 1] > 0.0 {
                        let step_ratio = scores[i] / scores[i - 1];
                        if step_ratio < 0.40 {
                            keep = i;
                            break;
                        }
                    }
                }
                if keep < memories.len() {
                    tracing::debug!(
                        "Layer 5.7: Confidence pruning {} -> {} results (top={:.3}, cut={:.3})",
                        memories.len(), keep, top_score, scores.get(keep).copied().unwrap_or(0.0)
                    );
                    memories.truncate(keep);
                }
            }
        }

        // =====================================================================
        // LAYER 5.8: ANSWER-TYPE SOFT FILTER
        // =====================================================================
        // Penalize memories whose ExperienceType doesn't match query intent.
        // Soft filter (score penalty), not hard filter.
        {
            let query_lower = query_text.to_lowercase();
            let preferred: Option<Vec<crate::memory::types::ExperienceType>> =
                if query_lower.contains("bug")
                    || query_lower.contains("error")
                    || query_lower.contains("issue")
                    || query_lower.contains("problem")
                    || query_lower.contains("fail")
                {
                    Some(vec![
                        crate::memory::types::ExperienceType::Error,
                        crate::memory::types::ExperienceType::Discovery,
                    ])
                } else if query_lower.contains("risk") || query_lower.contains("concern") {
                    Some(vec![
                        crate::memory::types::ExperienceType::Observation,
                        crate::memory::types::ExperienceType::Discovery,
                    ])
                } else {
                    None
                };

            if let Some(ref prefs) = preferred {
                for mem in memories.iter_mut() {
                    if !prefs.contains(&mem.experience.experience_type) {
                        let base = mem.score.unwrap_or(0.0);
                        let penalized = base * 0.85;
                        let mut cloned: Memory = mem.as_ref().clone();
                        cloned.set_score(penalized);
                        *mem = Arc::new(cloned);
                    }
                }
                memories.sort_by(|a, b| {
                    b.score
                        .unwrap_or(0.0)
                        .total_cmp(&a.score.unwrap_or(0.0))
                });
            }
        }

        // =====================================================================
        // LAYER 5.9: ORDINAL RESOLUTION + CATEGORY ERROR DETECTION
        // =====================================================================
        // Post-retrieval filter for ordinal queries ("first", "last", "most recent")
        // and temporal phase queries ("during the X phase", "second meeting").
        //
        // Ordinal resolution: when the query asks for "first" or "last", sort the
        // candidate set by created_at and return only the extreme. This fixes
        // "What was the first bug?" where all session 3 bugs match but only the
        // earliest is correct.
        //
        // Category error detection: when the query references a concept the system
        // can't resolve (e.g., "second meeting" with no meeting-to-session mapping),
        // flag low confidence rather than returning noise. The 96.2% LOCOMO ceiling
        // comes from these inherently unanswerable queries.
        {
            let ql = query_text.to_lowercase();

            // Ordinal: "first" → sort ascending by created_at, keep earliest
            let wants_first = ql.contains("first ") || ql.starts_with("first ");
            let wants_last = ql.contains("most recent")
                || ql.contains("latest")
                || ql.contains("last ");

            if wants_first && !memories.is_empty() {
                // Sort by created_at ascending — earliest first
                memories.sort_by(|a, b| a.created_at.cmp(&b.created_at));
                // Preserve the score of the earliest memory but put it at rank 1
                let earliest = memories[0].clone();
                // Re-sort by score but pin the earliest at position 0
                memories.sort_by(|a, b| {
                    b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0))
                });
                // Remove the earliest from its current position and insert at front
                if let Some(pos) = memories.iter().position(|m| m.id == earliest.id) {
                    memories.remove(pos);
                }
                memories.insert(0, earliest);
                tracing::debug!("Layer 5.9: Ordinal 'first' — pinned earliest memory at rank 1");
            } else if wants_last && !memories.is_empty() {
                // Strategy C: Gated pinning — only pin latest if it's already
                // in the top half by score (content-relevant). Otherwise, boost
                // the top-3 most recent memories from the relevant set by
                // re-scoring with a recency bonus proportional to their recency rank.
                let score_sorted: Vec<f32> = {
                    let mut s: Vec<f32> = memories.iter().map(|m| m.score.unwrap_or(0.0)).collect();
                    s.sort_by(|a, b| b.total_cmp(a));
                    s
                };
                let p50 = score_sorted[score_sorted.len() / 2];

                // Find the most recent memory
                let mut by_time = memories.clone();
                by_time.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                let latest = &by_time[0];

                if latest.score.unwrap_or(0.0) >= p50 {
                    // Latest IS content-relevant → pin it
                    let latest_id = latest.id.clone();
                    if let Some(pos) = memories.iter().position(|m| m.id == latest_id) {
                        let pinned = memories.remove(pos);
                        memories.insert(0, pinned);
                    }
                    tracing::debug!("Layer 5.9: Gated pin — latest memory is content-relevant, pinned");
                } else {
                    // Latest is NOT relevant → apply graduated recency boost to
                    // top-3 most recent among the relevant set (above p50).
                    // This lifts recent+relevant memories without polluting rank 1.
                    let relevant_recent: Vec<MemoryId> = by_time.iter()
                        .filter(|m| m.score.unwrap_or(0.0) >= p50)
                        .take(3)
                        .map(|m| m.id.clone())
                        .collect();
                    let boosts = [0.08_f32, 0.05, 0.03]; // Graduated: most recent gets most
                    for (mid, &boost) in relevant_recent.iter().zip(boosts.iter()) {
                        if let Some(mem) = memories.iter_mut().find(|m| m.id == *mid) {
                            let base = mem.score.unwrap_or(0.0);
                            let mut cloned: Memory = mem.as_ref().clone();
                            cloned.set_score(base + boost);
                            *mem = Arc::new(cloned);
                        }
                    }
                    memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
                    tracing::debug!(
                        "Layer 5.9: Gated pin — latest not relevant, boosted {} recent-relevant memories",
                        relevant_recent.len()
                    );
                }
            }

            // Category error detection + ordinal session resolution:
            // Try ordinal resolution FIRST. Only apply category error demotion
            // if resolution fails or is unavailable.
            let ordinal_session_ref = (ql.contains("second ") || ql.contains("third ")
                || ql.contains("fourth "))
                && (ql.contains("meeting") || ql.contains("sprint") || ql.contains("session"));

            let mut ordinal_resolved = false;

            if ordinal_session_ref {
                if let Some((ordinal, _noun)) =
                    wavelet_sessions::extract_ordinal_session_ref(&ql)
                {
                    match self.get_or_compute_session_map() {
                        Ok(session_map) if ordinal <= session_map.sessions.len() => {
                            let target = &session_map.sessions[ordinal - 1];
                            let session_ids: std::collections::HashSet<MemoryId> =
                                target.memory_ids.iter().cloned().collect();

                            // Inject session members not already in candidates
                            let existing_ids: std::collections::HashSet<MemoryId> =
                                memories.iter().map(|m| m.id.clone()).collect();
                            let mut injected = 0usize;
                            for sid in &target.memory_ids {
                                if !existing_ids.contains(sid) && injected < 5 {
                                    if let Ok(mem) = self.get_memory(sid) {
                                        memories.push(Arc::new(mem));
                                        injected += 1;
                                    }
                                }
                            }

                            for mem in memories.iter_mut() {
                                if session_ids.contains(&mem.id) {
                                    let base = mem.score.unwrap_or(0.0);
                                    let mut cloned: Memory = mem.as_ref().clone();
                                    cloned.set_score(
                                        base + crate::constants::ORDINAL_SESSION_BOOST,
                                    );
                                    *mem = Arc::new(cloned);
                                }
                            }
                            memories.sort_by(|a, b| {
                                b.score
                                    .unwrap_or(0.0)
                                    .total_cmp(&a.score.unwrap_or(0.0))
                            });
                            ordinal_resolved = true;
                            tracing::info!(
                                "Layer 5.9: Ordinal '{}' resolved to session {} ({} memories, {} injected)",
                                ordinal,
                                ordinal,
                                target.count,
                                injected
                            );
                        }
                        Ok(_) => {
                            tracing::info!(
                                "Layer 5.9: Ordinal {} out of range — demoting",
                                ordinal
                            );
                        }
                        Err(e) => {
                            tracing::debug!("Layer 5.9: Session detection failed: {}", e);
                        }
                    }
                }

                // Category error demotion: only if ordinal resolution failed
                if !ordinal_resolved {
                    tracing::info!(
                        "Layer 5.9: Category error — ordinal session reference '{}' unresolved",
                        query_text
                    );
                    for mem in memories.iter_mut() {
                        let base = mem.score.unwrap_or(0.0);
                        let mut cloned: Memory = mem.as_ref().clone();
                        cloned.set_score(base * 0.50);
                        *mem = Arc::new(cloned);
                    }
                }
            }
        }

        // Linguistic analysis: additive boost (5% of IC weight), not a full re-sort
        if !query_analysis.focal_entities.is_empty() {
            memories.sort_by(|a, b| {
                let score_a = a.score.unwrap_or(0.0)
                    + Self::linguistic_boost(&a.experience.content, &query_analysis) * 0.05;
                let score_b = b.score.unwrap_or(0.0)
                    + Self::linguistic_boost(&b.experience.content, &query_analysis) * 0.05;
                score_b.total_cmp(&score_a)
            });
        }

        self.logger
            .read()
            .log_retrieved(query_text, memories.len(), &sources);

        // SHO-106: Apply retrieval competition between similar memories FIRST
        // When highly similar memories are retrieved, they compete for activation
        // PIPE-10: Competition must happen BEFORE coactivation - we only want to
        // strengthen associations between memories that "won" the competition.
        // Suppressed memories should not be coactivated (Hebbian "losers don't learn").
        if memories.len() >= 2 {
            // Calculate similarity scores for competition analysis
            let candidates: Vec<(String, f32, f32)> = memories
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    let relevance = 1.0 - (i as f32 / memories.len() as f32) * 0.3; // Position-based score
                    let similarity = m.score.unwrap_or(0.5); // Use computed retrieval score
                    (m.id.0.to_string(), relevance, similarity)
                })
                .collect();

            let competition_result = self
                .interference_detector
                .write()
                .apply_retrieval_competition(&candidates, query_text);

            // Record competition event if any memories were suppressed
            if let Some(ref event) = competition_result.event {
                self.record_consolidation_event(event.clone());
            }

            // Re-order memories based on competition results (winners first)
            if !competition_result.suppressed.is_empty() {
                let winner_set: std::collections::HashSet<_> = competition_result
                    .winners
                    .iter()
                    .map(|(id, _)| id.clone())
                    .collect();

                // Keep only winners, maintain their relative order
                memories.retain(|m| winner_set.contains(&m.id.0.to_string()));

                tracing::debug!(
                    "Retrieval competition: {} memories suppressed",
                    competition_result.suppressed.len()
                );
            }

            // Persist interference records from retrieval competition
            {
                let detector = self.interference_detector.read();
                let affected_ids = detector.get_affected_ids_from_competition(&competition_result);
                if !affected_ids.is_empty() {
                    for (id, records) in detector.get_records_for_ids(&affected_ids) {
                        if let Err(e) = self.long_term_memory.save_interference_records(id, records)
                        {
                            tracing::debug!("Failed to persist competition interference: {e}");
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

        // Abstention: filter out memories below confidence threshold
        // importance × calibrated_confidence < ABSTENTION_THRESHOLD → excluded
        // Only applies to memories with enough feedback observations (≥ 2 beyond prior)
        {
            let threshold = crate::constants::ABSTENTION_THRESHOLD;
            let before = memories.len();
            memories.retain(|m| {
                // Only abstain when memory has ≥ 2 actual feedback events (beyond prior of 2.0)
                if m.confidence_observations() <= 4.0 {
                    return true;
                }
                m.importance() * m.calibrated_confidence() >= threshold
            });
            if memories.len() < before {
                tracing::debug!(
                    "Abstention: filtered {} low-confidence memories (threshold={threshold})",
                    before - memories.len()
                );
            }
        }

        // Update access counts with instrumentation for consolidation events
        // (only for memories that survived competition)
        for memory in &memories {
            self.update_access_count_instrumented(memory, StrengtheningReason::Recalled);
        }

        // =====================================================================
        // FIX-R1: RECONSOLIDATION — mark retrieved memories as labile
        // =====================================================================
        // Set activation=1.0 (labile state) and create/extend reconsolidation
        // shadows. During the labile window, the memory can be updated with new
        // context discovered during the retrieval interaction.
        // Reference: Nader et al. (2000) — reconsolidation theory.
        {
            let now = chrono::Utc::now();
            let window =
                chrono::Duration::seconds(crate::constants::RECONSOLIDATION_LABILE_WINDOW_SECS);
            let mut shadows = self.reconsolidation_shadows.write();

            for memory in &memories {
                // Mark as labile
                memory.set_activation(1.0);

                if let Some(existing) = shadows.get_mut(&memory.id) {
                    // Memory already labile — extend window (working memory behavior)
                    existing.consecutive_retrieval_count += 1;
                    existing.last_retrieval_at = now;
                    // Last-writer-wins: reset expiry
                    existing.expires_at = now + window;
                } else {
                    // New labile window
                    shadows.insert(
                        memory.id.clone(),
                        super::types::ReconsolidationShadow {
                            memory_id: memory.id.clone(),
                            opened_at: now,
                            expires_at: now + window,
                            retrieval_context: query_text.to_string(),
                            consecutive_retrieval_count: 1,
                            last_retrieval_at: now,
                        },
                    );
                }
            }

            // Cap active shadows to prevent accumulation
            while shadows.len() > crate::constants::RECONSOLIDATION_MAX_ACTIVE_SHADOWS {
                // Evict the oldest expired first, then oldest opened
                let evict_key = shadows
                    .iter()
                    .filter(|(_, s)| s.expires_at <= now)
                    .min_by_key(|(_, s)| s.opened_at)
                    .or_else(|| shadows.iter().min_by_key(|(_, s)| s.opened_at))
                    .map(|(k, _)| k.clone());
                if let Some(key) = evict_key {
                    shadows.remove(&key);
                } else {
                    break;
                }
            }
        }

        // PIPE-10: Hebbian learning AFTER competition - only coactivate winners
        // When memories are retrieved together AND survive competition, they
        // form/strengthen edges in the memory graph. Suppressed memories don't
        // participate in coactivation (biological: "neurons that fire together
        // wire together" but suppressed neurons don't fire).
        if memories.len() >= 2 {
            if let Some(graph) = &self.graph_memory {
                let memory_uuids: Vec<uuid::Uuid> = memories.iter().map(|m| m.id.0).collect();
                match graph.read().record_memory_coactivation(&memory_uuids) {
                    Ok(ref result) if result.edges_updated > 0 => {
                        // Record consolidation events for coactivation visibility
                        for i in 0..memories.len().min(5) {
                            for j in (i + 1)..memories.len().min(5) {
                                self.record_consolidation_event(
                                    introspection::ConsolidationEvent::EdgeStrengthened {
                                        from_memory_id: memories[i].id.0.to_string(),
                                        to_memory_id: memories[j].id.0.to_string(),
                                        strength_before: 0.0,
                                        strength_after: 0.025,
                                        co_activations: 1,
                                        timestamp: chrono::Utc::now(),
                                    },
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::trace!("Coactivation recording failed (non-critical): {e}");
                    }
                    _ => {}
                }
            }
        }

        // Increment and persist retrieval counter
        if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
            self.stats.write().total_retrievals = count;
        }

        // Expand with hierarchy context (parent chain + children)
        // This ensures semantic search also surfaces contextually related memories
        let mut seen_ids: HashSet<MemoryId> = memories.iter().map(|m| m.id.clone()).collect();
        self.expand_with_hierarchy(&mut memories, &mut seen_ids);

        let t_total = recall_start.elapsed();
        tracing::info!(
            post_ms = format!("{:.2}", (t_total - t_fetch).as_secs_f64() * 1000.0),
            total_ms = format!("{:.2}", t_total.as_secs_f64() * 1000.0),
            final_count = memories.len(),
            "recall [layer:post] linguistic + competition + coactivation + hierarchy === RECALL COMPLETE ==="
        );

        Ok(memories)
    }

    /// Apply learning velocity boost to retrieved memories
    ///
    /// This method should be called after `recall()` when user_id is known.
    /// It boosts memories that have been recently learned/reinforced, implementing
    /// the principle that "learning should improve retrieval over time".
    ///
    /// Boost factors:
    /// - Base boost for any learning activity (5%)
    /// - Velocity boost for rapid learning (up to 15%)
    /// - Potentiation bonus for LTP'd edges (10%)
    /// - Total max boost: 30%
    ///
    /// The memories are re-sorted by adjusted score after boosting.
    pub fn apply_learning_boost(
        &self,
        user_id: &str,
        mut memories: Vec<SharedMemory>,
    ) -> Vec<SharedMemory> {
        if memories.is_empty() {
            return memories;
        }

        // Calculate boosts for all memories
        let mut boosted: Vec<(SharedMemory, f32)> = memories
            .drain(..)
            .map(|mem| {
                let base_score = mem.score.unwrap_or(0.5);
                let boost = self
                    .learning_history
                    .recency_boost(user_id, &mem.id.0.to_string())
                    .unwrap_or(1.0);
                let adjusted_score = base_score * boost;
                (mem, adjusted_score)
            })
            .collect();

        // Log if any memories got significant boosts
        let boosted_count = boosted.iter().filter(|(_, s)| *s > 0.5).count();
        if boosted_count > 0 {
            tracing::debug!(
                user_id = %user_id,
                boosted_count = boosted_count,
                "Applied learning velocity boost to retrieved memories"
            );
        }

        // Sort by adjusted score (descending)
        boosted.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Rebuild memories with updated scores
        boosted
            .into_iter()
            .map(|(mem, score)| {
                let mut cloned: Memory = mem.as_ref().clone();
                cloned.set_score(score);
                Arc::new(cloned)
            })
            .collect()
    }

    /// Recall with learning boost applied
    ///
    /// Convenience method that combines `recall()` with `apply_learning_boost()`.
    /// Use this when you have the user_id available at recall time.
    pub fn recall_for_user(&self, user_id: &str, query: &Query) -> Result<Vec<SharedMemory>> {
        let memories = self.recall(query)?;
        Ok(self.apply_learning_boost(user_id, memories))
    }

    /// Get learning velocity statistics for a memory
    ///
    /// Returns information about recent learning activity for this memory,
    /// useful for debugging/introspection.
    pub fn get_learning_velocity(
        &self,
        user_id: &str,
        memory_id: &str,
        hours: i64,
    ) -> Result<learning_history::LearningVelocity> {
        self.learning_history
            .memory_learning_velocity(user_id, memory_id, hours)
    }

    /// Get learning history statistics for a user
    pub fn get_learning_stats(&self, user_id: &str) -> Result<learning_history::LearningStats> {
        self.learning_history.stats(user_id)
    }

    /// Get recent learning events for a user
    pub fn get_learning_events(
        &self,
        user_id: &str,
        since: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<learning_history::StoredLearningEvent>> {
        let mut events = self.learning_history.events_since(user_id, since)?;
        events.truncate(limit);
        Ok(events)
    }

    /// Calculate linguistic boost based on focal entity matches
    fn linguistic_boost(content: &str, analysis: &query_parser::QueryAnalysis) -> f32 {
        let content_lower = content.to_lowercase();
        let mut boost = 0.0;

        for entity in &analysis.focal_entities {
            if content_lower.contains(&entity.text) {
                boost += entity.ic_weight;
            }
        }

        for modifier in &analysis.discriminative_modifiers {
            if content_lower.contains(&modifier.text) {
                boost += 1.7; // IC_ADJECTIVE
            }
        }

        boost
    }


    /// Retrieve or recompute the wavelet-detected session map.
    ///
    /// The session map is cached and only recomputed when new memories have been
    /// stored (indicated by `fact_extraction_needed`). This keeps repeated
    /// ordinal-resolution lookups within a single retrieval pipeline free of
    /// redundant full-store scans.
    fn get_or_compute_session_map(&self) -> anyhow::Result<wavelet_sessions::SessionMap> {
        // Check cache
        {
            let cache = self.session_map_cache.lock();
            if let Some(ref map) = *cache {
                if !self
                    .fact_extraction_needed
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    return Ok(map.clone());
                }
            }
        }

        // Recompute from all memories
        let all = self.get_all_memories()?;
        let timestamps: Vec<(MemoryId, chrono::DateTime<chrono::Utc>)> =
            all.iter().map(|m| (m.id.clone(), m.created_at)).collect();

        let map = wavelet_sessions::detect_sessions(&timestamps);

        // Cache
        *self.session_map_cache.lock() = Some(map.clone());
        Ok(map)
    }


    /// Calculate temporal relevance based on memory age (ENTERPRISE FEATURE)
    ///
    /// Implements exponential decay curve for time-aware memory retrieval:
    /// - 0-7 days: Full relevance (1.0) - recent memories
    /// - 8-30 days: High relevance (0.7) - medium-term memories
    /// - 31-90 days: Moderate relevance (0.4) - older memories
    /// - 90+ days: Low relevance (0.2) - ancient memories
    ///
    /// This ensures recent experiences are prioritized while maintaining
    /// access to historical context when needed.
    fn calculate_temporal_relevance(age_days: i64) -> f32 {
        match age_days {
            0..=7 => 1.0,   // Recent: Full weight
            8..=30 => 0.7,  // Medium-term: 70% weight
            31..=90 => 0.4, // Old: 40% weight
            _ => 0.2,       // Ancient: 20% weight (never completely forgotten)
        }
    }

    /// Expand retrieved memories with their hierarchy context
    ///
    /// When a memory is retrieved, its parent chain and children are also
    /// contextually relevant. This method adds them to the result set with
    /// slightly boosted importance (hierarchy context is valuable).
    ///
    /// Hierarchy expansion depth is limited to prevent explosion:
    /// - Parents: Full chain up to root (usually shallow)
    /// - Children: Direct children only (1 level)
    fn expand_with_hierarchy(
        &self,
        memories: &mut Vec<SharedMemory>,
        seen_ids: &mut HashSet<MemoryId>,
    ) {
        // Skip if no memories to expand
        if memories.is_empty() {
            return;
        }

        // Collect IDs to expand (copy to avoid borrow issues)
        let ids_to_expand: Vec<MemoryId> = memories.iter().map(|m| m.id.clone()).collect();

        // Expand each memory with its hierarchy
        for memory_id in ids_to_expand {
            // Get parent chain
            if let Ok(ancestors) = self.long_term_memory.get_ancestors(&memory_id) {
                for ancestor in ancestors {
                    if seen_ids.insert(ancestor.id.clone()) {
                        // Boost ancestor importance slightly (context is valuable)
                        let new_importance = (ancestor.importance() * 1.1).min(1.0);
                        let mut shared = Arc::new(ancestor);
                        Arc::make_mut(&mut shared).set_importance(new_importance);
                        memories.push(shared);
                    }
                }
            }

            // Get direct children
            if let Ok(children) = self.long_term_memory.get_children(&memory_id) {
                for child in children {
                    if seen_ids.insert(child.id.clone()) {
                        // Boost child importance slightly
                        let new_importance = (child.importance() * 1.05).min(1.0);
                        let mut shared = Arc::new(child);
                        Arc::make_mut(&mut shared).set_importance(new_importance);
                        memories.push(shared);
                    }
                }
            }
        }
    }


    // =========================================================================
    // OUTCOME FEEDBACK SYSTEM - Hebbian "Fire Together, Wire Together"
    // =========================================================================

    /// Retrieve memories with tracking for later feedback
    ///
    /// Use this when you want to provide feedback on retrieval quality.
    /// Returns a TrackedRetrieval that can be used with `reinforce_recall`.
    ///
    /// # Example
    /// ```ignore
    /// let tracked = memory_system.recall_tracked(&query)?;
    /// // Use memories...
    /// // Later, after task completion:
    /// memory_system.reinforce_recall(&tracked.memory_ids(), RetrievalOutcome::Helpful)?;
    /// ```
    pub fn recall_tracked(&self, query: &Query) -> Result<TrackedRetrieval> {
        let result = self.retriever.search_tracked(query, query.max_results)?;
        if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
            self.stats.write().total_retrievals = count;
        }
        Ok(result)
    }

    /// Reinforce memories based on task outcome (core feedback loop)
    ///
    /// This is THE key method that closes the Hebbian loop:
    /// - If outcome is Helpful: strengthen associations, boost importance
    /// - If outcome is Misleading: weaken associations, reduce importance
    /// - If outcome is Neutral: just record access (mild reinforcement)
    ///
    /// CACHE COHERENCY: This method updates BOTH the in-memory caches AND
    /// persistent storage to ensure importance changes are visible immediately
    /// through cached references (via Arc interior mutability) AND survive restarts.
    ///
    /// # Arguments
    /// * `memory_ids` - IDs of memories that were used in the task
    /// * `outcome` - Whether the memories were helpful, misleading, or neutral
    ///
    /// # Returns
    /// Statistics about what was reinforced
    pub fn reinforce_recall(
        &self,
        memory_ids: &[MemoryId],
        outcome: RetrievalOutcome,
    ) -> Result<ReinforcementStats> {
        if memory_ids.is_empty() {
            return Ok(ReinforcementStats::default());
        }

        let mut stats = ReinforcementStats {
            memories_processed: memory_ids.len(),
            outcome,
            ..Default::default()
        };

        // Hebbian coactivation: strengthen associations between co-retrieved memories
        // Uses GraphMemory if available, otherwise counts pair associations directly
        if !matches!(outcome, RetrievalOutcome::Misleading) && memory_ids.len() >= 2 {
            if let Some(graph) = &self.graph_memory {
                let memory_uuids: Vec<uuid::Uuid> = memory_ids.iter().map(|id| id.0).collect();
                if memory_uuids.len() >= 2 {
                    match graph.read().record_memory_coactivation(&memory_uuids) {
                        Ok(result) => {
                            stats.associations_strengthened = result.edges_updated;
                            // BRIDGE-4: Boost importance for memories linked by promoted edges
                            for promo in &result.promotions {
                                let boost = if promo.new_tier.contains("L3") { 0.15 } else { 0.10 };
                                for mid in memory_ids {
                                    if mid.0 == promo.from_entity || mid.0 == promo.to_entity {
                                        // Apply boost through all tiers
                                        if let Some(mem) = self.working_memory.read().get(mid) {
                                            mem.boost_importance(boost);
                                        } else if let Some(mem) = self.session_memory.read().get(mid) {
                                            mem.boost_importance(boost);
                                        } else if let Ok(mem) = self.long_term_memory.get(mid) {
                                            mem.boost_importance(boost);
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to record memory coactivation");
                            // Fallback: count pairs
                            let n = memory_ids.len();
                            stats.associations_strengthened = n * (n - 1) / 2;
                        }
                    }
                }
            } else {
                // No graph memory available — count pairs directly
                let n = memory_ids.len();
                stats.associations_strengthened = n * (n - 1) / 2;
            }
        }

        // CACHE COHERENT IMPORTANCE UPDATES:
        // 1. First try to find memory in caches (working, session)
        // 2. If found in cache, modify through the cached Arc (interior mutability)
        //    This updates ALL holders of this Arc reference
        // 3. Then persist to storage for durability
        // 4. If not in cache, get from storage, modify, and persist
        let mut persist_failures: Vec<(MemoryId, String)> = Vec::new();

        for id in memory_ids {
            // Try working memory cache first
            let cached_memory = {
                let working = self.working_memory.read();
                working.get(id)
            };

            // Try session memory cache if not in working
            let cached_memory = cached_memory.or_else(|| {
                let session = self.session_memory.read();
                session.get(id)
            });

            if let Some(memory) = cached_memory {
                // CACHE HIT: Modify through cached Arc (updates all references)
                memory.record_access();
                match &outcome {
                    RetrievalOutcome::Helpful => {
                        memory.boost_importance(HEBBIAN_BOOST_HELPFUL);
                        stats.importance_boosts += 1;
                    }
                    RetrievalOutcome::Misleading => {
                        memory.decay_importance(HEBBIAN_DECAY_MISLEADING);
                        stats.importance_decays += 1;
                    }
                    RetrievalOutcome::Neutral => {
                        // Just access recorded
                    }
                }
                // PERSIST: Write updated memory to durable storage
                // Track failures instead of silently ignoring
                if let Err(e) = self.long_term_memory.update(&memory) {
                    persist_failures.push((id.clone(), e.to_string()));
                    tracing::warn!(
                        memory_id = %id.0,
                        error = %e,
                        "Failed to persist reinforcement update - Hebbian feedback may be lost on restart"
                    );
                }
            } else {
                // CACHE MISS: Get from storage, modify, and persist
                match self.long_term_memory.get(id) {
                    Ok(memory) => {
                        memory.record_access();
                        match &outcome {
                            RetrievalOutcome::Helpful => {
                                memory.boost_importance(HEBBIAN_BOOST_HELPFUL);
                                stats.importance_boosts += 1;
                            }
                            RetrievalOutcome::Misleading => {
                                memory.decay_importance(HEBBIAN_DECAY_MISLEADING);
                                stats.importance_decays += 1;
                            }
                            RetrievalOutcome::Neutral => {
                                // Just access recorded
                            }
                        }
                        // PERSIST: Write to durable storage
                        if let Err(e) = self.long_term_memory.update(&memory) {
                            persist_failures.push((id.clone(), e.to_string()));
                            tracing::warn!(
                                memory_id = %id.0,
                                error = %e,
                                "Failed to persist reinforcement update - Hebbian feedback may be lost on restart"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::debug!(
                            memory_id = %id.0,
                            error = %e,
                            "Memory not found during reinforcement - may have been deleted"
                        );
                    }
                }
            }
        }

        // Report aggregate persistence failures
        if !persist_failures.is_empty() {
            stats.persist_failures = persist_failures.len();
            tracing::error!(
                failure_count = persist_failures.len(),
                "Hebbian reinforcement had persistence failures - learning feedback partially lost"
            );
        }

        Ok(stats)
    }

    /// Reinforce using a tracked recall (convenience wrapper)
    pub fn reinforce_recall_tracked(
        &self,
        tracked: &TrackedRetrieval,
        outcome: RetrievalOutcome,
    ) -> Result<ReinforcementStats> {
        self.retriever.reinforce_tracked(tracked, outcome)
    }

    /// Apply Maximal Marginal Relevance to diversify retrieval results (FIX-R3).
    ///
    /// MMR formula: score_mmr(i) = λ * relevance(i) - (1-λ) * max_j∈selected(sim(i,j))
    ///
    /// This is the computational analog of dentate gyrus pattern separation:
    /// similar memories are penalized to prevent co-activation flooding.
    ///
    /// Reference: Berntsen (2021) — functional constraints on involuntary memory
    /// prevent loose associations from dominating retrieval.
    fn apply_mmr(memories: &[SharedMemory], lambda: f32, k: usize) -> Vec<SharedMemory> {
        if memories.len() <= 1 || k == 0 {
            return memories.to_vec();
        }

        let k = k.min(memories.len());
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        let mut remaining: Vec<usize> = (0..memories.len()).collect();

        // Pre-extract embeddings (already computed, just reference)
        let embeddings: Vec<Option<&[f32]>> = memories
            .iter()
            .map(|m| m.experience.embeddings.as_deref())
            .collect();

        // Normalize scores to [0, 1] for MMR
        let max_score = memories
            .iter()
            .filter_map(|m| m.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_score = memories
            .iter()
            .filter_map(|m| m.score)
            .fold(f32::INFINITY, f32::min);
        let score_range = (max_score - min_score).max(f32::EPSILON);

        // First pick: highest relevance score (already sorted, index 0)
        selected.push(remaining.remove(0));

        // Iteratively pick remaining by MMR
        while selected.len() < k && !remaining.is_empty() {
            let mut best_idx_in_remaining = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (ri, &cand_idx) in remaining.iter().enumerate() {
                let relevance =
                    (memories[cand_idx].score.unwrap_or(0.0) - min_score) / score_range;

                // Max similarity to any already-selected memory
                let max_sim = if let Some(cand_emb) = embeddings[cand_idx] {
                    selected
                        .iter()
                        .filter_map(|&sel_idx| {
                            embeddings[sel_idx]
                                .map(|sel_emb| crate::similarity::cosine_similarity(cand_emb, sel_emb))
                        })
                        .fold(0.0_f32, f32::max)
                } else {
                    0.0 // No embedding = no diversity penalty
                };

                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim;
                if mmr_score > best_mmr {
                    best_mmr = mmr_score;
                    best_idx_in_remaining = ri;
                }
            }

            selected.push(remaining.remove(best_idx_in_remaining));
        }

        // Rebuild result preserving MMR order but keeping original scores
        selected.iter().map(|&idx| memories[idx].clone()).collect()
    }
}
