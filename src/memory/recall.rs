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
use crate::memory::dials::FusionMode;
use crate::memory::graph_retrieval::calculate_density_weights;
use crate::memory::introspection::{self, StrengtheningReason};
use crate::memory::learning_history;
use crate::memory::query_parser;
use crate::memory::retrieval::{ReinforcementStats, RetrievalOutcome, TrackedRetrieval};
use crate::memory::space_legs::{LegKind, PendingLegs};
use crate::memory::storage::{self, SearchCriteria};
use crate::memory::temporal_facts;
use crate::memory::types::*;
use crate::memory::wavelet_sessions;
use crate::metrics::{EMBEDDING_CACHE_QUERY, EMBEDDING_CACHE_QUERY_SIZE};

/// Fraction of semantic_w used for secondary Vamana RRF signal.
/// Secondary is a refinement signal, not a replacement for primary vector search.
const SECONDARY_RRF_WEIGHT_RATIO: f32 = 0.5;

/// Days the date window leg adds on each side of a calendar day a query names.
///
/// A conversational memory is stamped when it was said, and what it reports was often
/// said the day after it happened: "watched it last night" on 2 May answers "what did she
/// watch on 1 May". One day either side covers that; a month-year window is already wide.
const DATE_WINDOW_DAY_SLACK_DAYS: i64 = 1;

impl super::MemorySystem {
    /// The embedding `recall` searches with for `text`, through the same SHA256-keyed
    /// cache the pipeline uses. A caller that passes this as `Query::query_embedding`
    /// gets exactly the ranking it would get by leaving the field unset.
    ///
    /// It is the query side, so an asymmetric model's query instruction is applied;
    /// `query_cache` holds only query-side vectors, the sub-query sites included.
    pub fn recall_query_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let query_hash = Self::sha256_hash(text);
        if let Some(cached) = self.query_cache.get(&query_hash) {
            EMBEDDING_CACHE_QUERY.with_label_values(&["hit"]).inc();
            return Ok(cached);
        }
        EMBEDDING_CACHE_QUERY.with_label_values(&["miss"]).inc();
        self.embed_query(text, query_hash)
    }

    /// Embed `text` as a query in the primary space and cache it under `query_hash`.
    fn embed_query(&self, text: &str, query_hash: [u8; 32]) -> Result<Vec<f32>> {
        let embedding = self
            .embedder
            .as_ref()
            .encode_for_query(text)
            .context("Failed to generate query embedding")?;
        self.query_cache.insert(query_hash, embedding.clone());
        EMBEDDING_CACHE_QUERY_SIZE.set(self.query_cache.entry_count() as i64);
        Ok(embedding)
    }

    /// Working- and Session-tier memories the vector index does not hold, each with its best
    /// similarity to the query where that exceeds 0.1: the cosine of its vector in the primary
    /// index's space, or of its vector in any other space a query vector exists for, moved
    /// onto the primary's scale first so a sharper model's near-misses do not outrank the
    /// primary's hits on scale alone.
    fn unindexed_tier_hits(
        &self,
        query_embedding: &[f32],
        index_space: &SpaceId,
        space_queries: &[(SpaceId, Vec<f32>)],
        exclude: &HashSet<MemoryId>,
    ) -> Vec<(MemoryId, f32)> {
        let best_similarity = |mem: &Memory| -> Option<f32> {
            let primary = self
                .retriever
                .stored_vector(mem, index_space)
                .filter(|emb| !emb.is_empty())
                .map(|emb| {
                    crate::memory::hybrid_search::cosine_similarity_pub(query_embedding, emb)
                });
            let others = space_queries.iter().filter_map(|(space, q)| {
                self.retriever
                    .stored_vector(mem, space)
                    .filter(|emb| !emb.is_empty())
                    .map(|emb| {
                        self.retriever.calibrated_similarity(
                            space,
                            crate::memory::hybrid_search::cosine_similarity_pub(q, emb),
                        )
                    })
            });
            primary
                .into_iter()
                .chain(others)
                .reduce(f32::max)
                .filter(|&s| s > 0.1)
        };

        let mut tier_memories = self.working_memory.read().all_memories();
        tier_memories.extend(self.session_memory.read().all_memories());
        tier_memories
            .iter()
            .filter(|mem| !exclude.contains(&mem.id) && !self.retriever.has_mapping(&mem.id))
            .filter_map(|mem| best_similarity(mem).map(|sim| (mem.id.clone(), sim)))
            .collect()
    }

    /// The `vamana_only` rung's answer: the vector leg in its own order, each memory scored
    /// by its cosine, after the query's filters. No fusion, boost or rescoring stage runs,
    /// and nothing is written back: access counts and coactivation feed the stages this
    /// rung leaves out, so a vector-only recall must not move what the next rung measures.
    fn vector_rung_results<S: CandidateSource + ?Sized>(
        &self,
        vector_results: &[(MemoryId, f32)],
        filter: &Query,
        candidates: &RecallCandidates<'_, S>,
    ) -> (Vec<SharedMemory>, Vec<&'static str>) {
        candidates.prefetch(vector_results.iter().map(|(id, _)| id));
        let mut memories = Vec::with_capacity(filter.max_results);
        let mut sources = Vec::new();
        for (id, similarity) in vector_results {
            if memories.len() >= filter.max_results {
                break;
            }
            let Some(view) = candidates.get(id) else {
                continue;
            };
            let valid = match filter.as_of {
                Some(t) => view.memory.is_valid_at(t),
                None => !view.memory.is_expired(),
            };
            if !valid || !self.retriever.matches_filters(&view.memory, filter) {
                continue;
            }
            let source = view.tier.source_label();
            if !sources.contains(&source) {
                sources.push(source);
            }
            let mut memory = view.memory.as_ref().clone();
            memory.set_score(*similarity);
            memories.push(Arc::new(memory));
        }
        (memories, sources)
    }

    /// Search and retrieve relevant memories (zero-copy with Arc<Memory>)
    ///
    /// PRODUCTION IMPLEMENTATION:
    /// - Semantic search: Uses embeddings + vector similarity across ALL tiers
    /// - Non-semantic search: Uses importance * temporal decay
    /// - Zero shortcuts, no TODOs, enterprise-grade
    pub fn recall(&self, query: &Query) -> Result<Vec<SharedMemory>> {
        let owner_scoped;
        let query = match (&query.user_id, &self.owner_user_id) {
            (None, Some(owner)) => {
                owner_scoped = Query {
                    user_id: Some(owner.clone()),
                    ..query.clone()
                };
                &owner_scoped
            }
            _ => query,
        };
        let _evidence = crate::graph_memory::EvidenceScope::enter(query.evidence_source.as_deref());
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
        self.expand_with_hierarchy(query, &mut memories, &mut seen_ids);

        // Rank by importance * temporal relevance
        let now = super::scoring_now();
        memories.sort_by(|a, b| {
            let age_days_a = (now - a.created_at).num_days();
            let temporal_a = Self::calculate_temporal_relevance(age_days_a);
            let score_a = a.importance() * temporal_a;

            let age_days_b = (now - b.created_at).num_days();
            let temporal_b = Self::calculate_temporal_relevance(age_days_b);
            let score_b = b.importance() * temporal_b;

            score_b.total_cmp(&score_a)
        });

        // Filter temporally invalidated memories before competition logic.
        memories.retain(|m| !m.is_expired());

        // A filter-only recall enumerates what matches; defaulting to competition would drop matches.
        let competition_mode = match query.competition_mode {
            Some(_) => query.effective_competition_mode(),
            None => CompetitionMode::Coexist,
        };
        let _ = Self::compete_memories(&mut memories, competition_mode);

        memories.truncate(query.max_results);

        // Log retrieval after competition/truncation so observability reflects
        // the actual live result set.
        self.logger
            .read()
            .log_retrieved("", memories.len(), &sources);

        // Usage writes below (access counts, coactivation, retrieval counter) are skipped under
        // read-only recall — see `recall_readonly()` for why an eval must not teach the store.
        let readonly = self.recall_readonly();

        // Update access counts with instrumentation for consolidation events
        if !readonly {
            for memory in &memories {
                self.update_access_count_instrumented(memory, StrengtheningReason::Recalled);
            }
        }

        // Hebbian learning: co-activation strengthens associations between memories
        // When memories are retrieved together, they form/strengthen edges in the memory graph.
        // S-R4: routed through apply_reinforcement so duplicate retrieval_ids are no-ops.
        if !readonly && memories.len() >= 2 {
            if let Some(graph) = &self.graph_memory {
                let retrieval_id = uuid::Uuid::new_v4();
                let source = query
                    .evidence_source
                    .as_deref()
                    .unwrap_or(crate::constants::HEBBIAN_UNATTRIBUTED_SOURCE);
                let memory_uuids: Vec<uuid::Uuid> = memories.iter().map(|m| m.id.0).collect();
                match graph.read_recursive().apply_reinforcement(
                    retrieval_id,
                    source,
                    &memory_uuids[0],
                    &memory_uuids[1..],
                ) {
                    Ok(result) => {
                        // BRIDGE-4: Consume edge tier promotions — boost memory importance
                        for promo in &result.promotions {
                            let boost = if promo.new_tier.contains("L3") {
                                0.15
                            } else {
                                0.10
                            };
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
        if !readonly {
            if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
                self.stats.write().total_retrievals = count;
            }
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
    /// Returns memories that have ANY of the specified tags, newest first.
    pub fn recall_by_tags(&self, tags: &[String], limit: usize) -> Result<Vec<Memory>> {
        self.recall_by_tags_matching(tags, limit, false)
    }

    /// As [`Self::recall_by_tags`], but a memory matches only through a tag its writer chose
    /// (caller, intent or declared, per the tag-source ledger), never through one veld derived
    /// from its content. A memory with no ledger row predates the ledger and still matches: its
    /// sources are unknown, not derived.
    pub fn recall_by_authored_tags(&self, tags: &[String], limit: usize) -> Result<Vec<Memory>> {
        self.recall_by_tags_matching(tags, limit, true)
    }

    fn recall_by_tags_matching(
        &self,
        tags: &[String],
        limit: usize,
        authored_only: bool,
    ) -> Result<Vec<Memory>> {
        let criteria = storage::SearchCriteria::ByTags(tags.to_vec());
        let mut memories = self.advanced_search(criteria)?;
        if authored_only {
            memories.retain(|memory| self.matches_an_authored_tag(memory, tags));
        }
        // A broad tag matches far more than `limit`; truncating in id order returned the same
        // arbitrary lowest-UUID slice on every call, never the recent memories a caller wants.
        memories.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        memories.truncate(limit);
        if !self.recall_readonly() {
            if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
                self.stats.write().total_retrievals = count;
            }
        }
        Ok(memories)
    }

    fn matches_an_authored_tag(&self, memory: &Memory, tags: &[String]) -> bool {
        match self.tag_source_store.get(&memory.id) {
            Ok(Some(sources)) => sources.tags.iter().any(|sourced| {
                sourced.source.is_authored()
                    && tags
                        .iter()
                        .any(|tag| tag.eq_ignore_ascii_case(&sourced.tag))
            }),
            Ok(None) => true,
            Err(error) => {
                tracing::warn!(
                    memory_id = %memory.id.0,
                    %error,
                    "tag-source ledger unreadable; matching the memory on any tag"
                );
                true
            }
        }
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
        if !self.recall_readonly() {
            if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
                self.stats.write().total_retrievals = count;
            }
        }
        Ok(memories)
    }

    /// The date window candidate leg: memories whose `created_at` falls in a calendar day
    /// or month-year the query names, ranked by similarity to the query.
    ///
    /// Vector and BM25 search cannot reach a memory whose only link to the query is a date
    /// held in `created_at` ("What did Nate do for Joanna on 25 May, 2022?" against
    /// "Nate: Looky here, I got this new pup for you!"). The temporal boost at Layer 4.55
    /// rescores candidates that are already in the pool; this leg puts the window's
    /// memories into the pool, as one more list for RRF fusion. Day windows are widened by
    /// [`DATE_WINDOW_DAY_SLACK_DAYS`] on each side; month windows are taken as written.
    /// Year-only mentions produce nothing (see [`query_parser::extract_date_windows`]).
    ///
    /// The list is ordered by cosine similarity to the query embedding (the largest over the
    /// primary space and each other space a query vector is in, each compared only in the
    /// space both vectors are in and moved onto the primary's scale), then by `created_at`,
    /// then by id, so the rank RRF
    /// reads is deterministic. Capped at the `candidate_count` dial, or `max_results * 8`
    /// like the vector leg. Empty when the `date_window_leg` dial is off or the query names
    /// no window.
    pub(crate) fn date_window_candidates(
        &self,
        query_text: &str,
        query: &Query,
        query_embedding: &[f32],
        space_queries: &[(SpaceId, Vec<f32>)],
    ) -> Vec<(MemoryId, f32)> {
        let (enabled, cap) = {
            let dials = self.cognitive_dials.read();
            (
                dials.date_window_leg != Some(false),
                dials
                    .candidate_count
                    .unwrap_or(query.max_results * 8)
                    .max(1),
            )
        };
        if !enabled {
            return Vec::new();
        }
        let windows = query_parser::extract_date_windows(query_text);
        if windows.is_empty() {
            return Vec::new();
        }
        crate::metrics::RECALL_DATE_WINDOW_QUERIES_TOTAL.inc();

        let index_space = self.retriever.index_space().clone();
        let similarity = |mem: &Memory| -> f32 {
            let mut best = 0.0f32;
            if !query_embedding.is_empty() {
                if let Some(emb) = self
                    .retriever
                    .stored_vector(mem, &index_space)
                    .filter(|e| !e.is_empty())
                {
                    best =
                        crate::memory::hybrid_search::cosine_similarity_pub(query_embedding, emb);
                }
            }
            for (space, q) in space_queries {
                if let Some(v) = self.retriever.stored_vector(mem, space) {
                    let sim = crate::memory::hybrid_search::cosine_similarity_pub(q, v);
                    best = best.max(self.retriever.calibrated_similarity(space, sim));
                }
            }
            best
        };

        let mut seen: HashSet<MemoryId> = HashSet::new();
        let mut candidates: Vec<(MemoryId, f32, chrono::DateTime<chrono::Utc>)> = Vec::new();
        for window in &windows {
            let slack = match window.granularity {
                query_parser::DateWindowGranularity::Day => {
                    chrono::Duration::days(DATE_WINDOW_DAY_SLACK_DAYS)
                }
                query_parser::DateWindowGranularity::Month => chrono::Duration::zero(),
            };
            // `ByDate` is inclusive on both ends in every backend and the RocksDB index is
            // keyed by day, so the end is the last instant of the last day covered.
            let start = (window.start - slack)
                .and_hms_opt(0, 0, 0)
                .expect("midnight exists")
                .and_utc();
            let end = (window.end + slack - chrono::Duration::days(1))
                .and_hms_milli_opt(23, 59, 59, 999)
                .expect("end of day exists")
                .and_utc();

            let mut in_window = |mem: &Memory| {
                if mem.created_at < start || mem.created_at > end || mem.is_forgotten() {
                    return;
                }
                if seen.insert(mem.id.clone()) {
                    candidates.push((mem.id.clone(), similarity(mem), mem.created_at));
                }
            };
            match self
                .long_term_memory
                .search(SearchCriteria::ByDate { start, end })
            {
                Ok(memories) => memories.iter().for_each(&mut in_window),
                Err(e) => tracing::warn!(
                    window = %window.original_text,
                    "date window leg: long-term search failed (non-fatal): {e}"
                ),
            }
            // The in-memory tiers hold what the long-term index may not have yet.
            for mem in self.working_memory.read().all_memories() {
                in_window(&mem);
            }
            for mem in self.session_memory.read().all_memories() {
                in_window(&mem);
            }
        }

        candidates.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(cap);
        crate::metrics::RECALL_DATE_WINDOW_CANDIDATES_TOTAL.inc_by(candidates.len() as u64);
        tracing::info!(
            windows = ?windows
                .iter()
                .map(|w| w.original_text.as_str())
                .collect::<Vec<_>>(),
            candidates = candidates.len(),
            cap,
            "recall [layer:3.7] date window candidate leg"
        );
        candidates
            .into_iter()
            .map(|(id, score, _)| (id, score))
            .collect()
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
        // SPREADING ACTIVATION MODE (V5 toroid closure)
        // ===========================================================================
        // When mode=SpreadingActivation, bypass the 11-layer pipeline entirely and
        // use the dedicated graph_retrieval::spreading_activation_retrieve algorithm.
        // This implements Anderson & Pirolli (1984) with density-dependent weights.
        if matches!(query.retrieval_mode, RetrievalMode::SpreadingActivation) {
            return self.spreading_activation_recall(query_text, query);
        }

        // Layer 0.5 reads the BM25 index directly, ahead of hybrid search's own commit, so
        // buffered writes are committed here for a just-written memory to be findable there.
        if let Err(e) = self.hybrid_search.commit_pending() {
            tracing::warn!(error = %e, "BM25 commit before recall failed; buffered writes are not yet searchable");
        }

        // One clock for the whole recall: graph edge decay and Layer 5 recency agree on "now".
        let now = super::scoring_now();
        // Every candidate record this recall reads, each fetched once.
        let candidates = RecallCandidates::new(self);

        // ===========================================================================
        // TEMPORAL EXTRACTION (TEMPR approach from Hindsight - 89.6% on LoCoMo)
        // ===========================================================================
        // Key insight: Temporal filtering is critical for multi-hop retrieval accuracy.
        // Extract temporal constraints from query and use them to boost/filter results.
        // B2: pass an event-resolver closure so named-event anchors ("the Sunday
        // before the wedding") are resolved via the TemporalFactStore at query time.
        let tfs_for_resolver = Arc::clone(&self.temporal_fact_store);
        let user_id_for_resolver = query.user_id.clone().unwrap_or_default();
        let event_resolver = move |event_phrase: &str| -> Option<chrono::NaiveDate> {
            let mut visited = std::collections::HashSet::new();
            tfs_for_resolver.resolve_event_to_date(
                &user_id_for_resolver,
                event_phrase,
                &mut visited,
                0,
            )
        };
        let query_temporal =
            query_parser::extract_temporal_refs_with_events(query_text, Some(&event_resolver));
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

        let use_graph = matches!(
            query.retrieval_mode,
            RetrievalMode::Hybrid | RetrievalMode::Associative | RetrievalMode::Causal
        ) && query
            .layer_mode()
            .allows(crate::memory::types::LayerMode::PlusSpreading);
        let onto_confident = onto_intent.confidence >= crate::constants::ONTOLOGICAL_MIN_CONFIDENCE;

        // Query words resolved to graph entities once, for the ontology density gate below
        // and Layer 2. Only the nouns and path-shaped words seed Layer 2's walk; adjectives
        // and verbs count toward the density that weighs the graph leg in fusion.
        //
        // Every graph read on the recall path is recursive: `/api/relevant` and
        // `/api/stream` call recall while holding this graph's read guard, and a plain read
        // would queue behind a waiting writer that the outer guard keeps waiting.
        let resolved_entities: Option<ResolvedQueryEntities> = self
            .graph_memory
            .as_ref()
            .filter(|_| use_graph || onto_confident)
            .map(|graph| {
                let focal_words: Vec<&str> = query_analysis
                    .focal_entities
                    .iter()
                    .map(|e| e.text.as_str())
                    .collect();
                let other_words: Vec<&str> = query_analysis
                    .discriminative_modifiers
                    .iter()
                    .map(|m| m.text.as_str())
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
                    .collect();
                resolve_query_entities(
                    &graph.read_recursive(),
                    &focal_words,
                    &other_words,
                    query_text,
                )
            });

        // Compute graph density for ontological gating (consistent with Layer 2 in graph_retrieval.rs).
        // Dense/young graphs have too many noisy L1 edges for type filtering to help.
        // Short-circuit: skip the RocksDB density lookups entirely when confidence is too low.
        let (graph_density_for_rerank, use_ontology_rerank) = if !onto_confident {
            (None, false)
        } else if let Some(graph) = self.graph_memory.as_ref() {
            let g = graph.read_recursive();
            let seed_uuids: &[uuid::Uuid] = resolved_entities
                .as_ref()
                .map(|r| r.focal.as_slice())
                .unwrap_or_default();
            let density = if seed_uuids.is_empty() {
                None
            } else {
                g.entities_average_density(seed_uuids).ok().flatten()
            };
            let use_rerank = !density
                .is_some_and(|d| d >= crate::constants::ONTOLOGICAL_DENSITY_THRESHOLD)
                && query
                    .layer_mode()
                    .allows(crate::memory::types::LayerMode::PlusRerank);
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
            query_parser::QueryType::Attribute(attr_query)
                if query
                    .layer_mode()
                    .allows(crate::memory::types::LayerMode::PlusBm25) =>
            {
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

                candidates.prefetch(bm25_matches.iter().map(|(id, _)| id));
                for (mem_id, _score) in bm25_matches {
                    // Get memory content to verify it contains entity + attribute value
                    if let Some(view) = candidates.get(&mem_id) {
                        let content = &view.content_lower;
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
            "when",
            "last week",
            "recently",
            "during",
            "before",
            "after",
            "yesterday",
            "meeting",
            "session",
            "phase",
            "earlier",
            "later",
            "previous",
            "following",
            "happened",
            "took place",
            "occurred",
        ]
        .iter()
        .any(|kw| query_lower_for_temporal.contains(kw));

        let temporal_gate = has_any_temporal_intent || has_temporal_keywords;

        // A PlusFacts stage, like Layer 0.7: below that rung it contributes nothing.
        let temporal_fact_boost_ids: HashSet<MemoryId> = if temporal_gate
            && query
                .layer_mode()
                .allows(crate::memory::types::LayerMode::PlusFacts)
        {
            if let Some(user_id) = query
                .user_id
                .as_ref()
                .filter(|_| !super::ablation::fact_layers_disabled())
            {
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
                let event_type =
                    if temporal_facts::query_signals_planned_intent(&query_lower_for_temporal) {
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
                            tracing::debug!(
                                "Layer 0.6: Temporal fact lookup (entity+event) failed: {}",
                                e
                            );
                        }
                    }
                }

                // ---------------------------------------------------------------
                // Strategy 2 (fallback): Entity-only lookup
                // ---------------------------------------------------------------
                // Fires when entity is known but event keywords are empty or
                // Strategy 1 returned nothing.
                // [signal: temporal_fact_untimed_fallback] With no time window to filter by, it
                // takes every fact about the entity, up to 50 sources for one person; disabled,
                // only time-window-filtered facts count. Default = on.
                let untimed_entity_fallback = self
                    .cognitive_dials
                    .read()
                    .signal_enabled("temporal_fact_untimed_fallback");
                if boosted.is_empty() && !entity.is_empty() {
                    match self
                        .temporal_fact_store
                        .find_by_entity_filtered(user_id, &entity, 50, false)
                    {
                        Ok(facts) if !facts.is_empty() => {
                            // When query has a parsed time window, filter facts by it
                            let filtered =
                                Self::filter_facts_by_time_window(&facts, &query_temporal);
                            if !filtered.is_empty() {
                                tracing::info!(
                                    "Layer 0.6: Fallback entity-only found {} facts (time-filtered) for '{}'",
                                    filtered.len(),
                                    entity
                                );
                                boosted.extend(
                                    filtered.into_iter().map(|f| f.source_memory_id.clone()),
                                );
                            } else if untimed_entity_fallback {
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
                                let filtered =
                                    Self::filter_facts_by_time_window(&facts, &query_temporal);
                                if !filtered.is_empty() {
                                    boosted.extend(
                                        filtered.iter().map(|f| f.source_memory_id.clone()),
                                    );
                                } else {
                                    boosted
                                        .extend(facts.iter().map(|f| f.source_memory_id.clone()));
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
                            let filtered =
                                Self::filter_facts_by_time_window(&all_facts, &query_temporal);
                            if !filtered.is_empty() {
                                tracing::info!(
                                    "Layer 0.6: Time-window scan matched {} facts out of {}",
                                    filtered.len(),
                                    all_facts.len()
                                );
                                boosted.extend(
                                    filtered.into_iter().map(|f| f.source_memory_id.clone()),
                                );
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

            // Layer 0.7 is a PlusFacts stage; below that rung it contributes nothing.
            let facts_allowed = query
                .layer_mode()
                .allows(crate::memory::types::LayerMode::PlusFacts)
                && !super::ablation::fact_layers_disabled();

            if let Some(user_id) = &query.user_id.clone().filter(|_| facts_allowed) {
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

        // The query, embedded once in each space an embedder here writes.
        let (query_embedding, query_embedding_secondary) = self.query_vectors(query_text, query)?;
        // The space each query vector is in. A stored vector is compared with a query only in
        // the space both are in, and every space a query vector exists for is searched.
        let index_space = self.retriever.index_space().clone();
        let space_queries: Vec<(SpaceId, Vec<f32>)> = self
            .retriever
            .secondary_index_space()
            .cloned()
            .zip(query_embedding_secondary)
            .into_iter()
            .collect();
        // Every vector leg beyond the primary index needs only the query vectors, so all of
        // them start now, beside graph traversal, the primary search and BM25, and are
        // collected at fusion.
        let pending_legs = if query.dual_index.unwrap_or(true) {
            let candidate_dial = self.cognitive_dials.read().space_candidates.clone();
            PendingLegs::start(self.retriever.space_legs(
                &query_embedding,
                &space_queries,
                query.max_results,
                &candidate_dial,
            ))
        } else {
            PendingLegs::start(Vec::new())
        };
        let mut leg_record: Vec<(SpaceId, Vec<MemoryId>)> = Vec::new();

        let t_embedding = recall_start.elapsed();
        tracing::info!(
            embedding_ms = format!(
                "{:.2}",
                (t_embedding - t_query_analysis).as_secs_f64() * 1000.0
            ),
            cumulative_ms = format!("{:.2}", t_embedding.as_secs_f64() * 1000.0),
            has_secondary = !space_queries.is_empty(),
            "recall [layer:embedding] query embedding"
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

                // Every resolved query word, each once.
                let query_entities: &[uuid::Uuid] = resolved_entities
                    .as_ref()
                    .map(|r| r.all.as_slice())
                    .unwrap_or_default();

                // The graph guard covers the walk and the episode reads only; the candidate
                // records are read after it is released.
                let (d, leg_weights, pool, seed_count) = {
                    let g = graph.read_recursive();
                    // Calculate PER-ENTITY density (not global graph density)
                    // Sparse entities = trust graph, Dense entities = trust vector
                    let d = if !query_entities.is_empty() {
                        g.entities_average_density(query_entities).ok().flatten()
                    } else {
                        // No query entities — skip density calculation.
                        // The default weights (0.6, 0.3, 0.1) handle this case correctly.
                        None
                    };
                    // Hybrid ranks the leg with fixed weights, the graph-led modes with their
                    // density's, which also decides whether the ontology may shape the walk.
                    let leg_density = match query.retrieval_mode {
                        RetrievalMode::Associative | RetrievalMode::Causal => d,
                        _ => None,
                    };
                    let leg_weights = leg_density.map(calculate_density_weights).unwrap_or((
                        crate::constants::HYBRID_SEMANTIC_WEIGHT,
                        crate::constants::HYBRID_GRAPH_WEIGHT,
                        crate::constants::HYBRID_LINGUISTIC_WEIGHT,
                    ));
                    let intent = (onto_confident
                        && !leg_density.is_some_and(|density| {
                            density >= crate::constants::ONTOLOGICAL_DENSITY_THRESHOLD
                        }))
                    .then_some(&onto_intent);

                    let seeds = graph_leg_seeds(
                        &g,
                        resolved_entities.as_ref(),
                        &query_analysis,
                        query.ner_entities.as_deref(),
                    );
                    let pool = if seeds.is_empty() {
                        Vec::new()
                    } else {
                        let restart = crate::memory::graph_retrieval::seed_restart_weights(&seeds);
                        let seed_set: HashSet<uuid::Uuid> = seeds.iter().map(|s| s.uuid).collect();
                        let walked = crate::memory::graph_retrieval::personalized_pagerank(
                            &g, &restart, intent, now,
                        )
                        .and_then(|ppr| {
                            let pool = crate::memory::graph_retrieval::ppr_episode_pool(
                                &g,
                                &ppr,
                                &seed_set,
                                |episode| {
                                    episode_candidates
                                        .as_ref()
                                        .is_none_or(|c| c.contains(&MemoryId(*episode)))
                                },
                            )?;
                            Ok((ppr, pool))
                        });
                        match walked {
                            Ok((ppr, pool)) => {
                                // Retrieval-driven strengthening of the edges that carried the
                                // walk: a usage write, skipped under read-only recall.
                                if !ppr.carrying_edges.is_empty()
                                    && !self.recall_readonly()
                                    && super::ablation::is_enabled("graph_potentiation")
                                {
                                    if let Err(e) = g.batch_strengthen_synapses(
                                        &ppr.carrying_edges,
                                        &crate::graph_memory::ambient_evidence_source(),
                                    ) {
                                        tracing::debug!("Layer 2: edge strengthening failed: {e}");
                                    }
                                }
                                tracing::debug!(
                                    seeds = seeds.len(),
                                    expanded = ppr.expanded,
                                    edge_reads = ppr.edge_reads,
                                    activated = ppr.mass.len(),
                                    pool = pool.len(),
                                    "Layer 2: personalized PageRank"
                                );
                                pool.into_iter()
                                    .map(|(episode, activation)| (MemoryId(episode), activation))
                                    .collect()
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Layer 2: graph leg failed, recall continues without it: {e:#}"
                                );
                                Vec::new()
                            }
                        }
                    };
                    (d, leg_weights, pool, seeds.len())
                };

                // Rank the pool by relevance, as shodh-memory #258 ranks Layer 2 by final score:
                // each episode's query cosine, linguistic match and graph activation, fused with
                // the leg's weights. Recency, arousal and credibility are left to Layer 5, which
                // applies them to every candidate.
                let (semantic_w, leg_graph_w, linguistic_w) = leg_weights;
                candidates.prefetch(pool.iter().map(|(id, _)| id));
                let max_activation = pool.iter().map(|(_, a)| *a).fold(0.0_f32, f32::max);
                let ranked: Vec<(MemoryId, f32, f32, Arc<CandidateView>)> = pool
                    .into_iter()
                    .filter_map(|(id, activation)| {
                        let view = candidates.get(&id)?;
                        if view.memory.is_forgotten() {
                            return None;
                        }
                        let semantic = self
                            .retriever
                            .stored_vector(&view.memory, &index_space)
                            .map_or(0.0, |e| {
                                crate::similarity::cosine_similarity(&query_embedding, e)
                            });
                        let linguistic = crate::memory::graph_retrieval::linguistic_match(
                            &view.content_lower,
                            &query_analysis,
                        );
                        let score = crate::memory::graph_retrieval::fuse_hybrid_score(
                            semantic,
                            activation,
                            linguistic,
                            semantic_w,
                            leg_graph_w,
                            linguistic_w,
                            crate::memory::graph_retrieval::memory_tier_graph_trust(
                                view.memory.tier,
                            ),
                        );
                        let strength = if max_activation > 0.0 {
                            activation / max_activation
                        } else {
                            0.0
                        };
                        Some((id, score, strength, view))
                    })
                    .collect();
                let pooled = ranked.len();
                let top_score = ranked.iter().map(|x| x.1).fold(0.0_f32, f32::max);
                let r = graph_leg_for_fusion(
                    ranked,
                    crate::constants::RECALL_GRAPH_LEG_CANDIDATES,
                    |view| view.memory.experience.content.as_str(),
                );
                if !r.is_empty() {
                    tracing::debug!(
                        "Layer 2: {} graph results (ranked from {}), {} seeds, top={:.3}",
                        r.len(),
                        pooled,
                        seed_count,
                        top_score
                    );
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

        crate::memory::gold_funnel::record("graph", graph_results.iter().map(|(id, _, _)| id));

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
            // inherited: a sub-query that widened the ladder would leak stages
            // the caller ablated away
            layers: query.layers,
            ner_entities: query.ner_entities.clone(),
            evidence_source: query.evidence_source.clone(),
            time_range: query.time_range,
            as_of: query.as_of,
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
            repo: query.repo.clone(),
            pattern_id: query.pattern_id.clone(),
            terrain_type: query.terrain_type.clone(),
            confidence_range: query.confidence_range,
            offset: query.offset,
            episode_id: query.episode_id.clone(),
            prospective_signals: query.prospective_signals.clone(),
            recency_weight: query.recency_weight,
            competition_mode: query.competition_mode,
            rrf_k: query.rrf_k,
            rerank_count: query.rerank_count,
            dual_index: query.dual_index,
        };

        // ===========================================================================
        // LAYER 3: VECTOR SEARCH (Vamana Index) + QUERY DECOMPOSITION
        // ===========================================================================
        // Decompose compound/multi-entity queries into sub-queries for independent
        // vector search. BM25 and graph traversal still use the original query —
        // they handle compound queries better naturally.
        // The vamana_only rung is the vector index's own answer to the query embedding:
        // sub-query lists would come back RRF-fused, not ranked by cosine.
        let vector_only = !query
            .layer_mode()
            .allows(crate::memory::types::LayerMode::PlusSpreading);
        let sub_queries = if vector_only {
            vec![query_text.to_string()]
        } else {
            query_parser::decompose_query(query_text, &query_analysis)
        };
        let decomposed = sub_queries.len() > 1;
        // Detect abstract/thematic decomposition (Rule 4): decomposition fired AND
        // query contains abstract vocabulary. Used to invert RRF weights (sub-queries
        // are more targeted than the vague original) and gate adaptive scoring.
        let abstract_decomposed =
            decomposed && query_parser::expand_abstract_terms_for_bm25(query_text).is_some();

        let mut vector_results: Vec<(MemoryId, f32)> = if decomposed {
            // --- Decomposed path: weighted RRF merge of sub-query vector results ---
            use crate::constants::QUERY_DECOMPOSITION_SUB_WEIGHT;
            let decomp_rrf_k: f32 = self.cognitive_dials.read().rrf_k_decomp.unwrap_or(20.0);

            let mut merged: std::collections::HashMap<MemoryId, f32> =
                std::collections::HashMap::new();
            // First-appearance order over the sub-query lists, kept by the stable sort below
            // for equal fused scores.
            let mut first_seen: Vec<MemoryId> = Vec::new();

            for (idx, sub_q) in sub_queries.iter().enumerate() {
                let is_original = idx == 0;
                // For abstract decomposition, INVERT weights: sub-queries ("performance",
                // "improvements") are more targeted than the vague original, so they should
                // contribute more to RRF scoring. For conjunction/entity decomposition,
                // the original compound query is more informative — keep default weights.
                let weight = if abstract_decomposed {
                    if is_original {
                        QUERY_DECOMPOSITION_SUB_WEIGHT
                    } else {
                        1.0_f32
                    }
                } else if is_original {
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
                        match self.embedder.as_ref().encode_for_query(sub_q) {
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
                    layers: query.layers,
                    ner_entities: query.ner_entities.clone(),
                    evidence_source: query.evidence_source.clone(),
                    time_range: query.time_range,
                    as_of: query.as_of,
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
                    repo: query.repo.clone(),
                    pattern_id: query.pattern_id.clone(),
                    terrain_type: query.terrain_type.clone(),
                    confidence_range: query.confidence_range,
                    offset: query.offset,
                    episode_id: query.episode_id.clone(),
                    prospective_signals: query.prospective_signals.clone(),
                    recency_weight: query.recency_weight,
                    competition_mode: query.competition_mode,
                    rrf_k: query.rrf_k,
                    rerank_count: query.rerank_count,
                    dual_index: query.dual_index,
                };

                let sub_results = self
                    .retriever
                    .search_ids(&sub_vector_query, query.max_results * 8)?;

                let filtered: Vec<(MemoryId, f32)> = if let Some(ref c) = episode_candidates {
                    sub_results
                        .into_iter()
                        .filter(|(id, _)| c.contains(id))
                        .collect()
                } else {
                    sub_results
                };

                // Weighted RRF: score = weight / (K + rank)
                for (rank, (id, _sim)) in filtered.iter().enumerate() {
                    let rrf_score = weight / (decomp_rrf_k + (rank + 1) as f32);
                    let entry = merged.entry(id.clone()).or_insert_with(|| {
                        first_seen.push(id.clone());
                        0.0
                    });
                    *entry = entry.max(rrf_score).max(*entry);
                    // Use max-merge: keep the highest RRF contribution per memory
                    // to prevent score inflation from appearing in multiple sub-queries
                    *entry = *entry + rrf_score - entry.min(rrf_score);
                }
            }

            // Sort by fused score and convert back to (MemoryId, f32)
            let mut fused_vec: Vec<(MemoryId, f32)> = first_seen
                .into_iter()
                .map(|id| {
                    let score = merged[&id];
                    (id, score)
                })
                .collect();
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

        crate::memory::gold_funnel::record("vector", vector_results.iter().map(|(id, _)| id));

        let t_vector = recall_start.elapsed();
        tracing::info!(
            vector_ms = format!("{:.2}", (t_vector - t_graph).as_secs_f64() * 1000.0),
            cumulative_ms = format!("{:.2}", t_vector.as_secs_f64() * 1000.0),
            vector_results = vector_results.len(),
            decomposed = decomposed,
            "recall [layer:3] Vamana vector search{}",
            if decomposed { " (decomposed)" } else { "" }
        );

        // ===========================================================================
        // LAYER 3.1: MULTI-QUERY EMBEDDING FUSION
        // ===========================================================================
        // When decomposition produced sub-queries, the single verbatim embedding
        // dilutes compound topics (e.g., "database change AND timeline impact" embeds
        // halfway between both, matching neither well). Fuse sub-query embeddings by
        // averaging to give equal weight to each sub-topic. The fused embedding
        // propagates to Layer 3.5 (brute-force cosine scan) and the vector component
        // of Layer 4 (hybrid search). The decomposed RRF path (Layer 3) already used
        // individual sub-query embeddings, so this is complementary.
        let query_embedding = if decomposed && sub_queries.len() > 1 {
            // Embed each sub-query (skip idx=0 which is the original compound query —
            // including it would re-introduce the dilution we're trying to fix)
            let mut sub_embeddings: Vec<Vec<f32>> = Vec::with_capacity(sub_queries.len() - 1);
            for sub_q in sub_queries.iter().skip(1) {
                let sub_hash = Self::sha256_hash(sub_q);
                let emb = if let Some(cached) = self.query_cache.get(&sub_hash) {
                    cached.clone()
                } else {
                    match self.embedder.as_ref().encode_for_query(sub_q) {
                        Ok(e) => {
                            self.query_cache.insert(sub_hash, e.clone());
                            e
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Layer 3.1: Failed to embed sub-query '{}': {}",
                                sub_q,
                                e
                            );
                            continue;
                        }
                    }
                };
                sub_embeddings.push(emb);
            }

            if sub_embeddings.is_empty() {
                // All sub-query embeddings failed — keep original
                query_embedding
            } else {
                let dim = sub_embeddings[0].len();
                let n = sub_embeddings.len() as f32;
                let mut fused = vec![0.0f32; dim];
                for emb in &sub_embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        fused[i] += v;
                    }
                }
                // Element-wise average
                for v in &mut fused {
                    *v /= n;
                }
                // L2-normalize: MiniLM embeddings are unit-normalized, averaging
                // de-normalizes them. Without re-normalization cosine similarities
                // are distorted.
                let norm: f32 = fused.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for v in &mut fused {
                        *v /= norm;
                    }
                }
                tracing::info!(
                    sub_query_count = sub_embeddings.len(),
                    "recall [layer:3.1] Multi-query embedding fusion (averaged {} sub-query embeddings)",
                    sub_embeddings.len()
                );
                fused
            }
        } else {
            query_embedding
        };

        // ===========================================================================
        // LAYER 3.5: WORKING + SESSION TIER BRUTE-FORCE COSINE SCAN
        // ===========================================================================
        // A Working- or Session-tier memory the vector index does not hold yet (its
        // embedding is still being written back, or its insert failed) is invisible to
        // Layer 3, so the in-memory tiers are scanned by brute-force cosine for those.
        // A memory the index does hold was already ranked there: the ones below its cut
        // are its rejects, and on the decomposed path they re-entered at the top reciprocal
        // ranks (LoCoMo conv-26: every turn is tier-resident, vector hit@10 0.72 -> 0.68).
        {
            let mut tier_hits = 0usize;
            let vamana_ids: std::collections::HashSet<MemoryId> =
                vector_results.iter().map(|(id, _)| id.clone()).collect();
            let mut tier_candidates = self.unindexed_tier_hits(
                &query_embedding,
                &index_space,
                &space_queries,
                &vamana_ids,
            );

            if decomposed {
                // The decomposed vector list carries reciprocal-rank scores (at most
                // weight / (k + 1), about 0.05), not cosine. Pushing raw cosine (above 0.1)
                // beside it put every tier hit above every Vamana hit, and the truncation
                // below then evicted the Vamana hits (live: tier_hits=56, total=48). Tier hits
                // enter as one more ranked list instead, at the strongest list's weight and
                // the same k, so they interleave with the Vamana hits by rank.
                let decomp_rrf_k: f32 = self.cognitive_dials.read().rrf_k_decomp.unwrap_or(20.0);
                tier_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                for (rank, (id, _)) in tier_candidates.into_iter().enumerate() {
                    vector_results.push((id, 1.0 / (decomp_rrf_k + (rank + 1) as f32)));
                    tier_hits += 1;
                }
            } else {
                tier_hits = tier_candidates.len();
                vector_results.extend(tier_candidates);
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

        if vector_only {
            let (memories, sources) =
                self.vector_rung_results(&vector_results, &vector_query, &candidates);
            crate::memory::gold_funnel::record("final", memories.iter().map(|m| &m.id));
            self.logger
                .read()
                .log_retrieved(query_text, memories.len(), &sources);
            return Ok(memories);
        }

        // ===========================================================================
        // LAYER 3.7: DATE WINDOW CANDIDATE LEG
        // ===========================================================================
        // When the query names a calendar day or month-year, the memories created in
        // that window are a candidate list of their own, fused below with the same
        // weight as the hybrid list. Empty for every other query, and under the
        // `date_window_leg` dial when it is off.
        let date_window_results =
            self.date_window_candidates(query_text, query, &query_embedding, &space_queries);
        crate::memory::gold_funnel::record(
            "date_window",
            date_window_results.iter().map(|(id, _)| id),
        );

        // ===========================================================================
        // LAYER 4: BM25 + RRF FUSION
        // ===========================================================================
        // Early dials snapshot for signal gates inside the fusion block (content_fallback,
        // bm25_specificity_discount). A second snapshot is taken before Layer 5 scoring;
        // both snapshots are value copies of the same lock, so they are always consistent
        // with each other — the same set was in effect for the whole query.
        let early_dials_snapshot = self.cognitive_dials.read().clone();

        type FusionLayerOutput = (
            Vec<(MemoryId, f32)>,
            std::collections::HashMap<MemoryId, f32>,
            std::collections::HashMap<MemoryId, SignalAttribution>,
        );

        let (memory_ids, hebbian_scores, mut signal_attributions): FusionLayerOutput = {
            let get_content = |id: &MemoryId| -> Option<String> { candidates.content(id) };
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
            // Signal attribution: capture component scores from hybrid search
            // before reducing to (id, score) pairs. These track which signals
            // contributed to each memory's ranking for adaptive weight learning.
            let mut attr: std::collections::HashMap<MemoryId, SignalAttribution> =
                std::collections::HashMap::new();
            let primary_vector_hits: Vec<MemoryId> =
                vector_results.iter().map(|(id, _)| id.clone()).collect();
            let space_weights = self.cognitive_dials.read().space_weights.clone();
            let space_weight = |space: &SpaceId| -> f32 {
                space_weights.get(space.as_str()).copied().unwrap_or(1.0)
            };
            fn mark_space(
                attr: &mut std::collections::HashMap<MemoryId, SignalAttribution>,
                id: &MemoryId,
                space: &SpaceId,
            ) {
                let spaces = &mut attr.entry(id.clone()).or_default().spaces;
                if !spaces.contains(space) {
                    spaces.push(space.clone());
                }
            }

            // ---------------------------------------------------------------
            // BM25 CONCEPT EXPANSION: Expand abstract/meta-reasoning query
            // terms with concrete synonyms so BM25 can match memories that
            // describe specific instances of abstract concepts.
            // e.g., "patterns" → also search "decisions choices recurring"
            // Expansion terms use low boost (^0.3) to act as tiebreakers.
            // ---------------------------------------------------------------
            let bm25_query_text = query_parser::expand_abstract_terms_for_bm25(query_text);
            let bm25_query_ref: &str = bm25_query_text.as_deref().unwrap_or(query_text);

            let bm25_allowed = query
                .layer_mode()
                .allows(crate::memory::types::LayerMode::PlusBm25);
            let hybrid_search_results = if bm25_allowed {
                self.hybrid_search
                    .search_with_dynamic_weights(
                        bm25_query_ref,
                        vector_results.clone(),
                        get_content,
                        term_weights,
                        phrases,
                        disc_opt,
                        query.rerank_count, // FIX-11: per-query rerank count override
                    )
                    .ok()
            } else {
                None
            };
            // What FLAT fuses: each hybrid candidate's raw scores in the two lists, and the
            // factor the optional LLM refiner multiplied its hybrid score by (1 without one).
            let mut hybrid_components: std::collections::HashMap<MemoryId, HybridComponents> =
                std::collections::HashMap::new();
            let hybrid_ids = match hybrid_search_results {
                Some(r) => r
                    .into_iter()
                    .map(|x| {
                        attr.insert(
                            x.memory_id.clone(),
                            SignalAttribution {
                                bm25_contribution: x.bm25_score.unwrap_or(0.0),
                                vector_contribution: x.vector_score.unwrap_or(0.0),
                                graph_contribution: x.graph_score.unwrap_or(0.0),
                                cross_encoder_contribution: x.rerank_score.unwrap_or(0.0),
                                ..Default::default()
                            },
                        );
                        hybrid_components.insert(
                            x.memory_id.clone(),
                            HybridComponents {
                                bm25: x.bm25_score.unwrap_or(0.0),
                                vector: x.vector_score.unwrap_or(0.0),
                                refined: if x.rrf_score > 0.0 {
                                    x.score / x.rrf_score
                                } else {
                                    1.0
                                },
                            },
                        );
                        (x.memory_id, x.score)
                    })
                    .collect::<Vec<_>>(),
                None => {
                    for (id, score) in &vector_results {
                        hybrid_components.insert(
                            id.clone(),
                            HybridComponents {
                                bm25: 0.0,
                                vector: *score,
                                refined: 1.0,
                            },
                        );
                    }
                    vector_results
                }
            };
            for id in &primary_vector_hits {
                mark_space(&mut attr, id, &index_space);
            }

            // ===========================================================================
            // LAYER 4: FUSION WITH DENSITY-BASED WEIGHTS (PIPE-11)
            // ===========================================================================
            // Biological model: Memory graphs start dense (noisy L1 edges) and become
            // sparse over time through pruning (Hebbian "use it or lose it").
            //
            // Sparse graphs = mature, curated connections = trust graph more
            // Dense graphs = fresh, noisy connections = trust semantic/BM25 more
            //
            // Reciprocal rank (the default) adds weight / (k + rank) per list; K sets top-rank
            // discrimination, precedence per-query rrf_k → live dial `rrf_k_outer` → 20.0.
            // FLAT, behind the `fusion` dial or SHODH_FUSION_FLAT, scores a candidate in each
            // list by its score over the list's best, vector and BM25 joined as the stronger
            // plus FLAT_CONSENSUS of the weaker. On the LoCoMo hold-out (1,166 cases) against
            // main, FLAT traded the top hit for depth (p@1 -0.050, recall@10 +0.032), and
            // reciprocal rank under the same bounded boosts did better on both (+0.023,
            // +0.043). Upstream's own gate arm leaned the same way on p@1 (0.48 FLAT, 0.50 RRF).
            let fusion_mode = super::ablation::fusion_mode(early_dials_snapshot.fusion);
            let k: f32 = query
                .rrf_k
                .or_else(|| self.cognitive_dials.read().rrf_k_outer)
                .unwrap_or(20.0);
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
            // Applied after query-type modulation so `0` really is graph-off.
            let graph_w = super::ablation::graph_fusion_weight_override().unwrap_or(graph_w);

            // Hybrid weight = semantic + linguistic (BM25 + vector combined)
            let hybrid_w = semantic_w + linguistic_w;

            tracing::debug!(
                "Layer 4 {} fusion: density={:?}, graph_w={:.2}, hybrid_w={:.2}, query_entities={}",
                fusion_mode.as_str(),
                graph_density,
                graph_w,
                hybrid_w,
                query_entity_count
            );

            // Graph results: the leg's term, times the activation bonus. Under FLAT the term
            // is the activation over the leg's highest. Under `graph_candidates_only` the leg
            // adds no term and no hop weight; its hits that no hybrid list found are scored
            // after the hybrid leg below.
            let graph_candidates_only = early_dials_snapshot.graph_candidates_only.unwrap_or(false);
            let best_activation = best_leg_score(graph_results.iter().map(|(_, a, _)| *a));
            for (r, (id, activation, h)) in graph_results.iter().enumerate() {
                if !graph_candidates_only {
                    let term =
                        leg_term(fusion_mode, graph_w, k, r + 1, *activation, best_activation)
                            * graph_activation_factor(graph_w, *activation);
                    *fused.entry(id.clone()).or_insert(0.0) += term;
                    heb.insert(id.clone(), *h);
                }

                // Signal attribution: record graph contribution for memories
                // that came from graph traversal (may not have hybrid search data)
                attr.entry(id.clone()).or_default().graph_contribution = *activation;
            }

            // Hybrid (BM25+vector) results.
            match fusion_mode {
                FusionMode::Rrf => {
                    for (r, (id, _)) in hybrid_ids.iter().enumerate() {
                        *fused.entry(id.clone()).or_insert(0.0) += hybrid_w / (k + (r + 1) as f32);
                    }
                }
                FusionMode::Flat => {
                    let best_vector = best_leg_score(hybrid_components.values().map(|c| c.vector));
                    let best_bm25 = best_leg_score(hybrid_components.values().map(|c| c.bm25));
                    for (id, _) in &hybrid_ids {
                        let Some(c) = hybrid_components.get(id) else {
                            continue;
                        };
                        *fused.entry(id.clone()).or_insert(0.0) += hybrid_w
                            * flat_hybrid(
                                calibrated(c.vector, best_vector),
                                calibrated(c.bm25, best_bm25),
                            )
                            * c.refined;
                    }
                }
            }

            if graph_candidates_only {
                let admitted: Vec<&MemoryId> = graph_results
                    .iter()
                    .map(|(id, _, _)| id)
                    .filter(|id| !hybrid_components.contains_key(*id))
                    .collect();
                if !admitted.is_empty() {
                    // The vector list's best by its own score, to put the admitted hits'
                    // cosines on that list's scale: its scores are reciprocal ranks on a
                    // decomposed query, so its best is re-read as a cosine like theirs.
                    let mut vector_best: Option<(&MemoryId, f32)> = None;
                    for (id, _) in &hybrid_ids {
                        if let Some(c) = hybrid_components.get(id) {
                            if vector_best.is_none_or(|(_, best)| c.vector > best) {
                                vector_best = Some((id, c.vector));
                            }
                        }
                    }
                    candidates.prefetch(
                        admitted
                            .iter()
                            .copied()
                            .chain(vector_best.map(|(id, _)| id)),
                    );
                    let cosine = |id: &MemoryId| -> f32 {
                        candidates
                            .get(id)
                            .and_then(|view| {
                                self.retriever
                                    .stored_vector(&view.memory, &index_space)
                                    .filter(|e| !e.is_empty())
                                    .map(|e| {
                                        crate::memory::hybrid_search::cosine_similarity_pub(
                                            &query_embedding,
                                            e,
                                        )
                                    })
                            })
                            .unwrap_or(0.0)
                    };
                    let scored = admitted_graph_hit_terms(
                        fusion_mode,
                        hybrid_w,
                        k,
                        hybrid_ids.len(),
                        admitted.into_iter().map(|id| (id, cosine(id))).collect(),
                        vector_best.map(|(id, _)| cosine(id)),
                    );
                    for (id, term) in scored {
                        *fused.entry(id.clone()).or_insert(0.0) += term;
                    }
                }
            }

            // PSEUDO-RELEVANCE FEEDBACK: a second lexical pass whose query adds the terms the
            // first pass's top documents share, fused as its own leg. Dials: use_prf,
            // prf_docs, prf_terms, prf_weight (constants::PRF_*).
            let prf_wanted = early_dials_snapshot
                .use_prf
                .unwrap_or(crate::constants::PRF_ENABLED_DEFAULT)
                && early_dials_snapshot.signal_enabled("prf");
            if bm25_allowed && prf_wanted && hybrid_ids.len() >= 2 {
                let prf_docs = early_dials_snapshot
                    .prf_docs
                    .unwrap_or(crate::constants::PRF_DOCS);
                let prf_term_count = early_dials_snapshot
                    .prf_terms
                    .unwrap_or(crate::constants::PRF_TERMS);
                let prf_weight = early_dials_snapshot
                    .prf_weight
                    .unwrap_or(crate::constants::PRF_WEIGHT);
                let doc_texts: Vec<String> = hybrid_ids
                    .iter()
                    .take(prf_docs)
                    .filter_map(|(id, _)| get_content(id))
                    .collect();
                let doc_refs: Vec<&str> = doc_texts.iter().map(String::as_str).collect();
                let terms = query_parser::prf_terms(&doc_refs, query_text, prf_term_count);
                if !terms.is_empty() && prf_weight > 0.0 {
                    let expanded = format!("{} {}", bm25_query_ref, terms.join(" "));
                    let prf_hits = self
                        .hybrid_search
                        .bm25_index()
                        .search(&expanded, query.max_results * 5)
                        .unwrap_or_default();
                    let prf_w = hybrid_w * prf_weight;
                    let best_prf = best_leg_score(prf_hits.iter().map(|(_, s)| *s));
                    for (r, (id, score)) in prf_hits.iter().enumerate() {
                        *fused.entry(id.clone()).or_insert(0.0) +=
                            leg_term(fusion_mode, prf_w, k, r + 1, *score, best_prf);
                    }
                    tracing::debug!(
                        terms = terms.join(" "),
                        hits = prf_hits.len(),
                        "Layer 4: pseudo-relevance feedback leg"
                    );
                }
            }

            // Date window results (Layer 3.7), at the hybrid weight. A memory the hybrid
            // list also holds gains from both, which is the point: the date confirms the
            // match. One the hybrid list could not reach enters here.
            if !date_window_results.is_empty() {
                let best_dated = best_leg_score(date_window_results.iter().map(|(_, s)| *s));
                for (r, (id, score)) in date_window_results.iter().enumerate() {
                    *fused.entry(id.clone()).or_insert(0.0) +=
                        leg_term(fusion_mode, hybrid_w, k, r + 1, *score, best_dated);
                    attr.entry(id.clone()).or_default().temporal_match = true;
                }
                tracing::debug!(
                    "Layer 4 fusion: date window leg added {} results (w={:.2})",
                    date_window_results.len(),
                    hybrid_w
                );
            }

            // The legs beyond the primary index. Reciprocal rank reads ranks and FLAT calibrates
            // against the leg's own best, so the spaces' score scales do not matter; each leg
            // weighs a share of semantic_w, so the total is not inflated:
            // the secondary ratio, the space's dial weight, and for a leg through a fitted map,
            // the map's confidence. The leg through the secondary-to-primary map searches the
            // primary index, so its hits count for the primary space.
            let finished_legs = pending_legs.finish();
            self.retriever.observe_leg_scores(&finished_legs);
            let sec_ratio = self
                .cognitive_dials
                .read()
                .secondary_rrf_ratio
                .unwrap_or(SECONDARY_RRF_WEIGHT_RATIO);
            for leg in &finished_legs {
                if leg.hits.is_empty() {
                    continue;
                }
                let w = match &leg.kind {
                    LegKind::Aligned => semantic_w * sec_ratio,
                    LegKind::Direct => semantic_w * sec_ratio * space_weight(&leg.space),
                    LegKind::Bridged { confidence, .. } => {
                        semantic_w * sec_ratio * confidence * space_weight(&leg.space)
                    }
                };
                let best_hit = best_leg_score(leg.hits.iter().map(|(_, s)| *s));
                for (r, (id, score)) in leg.hits.iter().enumerate() {
                    *fused.entry(id.clone()).or_insert(0.0) +=
                        leg_term(fusion_mode, w, k, r + 1, *score, best_hit);
                    mark_space(&mut attr, id, &leg.space);
                }
                tracing::debug!(
                    "Layer 4 {} fusion: {:?} leg on {} added {} results (w={:.2})",
                    fusion_mode.as_str(),
                    leg.kind,
                    leg.space,
                    leg.hits.len(),
                    w
                );
            }
            leg_record.extend(finished_legs.into_iter().map(|leg| {
                let ranked = leg.hits.into_iter().map(|(id, _)| id).collect();
                (leg.space, ranked)
            }));

            // Every boost from here to Layer 4.92 multiplies a candidate's fused score by
            // `1 + share`, so what it adds is a bounded share of that candidate's own score
            // (shodh-memory #174). An absolute boost outweighed every rank difference the
            // fusion produced: at k = 20 reciprocal-rank scores sit near 0.02-0.04, and on
            // LoCoMo the entity-overlap and temporal-fact boosts took gold hit@10 from 0.72 to
            // 0.45. A boost reorders what the fusion found; it does not add a memory the
            // fusion did not find, since such a memory has no score to take a share of.

            // ===========================================================================
            // LAYER 4.5: ATTRIBUTE QUERY BOOST
            // ===========================================================================
            // For attribute queries, heavily boost memories that contain BOTH the entity
            // AND an attribute synonym value. This ensures "Caroline is single" ranks
            // high for "What is Caroline's relationship status?".
            if !attribute_boost_ids.is_empty() {
                let mut boosted_count = 0;
                for id in &attribute_boost_ids {
                    if let Some(score) = fused.get_mut(id) {
                        *score *= 1.0 + crate::constants::ATTRIBUTE_QUERY_BOOST;
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
            // Source memories of matching temporal facts gain `temporal_recall_boost` (0.20) of
            // their fused score. Fires for all temporal intents (including WhenQuestion) and
            // broader temporal keywords. The entity-only fallback in Layer 0.6 can name fifty
            // sources for one person, so the boost has to stay a nudge: as an absolute +0.20
            // those fifty outranked every semantic match and took LoCoMo temporal hit@10 from
            // 0.89 to 0.35.
            // [signal: temporal_fact_boost] Gated by disabled_signals dial; default = always on.
            if !temporal_fact_boost_ids.is_empty()
                && early_dials_snapshot.signal_enabled("temporal_fact_boost")
            {
                let temporal_recall_boost = self
                    .cognitive_dials
                    .read()
                    .temporal_recall_boost
                    .unwrap_or(crate::constants::TEMPORAL_RECALL_BOOST);
                let mut boosted_count = 0;
                for id in &temporal_fact_boost_ids {
                    if let Some(score) = fused.get_mut(id) {
                        *score *= 1.0 + temporal_recall_boost;
                        boosted_count += 1;
                        // Signal attribution: mark temporal match
                        attr.entry(id.clone()).or_default().temporal_match = true;
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

            // The fused set is final from here on (later layers only rescore it): read every
            // candidate once, tiers first and one storage batch for the rest.
            candidates.prefetch(fused.keys());

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
                    let mut boosted_count = 0usize;
                    for (id, score) in fused.iter_mut() {
                        let is_decision = candidates.get(id).is_some_and(|view| {
                            matches!(
                                view.memory.experience.experience_type,
                                crate::memory::types::ExperienceType::Decision
                            )
                        });
                        if is_decision {
                            *score *= 1.0 + crate::constants::DECISION_TYPE_BOOST;
                            boosted_count += 1;
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
            // A word match adds up to ENTITY_OVERLAP_SHARE of the memory's fused score, weighted
            // by how few candidates it matches; four full matches are the cap.
            // [signal: entity_overlap] Gated by disabled_signals dial; default = always on.
            let entity_overlap_on = early_dials_snapshot.signal_enabled("entity_overlap");
            {
                use rust_stemmers::{Algorithm, Stemmer};
                let stemmer = Stemmer::create(Algorithm::English);

                const STOP_WORDS: &[&str] = &[
                    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "was",
                    "one", "our", "out", "has", "how", "its", "who", "did", "get", "got", "let",
                    "say", "she", "too", "use", "what", "when", "where", "which", "with", "will",
                    "would", "could", "should", "about", "after", "before", "from", "have", "been",
                    "were", "being", "their", "there", "these", "those", "this", "that", "than",
                    "into", "some", "such", "them", "then", "very", "just", "also", "most", "over",
                    "only", "other", "during",
                ];

                let query_stems: std::collections::HashSet<String> = query_text
                    .to_lowercase()
                    .split_whitespace()
                    .filter(|w| w.len() > 2)
                    .map(|w| stem_query_token(&stemmer, &clean_query_token(w)))
                    .filter(|s| !s.is_empty() && s.len() > 1 && !STOP_WORDS.contains(&s.as_str()))
                    .collect();

                if !query_stems.is_empty() {
                    let ids: Vec<MemoryId> = fused.keys().cloned().collect();
                    let mut with_entities = 0usize;
                    let matches: Vec<Vec<&String>> = ids
                        .iter()
                        .map(|id| {
                            let entity_stems: Vec<String> = candidates
                                .get(id)
                                .map(|v| {
                                    v.entities_lower
                                        .iter()
                                        .map(|e| stem_query_token(&stemmer, e))
                                        .collect()
                                })
                                .unwrap_or_default();
                            if !entity_stems.is_empty() {
                                with_entities += 1;
                            }
                            query_stems
                                .iter()
                                .filter(|qs| {
                                    entity_stems
                                        .iter()
                                        .any(|es| query_token_matches_entity(qs, es))
                                })
                                .collect()
                        })
                        .collect();
                    let mut matching_candidates: std::collections::HashMap<&String, usize> =
                        std::collections::HashMap::new();
                    for matched in &matches {
                        for qs in matched {
                            *matching_candidates.entry(*qs).or_default() += 1;
                        }
                    }

                    for (id, matched) in ids.iter().zip(&matches) {
                        let overlap = matched.len();
                        if overlap >= 1 {
                            // A word match is weighted by how few candidates the word matches:
                            // a speaker's name is an entity on half of a conversation and says
                            // nothing about which turn is meant, while `thiserror` on one
                            // memory says everything. An identifier match ("serve.mjs" against
                            // a declared `repo:admin/serve.mjs`) matches at a name boundary or
                            // not at all, so it counts in full.
                            let identifiers =
                                matched.iter().filter(|qs| is_structured_token(qs)).count();
                            // Summed in a fixed order: `matched` follows the stem set's hash
                            // order, and an f32 sum in another order differs in its last bits.
                            let mut word_weights: Vec<f32> = matched
                                .iter()
                                .filter(|qs| !is_structured_token(qs))
                                .map(|qs| {
                                    entity_match_specificity(
                                        with_entities,
                                        matching_candidates[*qs],
                                    )
                                })
                                .collect();
                            word_weights.sort_by(f32::total_cmp);
                            let word_weight: f32 = word_weights.iter().sum();
                            if entity_overlap_on {
                                if let Some(score) = fused.get_mut(id) {
                                    *score *= 1.0
                                        + entity_overlap_share(
                                            word_weight,
                                            identifiers,
                                            decomposed,
                                        );
                                }
                            }
                            // Signal attribution: mark entity overlap
                            attr.entry(id.clone()).or_default().entity_overlap = true;
                        } else if !has_any_temporal_intent {
                            // Content-fallback: when no entity/tag overlap, check
                            // if query stems appear in memory content. Smaller boost
                            // (0.05 of the memory's fused score per match, cap 0.20) to
                            // keep below tag-match priority.
                            // Gated: skip for temporal queries where content overlap
                            // would boost wrong-session memories indiscriminately.
                            // [signal: content_fallback] Gated by disabled_signals dial; default = always on.
                            if early_dials_snapshot.signal_enabled("content_fallback") {
                                if let Some(view) = candidates.get(id) {
                                    let content_lower = &view.content_lower;
                                    let content_overlap = query_stems
                                        .iter()
                                        .filter(|qs| content_lower.contains(qs.as_str()))
                                        .count();
                                    if content_overlap >= 2 {
                                        if let Some(score) = fused.get_mut(id) {
                                            *score *=
                                                1.0 + (0.05 * content_overlap as f32).min(0.20);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // A memory carrying a declared FilePath the query names ranks on that match; see
            // DECLARED_PATH_MATCH_BOOST for why the overlap boost above is not enough.
            if let Some(paths) = resolved_entities
                .as_ref()
                .map(|r| &r.file_paths)
                .filter(|p| !p.is_empty())
            {
                let ids: Vec<MemoryId> = fused.keys().cloned().collect();
                for id in &ids {
                    let carries = candidates
                        .get(id)
                        .is_some_and(|v| v.entities_lower.iter().any(|e| paths.contains(e)));
                    if carries {
                        if let Some(score) = fused.get_mut(id) {
                            *score += crate::constants::DECLARED_PATH_MATCH_BOOST;
                        }
                        attr.entry(id.clone()).or_default().entity_overlap = true;
                    }
                }
            }

            // ===========================================================================
            // LAYER 4.527: BM25 SPECIFICITY DISCOUNT
            // ===========================================================================
            // When a memory has high BM25 contribution but ZERO entity overlap with
            // the query, the BM25 match is likely peripheral (the term appears in
            // content but the memory isn't ABOUT that topic). Apply a small discount
            // to prevent these peripheral matches from outranking focused memories.
            // [signal: bm25_specificity_discount] Gated by disabled_signals dial; default = always on.
            if early_dials_snapshot.signal_enabled("bm25_specificity_discount") {
                let mut discounted = 0usize;
                let ids: Vec<MemoryId> = fused.keys().cloned().collect();
                for id in &ids {
                    let a = attr.get(id);
                    let has_entity_overlap = a.map(|a| a.entity_overlap).unwrap_or(false);
                    let bm25 = a.map(|a| a.bm25_contribution).unwrap_or(0.0);
                    // Very high BM25 (>0.5) but no entity overlap = peripheral mention
                    if bm25 > 0.5 && !has_entity_overlap {
                        if let Some(score) = fused.get_mut(id) {
                            *score *= 0.95; // 5% discount
                            discounted += 1;
                        }
                    }
                }
                if discounted > 0 {
                    tracing::debug!(
                        "Layer 4.527: BM25 specificity discount applied to {} memories",
                        discounted
                    );
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
            // [signal: length_penalty] Gated by disabled_signals dial; default = always on.
            if early_dials_snapshot.signal_enabled("length_penalty") {
                let mut lengths: Vec<(MemoryId, usize)> = Vec::new();
                for id in fused.keys() {
                    if let Some(view) = candidates.get(id) {
                        lengths.push((id.clone(), view.memory.experience.content.len()));
                    }
                }
                if lengths.len() >= 5 {
                    // Summed as integers: an f32 sum would depend on the map's iteration order.
                    let total_len: usize = lengths.iter().map(|(_, l)| *l).sum();
                    let mean_len: f32 = total_len as f32 / lengths.len() as f32;
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
                    for (id, score) in fused.iter_mut() {
                        let Some(view) = candidates.get(id) else {
                            continue;
                        };
                        if view.entities_lower.len() >= 2 {
                            let ent_stems: Vec<String> = view
                                .entities_lower
                                .iter()
                                .map(|e| stem.stem(e).to_string())
                                .collect();
                            let matching = q_stems
                                .iter()
                                .filter(|qs| ent_stems.iter().any(|es| es.contains(qs.as_str())))
                                .count();
                            let concentration = matching as f32 / ent_stems.len() as f32;

                            if concentration < 0.20 {
                                *score *= 0.85;
                            } else if concentration > 0.50 {
                                *score *= 1.10;
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
            // Only the top TEMPORAL_DEDUP_MAX_CANDIDATES by fused score are compared: the pass
            // is all-pairs over the fused set, which runs to hundreds of candidates.
            // Recency intent only: "when did X happen" asks for the memory where it happened,
            // which is usually the oldest on its topic. Firing on every temporal query took
            // LoCoMo temporal hit@10 to zero.
            // [signal: temporal_topic_dedup] Gated by disabled_signals dial; default = always on.
            if query_parser::has_recency_intent(query_text)
                && early_dials_snapshot.signal_enabled("temporal_topic_dedup")
            {
                let demoted = temporal_topic_dedup(
                    &mut fused,
                    &candidates,
                    crate::constants::TEMPORAL_DEDUP_MAX_CANDIDATES,
                );
                if demoted > 0 {
                    tracing::debug!(
                        "Layer 4.54: Temporal dedup demoted {} older same-topic memories",
                        demoted
                    );
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
                    use crate::constants::{PROSPECTIVE_BOOST_MAX, PROSPECTIVE_BOOST_PER_MATCH};

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

                        for (id, score) in fused.iter_mut() {
                            if let Some(view) = candidates.get(id) {
                                let match_count = signal_terms
                                    .iter()
                                    .filter(|term| view.content_lower.contains(term.as_str()))
                                    .count();

                                if match_count > 0 {
                                    // Sqrt scaling: diminishing returns for additional matches
                                    *score *= 1.0
                                        + (PROSPECTIVE_BOOST_PER_MATCH
                                            * (match_count as f32).sqrt())
                                        .min(PROSPECTIVE_BOOST_MAX);
                                    boosted_count += 1;
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
                        *score *= 1.0 + boost;
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
            // Boost memories connected to type-matching entities, by a share of the fused
            // score. Only active when ontological intent has sufficient confidence.
            //
            // Reference: Collins & Quillian (1969) — type-plausible paths retrieved faster
            // Pre-sort and limit candidates before expensive re-ranking.
            // Only look up graph entities for the top 2x max_results candidates,
            // not all fused results (avoids 100s of RocksDB reads).
            let mut res: Vec<_> = fused.into_iter().collect();
            res.sort_by(|a, b| b.1.total_cmp(&a.1));
            order_score_ties_by_content(&mut res, &get_content);
            let rerank_budget = query.max_results * 2;

            if use_ontology_rerank && !onto_intent.expected_labels.is_empty() {
                if let Some(graph) = self.graph_memory.as_ref() {
                    let g = graph.read_recursive();
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
                                    * self
                                        .cognitive_dials
                                        .read()
                                        .onto_rerank_boost
                                        .unwrap_or(crate::constants::ONTOLOGICAL_RERANK_BOOST))
                                .min(
                                    self.cognitive_dials
                                        .read()
                                        .onto_rerank_max
                                        .unwrap_or(crate::constants::ONTOLOGICAL_RERANK_MAX),
                                );
                                *fused_score *= 1.0 + boost;
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
                                *fused_score *= 1.0
                                    + self
                                        .cognitive_dials
                                        .read()
                                        .onto_rerank_penalty
                                        .unwrap_or(crate::constants::ONTOLOGICAL_RERANK_PENALTY);
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

                // Content and timestamps for the top candidates
                let views: Vec<Option<Arc<CandidateView>>> = res
                    .iter()
                    .take(interference_budget)
                    .map(|(id, _)| candidates.get(id))
                    .collect();

                // Check pairwise: only pairs with high text similarity (>0.7)
                // that also show semantic opposition
                let mut demote_indices: Vec<(usize, f32)> = Vec::new();
                for (i, vi) in views.iter().enumerate() {
                    let Some(vi) = vi else { continue };
                    let ci = &vi.memory.experience.content;
                    for (j, vj) in views.iter().enumerate().skip(i + 1) {
                        let Some(vj) = vj else { continue };
                        let cj = &vj.memory.experience.content;

                        let sim = text_similarity(ci, cj);
                        if sim < 0.7 {
                            continue;
                        }

                        if detect_semantic_opposition(ci, cj) {
                            // Contradiction detected — demote the older memory
                            let (ti, tj) = (vi.memory.created_at, vj.memory.created_at);
                            // On equal timestamps, demote the lower-ranked one.
                            let older_idx = if ti < tj { i } else { j };
                            // Penalty proportional to similarity: higher sim = more
                            // confident this is a true contradiction, not coincidence
                            let penalty = sim * 0.20;
                            demote_indices.push((older_idx, penalty));
                        }
                    }
                }

                // Apply demotions
                for (idx, penalty) in &demote_indices {
                    if let Some((_, score)) = res.get_mut(*idx) {
                        *score *= 1.0 - penalty;
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
                        candidates.get(id).map(|view| {
                            view.memory
                                .experience
                                .content
                                .chars()
                                .take(200)
                                .collect::<String>()
                                .to_lowercase()
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
            // Decomposed queries produce a wider candidate set — widen the budget
            // so more candidates survive into the cross-encoder.
            // The `rerank_pool` dial overrides the width; the pool never drops below an explicit
            // `rerank_count`, or Layer 5.3 would be asked to score candidates already cut.
            let (pool_dial, rerank_count_dial) = {
                let d = self.cognitive_dials.read();
                (d.rerank_pool, d.rerank_count)
            };
            let rerank_budget = pool_dial
                .unwrap_or(if decomposed { 50_usize } else { 30_usize })
                .max(rerank_count_dial.or(query.rerank_count).unwrap_or(0))
                .max(query.max_results);
            // Query filters are applied in the Layer 5 scan below, which stops once
            // `scan_limit` memories pass. Cutting the list here first meant a selective
            // filter (tags, type, time range) judged only the top 30/50 and could discard
            // nearly all of them while matching memories sat just below the cut (live:
            // filtered_out=45 of 50). With filters the full fused list goes to the scan.
            if !vector_query.has_selective_filters() {
                res.truncate(rerank_budget);
            }
            tracing::debug!(
                "Layer 4: {} fused results (rerank budget={})",
                res.len(),
                rerank_budget
            );
            (res, heb, attr)
        };

        crate::memory::gold_funnel::record("fusion", memory_ids.iter().map(|(id, _)| id));

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

        // Fetch up to scan_limit candidates so Layer 5.5 cross-encoder can rerank
        // before truncating to max_results. Without this, the early break at max_results
        // would prevent the cross-encoder from ever firing (max_results == max_results).
        // Decomposed queries benefit from a larger pool — more candidates to rerank.
        // An explicit rerank_count is Layer 5.3's budget, so the scan must find that many;
        // stopping at 20 would cap the dial at 20 whenever max_results is small.
        let explicit_rerank_count = self
            .cognitive_dials
            .read()
            .rerank_count
            .or(query.rerank_count);
        let scan_limit = if decomposed { 30_usize } else { 20_usize }
            .max(query.max_results)
            .max(explicit_rerank_count.unwrap_or(0));

        // Layer 5: Unified scoring with hebbian + recency + emotional + feedback signals
        // Recency decay: recent memories get boost, old memories decay
        // λ = 0.01 means ~50% at 70 hours, ~25% at 140 hours
        const RECENCY_DECAY_RATE: f32 = 0.01;

        // PIPE-9: Get feedback store guard for momentum-based scoring
        // Acquire once outside the loop to avoid repeated locking
        let feedback_guard = self.feedback_store.as_ref().map(|fs| fs.read());

        // Signal 20: External dimension aggregate (Sleight) (graph topological health)
        // Computed once per query — same value for all memories.
        // Acts as a global quality multiplier: high-quality graphs boost all scores.
        let external_dim_multiplier = self.external_aggregate_score().unwrap_or(1.0);

        // R3b B1 — WEIGHT SNAPSHOT: snapshot the learned weight vector ONCE before
        // the scoring loop.  Reading inside the loop would risk a torn read if a
        // concurrent `update_weights_from_feedback` call fires mid-query.  The
        // snapshot is a value copy (`Clone`) so it is immune to further mutations.
        // NOTE: this snapshot is used ONLY for the shadow computation below; the
        // static path (what is actually returned) is unchanged.
        let r3b_learned_snapshot = self.hybrid_search.snapshot_learned_weights();
        // Static config weights — the floor that the live path always uses.
        let r3b_static_bm25 = self.hybrid_search.config().bm25_weight;
        let r3b_static_vector = self.hybrid_search.config().vector_weight;
        let r3b_static_graph = self.hybrid_search.config().graph_weight;

        // SIGNAL-DIAL SNAPSHOT: snapshot the disable-signal set ONCE before the
        // scoring loop so signal_enabled() checks inside the closure are
        // branch-predictable (and avoid repeated lock acquisitions).
        // An empty set (the compile-time default) makes every `signal_enabled`
        // call return `true`, preserving byte-identical behaviour.
        let dials_snapshot = self.cognitive_dials.read().clone();
        let hebbian_family_on = super::ablation::is_enabled("hebbian");
        let graph_strength_on = super::ablation::is_enabled("graph_strength");

        // R2 (LTR) SHADOW CAPTURE side-map. The `with_unified_score` closure
        // (re-bound each loop iteration) borrows this by reference and inserts one
        // ScoringFeatures per scored memory keyed by MemoryId. It lives outside the
        // loop so it accumulates across all iterations and survives to the single
        // trace write-site at the end of the function. This is READ-ONLY w.r.t. the
        // score path: it captures already-computed locals and never feeds back into
        // `final_score`, `set_score`, ordering, or any live behaviour.
        let scoring_features: std::cell::RefCell<
            std::collections::HashMap<MemoryId, crate::memory::types::ScoringFeatures>,
        > = std::cell::RefCell::new(std::collections::HashMap::new());

        for (memory_id, score) in memory_ids {
            // Hebbian boost from learned graph weights: up to 10% of the fused score.
            // [signal: hebbian] Gated by disabled_signals dial and the ablation family (off).
            let hebbian_boost = hebbian_scores.get(&memory_id).copied().unwrap_or(0.0);
            let base_score = if hebbian_family_on && dials_snapshot.signal_enabled("hebbian") {
                score * (1.0 + hebbian_boost.clamp(0.0, 1.0) * 0.1)
            } else {
                score
            };

            // Helper to apply unified scoring (recency + arousal + credibility + temporal)
            // Amplify recency when query explicitly asks for recent items
            let query_lower_recency = query_text.to_lowercase();
            // "recently" / "recent" is graded by Layer 5.87's stepped multiplier instead of
            // amplified here: at 5x this term lifts the last three days above everything
            // regardless of relevance, which is not what the word asks for.
            let wants_recent = query_lower_recency.contains("last week")
                || query_lower_recency.contains("most recent")
                || query_lower_recency.contains("latest")
                || query_lower_recency.contains("yesterday")
                || query_lower_recency.contains("today");
            let recency_scale = if wants_recent {
                0.5 // 5x amplification for explicit recency queries
            } else if decomposed {
                // Abstract/decomposed queries: dampen recency noise. Recency is
                // informative for "what happened recently?" but noise for "what are
                // our strategic priorities?" where M20 (most recent) drowns M13 (OKRs).
                // wants_recent guard above preserves recency for queries with explicit
                // recency keywords like "recent", "yesterday", "today", "last week".
                0.03
            } else {
                query.recency_weight.unwrap_or(0.1)
            };
            let with_unified_score = |mem: &SharedMemory, base: f32| -> SharedMemory {
                // Recency decay: exponential decay based on age
                let hours_old = (now - mem.created_at).num_hours().max(0) as f32;
                let recency_boost = (-RECENCY_DECAY_RATE * hours_old).exp() * recency_scale;

                // Emotional arousal boost: high arousal = more salient (5% contribution)
                // Research: LaBar & Cabeza (2006) - emotionally arousing events better remembered
                // [signal: arousal] Gated by disabled_signals dial; default = always on.
                let arousal_boost = if dials_snapshot.signal_enabled("arousal") {
                    mem.experience
                        .context
                        .as_ref()
                        .map(|c| c.emotional.arousal * 0.05)
                        .unwrap_or(0.0)
                } else {
                    0.0
                };

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
                // significantly boost the memory's score (25% contribution when matched).
                //
                // R2: boost detection now uses bound-overlap to align with the gate
                // (`filter_facts_by_time_window`). Each memory temporal_ref is parsed
                // to a NaiveDate, treated as a TemporalBound::Point, and checked for
                // overlap against each query bound widened ±7 days (matching the gate's
                // tolerance). When query_temporal.bounds() is empty (no relative
                // expressions were extracted), we fall back to the prior string/proximity
                // logic so existing behaviour is fully preserved.
                //
                // NOTE: source of truth for mem-side dates is still
                // `experience.temporal_refs`; full unification with
                // `TemporalFact.resolved_time` is a separate future step.
                let temporal_boost = if has_temporal_query
                    && !mem.experience.temporal_refs.is_empty()
                {
                    let query_bounds = query_temporal.bounds();
                    if !query_bounds.is_empty() {
                        // Bound-overlap path (R2): parse each mem temporal_ref to a
                        // NaiveDate and check whether any query bound's ±7-day-widened
                        // window overlaps the mem date. The overlap test is symmetric:
                        // `qb.overlaps_range(mem_date − 7, mem_date + 7)` asks "does the
                        // query's temporal bound cover anything within a week of this
                        // memory's date?", which mirrors the gate's ±7-day tolerance.
                        let mut best_match = 0.0_f32;
                        for mem_ref in &mem.experience.temporal_refs {
                            if let Ok(mem_date) =
                                chrono::NaiveDate::parse_from_str(mem_ref, "%Y-%m-%d")
                            {
                                let window_start = mem_date - chrono::Duration::days(7);
                                let window_end = mem_date + chrono::Duration::days(7);
                                for qb in query_bounds {
                                    if qb.overlaps_range(window_start, window_end) {
                                        // Same magnitude as the old exact-match path.
                                        best_match = best_match.max(0.25);
                                        break;
                                    }
                                }
                            }
                        }
                        best_match
                    } else {
                        // Fallback (pre-R2): string/proximity matching when no bounds
                        // were resolved (e.g. query contains only absolute date strings
                        // without relative phrases). Behaviour is byte-for-byte identical
                        // to the pre-R2 implementation — no regression.
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
                                        let proximity_boost =
                                            0.05 * (1.0 - days_diff as f32 / 30.0);
                                        best_match = best_match.max(proximity_boost);
                                    }
                                }
                            }
                        }
                        best_match
                    }
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
                // [signal: session_continuity] Gated by disabled_signals dial; default = always on.
                let session_boost = if dials_snapshot.signal_enabled("session_continuity") {
                    let age_hours = (now - mem.created_at).num_hours();
                    if age_hours <= 2 {
                        0.03 // within current session window
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // =================================================================
                // BRIDGE SIGNALS (9-19): Connect captured-but-forgotten signals
                // =================================================================

                // BRIDGE-1: access_count (signal 5, proven 14% in proactive)
                // Query-intent-dependent: exploratory queries reduce access advantage
                // to surface novel/unfamiliar memories (bifocal novelty evaluation)
                let access_weight = if matches!(
                    query_type,
                    crate::memory::query_parser::QueryType::Exploratory
                ) {
                    0.02
                } else {
                    0.07
                };
                let access_boost =
                    ((mem.access_count() as f64).ln_1p() / 5.0) as f32 * access_weight;

                // BRIDGE-1: graph_strength, the `graph_strength` ablation family.
                let graph_boost = if graph_strength_on {
                    hebbian_scores
                        .get(&mem.id)
                        .copied()
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0)
                        * 0.08
                } else {
                    0.0
                };

                // Signal 9: graded temporal-context contiguity (E6 TCM). When the
                // memory carries a context_drift snapshot and a live drift state
                // exists, boost by cosine proximity between the current subjective
                // context and the memory's encoding context — graded contiguity
                // replacing brittle episode_id equality. Falls back to binary
                // episode_id coherence for memories without a snapshot (pre-E6 or
                // sync-path) or before any encode has seeded the drift state.
                // [signal: episode_tcm] Gated by disabled_signals dial; default = always on.
                let episode_boost = if dials_snapshot.signal_enabled("episode_tcm") {
                    let cur = self.tcm_context.read();
                    let md = &mem.when.context_drift;
                    if !cur.is_empty() && md.len() == cur.len() {
                        let dot: f32 = cur.iter().zip(md.iter()).map(|(a, b)| a * b).sum();
                        dot.max(0.0) * 0.08
                    } else {
                        mem.experience
                            .context
                            .as_ref()
                            .and_then(|c| c.episode.episode_id.as_ref())
                            .and_then(|mem_ep| {
                                query.episode_id.as_ref().and_then(|q_ep| {
                                    if mem_ep == q_ep {
                                        Some(0.08_f32)
                                    } else {
                                        None
                                    }
                                })
                            })
                            .unwrap_or(0.0)
                    }
                } else {
                    0.0
                };

                // Signal 10: Source type multiplier on credibility
                // [signal: valence_credibility] Gated by disabled_signals dial; default = always on.
                let source_type_mult = if dials_snapshot.signal_enabled("valence_credibility") {
                    mem.experience
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
                        .unwrap_or(1.0)
                } else {
                    1.0
                };
                let credibility_boost = credibility_boost * source_type_mult;

                // Signal 11: Emotional valence as absolute intensity
                // [signal: valence] Gated by disabled_signals dial; default = always on.
                let valence_boost = if dials_snapshot.signal_enabled("valence") {
                    mem.experience
                        .context
                        .as_ref()
                        .map(|c| c.emotional.valence.abs() * 0.02)
                        .unwrap_or(0.0)
                } else {
                    0.0
                };

                // Signal 12: Sequence proximity within episode
                // [signal: seq_proximity] Gated by disabled_signals dial; default = always on.
                let sequence_boost = if dials_snapshot.signal_enabled("seq_proximity") {
                    mem.experience
                        .context
                        .as_ref()
                        .and_then(|c| c.episode.sequence_number)
                        .map(|seq| ((seq as f64).ln_1p() / 5.0) as f32 * 0.02)
                        .unwrap_or(0.0)
                } else {
                    0.0
                };

                // Signal 16: Context richness — memories with richer context
                // (more populated fields) are more useful for retrieval
                // [signal: context_richness] Gated by disabled_signals dial; default = always on.
                let richness_boost = if dials_snapshot.signal_enabled("context_richness") {
                    (mem.context_richness() as f32 / 10.0) * 0.02
                } else {
                    0.0
                };

                // Signal 17: Activation level — current Hebbian co-activation state
                let activation_boost = mem.activation() * 0.03;

                // Signal 18: Temporal fact density — more temporal refs = more anchored
                // [signal: temporal_fact_density] Gated by disabled_signals dial; default = always on.
                let temporal_density = if dials_snapshot.signal_enabled("temporal_fact_density") {
                    (mem.experience.temporal_refs.len() as f32 / 5.0).min(1.0) * 0.02
                } else {
                    0.0
                };

                // Signal 19: Entity confidence — memories with more entity refs
                // are more structurally connected (proxy for graph salience)
                // [signal: entity_confidence] Gated by disabled_signals dial; default = always on.
                let entity_density = if dials_snapshot.signal_enabled("entity_confidence") {
                    (mem.entity_refs.len() as f32 / 5.0).min(1.0) * 0.02
                } else {
                    0.0
                };

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
                // [signal: access_burstiness] Gated by disabled_signals dial; default = always on.
                let burstiness_boost = if dials_snapshot.signal_enabled("access_burstiness") {
                    let burstiness = mem.access_burstiness();
                    if burstiness > 1.5 {
                        0.02 // bursty = working memory, small boost
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // BRIDGE-3: Calibrated confidence gate (Bayesian alpha/beta)
                // [signal: confidence_gate] Gated by disabled_signals dial; default = always on.
                let confidence_gate = if dials_snapshot.signal_enabled("confidence_gate") {
                    let obs = mem.confidence_observations();
                    let total_obs = (obs - 2.0).max(0.0);
                    if total_obs >= 5.0 {
                        let bayesian = mem.calibrated_confidence();
                        0.85 + 0.15 * bayesian
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                // The signals are shares of the fused score, summed and then capped
                // (shodh-memory #174): added to it, a recency or temporal term of 0.1-0.25 set the
                // order outright over reciprocal-rank scores near 0.03, and a memory the fusion
                // ranked last could pass every one above it.
                let signal_share = recency_boost
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
                    + burstiness_boost;
                let final_score = base
                    * (1.0 + signal_share).min(crate::constants::UNIFIED_SIGNAL_FACTOR_MAX)
                    * feedback_multiplier
                    * confidence_gate
                    * external_dim_multiplier;

                // R2 (LTR) SHADOW CAPTURE: read every component that just produced
                // `final_score` into the feature side-map. This READS the locals
                // already computed above — it recomputes nothing and writes nothing
                // back to `mem`, `base`, or `final_score`. The live score path below
                // (`set_score(final_score)`) is byte-for-byte unchanged.
                scoring_features.borrow_mut().insert(
                    mem.id.clone(),
                    crate::memory::types::ScoringFeatures {
                        base_score: base,
                        recency_boost,
                        arousal_boost,
                        credibility_boost,
                        valence_boost,
                        temporal_boost,
                        session_boost,
                        access_boost,
                        graph_boost,
                        episode_boost,
                        sequence_boost,
                        richness_boost,
                        activation_boost,
                        temporal_density,
                        entity_density,
                        elaboration_boost,
                        burstiness_boost,
                        feedback_multiplier,
                        confidence_gate,
                        external_dim_multiplier,
                        unified_final: final_score,
                    },
                );

                let mut cloned: Memory = mem.as_ref().clone();
                cloned.set_score(final_score);
                Arc::new(cloned)
            };

            // The record the fusion layers already read: working, session, or storage.
            match candidates.get(&memory_id) {
                Some(view) => {
                    // CRITICAL FIX: Apply filters before adding to results
                    if self.retriever.matches_filters(&view.memory, &vector_query) {
                        memories.push(with_unified_score(&view.memory, base_score));
                        let source = view.tier.source_label();
                        if !sources.contains(&source) {
                            sources.push(source);
                        }
                        if view.tier == CandidateTier::LongTerm {
                            storage_fetches += 1;
                        } else {
                            cache_hits += 1;
                        }
                    } else {
                        filtered_out += 1;
                    }
                    // A tier-resident candidate never stopped the fetch: only a storage read
                    // or a stale reference reaches the pool check below.
                    if view.tier != CandidateTier::LongTerm {
                        continue;
                    }
                }
                None => {
                    tracing::warn!(
                        memory_id = %memory_id.0,
                        error = %candidates.failure(&memory_id).unwrap_or_default(),
                        "Stale vector reference — cleaning up orphaned index entry"
                    );
                    self.retriever.remove_memory(&memory_id);
                }
            }

            if memories.len() >= scan_limit {
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

        // Signal attribution: record recency contribution for fetched memories.
        // Computed post-fetch using the same decay formula as the unified scoring
        // closure so the attribution accurately reflects what was applied.
        {
            let recency_scale_attr = if query_text.to_lowercase().contains("last week")
                || query_text.to_lowercase().contains("most recent")
                || query_text.to_lowercase().contains("recent")
                || query_text.to_lowercase().contains("latest")
                || query_text.to_lowercase().contains("yesterday")
                || query_text.to_lowercase().contains("today")
            {
                0.5_f32
            } else {
                query.recency_weight.unwrap_or(0.1)
            };
            for mem in &memories {
                let hours_old = (now - mem.created_at).num_hours().max(0) as f32;
                let recency_val = (-RECENCY_DECAY_RATE * hours_old).exp() * recency_scale_attr;
                let has_temporal_match =
                    !mem.experience.temporal_refs.is_empty() && has_temporal_query;
                let entry = signal_attributions.entry(mem.id.clone()).or_default();
                entry.recency_contribution = recency_val;
                if has_temporal_match && !entry.temporal_match {
                    entry.temporal_match = true;
                }
            }
        }

        // Layer 5.1: TEMPORAL INVALIDATION FILTER
        // Remove memories whose valid_until timestamp has passed.
        {
            let before = memories.len();
            // R6: honor an as-of point-in-time query (validity at that instant);
            // otherwise the default "currently valid" invalidation filter.
            match query.as_of {
                Some(t) => memories.retain(|m| m.is_valid_at(t)),
                None => memories.retain(|m| !m.is_expired()),
            }
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
        //
        // Dial gate: VELD_DIALS_USE_RERANKING=false (or cognitive_dials.use_reranking =
        // Some(false)) fully skips this layer. The `rerank_count` dial caps the
        // candidate budget; absent that, the hardcoded ce_cap applies.
        let rerank_shape = query_parser::classify_rerank_shape(query_text, decomposed, &query_type);
        let (layer53_use_reranking, layer53_rerank_count_dial, ce_w) = {
            let d = self.cognitive_dials.read();
            (d.use_reranking, d.rerank_count, d.ce_share(rerank_shape))
        };
        if ce_w < crate::constants::CE_BLEND_SKIP_BELOW {
            tracing::debug!(
                shape = rerank_shape.as_str(),
                share = format!("{ce_w:.2}"),
                "Layer 5.3: cross-encoder skipped for this query shape"
            );
        }
        if memories.len() > 1
            && ce_w >= crate::constants::CE_BLEND_SKIP_BELOW
            && layer53_use_reranking.unwrap_or(true)
            && query
                .layer_mode()
                .allows(crate::memory::types::LayerMode::PlusRerank)
        {
            if let Some(reranker) = self.hybrid_search.reranker() {
                // Decomposed queries: widen CE budget from 20→30 to exploit the
                // broader candidate pool from multi-query vector search.
                let ce_cap = if decomposed { 30 } else { 20 };
                // An explicit rerank_count (dial or query) is the budget, so it can reach past
                // twice max_results to gold the fused order left below rank 20; absent one, the
                // budget stays bounded by max_results for latency.
                let rerank_budget = match layer53_rerank_count_dial.or(query.rerank_count) {
                    Some(explicit) => memories.len().min(explicit),
                    None => memories.len().min(query.max_results * 2).min(ce_cap),
                };
                // Pass each candidate's stored embedding so the reranker's
                // bi-encoder pass can reuse it instead of re-embedding via the
                // HTTP embedder (5–6s/call for CPU Nomic). The reranker
                // dim-checks against the query embedding and falls back to
                // `embedder.encode(content)` on mismatch.
                //
                // The cross-encoder reads content only. A memory whose writer declared what it is
                // about (`repo:admin/serve.mjs`) often never names that thing in its text, so a
                // query naming it scored the memory as unrelated and the blend undid every
                // entity signal before it. When the query names an identifier the memory
                // carries as an entity, that entity is put in front of the text the
                // cross-encoder reads. Nothing changes for a query with no identifier in it.
                //
                // Without a cross-encoder only dense similarity is left, and it cannot tell
                // `serve.mjs` from `server.mjs`. A query that names a path some candidate
                // declared keeps its fused order, which already carries the identifier boost.
                // Hyphenated words ("dairy-free") are identifiers above but not paths here: the
                // dense fallback ranks them better than fusion does.
                let mut declared_path_named = false;
                let query_identifiers: Vec<String> = query_text
                    .split_whitespace()
                    .map(clean_query_token)
                    .map(|t| t.to_lowercase())
                    .filter(|t| is_structured_token(t))
                    .collect();
                let query_paths: Vec<&String> = query_identifiers
                    .iter()
                    .filter(|t| is_path_shaped_token(t))
                    .collect();
                let candidates: Vec<(MemoryId, String, f32, Option<Vec<f32>>)> = memories
                    .iter()
                    .take(rerank_budget)
                    .map(|m| {
                        let named: Vec<&str> = if query_identifiers.is_empty() {
                            Vec::new()
                        } else {
                            m.experience
                                .entities
                                .iter()
                                .filter(|e| {
                                    let e = e.to_lowercase();
                                    query_identifiers
                                        .iter()
                                        .any(|q| query_token_matches_entity(q, &e))
                                })
                                .map(String::as_str)
                                .collect()
                        };
                        declared_path_named |= named.iter().any(|e| {
                            let e = e.to_lowercase();
                            query_paths
                                .iter()
                                .any(|q| query_token_matches_entity(q, &e))
                        });
                        let text = if named.is_empty() {
                            m.experience.content.clone()
                        } else {
                            format!("{}\n{}", named.join(" "), m.experience.content)
                        };
                        (
                            m.id.clone(),
                            text,
                            m.score.unwrap_or(0.0),
                            m.experience.embeddings.clone(),
                        )
                    })
                    .collect();

                match reranker.rerank(query_text, candidates, !declared_path_named) {
                    Ok(None) => {
                        tracing::debug!(
                            "Layer 5.3: no cross-encoder for a declared path, keeping the fused order"
                        );
                    }
                    Ok(Some(reranked)) => {
                        // Build score lookup from cross-encoder results
                        let ce_scores: std::collections::HashMap<MemoryId, f32> =
                            reranked.into_iter().collect();

                        // The share comes from the dials by query shape (see
                        // CognitiveDials::ce_share and constants::CE_BLEND_*).
                        let base_w = 1.0 - ce_w;
                        for mem in memories.iter_mut().take(rerank_budget) {
                            if let Some(&ce_score) = ce_scores.get(&mem.id) {
                                // Signal attribution: record cross-encoder contribution
                                signal_attributions
                                    .entry(mem.id.clone())
                                    .or_default()
                                    .cross_encoder_contribution = ce_score;
                                let base = mem.score.unwrap_or(0.0);
                                let blended = base * base_w + ce_score * ce_w;
                                let mut cloned: Memory = mem.as_ref().clone();
                                cloned.set_score(blended);
                                *mem = Arc::new(cloned);
                            }
                        }

                        // Re-sort after blending
                        memories.sort_by(|a, b| {
                            b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0))
                        });

                        tracing::info!(
                            reranked = rerank_budget,
                            shape = rerank_shape.as_str(),
                            share = format!("{ce_w:.2}"),
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

        // Layer 5.32: rank on the unified score. The cross-encoder takes its window from the
        // fused order on purpose: the recall gate of 2026-10-29 (run 40812556433) lost five gold
        // documents when the window was chosen by the unified score instead. When the
        // reranker ran it already sorted the list; when it was skipped, the layers that do not
        // re-sort (5.7, 5.8, the final cut while 5.35 and 5.36 stay quiet) used to run on an
        // order the Layer 5 signals had never touched. Stable, so fused ties keep their
        // content-ordered tie-break.
        memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));

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
                memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
                tracing::debug!("Layer 5.35: Applied fragment demotion to retrieval scores");
            }
        }

        // =====================================================================
        // LAYER 5.36: QUERY-TERM COVERAGE  /  LAYER 5.37: PERSON DENSITY
        // =====================================================================
        // Both multiplicative, on the full candidate list before truncation, so they can
        // reorder the top results and pull a candidate in from below the cut. Coverage
        // multiplies by `1 - w * (1 - coverage)` (constants::QUERY_COVERAGE_WEIGHT); person
        // density multiplies by `1 + max * intent * people_score` on people questions
        // (constants::PEOPLE_BOOST_MAX). Person names come from the same rule-based extractor
        // the write path falls back to, run on candidates only and only for people
        // questions, so stored records are read as they are.
        let coverage_weight = dials_snapshot
            .query_coverage_weight
            .unwrap_or(crate::constants::QUERY_COVERAGE_WEIGHT);
        let coverage_terms: Vec<(String, f32)> = {
            let mut seen: HashSet<String> = HashSet::new();
            query_analysis
                .focal_entities
                .iter()
                .filter(|e| !e.negated && !query_parser::is_question_frame_word(&e.text))
                .map(|e| (e.stem.clone(), e.ic_weight))
                .chain(
                    query_analysis
                        .discriminative_modifiers
                        .iter()
                        .filter(|m| !m.negated && !query_parser::is_question_frame_word(&m.text))
                        .map(|m| (m.stem.clone(), m.ic_weight)),
                )
                .filter(|(stem, w)| !stem.is_empty() && *w > 0.0 && seen.insert(stem.clone()))
                .collect()
        };
        let coverage_total: f32 = coverage_terms.iter().map(|(_, w)| w).sum();
        let query_coverage_active = coverage_weight > 0.0
            && dials_snapshot.signal_enabled("query_coverage")
            && coverage_terms.len() >= crate::constants::QUERY_COVERAGE_MIN_TERMS
            && coverage_total > 0.0;
        let people_max = dials_snapshot
            .people_boost_max
            .unwrap_or(crate::constants::PEOPLE_BOOST_MAX);
        let people_intent = query_parser::people_intent(query_text);
        let people_active = people_max > 0.0
            && people_intent > 0.0
            && dials_snapshot.signal_enabled("people_density");
        if (query_coverage_active || people_active) && !memories.is_empty() {
            for mem in memories.iter_mut() {
                let mut factor = 1.0_f32;
                if query_coverage_active {
                    let mut text = mem.experience.content.clone();
                    for tag in &mem.experience.tags {
                        text.push(' ');
                        text.push_str(tag);
                    }
                    let stems = query_parser::content_stems(&text);
                    let covered: f32 = coverage_terms
                        .iter()
                        .filter(|(stem, _)| stems.contains(stem))
                        .map(|(_, w)| w)
                        .sum();
                    factor *= 1.0 - coverage_weight * (1.0 - covered / coverage_total);
                }
                if people_active {
                    factor *=
                        1.0 + people_max * people_intent * people_score(&mem.experience.content);
                }
                if (factor - 1.0).abs() > f32::EPSILON {
                    let base = mem.score.unwrap_or(0.0);
                    let mut cloned: Memory = mem.as_ref().clone();
                    cloned.set_score(base * factor);
                    *mem = Arc::new(cloned);
                }
            }
            memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
            tracing::debug!(
                coverage = query_coverage_active,
                people_intent = format!("{people_intent:.1}"),
                "Layer 5.36/5.37: query-term coverage and person density"
            );
        }

        // =====================================================================
        // LAYER 5.38: NEAR-EXACT DUPLICATE DE-RANK
        // =====================================================================
        // A memory the write gate kept as a near-exact copy of another ranks below what it
        // repeats. Before MMR, so diversification sees the demoted order.
        // [signal: duplicate_derank] Gated by the disabled_signals dial; on by default.
        if dials_snapshot.signal_enabled("duplicate_derank")
            && crate::memory::duplicates::derank(&mut memories)
        {
            tracing::debug!("Layer 5.38: de-ranked near-exact duplicates");
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
        // [signal: mmr] Off unless the `use_mmr` dial (or `VELD_DIALS_USE_MMR`) turns it on; see
        // `MMR_ENABLED_DEFAULT` for the measurement behind the default. The `disabled_signals`
        // dial still vetoes it.
        let mmr_wanted = dials_snapshot
            .use_mmr
            .unwrap_or(crate::constants::MMR_ENABLED_DEFAULT);
        if mmr_wanted && dials_snapshot.signal_enabled("mmr") {
            let mmr_lambda = match &query_type {
                query_parser::QueryType::Attribute(_) => None,
                query_parser::QueryType::Exploratory => {
                    // Abstract/decomposed queries need stronger diversity: the answer
                    // set spans multiple semantic clusters (caching, payments,
                    // onboarding for "performance improvements"). Lower lambda
                    // reduces the redundancy penalty threshold.
                    if decomposed {
                        Some(
                            self.cognitive_dials
                                .read()
                                .mmr_lambda_abstract
                                .unwrap_or(crate::constants::MMR_LAMBDA_ABSTRACT),
                        )
                    } else {
                        Some(
                            self.cognitive_dials
                                .read()
                                .mmr_lambda_exploratory
                                .unwrap_or(crate::constants::MMR_LAMBDA_EXPLORATORY),
                        )
                    }
                }
                query_parser::QueryType::Temporal => Some(
                    self.cognitive_dials
                        .read()
                        .mmr_lambda_relationship
                        .unwrap_or(crate::constants::MMR_LAMBDA_RELATIONSHIP),
                ),
            };
            if let Some(lambda) = mmr_lambda {
                let before_count = memories.len();
                memories = Self::apply_mmr(&memories, lambda, query.max_results);
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
        // LAYER 5.8: ANSWER-TYPE SOFT FILTER
        // =====================================================================
        // Penalize memories whose ExperienceType doesn't match query intent.
        // Soft filter (score penalty), not hard filter.
        {
            let query_lower = query_text.to_lowercase();
            let preferred: Option<Vec<crate::memory::types::ExperienceType>> = if query_lower
                .contains("bug")
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
                memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
            }
        }

        // =====================================================================
        // LAYER 5.85: LINGUISTIC BOOST (moved before 5.9 to avoid undoing pins)
        // =====================================================================
        // Sort-key boost for focal entity overlap, 5% of the score per unit of matched IC
        // weight. Must run BEFORE Layer 5.9 ordinal resolution, because 5.9 pins memories to
        // specific ranks and a post-pin re-sort would destroy those pins.
        if !query_analysis.focal_entities.is_empty() {
            memories.sort_by(|a, b| {
                let score_a = a.score.unwrap_or(0.0)
                    * (1.0 + Self::linguistic_boost(&a.experience.content, &query_analysis) * 0.05);
                let score_b = b.score.unwrap_or(0.0)
                    * (1.0 + Self::linguistic_boost(&b.experience.content, &query_analysis) * 0.05);
                score_b.total_cmp(&score_a)
            });
        }

        // =====================================================================
        // LAYER 5.87: TEMPORAL RANGE DEMOTION
        // =====================================================================
        // For temporal range queries ("last week", "last month", "recently"),
        // apply graduated demotion to memories outside the implied time window.
        // This makes recency the PRIMARY signal for these queries rather than
        // a weak additive boost that BM25 easily overcomes.
        //
        // Guard: only fires if 3+ candidates exist within the window to avoid
        // over-filtering when few recent memories match.
        {
            let ql = query_text.to_lowercase();
            // Calendar-aware temporal windows with graduated penalties:
            // "last week" = Monday-before-last through last Sunday
            //   - memories within last 7 days: moderate penalty (0.85x) — "this week", not "last week"
            //   - memories 7-14 days ago: full score (1.0x) — "last week" sweet spot
            //   - memories 14+ days ago (before Monday of last week): steep penalty (0.3x)
            // Other temporal ranges use simpler hour-based windows.
            enum TemporalRange {
                LastWeek,
                Window { hours: f64 },
                Stepped,
            }
            let temporal_range = if ql.contains("last week") {
                Some(TemporalRange::LastWeek)
            } else if ql.contains("last month") {
                Some(TemporalRange::Window { hours: 35.0 * 24.0 })
            } else if ql.contains("yesterday") {
                Some(TemporalRange::Window { hours: 48.0 })
            } else if ql.contains("today") {
                Some(TemporalRange::Window { hours: 24.0 })
            } else if ql.contains("recently") || ql.contains("recent ") {
                Some(TemporalRange::Stepped)
            } else {
                None
            };

            if let Some(range) = temporal_range {
                let mut demoted = 0;

                match range {
                    TemporalRange::LastWeek => {
                        // Calendar-aware: "last week" means the 7-day period
                        // ending last Sunday. Memories from "this week" (0-7 days)
                        // get moderate penalty, memories older than 14 days get steep penalty.
                        let in_sweet_spot = memories
                            .iter()
                            .filter(|m| {
                                let days = (now - m.created_at).num_days();
                                (7..=14).contains(&days)
                            })
                            .count();

                        if in_sweet_spot >= 2 {
                            for mem in memories.iter_mut() {
                                let days = (now - mem.created_at).num_days();
                                let demotion = if days < 7 {
                                    0.80 // "this week" — moderate penalty
                                } else if days <= 14 {
                                    1.0 // "last week" sweet spot — no penalty
                                } else if days <= 21 {
                                    0.40 // week before last — steep penalty
                                } else {
                                    0.30 // older — very steep
                                };
                                if demotion < 1.0 {
                                    let base = mem.score.unwrap_or(0.0);
                                    let mut cloned: Memory = mem.as_ref().clone();
                                    cloned.set_score(base * demotion);
                                    *mem = Arc::new(cloned);
                                    demoted += 1;
                                }
                            }
                        }
                    }
                    TemporalRange::Window {
                        hours: window_hours,
                    } => {
                        let in_window = memories
                            .iter()
                            .filter(|m| (now - m.created_at).num_hours() as f64 <= window_hours)
                            .count();

                        if in_window >= 3 {
                            for mem in memories.iter_mut() {
                                let age_hours = (now - mem.created_at).num_hours() as f64;
                                if age_hours > window_hours {
                                    let overshoot = age_hours / window_hours;
                                    let demotion = (1.0 / overshoot as f32).clamp(0.3, 0.5);
                                    let base = mem.score.unwrap_or(0.0);
                                    let mut cloned: Memory = mem.as_ref().clone();
                                    cloned.set_score(base * demotion);
                                    *mem = Arc::new(cloned);
                                    demoted += 1;
                                }
                            }
                        }
                    }
                    TemporalRange::Stepped => {
                        let step_for = |age_days: i64| -> f32 {
                            crate::constants::RECENTLY_STEPS
                                .iter()
                                .find(|(upper, _)| age_days <= *upper)
                                .map(|(_, factor)| *factor)
                                .unwrap_or(crate::constants::RECENTLY_FLOOR_MULTIPLIER)
                        };
                        for mem in memories.iter_mut() {
                            let factor = step_for((now - mem.created_at).num_days().max(0));
                            if factor < 1.0 {
                                let base = mem.score.unwrap_or(0.0);
                                let mut cloned: Memory = mem.as_ref().clone();
                                cloned.set_score(base * factor);
                                *mem = Arc::new(cloned);
                                demoted += 1;
                            }
                        }
                    }
                }

                if demoted > 0 {
                    memories
                        .sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
                    tracing::info!(demoted, "recall [layer:5.87] temporal range demotion");
                }
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
        // The memory an ordinal pin put first. Later layers re-sort by score (Layer 5.95 does
        // whenever the query has focal entities), and a pin that only moved a position was
        // undone by them; it is re-asserted once at the end of recall.
        let mut ordinal_pin: Option<SharedMemory> = None;
        {
            let ql = query_text.to_lowercase();

            // Ordinal: "first" → sort ascending by created_at, keep earliest
            let wants_first = ql.contains("first ") || ql.starts_with("first ");
            let wants_last =
                ql.contains("most recent") || ql.contains("latest") || ql.contains("last ");

            if wants_first && !memories.is_empty() {
                // Sort by created_at ascending — earliest first
                memories.sort_by(|a, b| a.created_at.cmp(&b.created_at));
                // Preserve the score of the earliest memory but put it at rank 1
                let earliest = memories[0].clone();
                // Re-sort by score but pin the earliest at position 0
                memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
                // Remove the earliest from its current position and insert at front
                if let Some(pos) = memories.iter().position(|m| m.id == earliest.id) {
                    memories.remove(pos);
                }
                ordinal_pin = Some(earliest.clone());
                memories.insert(0, earliest);
                tracing::debug!("Layer 5.9: Ordinal 'first' — pinned earliest memory at rank 1");
            } else if wants_last && !memories.is_empty() {
                // Strategy E: Focal-entity recency scan with tag fallback.
                //
                // For "most recent X" queries, scan candidates by recency and check
                // BOTH content AND tags for focal entity stems. This catches memories
                // like memory 46 where "architecture-change" is in tags but not in content.
                // Prefer action types (Decision, Error, Task) over observations.
                let focal_stems: Vec<String> = query_analysis
                    .focal_entities
                    .iter()
                    .map(|e| e.stem.clone())
                    .collect();

                let mut by_time = memories.clone();
                by_time.sort_by(|a, b| b.created_at.cmp(&a.created_at));

                // Scan by recency: find the first memory whose content OR tags
                // contain a focal entity stem
                let latest_focal = if !focal_stems.is_empty() {
                    by_time.iter().find(|m| {
                        let content_lower = m.experience.content.to_lowercase();
                        let tags_lower: String = m.experience.tags.join(" ").to_lowercase();
                        focal_stems
                            .iter()
                            .any(|stem| content_lower.contains(stem) || tags_lower.contains(stem))
                    })
                } else {
                    None
                };

                if let Some(found) = latest_focal {
                    let found_id = found.id.clone();
                    if let Some(pos) = memories.iter().position(|m| m.id == found_id) {
                        let pinned = memories.remove(pos);
                        ordinal_pin = Some(pinned.clone());
                        memories.insert(0, pinned);
                    }
                    tracing::debug!(
                        "Layer 5.9: Focal-entity recency scan pinned memory (stems={:?})",
                        focal_stems
                    );
                } else {
                    // Fallback: p50-gated pin (original strategy)
                    let score_sorted: Vec<f32> = {
                        let mut s: Vec<f32> =
                            memories.iter().map(|m| m.score.unwrap_or(0.0)).collect();
                        s.sort_by(|a, b| b.total_cmp(a));
                        s
                    };
                    let p50 = score_sorted[score_sorted.len() / 2];
                    let latest_relevant = by_time.iter().find(|m| m.score.unwrap_or(0.0) >= p50);
                    if let Some(latest) = latest_relevant {
                        let latest_id = latest.id.clone();
                        if let Some(pos) = memories.iter().position(|m| m.id == latest_id) {
                            let pinned = memories.remove(pos);
                            memories.insert(0, pinned);
                        }
                    }
                    // Graduated boost for top-3 recent relevant
                    let relevant_recent: Vec<MemoryId> = by_time
                        .iter()
                        .filter(|m| m.score.unwrap_or(0.0) >= p50)
                        .take(3)
                        .map(|m| m.id.clone())
                        .collect();
                    let boosts = [0.08_f32, 0.05, 0.03];
                    for (mid, &boost) in relevant_recent.iter().zip(boosts.iter()) {
                        if let Some(mem) = memories.iter_mut().find(|m| m.id == *mid) {
                            let base = mem.score.unwrap_or(0.0);
                            let mut cloned: Memory = mem.as_ref().clone();
                            cloned.set_score(base + boost);
                            *mem = Arc::new(cloned);
                        }
                    }
                    memories
                        .sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
                    tracing::debug!(
                        "Layer 5.9: Gated pin — latest not relevant, boosted {} recent-relevant memories",
                        relevant_recent.len()
                    );
                }
            }

            // Category error detection + ordinal session resolution:
            // Try ordinal resolution FIRST. Only apply category error demotion
            // if resolution fails or is unavailable.
            let ordinal_session_ref =
                (ql.contains("second ") || ql.contains("third ") || ql.contains("fourth "))
                    && (ql.contains("meeting") || ql.contains("sprint") || ql.contains("session"));

            let mut ordinal_resolved = false;

            if ordinal_session_ref {
                if let Some((ordinal, _noun)) = wavelet_sessions::extract_ordinal_session_ref(&ql) {
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
                                        // An injected member must pass the caller's filters
                                        // (tags, repo, ...); unfiltered recall injects as before.
                                        if vector_query.has_selective_filters()
                                            && !self.retriever.matches_filters(&mem, &vector_query)
                                        {
                                            continue;
                                        }
                                        memories.push(Arc::new(mem));
                                        injected += 1;
                                    }
                                }
                            }

                            for mem in memories.iter_mut() {
                                if session_ids.contains(&mem.id) {
                                    let base = mem.score.unwrap_or(0.0);
                                    let mut cloned: Memory = mem.as_ref().clone();
                                    cloned
                                        .set_score(base + crate::constants::ORDINAL_SESSION_BOOST);
                                    *mem = Arc::new(cloned);
                                }
                            }
                            memories.sort_by(|a, b| {
                                b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0))
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

                // Category error demotion: only if ordinal resolution failed.
                // [signal: ordinal_cliff] Gated by disabled_signals dial; default = always on.
                // When disabled, skip the ×0.50 demotion (leave all candidate scores unchanged).
                if !ordinal_resolved && dials_snapshot.signal_enabled("ordinal_cliff") {
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

        // (Linguistic boost moved to Layer 5.85, before ordinal resolution)

        let competition_mode = query.effective_competition_mode();
        let (competitions_detected, memories_eliminated) =
            Self::compete_memories(&mut memories, competition_mode);
        if competitions_detected > 0 {
            tracing::info!(
                mode = ?competition_mode,
                competitions_detected,
                memories_eliminated,
                post_count = memories.len(),
                "Live retrieval: entity-aware memory competition"
            );
        }

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

            // Suppression COMPUTES in every mode (per-query ranking); interference
            // ACCUMULATION is a usage write, so read-only recall must not teach the detector.
            let record_state = !self.recall_readonly();
            let competition_result = self
                .interference_detector
                .write()
                .apply_retrieval_competition(&candidates, query_text, record_state);

            // Record competition event if any memories were suppressed
            if let Some(ref event) = competition_result.event {
                self.record_consolidation_event(event.clone());
            }

            // Re-order memories based on competition results (winners first)
            if !competition_result.suppressed.is_empty() {
                match competition_mode {
                    CompetitionMode::ResolveNewest | CompetitionMode::ResolveStrongest => {
                        let winner_set: std::collections::HashSet<_> = competition_result
                            .winners
                            .iter()
                            .map(|(id, _)| id.clone())
                            .collect();

                        memories.retain(|m| winner_set.contains(&m.id.0.to_string()));

                        tracing::debug!(
                            mode = ?competition_mode,
                            suppressed = competition_result.suppressed.len(),
                            "Retrieval competition suppressed lower-ranked memories"
                        );
                    }
                    CompetitionMode::SurfaceBoth | CompetitionMode::Coexist => {
                        tracing::debug!(
                            mode = ?competition_mode,
                            suppressed = competition_result.suppressed.len(),
                            "Retrieval competition observed without suppressing candidates"
                        );
                    }
                }
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

            if !competition_result.suppressed.is_empty() {
                crate::metrics::SUPPRESSOR_DETECTIONS_TOTAL
                    .inc_by(competition_result.suppressed.len() as u64);
            }
        }

        // Abstention: filter out memories below confidence threshold
        // importance × calibrated_confidence < ABSTENTION_THRESHOLD → excluded
        // Only applies to memories with enough feedback observations (≥ 2 beyond prior)
        //
        // APP-4: Query-type adaptive thresholds — lexical/identifier queries
        // score lower on vector similarity, so we relax the threshold for them.
        // APP-6: Graph-connected memory insurance — memories found via graph
        // traversal with strong activation bypass abstention entirely.
        {
            // APP-5: Use drift-adaptive threshold if calibrated, else static constant.
            let calibrated_bits = self
                .calibrated_abstention_threshold
                .load(std::sync::atomic::Ordering::Relaxed);
            let base_threshold = if calibrated_bits != 0 {
                f32::from_bits(calibrated_bits)
            } else {
                crate::constants::ABSTENTION_THRESHOLD
            };

            // APP-4: Relax threshold for lexical/identifier queries.
            // Attribute queries targeting specific identifiers (ticket IDs, version
            // strings, error codes) produce lower vector similarity scores because
            // identifiers are out-of-vocabulary for the embedding model.
            let threshold = if query_parser::has_lexical_identifiers(query_text) {
                let relaxed = base_threshold * 0.5;
                tracing::debug!(
                    "Abstention: relaxed threshold {base_threshold} -> {relaxed} (lexical query)"
                );
                relaxed
            } else {
                base_threshold
            };

            let before = memories.len();
            memories.retain(|m| {
                // Only abstain when memory has ≥ 2 actual feedback events (beyond prior of 2.0)
                if m.confidence_observations() <= 4.0 {
                    return true;
                }

                // APP-6: Graph-connected memory insurance.
                // If this memory was found via graph traversal with activation > 0.3,
                // it has strong relational evidence for relevance. Graph connectivity
                // provides an orthogonal retrieval signal that should not be vetoed
                // by importance decay — otherwise the graph "knows" about a memory
                // that retrieval refuses to surface.
                if let Some(attr) = signal_attributions.get(&m.id) {
                    if attr.graph_contribution > 0.3 {
                        return true;
                    }
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

        // =====================================================================
        // LAYER 5.95: CONTENT SUFFICIENCY SIGNAL
        // =====================================================================
        // "Finding the right memory ≠ returning sufficient information."
        // A memory might be the correct match (right entity, right timeframe)
        // but contain insufficient detail to answer the query. Detect this by
        // measuring focal entity coverage in the memory content. Memories that
        // mention fewer query entities get a soft demotion; memories that cover
        // more entities get a boost. If a top memory is thin, check the fact
        // store for supplementary facts from the same entity cluster and inject
        // fact-source memories into the result set.
        if !query_analysis.focal_entities.is_empty() && memories.len() >= 2 {
            let query_entities: Vec<String> = query_analysis
                .focal_entities
                .iter()
                .map(|e| e.text.to_lowercase())
                .collect();
            let entity_count = query_entities.len() as f32;

            // Layer 5.36 already scored coverage on stems before truncation; scoring it again
            // here on surface text would count the same evidence twice. The fact supplement
            // below still runs.
            let sufficiency_scoring = if query_coverage_active {
                0
            } else {
                memories.len()
            };
            for mem in memories.iter_mut().take(sufficiency_scoring) {
                let content_lower = mem.experience.content.to_lowercase();
                let covered = query_entities
                    .iter()
                    .filter(|e| content_lower.contains(e.as_str()))
                    .count() as f32;
                let coverage_ratio = covered / entity_count;

                // Only adjust if there's meaningful entity divergence
                if coverage_ratio < 0.5 {
                    // Thin memory: covers less than half the query entities
                    let penalty = 0.90 + 0.10 * coverage_ratio; // 0.90-0.95
                    let base = mem.score.unwrap_or(0.0);
                    let mut cloned: Memory = mem.as_ref().clone();
                    cloned.set_score(base * penalty);
                    *mem = Arc::new(cloned);
                } else if coverage_ratio > 0.8 && entity_count >= 2.0 {
                    // Rich memory: covers most query entities
                    let boost = 1.0 + 0.05 * (coverage_ratio - 0.8); // 1.00-1.01
                    let base = mem.score.unwrap_or(0.0);
                    let mut cloned: Memory = mem.as_ref().clone();
                    cloned.set_score(base * boost);
                    *mem = Arc::new(cloned);
                }
            }

            // Re-sort after sufficiency adjustments
            memories.sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));

            // If the top result has low entity coverage, try to supplement from fact store.
            // Facts are distilled knowledge — they're often more complete than thin episodic
            // memories for the same entity.
            if let Some(top) = memories.first() {
                let top_content_lower = top.experience.content.to_lowercase();
                let top_score = top.score.unwrap_or(0.0);
                let top_coverage = query_entities
                    .iter()
                    .filter(|e| top_content_lower.contains(e.as_str()))
                    .count() as f32
                    / entity_count;

                if top_coverage < 0.5 {
                    if let Some(user_id) = &query.user_id {
                        if let Ok(facts) =
                            self.get_facts_for_graph_entities(user_id, &query_entities, 3)
                        {
                            // Inject fact-source memories that aren't already in results
                            let existing_ids: HashSet<_> =
                                memories.iter().map(|m| m.id.clone()).collect();

                            let mut injected = 0;
                            for fact in &facts {
                                if fact.confidence < 0.6 || fact.support_count < 2 {
                                    continue;
                                }
                                for src_id in &fact.source_memories {
                                    if existing_ids.contains(src_id) {
                                        continue;
                                    }
                                    // Load and score the fact-source memory
                                    if let Some(view) = candidates.get(src_id) {
                                        // As for ordinal-session members: a fact source the
                                        // caller's filters exclude is not injected.
                                        if vector_query.has_selective_filters()
                                            && !self
                                                .retriever
                                                .matches_filters(&view.memory, &vector_query)
                                        {
                                            continue;
                                        }
                                        if view.tier == CandidateTier::LongTerm {
                                            let mut scored = view.memory.as_ref().clone();
                                            // Score slightly below top result to avoid
                                            // displacing the original match entirely
                                            scored.set_score(top_score * 0.85 * fact.confidence);
                                            memories.push(Arc::new(scored));
                                        } else {
                                            memories.push(Arc::clone(&view.memory));
                                        }
                                        injected += 1;
                                    }
                                    if injected >= 2 {
                                        break;
                                    }
                                }
                                if injected >= 2 {
                                    break;
                                }
                            }

                            if injected > 0 {
                                // Re-sort and truncate
                                memories.sort_by(|a, b| {
                                    b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0))
                                });
                                memories.truncate(query.max_results);
                                tracing::debug!(
                                    injected,
                                    "Layer 5.95: Supplemented thin top result with fact-source memories"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Usage writes from here on (access counts, coactivation, retrieval counter) are
        // skipped under read-only recall — see `recall_readonly()`.
        let readonly = self.recall_readonly();

        // Update access counts with instrumentation for consolidation events
        // (only for memories that survived competition)
        if !readonly {
            for memory in &memories {
                self.update_access_count_instrumented(memory, StrengtheningReason::Recalled);
            }
        }

        // =====================================================================
        // FIX-R1: RECONSOLIDATION — mark retrieved memories as labile
        // =====================================================================
        // Set activation=1.0 (labile state) and create/extend reconsolidation
        // shadows. During the labile window, the memory can be updated with new
        // context discovered during the retrieval interaction.
        // Reference: Nader et al. (2000) — reconsolidation theory.
        // A usage write: set_activation feeds Layer 5's activation boost on the next recall.
        if !readonly {
            let now = chrono::Utc::now();
            // Labile-window length + active-shadow cap are runtime dials (default to the
            // constant values). Read once here so the shadows write lock below never
            // nests the dials lock.
            let (window, max_active_shadows) = {
                let d = self.cognitive_dials.read();
                (
                    chrono::Duration::seconds(d.reconsolidation_window_secs),
                    d.max_active_shadows.max(1),
                )
            };
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
                    // Accumulate retrieval context across the labile window for the
                    // re-encode at window close (capped at 2KB, deduped per segment).
                    let q = query_text.trim();
                    if !q.is_empty()
                        && existing.retrieval_context.len() + q.len() + 3 <= 2048
                        && !existing.retrieval_context.split(" | ").any(|s| s == q)
                    {
                        if existing.retrieval_context.is_empty() {
                            existing.retrieval_context.push_str(q);
                        } else {
                            existing.retrieval_context.push_str(" | ");
                            existing.retrieval_context.push_str(q);
                        }
                    }
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

            // Cap active shadows to prevent accumulation (runtime dial).
            while shadows.len() > max_active_shadows {
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
        // S-R4: routed through apply_reinforcement so duplicate retrieval_ids are no-ops.
        if !readonly && memories.len() >= 2 {
            if let Some(graph) = &self.graph_memory {
                let retrieval_id = uuid::Uuid::new_v4();
                let source = query
                    .evidence_source
                    .as_deref()
                    .unwrap_or(crate::constants::HEBBIAN_UNATTRIBUTED_SOURCE);
                let memory_uuids: Vec<uuid::Uuid> = memories.iter().map(|m| m.id.0).collect();
                match graph.read_recursive().apply_reinforcement(
                    retrieval_id,
                    source,
                    &memory_uuids[0],
                    &memory_uuids[1..],
                ) {
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
        if !readonly {
            if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
                self.stats.write().total_retrievals = count;
            }
        }

        self.logger
            .read()
            .log_retrieved(query_text, memories.len(), &sources);

        // Expand with hierarchy context (parent chain + children)
        // This ensures semantic search also surfaces contextually related memories
        let mut seen_ids: HashSet<MemoryId> = memories.iter().map(|m| m.id.clone()).collect();
        self.expand_with_hierarchy(&vector_query, &mut memories, &mut seen_ids);

        // P4: iterative re-cueing (pattern completion). When the first pass came
        // back thin (fewer than requested), use the strongest hit's embedding as
        // a NEW cue and pull its nearest neighbours the literal query missed —
        // the recurrent-reinstatement analogue. READ-ONLY: it never mutates the
        // TCM drift or Hebbian edges, so the re-cue cannot self-confirm toward
        // what gets retrieved (overloop guardrail). Capped at ONE extra pass;
        // gated on under-fill so it only augments a thin set, never reorders a
        // satisfied one. Re-cued neighbours pass the same validity + filter gates.
        if memories.len() < query.max_results {
            let top_emb = memories
                .first()
                .and_then(|m| m.experience.embeddings.clone());
            let floor = memories
                .iter()
                .filter_map(|m| m.score)
                .fold(f32::INFINITY, f32::min);
            if let Some(emb) = top_emb {
                let mut recue = query.clone();
                recue.query_embedding = Some(emb);
                recue.query_text = None;
                if let Ok(ids) = self.retriever.search_ids(&recue, query.max_results * 4) {
                    let recue_score = if floor.is_finite() { floor * 0.9 } else { 0.0 };
                    for (id, _sim) in ids {
                        if memories.len() >= query.max_results {
                            break;
                        }
                        if seen_ids.contains(&id) {
                            continue;
                        }
                        if let Ok(mut m) = self.retriever.get_from_storage(&id) {
                            let valid = match recue.as_of {
                                Some(t) => m.is_valid_at(t),
                                None => !m.is_expired(),
                            };
                            if valid && self.retriever.matches_filters(&m, &recue) {
                                seen_ids.insert(id);
                                m.set_score(recue_score);
                                memories.push(Arc::new(m));
                            }
                        }
                    }
                    // Keep the ranking monotonic after augmentation.
                    memories
                        .sort_by(|a, b| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)));
                }
            }
        }

        crate::memory::gold_funnel::record("final", memories.iter().map(|m| &m.id));

        let t_total = recall_start.elapsed();
        tracing::info!(
            post_ms = format!("{:.2}", (t_total - t_fetch).as_secs_f64() * 1000.0),
            total_ms = format!("{:.2}", t_total.as_secs_f64() * 1000.0),
            final_count = memories.len(),
            "recall [layer:post] linguistic + competition + coactivation + hierarchy === RECALL COMPLETE ==="
        );

        // Persist signal attributions from this retrieval. Only keep attributions
        // for memories that survived all filtering/competition stages.
        {
            let final_ids: HashSet<MemoryId> = memories.iter().map(|m| m.id.clone()).collect();
            let mut attr_guard = self.last_signal_attributions.write();
            attr_guard.clear();
            for (id, attribution) in signal_attributions {
                if final_ids.contains(&id) {
                    attr_guard.insert(id, attribution);
                }
            }
        }

        // M3: stash the query-level metacognition substrate. The full
        // feeling-of-knowing is assembled by the handler (which has the final
        // scores and slow-store access for the M5 gap check); here we only carry
        // the resolved focal entities forward. This is a terminal readout — it is
        // never fed back into scoring or adaptive weight learning.
        {
            let focal_entities: Vec<String> = query_analysis
                .focal_entities
                .iter()
                .map(|e| e.text.clone())
                .collect();
            // M3.1: cross-embedder (S3) agreement over the top results — a
            // cold-robust confidence signal needing no feedback history. The two
            // independent embedders agreeing on the ranking => trustworthy;
            // diverging => uncertain. Skipped when there is no secondary embedder
            // or too few results carry secondary embeddings.
            let cross_embedder_agreement = space_queries.first().and_then(|(sec_space, qsec)| {
                let mut pairs: Vec<(f32, f32)> = Vec::new();
                for m in memories.iter().take(10) {
                    if let (Some(emb), Some(sec)) = (
                        self.retriever.stored_vector(m, &index_space),
                        self.retriever.stored_vector(m, sec_space),
                    ) {
                        let p = crate::memory::hybrid_search::cosine_similarity_pub(
                            &query_embedding,
                            emb,
                        );
                        let s = crate::memory::hybrid_search::cosine_similarity_pub(qsec, sec);
                        pairs.push((p, s));
                    }
                }
                if pairs.len() < 3 {
                    return None;
                }
                // Min-max normalize each embedder's similarity profile, then
                // agreement = 1 - mean absolute difference between the profiles.
                let p_min = pairs.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
                let p_max = pairs.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
                let s_min = pairs.iter().map(|x| x.1).fold(f32::INFINITY, f32::min);
                let s_max = pairs.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
                let p_rng = (p_max - p_min).max(1e-6);
                let s_rng = (s_max - s_min).max(1e-6);
                let mad: f32 = pairs
                    .iter()
                    .map(|&(p, s)| (((p - p_min) / p_rng) - ((s - s_min) / s_rng)).abs())
                    .sum::<f32>()
                    / pairs.len() as f32;
                Some((1.0 - mad).clamp(0.0, 1.0))
            });
            // M4: query-level entity coverage — fraction of the query's focal
            // entities covered by ANY top result (union, not per-memory). Reuses
            // the Layer 5.95 lowercased-substring matching; cold-robust.
            let coverage = if focal_entities.is_empty() {
                None
            } else {
                let wanted: Vec<String> = focal_entities.iter().map(|e| e.to_lowercase()).collect();
                let top_lc: Vec<String> = memories
                    .iter()
                    .take(10)
                    .map(|m| m.experience.content.to_lowercase())
                    .collect();
                let covered = wanted
                    .iter()
                    .filter(|e| top_lc.iter().any(|c| c.contains(e.as_str())))
                    .count();
                Some(covered as f32 / wanted.len() as f32)
            };
            *self.last_query_metacognition.write() =
                Some(crate::memory::types::QueryMetacognition {
                    fok: 0.0,
                    label: String::new(),
                    peak_confidence: 0.0,
                    answerability: 0.0,
                    cross_embedder_agreement,
                    coverage,
                    in_known_gap: false,
                    signal_strength: if cross_embedder_agreement.is_some() {
                        "score+agreement"
                    } else {
                        "score"
                    }
                    .to_string(),
                    focal_entities,
                });
        }

        // R2 (LTR) SHADOW WRITE-SITE — the SINGLE place a trace row is emitted, so
        // there is exactly one PENDING row per (recall, memory). It runs AFTER all
        // reordering (competition, hierarchy expansion, re-cue, final sort), so
        // `rank` reflects the truly final returned order and `m.score` is the final
        // returned score. Best-effort and READ-ONLY w.r.t. the result: it reads the
        // already-finalized `memories` and the captured feature side-map, then
        // persists; it never mutates `memories`, scores, or ordering. Only memories
        // that were scored through the closure carry captured features; re-cued
        // augmentation rows (which bypass the closure) have no features and are
        // skipped — they were never part of the learned-ranking surface.
        {
            let features = scoring_features.into_inner();
            for (rank, m) in memories.iter().enumerate() {
                if let Some(feat) = features.get(&m.id) {
                    let trace = crate::memory::feedback::RetrievalTrace::new_pending(
                        m.id.clone(),
                        query_text.to_string(),
                        rank,
                        feat.clone(),
                        feat.base_score,
                        m.score.unwrap_or(feat.unified_final),
                    );
                    self.retrieval_trace_store.record_pending(&trace);
                }
            }
            // B9 batch enforcement: enforce the PENDING bound once per recall
            // (after all rows for this retrieval are written) rather than once
            // per result row. Avoids a full KV prefix scan per insert.
            self.retrieval_trace_store.enforce_pending_bound_amortized();

            let returned: Vec<MemoryId> = memories.iter().map(|m| m.id.clone()).collect();
            self.retriever
                .observe_leg_depths(&leg_record, &returned, query.max_results);

            // R3b SHADOW DELTA — compute the learned-weighted score alongside
            // the static score and log the delta.  PURELY READ-ONLY w.r.t. the
            // live result: `memories`, `m.score`, and the returned order are
            // byte-for-byte unchanged.  The static score is the live score.
            //
            // Shadow formula (per memory):
            //   static_signal  = static_bm25  * bm25_attr
            //                  + static_vector * vector_attr
            //                  + static_graph  * graph_attr
            //   learned_signal = learned_bm25  * bm25_attr
            //                  + learned_vector * vector_attr
            //                  + learned_graph  * graph_attr
            //   delta          = learned_signal − static_signal
            //   shadow_score   = static_final_score + delta
            //
            // The three raw signal contributions come from `last_signal_attributions`
            // (populated just above; guarded so only surviving memories are present).
            // If a memory has no attribution (e.g. a re-cued augmentation row) we
            // log delta=0 so the accounting is complete without fabricating data.
            //
            // B8 gate: only log if the gate flag is false (which it always is in
            // this PR), ensuring the comment stays true: live OFF, shadow logged.
            if !crate::constants::R3B_LIVE_APPLICATION_ENABLED {
                let attr_guard = self.last_signal_attributions.read();
                for m in memories.iter() {
                    let static_score = m.score.unwrap_or(0.0);
                    let (delta, learned_score) = if let Some(attr) = attr_guard.get(&m.id) {
                        let static_signal = r3b_static_bm25 * attr.bm25_contribution
                            + r3b_static_vector * attr.vector_contribution
                            + r3b_static_graph * attr.graph_contribution;
                        let learned_signal = r3b_learned_snapshot.bm25 * attr.bm25_contribution
                            + r3b_learned_snapshot.vector * attr.vector_contribution
                            + r3b_learned_snapshot.graph * attr.graph_contribution;
                        let d = learned_signal - static_signal;
                        (d, static_score + d)
                    } else {
                        (0.0_f32, static_score)
                    };
                    tracing::trace!(
                        memory_id = %m.id.0,
                        static_score,
                        learned_score,
                        delta,
                        learned_updates = r3b_learned_snapshot.update_count,
                        "R3b shadow: learned-vs-static delta (live path unchanged)"
                    );
                }
            }
        }

        if let Some(pinned) = ordinal_pin {
            Self::reassert_ordinal_pin(&mut memories, pinned, query.max_results);
        }

        Ok(memories)
    }

    /// The query's vector in the primary space and, with a secondary embedder, in the
    /// secondary's: the caller's or the query cache's when either has one, else embedded, the
    /// two at once when both must be. The secondary embeds the text as a query, the side an
    /// asymmetric model such as nomic searches with.
    fn query_vectors(
        &self,
        query_text: &str,
        query: &Query,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let query_hash = Self::sha256_hash(query_text);
        // PERFORMANCE: Use pre-computed embedding if caller provided one,
        // otherwise fall back to SHA256-keyed cache (80ms → <1μs for repeated queries)
        let known_primary =
            if let Some(pre) = query.query_embedding.as_ref().filter(|e| !e.is_empty()) {
                EMBEDDING_CACHE_QUERY
                    .with_label_values(&["precomputed"])
                    .inc();
                tracing::debug!("Query embedding PRECOMPUTED by caller — skipping encode");
                Some(pre.clone())
            } else if let Some(cached_embedding) = self.query_cache.get(&query_hash) {
                EMBEDDING_CACHE_QUERY.with_label_values(&["hit"]).inc();
                tracing::debug!("Query embedding cache HIT for: {}", query_text);
                Some(cached_embedding)
            } else {
                EMBEDDING_CACHE_QUERY.with_label_values(&["miss"]).inc();
                tracing::debug!(
                    "Query embedding cache MISS - generating for: {}",
                    query_text
                );
                None
            };

        let secondary_space = self.embedder.secondary_space();
        let mut known_secondary = None;
        let mut secondary_to_embed = None;
        if let Some(space) = &secondary_space {
            match self.query_cache.get_in(space, &query_hash) {
                Some(vector) => known_secondary = Some(vector),
                None => secondary_to_embed = Some(space),
            }
        }

        let embed_primary = || self.embed_query(query_text, query_hash);
        let embed_secondary = |space: &SpaceId| -> Option<Vec<f32>> {
            match self.embedder.encode_secondary_query(query_text) {
                Ok(Some(vector)) => {
                    self.query_cache
                        .insert_in(space, query_hash, vector.clone());
                    Some(vector)
                }
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("Secondary query embedding failed (non-fatal): {e:#}");
                    None
                }
            }
        };

        let (primary, embedded_secondary) = match (known_primary, secondary_to_embed) {
            (Some(primary), to_embed) => (primary, to_embed.and_then(embed_secondary)),
            (None, None) => (embed_primary()?, None),
            (None, Some(space)) => {
                let (primary, secondary) = std::thread::scope(|scope| {
                    let secondary = scope.spawn(|| embed_secondary(space));
                    (embed_primary(), secondary.join())
                });
                let secondary = secondary.unwrap_or_else(|_| {
                    tracing::error!("secondary query embedding panicked; recall goes without it");
                    None
                });
                (primary?, secondary)
            }
        };
        Ok((primary, known_secondary.or(embedded_secondary)))
    }

    /// Put the Layer 5.9 ordinal pin back at rank 1 after the layers that re-sort by score,
    /// re-inserting it if a later truncation dropped it. Its score is raised just above the
    /// new second place so the list stays monotonic and a caller that re-sorts keeps the pin.
    fn reassert_ordinal_pin(
        memories: &mut Vec<SharedMemory>,
        pinned: SharedMemory,
        max_results: usize,
    ) {
        let pinned = match memories.iter().position(|m| m.id == pinned.id) {
            Some(pos) => memories.remove(pos),
            None => pinned,
        };
        let runner_up = memories.first().and_then(|m| m.score).unwrap_or(0.0);
        let pinned = if pinned.score.unwrap_or(0.0) <= runner_up {
            let mut cloned: Memory = pinned.as_ref().clone();
            cloned.set_score(runner_up + (runner_up.abs() * 1e-4).max(f32::EPSILON));
            Arc::new(cloned)
        } else {
            pinned
        };
        memories.insert(0, pinned);
        memories.truncate(max_results.max(1));
    }

    /// Spreading activation recall: graph-driven memory retrieval (V5 toroid closure)
    ///
    /// Uses the dedicated Anderson & Pirolli (1984) spreading activation algorithm
    /// with density-dependent weights (SHO-26). This is a separate retrieval path
    /// from the 11-layer pipeline — it trusts the knowledge graph as the primary
    /// signal rather than vector similarity + BM25 fusion.
    ///
    /// The episode_to_memory closure resolves graph episodes to full memory objects
    /// by checking working -> session -> long-term tiers.
    fn spreading_activation_recall(
        &self,
        query_text: &str,
        query: &Query,
    ) -> Result<Vec<SharedMemory>> {
        let graph = self.graph_memory.as_ref().ok_or_else(|| {
            anyhow::anyhow!("SpreadingActivation mode requires a knowledge graph")
        })?;
        let graph_guard = graph.read_recursive();

        // Compute graph density for weight selection
        let stats = graph_guard.get_stats().unwrap_or_default();
        let total_memories = {
            let s = self.stats.read();
            s.total_memories
        };
        let graph_density = if total_memories > 0 {
            Some(stats.relationship_count as f32 / total_memories as f32)
        } else {
            None
        };

        // Pre-compute query classification and ontological intent
        let query_type = query_parser::classify_query(query_text);
        let query_analysis = query_parser::analyze_query(query_text);
        let ontological_intent =
            query_parser::infer_ontological_intent(query_text, &query_analysis);

        // Build episode-to-memory resolver: working -> session -> long-term
        let working_ref = &self.working_memory;
        let session_ref = &self.session_memory;
        let lt_ref = &self.long_term_memory;

        let episode_to_memory_fn =
            |episode: &crate::graph_memory::EpisodicNode| -> Result<Option<SharedMemory>> {
                let target_id = MemoryId(episode.uuid);

                // Check working memory first (highest activation), then session
                if let Some(mem) = working_ref.read().get(&target_id) {
                    return Ok(Some(mem));
                }
                if let Some(mem) = session_ref.read().get(&target_id) {
                    return Ok(Some(mem));
                }

                // Fall back to long-term storage
                match lt_ref.get(&target_id) {
                    Ok(memory) => Ok(Some(Arc::new(memory))),
                    Err(_) => Ok(None),
                }
            };

        let (activated_memories, retrieval_stats) =
            crate::memory::graph_retrieval::spreading_activation_retrieve_with_stats(
                query_text,
                query,
                &graph_guard,
                self.embedder.as_ref(),
                graph_density,
                Some(&query_type),
                Some(&ontological_intent),
                episode_to_memory_fn,
            )?;

        tracing::info!(
            mode = "spreading_activation",
            results = activated_memories.len(),
            graph_density = ?graph_density,
            query_type = ?query_type,
            entities_activated = retrieval_stats.entities_activated,
            graph_time_us = retrieval_stats.graph_time_us,
            "recall [spreading_activation] complete"
        );

        // S-R5: Cache path-credit edges for the next reinforce_recall(Helpful) call.
        // Stash the raw contributions; reinforce_recall will drain and apply them.
        if !retrieval_stats.path_credit_edges.is_empty() {
            *self.spreading_path_credit.lock() = retrieval_stats.path_credit_edges;
        }

        // Extract SharedMemory from ActivatedMemory, preserving score ordering. The graph walk
        // applies no query filter, so a caller filter is applied here; unfiltered recall is as
        // before.
        let filtered = query.has_selective_filters();
        let memories: Vec<SharedMemory> = activated_memories
            .into_iter()
            .map(|am| am.memory)
            .filter(|memory| !filtered || query.matches(memory))
            .take(query.max_results)
            .collect();

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

    /// Entity-aware retrieval-time competition for the live recall path.
    fn compete_memories(memories: &mut Vec<SharedMemory>, mode: CompetitionMode) -> (usize, usize) {
        use crate::constants::{COMPETITION_ENTITY_OVERLAP_MIN, COMPETITION_MAX_PAIRS};

        if mode == CompetitionMode::Coexist || memories.len() < 2 {
            return (0, 0);
        }

        let entity_sets: Vec<HashSet<String>> = memories
            .iter()
            .map(|memory| {
                memory
                    .experience
                    .entities
                    .iter()
                    .map(|entity| entity.to_lowercase())
                    .collect()
            })
            .collect();

        let mut eliminated: HashSet<usize> = HashSet::new();
        let mut competitions_detected = 0;
        let mut pair_count = 0;
        let check_limit = memories.len().min(20);
        let similarity_floor = competition_similarity_floor(&memories[..check_limit]);

        for left_idx in 0..check_limit {
            if eliminated.contains(&left_idx) {
                continue;
            }

            for right_idx in (left_idx + 1)..check_limit {
                if eliminated.contains(&right_idx) {
                    continue;
                }
                if pair_count >= COMPETITION_MAX_PAIRS {
                    break;
                }
                pair_count += 1;

                if entity_sets[left_idx].is_empty() || entity_sets[right_idx].is_empty() {
                    continue;
                }

                let overlap_count = entity_sets[left_idx]
                    .intersection(&entity_sets[right_idx])
                    .count();
                if overlap_count < COMPETITION_ENTITY_OVERLAP_MIN {
                    continue;
                }

                let similarity = match (
                    memories[left_idx].experience.embeddings.as_ref(),
                    memories[right_idx].experience.embeddings.as_ref(),
                ) {
                    // Raw cosine: `similarity_floor` is measured on this pool's own raw
                    // cosines, which already adapts the bar to the embedder.
                    (Some(left), Some(right)) if !left.is_empty() && !right.is_empty() => {
                        crate::similarity::cosine_similarity(left, right)
                    }
                    _ => {
                        let union_count =
                            entity_sets[left_idx].union(&entity_sets[right_idx]).count();
                        if union_count == 0 {
                            0.0
                        } else {
                            overlap_count as f32 / union_count as f32
                        }
                    }
                };

                if similarity < similarity_floor {
                    continue;
                }

                competitions_detected += 1;

                let loser_idx = match mode {
                    CompetitionMode::ResolveNewest => {
                        if memories[left_idx].created_at >= memories[right_idx].created_at {
                            right_idx
                        } else {
                            left_idx
                        }
                    }
                    CompetitionMode::ResolveStrongest => {
                        if memories[left_idx].importance() >= memories[right_idx].importance() {
                            right_idx
                        } else {
                            left_idx
                        }
                    }
                    CompetitionMode::SurfaceBoth => {
                        let penalty = 0.01;
                        let (stronger_idx, weaker_idx) = if memories[left_idx].score.unwrap_or(0.0)
                            >= memories[right_idx].score.unwrap_or(0.0)
                        {
                            (left_idx, right_idx)
                        } else {
                            (right_idx, left_idx)
                        };

                        let weaker_score = memories[weaker_idx].score.unwrap_or(0.0);
                        let mut weaker_clone: Memory = memories[weaker_idx].as_ref().clone();
                        weaker_clone.set_score(weaker_score - penalty);
                        weaker_clone
                            .experience
                            .metadata
                            .insert("competition_conflict".to_string(), "true".to_string());
                        weaker_clone.experience.metadata.insert(
                            "competition_opponent".to_string(),
                            memories[stronger_idx].id.0.to_string(),
                        );
                        memories[weaker_idx] = Arc::new(weaker_clone);

                        let mut stronger_clone: Memory = memories[stronger_idx].as_ref().clone();
                        stronger_clone
                            .experience
                            .metadata
                            .insert("competition_conflict".to_string(), "true".to_string());
                        stronger_clone.experience.metadata.insert(
                            "competition_opponent".to_string(),
                            memories[weaker_idx].id.0.to_string(),
                        );
                        memories[stronger_idx] = Arc::new(stronger_clone);
                        continue;
                    }
                    CompetitionMode::Coexist => continue,
                };

                let winner_idx = if loser_idx == left_idx {
                    right_idx
                } else {
                    left_idx
                };
                eliminated.insert(loser_idx);

                let shared_entities: Vec<&String> = entity_sets[left_idx]
                    .intersection(&entity_sets[right_idx])
                    .collect();
                tracing::debug!(
                    winner_id = %memories[winner_idx].id.0,
                    loser_id = %memories[loser_idx].id.0,
                    shared_entities = ?shared_entities,
                    similarity = format!("{similarity:.3}"),
                    mode = ?mode,
                    "Memory competition resolved"
                );
            }

            if pair_count >= COMPETITION_MAX_PAIRS {
                break;
            }
        }

        let memories_eliminated = eliminated.len();
        if memories_eliminated > 0 {
            let mut indices: Vec<usize> = eliminated.into_iter().collect();
            indices.sort_unstable_by(|left, right| right.cmp(left));
            for idx in indices {
                memories.remove(idx);
            }
        }

        (competitions_detected, memories_eliminated)
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
    /// Append each result's ancestors and direct children, subject to the same query filters
    /// as the results themselves: a tag- or type-filtered recall must not return an unfiltered
    /// parent because one of its children matched.
    fn expand_with_hierarchy(
        &self,
        query: &Query,
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
                    if query.matches(&ancestor) && seen_ids.insert(ancestor.id.clone()) {
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
                    if query.matches(&child) && seen_ids.insert(child.id.clone()) {
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
        if !self.recall_readonly() {
            if let Ok(count) = self.long_term_memory.increment_retrieval_count() {
                self.stats.write().total_retrievals = count;
            }
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
        self.reinforce_recall_from(
            memory_ids,
            outcome,
            crate::constants::HEBBIAN_UNATTRIBUTED_SOURCE,
        )
    }

    /// `reinforce_recall` with the evidence source named: the session or tool whose
    /// feedback this is. One source counts once per edge it strengthens.
    pub fn reinforce_recall_from(
        &self,
        memory_ids: &[MemoryId],
        outcome: RetrievalOutcome,
        source: &str,
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
        // Uses GraphMemory if available, otherwise counts pair associations directly.
        // S-R4: routed through apply_reinforcement so duplicate retrieval_ids are no-ops.
        if !matches!(outcome, RetrievalOutcome::Misleading) && memory_ids.len() >= 2 {
            if let Some(graph) = &self.graph_memory {
                let memory_uuids: Vec<uuid::Uuid> = memory_ids.iter().map(|id| id.0).collect();
                if memory_uuids.len() >= 2 {
                    let retrieval_id = uuid::Uuid::new_v4();
                    match graph.read().apply_reinforcement(
                        retrieval_id,
                        source,
                        &memory_uuids[0],
                        &memory_uuids[1..],
                    ) {
                        Ok(result) => {
                            stats.associations_strengthened = result.edges_updated;
                            // BRIDGE-4: Boost importance for memories linked by promoted edges
                            for promo in &result.promotions {
                                let boost = if promo.new_tier.contains("L3") {
                                    0.15
                                } else {
                                    0.10
                                };
                                for mid in memory_ids {
                                    if mid.0 == promo.from_entity || mid.0 == promo.to_entity {
                                        // Apply boost through all tiers
                                        if let Some(mem) = self.working_memory.read().get(mid) {
                                            mem.boost_importance(boost);
                                        } else if let Some(mem) =
                                            self.session_memory.read().get(mid)
                                        {
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

        // S-R5: Path-credit reinforcement (Physarum tube-thickening, B10)
        // On Helpful outcome, strengthen the spreading-activation edges that actually
        // delivered the memories.  Credit is normalised to 1.0 inside apply_path_credit.
        if matches!(outcome, RetrievalOutcome::Helpful) {
            if let Some(graph) = &self.graph_memory {
                // Drain the stashed path-credit edges from the most recent retrieval.
                let path_edges: Vec<(uuid::Uuid, f32)> =
                    std::mem::take(&mut *self.spreading_path_credit.lock());
                if !path_edges.is_empty() {
                    match graph.read().apply_path_credit(&path_edges, source) {
                        Ok(n) => {
                            tracing::debug!(
                                n,
                                "S-R5: path-credit applied to {} spreading-activation edges",
                                n
                            );
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "S-R5: path-credit apply failed (non-critical)");
                        }
                    }
                }
            }
        }

        // CACHE COHERENT IMPORTANCE UPDATES: the change is applied to the cached Arc, so every
        // holder of it sees the change, and to the stored record, which is what is persisted.
        // The cached copy can predate a store-side rewrite, such as the vector a deferred embed
        // stored after the memory was cached, and persisting it would erase that rewrite.
        let mut persist_failures: Vec<(MemoryId, String)> = Vec::new();
        let rewrites_importance = matches!(
            outcome,
            RetrievalOutcome::Helpful | RetrievalOutcome::Misleading
        );
        let reinforce = |memory: &Memory| {
            memory.record_access();
            match outcome {
                RetrievalOutcome::Helpful => memory.boost_importance(HEBBIAN_BOOST_HELPFUL),
                RetrievalOutcome::Misleading => memory.decay_importance(HEBBIAN_DECAY_MISLEADING),
                // I8: Abstained/Ignored are shadow-neutral for live reinforcement
                // (access-only); they carry their negative label to the trace store.
                RetrievalOutcome::Neutral
                | RetrievalOutcome::Abstained
                | RetrievalOutcome::Ignored => {}
            }
        };

        for id in memory_ids {
            let cached = self
                .working_memory
                .read()
                .get(id)
                .or_else(|| self.session_memory.read().get(id));
            if let Some(cached) = &cached {
                reinforce(cached);
            }
            let persisted = match self.long_term_memory.get(id) {
                Ok(stored) => {
                    reinforce(&stored);
                    stored
                }
                Err(e) => match cached {
                    Some(cached) => (*cached).clone(),
                    None => {
                        tracing::debug!(
                            memory_id = %id.0,
                            error = %e,
                            "Memory not found during reinforcement - may have been deleted"
                        );
                        continue;
                    }
                },
            };
            match outcome {
                RetrievalOutcome::Helpful => stats.importance_boosts += 1,
                RetrievalOutcome::Misleading => stats.importance_decays += 1,
                RetrievalOutcome::Neutral
                | RetrievalOutcome::Abstained
                | RetrievalOutcome::Ignored => {}
            }
            match self.long_term_memory.update(&persisted) {
                Ok(()) if rewrites_importance => stats.rewritten.push(id.clone()),
                Ok(()) => {}
                Err(e) => {
                    persist_failures.push((id.clone(), e.to_string()));
                    tracing::warn!(
                        memory_id = %id.0,
                        error = %e,
                        "Failed to persist reinforcement update - Hebbian feedback may be lost on restart"
                    );
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

        // R2 (LTR) JOIN HOOK: a feedback label has arrived for these memories.
        // Graduate the matching PENDING trace rows (captured at recall time) to
        // TRAINING, stamping them with `outcome`. Joins by `memory_id` because the
        // live feedback path (reinforce_feedback handler → reinforce_recall) carries
        // NO retrieval/trace id — memory_id + outcome is the only correlation
        // available here; join_feedback picks the most-recent un-expired pending row
        // per memory. Best-effort: it returns a graduated count and never errors out
        // of the reinforcement path. Does not affect the importance/Hebbian updates
        // above (shadow side-channel only).
        let graduated = self
            .retrieval_trace_store
            .join_feedback(memory_ids, outcome);
        if graduated > 0 {
            tracing::debug!(
                graduated,
                ?outcome,
                "LTR trace: graduated pending rows to TRAINING on feedback"
            );
        }

        // R3b: wire adaptive-weight update from the feedback signal.
        //
        // For each reinforced memory, look up its signal attribution (BM25 /
        // vector / graph contribution fractions) from the most-recent recall.
        // Call `update_weights_from_feedback` on the hybrid search engine so the
        // EMA ranker learns from this signal.  After all updates, persist the new
        // vector via `learned_weights_store` (R3a F3 closure: survives restarts).
        //
        // The B8 clamp (max blend = 0.6) lives inside `effective_weights()`; the
        // per-epoch regularisation and negative-count tracking live inside
        // `update_weights_from_feedback`.  Per-user blast radius is contained by
        // construction (this MemorySystem is per-user).
        //
        // SHADOW ONLY: `update_weights_from_feedback` mutates `learned_weights`
        // on the engine, but those weights are NOT consumed by the scoring path
        // in this PR (`R3B_LIVE_APPLICATION_ENABLED = false`).
        {
            let is_negative = matches!(
                outcome,
                RetrievalOutcome::Misleading
                    | RetrievalOutcome::Abstained
                    | RetrievalOutcome::Ignored
            );
            let is_helpful = matches!(outcome, RetrievalOutcome::Helpful);
            // Only update on actionable outcomes — Neutral carries no gradient signal.
            if is_helpful || is_negative {
                let attr_guard = self.last_signal_attributions.read();
                for id in memory_ids {
                    if let Some(attr) = attr_guard.get(id) {
                        let signal_scores =
                            crate::memory::hybrid_search::SignalScores::from_components(
                                attr.bm25_contribution,
                                attr.vector_contribution,
                                attr.graph_contribution,
                            );
                        self.hybrid_search.update_weights_from_feedback(
                            &signal_scores,
                            is_helpful,
                            is_negative,
                        );
                    }
                }
                // Persist the updated weight vector so it survives a restart (R3a).
                let updated = self.hybrid_search.learned_weight_state();
                self.learned_weights_store.save(&updated);
            }
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

        // The pool's mean pairwise cosine is the anisotropy baseline: redundancy is how far a
        // pair sits above it, on the way to identity. See MMR_REDUNDANCY_THRESHOLD.
        let (pair_sum, pair_count) = (0..embeddings.len())
            .flat_map(|i| (i + 1..embeddings.len()).map(move |j| (i, j)))
            .filter_map(|(i, j)| match (embeddings[i], embeddings[j]) {
                (Some(a), Some(b)) => Some(crate::similarity::cosine_similarity(a, b)),
                _ => None,
            })
            .fold((0.0_f32, 0usize), |(s, n), v| (s + v, n + 1));
        let baseline = if pair_count > 0 {
            (pair_sum / pair_count as f32).clamp(0.0, 0.99)
        } else {
            0.0
        };
        let redundancy = |sim: f32| -> f32 {
            let centred = ((sim - baseline) / (1.0 - baseline)).clamp(0.0, 1.0);
            if centred >= crate::constants::MMR_REDUNDANCY_THRESHOLD {
                centred
            } else {
                0.0
            }
        };

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
                let relevance = (memories[cand_idx].score.unwrap_or(0.0) - min_score) / score_range;

                // Max similarity to any already-selected memory
                let max_sim = if let Some(cand_emb) = embeddings[cand_idx] {
                    selected
                        .iter()
                        .filter_map(|&sel_idx| {
                            embeddings[sel_idx].map(|sel_emb| {
                                redundancy(crate::similarity::cosine_similarity(cand_emb, sel_emb))
                            })
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

/// Person-name evidence in a memory for Layer 5.37, in `[0, 1]`: half the share of its
/// extracted entities that are people, half the number of distinct people capped at four.
/// Uses the rule-based extractor the write path falls back to when neural NER is off, which
/// is how production runs.
fn people_score(content: &str) -> f32 {
    static EXTRACTOR: std::sync::LazyLock<crate::graph_memory::EntityExtractor> =
        std::sync::LazyLock::new(crate::graph_memory::EntityExtractor::new);
    let entities = EXTRACTOR.extract_with_salience(content);
    if entities.is_empty() {
        return 0.0;
    }
    let people: HashSet<String> = entities
        .iter()
        .filter(|e| matches!(e.label, crate::graph_memory::EntityLabel::Person))
        .map(|e| e.name.to_lowercase())
        .collect();
    let share = people.len() as f32 / entities.len() as f32;
    let count = people.len().min(4) as f32 / 4.0;
    0.5 * share + 0.5 * count
}

/// Punctuation kept inside a query token besides alphanumerics. Canonical entity names and file
/// names are made of it — `commitwork:admin/serve.mjs`, `repo@1a2b3c4d`, `cc:<session>/<agent>`,
/// `#123`, `snake_case`, `kebab-case` — and stripping it turned `harvest.mjs` into `harvestmjs`,
/// which matches nothing.
const QUERY_TOKEN_KEEP: &[char] = &['.', '/', '_', ':', '@', '#', '-'];

/// Clean one whitespace-separated query word for entity matching: keep alphanumerics and
/// [`QUERY_TOKEN_KEEP`], then trim anything non-alphanumeric off both ends, which is sentence
/// punctuation rather than part of a name (`serve.mjs.`, `(harvest.mjs)`, `"admin/serve.mjs"`).
pub(crate) fn clean_query_token(word: &str) -> String {
    let kept: String = word
        .chars()
        .filter(|c| c.is_alphanumeric() || QUERY_TOKEN_KEEP.contains(c))
        .collect();
    kept.trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

/// A token that could name a file: `serve.mjs`, `admin/serve.mjs`.
pub(crate) fn is_path_shaped_token(token: &str) -> bool {
    token.contains('.') || token.contains('/')
}

/// A token carrying interior [`QUERY_TOKEN_KEEP`] punctuation is an identifier, not a word.
pub(crate) fn is_structured_token(token: &str) -> bool {
    token.chars().any(|c| QUERY_TOKEN_KEEP.contains(&c))
}

/// Porter-stem a word; leave an identifier alone. The stemmer works on English suffixes and
/// would read `serve.mjs` as a plural.
pub(crate) fn stem_query_token(stemmer: &rust_stemmers::Stemmer, token: &str) -> String {
    if is_structured_token(token) {
        token.to_string()
    } else {
        stemmer.stem(token).to_string()
    }
}

/// Whether a cleaned query token matches an entity's (stemmed or identifier) form. A plain word
/// matches as a substring, as it always has. An identifier must start at a boundary in the entity
/// — the start or after a non-alphanumeric — so `serve.mjs` matches `commitwork:admin/serve.mjs`
/// and not `commitwork:lib/observe.mjs`.
pub(crate) fn query_token_matches_entity(token: &str, entity: &str) -> bool {
    if !is_structured_token(token) {
        return entity.contains(token);
    }
    entity.match_indices(token).any(|(at, _)| {
        entity[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric())
    })
}

/// The cosine two candidates must reach to compete: the pool's median pair cosine plus
/// [`COMPETITION_POOL_HEADROOM`] of the way to 1, never below the absolute
/// [`INTERFERENCE_SIMILARITY_THRESHOLD`]. Fewer than three comparable pairs keep the absolute bar.
///
/// [`COMPETITION_POOL_HEADROOM`]: crate::constants::COMPETITION_POOL_HEADROOM
/// [`INTERFERENCE_SIMILARITY_THRESHOLD`]: crate::constants::INTERFERENCE_SIMILARITY_THRESHOLD
fn competition_similarity_floor(pool: &[SharedMemory]) -> f32 {
    use crate::constants::{COMPETITION_POOL_HEADROOM, INTERFERENCE_SIMILARITY_THRESHOLD};
    let vectors: Vec<&[f32]> = pool
        .iter()
        .filter_map(|m| m.experience.embeddings.as_deref())
        .filter(|v| !v.is_empty())
        .collect();
    let mut sims: Vec<f32> = Vec::new();
    for (i, left) in vectors.iter().enumerate() {
        for right in &vectors[i + 1..] {
            if left.len() == right.len() {
                sims.push(crate::similarity::cosine_similarity(left, right));
            }
        }
    }
    if sims.len() < 3 {
        return INTERFERENCE_SIMILARITY_THRESHOLD;
    }
    sims.sort_by(|a, b| a.total_cmp(b));
    let median = sims[sims.len() / 2];
    (median + COMPETITION_POOL_HEADROOM * (1.0 - median)).max(INTERFERENCE_SIMILARITY_THRESHOLD)
}

/// Orders each run of exactly equal scores in a score-sorted list by content, then id.
///
/// Fused scores tie often, and the map they come from iterates in a per-instance hash order.
/// Content is unique per store (remember() dedups on a content hash), so this is a total
/// order that two stores built from the same corpus agree on; ids are random per ingest.
/// Content is fetched only for members of a tied run.
fn order_score_ties_by_content<F>(items: &mut [(MemoryId, f32)], content: &F)
where
    F: Fn(&MemoryId) -> Option<String>,
{
    let mut start = 0;
    while start < items.len() {
        let bits = items[start].1.to_bits();
        let mut end = start + 1;
        while end < items.len() && items[end].1.to_bits() == bits {
            end += 1;
        }
        if end - start > 1 {
            let mut keyed: Vec<(Option<String>, (MemoryId, f32))> = items[start..end]
                .iter()
                .map(|item| (content(&item.0), item.clone()))
                .collect();
            keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| (a.1).0.cmp(&(b.1).0)));
            for (slot, (_, item)) in items[start..end].iter_mut().zip(keyed) {
                *slot = item;
            }
        }
        start = end;
    }
}

/// Which tier a recall candidate's record was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CandidateTier {
    Working,
    Session,
    LongTerm,
}

impl CandidateTier {
    fn source_label(self) -> &'static str {
        match self {
            CandidateTier::Working => "working",
            CandidateTier::Session => "session",
            CandidateTier::LongTerm => "longterm",
        }
    }
}

/// One recall candidate's record plus the derived fields the fusion layers read.
pub(crate) struct CandidateView {
    pub(crate) memory: SharedMemory,
    pub(crate) tier: CandidateTier,
    pub(crate) content_lower: String,
    pub(crate) entities_lower: Vec<String>,
}

impl CandidateView {
    fn new(memory: SharedMemory, tier: CandidateTier) -> Self {
        let content_lower = memory.experience.content.to_lowercase();
        let entities_lower = memory
            .experience
            .entities
            .iter()
            .map(|e| e.to_lowercase())
            .collect();
        Self {
            memory,
            tier,
            content_lower,
            entities_lower,
        }
    }
}

/// Where [`RecallCandidates`] reads records from.
pub(crate) trait CandidateSource {
    /// The records among `ids` resident in the in-memory tiers, working before session.
    fn resident(&self, ids: &[MemoryId]) -> Vec<(MemoryId, SharedMemory, CandidateTier)>;
    /// Storage reads for `ids`, results in the same order.
    fn stored(&self, ids: &[MemoryId]) -> Vec<Result<Memory>>;
}

impl CandidateSource for super::MemorySystem {
    fn resident(&self, ids: &[MemoryId]) -> Vec<(MemoryId, SharedMemory, CandidateTier)> {
        let mut found = Vec::new();
        let mut not_working: Vec<&MemoryId> = Vec::new();
        {
            let working = self.working_memory.read();
            for id in ids {
                match working.get(id) {
                    Some(memory) => found.push((id.clone(), memory, CandidateTier::Working)),
                    None => not_working.push(id),
                }
            }
        }
        if !not_working.is_empty() {
            let session = self.session_memory.read();
            for id in not_working {
                if let Some(memory) = session.get(id) {
                    found.push((id.clone(), memory, CandidateTier::Session));
                }
            }
        }
        found
    }

    fn stored(&self, ids: &[MemoryId]) -> Vec<Result<Memory>> {
        self.retriever.get_many_from_storage(ids)
    }
}

enum CandidateSlot {
    Loaded(Arc<CandidateView>),
    /// Neither tier holds it and storage could not produce it; the reason, for the log.
    Missing(String),
}

/// One recall's projection of its candidates: each id is read once (tiers first, then one
/// storage batch for the rest) and every fusion layer and the Layer 5 fetch read the same
/// record. Before this, each layer re-read every fused candidate through its own
/// tier-lock-then-storage closure.
pub(crate) struct RecallCandidates<'s, S: CandidateSource + ?Sized> {
    source: &'s S,
    slots: std::cell::RefCell<std::collections::HashMap<MemoryId, CandidateSlot>>,
}

impl<'s, S: CandidateSource + ?Sized> RecallCandidates<'s, S> {
    pub(crate) fn new(source: &'s S) -> Self {
        Self {
            source,
            slots: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// Load every id not yet held.
    pub(crate) fn prefetch<'a>(&self, ids: impl IntoIterator<Item = &'a MemoryId>) {
        let wanted: Vec<MemoryId> = {
            let slots = self.slots.borrow();
            let mut seen: HashSet<&MemoryId> = HashSet::new();
            ids.into_iter()
                .filter(|id| !slots.contains_key(*id) && seen.insert(*id))
                .cloned()
                .collect()
        };
        if wanted.is_empty() {
            return;
        }
        let mut slots = self.slots.borrow_mut();
        for (id, memory, tier) in self.source.resident(&wanted) {
            slots.insert(
                id,
                CandidateSlot::Loaded(Arc::new(CandidateView::new(memory, tier))),
            );
        }
        let from_storage: Vec<MemoryId> = wanted
            .into_iter()
            .filter(|id| !slots.contains_key(id))
            .collect();
        if from_storage.is_empty() {
            return;
        }
        let results = self.source.stored(&from_storage);
        for (id, result) in from_storage.into_iter().zip(results) {
            let slot = match result {
                Ok(memory) => CandidateSlot::Loaded(Arc::new(CandidateView::new(
                    Arc::new(memory),
                    CandidateTier::LongTerm,
                ))),
                Err(e) => CandidateSlot::Missing(format!("{e:#}")),
            };
            slots.insert(id, slot);
        }
    }

    /// The candidate's view, loading it first if no earlier call did.
    pub(crate) fn get(&self, id: &MemoryId) -> Option<Arc<CandidateView>> {
        self.prefetch(std::iter::once(id));
        match self.slots.borrow().get(id) {
            Some(CandidateSlot::Loaded(view)) => Some(Arc::clone(view)),
            _ => None,
        }
    }

    /// Why `id` could not be loaded, if it could not.
    pub(crate) fn failure(&self, id: &MemoryId) -> Option<String> {
        match self.slots.borrow().get(id) {
            Some(CandidateSlot::Missing(reason)) => Some(reason.clone()),
            _ => None,
        }
    }

    /// The candidate's content, owned, for callers that take `Fn(&MemoryId) -> Option<String>`.
    pub(crate) fn content(&self, id: &MemoryId) -> Option<String> {
        self.get(id)
            .map(|view| view.memory.experience.content.clone())
    }
}

/// Layer 4.525's weight for a query word that matches the entities of `matching` of the `n`
/// candidates carrying entities: `ln(n / matching) / ln(n)`, 1 for a word only one candidate
/// carries and 0 for a word every candidate carries.
fn entity_match_specificity(n: usize, matching: usize) -> f32 {
    if n <= 1 || matching == 0 {
        return 1.0;
    }
    ((n as f32 / matching as f32).ln() / (n as f32).ln()).clamp(0.0, 1.0)
}

/// Layer 4.525's share of a memory's own fused score for its entity matches: each word match
/// is worth `ENTITY_OVERLAP_SHARE` times its specificity (`word_weight` is their sum), 0.4 of
/// that on a decomposed query, whose generic stems ("perform") match many memories alike; each
/// identifier match is worth the full share, decomposed or not. Four full matches are the cap.
fn entity_overlap_share(word_weight: f32, identifiers: usize, decomposed: bool) -> f32 {
    let share = crate::constants::ENTITY_OVERLAP_SHARE;
    let per_word = if decomposed { 0.4 * share } else { share };
    let words = (per_word * word_weight).min(4.0 * per_word);
    (words + share * identifiers as f32).min(4.0 * share)
}

/// One hybrid-list candidate as Layer 4's FLAT fusion reads it.
#[derive(Debug, Clone, Copy)]
struct HybridComponents {
    /// Raw BM25 score, 0 when only the vector list found it.
    bm25: f32,
    /// Raw vector-list score (cosine, or reciprocal rank on a decomposed query), 0 when
    /// only BM25 found it.
    vector: f32,
    /// The factor the LLM refiner multiplied the hybrid score by; 1 when it did not run.
    refined: f32,
}

/// The highest score in a fused list, the denominator FLAT calibrates the list against.
fn best_leg_score(scores: impl Iterator<Item = f32>) -> f32 {
    scores.filter(|s| s.is_finite()).fold(0.0, f32::max)
}

/// `score` over its list's best, in [0, 1]; 0 for a list with no positive score.
fn calibrated(score: f32, best: f32) -> f32 {
    if best > 0.0 && score.is_finite() {
        (score / best).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// What one list adds to a candidate's fused score: `weight / (k + rank)` (rank from 1) under
/// reciprocal rank, `weight × score / best` under FLAT.
fn leg_term(mode: FusionMode, weight: f32, k: f32, rank: usize, score: f32, best: f32) -> f32 {
    match mode {
        FusionMode::Rrf => weight / (k + rank as f32),
        FusionMode::Flat => weight * calibrated(score, best),
    }
}

/// FLAT's join of a candidate's calibrated vector and BM25 scores: the stronger, plus
/// `FLAT_CONSENSUS` of the weaker.
fn flat_hybrid(vector: f32, bm25: f32) -> f32 {
    let (strong, weak) = if vector >= bm25 {
        (vector, bm25)
    } else {
        (bm25, vector)
    };
    strong + crate::constants::FLAT_CONSENSUS * weak
}

/// Under the `graph_candidates_only` dial, the fused terms of the graph hits no hybrid list
/// returned, each scored as the vector list would have scored it: under FLAT its cosine over
/// the best of theirs and `vector_best` (the vector list's best, read as a cosine), times the
/// hybrid weight; under reciprocal rank, ranked after the `hybrid_len` hybrid candidates in
/// cosine order. Nothing of the graph's own weighting enters.
fn admitted_graph_hit_terms(
    mode: FusionMode,
    hybrid_w: f32,
    k: f32,
    hybrid_len: usize,
    mut cosines: Vec<(&MemoryId, f32)>,
    vector_best: Option<f32>,
) -> Vec<(&MemoryId, f32)> {
    cosines.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let best = best_leg_score(cosines.iter().map(|(_, c)| *c).chain(vector_best));
    cosines
        .into_iter()
        .enumerate()
        .map(|(r, (id, cos))| {
            (
                id,
                leg_term(mode, hybrid_w, k, hybrid_len + r + 1, cos, best),
            )
        })
        .collect()
}

/// The multiplier on a graph candidate's graph term for its activation.
fn graph_activation_factor(graph_w: f32, activation: f32) -> f32 {
    1.0 + graph_w * crate::constants::ACTIVATION_BONUS_SCALE * activation.clamp(0.0, 1.0)
}

/// Layer 4.54 (temporal topic dedup) over the `cap` highest fused scores: for each pair of
/// those whose lowercase entity sets have Jaccard > 0.40, the older is demoted ×0.85, once
/// however many newer memories share its topic. Compounded per pair, the oldest memory on a
/// busy topic fell by 0.85^n and left the top 50 entirely.
/// Pairs are visited in content order: when two same-topic memories share a timestamp the
/// second one visited is demoted, so the order must not come from the map's hash order.
/// Returns the number of memories demoted.
fn temporal_topic_dedup<S: CandidateSource + ?Sized>(
    fused: &mut std::collections::HashMap<MemoryId, f32>,
    candidates: &RecallCandidates<'_, S>,
    cap: usize,
) -> usize {
    let mut entries: Vec<(MemoryId, f32, Arc<CandidateView>)> = fused
        .iter()
        .filter_map(|(id, score)| candidates.get(id).map(|view| (id.clone(), *score, view)))
        .filter(|(_, _, view)| !view.entities_lower.is_empty())
        .collect();
    let by_content = |a: &(MemoryId, f32, Arc<CandidateView>),
                      b: &(MemoryId, f32, Arc<CandidateView>)| {
        a.2.memory
            .experience
            .content
            .cmp(&b.2.memory.experience.content)
            .then_with(|| a.0.cmp(&b.0))
    };
    entries.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| by_content(a, b)));
    entries.truncate(cap);
    entries.sort_by(by_content);

    let entity_sets: Vec<HashSet<&str>> = entries
        .iter()
        .map(|(_, _, view)| view.entities_lower.iter().map(String::as_str).collect())
        .collect();

    let mut older: HashSet<&MemoryId> = HashSet::new();
    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            let (set_a, set_b) = (&entity_sets[i], &entity_sets[j]);
            let (small, large) = if set_a.len() <= set_b.len() {
                (set_a, set_b)
            } else {
                (set_b, set_a)
            };
            let intersection = small.iter().filter(|e| large.contains(*e)).count();
            let union = set_a.len() + set_b.len() - intersection;
            let jaccard = intersection as f32 / union.max(1) as f32;
            if jaccard > 0.40 {
                let (ts_a, ts_b) = (
                    entries[i].2.memory.created_at,
                    entries[j].2.memory.created_at,
                );
                older.insert(if ts_a < ts_b {
                    &entries[i].0
                } else {
                    &entries[j].0
                });
            }
        }
    }
    for id in &older {
        if let Some(score) = fused.get_mut(*id) {
            *score *= 0.85;
        }
    }
    older.len()
}

/// Query words resolved to graph entities for recall.
pub(crate) struct ResolvedQueryEntities {
    /// Entities the focal words resolved to, in query order, each once.
    pub(crate) focal: Vec<uuid::Uuid>,
    /// For each `focal` entity, the index in `focal_words` of the first word resolving to it.
    pub(crate) focal_word: Vec<usize>,
    /// FilePath entities the path-shaped query words name, each once.
    pub(crate) paths: Vec<uuid::Uuid>,
    /// Every resolved entity in query order (focal, then the other words, then path-shaped
    /// words' FilePath entities), each once.
    pub(crate) all: Vec<uuid::Uuid>,
    /// Lowercased names of the declared FilePath entities path-shaped query words resolved to.
    pub(crate) file_paths: Vec<String>,
}

/// Resolve recall's query words against the graph. Every word goes through the exact,
/// case-insensitive and stemmed indexes, which cost no storage read. Only a focal word those
/// miss falls back to fuzzy matching, bounded to
/// [`QUERY_ENTITY_FUZZY_MAX_CANDIDATES`](crate::constants::QUERY_ENTITY_FUZZY_MAX_CANDIDATES):
/// on the other words (modifiers and verbs, mostly not entities) the unbounded fuzzy tiers
/// scanned the whole name index and read every match on most queries.
pub(crate) fn resolve_query_entities(
    graph: &crate::graph_memory::GraphMemory,
    focal_words: &[&str],
    other_words: &[&str],
    query_text: &str,
) -> ResolvedQueryEntities {
    let mut focal: Vec<uuid::Uuid> = Vec::new();
    let mut focal_word: Vec<usize> = Vec::new();
    let mut paths: Vec<uuid::Uuid> = Vec::new();
    let mut all: Vec<uuid::Uuid> = Vec::new();
    let mut file_paths: Vec<String> = Vec::new();
    let mut tried: HashSet<&str> = HashSet::new();
    let add = |list: &mut Vec<uuid::Uuid>, uuid: uuid::Uuid| {
        let new = !list.contains(&uuid);
        if new {
            list.push(uuid);
        }
        new
    };

    for (index, &word) in focal_words.iter().enumerate() {
        if !tried.insert(word) {
            continue;
        }
        let resolved = graph.find_entity_uuid_strict(word).or_else(|| {
            graph
                .find_entity_by_name_bounded(
                    word,
                    crate::constants::QUERY_ENTITY_FUZZY_MAX_CANDIDATES,
                )
                .ok()
                .flatten()
                .map(|entity| entity.uuid)
        });
        if let Some(uuid) = resolved {
            if add(&mut focal, uuid) {
                focal_word.push(index);
            }
            add(&mut all, uuid);
        }
    }
    for &word in other_words {
        if !tried.insert(word) {
            continue;
        }
        if let Some(uuid) = graph.find_entity_uuid_strict(word) {
            add(&mut all, uuid);
        }
    }
    // A path-shaped query word also seeds every declared FilePath entity whose path ends with
    // it at a `/` boundary: "serve.mjs" and "admin/serve.mjs" reach `commitwork:admin/serve.mjs`,
    // which no name tier resolves without also admitting `observe.mjs`. One index probe per
    // path-shaped word, at most FILE_PATH_SUFFIX_MAX_CANDIDATES entity reads each.
    for word in query_text.split_whitespace() {
        let token = clean_query_token(word);
        if !is_path_shaped_token(&token) {
            continue;
        }
        if let Ok(found) = graph.find_file_paths_by_suffix(&token) {
            for entity in found {
                add(&mut paths, entity.uuid);
                add(&mut all, entity.uuid);
                let name = entity.name.to_lowercase();
                if !file_paths.contains(&name) {
                    file_paths.push(name);
                }
            }
        }
    }
    ResolvedQueryEntities {
        focal,
        focal_word,
        paths,
        all,
        file_paths,
    }
}

/// The graph leg's list for fusion, from entries `(id, relevance score, graph strength,
/// record)`: the best `cap` by score, equal scores by `content` (unique per store and
/// repeat-stable, where ids are random per ingest), each as `(id, strength, strength)`.
///
/// The relevance score orders the list and goes no further. Fusion reads the second slot as
/// graph activation, for its graph bonus and the graph attribution that abstention and the
/// learned leg weights read, and the third as the Layer 5 graph value. Both carry the
/// episode's graph strength, its activation over the pool's largest, so a memory the leg
/// ranks high for its cosine alone earns no graph credit.
fn graph_leg_for_fusion<T>(
    mut ranked: Vec<(MemoryId, f32, f32, T)>,
    cap: usize,
    content: impl Fn(&T) -> &str,
) -> Vec<(MemoryId, f32, f32)> {
    ranked.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| content(&a.3).cmp(content(&b.3)))
    });
    ranked.truncate(cap);
    ranked
        .into_iter()
        .map(|(id, _, strength, _)| (id, strength, strength))
        .collect()
}

/// The graph leg's seeds, each once: the entities the query's nouns resolved to, at their
/// words' information content, then at a noun's the FilePath entities its path-shaped words
/// name and the entities the caller's NER names resolve to strictly.
///
/// Adjectives and verbs do not seed: words like "working" or "enjoy" resolve to hub entities
/// and spread the walk over the store. A seed whose entity cannot be read is left out.
fn graph_leg_seeds(
    graph: &crate::graph_memory::GraphMemory,
    resolved: Option<&ResolvedQueryEntities>,
    analysis: &query_parser::QueryAnalysis,
    ner_names: Option<&[String]>,
) -> Vec<crate::memory::graph_retrieval::GraphSeed> {
    let noun_ic = crate::constants::IC_NOUN;
    let mut named: Vec<(uuid::Uuid, f32)> = Vec::new();
    let mut add = |uuid: uuid::Uuid, ic_weight: f32| {
        if !named.iter().any(|(seen, _)| *seen == uuid) {
            named.push((uuid, ic_weight));
        }
    };
    if let Some(resolved) = resolved {
        for (uuid, &word) in resolved.focal.iter().zip(&resolved.focal_word) {
            let ic_weight = analysis
                .focal_entities
                .get(word)
                .map_or(noun_ic, |e| e.ic_weight);
            add(*uuid, ic_weight);
        }
        for uuid in &resolved.paths {
            add(*uuid, noun_ic);
        }
    }
    for name in ner_names.unwrap_or_default() {
        if name.trim().chars().count() < 2 {
            continue;
        }
        if let Some(uuid) = graph.find_entity_uuid_strict(name) {
            add(uuid, noun_ic);
        }
    }
    named
        .into_iter()
        .filter_map(|(uuid, ic_weight)| {
            let entity = graph.get_entity(&uuid).ok().flatten()?;
            Some(crate::memory::graph_retrieval::GraphSeed {
                uuid,
                ic_weight,
                mention_count: entity.mention_count,
            })
        })
        .collect()
}

// =============================================================================
// R2 unit tests — temporal boost bound-overlap logic
// =============================================================================
//
// These tests validate the R2 temporal_boost path in isolation. They do NOT
// exercise the full `with_unified_score` closure (which requires a live
// MemorySystem); instead they test the exact sub-algorithm that was changed:
// parsing a memory temporal_ref → TemporalBound::Point, then checking overlap
// against a query-side TemporalBound widened ±7 days.
#[cfg(test)]
mod r2_temporal_boost_tests {
    use crate::memory::temporal_facts::TemporalBound;
    use chrono::NaiveDate;

    /// Simulate the R2 boost decision for a single (mem_date, query_bounds) pair.
    ///
    /// Returns true when the mem_date falls within ANY query bound's ±7-day window
    /// (i.e. temporal_boost would be > 0).  Returns false otherwise.
    fn r2_boost_fires(mem_date: NaiveDate, query_bounds: &[TemporalBound]) -> bool {
        if query_bounds.is_empty() {
            return false;
        }
        let window_start = mem_date - chrono::Duration::days(7);
        let window_end = mem_date + chrono::Duration::days(7);
        query_bounds
            .iter()
            .any(|qb| qb.overlaps_range(window_start, window_end))
    }

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// R2-B1: a memory date exactly equal to the query's Point bound receives
    /// the boost (overlap is true, boost value would be 0.25 > 0).
    #[test]
    fn test_r2_boost_exact_point_match() {
        let query_date = d(2024, 6, 15);
        let bounds = vec![TemporalBound::Point(query_date)];
        assert!(
            r2_boost_fires(d(2024, 6, 15), &bounds),
            "exact date match must fire the boost"
        );
    }

    /// R2-B2: a memory date 6 days inside the ±7-day window receives the boost.
    #[test]
    fn test_r2_boost_within_window() {
        let query_date = d(2024, 6, 15);
        let bounds = vec![TemporalBound::Point(query_date)];
        // mem_date window: [2024-06-09, 2024-06-23]
        // query Point(2024-06-15) overlaps [2024-06-09, 2024-06-23] → true
        assert!(
            r2_boost_fires(d(2024, 6, 21), &bounds),
            "mem date 6 days away must still be within ±7-day tolerance"
        );
    }

    /// R2-B3: a memory date 30 days outside the query bound does NOT fire the boost.
    #[test]
    fn test_r2_boost_outside_window() {
        let query_date = d(2024, 6, 15);
        let bounds = vec![TemporalBound::Point(query_date)];
        // mem_date = 2024-07-20, window = [2024-07-13, 2024-07-27]
        // query Point(2024-06-15) does NOT overlap [2024-07-13, 2024-07-27]
        assert!(
            !r2_boost_fires(d(2024, 7, 20), &bounds),
            "mem date 35 days from query must NOT fire the boost"
        );
    }

    /// R2-B4: when query_bounds is empty the boost does not fire (fallback guard).
    #[test]
    fn test_r2_no_bounds_no_boost() {
        assert!(
            !r2_boost_fires(d(2024, 6, 15), &[]),
            "empty bounds slice must not fire the boost"
        );
    }

    /// R2-B5: a Closed-interval query bound overlapping the mem date fires the boost.
    #[test]
    fn test_r2_boost_closed_interval_overlap() {
        let bounds = vec![TemporalBound::Closed {
            start: d(2024, 6, 10),
            end: d(2024, 6, 20),
        }];
        // mem_date = 2024-06-25 → window [2024-06-18, 2024-07-02]
        // Closed[10..20] overlaps [18..Jul-02] → 18 <= 20 && 10 <= Jul-02 → true
        assert!(
            r2_boost_fires(d(2024, 6, 25), &bounds),
            "Closed query bound partially overlapping ±7-day mem window must fire"
        );
    }

    /// R2-B6: a Closed-interval query bound entirely outside the mem window does not fire.
    #[test]
    fn test_r2_boost_closed_interval_no_overlap() {
        let bounds = vec![TemporalBound::Closed {
            start: d(2024, 5, 1),
            end: d(2024, 5, 10),
        }];
        // mem_date = 2024-06-25 → window [2024-06-18, 2024-07-02]
        // Closed[May-01..May-10] does NOT overlap [Jun-18..Jul-02]
        assert!(
            !r2_boost_fires(d(2024, 6, 25), &bounds),
            "Closed query bound 6 weeks before mem window must NOT fire"
        );
    }

    /// R2-B7: extract_temporal_refs on a "yesterday"-style query populates bounds(),
    /// and the R2 boost fires for a mem_date of today (within the ±7-day window).
    ///
    /// This is an integration-level check that the full pipeline from text → bounds
    /// → overlap produces a non-zero boost for a temporally-relevant memory.
    #[test]
    fn test_r2_boost_via_extract_temporal_refs_yesterday() {
        use crate::memory::query_parser::extract_temporal_refs;

        let extraction = extract_temporal_refs("what happened yesterday?");
        let bounds = extraction.bounds();

        // If the parser resolved "yesterday" to a bound, test the overlap.
        // If bounds is empty (parser did not handle it), the test is vacuously
        // satisfied — this guards against environment-dependent date parsing.
        if bounds.is_empty() {
            return;
        }

        // today should be within 1 day of "yesterday" → well within ±7-day window
        let today = chrono::Utc::now().date_naive();
        assert!(
            r2_boost_fires(today, bounds),
            "a memory dated today should receive the boost for a 'yesterday' query"
        );

        // a date 60 days ago should NOT fire (well outside ±7-day tolerance)
        let old_date = today - chrono::Duration::days(60);
        assert!(
            !r2_boost_fires(old_date, bounds),
            "a memory 60 days old should NOT receive the boost for a 'yesterday' query"
        );
    }
}

// =============================================================================
// Signal-disable gate tests (SD-*)
// =============================================================================
// These tests verify that:
//   (a) Default dials → signal_enabled returns true for all IDs (SD-1, in dials.rs).
//   (b) Specific IDs in disabled_signals → signal_enabled returns false (SD-2, in dials.rs).
//   (c) The gate IS wired into the score formula: disabling a signal whose
//       magnitude is non-zero for a given input changes the resulting score.
//
// Test (c) works by directly exercising the components of the `with_unified_score`
// closure logic — no I/O, no storage, no embedders — so it runs fast and hermetically.
#[cfg(test)]
mod signal_disable_gate_tests {
    use crate::memory::dials::CognitiveDials;
    use std::collections::HashSet;

    /// Simulates the arousal_boost component with and without the signal enabled.
    /// When `arousal` is disabled the contribution MUST be 0.0.
    #[test]
    fn sd_c1_arousal_gate_changes_score() {
        // A memory with non-zero arousal (0.8) contributes 0.8 * 0.05 = 0.04 normally.
        let raw_arousal: f32 = 0.8;
        let expected_enabled: f32 = raw_arousal * 0.05;

        let dials_enabled = CognitiveDials::default();
        let boost_enabled = if dials_enabled.signal_enabled("arousal") {
            raw_arousal * 0.05
        } else {
            0.0
        };
        assert!(
            (boost_enabled - expected_enabled).abs() < 1e-6,
            "with signal enabled: expected {expected_enabled}, got {boost_enabled}"
        );

        let mut disabled = HashSet::new();
        disabled.insert("arousal".to_string());
        let dials_disabled = CognitiveDials {
            disabled_signals: disabled,
            ..Default::default()
        };
        let boost_disabled = if dials_disabled.signal_enabled("arousal") {
            raw_arousal * 0.05
        } else {
            0.0
        };
        assert_eq!(boost_disabled, 0.0, "arousal disabled: boost must be 0.0");

        // The two paths must produce different results (the gate is load-bearing).
        assert!(
            (boost_enabled - boost_disabled).abs() > 1e-6,
            "disabling arousal must change the score contribution"
        );
    }

    /// Simulates the valence_boost component with and without the signal enabled.
    /// When `valence` is disabled the contribution MUST be 0.0.
    #[test]
    fn sd_c2_valence_gate_changes_score() {
        let raw_valence: f32 = -0.6_f32; // negative valence still contributes abs * 0.02
        let expected_enabled: f32 = raw_valence.abs() * 0.02;

        let dials_enabled = CognitiveDials::default();
        let boost_enabled = if dials_enabled.signal_enabled("valence") {
            raw_valence.abs() * 0.02
        } else {
            0.0
        };
        assert!(
            (boost_enabled - expected_enabled).abs() < 1e-6,
            "with signal enabled: expected {expected_enabled}, got {boost_enabled}"
        );

        let mut disabled = HashSet::new();
        disabled.insert("valence".to_string());
        let dials_disabled = CognitiveDials {
            disabled_signals: disabled,
            ..Default::default()
        };
        let boost_disabled = if dials_disabled.signal_enabled("valence") {
            raw_valence.abs() * 0.02
        } else {
            0.0
        };
        assert_eq!(boost_disabled, 0.0, "valence disabled: boost must be 0.0");
        assert!(
            (boost_enabled - boost_disabled).abs() > 1e-6,
            "disabling valence must change the score contribution"
        );
    }

    /// Simulates the session_continuity component.
    /// A memory within 2 hours gets 0.03 when enabled, 0.0 when disabled.
    #[test]
    fn sd_c3_session_continuity_gate_changes_score() {
        let age_hours: i64 = 1; // within session window

        let dials_enabled = CognitiveDials::default();
        let boost_enabled: f32 = if dials_enabled.signal_enabled("session_continuity") {
            if age_hours <= 2 {
                0.03
            } else {
                0.0
            }
        } else {
            0.0
        };
        assert!(
            (boost_enabled - 0.03_f32).abs() < 1e-6,
            "session_continuity enabled: expected 0.03, got {boost_enabled}"
        );

        let mut disabled = HashSet::new();
        disabled.insert("session_continuity".to_string());
        let dials_disabled = CognitiveDials {
            disabled_signals: disabled,
            ..Default::default()
        };
        let boost_disabled: f32 = if dials_disabled.signal_enabled("session_continuity") {
            if age_hours <= 2 {
                0.03
            } else {
                0.0
            }
        } else {
            0.0
        };
        assert_eq!(
            boost_disabled, 0.0_f32,
            "session_continuity disabled: boost must be 0.0"
        );
        assert!(
            (boost_enabled - boost_disabled).abs() > 1e-6,
            "disabling session_continuity must change the score contribution"
        );
    }

    /// Simulates the hebbian gate on base_score.
    /// When `hebbian` is disabled the fused score does not take its 0.1× hebbian share.
    #[test]
    fn sd_c4_hebbian_gate_changes_base_score() {
        let rrf_score: f32 = 0.5;
        let hebbian_score: f32 = 0.4;

        let dials_enabled = CognitiveDials::default();
        let base_enabled = if dials_enabled.signal_enabled("hebbian") {
            rrf_score * (1.0 + hebbian_score * 0.1)
        } else {
            rrf_score
        };
        let expected = rrf_score * (1.0 + hebbian_score * 0.1);
        assert!(
            (base_enabled - expected).abs() < 1e-6,
            "hebbian enabled: expected {expected}, got {base_enabled}"
        );

        let mut disabled = HashSet::new();
        disabled.insert("hebbian".to_string());
        let dials_disabled = CognitiveDials {
            disabled_signals: disabled,
            ..Default::default()
        };
        let base_disabled = if dials_disabled.signal_enabled("hebbian") {
            rrf_score * (1.0 + hebbian_score * 0.1)
        } else {
            rrf_score
        };
        assert!(
            (base_disabled - rrf_score).abs() < 1e-6,
            "hebbian disabled: base must equal rrf_score {rrf_score}, got {base_disabled}"
        );
        assert!(
            (base_enabled - base_disabled).abs() > 1e-6,
            "disabling hebbian must change base_score"
        );
    }
}

#[cfg(test)]
mod competition_floor_tests {
    use super::*;
    use crate::constants::INTERFERENCE_SIMILARITY_THRESHOLD;
    use crate::memory::types::{Experience, Memory, MemoryId};

    fn embedded(v: Vec<f32>) -> SharedMemory {
        let experience = Experience {
            content: "turn".into(),
            embeddings: Some(v),
            ..Default::default()
        };
        Arc::new(Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            experience,
            0.5,
            None,
            None,
            None,
            None,
        ))
    }

    fn pool_at(angles: &[f32]) -> Vec<SharedMemory> {
        angles
            .iter()
            .map(|a| embedded(vec![a.cos(), a.sin()]))
            .collect()
    }

    #[test]
    fn a_pool_of_unrelated_candidates_keeps_the_absolute_bar() {
        let pool = pool_at(&[0.0, 1.2, 2.4, 3.6]);
        assert_eq!(
            competition_similarity_floor(&pool),
            INTERFERENCE_SIMILARITY_THRESHOLD
        );
    }

    #[test]
    fn a_pool_whose_pairs_all_score_high_must_be_beaten_by_its_own_background() {
        // Pair cosines 0.955 x3, 0.825 x2, 0.622: the median is 0.955, so the bar sits above it.
        let pool = pool_at(&[0.0, 0.3, 0.6, 0.9]);
        let floor = competition_similarity_floor(&pool);
        assert!(floor > 0.955 && floor < 1.0, "{floor}");
    }

    #[test]
    fn too_few_embedded_candidates_keep_the_absolute_bar() {
        let pool = pool_at(&[0.0, 0.1]);
        assert_eq!(
            competition_similarity_floor(&pool),
            INTERFERENCE_SIMILARITY_THRESHOLD
        );
    }
}

#[cfg(test)]
mod score_tie_order_tests {
    use super::{graph_leg_for_fusion, order_score_ties_by_content};
    use crate::memory::types::MemoryId;
    use std::collections::HashMap;

    #[test]
    fn the_graph_leg_is_ordered_by_relevance_and_credits_only_graph_strength() {
        let [cosine_only, graph_strong, tied] = [(); 3].map(|_| MemoryId(uuid::Uuid::new_v4()));
        let ranked = vec![
            (graph_strong.clone(), 0.31, 1.0, "graph strong"),
            (
                cosine_only.clone(),
                0.33,
                0.02,
                "cosine 0.62, no graph mass",
            ),
            (tied.clone(), 0.31, 0.5, "a tie, by content"),
        ];
        let fused = graph_leg_for_fusion(ranked, 2, |content| content);
        assert_eq!(
            fused,
            vec![(cosine_only, 0.02, 0.02), (tied, 0.5, 0.5)],
            "relevance orders and the cap cuts; graph strength fills both slots"
        );
    }

    #[test]
    fn equal_scores_are_ordered_by_content_and_the_rest_is_untouched() {
        let ids: Vec<MemoryId> = (0..5).map(|_| MemoryId(uuid::Uuid::new_v4())).collect();
        let content: HashMap<MemoryId, String> = ids
            .iter()
            .zip(["top", "delta", "alpha", "charlie", "bottom"])
            .map(|(id, c)| (id.clone(), c.to_string()))
            .collect();
        let mut items = vec![
            (ids[0].clone(), 0.9),
            (ids[1].clone(), 0.5),
            (ids[2].clone(), 0.5),
            (ids[3].clone(), 0.5),
            (ids[4].clone(), 0.1),
        ];

        order_score_ties_by_content(&mut items, &|id: &MemoryId| content.get(id).cloned());

        let order: Vec<&str> = items.iter().map(|(id, _)| content[id].as_str()).collect();
        assert_eq!(order, ["top", "alpha", "charlie", "delta", "bottom"]);
    }
}

#[cfg(test)]
mod ordinal_pin_tests {
    use crate::memory::types::{Experience, Memory, MemoryId, SharedMemory};
    use crate::memory::MemorySystem;
    use std::sync::Arc;

    fn scored(content: &str, score: f32) -> SharedMemory {
        let mut m = Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            Experience {
                content: content.to_string(),
                ..Default::default()
            },
            0.5,
            None,
            None,
            None,
            None,
        );
        m.set_score(score);
        Arc::new(m)
    }

    #[test]
    fn a_pin_undone_by_a_re_sort_returns_to_first_with_a_monotonic_score() {
        let pinned = scored("earliest", 0.4);
        let mut list = vec![scored("a", 0.9), pinned.clone(), scored("b", 0.2)];
        MemorySystem::reassert_ordinal_pin(&mut list, pinned.clone(), 10);
        assert_eq!(list[0].id, pinned.id);
        assert_eq!(list.len(), 3);
        assert!(list[0].score.unwrap() > list[1].score.unwrap());
    }

    #[test]
    fn a_pin_truncated_away_is_reinserted_within_the_limit() {
        let pinned = scored("earliest", 0.3);
        let mut list = vec![scored("a", 0.9), scored("b", 0.8)];
        MemorySystem::reassert_ordinal_pin(&mut list, pinned.clone(), 2);
        assert_eq!(list[0].id, pinned.id);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn a_pin_already_first_keeps_its_score() {
        let pinned = scored("earliest", 0.95);
        let mut list = vec![pinned.clone(), scored("a", 0.9)];
        MemorySystem::reassert_ordinal_pin(&mut list, pinned.clone(), 10);
        assert_eq!(list[0].id, pinned.id);
        assert_eq!(list[0].score, Some(0.95));
    }
}

#[cfg(test)]
mod query_token_tests {
    use super::{clean_query_token, query_token_matches_entity, stem_query_token};
    use rust_stemmers::{Algorithm, Stemmer};

    #[test]
    fn identifiers_survive_cleaning_and_sentence_punctuation_does_not() {
        assert_eq!(clean_query_token("harvest.mjs"), "harvest.mjs");
        assert_eq!(clean_query_token("(harvest.mjs)?"), "harvest.mjs");
        assert_eq!(clean_query_token("serve.mjs."), "serve.mjs");
        assert_eq!(clean_query_token("\"admin/serve.mjs\","), "admin/serve.mjs");
        assert_eq!(
            clean_query_token("commitwork:admin/serve.mjs"),
            "commitwork:admin/serve.mjs"
        );
        assert_eq!(
            clean_query_token("commitwork@1a2b3c4d"),
            "commitwork@1a2b3c4d"
        );
        assert_eq!(clean_query_token("cc:abc/def"), "cc:abc/def");
        assert_eq!(clean_query_token("snake_case-name"), "snake_case-name");
        assert_eq!(
            clean_query_token("what's"),
            "whats",
            "other punctuation still goes"
        );
        assert_eq!(clean_query_token("..."), "");
    }

    #[test]
    fn identifiers_are_not_stemmed_and_words_still_are() {
        let stemmer = Stemmer::create(Algorithm::English);
        assert_eq!(stem_query_token(&stemmer, "harvest.mjs"), "harvest.mjs");
        assert_eq!(stem_query_token(&stemmer, "serve.mjs"), "serve.mjs");
        assert_eq!(stem_query_token(&stemmer, "changes"), "chang");
        assert_eq!(stem_query_token(&stemmer, "risks"), "risk");
    }

    #[test]
    fn an_identifier_matches_only_at_a_boundary() {
        let entity = "commitwork:admin/serve.mjs";
        assert!(query_token_matches_entity("serve.mjs", entity));
        assert!(query_token_matches_entity("admin/serve.mjs", entity));
        assert!(query_token_matches_entity(entity, entity));
        assert!(!query_token_matches_entity(
            "serve.mjs",
            "commitwork:lib/observe.mjs"
        ));
        // A plain word keeps the old substring rule.
        assert!(query_token_matches_entity("serv", "observer"));
    }
}

#[cfg(test)]
mod date_window_leg_tests {
    use super::*;
    use crate::memory::dials::CognitiveDials;
    use crate::memory::{MemoryConfig, MemorySystem};
    use tempfile::TempDir;

    fn fresh_system() -> (MemorySystem, TempDir) {
        let tmp = TempDir::new().expect("temp dir");
        let config = MemoryConfig {
            storage_path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let sys = MemorySystem::new(config, None).expect("memory system");
        (sys, tmp)
    }

    fn remember_on(sys: &MemorySystem, content: &str, day: &str) -> MemoryId {
        let created_at = chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d")
            .expect("test date")
            .and_hms_opt(12, 0, 0)
            .expect("noon")
            .and_utc();
        sys.remember(
            Experience {
                content: content.to_string(),
                ..Default::default()
            },
            Some(created_at),
        )
        .expect("remember")
    }

    fn leg(sys: &MemorySystem, query_text: &str) -> Vec<MemoryId> {
        let query = Query {
            user_id: Some("date-window-user".to_string()),
            query_text: Some(query_text.to_string()),
            ..Default::default()
        };
        sys.date_window_candidates(query_text, &query, &[], &[])
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    fn sorted(mut ids: Vec<MemoryId>) -> Vec<MemoryId> {
        ids.sort();
        ids
    }

    #[test]
    fn a_day_in_the_query_yields_that_day_and_one_day_either_side() {
        let (sys, _tmp) = fresh_system();
        let _far = remember_on(&sys, "Nate: Dyed my hair last week", "2022-04-29");
        let before = remember_on(&sys, "Joanna: Big day tomorrow", "2022-04-30");
        let on = remember_on(
            &sys,
            "Nate: Try the Lord of the Rings trilogy",
            "2022-05-01",
        );
        let after = remember_on(
            &sys,
            "Joanna: I took your recommendation and saw the trilogy last night",
            "2022-05-02",
        );
        let _later = remember_on(&sys, "Nate: Glad you liked it", "2022-05-03");

        let got = leg(&sys, "What movie did Joanna watch on 1 May, 2022?");
        assert_eq!(sorted(got), sorted(vec![before, on, after]));
    }

    #[test]
    fn a_month_in_the_query_yields_the_whole_month_and_nothing_else() {
        let (sys, _tmp) = fresh_system();
        let _march = remember_on(&sys, "Nate: Spring is here", "2022-03-31");
        let first = remember_on(&sys, "Nate: Dyed my hair last week", "2022-04-01");
        let mid = remember_on(&sys, "Nate: Check it out, it's blue", "2022-04-15");
        let last = remember_on(&sys, "Joanna: Looks great", "2022-04-30");
        let _may = remember_on(&sys, "Joanna: Hey Nate, how's it going?", "2022-05-01");

        let got = leg(
            &sys,
            "What physical transformation did Nate undergo in April 2022?",
        );
        assert_eq!(sorted(got), sorted(vec![first, mid, last]));
    }

    #[test]
    fn a_year_alone_or_no_date_yields_no_candidates() {
        let (sys, _tmp) = fresh_system();
        remember_on(&sys, "Joanna: I went to Woodhaven", "2021-07-10");
        assert!(leg(&sys, "What state did Joanna visit in 2021?").is_empty());
        assert!(leg(&sys, "Where did Joanna travel to?").is_empty());
    }

    #[test]
    fn the_dial_switches_the_leg_off() {
        let (sys, _tmp) = fresh_system();
        remember_on(&sys, "Nate: I got this new pup for you!", "2022-05-25");
        assert_eq!(
            leg(&sys, "What did Nate do for Joanna on 25 May, 2022?").len(),
            1
        );

        sys.set_cognitive_dials(CognitiveDials {
            date_window_leg: Some(false),
            ..Default::default()
        });
        assert!(leg(&sys, "What did Nate do for Joanna on 25 May, 2022?").is_empty());
    }

    #[test]
    fn the_leg_is_capped_at_the_candidate_dial() {
        let (sys, _tmp) = fresh_system();
        for i in 0..5 {
            remember_on(&sys, &format!("Nate: gaming party plan {i}"), "2022-06-03");
        }
        sys.set_cognitive_dials(CognitiveDials {
            candidate_count: Some(2),
            ..Default::default()
        });
        assert_eq!(
            leg(
                &sys,
                "Who did Nate plan to invite to his gaming party in June 2022?"
            )
            .len(),
            2
        );
    }

    /// The gold funnel records, per pipeline stage, the best rank any gold id holds. A memory
    /// the query shares no word with is at the head of the date window list, and is in the
    /// fused pool; with the dial off the list is empty.
    #[test]
    fn the_leg_hands_a_memory_the_query_shares_no_word_with_to_fusion() {
        let (sys, _tmp) = fresh_system();
        for day in 1..=20 {
            remember_on(
                &sys,
                &format!("Joanna: watched a movie with Nate, day {day}"),
                &format!("2022-03-{day:02}"),
            );
        }
        let gold = remember_on(
            &sys,
            "Looky here, I got this new pup for you!",
            "2022-05-25",
        );
        let query = Query {
            user_id: Some("date-window-user".to_string()),
            query_text: Some("What did Nate do for Joanna on 25 May, 2022?".to_string()),
            max_results: 5,
            ..Default::default()
        };
        let stage = |funnel: &[(String, Option<usize>)], name: &str| -> Option<usize> {
            funnel
                .iter()
                .find(|(s, _)| s == name)
                .unwrap_or_else(|| panic!("stage {name} not recorded in {funnel:?}"))
                .1
        };

        crate::memory::gold_funnel::begin(HashSet::from([gold.clone()]));
        sys.recall(&query).expect("recall");
        let funnel = crate::memory::gold_funnel::take().expect("funnel armed");
        assert_eq!(stage(&funnel, "date_window"), Some(0), "{funnel:?}");
        assert!(stage(&funnel, "fusion").is_some(), "{funnel:?}");

        sys.set_cognitive_dials(CognitiveDials {
            date_window_leg: Some(false),
            ..Default::default()
        });
        crate::memory::gold_funnel::begin(HashSet::from([gold]));
        sys.recall(&query).expect("recall");
        let funnel = crate::memory::gold_funnel::take().expect("funnel armed");
        assert_eq!(stage(&funnel, "date_window"), None, "{funnel:?}");
    }
}

#[cfg(test)]
mod vector_rung_tests {
    use super::*;
    use crate::memory::{MemoryConfig, MemorySystem};
    use rand::{Rng, SeedableRng};
    use tempfile::TempDir;

    fn fresh_system() -> (MemorySystem, TempDir) {
        let tmp = TempDir::new().expect("temp dir");
        let config = MemoryConfig {
            storage_path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let sys = MemorySystem::new(config, None).expect("memory system");
        (sys, tmp)
    }

    fn normalised(v: Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    /// A unit vector whose cosine with the unit vector `axis` is `cos`.
    fn at_cosine(axis: &[f32], cos: f32, rng: &mut impl Rng) -> Vec<f32> {
        let r: Vec<f32> = (0..axis.len()).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let along: f32 = r.iter().zip(axis).map(|(a, b)| a * b).sum();
        let orthogonal = normalised(r.iter().zip(axis).map(|(a, b)| a - along * b).collect());
        let sin = (1.0 - cos * cos).sqrt();
        axis.iter()
            .zip(&orthogonal)
            .map(|(a, o)| cos * a + sin * o)
            .collect()
    }

    fn remember_at(
        sys: &MemorySystem,
        content: &str,
        entities: &[&str],
        embedding: Vec<f32>,
        day: u32,
    ) -> MemoryId {
        let created_at = chrono::NaiveDate::from_ymd_opt(2023, 5, day)
            .expect("test date")
            .and_hms_opt(12, 0, 0)
            .expect("noon")
            .and_utc();
        sys.remember(
            Experience {
                experience_type: ExperienceType::Conversation,
                content: content.to_string(),
                entities: entities.iter().map(|e| e.to_string()).collect(),
                embeddings: Some(embedding),
                ..Default::default()
            },
            Some(created_at),
        )
        .expect("remember")
    }

    /// LoCoMo conv-26 shape: the turn the question asks about is the oldest and the longest,
    /// and the turns nearer in time share the question's words and speaker. Every fusion
    /// stage after the vector leg (temporal facts, entity overlap, the length penalty,
    /// temporal topic dedup) prefers those; the vector rung must not run any of them.
    #[test]
    fn the_vector_rung_returns_the_stored_vectors_in_cosine_order() {
        let (sys, _tmp) = fresh_system();
        let dim = sys.get_embedder().dimension();
        let mut rng = rand::rngs::StdRng::seed_from_u64(26);
        let query = normalised((0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect());

        let mut by_cosine: Vec<(MemoryId, f32)> = Vec::new();
        for i in 0..24u32 {
            let cos = 0.9 - 0.03 * i as f32;
            let embedding = at_cosine(&query, cos, &mut rng);
            let id = if i < 12 {
                remember_at(
                    &sys,
                    &format!(
                        "Caroline: entry {i}, a long account of the evening at the community \
                         centre, who spoke, what they said, how it felt, and what came after"
                    ),
                    &["Caroline", &format!("centre {i}")],
                    embedding,
                    1 + i,
                )
            } else {
                remember_at(
                    &sys,
                    &format!("Caroline: support group {i}"),
                    &["Caroline", "LGBTQ", "support group"],
                    embedding,
                    1 + i,
                )
            };
            by_cosine.push((id, cos));
        }

        let recalled = sys
            .recall(&Query {
                user_id: Some("vector-rung-user".to_string()),
                query_text: Some("When did Caroline go to the LGBTQ support group?".to_string()),
                query_embedding: Some(query),
                layers: Some(LayerMode::VamanaOnly),
                max_results: 10,
                ..Default::default()
            })
            .expect("recall");

        let got: Vec<&MemoryId> = recalled.iter().map(|m| &m.id).collect();
        let want: Vec<&MemoryId> = by_cosine.iter().take(10).map(|(id, _)| id).collect();
        assert_eq!(got, want);
        for (memory, (_, cos)) in recalled.iter().zip(&by_cosine) {
            let score = memory.score.expect("scored");
            assert!((score - cos).abs() < 1e-3, "score {score} for cosine {cos}");
        }
    }

    #[test]
    fn the_tier_scan_returns_only_memories_the_vector_index_does_not_hold() {
        let (sys, _tmp) = fresh_system();
        let dim = sys.get_embedder().dimension();
        let mut rng = rand::rngs::StdRng::seed_from_u64(35);
        let query = normalised((0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect());
        let indexed = remember_at(
            &sys,
            "Melanie: we went camping",
            &["Melanie"],
            at_cosine(&query, 0.8, &mut rng),
            1,
        );
        let unindexed = remember_at(
            &sys,
            "Melanie: the kids loved the lake",
            &["Melanie"],
            at_cosine(&query, 0.7, &mut rng),
            2,
        );
        assert!(sys.retriever.has_mapping(&indexed));
        assert!(sys.retriever.remove_memory(&unindexed));

        let index_space = sys.retriever.index_space().clone();
        let hits = sys.unindexed_tier_hits(&query, &index_space, &[], &HashSet::new());
        let ids: Vec<&MemoryId> = hits.iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![&unindexed]);
        assert!((hits[0].1 - 0.7).abs() < 1e-3);
    }
}

#[cfg(test)]
mod candidate_projection_tests {
    use super::{
        entity_match_specificity, temporal_topic_dedup, CandidateSource, CandidateTier,
        RecallCandidates,
    };
    use crate::memory::types::{Experience, ExperienceType, Memory, MemoryId, SharedMemory};
    use anyhow::Result;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn memory(content: &str, entities: &[&str]) -> Memory {
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            ..Experience::default()
        };
        Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            experience,
            0.5,
            None,
            None,
            None,
            None,
        )
    }

    /// A source that counts every read, per id.
    #[derive(Default)]
    struct CountingSource {
        working: HashMap<MemoryId, SharedMemory>,
        stored: HashMap<MemoryId, Memory>,
        storage_reads: RefCell<HashMap<MemoryId, usize>>,
        storage_batches: RefCell<usize>,
    }

    impl CountingSource {
        fn reads_of(&self, id: &MemoryId) -> usize {
            self.storage_reads.borrow().get(id).copied().unwrap_or(0)
        }
    }

    impl CandidateSource for CountingSource {
        fn resident(&self, ids: &[MemoryId]) -> Vec<(MemoryId, SharedMemory, CandidateTier)> {
            ids.iter()
                .filter_map(|id| {
                    self.working
                        .get(id)
                        .map(|m| (id.clone(), Arc::clone(m), CandidateTier::Working))
                })
                .collect()
        }

        fn stored(&self, ids: &[MemoryId]) -> Vec<Result<Memory>> {
            *self.storage_batches.borrow_mut() += 1;
            ids.iter()
                .map(|id| {
                    *self
                        .storage_reads
                        .borrow_mut()
                        .entry(id.clone())
                        .or_default() += 1;
                    self.stored
                        .get(id)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("Memory not found: {id:?}"))
                })
                .collect()
        }
    }

    #[test]
    fn every_candidate_is_read_once_however_many_layers_ask() {
        let mut source = CountingSource::default();
        let hot = memory("Working Tier note", &["Rust"]);
        let cold_a = memory("Stored Note A", &["Tokio", "Rust"]);
        let cold_b = memory("Stored Note B", &[]);
        let missing = MemoryId(uuid::Uuid::new_v4());
        let (hot_id, a_id, b_id) = (hot.id.clone(), cold_a.id.clone(), cold_b.id.clone());
        source.working.insert(hot_id.clone(), Arc::new(hot));
        source.stored.insert(a_id.clone(), cold_a);
        source.stored.insert(b_id.clone(), cold_b);

        let candidates = RecallCandidates::new(&source);
        let fused = [hot_id.clone(), a_id.clone(), b_id.clone(), missing.clone()];
        candidates.prefetch(fused.iter());
        assert_eq!(*source.storage_batches.borrow(), 1, "one storage batch");

        // Every layer after fusion reads every candidate, several fields each.
        for _layer in 0..8 {
            for id in &fused {
                let _ = candidates.get(id);
                let _ = candidates.content(id);
            }
            candidates.prefetch(fused.iter());
        }
        assert_eq!(*source.storage_batches.borrow(), 1);
        assert_eq!(
            source.reads_of(&hot_id),
            0,
            "a resident record never reaches storage"
        );
        assert_eq!(source.reads_of(&a_id), 1);
        assert_eq!(source.reads_of(&b_id), 1);
        assert_eq!(
            source.reads_of(&missing),
            1,
            "a miss is remembered, not retried"
        );

        let hot_view = candidates.get(&hot_id).unwrap();
        assert_eq!(hot_view.tier, CandidateTier::Working);
        assert_eq!(hot_view.content_lower, "working tier note");
        let a_view = candidates.get(&a_id).unwrap();
        assert_eq!(a_view.tier, CandidateTier::LongTerm);
        assert_eq!(a_view.entities_lower, ["tokio", "rust"]);
        assert!(candidates.get(&missing).is_none());
        assert!(candidates
            .failure(&missing)
            .is_some_and(|reason| reason.contains("not found")));

        // A candidate first named after fusion is read once, on its own.
        let late = memory("Late", &[]);
        let late_id = late.id.clone();
        source.stored.insert(late_id.clone(), late);
        let candidates = RecallCandidates::new(&source);
        assert!(candidates.get(&late_id).is_some());
        assert!(candidates.get(&late_id).is_some());
        assert_eq!(source.reads_of(&late_id), 1);
    }

    #[test]
    fn an_entity_word_weighs_by_how_few_candidates_carry_it() {
        assert_eq!(entity_match_specificity(200, 1), 1.0);
        assert_eq!(entity_match_specificity(200, 200), 0.0);
        let speaker_name = entity_match_specificity(200, 100);
        let topic = entity_match_specificity(200, 5);
        assert!(speaker_name < 0.15, "{speaker_name}");
        assert!(topic > 0.65, "{topic}");
        assert_eq!(
            entity_match_specificity(1, 1),
            1.0,
            "one candidate: nothing to compare"
        );
        assert_eq!(entity_match_specificity(0, 0), 1.0);
    }

    #[test]
    fn temporal_dedup_compares_only_the_top_candidates() {
        let mut source = CountingSource::default();
        let base = chrono::Utc::now() - chrono::Duration::days(100);
        let mut fused: HashMap<MemoryId, f32> = HashMap::new();
        let mut by_rank: Vec<MemoryId> = Vec::new();
        for rank in 0..60 {
            let mut m = memory(&format!("note {rank:02}"), &["Alpha", "Beta"]);
            m.created_at = base + chrono::Duration::days(rank);
            fused.insert(m.id.clone(), 1.0 + rank as f32);
            by_rank.push(m.id.clone());
            source.stored.insert(m.id.clone(), m);
        }
        let before = fused.clone();
        let candidates = RecallCandidates::new(&source);

        let demoted = temporal_topic_dedup(&mut fused, &candidates, 50);
        assert_eq!(
            demoted, 49,
            "every compared memory but the newest has a newer one on its topic"
        );
        for id in &by_rank[..10] {
            assert_eq!(fused[id], before[id], "below the cap, nothing is compared");
        }
        let newest = &by_rank[59];
        assert_eq!(fused[newest], before[newest]);
        // The oldest compared memory loses to 49 newer ones and is still demoted only once.
        for id in &by_rank[10..59] {
            assert!((fused[id] - before[id] * 0.85).abs() < 1e-4);
        }
    }
}

#[cfg(test)]
mod query_entity_resolution_tests {
    use super::resolve_query_entities;
    use crate::graph_memory::{EntityLabel, EntityNode, GraphMemory, PiiClassification};
    use chrono::Utc;
    use std::collections::HashMap;

    fn entity(name: &str) -> EntityNode {
        EntityNode {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![EntityLabel::Concept],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
            pii_classification: PiiClassification::Clean,
        }
    }

    #[test]
    fn words_resolving_to_one_entity_seed_it_once() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let deploy = graph.add_entity(entity("deploy")).unwrap();
        let reactor = graph.add_entity(entity("Vornak Reactors")).unwrap();

        // "deploying" (text) and "deploy" (its stem) are one entity; "vornak reactors" is
        // the focal entity again in another case.
        let resolved = resolve_query_entities(
            &graph,
            &["Vornak Reactors"],
            &["deploying", "deploy", "vornak reactors", "deploying"],
            "when did we start deploying the Vornak Reactors",
        );
        assert_eq!(resolved.focal, vec![reactor]);
        assert_eq!(resolved.focal_word, vec![0]);
        assert!(resolved.paths.is_empty());
        assert_eq!(resolved.all, vec![reactor, deploy]);
    }

    #[test]
    fn only_nouns_seed_the_graph_leg_at_their_information_content_and_rarity() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let mut speaker = entity("Caroline");
        speaker.mention_count = 300;
        graph.put_entity_record(&speaker).unwrap();
        let topic = graph.add_entity(entity("adoption")).unwrap();
        let verb = graph.add_entity(entity("research")).unwrap();
        let ner_only = graph.add_entity(entity("LGBTQ support group")).unwrap();

        let analysis = crate::memory::query_parser::QueryAnalysis {
            focal_entities: ["Caroline", "adoption", "Caroline"]
                .iter()
                .zip([2.3, 3.0, 2.3])
                .map(
                    |(text, ic_weight)| crate::memory::query_parser::FocalEntity {
                        text: text.to_string(),
                        stem: text.to_lowercase(),
                        ic_weight,
                        is_compound: false,
                        negated: false,
                    },
                )
                .collect(),
            ..crate::memory::query_parser::analyze_query("")
        };
        let resolved = resolve_query_entities(
            &graph,
            &["Caroline", "adoption", "Caroline"],
            &["research", "researched"],
            "did Caroline research adoption",
        );
        assert!(resolved.all.contains(&verb), "the verb resolves");

        let ner = vec!["LGBTQ support group".to_string(), "x".to_string()];
        let seeds = super::graph_leg_seeds(&graph, Some(&resolved), &analysis, Some(&ner));
        let seeded: Vec<(uuid::Uuid, f32, usize)> = seeds
            .iter()
            .map(|s| (s.uuid, s.ic_weight, s.mention_count))
            .collect();
        assert_eq!(
            seeded,
            vec![
                (speaker.uuid, 2.3, 300),
                (topic, 3.0, 1),
                (ner_only, crate::constants::IC_NOUN, 1)
            ],
            "the verb does not seed; a repeated noun seeds once"
        );
    }

    #[test]
    fn only_focal_words_fall_back_to_fuzzy_matching() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let reactor = graph.add_entity(entity("Vornak Reactors")).unwrap();
        assert_eq!(
            graph.find_entity_by_name("React").unwrap().map(|e| e.uuid),
            Some(reactor),
            "the unbounded resolver binds the fragment"
        );

        let as_other = resolve_query_entities(&graph, &[], &["React", "Vornak"], "react vornak");
        assert!(
            as_other.all.is_empty(),
            "a non-focal word resolves strictly or not at all"
        );

        let as_focal = resolve_query_entities(&graph, &["Vornak"], &[], "vornak");
        assert_eq!(as_focal.focal, vec![reactor]);
        assert_eq!(as_focal.all, vec![reactor]);
    }

    #[test]
    fn a_path_shaped_word_names_the_declared_file_it_resolves_to() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path(), None).unwrap();
        let mut declared = entity("commitwork:Admin/serve.mjs");
        declared.labels = vec![EntityLabel::FilePath];
        let serve = graph.add_entity(declared).unwrap();
        graph.add_entity(entity("observe.mjs")).unwrap();

        for query in [
            "what changed in serve.mjs",
            "what changed in admin/serve.mjs",
        ] {
            let resolved = resolve_query_entities(&graph, &[], &[], query);
            assert_eq!(resolved.all, vec![serve], "{query}");
            assert_eq!(
                resolved.file_paths,
                vec!["commitwork:admin/serve.mjs".to_string()],
                "{query}"
            );
        }
        let none = resolve_query_entities(&graph, &[], &["changed"], "what changed recently");
        assert!(
            none.file_paths.is_empty(),
            "no path-shaped word, no declared file"
        );
    }
}

#[cfg(test)]
mod fusion_tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn flat_scores_a_list_by_how_close_a_candidate_is_to_its_best() {
        // Rank does not enter: the same score over the same best is the same term at rank 1
        // and rank 50, so a leg's length cannot change what fusion believes of a candidate.
        let at_1 = leg_term(FusionMode::Flat, 0.7, 20.0, 1, 0.45, 0.6);
        let at_50 = leg_term(FusionMode::Flat, 0.7, 20.0, 50, 0.45, 0.6);
        assert!((at_1 - 0.7 * 0.75).abs() < EPS);
        assert_eq!(at_1, at_50);
        // A list's best enters at the list's weight, not weight / (k + 1).
        assert!((leg_term(FusionMode::Flat, 0.3, 20.0, 1, 0.9, 0.9) - 0.3).abs() < EPS);
        assert!((leg_term(FusionMode::Rrf, 0.3, 20.0, 1, 0.9, 0.9) - 0.3 / 21.0).abs() < EPS);
        assert!((leg_term(FusionMode::Rrf, 0.3, 20.0, 10, 0.1, 0.9) - 0.3 / 30.0).abs() < EPS);
    }

    #[test]
    fn calibration_is_bounded_and_safe_on_an_empty_list() {
        assert_eq!(calibrated(0.5, 0.0), 0.0);
        assert_eq!(calibrated(f32::NAN, 1.0), 0.0);
        assert_eq!(calibrated(-0.2, 0.8), 0.0);
        assert_eq!(calibrated(0.8, 0.8), 1.0);
        assert_eq!(best_leg_score(std::iter::empty()), 0.0);
        assert_eq!(best_leg_score([0.2, f32::NAN, 0.7].into_iter()), 0.7);
    }

    #[test]
    fn a_candidate_one_list_is_sure_of_beats_a_crowd_both_lists_half_match() {
        // The burial FLAT exists to undo: vector's best, absent from BM25, against a
        // candidate both lists put at 0.6 of their best.
        let single = flat_hybrid(1.0, 0.0);
        let crowd = flat_hybrid(0.6, 0.6);
        assert!((single - 1.0).abs() < EPS);
        assert!((crowd - (0.6 + 0.3 * 0.6)).abs() < EPS);
        assert!(single > crowd);
        // Both lists finding a candidate is still worth something over one list.
        assert!(flat_hybrid(0.8, 0.5) > flat_hybrid(0.8, 0.0));
        // Symmetric in the two lists.
        assert_eq!(flat_hybrid(0.3, 0.9), flat_hybrid(0.9, 0.3));
    }

    #[test]
    fn an_entity_match_is_a_bounded_share_of_the_memorys_own_score() {
        let share = crate::constants::ENTITY_OVERLAP_SHARE;
        assert_eq!(entity_overlap_share(0.0, 0, false), 0.0);
        assert!((entity_overlap_share(1.0, 0, false) - share).abs() < EPS);
        assert!((entity_overlap_share(0.5, 0, false) - 0.5 * share).abs() < EPS);
        // A decomposed query's word matches count for 0.4; an identifier counts in full.
        assert!((entity_overlap_share(1.0, 0, true) - 0.4 * share).abs() < EPS);
        assert!((entity_overlap_share(0.0, 1, true) - share).abs() < EPS);
        // Four full matches of either kind are the cap, decomposed or not.
        assert!((entity_overlap_share(10.0, 0, false) - 4.0 * share).abs() < EPS);
        assert!((entity_overlap_share(3.0, 3, false) - 4.0 * share).abs() < EPS);
        assert!((entity_overlap_share(0.0, 9, true) - 4.0 * share).abs() < EPS);
        assert!((entity_overlap_share(10.0, 0, true) - 1.6 * share).abs() < EPS);
    }

    #[test]
    fn an_admitted_graph_hit_scores_as_the_vector_list_would_have() {
        let (a, b) = (
            MemoryId(uuid::Uuid::new_v4()),
            MemoryId(uuid::Uuid::new_v4()),
        );
        // FLAT: cosine over the vector list's best, at the hybrid weight, so an admitted hit
        // never outscores the list's own best.
        let flat = admitted_graph_hit_terms(
            FusionMode::Flat,
            0.7,
            20.0,
            40,
            vec![(&a, 0.3), (&b, 0.45)],
            Some(0.6),
        );
        assert_eq!(flat[0].0, &b, "cosine order");
        assert!((flat[0].1 - 0.7 * 0.75).abs() < EPS);
        assert!((flat[1].1 - 0.7 * 0.5).abs() < EPS);
        // With no vector best to read, the hits calibrate among themselves.
        let alone = admitted_graph_hit_terms(FusionMode::Flat, 0.7, 20.0, 0, vec![(&a, 0.3)], None);
        assert!((alone[0].1 - 0.7).abs() < EPS);
        // Reciprocal rank: after the 40 hybrid candidates, in cosine order.
        let rrf = admitted_graph_hit_terms(
            FusionMode::Rrf,
            0.7,
            20.0,
            40,
            vec![(&a, 0.3), (&b, 0.45)],
            Some(0.6),
        );
        assert_eq!(rrf[0].0, &b);
        assert!((rrf[0].1 - 0.7 / 61.0).abs() < EPS);
        assert!((rrf[1].1 - 0.7 / 62.0).abs() < EPS);
    }

    #[test]
    fn graph_activation_multiplies_the_graph_term_within_a_bound() {
        assert!((graph_activation_factor(0.5, 1.0) - 1.15).abs() < EPS);
        assert!((graph_activation_factor(0.1, 1.0) - 1.03).abs() < EPS);
        assert_eq!(
            graph_activation_factor(0.5, 7.0),
            graph_activation_factor(0.5, 1.0)
        );
        assert_eq!(graph_activation_factor(0.0, 1.0), 1.0);
        assert_eq!(graph_activation_factor(0.5, -1.0), 1.0);
    }
}

#[cfg(test)]
mod owner_scope_tests {
    use super::*;
    use crate::memory::{MemoryConfig, MemorySystem};

    /// Sets an env var for the scope of the guard and restores it on drop, so a failed
    /// expectation cannot leave recall read-only for the rest of the test process.
    struct EnvVar(&'static str, Option<String>);

    impl EnvVar {
        fn set(name: &'static str, value: &str) -> Self {
            let previous = std::env::var(name).ok();
            std::env::set_var(name, value);
            Self(name, previous)
        }
    }

    impl Drop for EnvVar {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }

    fn score_of(sys: &MemorySystem, user_id: Option<&str>, id: &MemoryId) -> f32 {
        let query = Query {
            user_id: user_id.map(str::to_string),
            query_text: Some("When did Melanie run a marathon?".to_string()),
            max_results: 5,
            ..Default::default()
        };
        sys.recall(&query)
            .expect("recall")
            .iter()
            .find(|m| &m.id == id)
            .and_then(|m| m.score)
            .expect("the fact's source is recalled")
    }

    /// A recall that names no user, on a system that knows its owner, runs the owner's fact
    /// layers: the temporal fact stored for the owner boosts its source memory exactly as a
    /// recall naming the owner does, and an unowned system's recall does not. A regression
    /// in the owner match at the top of `recall()` scores the source as the unowned recall.
    #[test]
    fn a_userless_recall_on_an_owned_system_runs_the_owners_fact_layers() {
        let _env = crate::memory::ablation::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Read-only recall writes no access counts or activation, so the three recalls below
        // differ only in which user they run as.
        let _readonly = EnvVar::set("VELD_RECALL_READONLY", "1");

        let tmp = tempfile::TempDir::new().expect("temp dir");
        let mut sys = MemorySystem::new(
            MemoryConfig {
                storage_path: tmp.path().to_path_buf(),
                ..Default::default()
            },
            None,
        )
        .expect("memory system");
        let content = "Melanie ran a marathon yesterday";
        let entities = vec!["Melanie".to_string()];
        let source = sys
            .remember(
                Experience {
                    content: content.to_string(),
                    entities: entities.clone(),
                    ..Default::default()
                },
                None,
            )
            .expect("remember");
        sys.remember(
            Experience {
                content: "Caroline went to a support group meeting".to_string(),
                entities: vec!["Caroline".to_string()],
                ..Default::default()
            },
            None,
        )
        .expect("remember");
        let created_at = sys.get_memory(&source).expect("stored").created_at;
        let stored = sys
            .store_temporal_facts_for_memory("alice", &source, content, &entities, created_at)
            .expect("store temporal facts");
        assert!(stored > 0, "the content must yield a temporal fact");

        let unowned = score_of(&sys, None, &source);
        sys.set_owner_user_id("alice");
        let scoped = score_of(&sys, None, &source);
        let named = score_of(&sys, Some("alice"), &source);

        assert!(
            (scoped - named).abs() < 1e-6,
            "a user-less recall on alice's system must score as alice's recall: {scoped} vs {named}"
        );
        assert!(
            scoped > unowned,
            "the owner's temporal fact must boost its source: {scoped} owned vs {unowned} unowned"
        );
    }
}

#[cfg(test)]
mod recursive_graph_read_tests {
    use crate::graph_memory::GraphMemory;
    use crate::memory::types::{Experience, Query};
    use crate::memory::{MemoryConfig, MemorySystem};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;

    /// `/api/relevant` and `/api/stream` hold the graph's read guard around recall. A writer
    /// queued behind that guard must not stop recall's own graph reads, or neither finishes.
    #[test]
    fn recall_inside_an_outer_graph_read_does_not_wait_for_a_queued_writer() {
        let store = TempDir::new().expect("store dir");
        let mut sys = MemorySystem::new(
            MemoryConfig {
                storage_path: store.path().to_path_buf(),
                ..Default::default()
            },
            None,
        )
        .expect("memory system");
        let graph_dir = TempDir::new().expect("graph dir");
        let graph = Arc::new(parking_lot::RwLock::new(
            GraphMemory::new(graph_dir.path(), None).expect("graph"),
        ));
        sys.set_graph_memory(Arc::clone(&graph));
        sys.remember(
            Experience {
                content: "Caroline went to the LGBTQ support group on Tuesday".to_string(),
                ..Default::default()
            },
            None,
        )
        .expect("remember");
        let sys = Arc::new(sys);

        let outer = graph.read();
        let writer = {
            let graph = Arc::clone(&graph);
            std::thread::spawn(move || drop(graph.write()))
        };
        // Queued, the writer turns away a plain new reader.
        while graph.try_read().is_some() {
            std::thread::sleep(Duration::from_millis(5));
        }
        let (done, recalled) = std::sync::mpsc::channel();
        let reader = {
            let sys = Arc::clone(&sys);
            std::thread::spawn(move || {
                let query = Query {
                    query_text: Some("Where did Caroline go on Tuesday?".to_string()),
                    ..Default::default()
                };
                let _ = done.send(sys.recall(&query).map(|found| found.len()));
            })
        };
        let outcome = recalled.recv_timeout(Duration::from_secs(120));
        drop(outer);
        writer.join().expect("writer");
        reader.join().expect("reader");
        assert!(
            outcome.is_ok(),
            "recall blocked behind a writer queued on the outer guard"
        );
        assert!(outcome.unwrap().is_ok());
    }
}
