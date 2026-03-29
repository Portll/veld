//! Memory forget/deletion operations.
//!
//! Implements all forget criteria (by ID, age, importance, pattern, tags,
//! date range, type, and full erasure for GDPR compliance). Each variant
//! cleans up all secondary indices (vector, BM25, graph, interference)
//! and updates in-memory stats atomically.

use anyhow::Result;

use super::replay;
use super::storage;
use super::types::{ExperienceType, ForgetCriteria, MemoryId};

impl super::MemorySystem {
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
        // Deletion order: graph -> long-term -> session -> working -> stats
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
                    // Continue -- graph cleanup is best-effort for GDPR
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

        // Step 3: Clear session memory (ephemeral -- lost on restart anyway)
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

        // Step 4: Clear working memory (ephemeral -- lost on restart anyway)
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

        // Step 6: Clear semantic facts (GDPR -- knowledge derived from memories)
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
}
