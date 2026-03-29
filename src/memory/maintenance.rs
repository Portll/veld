//! Memory maintenance and consolidation routines.
//!
//! Contains periodic maintenance (tier promotion, activation decay, replay,
//! fact extraction) and the consolidation introspection API (event recording,
//! report generation).

use anyhow::Result;
use std::collections::HashSet;

use crate::constants::{
    ORPHAN_COMPENSATORY_BOOST, POTENTIATION_ACCESS_THRESHOLD, POTENTIATION_MAINTENANCE_BOOST,
    TIER_PROMOTION_SESSION_AGE_SECS, TIER_PROMOTION_SESSION_IMPORTANCE,
    TIER_PROMOTION_WORKING_AGE_SECS, TIER_PROMOTION_WORKING_IMPORTANCE,
};
use crate::embeddings::Embedder;
use crate::memory::compression::{SemanticConsolidator, SemanticFact};
use crate::memory::introspection::{
    ConsolidationEvent, ConsolidationEventBuffer, ConsolidationReport, StrengtheningReason,
};
use crate::memory::pattern_detection;
use crate::memory::replay;
use crate::memory::types::{
    EdgePromotionBoost, MaintenanceResult, Memory, MemoryId, SharedMemory,
};

impl super::MemorySystem {
    // =========================================================================
    // TIER PROMOTION & CONSOLIDATION
    // =========================================================================

    /// Consolidate memories based on Cowan's model (importance + time, not size)
    ///
    /// Tier promotion criteria:
    /// - Working -> Session: importance >= 0.4 AND age >= 5 minutes
    /// - Session -> LongTerm: importance >= 0.6 AND age >= 1 hour
    pub(crate) fn consolidate_if_needed(&self) -> Result<()> {
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

        let mut threshold = base_threshold;

        // 1. Graph health adjustment (existing)
        if let Some(graph) = &self.graph_memory {
            if !memory.entity_refs.is_empty() {
                let graph_guard = graph.read();
                let mut l2_plus_count = 0usize;

                for entity_ref in &memory.entity_refs {
                    if let Ok(edges) =
                        graph_guard.get_entity_relationships(&entity_ref.entity_id)
                    {
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
                    threshold *= 1.0 + GRAPH_HEALTH_NO_EDGES_PENALTY as f32;
                } else {
                    // Discount proportional to edge count, capped at saturation
                    let ratio =
                        (l2_plus_count as f64 / GRAPH_HEALTH_EDGE_SATURATION).min(1.0);
                    threshold *= 1.0 - (GRAPH_HEALTH_PROMOTION_DISCOUNT * ratio) as f32;
                }
            }
        }

        // 2. Feedback momentum adjustment (P2b — Berntsen functional constraints)
        // Memories with consistently positive feedback should promote faster
        // (lower threshold). Memories with negative momentum resist promotion
        // (higher threshold). This prevents "undead" memories — memories that
        // achieve high importance through access frequency alone but are
        // consistently unhelpful in retrieval.
        //
        // Adjustment range: ±15% of threshold
        // Positive momentum (proven helpful): threshold reduced by up to 15%
        // Negative momentum (frequently ignored): threshold increased by up to 15%
        if let Some(ref fb_store) = self.feedback_store {
            if let Some(fm) = fb_store.read().get_momentum(&memory.id) {
                let momentum = fm.ema_with_decay();
                // Clamp to ±1.0, then scale to ±15% adjustment
                let adjustment = momentum.clamp(-1.0, 1.0) * 0.15;
                // Negative momentum → positive adjustment (higher threshold)
                // Positive momentum → negative adjustment (lower threshold)
                threshold *= 1.0 - adjustment;
            }
        }

        threshold
    }

    /// Apply importance boosts to memories whose edges were promoted (Direction 1).
    ///
    /// When an edge promotes from L1->L2 or L2->L3, the memories involved get
    /// a small importance boost, reflecting that they participate in a consolidating
    /// relationship. Uses interior mutability -- `set_importance` works through Arc.
    pub fn apply_edge_promotion_boosts(
        &self,
        boosts: &[EdgePromotionBoost],
    ) -> Result<usize> {
        let mut applied = 0;

        for boost in boosts {
            let memory_id = match uuid::Uuid::parse_str(&boost.memory_id) {
                Ok(uuid) => MemoryId(uuid),
                Err(_) => continue,
            };

            // Search across tiers: working -> session -> long-term
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
        if orphaned_entity_ids.is_empty() {
            return Ok(0);
        }

        let orphaned_set: HashSet<&str> =
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
    pub(crate) fn update_access_count(&self, memory_id: &MemoryId) -> Result<()> {
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
    pub(crate) fn update_access_count_instrumented(
        &self,
        memory: &SharedMemory,
        reason: StrengtheningReason,
    ) {
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

    // =========================================================================
    // PERIODIC MAINTENANCE
    // =========================================================================

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

        // 2.45 FIX-R1: Process expired reconsolidation shadows
        // Close labile windows and decay activation back to resting state.
        // Working memory pattern (bursty access) keeps activation high.
        // Reference: Nader et al. (2000) — reconsolidation window closure.
        let mut reconsolidation_applied = 0_u32;
        {
            let expired: Vec<types::ReconsolidationShadow> = {
                let mut shadows = self.reconsolidation_shadows.write();
                let expired_keys: Vec<types::MemoryId> = shadows
                    .iter()
                    .filter(|(_, s)| {
                        s.expires_at <= now
                            && s.consecutive_retrieval_count
                                < crate::constants::RECONSOLIDATION_WORKING_MEMORY_THRESHOLD
                    })
                    .map(|(k, _)| k.clone())
                    .collect();

                expired_keys
                    .iter()
                    .filter_map(|k| shadows.remove(k))
                    .collect()
            };

            for shadow in &expired {
                // Find the memory across tiers and decay activation from labile (1.0)
                // to resting state. Bursty access patterns keep activation higher.
                let found: Option<SharedMemory> = self
                    .working_memory
                    .read()
                    .get(&shadow.memory_id)
                    .cloned()
                    .or_else(|| self.session_memory.read().get(&shadow.memory_id).cloned());

                if let Some(mem) = found {
                    let burstiness = mem.access_burstiness();
                    let resting_activation = if burstiness > 1.5 {
                        0.6 // Bursty = working memory, keep active
                    } else {
                        0.3 // Steady = long-term, decay to resting
                    };
                    mem.set_activation(resting_activation);
                    reconsolidation_applied += 1;

                    self.record_consolidation_event(ConsolidationEvent::MemoryStrengthened {
                        memory_id: shadow.memory_id.0.to_string(),
                        content_preview: format!(
                            "reconsolidated ({}x retrievals, burstiness={:.1})",
                            shadow.consecutive_retrieval_count, burstiness
                        ),
                        activation_before: 1.0,
                        activation_after: resting_activation,
                        reason: StrengtheningReason::Recalled,
                        timestamp: now,
                    });
                }
            }
        }
        if reconsolidation_applied > 0 {
            tracing::debug!(
                applied = reconsolidation_applied,
                "FIX-R1: Processed expired reconsolidation shadows"
            );
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

        // 3.7. Re-embed degraded memories (HEAVY ONLY)
        // When the circuit breaker was open, memories got hash-based fallback embeddings.
        // Now that we're in a heavy cycle, check if the embedder is healthy and re-embed.
        let mut reembedded_count = 0u32;
        if is_heavy && self.embedder.is_healthy() {
            for memory in &all_memories_for_heavy {
                if memory.experience.embedding_degraded {
                    match self.embedder.encode_with_status(&memory.experience.content) {
                        Ok((new_embedding, still_degraded)) if !still_degraded => {
                            // Update the stored memory with a real embedding
                            if let Ok(mut stored) = self.long_term_memory.get(&memory.id) {
                                stored.experience.embeddings = Some(new_embedding);
                                stored.experience.embedding_degraded = false;
                                let _ = self.long_term_memory.update(&stored);
                                reembedded_count += 1;
                            }
                        }
                        _ => {
                            // Still degraded or error — skip, will retry next heavy cycle
                        }
                    }
                    // Cap per cycle to avoid long maintenance stalls
                    if reembedded_count >= 100 {
                        tracing::info!(reembedded = reembedded_count, "Re-embed cap reached, deferring rest to next heavy cycle");
                        break;
                    }
                }
            }
            if reembedded_count > 0 {
                tracing::info!(reembedded = reembedded_count, "Re-embedded degraded memories with real embeddings");
            }
        }

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

                let consolidator = SemanticConsolidator::new();
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
                        .zip(fact_embeddings.into_iter())
                        .filter(|(fact, fact_emb)| {
                            let Some(ref fv) = fact_emb else { return true };

                            // Collect source memory content (sample up to max_sources)
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

                            // Encode sources and compute centroid
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

                    for (fact, embedding) in gated.into_iter() {
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

                        // FIX-R2: Fragment demotion — demote source memories when
                        // a quality-gated fact has been extracted from them.
                        // Reference: Ehlers & Clark (2000) — S-rep fragments should
                        // be deprioritized once a C-rep fact is consolidated.
                        let mut demoted_count = 0_u32;
                        for (fact, fact_emb) in &truly_new {
                            for source_id in &fact.source_memories {
                                let source_mem = memories.iter().find(|m| &m.id == source_id);
                                if let Some(source) = source_mem {
                                    // Similarity gate: only demote if fact faithfully
                                    // represents the source (prevents bad facts from
                                    // suppressing good fragments)
                                    let similarity = match (
                                        fact_emb.as_deref(),
                                        source.experience.embeddings.as_deref(),
                                    ) {
                                        (Some(fe), Some(se)) => {
                                            crate::similarity::cosine_similarity(fe, se)
                                        }
                                        _ => 0.0,
                                    };
                                    if similarity
                                        >= crate::constants::FRAGMENT_DEMOTION_SIMILARITY_GATE
                                    {
                                        let elab = source.elaboration_score();
                                        // Well-elaborated facts strongly demote; poorly-
                                        // elaborated facts barely demote. Uses fact
                                        // confidence as quality proxy.
                                        let quality = fact.confidence.max(elab);
                                        let demotion = 1.0
                                            - (quality
                                                * crate::constants::FRAGMENT_DEMOTION_MAX_FACTOR);
                                        source.set_fragment_demotion(demotion);
                                        // Persist the demotion
                                        if let Ok(mut stored) =
                                            self.long_term_memory.get(source_id)
                                        {
                                            stored.set_fragment_demotion(demotion);
                                            let _ = self.long_term_memory.update(&stored);
                                        }
                                        demoted_count += 1;
                                    }
                                }
                            }
                        }
                        if demoted_count > 0 {
                            tracing::debug!(
                                demoted = demoted_count,
                                "FIX-R2: Demoted source fragments after fact extraction"
                            );
                        }
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
                        ConsolidationEvent::PatternTriggeredReplay {
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

        // APP-5: Drift-adaptive abstention threshold calibration.
        // Sample memories from long-term storage and compute the 5th percentile of
        // importance × confidence. As the corpus grows, the score distribution shifts;
        // a static threshold silently over-filters. This recalibrates every maintenance
        // cycle so the system adapts to its own growth.
        if is_heavy {
            self.calibrate_abstention_threshold();
        }

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
    // DRIFT-ADAPTIVE ABSTENTION (APP-5)
    // =========================================================================

    /// Calibrate the abstention threshold from the current corpus distribution.
    ///
    /// Samples up to 200 memories from long-term storage, computes importance ×
    /// calibrated_confidence for each, and sets the threshold at the 5th percentile.
    /// This ensures the threshold tracks corpus growth: as more memories accumulate
    /// and the score distribution shifts, the threshold auto-adjusts rather than
    /// silently over-filtering.
    fn calibrate_abstention_threshold(&self) {
        use crate::memory::storage::SearchCriteria;

        // Sample memories across the importance spectrum (most representative of distribution)
        let sample = match self.long_term_memory.search(SearchCriteria::ByImportance {
            min: 0.0,
            max: 1.0,
        }) {
            Ok(mems) => mems,
            Err(e) => {
                tracing::debug!("Abstention calibration skipped: {e}");
                return;
            }
        };

        if sample.len() < 20 {
            // Too few memories to calibrate reliably — keep using static threshold
            return;
        }

        // Compute importance × confidence for each memory with sufficient observations
        let mut scores: Vec<f32> = sample
            .iter()
            .filter(|m| m.confidence_observations() > 4.0)
            .map(|m| m.importance() * m.calibrated_confidence())
            .collect();

        if scores.len() < 10 {
            // Not enough feedback-observed memories to calibrate
            return;
        }

        scores.sort_by(|a, b| a.total_cmp(b));

        // 5th percentile: low enough to avoid over-filtering, high enough to cut noise
        let p5_idx = (scores.len() as f32 * 0.05).ceil() as usize;
        let p5_idx = p5_idx.min(scores.len() - 1);
        let calibrated = scores[p5_idx];

        // Clamp: never go below half the static threshold (safety floor)
        // and never go above 2× (don't over-filter if distribution is skewed)
        let static_threshold = crate::constants::ABSTENTION_THRESHOLD;
        let clamped = calibrated.clamp(static_threshold * 0.5, static_threshold * 2.0);

        self.calibrated_abstention_threshold
            .store(clamped.to_bits(), std::sync::atomic::Ordering::Relaxed);

        tracing::info!(
            samples = scores.len(),
            p5 = calibrated,
            clamped,
            static_threshold,
            "APP-5: Calibrated abstention threshold"
        );
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
        let mut seen_keys: HashSet<(
            i64,
            std::mem::Discriminant<ConsolidationEvent>,
        )> = HashSet::new();

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
}
