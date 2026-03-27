//! Consolidation, Index Maintenance, and Backup Handlers
//!
//! Semantic consolidation for fact extraction, vector index maintenance,
//! and backup/restore operations.

use axum::{extract::State, response::Json};

use super::state::MultiUserMemoryManager;
use super::types::{
    BackupResponse, CleanupCorruptedRequest, CleanupCorruptedResponse, ConsolidateRequest,
    ConsolidateResponse, CreateBackupRequest, ListBackupsRequest, ListBackupsResponse, MemoryEvent,
    MigrateLegacyRequest, MigrateLegacyResponse, PurgeBackupsRequest, PurgeBackupsResponse,
    RebuildIndexRequest, RebuildIndexResponse, RepairIndexRequest, RepairIndexResponse,
    RestoreBackupRequest, RestoreBackupResponse, VerifyBackupRequest, VerifyBackupResponse,
    VerifyIndexRequest,
};
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory;
use crate::memory::gap_topology::{GapDetectionConfig, GapDetector};
use crate::metrics;
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// SEMANTIC CONSOLIDATION
// =============================================================================

/// Consolidate memories into semantic facts (SHO-AUD-7)
///
/// Spawns the full pipeline (fact extraction → replay → edge strengthening) as a
/// background task and returns immediately with 202 Accepted. This avoids the 60s
/// HTTP timeout killing the handler mid-flight for large memory stores.
/// Results are logged server-side and visible via `/api/consolidation/report`.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn consolidate_memories(
    State(state): State<AppState>,
    Json(req): Json<ConsolidateRequest>,
) -> Result<Json<ConsolidateResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Validate user exists before spawning background work
    let _ = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let min_support = req.min_support;
    let min_age_days = req.min_age_days;
    let state_clone = state.clone();

    // Spawn the entire pipeline as a detached background task.
    // This survives HTTP timeout cancellation — the work always completes.
    tokio::task::spawn(async move {
        let op_start = std::time::Instant::now();

        let memory = match state_clone.get_user_memory(&user_id) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(user_id = %user_id, "Consolidation: failed to get memory: {e}");
                return;
            }
        };

        // Step 1: Fact extraction
        let result = {
            let memory = memory.clone();
            let uid = user_id.clone();
            match tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                memory_guard.distill_facts(&uid, min_support, min_age_days)
            })
            .await
            {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    tracing::error!(user_id = %user_id, "Consolidation fact extraction failed: {e}");
                    return;
                }
                Err(e) => {
                    tracing::error!(user_id = %user_id, "Consolidation fact extraction panicked: {e}");
                    return;
                }
            }
        };

        // Step 2: Maintenance (replay + tier consolidation + decay)
        let decay_factor = state_clone.server_config().activation_decay_factor;
        let maintenance_result = {
            let memory = memory.clone();
            let uid = user_id.clone();
            match tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                memory_guard.run_maintenance(decay_factor, &uid, true)
            })
            .await
            {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    tracing::error!(user_id = %user_id, "Consolidation maintenance failed: {e}");
                    return;
                }
                Err(e) => {
                    tracing::error!(user_id = %user_id, "Consolidation maintenance panicked: {e}");
                    return;
                }
            }
        };

        // Step 3: Apply graph strengthening from replay results
        let mut edges_strengthened: usize = 0;
        let mut entity_edges_strengthened: usize = 0;

        // Direction 1: Edge strengthening + promotion boost propagation
        if !maintenance_result.edge_boosts.is_empty() {
            if let Ok(graph) = state_clone.get_user_graph(&user_id) {
                let graph_guard = graph.read();
                match graph_guard.strengthen_memory_edges(&maintenance_result.edge_boosts) {
                    Ok((count, promotion_boosts)) => {
                        edges_strengthened += count;
                        if !promotion_boosts.is_empty() {
                            let memory_guard = memory.read();
                            let _ = memory_guard.apply_edge_promotion_boosts(&promotion_boosts);
                        }
                    }
                    Err(e) => {
                        tracing::debug!("On-demand edge boost failed: {e}");
                    }
                }
            }
        }

        // Direction 3: Entity-entity Hebbian reinforcement for replayed memories
        if !maintenance_result.replay_memory_ids.is_empty() {
            if let Ok(graph) = state_clone.get_user_graph(&user_id) {
                let graph_guard = graph.read();
                for mem_id_str in &maintenance_result.replay_memory_ids {
                    if let Ok(uuid) = uuid::Uuid::parse_str(mem_id_str) {
                        match graph_guard.strengthen_episode_entity_edges(&uuid) {
                            Ok(count) => entity_edges_strengthened += count,
                            Err(e) => {
                                tracing::debug!(
                                    "Entity edge strengthening failed for {mem_id_str}: {e}"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Direction 2: Lazy decay — flush opportunistic pruning queue
        if let Ok(graph) = state_clone.get_user_graph(&user_id) {
            let graph_guard = graph.read();
            let _ = graph_guard.flush_pending_maintenance();
        }

        // Step 3.5: Dream replay — random memory pair comparison discovers
        // latent cross-topic connections (Wilson & McNaughton 1994).
        // Samples random pairs, computes cosine similarity, creates weak
        // RelatedTo edges for pairs in the discovery band (0.55-0.85).
        let mut dream_edges_created: usize = 0;
        {
            let memory_clone = memory.clone();
            let graph_arc = state_clone.get_user_graph(&user_id).ok();
            let uid = user_id.clone();
            match tokio::task::spawn_blocking(move || -> usize {
                use crate::constants::{
                    DREAM_REPLAY_EDGE_CONFIDENCE, DREAM_REPLAY_PAIR_COUNT,
                    DREAM_REPLAY_SIMILARITY_CEILING, DREAM_REPLAY_SIMILARITY_THRESHOLD,
                };
                use crate::similarity::cosine_similarity;

                let memory_guard = memory_clone.read();
                let Some(graph_arc) = graph_arc else {
                    return 0;
                };
                let graph = graph_arc.read();

                // Get all memory IDs and sample random pairs
                let all_ids = match memory_guard.get_long_term_ids() {
                    Ok(ids) => ids,
                    Err(_) => return 0,
                };
                if all_ids.len() < 10 {
                    return 0; // Not enough memories for meaningful replay
                }

                let mut created = 0usize;
                let mut rng_state = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                // Simple xorshift64 for lightweight randomness (no external dep)
                let mut next_rand = || -> usize {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    rng_state as usize
                };

                let pair_count = DREAM_REPLAY_PAIR_COUNT.min(all_ids.len() * (all_ids.len() - 1) / 2);

                for _ in 0..pair_count {
                    let idx_a = next_rand() % all_ids.len();
                    let mut idx_b = next_rand() % all_ids.len();
                    if idx_b == idx_a {
                        idx_b = (idx_a + 1) % all_ids.len();
                    }

                    let mem_a = match memory_guard.get_memory(&all_ids[idx_a]) {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    let mem_b = match memory_guard.get_memory(&all_ids[idx_b]) {
                        Ok(m) => m,
                        Err(_) => continue,
                    };

                    let (Some(emb_a), Some(emb_b)) = (
                        &mem_a.experience.embeddings,
                        &mem_b.experience.embeddings,
                    ) else {
                        continue;
                    };

                    let sim = cosine_similarity(emb_a, emb_b);
                    metrics::DREAM_REPLAY_PAIRS_EVALUATED.inc();

                    // Discovery band: interesting but not already-connected similarity
                    if sim < DREAM_REPLAY_SIMILARITY_THRESHOLD
                        || sim > DREAM_REPLAY_SIMILARITY_CEILING
                    {
                        continue;
                    }

                    // Check if entities from both memories have any existing edge
                    let entities_a: Vec<uuid::Uuid> =
                        mem_a.entity_refs.iter().map(|e| e.entity_id).collect();
                    let entities_b: Vec<uuid::Uuid> =
                        mem_b.entity_refs.iter().map(|e| e.entity_id).collect();

                    if entities_a.is_empty() || entities_b.is_empty() {
                        continue;
                    }

                    // Check if any cross-entity edge already exists
                    let mut already_connected = false;
                    'outer: for ea in &entities_a {
                        for eb in &entities_b {
                            if graph.find_relationship_between(ea, eb).ok().flatten().is_some() {
                                already_connected = true;
                                break 'outer;
                            }
                        }
                    }
                    if already_connected {
                        continue;
                    }

                    // Create a discovery edge between the first entity pair
                    let now = chrono::Utc::now();
                    let edge = crate::graph_memory::RelationshipEdge {
                        uuid: uuid::Uuid::new_v4(),
                        from_entity: entities_a[0],
                        to_entity: entities_b[0],
                        relation_type: crate::graph_memory::RelationType::RelatedTo,
                        strength: DREAM_REPLAY_EDGE_CONFIDENCE,
                        created_at: now,
                        valid_at: now,
                        invalidated_at: None,
                        source_episode_id: None,
                        context: format!(
                            "dream-replay: cosine={:.3} between memories {}..{}",
                            sim,
                            &mem_a.id.0.to_string()[..8],
                            &mem_b.id.0.to_string()[..8],
                        ),
                        last_activated: now,
                        activation_count: 0,
                        ltp_status: Default::default(),
                        tier: Default::default(),
                        activation_timestamps: None,
                        entity_confidence: None,
                    };
                    match graph.add_relationship(edge) {
                        Ok(_) => {
                            created += 1;
                            metrics::DREAM_REPLAY_EDGES_CREATED.inc();
                        }
                        Err(e) => {
                            tracing::debug!(
                                "Dream replay edge creation failed: {e}"
                            );
                        }
                    }
                }

                tracing::info!(
                    user_id = %uid,
                    pairs_evaluated = pair_count,
                    edges_created = created,
                    "Dream replay complete"
                );
                created
            })
            .await
            {
                Ok(count) => dream_edges_created = count,
                Err(e) => {
                    tracing::debug!("Dream replay task panicked: {e}");
                }
            }
        }

        // Step 4: Gap analysis — sync graph to SQLite, detect structural gaps
        if let Ok(graph) = state_clone.get_user_graph(&user_id) {
            if let Ok(store) = state_clone.get_user_slow_store(&user_id) {
                let graph_guard = graph.read();
                match (graph_guard.get_all_entities(), graph_guard.get_all_relationships()) {
                    (Ok(entities), Ok(edges)) => {
                        if let Ok(_sync) = store.sync_from_graph(&entities, &edges) {
                            let config = GapDetectionConfig::default();
                            match GapDetector::detect(store.as_ref(), &config) {
                                Ok(result) => {
                                    tracing::info!(
                                        user_id = %user_id,
                                        gaps = result.gaps.len(),
                                        types = ?result.type_counts,
                                        "Gap detection complete (post-consolidation)"
                                    );
                                }
                                Err(e) => {
                                    tracing::debug!("Gap detection failed: {e}");
                                }
                            }
                        }
                    }
                    _ => {
                        tracing::debug!("Could not load graph data for gap analysis");
                    }
                }
            }
        }

        let duration = op_start.elapsed().as_secs_f64();
        metrics::CONSOLIDATE_DURATION.observe(duration);
        metrics::CONSOLIDATE_TOTAL
            .with_label_values(&["success"])
            .inc();

        tracing::info!(
            user_id = %user_id,
            memories_processed = result.memories_processed,
            facts_extracted = result.facts_extracted,
            facts_reinforced = result.facts_reinforced,
            memories_replayed = maintenance_result.replay_memory_ids.len(),
            edges_strengthened,
            entity_edges_strengthened,
            dream_edges_created,
            memories_decayed = maintenance_result.decayed_count,
            duration_secs = format!("{:.1}", duration),
            "Consolidation complete (background)"
        );
    });

    // Return immediately — work continues in background
    Ok(Json(ConsolidateResponse {
        memories_analyzed: 0,
        facts_extracted: 0,
        facts_reinforced: 0,
        fact_ids: vec![],
        memories_replayed: 0,
        edges_strengthened: 0,
        entity_edges_strengthened: 0,
        memories_decayed: 0,
        warnings: vec![
            "Consolidation started in background. Check /api/consolidation/report for results."
                .to_string(),
        ],
    }))
}

// =============================================================================
// SLEEP-PHASE CONSOLIDATION (A7: CLS dual-structure + LightMem offline)
// =============================================================================

use super::types::{SleepPhaseRequest, SleepPhaseResponse};

/// Sleep-phase consolidation — heavyweight offline pipeline (A7)
///
/// Composes the full CLS-inspired consolidation sequence:
/// 1. Fact extraction (cortical integration)
/// 2. Heavy maintenance (replay + tier promotion + decay)
/// 3. Hebbian edge strengthening from replay results
/// 4. Entity-entity reinforcement for replayed memories
/// 5. Opportunistic edge pruning flush
/// 6. Dream replay with enlarged batch (hippocampal replay)
/// 7. Gap analysis for structural integrity
///
/// Returns 202 immediately. Work runs in background.
/// Designed to be called during low-activity periods (cron, hooks).
///
/// Reference: Bai et al. (2026) §2.1.3 CLS Theory, §4.2.3 LightMem
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn sleep_phase_consolidation(
    State(state): State<AppState>,
    Json(req): Json<SleepPhaseRequest>,
) -> Result<Json<SleepPhaseResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let _ = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let replay_multiplier = req.replay_multiplier.unwrap_or(3);
    let state_clone = state.clone();

    tokio::task::spawn(async move {
        let op_start = std::time::Instant::now();
        tracing::info!(user_id = %user_id, replay_multiplier, "Sleep-phase consolidation starting");

        let memory = match state_clone.get_user_memory(&user_id) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(user_id = %user_id, "Sleep-phase: failed to get memory: {e}");
                return;
            }
        };

        // Phase 1: Fact extraction (with heat-score-aware consolidation from A2)
        {
            let memory = memory.clone();
            let uid = user_id.clone();
            match tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                memory_guard.distill_facts(
                    &uid,
                    crate::constants::CONSOLIDATION_MIN_SUPPORT,
                    crate::constants::CONSOLIDATION_MIN_AGE_DAYS,
                )
            })
            .await
            {
                Ok(Ok(r)) => {
                    tracing::info!(
                        user_id = %user_id,
                        facts_extracted = r.facts_extracted,
                        "Sleep-phase: fact extraction complete"
                    );
                }
                Ok(Err(e)) => tracing::warn!(user_id = %user_id, "Sleep-phase fact extraction failed: {e}"),
                Err(e) => tracing::warn!(user_id = %user_id, "Sleep-phase fact extraction panicked: {e}"),
            }
        }

        // Phase 2: Heavy maintenance (replay + tier consolidation + decay)
        let decay_factor = state_clone.server_config().activation_decay_factor;
        let maintenance_result = {
            let memory = memory.clone();
            let uid = user_id.clone();
            match tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                memory_guard.run_maintenance(decay_factor, &uid, true)
            })
            .await
            {
                Ok(Ok(r)) => Some(r),
                Ok(Err(e)) => {
                    tracing::warn!(user_id = %user_id, "Sleep-phase maintenance failed: {e}");
                    None
                }
                Err(e) => {
                    tracing::warn!(user_id = %user_id, "Sleep-phase maintenance panicked: {e}");
                    None
                }
            }
        };

        // Phase 3: Edge strengthening from replay
        if let Some(ref maint) = maintenance_result {
            if !maint.edge_boosts.is_empty() {
                if let Ok(graph) = state_clone.get_user_graph(&user_id) {
                    let graph_guard = graph.read();
                    match graph_guard.strengthen_memory_edges(&maint.edge_boosts) {
                        Ok((count, promotion_boosts)) => {
                            tracing::debug!(user_id = %user_id, count, "Sleep-phase: edges strengthened");
                            if !promotion_boosts.is_empty() {
                                let memory_guard = memory.read();
                                let _ = memory_guard.apply_edge_promotion_boosts(&promotion_boosts);
                            }
                        }
                        Err(e) => tracing::debug!("Sleep-phase edge boost failed: {e}"),
                    }
                }
            }

            // Phase 4: Entity-entity Hebbian reinforcement
            if !maint.replay_memory_ids.is_empty() {
                if let Ok(graph) = state_clone.get_user_graph(&user_id) {
                    let graph_guard = graph.read();
                    for mem_id_str in &maint.replay_memory_ids {
                        if let Ok(uuid) = uuid::Uuid::parse_str(mem_id_str) {
                            let _ = graph_guard.strengthen_episode_entity_edges(&uuid);
                        }
                    }
                }
            }
        }

        // Phase 5: Flush pending maintenance (opportunistic edge pruning)
        if let Ok(graph) = state_clone.get_user_graph(&user_id) {
            let graph_guard = graph.read();
            let _ = graph_guard.flush_pending_maintenance();
        }

        // Phase 6: Dream replay with enlarged batch
        {
            let memory_clone = memory.clone();
            let graph_arc = state_clone.get_user_graph(&user_id).ok();
            let uid = user_id.clone();
            let _ = tokio::task::spawn_blocking(move || -> usize {
                use crate::constants::{
                    DREAM_REPLAY_EDGE_CONFIDENCE, DREAM_REPLAY_PAIR_COUNT,
                    DREAM_REPLAY_SIMILARITY_CEILING, DREAM_REPLAY_SIMILARITY_THRESHOLD,
                };
                use crate::similarity::cosine_similarity;

                let memory_guard = memory_clone.read();
                let Some(graph_arc) = graph_arc else { return 0 };
                let graph = graph_arc.read();

                let all_ids = match memory_guard.get_long_term_ids() {
                    Ok(ids) => ids,
                    Err(_) => return 0,
                };
                if all_ids.len() < 10 { return 0; }

                let mut created = 0usize;
                let mut rng_state = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let mut next_rand = || -> usize {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    rng_state as usize
                };

                // Sleep-phase uses enlarged batch (multiplier × normal count)
                let pair_count = (DREAM_REPLAY_PAIR_COUNT * replay_multiplier)
                    .min(all_ids.len() * (all_ids.len() - 1) / 2);

                for _ in 0..pair_count {
                    let idx_a = next_rand() % all_ids.len();
                    let mut idx_b = next_rand() % all_ids.len();
                    if idx_b == idx_a { idx_b = (idx_a + 1) % all_ids.len(); }

                    let mem_a = match memory_guard.get_memory(&all_ids[idx_a]) { Ok(m) => m, Err(_) => continue };
                    let mem_b = match memory_guard.get_memory(&all_ids[idx_b]) { Ok(m) => m, Err(_) => continue };

                    let (Some(emb_a), Some(emb_b)) = (&mem_a.experience.embeddings, &mem_b.experience.embeddings) else { continue };

                    let sim = cosine_similarity(emb_a, emb_b);
                    if sim < DREAM_REPLAY_SIMILARITY_THRESHOLD || sim > DREAM_REPLAY_SIMILARITY_CEILING { continue; }

                    let entities_a: Vec<uuid::Uuid> = mem_a.entity_refs.iter().map(|e| e.entity_id).collect();
                    let entities_b: Vec<uuid::Uuid> = mem_b.entity_refs.iter().map(|e| e.entity_id).collect();
                    if entities_a.is_empty() || entities_b.is_empty() { continue; }

                    let mut already_connected = false;
                    'outer: for ea in &entities_a {
                        for eb in &entities_b {
                            if graph.find_relationship_between(ea, eb).ok().flatten().is_some() {
                                already_connected = true;
                                break 'outer;
                            }
                        }
                    }
                    if already_connected { continue; }

                    let now = chrono::Utc::now();
                    let edge = crate::graph_memory::RelationshipEdge {
                        uuid: uuid::Uuid::new_v4(),
                        from_entity: entities_a[0],
                        to_entity: entities_b[0],
                        relation_type: crate::graph_memory::RelationType::RelatedTo,
                        strength: DREAM_REPLAY_EDGE_CONFIDENCE,
                        created_at: now,
                        valid_at: now,
                        invalidated_at: None,
                        source_episode_id: None,
                        context: format!("sleep-replay: cosine={:.3}", sim),
                        last_activated: now,
                        activation_count: 0,
                        ltp_status: Default::default(),
                        tier: Default::default(),
                        activation_timestamps: None,
                        entity_confidence: None,
                    };
                    if graph.add_relationship(edge).is_ok() { created += 1; }
                }

                tracing::info!(user_id = %uid, pairs_evaluated = pair_count, edges_created = created, "Sleep-phase dream replay complete");
                created
            }).await;
        }

        // Phase 7: Gap analysis
        if let Ok(graph) = state_clone.get_user_graph(&user_id) {
            if let Ok(store) = state_clone.get_user_slow_store(&user_id) {
                let graph_guard = graph.read();
                if let (Ok(entities), Ok(edges)) = (graph_guard.get_all_entities(), graph_guard.get_all_relationships()) {
                    if let Ok(_sync) = store.sync_from_graph(&entities, &edges) {
                        let config = GapDetectionConfig::default();
                        if let Ok(result) = GapDetector::detect(store.as_ref(), &config) {
                            tracing::info!(user_id = %user_id, gaps = result.gaps.len(), "Sleep-phase gap detection complete");
                        }
                    }
                }
            }
        }

        let duration = op_start.elapsed().as_secs_f64();
        metrics::CONSOLIDATE_DURATION.observe(duration);
        metrics::CONSOLIDATE_TOTAL
            .with_label_values(&["sleep_phase"])
            .inc();

        tracing::info!(
            user_id = %user_id,
            duration_secs = format!("{:.1}", duration),
            "Sleep-phase consolidation complete"
        );
    });

    Ok(Json(SleepPhaseResponse {
        accepted: true,
        message: "Sleep-phase consolidation started in background. Check /api/consolidation/report for results.".to_string(),
    }))
}

// =============================================================================
// INDEX MAINTENANCE
// =============================================================================

/// Verify vector index integrity - diagnose orphaned memories
pub async fn verify_index_integrity(
    State(state): State<AppState>,
    Json(req): Json<VerifyIndexRequest>,
) -> Result<Json<memory::IndexIntegrityReport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let report = memory_guard
        .verify_index_integrity()
        .map_err(AppError::Internal)?;

    Ok(Json(report))
}

/// Repair vector index - re-index orphaned memories
pub async fn repair_vector_index(
    State(state): State<AppState>,
    Json(req): Json<RepairIndexRequest>,
) -> Result<Json<RepairIndexResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (total_storage, total_indexed, repaired, failed) = memory_guard
        .repair_vector_index()
        .map_err(AppError::Internal)?;

    Ok(Json(RepairIndexResponse {
        success: failed == 0,
        total_storage,
        total_indexed,
        repaired,
        failed,
        is_healthy: total_storage == total_indexed,
    }))
}

/// Cleanup corrupted memories that fail to deserialize
pub async fn cleanup_corrupted(
    State(state): State<AppState>,
    Json(req): Json<CleanupCorruptedRequest>,
) -> Result<Json<CleanupCorruptedResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let deleted_count = memory_guard
        .cleanup_corrupted()
        .map_err(AppError::Internal)?;

    // Broadcast DELETE event for real-time dashboard so TUI updates its count
    if deleted_count > 0 {
        state.emit_event(MemoryEvent {
            event_type: "DELETE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: req.user_id.clone(),
            memory_id: None,
            content_preview: Some(format!("cleanup: {} corrupted entries", deleted_count)),
            memory_type: None,
            importance: None,
            count: Some(deleted_count),
            results: None,
        });
    }

    Ok(Json(CleanupCorruptedResponse {
        success: true,
        deleted_count,
    }))
}

/// Migrate legacy memories to current format
/// This converts old storage formats to the current schema without data loss
pub async fn migrate_legacy(
    State(state): State<AppState>,
    Json(req): Json<MigrateLegacyRequest>,
) -> Result<Json<MigrateLegacyResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (migrated, already_current, failed) =
        memory_guard.migrate_legacy().map_err(AppError::Internal)?;

    // Broadcast event for real-time dashboard
    if migrated > 0 {
        state.emit_event(MemoryEvent {
            event_type: "MIGRATE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: req.user_id.clone(),
            memory_id: None,
            content_preview: Some(format!(
                "migrated {} memories, {} already current, {} failed",
                migrated, already_current, failed
            )),
            memory_type: None,
            importance: None,
            count: Some(migrated),
            results: None,
        });
    }

    Ok(Json(MigrateLegacyResponse {
        success: true,
        migrated_count: migrated,
        already_current_count: already_current,
        failed_count: failed,
    }))
}

/// Rebuild vector index from storage (removes orphaned index entries)
pub async fn rebuild_index(
    State(state): State<AppState>,
    Json(req): Json<RebuildIndexRequest>,
) -> Result<Json<RebuildIndexResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (storage_count, indexed_count) =
        memory_guard.rebuild_index().map_err(AppError::Internal)?;

    Ok(Json(RebuildIndexResponse {
        success: true,
        storage_count,
        indexed_count,
        is_healthy: storage_count == indexed_count,
    }))
}

// =============================================================================
// BACKUP & RESTORE
// =============================================================================

/// Create a comprehensive backup for a user (memories + secondary stores)
pub async fn create_backup(
    State(state): State<AppState>,
    Json(req): Json<CreateBackupRequest>,
) -> Result<Json<BackupResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let db = memory_guard.get_db();

    // Collect secondary store DB references for comprehensive backup
    let secondary_refs = state.collect_secondary_store_refs();
    let store_refs: Vec<crate::backup::SecondaryStoreRef<'_>> = secondary_refs
        .iter()
        .map(|(name, db)| crate::backup::SecondaryStoreRef { name, db })
        .collect();

    // Get graph DB reference for backup (per-user graph at {user_id}/graph/)
    let graph_lock = state.get_user_graph(&req.user_id).ok();
    let graph_guard = graph_lock.as_ref().map(|g| g.read());
    let graph_db_ref = graph_guard.as_ref().map(|g| g.get_db());

    let result = if store_refs.is_empty() && graph_db_ref.is_none() {
        state.backup_engine().create_backup(&db, &req.user_id)
    } else {
        state
            .backup_engine()
            .create_comprehensive_backup_with_graph(&db, &req.user_id, &store_refs, graph_db_ref)
    };

    match result {
        Ok(metadata) => {
            let secondary_count = metadata.secondary_stores.len();
            state.log_event(
                &req.user_id,
                "BACKUP_CREATED",
                &metadata.backup_id.to_string(),
                &format!(
                    "Backup created: {} bytes + {} secondary stores ({} bytes)",
                    metadata.size_bytes, secondary_count, metadata.secondary_size_bytes
                ),
            );
            Ok(Json(BackupResponse {
                success: true,
                backup: Some(metadata),
                message: "Backup created successfully".to_string(),
            }))
        }
        Err(e) => Ok(Json(BackupResponse {
            success: false,
            backup: None,
            message: format!("Backup failed: {}", e),
        })),
    }
}

/// List all backups for a user
pub async fn list_backups(
    State(state): State<AppState>,
    Json(req): Json<ListBackupsRequest>,
) -> Result<Json<ListBackupsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state.backup_engine().list_backups(&req.user_id) {
        Ok(backups) => {
            let count = backups.len();
            Ok(Json(ListBackupsResponse {
                success: true,
                backups,
                count,
            }))
        }
        Err(e) => Err(AppError::Internal(e)),
    }
}

/// Verify backup integrity
pub async fn verify_backup(
    State(state): State<AppState>,
    Json(req): Json<VerifyBackupRequest>,
) -> Result<Json<VerifyBackupResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .backup_engine()
        .verify_backup(&req.user_id, req.backup_id)
    {
        Ok(is_valid) => Ok(Json(VerifyBackupResponse {
            success: true,
            is_valid,
            message: if is_valid {
                "Backup integrity verified".to_string()
            } else {
                "Backup checksum mismatch - may be corrupted".to_string()
            },
        })),
        Err(e) => Ok(Json(VerifyBackupResponse {
            success: false,
            is_valid: false,
            message: format!("Verification failed: {}", e),
        })),
    }
}

/// Purge old backups
pub async fn purge_backups(
    State(state): State<AppState>,
    Json(req): Json<PurgeBackupsRequest>,
) -> Result<Json<PurgeBackupsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .backup_engine()
        .purge_old_backups(&req.user_id, req.keep_count)
    {
        Ok(purged_count) => {
            if purged_count > 0 {
                state.log_event(
                    &req.user_id,
                    "BACKUP_PURGE",
                    &format!("keep_{}", req.keep_count),
                    &format!("Purged {} old backups", purged_count),
                );
            }
            Ok(Json(PurgeBackupsResponse {
                success: true,
                purged_count,
            }))
        }
        Err(e) => Err(AppError::Internal(e)),
    }
}

/// Restore a user's data from a backup (memories, secondary stores, and graph)
///
/// This endpoint:
/// 1. Closes the user's current memory/graph sessions
/// 2. Restores main memories DB from backup
/// 3. Restores secondary stores (shared DB with todos, reminders, etc.)
/// 4. Restores graph DB if present in backup
/// 5. Re-initializes the user's memory and graph systems
pub async fn restore_backup(
    State(state): State<AppState>,
    Json(req): Json<RestoreBackupRequest>,
) -> Result<Json<RestoreBackupResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let user_id = req.user_id.clone();

    // Determine restore paths based on server's base path
    let memory_db_path = state.base_path().join(&user_id).join("storage");
    let graph_path = state.base_path().join(&user_id).join("graph").join("graph");

    // Evict user from caches so DB handles are released
    state.evict_user(&user_id);

    // Note: shared DB is NOT restored because it is a multi-user resource.
    // Restoring it from one user's backup would destroy all other users'
    // todos, reminders, audit logs, etc. Only per-user stores are restored.
    let secondary_restore_paths: Vec<(&str, &std::path::Path)> = vec![];

    // Execute restore
    let restored_stores = state
        .backup_engine()
        .restore_comprehensive_backup(
            &user_id,
            req.backup_id,
            &memory_db_path,
            &secondary_restore_paths,
        )
        .map_err(AppError::Internal)?;

    // Restore graph checkpoint if present in backup
    let resolved_backup_id = req.backup_id.unwrap_or_else(|| {
        state
            .backup_engine()
            .list_backups(&user_id)
            .ok()
            .and_then(|b| b.last().map(|m| m.backup_id))
            .unwrap_or(0)
    });
    let graph_checkpoint = state
        .backup_engine()
        .backup_path()
        .join(&user_id)
        .join(format!("secondary_{resolved_backup_id}"))
        .join("graph");

    let mut all_restored = restored_stores;
    if graph_checkpoint.exists() {
        // Remove existing graph and copy from backup
        if graph_path.exists() {
            let _ = std::fs::remove_dir_all(&graph_path);
        }
        if let Err(e) = crate::backup::copy_dir_recursive_pub(&graph_checkpoint, &graph_path) {
            tracing::warn!(error = %e, "Failed to restore graph DB from backup");
        } else {
            all_restored.push("graph".to_string());
            tracing::info!("Graph DB restored from backup");
        }
    }

    state.log_event(
        &user_id,
        "BACKUP_RESTORED",
        &format!("backup_{}", req.backup_id.unwrap_or(0)),
        &format!("Restored {} stores: {:?}", all_restored.len(), all_restored),
    );

    Ok(Json(RestoreBackupResponse {
        success: true,
        message: format!(
            "Restore complete for user '{}'. Restored: {:?}. Server restart recommended to re-initialize all caches.",
            user_id, all_restored
        ),
        restored_stores: all_restored,
    }))
}

// =============================================================================
// CONSOLIDATION INTROSPECTION
// =============================================================================

use serde::Deserialize;

/// Request for consolidation report
#[derive(Debug, Deserialize)]
pub struct ConsolidationReportRequest {
    pub user_id: String,
    #[serde(default)]
    pub since: Option<String>,
    #[serde(default)]
    pub until: Option<String>,
}

/// Request for consolidation events
#[derive(Debug, Deserialize)]
pub struct ConsolidationEventsRequest {
    pub user_id: String,
    #[serde(default)]
    pub since: Option<String>,
}

/// POST /api/consolidation/report - Get consolidation introspection report
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn get_consolidation_report(
    State(state): State<AppState>,
    Json(req): Json<ConsolidationReportRequest>,
) -> Result<Json<memory::ConsolidationReport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let now = chrono::Utc::now();
    let since = if let Some(since_str) = &req.since {
        chrono::DateTime::parse_from_rfc3339(since_str)
            .map_err(|e| AppError::InvalidInput {
                field: "since".to_string(),
                reason: format!("Invalid timestamp: {}", e),
            })?
            .with_timezone(&chrono::Utc)
    } else {
        now - chrono::Duration::hours(1)
    };

    let until = if let Some(until_str) = &req.until {
        Some(
            chrono::DateTime::parse_from_rfc3339(until_str)
                .map_err(|e| AppError::InvalidInput {
                    field: "until".to_string(),
                    reason: format!("Invalid timestamp: {}", e),
                })?
                .with_timezone(&chrono::Utc),
        )
    } else {
        None
    };

    let report = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_consolidation_report(since, until)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    Ok(Json(report))
}

/// GET /api/consolidation/events - Get raw consolidation events since a timestamp
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn get_consolidation_events(
    State(state): State<AppState>,
    axum::extract::Query(req): axum::extract::Query<ConsolidationEventsRequest>,
) -> Result<Json<Vec<memory::ConsolidationEvent>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let now = chrono::Utc::now();
    let since = if let Some(since_str) = &req.since {
        chrono::DateTime::parse_from_rfc3339(since_str)
            .map_err(|e| AppError::InvalidInput {
                field: "since".to_string(),
                reason: format!("Invalid timestamp: {}", e),
            })?
            .with_timezone(&chrono::Utc)
    } else {
        now - chrono::Duration::hours(1)
    };

    let events = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_consolidation_events_since(since)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    Ok(Json(events))
}
