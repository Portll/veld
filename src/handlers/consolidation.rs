//! Consolidation, Index Maintenance, and Backup Handlers
//!
//! Semantic consolidation for fact extraction, vector index maintenance,
//! and backup/restore operations.

use axum::{
    extract::{Extension, State},
    response::Json,
};

use super::state::MultiUserMemoryManager;
use super::types::{
    BackupResponse, CleanupCorruptedRequest, CleanupCorruptedResponse, ConsolidateRequest,
    ConsolidateResponse, CreateBackupRequest, ListBackupsRequest, ListBackupsResponse, MemoryEvent,
    MigrateLegacyRequest, MigrateLegacyResponse, PurgeBackupsRequest, PurgeBackupsResponse,
    RebuildIndexRequest, RebuildIndexResponse, RepairIndexRequest, RepairIndexResponse,
    RestoreBackupRequest, RestoreBackupResponse, VerifyBackupRequest, VerifyBackupResponse,
    VerifyIndexRequest,
};
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory;
use crate::memory::gap_topology::{GapDetectionConfig, GapDetector};

/// Hysteresis window (M5): a gap must be absent for this long before it is
/// marked resolved, so transient graph churn does not flap gaps
/// resolved/unresolved across consolidation cycles.
const GAP_STALE_RESOLVE_SECS: i64 = 3 * 3600;

/// Persist detected gaps to the slow store and resolve gaps gone stale (M5).
///
/// Closes the historical "computed -> logged -> dropped" leak: gap topology was
/// recomputed every consolidation and discarded. Each gap is persisted with a
/// deterministic, per-instance id (`type|shape|sorted-entity-uuids`) so
/// re-detection is idempotent via `ON CONFLICT` (which re-opens a previously
/// resolved gap). Gaps not re-verified within the hysteresis window are
/// resolved. Surface-only: never triggers acquisition (no curiosity feedback
/// loop); reads of these gaps are scope-filtered to prevent cross-user leakage.
fn persist_detected_gaps(
    store: &crate::memory::slow_store::SlowStore,
    scope: &str,
    result: &crate::memory::gap_topology::GapDetectionResult,
) {
    for gap in &result.gaps {
        let mut uuids: Vec<&str> = gap.entities.iter().map(|e| e.uuid.as_str()).collect();
        uuids.sort_unstable();
        let id = format!(
            "{}|{}|{}",
            gap.gap_type.as_str(),
            gap.shape.canonical,
            uuids.join(",")
        );
        let entities_json =
            serde_json::to_string(&gap.entities).unwrap_or_else(|_| "[]".to_string());
        let missing_links_json =
            serde_json::to_string(&gap.missing_links).unwrap_or_else(|_| "[]".to_string());
        if let Err(e) = store.store_gap(
            &id,
            gap.gap_type.as_str(),
            &gap.shape.canonical,
            &entities_json,
            &missing_links_json,
            gap.confidence,
            gap.embedding_similarity,
            gap.impact_score,
            scope,
        ) {
            tracing::debug!(scope = %scope, "M5: store_gap failed: {e}");
        }
    }
    let cutoff =
        (chrono::Utc::now() - chrono::Duration::seconds(GAP_STALE_RESOLVE_SECS)).to_rfc3339();
    if let Ok(n) = store.resolve_stale_gaps(scope, &cutoff) {
        if n > 0 {
            tracing::debug!(scope = %scope, resolved = n, "M5: resolved stale gaps");
        }
    }
}
use crate::metrics;
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Per-user consolidation locks. Prevents concurrent consolidation runs for the same
/// user from double-strengthening edges or duplicating work.
static CONSOLIDATION_LOCKS: std::sync::LazyLock<
    dashmap::DashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>,
> = std::sync::LazyLock::new(dashmap::DashMap::new);

/// `user_id`'s consolidation lock. Scheduled maintenance try-locks it as well: it runs
/// the same maintenance and edge strengthening, and two passes at once for one user
/// double-strengthen edges.
pub(crate) fn consolidation_lock(user_id: &str) -> std::sync::Arc<tokio::sync::Mutex<()>> {
    CONSOLIDATION_LOCKS
        .entry(user_id.to_string())
        .or_insert_with(|| std::sync::Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

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
        .get_user_memory_async(&req.user_id)
        .await
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let min_support = req.min_support;
    let min_age_days = req.min_age_days;
    let state_clone = state.clone();

    // Spawn the entire pipeline as a background task on the manager's tracker: it
    // survives HTTP timeout cancellation, and the shutdown drain waits for it before the
    // stores are flushed.
    state.task_tracker.spawn(async move {
        let op_start = std::time::Instant::now();

        // Guard against concurrent consolidation for the same user — concurrent runs
        // can double-strengthen edges and waste resources. Skip if already running.
        let lock_arc = consolidation_lock(&user_id);
        let _consolidation_guard = match lock_arc.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                tracing::info!(
                    user_id = %user_id,
                    "Consolidation already in progress, skipping"
                );
                return;
            }
        };

        let memory = match state_clone.get_user_memory_async(&user_id).await {
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

        // Step 2: Maintenance (replay + tier consolidation + decay). Its heavy cycle may run
        // the extraction step 1 just ran; ingest_candidate makes the repeat a no-op, and the
        // maintenance pass applies its own thresholds.
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
            if let Ok(graph) = state_clone.get_user_graph_async(&user_id).await {
                let graph_guard = graph.read();
                match graph_guard.strengthen_memory_edges(
                    &maintenance_result.edge_boosts,
                    "consolidation:replay",
                ) {
                    Ok((count, promotion_boosts)) => {
                        edges_strengthened += count;
                        if !promotion_boosts.is_empty() {
                            let memory_guard = memory.read();
                            if let Err(e) =
                                memory_guard.apply_edge_promotion_boosts(&promotion_boosts)
                            {
                                tracing::warn!(
                                    user_id = %user_id,
                                    "Failed to apply edge promotion boosts: {e}"
                                );
                            }
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
            if let Ok(graph) = state_clone.get_user_graph_async(&user_id).await {
                let graph_guard = graph.read();
                for mem_id_str in &maintenance_result.replay_memory_ids {
                    if let Ok(uuid) = uuid::Uuid::parse_str(mem_id_str) {
                        match graph_guard
                            .strengthen_episode_entity_edges(&uuid, "consolidation:replay")
                        {
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
        if let Ok(graph) = state_clone.get_user_graph_async(&user_id).await {
            let graph_guard = graph.read();
            if let Err(e) = graph_guard.flush_pending_maintenance() {
                tracing::warn!(user_id = %user_id, "Failed to flush pending maintenance: {e}");
            }
        }

        // Step 3.5: Dream replay — random memory pair comparison discovers
        // latent cross-topic connections (Wilson & McNaughton 1994).
        // Samples random pairs, computes cosine similarity, creates weak
        // RelatedTo edges for pairs in the discovery band (0.55-0.85).
        let mut dream_edges_created: usize = 0;
        {
            let memory_clone = memory.clone();
            let graph_arc = state_clone.get_user_graph_async(&user_id).await.ok();
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

                let pair_count =
                    DREAM_REPLAY_PAIR_COUNT.min(all_ids.len() * (all_ids.len() - 1) / 2);

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

                    let (Some(emb_a), Some(emb_b)) =
                        (&mem_a.experience.embeddings, &mem_b.experience.embeddings)
                    else {
                        continue;
                    };

                    let sim =
                        crate::similarity::scale::to_reference(cosine_similarity(emb_a, emb_b));
                    metrics::DREAM_REPLAY_PAIRS_EVALUATED.inc();

                    // Discovery band: interesting but not already-connected similarity
                    if !(DREAM_REPLAY_SIMILARITY_THRESHOLD..=DREAM_REPLAY_SIMILARITY_CEILING)
                        .contains(&sim)
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
                            if graph
                                .find_relationship_between(ea, eb)
                                .ok()
                                .flatten()
                                .is_some()
                            {
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
                        created_by: crate::graph_memory::EdgeSource::DreamReplay,
                        forward_strength: DREAM_REPLAY_EDGE_CONFIDENCE,
                        backward_strength: DREAM_REPLAY_EDGE_CONFIDENCE,
                        strengthening_sources: Default::default(),
                        last_corroborated: None,
                    };
                    match graph.add_relationship(edge) {
                        Ok(_) => {
                            created += 1;
                            metrics::DREAM_REPLAY_EDGES_CREATED.inc();
                        }
                        Err(e) => {
                            tracing::debug!("Dream replay edge creation failed: {e}");
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

        // Step 4: Gap analysis — refresh the SQLite graph mirror, then detect
        // structural gaps. The refresh is gated inside SlowStore (one rebuild
        // in flight, TTL since the last); when it is skipped, detection runs
        // on the existing mirror.
        if let (Ok(graph), Ok(store)) = (
            state_clone.get_user_graph_async(&user_id).await,
            state_clone.get_user_slow_store_async(&user_id).await,
        ) {
            let scope = user_id.clone();
            let ttl = state_clone.server_config().graph_mirror_sync_ttl();
            let gap_step = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                if let Some(sync) = store.begin_sync(ttl) {
                    // Hold the graph read lock only for the snapshot.
                    let (entities, edges) = {
                        let graph_guard = graph.read();
                        (
                            graph_guard.get_all_entities()?,
                            graph_guard.get_all_relationships()?,
                        )
                    };
                    sync.rebuild(&entities, &edges)?;
                }
                let result = GapDetector::detect(store.as_ref(), &GapDetectionConfig::default())?;
                persist_detected_gaps(&store, &scope, &result);
                tracing::info!(
                    user_id = %scope,
                    gaps = result.gaps.len(),
                    types = ?result.type_counts,
                    "Gap detection complete + persisted (post-consolidation)"
                );
                Ok(())
            })
            .await;
            match gap_step {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::debug!("Gap analysis failed: {e}"),
                Err(e) => tracing::debug!("Gap analysis task panicked: {e}"),
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
        .get_user_memory_async(&req.user_id)
        .await
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let replay_multiplier = req.replay_multiplier.unwrap_or(3);
    let state_clone = state.clone();

    // On the manager's tracker so the shutdown drain waits for it.
    state.task_tracker.spawn(async move {
        let op_start = std::time::Instant::now();
        tracing::info!(user_id = %user_id, replay_multiplier, "Sleep-phase consolidation starting");

        // Guard against concurrent consolidation runs for the same user.
        let lock_arc = consolidation_lock(&user_id);
        let _consolidation_guard = match lock_arc.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                tracing::info!(
                    user_id = %user_id,
                    "Sleep-phase consolidation already in progress, skipping"
                );
                return;
            }
        };

        let memory = match state_clone.get_user_memory_async(&user_id).await {
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
                Ok(Err(e)) => {
                    tracing::warn!(user_id = %user_id, "Sleep-phase fact extraction failed: {e}")
                }
                Err(e) => {
                    tracing::warn!(user_id = %user_id, "Sleep-phase fact extraction panicked: {e}")
                }
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
                if let Ok(graph) = state_clone.get_user_graph_async(&user_id).await {
                    let graph_guard = graph.read();
                    match graph_guard
                        .strengthen_memory_edges(&maint.edge_boosts, "consolidation:sleep")
                    {
                        Ok((count, promotion_boosts)) => {
                            tracing::debug!(user_id = %user_id, count, "Sleep-phase: edges strengthened");
                            if !promotion_boosts.is_empty() {
                                let memory_guard = memory.read();
                                if let Err(e) =
                                    memory_guard.apply_edge_promotion_boosts(&promotion_boosts)
                                {
                                    tracing::warn!(
                                        user_id = %user_id,
                                        "Sleep-phase: failed to apply edge promotion boosts: {e}"
                                    );
                                }
                            }
                        }
                        Err(e) => tracing::debug!("Sleep-phase edge boost failed: {e}"),
                    }
                }
            }

            // Phase 4: Entity-entity Hebbian reinforcement
            if !maint.replay_memory_ids.is_empty() {
                if let Ok(graph) = state_clone.get_user_graph_async(&user_id).await {
                    let graph_guard = graph.read();
                    for mem_id_str in &maint.replay_memory_ids {
                        if let Ok(uuid) = uuid::Uuid::parse_str(mem_id_str) {
                            if let Err(e) = graph_guard
                                .strengthen_episode_entity_edges(&uuid, "consolidation:sleep")
                            {
                                tracing::warn!(
                                    user_id = %user_id,
                                    "Sleep-phase: entity edge strengthening failed for {uuid}: {e}"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Phase 5: Flush pending maintenance (opportunistic edge pruning)
        if let Ok(graph) = state_clone.get_user_graph_async(&user_id).await {
            let graph_guard = graph.read();
            if let Err(e) = graph_guard.flush_pending_maintenance() {
                tracing::warn!(
                    user_id = %user_id,
                    "Sleep-phase: failed to flush pending maintenance: {e}"
                );
            }
        }

        // Phase 6: Dream replay with enlarged batch
        {
            let memory_clone = memory.clone();
            let graph_arc = state_clone.get_user_graph_async(&user_id).await.ok();
            let uid = user_id.clone();
            if let Err(e) = tokio::task::spawn_blocking(move || -> usize {
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

                    let sim = crate::similarity::scale::to_reference(cosine_similarity(emb_a, emb_b));
                    if !(DREAM_REPLAY_SIMILARITY_THRESHOLD..=DREAM_REPLAY_SIMILARITY_CEILING).contains(&sim) { continue; }

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
                        created_by: crate::graph_memory::EdgeSource::SleepReplay,
                        forward_strength: DREAM_REPLAY_EDGE_CONFIDENCE,
                        backward_strength: DREAM_REPLAY_EDGE_CONFIDENCE,
                        strengthening_sources: Default::default(),
                        last_corroborated: None,
                    };
                    if graph.add_relationship(edge).is_ok() { created += 1; }
                }

                tracing::info!(user_id = %uid, pairs_evaluated = pair_count, edges_created = created, "Sleep-phase dream replay complete");
                created
            }).await {
                tracing::warn!(user_id = %user_id, "Sleep-phase dream replay task panicked: {e}");
            }
        }

        // Phase 7: Gap analysis — same gated mirror refresh as the
        // post-consolidation step: snapshot under the graph lock, rebuild
        // without it, and detect on the existing mirror when the rebuild is
        // skipped.
        if let (Ok(graph), Ok(store)) = (
            state_clone.get_user_graph_async(&user_id).await,
            state_clone.get_user_slow_store_async(&user_id).await,
        ) {
            let scope = user_id.clone();
            let ttl = state_clone.server_config().graph_mirror_sync_ttl();
            let gap_step = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                if let Some(sync) = store.begin_sync(ttl) {
                    let (entities, edges) = {
                        let graph_guard = graph.read();
                        (
                            graph_guard.get_all_entities()?,
                            graph_guard.get_all_relationships()?,
                        )
                    };
                    sync.rebuild(&entities, &edges)?;
                }
                let result = GapDetector::detect(store.as_ref(), &GapDetectionConfig::default())?;
                persist_detected_gaps(&store, &scope, &result);
                tracing::info!(
                    user_id = %scope,
                    gaps = result.gaps.len(),
                    "Sleep-phase gap detection complete + persisted"
                );
                Ok(())
            })
            .await;
            match gap_step {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::warn!(user_id = %user_id, "Sleep-phase gap analysis failed: {e}")
                }
                Err(e) => {
                    tracing::warn!(user_id = %user_id, "Sleep-phase gap analysis task panicked: {e}")
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
        .get_user_memory_async(&req.user_id)
        .await
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
        .get_user_memory_async(&req.user_id)
        .await
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
        .get_user_memory_async(&req.user_id)
        .await
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
            entities: None,
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
        .get_user_memory_async(&req.user_id)
        .await
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
            entities: None,
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
        .get_user_memory_async(&req.user_id)
        .await
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

/// Re-embed all memories with context-prefixed embeddings (v0.7.2 migration)
///
/// Clears pre-computed embeddings and re-indexes every memory through the new
/// `extract_searchable_text()` path which prepends [project | topic | type].
pub async fn reembed_all(
    State(state): State<AppState>,
    Json(req): Json<RebuildIndexRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory_async(&req.user_id)
        .await
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (reembedded, failed) = memory_guard.reembed_all().map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "success": true,
        "reembedded": reembedded,
        "failed": failed,
    })))
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

    // Off the async runtime: a backup copies the stores, and opens the tenant and its graph
    // when they are not cached.
    let result = {
        let state = state.clone();
        let user_id = req.user_id.clone();
        tokio::task::spawn_blocking(move || state.create_user_backup(&user_id))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("backup task failed: {e}")))?
            .map_err(AppError::Internal)
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

    match state.list_backups_for_user(&req.user_id) {
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

    match state.verify_backup_for_user(&req.user_id, req.backup_id) {
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

    match state.purge_backups_for_user(&req.user_id, req.keep_count) {
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
    // Off the async runtime: a restore is file I/O and may wait for the tenant's store to close.
    let all_restored = {
        let state = state.clone();
        let user_id = user_id.clone();
        tokio::task::spawn_blocking(move || {
            state.restore_user_backup_with_options(&user_id, req.backup_id, req.max_lsn)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("restore task failed: {e}")))?
        .map_err(AppError::Internal)?
    };

    state.log_event(
        &user_id,
        "BACKUP_RESTORED",
        &format!("backup_{}", req.backup_id.unwrap_or(0)),
        &format!(
            "Restored {} stores: {:?} (max_lsn={:?})",
            all_restored.len(),
            all_restored,
            req.max_lsn
        ),
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
        .get_user_memory_async(&req.user_id)
        .await
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
        // Consolidation runs on multi-hour schedules, and clients document a 24-hour default.
        now - chrono::Duration::hours(24)
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
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    axum::extract::Query(req): axum::extract::Query<ConsolidationEventsRequest>,
) -> Result<Json<Vec<memory::ConsolidationEvent>>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&req.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory_async(&user_id)
        .await
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

#[derive(Debug, Deserialize)]
pub struct EmbeddingBacklogQuery {
    pub user_id: String,
}

/// GET /api/embeddings/backlog?user_id= - Memories still without a vector.
pub async fn get_embedding_backlog(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    axum::extract::Query(q): axum::extract::Query<EmbeddingBacklogQuery>,
) -> Result<Json<crate::memory::embedding_backlog::BacklogReport>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&q.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let memory = state
        .get_user_memory_async(&user_id)
        .await
        .map_err(AppError::Internal)?;
    let report = tokio::task::spawn_blocking(move || memory.read().embedding_backlog_report())
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("backlog report task failed: {e}")))?
        .map_err(AppError::Internal)?;
    Ok(Json(report))
}

#[derive(Debug, Deserialize)]
pub struct DrainEmbeddingBacklogRequest {
    pub user_id: String,
    pub limit: Option<usize>,
    #[serde(default)]
    pub reconcile: bool,
}

/// POST /api/embeddings/backlog/drain - Reconcile (optionally) and drain now.
pub async fn drain_embedding_backlog(
    State(state): State<AppState>,
    Json(req): Json<DrainEmbeddingBacklogRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let limit = req.limit.unwrap_or(50);
    if limit == 0 {
        return Err(AppError::InvalidInput {
            field: "limit".to_string(),
            reason: "must be at least 1".to_string(),
        });
    }
    let memory = state
        .get_user_memory_async(&req.user_id)
        .await
        .map_err(AppError::Internal)?;
    let reconcile = req.reconcile;
    let user_id = req.user_id.clone();
    let (reconciled, drained) = tokio::task::spawn_blocking(move || {
        let (reconciled, drained, rewritten) = {
            let guard = memory.read();
            let reconciled = if reconcile {
                Some(guard.reconcile_embedding_backlog()?)
            } else {
                None
            };
            let (drained, rewritten) = guard.drain_embedding_backlog_journaled(limit)?;
            (reconciled, drained, rewritten)
        };
        super::remember::journal_stored(
            &state,
            &memory,
            &user_id,
            &rewritten,
            super::remember::SnapshotKind::Update,
        );
        anyhow::Ok((reconciled, drained))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("backlog drain task failed: {e}")))?
    .map_err(AppError::Internal)?;
    Ok(Json(serde_json::json!({
        "reconciled": reconciled,
        "drained": drained,
    })))
}

#[cfg(test)]
mod embedding_backlog_route_tests {
    use std::fs;
    use std::path::Path;

    fn router_src() -> String {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/handlers/router.rs");
        fs::read_to_string(p).expect("router.rs must be readable")
    }

    #[test]
    fn both_backlog_routes_are_mounted_behind_auth() {
        let src = router_src();
        let protected_at = src
            .find("pub fn build_protected_routes")
            .expect("build_protected_routes must exist");
        for (route, handler) in [
            (
                "/api/embeddings/backlog\"",
                "get(consolidation::get_embedding_backlog)",
            ),
            (
                "/api/embeddings/backlog/drain",
                "post(consolidation::drain_embedding_backlog)",
            ),
        ] {
            let at = src.find(route).expect("route must be mounted");
            assert!(
                at > protected_at,
                "{route} sits on a public or probe router"
            );
            assert!(
                src[at..at + 200].contains(handler),
                "{route} must map to {handler}"
            );
        }
    }

    #[test]
    fn drain_request_defaults() {
        let req: super::DrainEmbeddingBacklogRequest =
            serde_json::from_str(r#"{"user_id":"u"}"#).unwrap();
        assert!(req.limit.is_none());
        assert!(!req.reconcile);
    }
}
