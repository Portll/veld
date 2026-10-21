//! Health and Infrastructure Handlers
//!
//! Kubernetes probes, metrics, and system health endpoints.

use axum::{
    extract::{Extension, Query, State},
    http::StatusCode,
    response::Json,
};
use serde::Deserialize;
use std::collections::HashMap;

use super::state::MultiUserMemoryManager;
use super::types::{ContextStatus, MemoryEvent};
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::AppError;
use crate::metrics;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Health response for main health endpoint
#[derive(serde::Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub build: String,
    pub built_at: String,
    pub requested_storage_backend: String,
    pub effective_storage_backend: String,
    pub users_count: usize,
    pub users_in_cache: usize,
    pub user_evictions: usize,
    pub max_cache_size: usize,
    /// The primary embedder this process chose (id, model, dimension, kind, reason). `null`
    /// until the first user's memory system has been constructed. Measured 2026-09-10: the
    /// running server was on hash-bigram vectors and nothing on this endpoint said so.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedder: Option<crate::embeddings::PrimaryEmbedderInfo>,
    /// Tenants holding data on each backend at boot. `users_count` over an empty
    /// store is the 2026-09-10 outage; this is what tells the two apart.
    pub tenants_with_data: crate::storage::boot_guard::BackendCensus,
    /// Embedding spaces across the users in cache, summed by space: memories each index holds,
    /// backlog rows waiting for a vector in the space, and memories stored in it that an index
    /// in another space refused. Space ids name models and widths, never a tenant.
    pub embedding_spaces: Vec<crate::embeddings::space::SpaceSummary>,
    /// Users in cache whose corpus no index here can search by vector: memories stored, none
    /// indexed, vectors waiting. Any makes `status` degraded. A count, never a tenant id.
    #[serde(skip_serializing_if = "is_zero")]
    pub orphaned_corpora: usize,
    /// Space migrations under way for users in cache: a change of embedder re-embedding in the
    /// background, or the one-time stamp of spaces. Absent when none is running.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub space_migrations: Vec<crate::memory::space_migration::MigrationProgress>,
    /// One entry per store currently refusing writes; absent when none is.
    /// Measured 2026-10-11: this endpoint served `healthy` through 11,088
    /// consecutive failed writes.
    ///
    /// An array rather than one object because this process runs one RocksDB per
    /// tenant: one latched store out of twelve is `degraded`, and the response
    /// has to say which. It says so by opaque `store` id — this route is
    /// unauthenticated, and `LatchSnapshot` documents why no path ships.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub write_latch: Vec<crate::storage::write_latch::LatchSnapshot>,
}

/// `status` and the latch detail from one rule, so the handler and its tests
/// cannot disagree about what "degraded" means.
fn write_latch_fields(
    latch: &crate::storage::write_latch::WriteLatch,
) -> (String, Vec<crate::storage::write_latch::LatchSnapshot>) {
    let latched = latch.snapshot();
    let status = if latched.is_empty() {
        "healthy"
    } else {
        "degraded"
    };
    (status.to_string(), latched)
}

/// Readiness under the write latch. 503 when ANY store is hard-latched; a soft
/// latch is still inside its bounded recovery and the next write may land.
/// Returns the hard-latched store count, which the status code cannot carry and
/// which is the difference between one tenant affected and all of them.
fn ready_decision(
    ready: bool,
    latch: &crate::storage::write_latch::WriteLatch,
) -> (StatusCode, &'static str, usize) {
    let hard_stores = latch.hard_count();
    if ready && hard_stores == 0 {
        (StatusCode::OK, "ready", 0)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not_ready", hard_stores)
    }
}

/// Main health check endpoint
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let users_in_cache = state.users_in_cache();
    let user_evictions = state.user_evictions();
    let config = state.server_config();
    let (status, write_latch) = write_latch_fields(crate::storage::write_latch::global());
    let (embedding_spaces, orphaned_corpora, space_migrations) = embedding_spaces(&state);
    let status = if orphaned_corpora > 0 {
        "degraded".to_string()
    } else {
        status
    };

    Json(HealthResponse {
        status,
        version: env!("VELD_VERSION_FULL").to_string(),
        build: env!("VELD_BUILD_NUMBER").to_string(),
        built_at: env!("VELD_BUILD_TIMESTAMP").to_string(),
        requested_storage_backend: config.requested_storage_backend.to_string(),
        effective_storage_backend: config.effective_storage_backend.to_string(),
        users_count: state.list_users().len(),
        users_in_cache,
        user_evictions,
        max_cache_size: config.max_users_in_memory,
        embedder: crate::embeddings::primary_embedder_info(),
        tenants_with_data: state.backend_census.clone(),
        embedding_spaces,
        orphaned_corpora,
        space_migrations,
        write_latch,
    })
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// Cached users only, and only those not busy: a probe must neither load a tenant nor wait on
/// one. Returns the summed spaces, how many corpora are cut off from vector search, and the
/// migrations under way.
fn embedding_spaces(
    state: &MultiUserMemoryManager,
) -> (
    Vec<crate::embeddings::space::SpaceSummary>,
    usize,
    Vec<crate::memory::space_migration::MigrationProgress>,
) {
    let mut summaries = Vec::new();
    let mut orphaned = 0usize;
    let mut migrations = Vec::new();
    for user_id in state.list_cached_users() {
        // The cache itself, not get_user_memory: a user evicted since the listing is skipped,
        // never loaded.
        let Some(memory) = state.user_memories.get(&user_id) else {
            continue;
        };
        let Some(guard) = memory.try_read() else {
            continue;
        };
        match guard.space_health() {
            Ok((spaces, cut_off)) => {
                summaries.extend(spaces);
                orphaned += usize::from(cut_off);
            }
            Err(e) => tracing::debug!("embedding spaces unreadable for a cached user: {e}"),
        }
        if let Some(m) = guard.space_migration().filter(|m| !m.is_finished()) {
            migrations.push((&m).into());
        }
    }
    (
        crate::embeddings::space::merge_summaries(summaries),
        orphaned,
        migrations,
    )
}

/// Liveness probe - indicates if process is alive and not deadlocked
/// Returns 200 OK if service is running (minimal check, always succeeds if reachable)
pub async fn health_live() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "alive",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// Readiness probe — system-level only. **PUBLIC, unauthenticated.**
///
/// Returns 503 if core shared storage was not initialized successfully.
///
/// Per-user readiness lives on the authenticated `/api/health/ready` route
/// (see `health_ready_user`). The previous `?user_id=` branch leaked
/// per-tenant cache residency, existence-on-disk, and memory/graph stats to
/// any unauthenticated caller. The structural rule is: nothing on the public
/// router reads `?user_id=` for per-tenant data — see `PUBLIC_PATHS` in
/// `router.rs` and the `public_router_has_no_per_user_handlers` test.
pub async fn health_ready(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    let (status_code, status_str, hard_stores) =
        ready_decision(state.is_ready(), crate::storage::write_latch::global());

    let users_in_cache = state.users_in_cache();
    let mut body = serde_json::json!({
        "status": status_str,
        "version": env!("VELD_VERSION_FULL"),
        "effective_storage_backend": state.server_config().effective_storage_backend.as_str(),
        "users_in_cache": users_in_cache,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    if hard_stores > 0 {
        if let Some(object) = body.as_object_mut() {
            object.insert(
                "reason".to_string(),
                serde_json::json!("storage_write_latched"),
            );
            object.insert(
                "hard_latched_stores".to_string(),
                serde_json::json!(hard_stores),
            );
        }
    }
    (status_code, Json(body))
}

/// Per-user readiness probe — **authenticated**. Tenant binding is enforced
/// via `resolve_request_user_id` (same pattern as `delete_memory`), so a
/// multi-tenant caller cannot probe another tenant via `?user_id=`.
///
/// Only checks already-cached users — never triggers lazy initialisation,
/// which would let a caller spin up arbitrary RocksDB instances by probing.
pub async fn health_ready_user(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<(StatusCode, Json<serde_json::Value>), AppError> {
    let user_id = resolve_request_user_id(
        params.get("user_id").map(String::as_str),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;

    let ready = state.is_ready();
    let status_code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let status_str = if ready { "ready" } else { "not_ready" };

    let cached_users = state.list_cached_users();
    let is_cached = cached_users.contains(&user_id);
    let exists_on_disk = state.list_users().contains(&user_id);

    if !is_cached {
        return Ok((
            status_code,
            Json(serde_json::json!({
                "status": status_str,
                "user_id": user_id,
                "user_ready": false,
                "cached": false,
                "exists_on_disk": exists_on_disk,
                "reason": if exists_on_disk {
                    "user exists but is not loaded in cache"
                } else {
                    "user does not exist"
                },
                "timestamp": chrono::Utc::now().to_rfc3339()
            })),
        ));
    }

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let Some(guard) = memory.try_read() else {
        return Ok((
            status_code,
            Json(serde_json::json!({
                "status": status_str,
                "user_id": user_id,
                "user_ready": true,
                "cached": true,
                "reason": "memory system is locked (concurrent operation in progress)",
                "timestamp": chrono::Utc::now().to_rfc3339()
            })),
        ));
    };

    let stats = guard.stats();
    let index_health = guard.index_health();
    let graph_stats = if let Ok(graph) = state.get_user_graph(&user_id) {
        let g = graph.read();
        let gs = g.get_stats().unwrap_or_default();
        serde_json::json!({
            "entity_count": gs.entity_count,
            "relationship_count": gs.relationship_count,
            "episode_count": gs.episode_count
        })
    } else {
        serde_json::json!(null)
    };

    Ok((
        status_code,
        Json(serde_json::json!({
            "status": status_str,
            "user_id": user_id,
            "user_ready": true,
            "cached": true,
            "memory_stats": {
                "total_memories": stats.total_memories,
                "working_memory_count": stats.working_memory_count,
                "session_memory_count": stats.session_memory_count,
                "long_term_memory_count": stats.long_term_memory_count,
                "vector_index_count": stats.vector_index_count,
            },
            "index_health": {
                "total_vectors": index_health.total_vectors,
                "incremental_inserts": index_health.incremental_inserts,
                "deleted_count": index_health.deleted_count,
                "deletion_ratio": index_health.deletion_ratio,
                "needs_rebuild": index_health.needs_rebuild,
                "needs_compaction": index_health.needs_compaction,
                "secondary": index_health.secondary.as_ref().map(|s| serde_json::json!({
                    "total_vectors": s.total_vectors,
                    "incremental_inserts": s.incremental_inserts,
                    "deleted_count": s.deleted_count,
                    "deletion_ratio": s.deletion_ratio,
                    "needs_rebuild": s.needs_rebuild,
                    "needs_compaction": s.needs_compaction,
                })),
            },
            "graph_stats": graph_stats,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    ))
}

/// Vector index health — aggregate-only. **PUBLIC, unauthenticated.**
///
/// Reports totals across currently-cached users without naming any of them.
/// The per-user branch (and the raw internal error body it leaked on the
/// failure path) has been moved behind auth in `health_index_user`. Same
/// disclosure class as the readiness probe: enumerating cached user IDs and
/// returning rebuild-threshold ratios is per-tenant metadata.
pub async fn health_index(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    // Read health across cached users only — never trigger lazy init from a
    // public probe (otherwise every probe opens RocksDB for every on-disk
    // user, exhausting FDs).
    let mut total_vectors: usize = 0;
    let mut total_incremental: usize = 0;
    let mut users_checked: usize = 0;
    let mut users_needing_rebuild: usize = 0;
    let mut users_with_shortfall: usize = 0;
    let mut users_coverage_unreadable: usize = 0;
    for user_id in state.list_cached_users() {
        if let Ok(memory) = state.get_user_memory(&user_id) {
            let guard = memory.read();
            let h = guard.index_health();
            total_vectors += h.total_vectors;
            total_incremental += h.incremental_inserts;
            if h.needs_rebuild {
                users_needing_rebuild += 1;
            }
            // Stored-vs-indexed, counted the same aggregate way as the rebuild
            // flag above. An unreadable coverage is its own count, never folded
            // into "no shortfall".
            match guard.index_coverage() {
                Ok((stored, vec_idx, lex_idx)) if vec_idx < stored || lex_idx < stored => {
                    users_with_shortfall += 1
                }
                Ok(_) => {}
                Err(_) => users_coverage_unreadable += 1,
            }
            users_checked += 1;
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "users_checked": users_checked,
            "total_vectors": total_vectors,
            "total_incremental_inserts": total_incremental,
            "users_needing_rebuild_count": users_needing_rebuild,
            "users_with_index_shortfall_count": users_with_shortfall,
            "users_coverage_unreadable_count": users_coverage_unreadable,
            "rebuild_threshold": crate::vector_db::vamana::REBUILD_THRESHOLD,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// Per-user vector index health — **authenticated**. Same tenant-binding
/// pattern as `delete_memory`: `resolve_request_user_id` rejects a `?user_id=`
/// that does not match the API key's tenant binding.
pub async fn health_index_user(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<(StatusCode, Json<serde_json::Value>), AppError> {
    let user_id = resolve_request_user_id(
        params.get("user_id").map(String::as_str),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let guard = memory.read();
    let health = guard.index_health();
    // Populations, not error counts: a memory stored and never indexed is
    // unretrievable and raises nothing for a counter to see.
    let coverage = guard.index_coverage().map_err(AppError::Internal)?;
    let (stored, vector_indexed, lexically_indexed) = coverage;
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "user_id": user_id,
            "index_coverage": {
                "stored": stored,
                "vector_indexed": vector_indexed,
                "lexically_indexed": lexically_indexed,
                "shortfall": vector_indexed < stored || lexically_indexed < stored,
            },
            "total_vectors": health.total_vectors,
            "incremental_inserts": health.incremental_inserts,
            "needs_rebuild": health.needs_rebuild,
            "rebuild_threshold": health.rebuild_threshold,
            "degradation_percent": if health.rebuild_threshold > 0 {
                (health.incremental_inserts as f64 / health.rebuild_threshold as f64 * 100.0).min(100.0)
            } else {
                0.0
            },
            "secondary": health.secondary.as_ref().map(|s| serde_json::json!({
                "total_vectors": s.total_vectors,
                "needs_rebuild": s.needs_rebuild,
                "deletion_ratio": s.deletion_ratio,
            })),
            "spaces": health.spaces.iter().map(|s| serde_json::json!({
                "space": s.space,
                "codec": s.codec,
                "memories": s.memories,
                "total_vectors": s.total_vectors,
                "vector_bytes": s.vector_bytes,
                "deleted_count": s.deleted_count,
                "needs_rebuild": s.needs_rebuild,
                "needs_compaction": s.needs_compaction,
            })).collect::<Vec<_>>(),
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    ))
}

/// Prometheus metrics endpoint for observability
pub async fn metrics_endpoint(State(state): State<AppState>) -> Result<String, StatusCode> {
    use prometheus::Encoder;

    // Update memory usage gauges before serving metrics
    let users_in_cache = state.users_in_cache();
    metrics::ACTIVE_USERS.set(users_in_cache as i64);

    // Aggregate metrics across all users
    let (mut total_working, mut total_session, mut total_longterm, mut total_heap) =
        (0i64, 0i64, 0i64, 0i64);
    let mut total_vectors = 0i64;

    // CACHED users only. `list_users()` is a FILESYSTEM listing, so pairing it with
    // `get_user_memory()` opened a RocksDB instance for every user not currently
    // resident — meaning a Prometheus scrape monotonically grew memory and FD usage
    // AND mutated the user cache. That churn drives eviction, which is the condition
    // the graph/user-memory reopen race needs, so scraping made an unrelated 500
    // more likely. Metrics now describe what is resident, which is also the honest
    // reading of a gauge that is set from resident state.
    //
    // The take(100) cap is retained from the original: it truncates the aggregate
    // silently, but max_users_in_memory bounds the cache well below it in practice.
    for user_id in state.list_cached_users().iter().take(100) {
        if let Ok(memory_sys) = state.get_user_memory(user_id) {
            if let Some(guard) = memory_sys.try_read() {
                let stats = guard.stats();
                total_working += stats.working_memory_count as i64;
                total_session += stats.session_memory_count as i64;
                total_longterm += stats.long_term_memory_count as i64;
                total_heap += (stats.total_memories * 250) as i64;
                total_vectors += stats.total_memories as i64;
            }
        }
    }

    // Set aggregate metrics
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["working"])
        .set(total_working);
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["session"])
        .set(total_session);
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["longterm"])
        .set(total_longterm);
    metrics::MEMORY_HEAP_BYTES_TOTAL.set(total_heap);
    metrics::VECTOR_INDEX_SIZE_TOTAL.set(total_vectors);

    // Gather and encode metrics
    let encoder = prometheus::TextEncoder::new();
    let metric_families = metrics::METRICS_REGISTRY.gather();

    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    String::from_utf8(buffer).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Context status request from Claude Code status line
#[derive(Debug, Deserialize)]
pub struct ContextStatusRequest {
    pub session_id: String,
    pub tokens_used: u64,
    pub tokens_budget: u64,
    pub current_dir: Option<String>,
    pub model: Option<String>,
}

/// Update context status from Claude Code status line script
pub async fn update_context_status(
    State(state): State<AppState>,
    Json(req): Json<ContextStatusRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Validate session_id length to prevent abuse
    if req.session_id.len() > 128 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "session_id must be 128 characters or fewer"
            })),
        );
    }

    // Enforce size cap on context_sessions map to prevent memory exhaustion
    if !state.context_sessions().contains_key(&req.session_id)
        && state.context_sessions().len() >= 10_000
    {
        tracing::warn!(
            "context_sessions at capacity (10,000), rejecting new session_id={}",
            &req.session_id
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "too many active context sessions"
            })),
        );
    }

    let percent_used = if req.tokens_budget > 0 {
        ((req.tokens_used as f64 / req.tokens_budget as f64) * 100.0) as u8
    } else {
        0
    };

    let status = ContextStatus {
        session_id: Some(req.session_id.clone()),
        tokens_used: req.tokens_used,
        tokens_budget: req.tokens_budget,
        percent_used,
        current_task: req.current_dir,
        model: req.model,
        updated_at: chrono::Utc::now(),
    };

    state
        .context_sessions()
        .insert(req.session_id.clone(), status.clone());

    state.broadcast_context(status);

    state.emit_event(MemoryEvent {
        event_type: "CONTEXT_UPDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: "system".to_string(),
        memory_id: Some(req.session_id),
        content_preview: Some(format!(
            "{}% ({}/{})",
            percent_used, req.tokens_used, req.tokens_budget
        )),
        memory_type: Some("Context".to_string()),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "success": true,
            "percent_used": percent_used
        })),
    )
}

/// Get all active context sessions (auto-cleans stale sessions > 5 mins old)
pub async fn get_context_status(State(state): State<AppState>) -> Json<Vec<ContextStatus>> {
    let now = chrono::Utc::now();
    let stale_threshold = chrono::Duration::minutes(5);

    let stale_ids: Vec<String> = state
        .context_sessions()
        .iter()
        .filter(|r| now - r.value().updated_at > stale_threshold)
        .map(|r| r.key().clone())
        .collect();

    for id in stale_ids {
        state.context_sessions().remove(&id);
    }

    let mut sessions: Vec<ContextStatus> = state
        .context_sessions()
        .iter()
        .map(|r| r.value().clone())
        .collect();
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Json(sessions)
}

#[cfg(test)]
mod write_latch_reporting {
    use super::*;
    use crate::storage::write_latch::WriteLatch;
    use std::path::Path;

    const PRODUCTION_MESSAGE: &str =
        "IO error: No space left on device: While open a file for appending: /store/000585.log";

    /// Every field but `status` / `write_latch` is filler — the assertions are
    /// about the two the latch decides.
    fn response_json(latch: &WriteLatch) -> serde_json::Value {
        let (status, write_latch) = write_latch_fields(latch);
        let response = HealthResponse {
            status,
            version: "test".to_string(),
            build: "0".to_string(),
            built_at: "test".to_string(),
            requested_storage_backend: "rocksdb".to_string(),
            effective_storage_backend: "rocksdb".to_string(),
            users_count: 0,
            users_in_cache: 0,
            user_evictions: 0,
            max_cache_size: 0,
            embedder: None,
            tenants_with_data: crate::storage::boot_guard::BackendCensus {
                rocksdb: 0,
                redb: None,
                measured_at: chrono::Utc::now(),
            },
            embedding_spaces: Vec::new(),
            orphaned_corpora: 0,
            space_migrations: Vec::new(),
            write_latch,
        };
        serde_json::to_value(&response).expect("HealthResponse serializes")
    }

    fn latch_at(latch: &WriteLatch, path: &str) {
        latch.note_write_error(Path::new(path), &anyhow::anyhow!("{PRODUCTION_MESSAGE}"));
    }

    #[test]
    fn an_unlatched_store_serializes_as_healthy_with_no_latch_field() {
        let body = response_json(&WriteLatch::new());
        assert_eq!(body["status"], "healthy");
        assert!(body.get("write_latch").is_none());
    }

    #[test]
    fn a_latched_store_serializes_as_degraded_and_carries_the_latch() {
        let latch = WriteLatch::new();
        latch_at(&latch, "/var/lib/veld/users/tenant-a/storage");

        let body = response_json(&latch);
        assert_eq!(body["status"], "degraded");
        let reported = body["write_latch"]
            .as_array()
            .expect("write_latch is an array");
        assert_eq!(reported.len(), 1);
        assert_eq!(
            reported[0]["store"],
            crate::storage::write_latch::store_id(Path::new(
                "/var/lib/veld/users/tenant-a/storage"
            ))
        );
        assert_eq!(reported[0]["condition"], "no_space_left_on_device");
        assert_eq!(reported[0]["failed_writes"], 1);
        assert_eq!(reported[0]["hard"], false);
    }

    #[test]
    fn two_latched_stores_both_appear_and_clearing_one_leaves_the_other() {
        let latch = WriteLatch::new();
        let a = "/var/lib/veld/users/tenant-a/storage";
        let b = "/var/lib/veld/users/tenant-b/storage";
        latch_at(&latch, a);
        latch_at(&latch, b);

        let body = response_json(&latch);
        assert_eq!(body["status"], "degraded");
        assert_eq!(
            body["write_latch"].as_array().expect("array").len(),
            2,
            "one process runs one store per tenant; the response must name both"
        );

        latch.note_write_ok(Path::new(a));
        let body = response_json(&latch);
        assert_eq!(
            body["status"], "degraded",
            "a write landing on one store says nothing about the other"
        );
        let remaining = body["write_latch"].as_array().expect("array");
        assert_eq!(remaining.len(), 1);
        assert_eq!(
            remaining[0]["store"],
            crate::storage::write_latch::store_id(Path::new(b))
        );
    }

    /// `/health` is on the unauthenticated probe router (`PROBE_PATHS`), and
    /// every per-user store directory is `base.join(user_id)`
    /// (`path_guard.rs:3`). The absence is asserted over the real serialized
    /// body rather than by checking that a redaction step ran.
    #[test]
    fn the_public_body_discloses_no_store_path_and_no_tenant_id() {
        const BASE: &str = "/var/lib/veld/users";
        const TENANT: &str = "tenant-7f3a9c2b";
        let store = format!("{BASE}/{TENANT}/storage");

        let latch = WriteLatch::new();
        latch.note_write_error(
            Path::new(&store),
            &anyhow::anyhow!(
                "IO error: No space left on device: While open a file for appending: {store}/000585.log"
            )
            .context("Failed to put memory 4c1f0b9e-77a2-4d35-9f10-8e2b6d4a1c33 in RocksDB"),
        );

        let body = response_json(&latch);
        let latch_field = serde_json::to_string(&body["write_latch"]).expect("serializes");
        let whole_body = serde_json::to_string(&body).expect("serializes");

        // Floor first: without this the absences below pass over an empty array.
        assert_eq!(body["status"], "degraded");
        assert!(latch_field.contains(&crate::storage::write_latch::store_id(Path::new(&store))));
        assert!(latch_field.contains("\"failed_writes\":1"));

        assert!(
            !whole_body.contains(TENANT),
            "the tenant id reached an unauthenticated body: {whole_body}"
        );
        assert!(
            !whole_body.contains(BASE),
            "the host filesystem layout reached an unauthenticated body: {whole_body}"
        );
        assert!(
            !whole_body.contains("000585.log"),
            "the failing file reached an unauthenticated body: {whole_body}"
        );
        assert!(
            !whole_body.contains("4c1f0b9e"),
            "the failing memory's id reached an unauthenticated body: {whole_body}"
        );
        assert!(
            !latch_field.contains('/'),
            "a path separator survived in the published latch: {latch_field}"
        );
    }

    #[test]
    fn ready_stays_200_through_a_soft_latch() {
        let latch = WriteLatch::new();
        latch_at(&latch, "/var/lib/veld/users/tenant-a/storage");

        let (code, status, hard_stores) = ready_decision(true, &latch);
        assert_eq!(code, StatusCode::OK);
        assert_eq!(status, "ready");
        assert_eq!(hard_stores, 0);
    }

    #[test]
    fn ready_returns_503_when_any_store_is_hard_latched() {
        let latch = WriteLatch::new();
        let a = "/var/lib/veld/users/tenant-a/storage";
        let b = "/var/lib/veld/users/tenant-b/storage";
        latch_at(&latch, a);
        latch_at(&latch, b);
        latch.mark_hard(Path::new(b));

        let (code, status, hard_stores) = ready_decision(true, &latch);
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(status, "not_ready");
        assert_eq!(
            hard_stores, 1,
            "one of the two is hard, and that is the count"
        );
    }

    #[test]
    fn ready_is_unaffected_when_nothing_is_latched() {
        let latch = WriteLatch::new();
        assert_eq!(ready_decision(true, &latch).0, StatusCode::OK);
        assert_eq!(
            ready_decision(false, &latch).0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(ready_decision(false, &latch).2, 0);
    }
}
