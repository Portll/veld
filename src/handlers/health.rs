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
}

/// Main health check endpoint
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let users_in_cache = state.users_in_cache();
    let user_evictions = state.user_evictions();
    let config = state.server_config();

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("VELD_VERSION_FULL").to_string(),
        build: env!("VELD_BUILD_NUMBER").to_string(),
        built_at: env!("VELD_BUILD_TIMESTAMP").to_string(),
        requested_storage_backend: config.requested_storage_backend.to_string(),
        effective_storage_backend: config.effective_storage_backend.to_string(),
        users_count: state.list_users().len(),
        users_in_cache,
        user_evictions,
        max_cache_size: config.max_users_in_memory,
    })
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
pub async fn health_ready(
    State(state): State<AppState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let ready = state.is_ready();
    let status_code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let status_str = if ready { "ready" } else { "not_ready" };

    let users_in_cache = state.users_in_cache();
    (
        status_code,
        Json(serde_json::json!({
            "status": status_str,
            "version": env!("VELD_VERSION_FULL"),
            "effective_storage_backend": state.server_config().effective_storage_backend.as_str(),
            "users_in_cache": users_in_cache,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
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
        .get_user_earth(&user_id)
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
pub async fn health_index(
    State(state): State<AppState>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Read health across cached users only — never trigger lazy init from a
    // public probe (otherwise every probe opens RocksDB for every on-disk
    // user, exhausting FDs).
    let mut total_vectors: usize = 0;
    let mut total_incremental: usize = 0;
    let mut users_checked: usize = 0;
    let mut users_needing_rebuild: usize = 0;
    for user_id in state.list_cached_users() {
        if let Ok(memory) = state.get_user_earth(&user_id) {
            let guard = memory.read();
            let h = guard.index_health();
            total_vectors += h.total_vectors;
            total_incremental += h.incremental_inserts;
            if h.needs_rebuild {
                users_needing_rebuild += 1;
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
        .get_user_earth(&user_id)
        .map_err(AppError::Internal)?;
    let guard = memory.read();
    let health = guard.index_health();
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "user_id": user_id,
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

    for user_id in state.list_users().iter().take(100) {
        if let Ok(memory_sys) = state.get_user_earth(user_id) {
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
