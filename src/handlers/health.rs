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

/// Readiness of the process, not of one tenant: 503 while the shared stores are
/// unusable or when ANY store is hard-latched. A soft latch is still inside its
/// bounded recovery and the next write may land; a tenant still opening delays
/// only that tenant's requests (they wait on the blocking pool, not on a worker),
/// so it is reported as `tenant_open_waiters` without failing readiness, which a
/// load balancer would otherwise apply to every tenant on every reopen after an
/// idle eviction. Returns the hard-latched store count, which the status code
/// cannot carry and which is the difference between one tenant affected and all
/// of them.
fn ready_decision(
    shared_stores_ready: bool,
    latch: &crate::storage::write_latch::WriteLatch,
) -> (StatusCode, &'static str, usize) {
    let hard_stores = latch.hard_count();
    if shared_stores_ready && hard_stores == 0 {
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
        write_latch,
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
pub async fn health_ready(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    let tenant_open_waiters = state.tenant_open_waiters();
    let (status_code, status_str, hard_stores) =
        ready_decision(state.is_ready(), crate::storage::write_latch::global());

    let users_in_cache = state.users_in_cache();
    let mut body = serde_json::json!({
        "status": status_str,
        "version": env!("VELD_VERSION_FULL"),
        "effective_storage_backend": state.server_config().effective_storage_backend.as_str(),
        "users_in_cache": users_in_cache,
        "tenant_open_waiters": tenant_open_waiters,
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

    // The cache itself, never the accessors: a tenant evicted since it was last seen would
    // otherwise be reopened by this probe.
    let Some(memory) = state.user_memories.get(&user_id) else {
        let exists_on_disk = state.list_users().contains(&user_id);
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
    };

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
    // The graph cache can expire a graph its tenant's memory still holds, so that handle is the
    // fallback; neither lookup opens anything.
    let graph = state
        .graph_memories
        .get(&user_id)
        .or_else(|| guard.graph_memory().cloned());
    let graph_stats = if let Some(graph) = graph {
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
    for user_id in state.list_cached_users() {
        // The cache itself, not get_user_memory: a tenant evicted since the listing
        // would otherwise be reopened by this probe, on the worker serving it.
        if let Some(memory) = state.user_memories.get(&user_id) {
            let guard = memory.read();
            let h = guard.index_health();
            total_vectors += h.total_vectors;
            total_incremental += h.incremental_inserts;
            if h.needs_rebuild {
                users_needing_rebuild += 1;
            }
            // Stored-vs-indexed, counted the same aggregate way as the rebuild
            // flag above. `stored` is the maintained count, not a key scan: an
            // unauthenticated probe must not cost a store read per cached user.
            // The exact population check is the authenticated per-user route.
            let (stored, vec_idx, lex_idx) = guard.index_coverage_cached();
            if vec_idx < stored || lex_idx < stored {
                users_with_shortfall += 1;
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
            // Kept for existing clients: a maintained count is never unreadable.
            "users_coverage_unreadable_count": 0,
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
        .get_user_memory_async(&user_id)
        .await
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
        // The cache itself: a tenant evicted since the listing is skipped, not reopened.
        if let Some(memory_sys) = state.user_memories.get(user_id) {
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

    /// Every kind of store driven through its real write paths into a NoSpace
    /// refusal: the database the write targeted latches and `/health` names it.
    ///
    /// The latch these sites record into is the process-global one, so another
    /// test's store may be in the body too; each assertion looks for its own.
    #[cfg(feature = "rocksdb")]
    mod write_sites {
        use super::*;
        use crate::graph_memory::EntityNode;
        use crate::memory::storage::wire_fixtures::memory_with_session;
        use crate::memory::storage::{MemoryStorage, Modality};
        use crate::storage::legacy_rocksdb::{
            RocksDbAuditStore, RocksDbColumnFamilyStore, RocksDbColumnStore,
            RocksDbDefaultKeyValueStore, RocksDbGraphStore,
        };
        use crate::storage::write_latch::{chain_is_enospc, global, injected, store_id};
        use crate::storage::{AuditLogEntry, AuditStore, ColumnStore, GraphStore, KeyValueStore};
        use std::sync::Arc;

        fn assert_reported(store: &Path, site: &str) {
            let latch = global();
            let snapshot = latch
                .snapshot_at(store)
                .unwrap_or_else(|| panic!("{site}: the refusal did not latch its store"));

            let body = response_json(latch);
            assert_eq!(body["status"], "degraded", "{site}");
            let entry = body["write_latch"]
                .as_array()
                .expect("write_latch is an array")
                .iter()
                .find(|entry| entry["store"] == store_id(store))
                .unwrap_or_else(|| panic!("{site}: /health does not name the store: {body}"));
            assert_eq!(entry["condition"], "no_space_left_on_device", "{site}");
            assert!(entry["failed_writes"].as_u64() >= Some(1), "{site}");

            // Hard only when the disk had room for the flush+retry, which then
            // met the same refusal; a full disk leaves it soft and ready.
            if snapshot.hard {
                let (code, _, hard_stores) = ready_decision(true, latch);
                assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "{site}");
                assert!(hard_stores >= 1, "{site}");
            }
        }

        /// Refuse `write` for want of space and check the latch and the report;
        /// then let the same write through and check that it clears the store.
        fn drive(store: &Path, site: &str, write: impl Fn() -> anyhow::Result<()>) {
            let _latching = crate::storage::write_latch::test_serial::latching();
            injected::arm(store);
            let refused = write();
            injected::disarm(store);

            let err = refused
                .err()
                .unwrap_or_else(|| panic!("{site}: the refusal never reached the caller"));
            assert!(
                chain_is_enospc(&err),
                "{site}: the caller lost the NoSpace cause, so it answers 500, not 503: {err:#}"
            );
            assert_reported(store, site);

            write().unwrap_or_else(|e| panic!("{site}: the write after the refusal failed: {e:#}"));
            assert!(
                !global().is_latched_at(store),
                "{site}: a write that landed left the latch standing"
            );
        }

        #[test]
        fn index_delete_and_multimodal_writes_latch_the_memory_database() {
            let dir = tempfile::tempdir().expect("tempdir");
            let storage = MemoryStorage::new(dir.path(), None).expect("open storage");
            let db = storage.db();
            let store = db.path();
            let memory = memory_with_session("a memory the disk refuses");
            storage.store(&memory).expect("seed the memory");
            let facts = RocksDbDefaultKeyValueStore::new(Arc::clone(&db));
            let holder = memory.id.0.to_string();

            drive(store, "multimodal store", || {
                storage.store_with_multimodal_vectors(&memory, Modality::Text, vec![7])
            });
            drive(store, "modality vectors", || {
                storage.update_modality_vectors(&memory.id, Modality::Image, vec![8])
            });
            drive(store, "index rebuild", || {
                storage.rebuild_index().map(|_| ())
            });
            drive(store, "retrieval counter", || {
                storage.increment_retrieval_count().map(|_| ())
            });
            drive(store, "interference records", || {
                storage.save_interference_records(&holder, &[])
            });
            drive(store, "default-keyspace view", || {
                facts.put_bytes(b"facts:latch", b"1")
            });
            drive(store, "delete", || storage.delete_with_vectors(&memory.id));
        }

        /// The double key: `store` once latched the tenant directory while the
        /// batch writer over the same database cleared `<tenant>/storage`, so a
        /// latch raised by one stood after the other had written.
        #[test]
        fn a_write_through_any_adapter_clears_the_latch_on_its_database() {
            let dir = tempfile::tempdir().expect("tempdir");
            let storage = MemoryStorage::new(dir.path(), None).expect("open storage");
            let db = storage.db();
            let memory = memory_with_session("refused by the record write");

            let _latching = crate::storage::write_latch::test_serial::latching();
            injected::arm(db.path());
            let refused = storage.store(&memory);
            injected::disarm(db.path());
            assert!(chain_is_enospc(
                &refused.expect_err("the armed refusal reaches the caller")
            ));
            assert_reported(db.path(), "store");

            let mut batch = crate::storage::WriteBatch::new();
            batch.put_cf("kv", b"dials".to_vec(), b"1".to_vec());
            storage.write_batch(&batch).expect("the batch lands");
            assert!(
                !global().is_latched_at(db.path()),
                "the batch writer cleared a different key than the record write latched"
            );
        }

        #[test]
        fn graph_writes_latch_the_graph_database() {
            let dir = tempfile::tempdir().expect("tempdir");
            let graph = RocksDbGraphStore::open(dir.path(), None).expect("open graph");
            let store = graph
                .inner()
                .rocksdb_for_backup()
                .expect("a rocksdb graph")
                .path()
                .to_path_buf();
            let now = chrono::Utc::now();
            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(),
                name: "harbour".to_string(),
                labels: Vec::new(),
                created_at: now,
                last_seen_at: now,
                mention_count: 1,
                summary: String::new(),
                attributes: Default::default(),
                name_embedding: None,
                salience: 0.4,
                is_proper_noun: true,
                pii_classification: Default::default(),
            };

            drive(&store, "entity put", || graph.put_entity(&entity));
            drive(&store, "entity delete", || {
                graph.inner().delete_entity(&entity.uuid).map(|_| ())
            });
            drive(&store, "graph flush", || graph.flush());
        }

        /// One database behind todos, reminders, files, feedback, context
        /// blocks and audit, as `bootstrap_shared_stores` opens it.
        #[test]
        fn shared_store_writes_latch_the_shared_database() {
            let dir = tempfile::tempdir().expect("tempdir");
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);
            opts.create_missing_column_families(true);
            let db = Arc::new(
                rocksdb::DB::open_cf(&opts, dir.path(), ["default", "audit", "todos", "feedback"])
                    .expect("open the shared database"),
            );
            let store = db.path().to_path_buf();
            let columns = RocksDbColumnStore::new(Arc::clone(&db));
            let feedback = RocksDbColumnFamilyStore::new(Arc::clone(&db), "feedback");
            let audit = RocksDbAuditStore::new(Arc::clone(&db), "audit");
            let event = AuditLogEntry {
                timestamp: chrono::Utc::now(),
                event_type: "remember".to_string(),
                memory_id: "m-1".to_string(),
                details: String::new(),
            };

            drive(&store, "todo put", || columns.put_cf("todos", b"t-1", b"1"));
            drive(&store, "todo delete", || columns.delete_cf("todos", b"t-1"));
            drive(&store, "feedback put", || feedback.put_bytes(b"f-1", b"1"));
            drive(&store, "audit append", || {
                audit.append_event("tenant", &event)
            });
            drive(&store, "audit rotation", || {
                audit
                    .rotate_events("tenant", 0, chrono::Utc::now())
                    .map(|_| ())
            });
            drive(&store, "shared flush", || columns.flush_cf("todos"));
        }
    }

    /// Every kind of redb store driven through its real write path into a NoSpace
    /// failure at its file. redb poisons the handle, so the file latches hard at
    /// once, `/health` names it, readiness fails, and the next write on the handle
    /// is refused with the latch (503) rather than redb's `PreviousIo` (500).
    ///
    /// Each test ends by dropping the store and opening the file again, which
    /// clears the latch: it is process-global, and a hard entry left standing
    /// would fail the readiness assertions of every test after it.
    #[cfg(feature = "storage-redb")]
    mod redb_write_sites {
        use super::*;
        use crate::errors::AppError;
        use crate::graph_memory::EntityNode;
        use crate::memory::types::Experience;
        use crate::memory::{Memory, MemoryId};
        use crate::storage::redb::{
            open_shared_database, RedbAuditStore, RedbBatchWriter, RedbColumnStore, RedbGraphStore,
            RedbKeyValueStore, RedbPrimaryMemoryStore,
        };
        use crate::storage::redb_latch::injected;
        use crate::storage::write_latch::{chain_is_enospc, global, store_id, test_serial};
        use crate::storage::{
            AuditLogEntry, AuditStore, BatchWriter, ColumnStore, GraphStore, KeyValueStore,
            PrimaryMemoryStore,
        };
        use std::sync::Arc;

        /// A write on a poisoned handle answers what a latched RocksDB write does.
        fn assert_refused_with_the_latch(site: &str, refused: anyhow::Result<()>) {
            let err = refused
                .err()
                .unwrap_or_else(|| panic!("{site}: a poisoned handle took a write"));
            assert!(
                chain_is_enospc(&err),
                "{site}: refused with redb's own error, which answers 500: {err:#}"
            );
            let response = AppError::from(err);
            assert_eq!(
                response.status_code(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{site}"
            );
            assert_eq!(response.code(), "STORAGE_WRITE_LATCHED", "{site}");
        }

        /// Fail `write` at `file` for want of space and check every reader of the
        /// latch, then repeat it on the poisoned handle.
        fn drive(file: &Path, site: &str, write: impl Fn() -> anyhow::Result<()>) {
            injected::arm(file);
            let failed = write();
            injected::disarm(file);
            let err = failed
                .err()
                .unwrap_or_else(|| panic!("{site}: the refusal never reached the caller"));
            assert!(chain_is_enospc(&err), "{site}: {err:#}");

            let latch = global();
            let snapshot = latch
                .snapshot_at(file)
                .unwrap_or_else(|| panic!("{site}: the failed write did not latch its file"));
            assert!(
                snapshot.hard,
                "{site}: a poisoned redb handle is hard at once"
            );

            let body = response_json(latch);
            assert_eq!(body["status"], "degraded", "{site}");
            let entry = body["write_latch"]
                .as_array()
                .expect("write_latch is an array")
                .iter()
                .find(|entry| entry["store"] == store_id(file))
                .unwrap_or_else(|| panic!("{site}: /health does not name the file: {body}"));
            assert_eq!(entry["condition"], "no_space_left_on_device", "{site}");
            assert_eq!(entry["hard"], true, "{site}");

            let (code, status, hard_stores) = ready_decision(true, latch);
            assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "{site}");
            assert_eq!(status, "not_ready", "{site}");
            assert!(hard_stores >= 1, "{site}");

            assert_refused_with_the_latch(site, write());
        }

        fn memory(content: &str) -> Memory {
            let experience = Experience {
                content: content.to_string(),
                ..Default::default()
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

        #[test]
        fn a_memory_write_latches_the_memory_store_file() {
            let _latching = test_serial::latching();
            let dir = tempfile::tempdir().expect("tempdir");
            let file = dir.path().join("storage.redb");
            let store = RedbPrimaryMemoryStore::open(dir.path()).expect("open");
            store.store(&memory("seed")).expect("seed write");
            let refused = memory("a memory the disk refuses");

            drive(&file, "memory store", || store.store(&refused));
            assert_refused_with_the_latch("memory delete", store.delete(&refused.id));
            assert_refused_with_the_latch("memory flush", store.flush());

            drop(store);
            RedbPrimaryMemoryStore::open(dir.path()).expect("reopen");
            assert!(
                !global().is_latched_at(&file),
                "the reopen did not clear it"
            );
        }

        #[test]
        fn a_graph_write_latches_the_graph_file() {
            let _latching = test_serial::latching();
            let dir = tempfile::tempdir().expect("tempdir");
            let file = dir.path().join("graph.redb");
            let graph = RedbGraphStore::open(dir.path()).expect("open");
            let now = chrono::Utc::now();
            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(),
                name: "harbour".to_string(),
                labels: Vec::new(),
                created_at: now,
                last_seen_at: now,
                mention_count: 1,
                summary: String::new(),
                attributes: Default::default(),
                name_embedding: None,
                salience: 0.4,
                is_proper_noun: true,
                pii_classification: Default::default(),
            };

            drive(&file, "entity put", || graph.put_entity(&entity));

            drop(graph);
            RedbGraphStore::open(dir.path()).expect("reopen");
            assert!(!global().is_latched_at(&file));
        }

        #[test]
        fn key_value_and_audit_writes_latch_their_files() {
            let _latching = test_serial::latching();
            let dir = tempfile::tempdir().expect("tempdir");
            let kv_file = dir.path().join("feedback.redb");
            let audit_file = dir.path().join("audit.redb");
            let kv = RedbKeyValueStore::open(dir.path(), "feedback").expect("open kv");
            let audit = RedbAuditStore::open(dir.path()).expect("open audit");
            let event = AuditLogEntry {
                timestamp: chrono::Utc::now(),
                event_type: "remember".to_string(),
                memory_id: "m-1".to_string(),
                details: String::new(),
            };

            drive(&kv_file, "kv put", || kv.put_bytes(b"f-1", b"1"));
            assert!(
                audit.append_event("tenant", &event).is_ok(),
                "a full disk under one file says nothing about another"
            );
            drive(&audit_file, "audit append", || {
                audit.append_event("tenant", &event)
            });

            drop(kv);
            drop(audit);
            RedbKeyValueStore::open(dir.path(), "feedback").expect("reopen kv");
            RedbAuditStore::open(dir.path()).expect("reopen audit");
            assert!(!global().is_latched_at(&kv_file));
            assert!(!global().is_latched_at(&audit_file));
        }

        /// One database behind todos, reminders, files, feedback, context blocks
        /// and audit, and every adapter over it is refused once it is poisoned.
        #[test]
        fn shared_store_writes_latch_the_shared_file() {
            let _latching = test_serial::latching();
            let dir = tempfile::tempdir().expect("tempdir");
            let file = dir.path().join("shared.redb");
            let database = open_shared_database(dir.path()).expect("open");
            let columns = RedbColumnStore::new(Arc::clone(&database));
            let writer = RedbBatchWriter::new(Arc::clone(&database));
            columns.put_cf("todos", b"t-0", b"1").expect("seed write");

            drive(&file, "todo put", || columns.put_cf("todos", b"t-1", b"1"));
            let mut batch = crate::storage::WriteBatch::new();
            batch.put_cf("todos", b"t-2".to_vec(), b"1".to_vec());
            assert_refused_with_the_latch("batch", writer.write_batch(&batch));
            assert_refused_with_the_latch("todo delete", columns.delete_cf("todos", b"t-0"));
            assert_refused_with_the_latch("shared flush", columns.flush_cf("todos"));

            drop(columns);
            drop(writer);
            drop(database);
            let reopened = open_shared_database(dir.path()).expect("reopen");
            assert!(!global().is_latched_at(&file));
            RedbColumnStore::new(reopened)
                .put_cf("todos", b"t-1", b"1")
                .expect("the reopened handle writes");
        }
    }
}

#[cfg(test)]
mod probes_while_a_tenant_opens {
    //! Each first request for a tenant still opening used to park a tokio worker on the
    //! tenant's init lock. On 2026-10-19 the live server took 11.7 s to open its largest
    //! tenant, and six websocket handshakes that had waited for a worker all logged within
    //! 0.4 ms of the open finishing; the redb replica's open of the same tenant took 10 min.
    use super::*;
    use crate::handlers::router::build_probe_routes;
    use crate::handlers::test_helpers::{get, get_unauthenticated, post_json, send, TestHarness};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    const WORKERS: usize = 2;
    const TENANT: &str = "tenant-mid-open";

    /// Sends `uri` from a spawned task, which needs a free worker as a real connection's
    /// does, and waits on this thread, which is not a worker.
    fn probe(app: &axum::Router, uri: &str) -> Option<(StatusCode, serde_json::Value)> {
        let (tx, rx) = mpsc::channel();
        let app = app.clone();
        let request = get_unauthenticated(uri);
        tokio::spawn(async move {
            let _ = tx.send(send(app, request).await);
        });
        rx.recv_timeout(Duration::from_secs(5)).ok()
    }

    fn wait_for_open_waiters(manager: &MultiUserMemoryManager, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while manager.tenant_open_waiters() < count {
            assert!(
                Instant::now() < deadline,
                "only {} of {count} requests reached the tenant open",
                manager.tenant_open_waiters()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn probes_answer_while_first_requests_wait_on_a_tenant_open() {
        let _quiet = crate::storage::write_latch::test_serial::quiet().await;
        let harness = TestHarness::new();
        let app = harness.router().merge(build_probe_routes(harness.state()));

        // An open holds the tenant's init lock until it is done. Hold it the same way.
        let init_lock = harness.manager.user_memory_init_lock(TENANT);
        let (held_tx, held_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let opener = std::thread::spawn(move || {
            let _open = init_lock.lock();
            held_tx.send(()).expect("test waits for the hold");
            let _ = release_rx.recv();
        });
        held_rx.recv().expect("init lock held");

        // Twice as many first requests as the runtime has workers, one per hot path.
        let requests = vec![
            post_json(
                "/api/remember",
                &serde_json::json!({"user_id": TENANT, "content": "written while the tenant opened"}),
            ),
            post_json(
                "/api/recall",
                &serde_json::json!({"user_id": TENANT, "query": "tenant opened"}),
            ),
            post_json(
                "/api/upsert",
                &serde_json::json!({"user_id": TENANT, "external_id": "probe:1", "content": "upserted while the tenant opened"}),
            ),
            get(&format!("/api/list/{TENANT}")),
        ];
        assert_eq!(requests.len(), WORKERS * 2);
        let waiters: Vec<_> = requests
            .into_iter()
            .map(|request| tokio::spawn(send(app.clone(), request)))
            .collect();
        wait_for_open_waiters(&harness.manager, WORKERS * 2);

        let health = probe(&app, "/health");
        let ready = probe(&app, "/health/ready");

        release_tx.send(()).expect("opener waits for release");
        opener.join().expect("opener thread");
        for waiter in waiters {
            let (status, body) = waiter.await.expect("request task");
            assert!(status.is_success(), "{status}: {body}");
        }
        let ready_after = probe(&app, "/health/ready");

        let (status, _) =
            health.expect("/health did not answer while requests waited on a tenant open");
        assert_eq!(status, StatusCode::OK);

        let (status, body) =
            ready.expect("/health/ready did not answer while requests waited on a tenant open");
        assert_eq!(
            status,
            StatusCode::OK,
            "an open delays its tenant, not the process: {body}"
        );
        assert_eq!(body["tenant_open_waiters"], WORKERS * 2);

        let (status, body) = ready_after.expect("/health/ready answers after the open");
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["tenant_open_waiters"], 0);
        assert!(body.get("reason").is_none(), "{body}");
    }

    #[tokio::test]
    async fn the_async_accessor_hands_back_the_one_cached_tenant() {
        let harness = TestHarness::new();
        let opened = harness
            .manager
            .get_user_memory_async("tenant-a")
            .await
            .expect("first open");
        let cached = harness
            .manager
            .get_user_memory_async("tenant-a")
            .await
            .expect("cached");
        let sync = harness.manager.get_user_memory("tenant-a").expect("sync");
        assert!(std::sync::Arc::ptr_eq(&opened, &cached));
        assert!(std::sync::Arc::ptr_eq(&opened, &sync));
        assert_eq!(harness.manager.tenant_open_waiters(), 0);
    }

    #[tokio::test]
    async fn a_failed_open_is_not_left_counted_as_opening() {
        let harness = TestHarness::new();
        assert!(harness
            .manager
            .get_user_memory_async("../escape")
            .await
            .is_err());
        assert_eq!(harness.manager.tenant_open_waiters(), 0);
    }

    #[test]
    fn a_cached_graph_comes_back_without_a_blocking_task() {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};

        let harness = TestHarness::new();
        let opened = harness.manager.get_user_graph("tenant-g").expect("open");

        // One poll, outside any runtime: a lookup that went to the blocking pool would panic
        // in spawn_blocking, and one that waited would not be ready.
        let mut lookup = std::pin::pin!(harness.manager.get_user_graph_async("tenant-g"));
        let Poll::Ready(cached) = lookup
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        else {
            panic!("a cached graph did not come back on the first poll");
        };
        assert!(std::sync::Arc::ptr_eq(&opened, &cached.expect("cached")));
    }

    /// The test runtime has one thread. An open made on it would stall the timeout below
    /// until the init lock is let go, so the lock is let go after 5 s at most.
    #[tokio::test(flavor = "current_thread")]
    async fn an_uncached_graph_opens_off_the_runtime_thread_and_is_cached() {
        const GRAPH_TENANT: &str = "tenant-graph-mid-open";
        let harness = TestHarness::new();
        assert!(harness.manager.graph_memories.get(GRAPH_TENANT).is_none());

        let init_lock = harness.manager.user_graph_init_lock(GRAPH_TENANT);
        let (held_tx, held_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let opener = std::thread::spawn(move || {
            let _open = init_lock.lock();
            held_tx.send(()).expect("test waits for the hold");
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
        });
        held_rx.recv().expect("init lock held");

        let manager = harness.manager.clone();
        let mut lookup =
            tokio::spawn(async move { manager.get_user_graph_async(GRAPH_TENANT).await });
        let waited = tokio::time::timeout(Duration::from_millis(200), &mut lookup).await;
        assert!(
            waited.is_err(),
            "the lookup finished while the graph's init lock was held, so it held the runtime \
             thread until the lock was let go"
        );

        release_tx.send(()).expect("opener waits for release");
        opener.join().expect("opener thread");
        let opened = lookup.await.expect("lookup task").expect("open");
        let cached = harness
            .manager
            .graph_memories
            .get(GRAPH_TENANT)
            .expect("the open cached the graph");
        assert!(std::sync::Arc::ptr_eq(&opened, &cached));
    }

    #[tokio::test]
    async fn per_user_readiness_and_metrics_leave_an_evicted_tenant_closed() {
        const EVICTED: &str = "tenant-evicted";
        let harness = TestHarness::new();
        harness
            .manager
            .get_user_memory_async(EVICTED)
            .await
            .expect("open");
        harness.manager.evict_user(EVICTED);

        let (status, body) = send(
            harness.router(),
            get(&format!("/api/health/ready?user_id={EVICTED}")),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["cached"], false, "{body}");
        assert_eq!(body["exists_on_disk"], true, "{body}");
        metrics_endpoint(State(harness.state()))
            .await
            .expect("metrics");

        assert!(
            harness.manager.user_memories.get(EVICTED).is_none(),
            "a cache-only probe reopened the tenant"
        );
        assert!(
            harness.manager.graph_memories.get(EVICTED).is_none(),
            "a cache-only probe reopened the tenant's graph"
        );
    }

    #[tokio::test]
    async fn per_user_readiness_reads_the_graph_its_tenant_holds() {
        const TENANT_G: &str = "tenant-graph-expired";
        let harness = TestHarness::new();
        harness
            .manager
            .get_user_memory_async(TENANT_G)
            .await
            .expect("open");
        harness.manager.graph_memories.invalidate(TENANT_G);
        harness.manager.graph_memories.run_pending_tasks();

        let (status, body) = send(
            harness.router(),
            get(&format!("/api/health/ready?user_id={TENANT_G}")),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["cached"], true, "{body}");
        assert!(body["graph_stats"].is_object(), "{body}");
        assert!(
            harness.manager.graph_memories.get(TENANT_G).is_none(),
            "the probe went through the graph accessor"
        );
    }
}
