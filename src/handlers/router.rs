//! Veld layer: product API surface (remember/recall/forget verbs, user-facing endpoints).
//! Router Configuration - Centralized route definitions
//!
//! This module builds the Axum router using handlers from the submodules.
//! Routes are organized by domain and split into public (no auth) and protected (auth required).

use axum::{
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;

use super::state::MultiUserMemoryManager;
use super::{
    ab_testing, admin, compression, consolidation, context_blocks, crud, external_dimensions,
    facts, files, gap_analysis, graph, health, ingest, integrations, lineage, mif, prompt_gen,
    recall, remember, search, seed, sessions, todos, user_auth, users, visualization, webhooks,
};

/// Application state type alias
pub type AppState = Arc<MultiUserMemoryManager>;

/// Structural circuit breaker — the **complete** list of paths reachable
/// without API-key authentication. `build_probe_routes` and
/// `build_public_routes` are derived from these consts; the
/// `public_router_has_no_per_user_handlers` test asserts no handler mounted
/// here reads `?user_id=` for per-tenant data.
///
/// Adding a path here must be a deliberate decision — if a future contributor
/// adds a route to one of these builders without updating the const, the
/// `public_router_paths_match_const` test fails. If the new route reads
/// `?user_id=`, the structural test fails. That stops the next
/// `health_ready`/`health_index` slip.
///
/// `/metrics` is conditionally public via `ServerConfig::metrics_public` and is
/// listed in `METRICS_PATH_IF_PUBLIC`; it is not in `PUBLIC_PATHS` because its
/// presence depends on runtime config.
pub const PROBE_PATHS: &[&str] = &[
    "/health",
    "/health/live",
    "/health/ready",
    "/health/index",
];

/// Non-probe public paths (no authentication required). See `PROBE_PATHS` for
/// the security contract and `public_router_has_no_per_user_handlers` for the
/// enforcement test.
pub const PUBLIC_PATHS: &[&str] = &[
    "/api/context/status",
    "/api/context_status",
    "/graph/view",
    "/api/admin/reset-rate-limit",
];

/// `/metrics` is mounted on the public router only when
/// `ServerConfig::metrics_public` is true; otherwise it lives behind auth.
pub const METRICS_PATH_IF_PUBLIC: &str = "/metrics";

/// Build the public Kubernetes probe routes — never rate-limited, always
/// public. These four paths must remain unconditionally accessible so
/// orchestrators can always determine liveness/readiness even under a
/// saturated rate limiter.
///
/// **Security contract:** none of these handlers may read `?user_id=` for
/// per-tenant data. Per-user readiness/index live on the authenticated
/// `/api/health/*` routes (see `build_protected_routes`).
pub fn build_probe_routes(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .route("/health/live", get(health::health_live))
        .route("/health/ready", get(health::health_ready))
        .route("/health/index", get(health::health_index))
        .with_state(state)
}

/// Build public (unauthenticated) routes.
///
/// These routes are optionally rate-limited (controlled by `VELD_PUBLIC_RATE_LIMIT`
/// in `ServerConfig`). Health probes are split into `build_probe_routes()` so
/// they stay accessible even when public-route rate limiting is enabled.
///
/// `/metrics` placement is controlled by `ServerConfig::metrics_public`:
///   - false (default): mounted under protected routes, requires API key.
///   - true: mounted here, rate-limited when `public_rate_limit` is true.
pub fn build_public_routes(state: AppState, metrics_public: bool) -> Router {
    let r = Router::new()
        // CONTEXT STATUS (GET only - status reads remain public)
        .route("/api/context/status", get(health::get_context_status))
        .route("/api/context_status", get(health::get_context_status)) // TUI GET alias
        // GRAPH VISUALIZATION (HTML viewer only — no memory data in the
        // static shell; dynamic data loads via authenticated API calls).
        .route("/graph/view", get(visualization::graph_view))
        // ADMIN OPERATIONAL ENDPOINTS — mounted on the public router so a
        // stuck rate limiter can never block its own recovery. The endpoint
        // enforces its own separate auth via X-Admin-API-Key +
        // VELD_ADMIN_API_KEY env var (handlers/admin.rs).
        .route(
            "/api/admin/reset-rate-limit",
            post(admin::reset_rate_limit),
        );

    let r = if metrics_public {
        r.route(METRICS_PATH_IF_PUBLIC, get(health::metrics_endpoint))
    } else {
        r
    };

    r.with_state(state)
}

/// Build the protected API routes (authentication required)
///
/// These routes require API key authentication and are rate-limited.
/// The auth middleware and rate limiter should be applied by the caller.
///
/// When `metrics_public` is false (default), `/metrics` is mounted here so
/// Prometheus scraping requires a valid API key.
pub fn build_protected_routes(state: AppState, metrics_public: bool) -> Router {
    let r = Router::new();

    // /metrics — protected by default, public when VELD_METRICS_PUBLIC=true
    let r = if !metrics_public {
        r.route("/metrics", get(health::metrics_endpoint))
    } else {
        r
    };

    // Phase C user-auth surface — session-token auth (separate from the
    // X-API-Key middleware that wraps the rest of the protected router).
    // The session middleware is applied here, not at the outer auth layer,
    // so the api-key check in `auth_middleware` is skipped for
    // `/api/user_auth/*` paths (see `crate::auth::auth_middleware`).
    let session_routes = Router::new()
        .route("/api/user_auth/2fa/enroll", post(user_auth::enroll_2fa))
        .route("/api/user_auth/2fa/confirm", post(user_auth::confirm_2fa))
        .route("/api/user_auth/logout", post(user_auth::logout))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            user_auth::require_user_session,
        ));
    let user_auth_routes = Router::new()
        .route("/api/user_auth/register", post(user_auth::register))
        .route("/api/user_auth/login", post(user_auth::login))
        .route("/api/user_auth/recover", post(user_auth::recover))
        .merge(session_routes);

    r
        .merge(user_auth_routes)
        // =================================================================
        // PER-USER HEALTH (auth required — tenant-bound via
        // resolve_request_user_id; replaces the leaked ?user_id= branches
        // formerly on the public probes)
        // =================================================================
        .route("/api/health/ready", get(health::health_ready_user))
        .route("/api/health/index", get(health::health_index_user))
        // =================================================================
        // CONTEXT SSE + STATUS (auth required — SSE leaks session/token/task info)
        // =================================================================
        .route("/api/context/sse", get(webhooks::context_status_sse))
        .route("/api/context/status", post(health::update_context_status))
        .route("/api/context_status", post(health::update_context_status)) // TUI POST alias
        // =================================================================
        // EXTERNAL WEBHOOKS (auth + rate limited — prevents unauthenticated memory injection)
        // =================================================================
        .route("/webhook/linear", post(integrations::linear_webhook))
        .route("/webhook/github", post(integrations::github_webhook))
        // =================================================================
        // REMEMBER/RECORD ENDPOINTS
        // =================================================================
        .route("/api/remember", post(remember::remember))
        .route("/api/remember/batch", post(remember::batch_remember))
        .route("/api/batch_remember", post(remember::batch_remember))
        .route("/api/upsert", post(remember::upsert_memory))
        // =================================================================
        // INGEST (multi-format text extraction + remember)
        // =================================================================
        .route("/api/ingest", post(ingest::ingest))
        // =================================================================
        // PROJECT SEEDING
        // =================================================================
        .route("/api/seed", post(seed::seed_project))
        // =================================================================
        // RECALL ENDPOINTS
        // =================================================================
        .route("/api/recall", post(recall::recall))
        .route("/api/recall/tracked", post(recall::recall_tracked))
        .route("/api/recall/tags", post(recall::recall_by_tags))
        .route("/api/recall/by-tags", post(recall::recall_by_tags)) // OpenAPI alias
        .route("/api/recall/date", post(recall::recall_by_date))
        // =================================================================
        // PROACTIVE CONTEXT & RELEVANCE
        // =================================================================
        .route("/api/context_summary", post(recall::context_summary))
        .route("/api/proactive_context", post(recall::proactive_context))
        .route("/api/context", post(recall::proactive_context)) // OpenAPI alias
        .route("/api/relevant", post(recall::surface_relevant))
        .route("/api/reinforce", post(recall::reinforce_feedback))
        // =================================================================
        // PROMPT GENERATION & ENTITY RESOLUTION
        // =================================================================
        .route("/api/prompt/gen", post(prompt_gen::prompt_gen))
        .route("/api/prompt/generate", post(prompt_gen::prompt_gen)) // alias
        .route("/api/entity/resolve", post(prompt_gen::resolve_entity))
        .route("/api/entity/attribute", post(prompt_gen::set_entity_attribute))
        .route("/api/entity/merge", post(prompt_gen::merge_entities))
        .route("/api/entity/alias", post(prompt_gen::add_entity_alias))
        // =================================================================
        // MEMORY CRUD OPERATIONS
        // =================================================================
        .route("/api/memory/{memory_id}", get(crud::get_memory))
        .route("/api/memories/{memory_id}", get(crud::get_memory)) // Cloudflare compat alias
        .route("/api/memory/{memory_id}/health", get(crud::get_memory_health)) // FIX-02: observability
        .route("/api/memory/{memory_id}", put(crud::update_memory))
        .route("/api/memory/{memory_id}", delete(crud::delete_memory))
        .route("/api/forget/{memory_id}", delete(crud::delete_memory)) // OpenAPI alias
        .route("/api/list/{user_id}", get(crud::list_memories)) // TUI uses this
        .route("/api/memories", post(crud::list_memories_post)) // POST version
        .route("/api/memories", get(crud::list_memories_get)) // Cloudflare compat alias
        .route("/api/memories/bulk", post(crud::bulk_delete_memories))
        .route("/api/memories/clear", post(crud::clear_all_memories))
        // =================================================================
        // ANCHOR (DECAY RESISTANCE)
        // =================================================================
        .route("/api/anchor", post(crud::anchor_memory))
        // =================================================================
        // AGENT-DIRECTED TIER MOVES
        // =================================================================
        .route("/api/memory/tier", post(crud::move_memory_tier))
        // =================================================================
        // FORGET OPERATIONS
        // =================================================================
        .route("/api/forget", post(crud::forget_by_id))
        .route("/api/forget/age", post(crud::forget_by_age))
        .route("/api/forget/importance", post(crud::forget_by_importance))
        .route("/api/forget/pattern", post(crud::forget_by_pattern))
        .route("/api/forget/tags", post(crud::forget_by_tags))
        .route("/api/forget/date", post(crud::forget_by_date))
        // =================================================================
        // USER MANAGEMENT
        // =================================================================
        .route("/api/users", get(users::list_users))
        .route("/api/users/{user_id}/stats", get(users::get_user_stats))
        .route("/api/users/{user_id}", delete(users::delete_user))
        .route("/api/stats", get(users::get_stats_query))
        // =================================================================
        // COMPRESSION
        // =================================================================
        .route("/api/memory/compress", post(compression::compress_memory))
        .route(
            "/api/memory/decompress",
            post(compression::decompress_memory),
        )
        .route("/api/storage/stats", get(compression::get_storage_stats))
        // =================================================================
        // ADVANCED SEARCH
        // =================================================================
        .route("/api/search/advanced", post(search::advanced_search))
        // =================================================================
        // STORAGE & INDEX MANAGEMENT
        // =================================================================
        .route("/api/storage/uncompressed", post(mif::get_uncompressed_old))
        .route(
            "/api/index/verify",
            post(consolidation::verify_index_integrity),
        )
        .route(
            "/api/index/repair",
            post(consolidation::repair_vector_index),
        )
        .route("/api/index/rebuild", post(consolidation::rebuild_index))
        .route("/api/index/reembed", post(consolidation::reembed_all))
        .route(
            "/api/storage/cleanup",
            post(consolidation::cleanup_corrupted),
        )
        .route("/api/storage/migrate", post(consolidation::migrate_legacy))
        // =================================================================
        // CONSOLIDATION & BACKUPS
        // =================================================================
        .route(
            "/api/consolidate",
            post(consolidation::consolidate_memories),
        )
        .route(
            "/api/consolidation/sleep",
            post(consolidation::sleep_phase_consolidation),
        )
        .route(
            "/api/consolidation/report",
            post(consolidation::get_consolidation_report),
        )
        .route(
            "/api/consolidation/events",
            get(consolidation::get_consolidation_events),
        )
        .route("/api/backup/create", post(consolidation::create_backup))
        .route("/api/backup/list", post(consolidation::list_backups))
        .route("/api/backups", post(consolidation::list_backups)) // MCP alias
        .route("/api/backup/verify", post(consolidation::verify_backup))
        .route("/api/backup/purge", post(consolidation::purge_backups))
        .route("/api/backups/purge", post(consolidation::purge_backups)) // MCP alias
        .route("/api/backup/restore", post(consolidation::restore_backup))
        // =================================================================
        // FACTS
        // =================================================================
        .route("/api/facts/list", post(facts::list_facts))
        .route("/api/facts/search", post(facts::search_facts))
        .route("/api/facts/by-entity", post(facts::facts_by_entity))
        .route("/api/facts/stats", post(facts::get_facts_stats))
        // =================================================================
        // TEMPORAL FACTS
        // =================================================================
        .route("/api/facts/temporal", post(facts::list_temporal_facts))
        .route(
            "/api/facts/temporal/search",
            post(facts::search_temporal_facts),
        )
        // =================================================================
        // CONTEXT BLOCKS (LETTA-STYLE MUTABLE AGENT STATE)
        // =================================================================
        .route(
            "/api/context/blocks",
            get(context_blocks::list_context_blocks),
        )
        .route(
            "/api/context/blocks/{key}",
            get(context_blocks::get_context_block),
        )
        .route(
            "/api/context/blocks/{key}",
            put(context_blocks::set_context_block),
        )
        .route(
            "/api/context/blocks/{key}",
            delete(context_blocks::delete_context_block),
        )
        // =================================================================
        // LINEAGE
        // =================================================================
        .route("/api/lineage/trace", post(lineage::lineage_trace))
        .route("/api/lineage/edges", post(lineage::lineage_list_edges))
        .route("/api/lineage/confirm", post(lineage::lineage_confirm_edge))
        .route("/api/lineage/reject", post(lineage::lineage_reject_edge))
        .route("/api/lineage/link", post(lineage::lineage_add_edge))
        .route("/api/lineage/stats", post(lineage::lineage_stats))
        .route(
            "/api/lineage/branches",
            post(lineage::lineage_list_branches),
        )
        .route("/api/lineage/branch", post(lineage::lineage_create_branch))
        // =================================================================
        // KNOWLEDGE GRAPH (ADVANCED)
        // =================================================================
        .route("/api/graph/{user_id}/stats", get(graph::get_graph_stats))
        .route(
            "/api/graph/{user_id}/universe",
            get(graph::get_memory_universe),
        )
        .route(
            "/api/graph/{user_id}/clear",
            delete(graph::clear_user_graph),
        )
        .route(
            "/api/graph/{user_id}/rebuild",
            post(graph::rebuild_user_graph),
        )
        .route("/api/graph/entity/find", post(graph::find_entity))
        .route("/api/graph/entities/all", post(graph::get_all_entities))
        .route(
            "/api/graph/relationship/invalidate",
            post(graph::invalidate_relationship),
        )
        .route("/api/graph/traverse", post(graph::traverse_graph))
        .route("/api/graph/episode/get", post(graph::get_episode))
        // =================================================================
        // GAP ANALYSIS & THOUGHT SURFACING
        // =================================================================
        .route("/api/gap/analyze", post(gap_analysis::analyze_gaps))
        .route("/api/gap/stats", post(gap_analysis::gap_stats))
        .route("/api/gap/voronoi", post(gap_analysis::voronoi_analysis))
        .route("/api/gap/persistence", post(gap_analysis::persistence_analysis))
        .route("/api/gap/mapper", post(gap_analysis::mapper_analysis))
        // =================================================================
        // KNOWLEDGE GRAPH (BASIC)
        // =================================================================
        .route("/api/graph/entity/add", post(mif::add_entity))
        .route("/api/graph/relationship/add", post(mif::add_relationship))
        // =================================================================
        // VISUALIZATION
        // =================================================================
        .route("/api/brain/{user_id}", get(visualization::get_brain_state))
        .route(
            "/api/graph/data/{user_id}",
            get(visualization::get_graph_data),
        )
        .route(
            "/api/visualization/{user_id}/stats",
            get(visualization::get_visualization_stats),
        )
        .route(
            "/api/visualization/{user_id}/dot",
            get(visualization::get_visualization_dot),
        )
        .route(
            "/api/visualization/build",
            post(visualization::build_visualization),
        )
        // =================================================================
        // TODOS
        // =================================================================
        .route("/api/todos", post(todos::list_todos))
        .route("/api/todos/list", post(todos::list_todos)) // TUI compatibility
        .route("/api/todos/add", post(todos::create_todo))
        .route("/api/todos/update", post(todos::update_todo))
        .route("/api/todos/complete", post(todos::complete_todo))
        .route("/api/todos/delete", post(todos::delete_todo))
        .route("/api/todos/reorder", post(todos::reorder_todo))
        .route("/api/todos/due", post(todos::list_due_todos))
        .route("/api/todos/{todo_id}", get(todos::get_todo))
        .route("/api/todos/{todo_id}", delete(todos::delete_todo)) // TUI uses DELETE
        .route("/api/todos/{todo_id}/update", post(todos::update_todo)) // TUI path style
        .route("/api/todos/{todo_id}/complete", post(todos::complete_todo)) // TUI path style
        .route("/api/todos/{todo_id}/reorder", post(todos::reorder_todo)) // TUI path style
        .route("/api/todos/{todo_id}/subtasks", get(todos::list_subtasks))
        .route("/api/todos/{todo_id}/dependents", get(todos::list_dependents))
        .route("/api/todos/{todo_id}/dependency_chain", get(todos::dependency_chain))
        .route("/api/todos/ready", post(todos::list_ready_todos))
        .route(
            "/api/todos/{todo_id}/comments",
            get(todos::list_todo_comments),
        )
        .route(
            "/api/todos/{todo_id}/comments",
            post(todos::add_todo_comment),
        )
        .route(
            "/api/todos/{todo_id}/comments/{comment_id}",
            put(todos::update_todo_comment),
        )
        .route(
            "/api/todos/{todo_id}/comments/{comment_id}/update",
            post(todos::update_todo_comment), // MCP alias
        )
        .route(
            "/api/todos/{todo_id}/comments/{comment_id}",
            delete(todos::delete_todo_comment),
        )
        .route("/api/todos/stats", post(todos::get_todo_stats)) // TUI uses POST
        // =================================================================
        // PROJECTS
        // =================================================================
        .route("/api/projects", post(todos::create_project)) // MCP: POST /api/projects
        .route("/api/projects/list", post(todos::list_projects)) // MCP alias
        .route("/api/projects/add", post(todos::create_project)) // Legacy alias
        .route("/api/projects/{project_id}", get(todos::get_project))
        .route("/api/projects/{project_id}", delete(todos::delete_project)) // Cloudflare compat alias
        .route(
            "/api/projects/{project_id}/update",
            post(todos::update_project),
        )
        .route(
            "/api/projects/{project_id}/delete",
            post(todos::delete_project),
        )
        // =================================================================
        // FILE MEMORY / CODEBASE INTEGRATION
        // =================================================================
        .route(
            "/api/projects/{project_id}/files",
            post(files::list_project_files),
        )
        .route(
            "/api/projects/{project_id}/scan",
            post(files::scan_project_codebase),
        )
        .route(
            "/api/projects/{project_id}/index",
            post(files::index_project_codebase),
        )
        .route(
            "/api/projects/{project_id}/files/search",
            post(files::search_project_files),
        )
        .route("/api/files/stats", get(files::get_file_stats))
        // =================================================================
        // REMINDERS
        // =================================================================
        .route("/api/reminders", post(todos::list_reminders))
        .route("/api/reminders/set", post(todos::create_reminder))
        .route("/api/remind", post(todos::create_reminder)) // MCP alias
        .route("/api/reminders/due", post(todos::get_due_reminders))
        .route("/api/reminders/check", post(todos::check_context_reminders))
        .route(
            "/api/reminders/context",
            post(todos::check_context_reminders),
        ) // MCP alias
        .route(
            "/api/reminders/{reminder_id}/dismiss",
            post(todos::dismiss_reminder),
        )
        .route(
            "/api/reminders/{reminder_id}/delete",
            post(todos::delete_reminder),
        )
        // =================================================================
        // SESSIONS
        // =================================================================
        .route("/api/sessions", post(sessions::list_sessions))
        .route("/api/sessions/stats", get(sessions::get_session_stats))
        .route("/api/sessions/end", post(sessions::end_session))
        .route("/api/sessions/{session_id}", get(sessions::get_session))
        // =================================================================
        // A/B TESTING
        // =================================================================
        .route("/api/ab/tests", get(ab_testing::list_ab_tests))
        .route("/api/ab/tests", post(ab_testing::create_ab_test))
        .route("/api/ab/tests/{test_id}", get(ab_testing::get_ab_test))
        .route(
            "/api/ab/tests/{test_id}",
            delete(ab_testing::delete_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/start",
            post(ab_testing::start_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/pause",
            post(ab_testing::pause_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/resume",
            post(ab_testing::resume_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/complete",
            post(ab_testing::complete_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/analyze",
            get(ab_testing::analyze_ab_test),
        )
        .route(
            "/api/ab/tests/{test_id}/impression",
            post(ab_testing::record_ab_impression),
        )
        .route(
            "/api/ab/tests/{test_id}/click",
            post(ab_testing::record_ab_click),
        )
        .route(
            "/api/ab/tests/{test_id}/feedback",
            post(ab_testing::record_ab_feedback),
        )
        .route("/api/ab/summary", get(ab_testing::get_ab_summary))
        // =================================================================
        // EXTERNAL DIMENSION PUSH (GRAPH TOPOLOGICAL HEALTH)
        // Sleight integration — accepts topology scores from evaluator
        // =================================================================
        .route(
            "/api/sleight/dimensions",
            post(external_dimensions::push_dimensions),
        )
        // backward-compat aliases
        .route(
            "/api/external/dimensions",
            post(external_dimensions::push_dimensions),
        )
        .route(
            "/api/wintermute/dimensions",
            post(external_dimensions::push_dimensions),
        )
        // =================================================================
        // EXTERNAL INTEGRATIONS (BULK SYNC)
        // =================================================================
        .route("/api/sync/linear", post(integrations::linear_sync))
        .route("/api/sync/github", post(integrations::github_sync))
        // =================================================================
        // WEBHOOKS & SSE (STREAMING)
        // =================================================================
        .route("/api/context/monitor", get(webhooks::context_monitor_ws))
        .route("/api/events/sse", get(webhooks::memory_events_sse))
        .route("/api/events", get(webhooks::memory_events_sse)) // TUI alias
        .route("/api/stream", get(webhooks::streaming_memory_ws))
        // =================================================================
        // MULTIMODAL & ROBOTICS SEARCH
        // =================================================================
        .route("/api/search/multimodal", post(search::multimodal_search))
        .route("/api/search/robotics", post(search::robotics_search))
        // =================================================================
        // MIF (Memory Interchange Format) v2
        // =================================================================
        .route("/api/export/mif", post(mif::export_mif))
        .route("/api/import/mif", post(mif::import_mif))
        .route("/api/mif/adapters", get(mif::list_adapters))
        // =================================================================
        // STATE
        // =================================================================
        .with_state(state)
}

/// Build the complete router with probe, public and protected routes.
///
/// Note: This function does NOT apply auth middleware or rate limiting.
/// The caller (server.rs) applies those layers to the appropriate sub-routers.
pub fn build_router(state: AppState, metrics_public: bool) -> Router {
    let probes = build_probe_routes(state.clone());
    let public = build_public_routes(state.clone(), metrics_public);
    let protected = build_protected_routes(state, metrics_public);

    Router::new().merge(probes).merge(public).merge(protected)
}

#[cfg(test)]
mod public_router_structural_guard {
    //! Structural circuit breaker (REMEDIATION_PLAN.md W2 / A3).
    //!
    //! These tests don't drive the router — they read this file as source and
    //! assert two invariants:
    //!
    //!   1. `build_probe_routes` and `build_public_routes` route exactly the
    //!      paths declared in `PROBE_PATHS` / `PUBLIC_PATHS` (+
    //!      `METRICS_PATH_IF_PUBLIC` when `metrics_public`). A drifted const
    //!      means a route was added to one builder without updating the
    //!      const — which is exactly how `health_index` ended up as the
    //!      unfixed twin of `health_ready`.
    //!
    //!   2. The handler bodies in `health.rs` for the public probes never
    //!      read `params.get("user_id")`. If a future contributor reintroduces
    //!      a `?user_id=` branch on a public probe (as commit a8f1299 did to
    //!      `health_ready`), this test fails before the leak can ship.
    //!
    //! Source-text assertions over reflection: axum's `Router` does not expose
    //! its route table for inspection, and adding a runtime registry would be
    //! a larger change than the structural rule warrants.
    use super::*;

    fn router_source() -> String {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/handlers/router.rs"))
            .expect("read router.rs")
    }

    fn health_source() -> String {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/handlers/health.rs"))
            .expect("read health.rs")
    }

    fn extract_routed_paths(source: &str, fn_name: &str) -> Vec<String> {
        // Locate the function body and harvest the first string literal of
        // every `.route("…", …)` call inside it. Cheap, deterministic, and
        // doesn't need a real parser.
        let fn_sig = format!("pub fn {}(", fn_name);
        let start = source
            .find(&fn_sig)
            .unwrap_or_else(|| panic!("function `{}` not found in router.rs", fn_name));
        let after = &source[start..];
        // Walk braces to find the matching close.
        let body_start = after.find('{').expect("function body opening brace");
        let mut depth = 0i32;
        let mut body_end = body_start;
        for (i, ch) in after[body_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        body_end = body_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let body = &after[body_start..=body_end];

        let mut paths = Vec::new();
        let mut rest = body;
        while let Some(idx) = rest.find(".route(") {
            rest = &rest[idx + ".route(".len()..];
            let q1 = rest.find('"').expect("opening quote in .route call");
            let after_q1 = &rest[q1 + 1..];
            let q2 = after_q1.find('"').expect("closing quote in .route call");
            paths.push(after_q1[..q2].to_string());
            rest = &after_q1[q2 + 1..];
        }
        paths.sort();
        paths.dedup();
        paths
    }

    #[test]
    fn probe_router_paths_match_const() {
        let mut declared: Vec<String> = PROBE_PATHS.iter().map(|s| s.to_string()).collect();
        declared.sort();
        let routed = extract_routed_paths(&router_source(), "build_probe_routes");
        assert_eq!(
            routed, declared,
            "PROBE_PATHS drifted from build_probe_routes — update one to match the other"
        );
    }

    #[test]
    fn public_router_paths_match_const() {
        // We can't easily run the const-list comparison while also handling
        // the conditional `/metrics` branch, so check both shapes by parsing
        // the function once and treating `METRICS_PATH_IF_PUBLIC` as the only
        // permitted extra path.
        let routed = extract_routed_paths(&router_source(), "build_public_routes");
        let mut declared: Vec<String> = PUBLIC_PATHS.iter().map(|s| s.to_string()).collect();
        declared.push(METRICS_PATH_IF_PUBLIC.to_string());
        declared.sort();
        assert_eq!(
            routed, declared,
            "PUBLIC_PATHS (+ METRICS_PATH_IF_PUBLIC) drifted from build_public_routes — \
             update one to match the other"
        );
    }

    #[test]
    fn public_router_has_no_per_user_handlers() {
        // The public probe handlers must not read `?user_id=`. We check the
        // four named handlers in PROBE_PATHS (mapped 1:1 to health.rs fns).
        const PUBLIC_PROBE_HANDLERS: &[&str] = &[
            "pub async fn health(",
            "pub async fn health_live(",
            "pub async fn health_ready(",
            "pub async fn health_index(",
        ];
        let src = health_source();
        for sig in PUBLIC_PROBE_HANDLERS {
            let start = src.find(sig).unwrap_or_else(|| {
                panic!("public probe handler `{}` not found in health.rs", sig)
            });
            // Walk braces to find the matching close of THIS function only.
            let after = &src[start..];
            let body_start = after.find('{').expect("function body opening brace");
            let mut depth = 0i32;
            let mut body_end = body_start;
            for (i, ch) in after[body_start..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            body_end = body_start + i;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let body = &after[body_start..=body_end];
            assert!(
                !body.contains("\"user_id\""),
                "public probe handler `{}` references \"user_id\" — per-tenant data \
                 must not be reachable on the public router. Move it behind auth on \
                 the /api/health/* protected route.",
                sig.trim_end_matches('(')
            );
        }
    }
}
