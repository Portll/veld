//! Smoke tests for all HTTP handler endpoints.
//!
//! Each handler group (health, users, recall, etc.) gets at least one test
//! that verifies:
//! - Valid requests return 2xx on fresh (empty) state.
//! - The auth middleware rejects unauthenticated access to protected routes.
//!
//! Run with: `cargo test --test handler_tests`

use std::sync::{Arc, Once};

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use serde_json::json;
use tempfile::TempDir;
use tower::ServiceExt;

use veld::{
    config::ServerConfig,
    handlers::{
        build_probe_routes, build_protected_routes, build_public_routes, MultiUserMemoryManager,
    },
};

// ═══════════════════════════════════════════════════════════════════════
// Test infrastructure
// ═══════════════════════════════════════════════════════════════════════

const TEST_KEY: &str = "handler-smoke-test-key";
static ENV_INIT: Once = Once::new();

fn init_env() {
    ENV_INIT.call_once(|| {
        // SAFETY: called once before any parallel tests start.
        unsafe {
            std::env::set_var("VELD_API_KEYS", TEST_KEY);
            // The detached cross-encoder pre-warm races this binary's exit (its own
            // cfg!(test) guard does not apply to an integration test's dependency).
            std::env::set_var("VELD_LAZY_CROSS_ENCODER", "1");
        }
    });
}

/// Self-contained test harness with a fresh temp directory and RocksDB.
struct Harness {
    mgr: Arc<MultiUserMemoryManager>,
    _dir: TempDir,
}

impl Harness {
    fn new() -> Self {
        init_env();
        let dir = TempDir::new().expect("create temp dir");
        let cfg = ServerConfig {
            storage_path: dir.path().to_path_buf(),
            backup_enabled: false,
            ..ServerConfig::default()
        };
        let mgr = MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg)
            .expect("create MultiUserMemoryManager");
        Self {
            mgr: Arc::new(mgr),
            _dir: dir,
        }
    }

    #[cfg(feature = "storage-redb")]
    fn new_redb() -> Self {
        Self::with_backend(veld::config::StorageBackend::Redb)
    }

    fn with_backend(backend: veld::config::StorageBackend) -> Self {
        init_env();
        let dir = TempDir::new().expect("create temp dir");
        let cfg = ServerConfig {
            storage_path: dir.path().to_path_buf(),
            backup_enabled: false,
            effective_storage_backend: backend,
            ..ServerConfig::default()
        };
        let mgr = MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg)
            .unwrap_or_else(|e| panic!("create {backend} MultiUserMemoryManager: {e:#}"));
        Self {
            mgr: Arc::new(mgr),
            _dir: dir,
        }
    }

    fn app(&self) -> Router {
        // Mirror main.rs: auth middleware only wraps protected routes.
        let probes = build_probe_routes(self.mgr.clone());
        let public = build_public_routes(self.mgr.clone(), false);
        let protected = build_protected_routes(self.mgr.clone(), false)
            .layer(axum::middleware::from_fn(veld::auth::auth_middleware));
        Router::new().merge(probes).merge(public).merge(protected)
    }
}

// ── request helpers ──

fn authed_get(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .header("x-api-key", TEST_KEY)
        .body(Body::empty())
        .unwrap()
}

fn authed_post(uri: &str, body: serde_json::Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(bytes))
        .unwrap()
}

#[allow(dead_code)]
fn authed_put(uri: &str, body: serde_json::Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::PUT)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(bytes))
        .unwrap()
}

#[allow(dead_code)]
fn authed_delete(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::DELETE)
        .uri(uri)
        .header("x-api-key", TEST_KEY)
        .body(Body::empty())
        .unwrap()
}

#[allow(dead_code)]
fn authed_delete_json(uri: &str, body: serde_json::Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::DELETE)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(bytes))
        .unwrap()
}

fn noauth_get(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn noauth_post(uri: &str, body: serde_json::Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(bytes))
        .unwrap()
}

// ── response helpers ──

async fn status_of(app: Router, req: Request<Body>) -> StatusCode {
    app.oneshot(req).await.unwrap().status()
}

async fn json_of(app: Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let val = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&bytes).to_string())
        })
    };
    (status, val)
}

// ═══════════════════════════════════════════════════════════════════════
// AUTH MIDDLEWARE
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn auth_public_routes_need_no_key() {
    let h = Harness::new();
    // /health is public
    assert_eq!(
        status_of(h.app(), noauth_get("/health")).await,
        StatusCode::OK
    );
    // /health/live is public
    assert_eq!(
        status_of(h.app(), noauth_get("/health/live")).await,
        StatusCode::OK
    );
}

#[tokio::test]
async fn auth_protected_routes_reject_missing_key() {
    let h = Harness::new();
    let status = status_of(
        h.app(),
        noauth_post("/api/recall", json!({"user_id":"u","query":"test"})),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_graph_data_requires_key() {
    // Regression: /api/graph/data/{user_id} was accidentally public.
    let h = Harness::new();
    let status = status_of(h.app(), noauth_get("/api/graph/data/test-user")).await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "/api/graph/data must be behind auth"
    );
}

#[tokio::test]
async fn auth_context_status_reads_require_key() {
    // Regression: GET /api/context/status listed every session's id, model, token
    // counts and working directory to any caller, cross-origin included.
    let h = Harness::new();
    for path in ["/api/context/status", "/api/context_status"] {
        let status = status_of(h.app(), noauth_get(path)).await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "{path} must be behind auth"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// health.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn health_endpoint() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body.get("status").is_some(),
        "health response needs 'status' field"
    );
    let census = &body["tenants_with_data"];
    assert_eq!(
        census["rocksdb"].as_u64(),
        Some(0),
        "fresh install has no tenant data: {body}"
    );
    assert!(
        census["measured_at"].is_string(),
        "census must say when it was measured: {body}"
    );
}

#[tokio::test]
async fn health_live() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/health/live")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "alive");
}

#[tokio::test]
async fn health_ready() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/health/ready")).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn health_index() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/health/index?user_id=test-user")).await;
    assert_eq!(status, StatusCode::OK);
    // The aggregate must carry the reconciliation counts, and must count an
    // unreadable coverage separately rather than folding it into "no shortfall".
    for field in [
        "users_checked",
        "users_with_index_shortfall_count",
        "users_coverage_unreadable_count",
    ] {
        assert!(
            body.get(field).and_then(|v| v.as_u64()).is_some(),
            "aggregate index health is missing {field}: {body}"
        );
    }
}

#[tokio::test]
async fn health_index_user_reports_coverage() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/api/health/index?user_id=test-user")).await;
    assert_eq!(status, StatusCode::OK);

    let cov = body
        .get("index_coverage")
        .unwrap_or_else(|| panic!("per-user index health is missing index_coverage: {body}"));
    let n = |k: &str| {
        cov.get(k)
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("index_coverage.{k} missing or not a number: {cov}"))
    };
    let (stored, vector_indexed, lexically_indexed) =
        (n("stored"), n("vector_indexed"), n("lexically_indexed"));

    // The flag must be derived from these three numbers, not from something
    // else that happens to be false here.
    assert_eq!(
        cov.get("shortfall").and_then(|v| v.as_bool()),
        Some(vector_indexed < stored || lexically_indexed < stored),
        "shortfall flag disagrees with the counts it summarises: {cov}"
    );
}

#[tokio::test]
async fn metrics_endpoint() {
    let h = Harness::new();
    let status = status_of(h.app(), authed_get("/metrics")).await;
    // metrics handler should always return 200 in test harness
    assert!(
        status == StatusCode::OK,
        "unexpected metrics status: {status}"
    );
}

#[tokio::test]
async fn context_status_roundtrip() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/context/status",
            json!({
                "session_id": "test-session",
                "tokens_used": 1000,
                "tokens_budget": 100000
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // GET context statuses
    let (status, body) = json_of(h.app(), authed_get("/api/context/status")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

// ═══════════════════════════════════════════════════════════════════════
// users.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_users_empty() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/api/users")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

#[tokio::test]
async fn user_stats_fresh() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/users/test-user/stats")).await;
    // Creates user on demand → should succeed
    assert!(status.is_success(), "user stats returned {status}");
}

#[tokio::test]
async fn stats_query() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/stats?user_id=test-user")).await;
    assert!(status.is_success(), "stats query returned {status}");
}

// ═══════════════════════════════════════════════════════════════════════
// sessions.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_sessions_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/sessions", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn session_stats() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/sessions/stats")).await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// remember.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn remember_basic() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "test-user",
                "content": "The sky is blue because of Rayleigh scattering."
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "remember failed: {body}");
    // Should return a memory_id
    assert!(
        body.get("id").is_some() || body.get("memory_id").is_some(),
        "remember response should contain an id: {body}"
    );
}

#[tokio::test]
async fn remember_with_tags_and_type() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "test-user",
                "content": "Decided to use Rust for the memory engine.",
                "memory_type": "Decision",
                "tags": ["architecture", "rust"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn recall_by_tags_returns_the_newest_matches_without_embeddings() {
    let h = Harness::new();
    for (content, created_at) in [
        ("January note on the shared tag.", "2026-01-01T00:00:00Z"),
        ("March note on the shared tag.", "2026-03-01T00:00:00Z"),
        ("February note on the shared tag.", "2026-02-01T00:00:00Z"),
    ] {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "tag-order-user",
                    "content": content,
                    "tags": ["shared-tag"],
                    "created_at": created_at
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "remember failed: {body}");
    }

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/recall/tags",
            json!({"user_id": "tag-order-user", "tags": ["shared-tag"], "limit": 2}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "recall by tags failed: {body}");
    let memories = body["memories"].as_array().expect("memories array");
    let contents: Vec<&str> = memories
        .iter()
        .map(|m| m["experience"]["content"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(
        contents,
        [
            "March note on the shared tag.",
            "February note on the shared tag."
        ]
    );
    for memory in memories {
        assert!(memory["experience"]["embeddings"].is_null(), "{memory}");
        assert!(
            memory["experience"]["embeddings_secondary"].is_null(),
            "{memory}"
        );
    }
}

#[tokio::test]
async fn recall_by_tags_leaves_out_annotations_and_context_unless_asked() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "lean-tag-user",
                "content": "Caroline told Melanie in Wellington that the Redis cluster moved to Auckland.",
                "tags": ["lean-tag"],
                "repo": "Portll/veld"
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "remember failed: {body}");

    let recall = |extra: serde_json::Value| {
        let mut request = json!({"user_id": "lean-tag-user", "tags": ["lean-tag"]});
        request
            .as_object_mut()
            .expect("request object")
            .extend(extra.as_object().expect("extra object").clone());
        authed_post("/api/recall/tags", request)
    };

    let (status, lean) = json_of(h.app(), recall(json!({}))).await;
    assert_eq!(status, StatusCode::OK, "recall by tags failed: {lean}");
    let experience = &lean["memories"][0]["experience"];
    assert_eq!(
        experience["content"],
        "Caroline told Melanie in Wellington that the Redis cluster moved to Auckland."
    );
    assert!(experience["context"].is_null(), "{experience}");
    assert_eq!(experience["ner_entities"], json!([]), "{experience}");
    assert_eq!(experience["cooccurrence_pairs"], json!([]), "{experience}");

    let (status, full) = json_of(
        h.app(),
        recall(json!({"include_annotations": true, "include_context": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "recall by tags failed: {full}");
    let experience = &full["memories"][0]["experience"];
    assert!(
        experience["context"].is_object(),
        "the repo was stored in the context: {experience}"
    );
    assert!(experience["ner_entities"].is_array(), "{experience}");
}

#[tokio::test]
async fn recall_by_tags_matches_derived_tags_only_when_asked() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "derived-tag-user",
                "content": "The quokka migration moved the quokka ledger onto the quokka cluster in Wellington.",
                "tags": ["caller-chosen"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "remember failed: {body}");

    let recall = |tags: Vec<&str>, include_derived: bool| {
        authed_post(
            "/api/recall/tags",
            json!({
                "user_id": "derived-tag-user",
                "tags": tags,
                "include_derived": include_derived
            }),
        )
    };
    let (_, by_caller_tag) = json_of(h.app(), recall(vec!["caller-chosen"], false)).await;
    assert_eq!(by_caller_tag["count"], 1, "{by_caller_tag}");
    let stored_tags: Vec<String> = by_caller_tag["memories"][0]["experience"]["tags"]
        .as_array()
        .expect("stored tags")
        .iter()
        .filter_map(|t| t.as_str().map(str::to_string))
        .collect();
    let derived = stored_tags
        .iter()
        .find(|t| !t.eq_ignore_ascii_case("caller-chosen"))
        .unwrap_or_else(|| panic!("no derived tag was stored: {stored_tags:?}"))
        .clone();

    let (_, default_match) = json_of(h.app(), recall(vec![derived.as_str()], false)).await;
    assert_eq!(
        default_match["count"], 0,
        "derived `{derived}`: {default_match}"
    );
    let (_, derived_match) = json_of(h.app(), recall(vec![derived.as_str()], true)).await;
    assert_eq!(
        derived_match["count"], 1,
        "derived `{derived}`: {derived_match}"
    );
}

#[tokio::test]
async fn unknown_memory_type_is_rejected_not_stored_as_observation() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "test-user",
                "content": "A memory whose type is a typo.",
                "memory_type": "Decsion"
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.to_string().contains("Decsion"), "{body}");

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember/batch",
            json!({
                "user_id": "test-user",
                "memories": [
                    {"content": "First batch item with a valid type.", "memory_type": "Learning"},
                    {"content": "Second batch item with a typo type.", "memory_type": "Lerning"}
                ]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["created"], 1, "{body}");
    assert_eq!(body["failed"], 1, "{body}");
    assert_eq!(body["errors"][0]["index"], 1, "{body}");
}

#[tokio::test]
async fn todo_filters_and_fields_reject_unknown_values() {
    let h = Harness::new();
    for (uri, body) in [
        (
            "/api/todos/add",
            json!({"user_id": "test-user", "content": "Bad status", "status": "finished"}),
        ),
        (
            "/api/todos/add",
            json!({"user_id": "test-user", "content": "Bad priority", "priority": "critical"}),
        ),
        (
            "/api/todos/add",
            json!({"user_id": "test-user", "content": "Bad recurrence", "recurrence": "yearly"}),
        ),
        (
            "/api/todos",
            json!({"user_id": "test-user", "status": ["todo", "finished"]}),
        ),
        (
            "/api/todos",
            json!({"user_id": "test-user", "due": "tomorrow"}),
        ),
        (
            "/api/todos",
            json!({"user_id": "test-user", "priority": "critical"}),
        ),
        (
            "/api/reminders",
            json!({"user_id": "test-user", "status": "snoozed"}),
        ),
    ] {
        let (status, resp) = json_of(h.app(), authed_post(uri, body.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri} {body}: {resp}");
    }

    let (status, body) = json_of(
        h.app(),
        authed_post("/api/todos", json!({"user_id": "test-user", "due": "all"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "due=all lists everything: {body}");
}

#[tokio::test]
async fn multimodal_search_refuses_modes_it_cannot_parameterise() {
    let h = Harness::new();
    for mode in ["spatial", "mission"] {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/search/multimodal",
                json!({"user_id": "test-user", "query_text": "dock", "mode": mode}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{mode}: {body}");
        assert!(body.to_string().contains("/api/search/robotics"), "{body}");
    }
}

/// Names of the entities a memory's graph episode references.
fn episode_entity_names(h: &Harness, user: &str, id: &veld::uuid::Uuid) -> Vec<String> {
    let graph = h.mgr.get_user_graph(user).expect("graph");
    let graph = graph.read();
    match graph.get_episode(id).expect("read episode") {
        Some(episode) => episode
            .entity_refs
            .iter()
            .filter_map(|e| graph.get_entity(e).ok().flatten())
            .map(|e| e.name.to_lowercase())
            .collect(),
        None => Vec::new(),
    }
}

/// Waits for the background graph pass to give a memory's episode an entity.
async fn wait_for_episode_entity(h: &Harness, user: &str, id: veld::uuid::Uuid, name: &str) {
    for _ in 0..300 {
        if episode_entity_names(h, user, &id).iter().any(|n| n == name) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    panic!(
        "the graph episode never gained '{name}': {:?}",
        episode_entity_names(h, user, &id)
    );
}

/// Waits until the work `remember` hands to the background has finished. The episode lists
/// its entities before their edges are linked, and an edge is inserted before its endpoints
/// are pruned back to the degree cap, so a read in between sees one edge over the cap.
async fn wait_for_background_work(h: &Harness) {
    for _ in 0..300 {
        if h.mgr.task_tracker.is_empty() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    panic!(
        "{} background tasks still running after 30s",
        h.mgr.task_tracker.len()
    );
}

#[tokio::test]
async fn a_duplicate_with_new_tags_reaches_the_stored_memory_and_its_graph_episode() {
    use veld::memory::MemoryId;

    let h = Harness::new();
    let user = "merge-user";
    // Short enough that the episode stays under max_entities_per_memory (10) after the merge.
    let content = "The tender drifted overnight.";

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": user, "content": content, "tags": ["mooring"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id = veld::uuid::Uuid::parse_str(body["id"].as_str().expect("id")).unwrap();
    wait_for_episode_entity(&h, user, id, "mooring").await;

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": user, "content": content, "tags": ["kestrel"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"].as_str(), Some(id.to_string().as_str()));

    let stored = {
        let memory = h.mgr.get_user_memory(user).expect("memory");
        let guard = memory.read();
        guard.get_memory(&MemoryId(id)).expect("stored")
    };
    for name in ["mooring", "kestrel"] {
        assert!(
            stored.experience.entities.iter().any(|e| e == name),
            "{name} missing from {:?}",
            stored.experience.entities
        );
    }
    wait_for_episode_entity(&h, user, id, "kestrel").await;
}

#[tokio::test]
async fn batch_remember() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember/batch",
            json!({
                "user_id": "test-user",
                "memories": [
                    {"content": "First memory item"},
                    {"content": "Second memory item", "tags": ["batch"]}
                ]
            }),
        ),
    )
    .await;
    assert!(status.is_success(), "batch remember returned {status}");
}

#[tokio::test]
async fn upsert_memory() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            json!({
                "user_id": "test-user",
                "external_id": "ext-001",
                "content": "Upserted memory content."
            }),
        ),
    )
    .await;
    assert!(status.is_success(), "upsert returned {status}");
}

/// Every memory carrying `external_id`, read from the store rather than from the response.
fn upserted(h: &Harness, user: &str, external_id: &str) -> Vec<veld::memory::Memory> {
    let memory = h.mgr.get_user_memory(user).expect("memory");
    let guard = memory.read();
    guard
        .get_all_memories()
        .expect("all memories")
        .into_iter()
        .filter(|m| m.external_id.as_deref() == Some(external_id))
        .map(|m| (*m).clone())
        .collect()
}

fn the_one(h: &Harness, user: &str, external_id: &str) -> veld::memory::Memory {
    let mut found = upserted(h, user, external_id);
    assert_eq!(found.len(), 1, "expected one memory for {external_id}");
    found.remove(0)
}

const TRANSCRIPT_CREATED: &str = "2026-09-01T10:00:00Z";

fn transcript_upsert(user: &str, content: &str) -> serde_json::Value {
    json!({
        "user_id": user,
        "external_id": "cc:transcript:sess-1",
        "content": content,
        "episode_id": "ep-sess-1",
        "agent_id": "agent-main",
        "created_at": TRANSCRIPT_CREATED,
        "writer": {"session_id": "sess-1", "transcript_id": "tx-1", "model_id": "model-a"}
    })
}

#[tokio::test]
async fn upsert_stores_provenance_on_insert_as_remember_does() {
    let h = Harness::new();
    let user = "upsert-provenance";
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            transcript_upsert(user, "Transcript sess-1, first pass."),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["was_update"], json!(false), "{body}");

    let stored = the_one(&h, user, "cc:transcript:sess-1");
    let created: veld::chrono::DateTime<veld::chrono::Utc> = TRANSCRIPT_CREATED.parse().unwrap();
    assert_eq!(stored.agent_id.as_deref(), Some("agent-main"));
    assert_eq!(stored.created_at, created);
    let ctx = stored.experience.context.as_ref().expect("context");
    assert_eq!(ctx.episode.episode_id.as_deref(), Some("ep-sess-1"));
    assert_eq!(ctx.session.declared.transcript_id.as_deref(), Some("tx-1"));
    assert_eq!(ctx.session.declared.session_id.as_deref(), Some("sess-1"));
    assert_eq!(ctx.session.declared.model_id.as_deref(), Some("model-a"));

    // Parity: the same provenance sent to /api/remember is stored identically.
    let mut via_remember = transcript_upsert(user, "Same provenance, written by remember.");
    via_remember.as_object_mut().unwrap().remove("external_id");
    let (status, body) = json_of(h.app(), authed_post("/api/remember", via_remember)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let remembered = {
        let id = veld::uuid::Uuid::parse_str(body["id"].as_str().expect("id")).unwrap();
        let memory = h.mgr.get_user_memory(user).expect("memory");
        let guard = memory.read();
        guard
            .get_memory(&veld::memory::MemoryId(id))
            .expect("stored")
    };
    let rctx = remembered.experience.context.as_ref().expect("context");
    assert_eq!(remembered.agent_id, stored.agent_id);
    assert_eq!(remembered.created_at, stored.created_at);
    assert_eq!(rctx.episode.episode_id, ctx.episode.episode_id);
    assert_eq!(rctx.session.declared, ctx.session.declared);
    assert_eq!(rctx.source.source_type, ctx.source.source_type);
    assert_eq!(rctx.source.credibility, ctx.source.credibility);
}

#[tokio::test]
async fn a_second_upsert_updates_the_one_memory() {
    let h = Harness::new();
    let user = "upsert-twice";
    let (status, first) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            transcript_upsert(user, "Transcript sess-1, first pass."),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (status, second) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            transcript_upsert(user, "Transcript sess-1, grown."),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["was_update"], json!(true), "{second}");
    assert_eq!(second["id"], first["id"]);
    assert_eq!(second["version"], json!(2), "{second}");

    let stored = the_one(&h, user, "cc:transcript:sess-1");
    assert_eq!(stored.experience.content, "Transcript sess-1, grown.");
    assert_eq!(stored.history.len(), 1, "the first pass is kept in history");
}

#[tokio::test]
async fn an_update_keeps_omitted_provenance_and_overwrites_provided() {
    let h = Harness::new();
    let user = "upsert-omit";
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            transcript_upsert(user, "Transcript sess-1, first pass."),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // No episode_id, agent_id or created_at; a different writer.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            json!({
                "user_id": user,
                "external_id": "cc:transcript:sess-1",
                "content": "Transcript sess-1, grown.",
                "writer": {"session_id": "sess-1", "transcript_id": "tx-2"}
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["was_update"], json!(true), "{body}");

    let stored = the_one(&h, user, "cc:transcript:sess-1");
    let created: veld::chrono::DateTime<veld::chrono::Utc> = TRANSCRIPT_CREATED.parse().unwrap();
    let ctx = stored.experience.context.as_ref().expect("context");
    assert_eq!(ctx.episode.episode_id.as_deref(), Some("ep-sess-1"));
    assert_eq!(stored.agent_id.as_deref(), Some("agent-main"));
    assert_eq!(stored.created_at, created);
    assert_eq!(ctx.session.declared.transcript_id.as_deref(), Some("tx-2"));
    assert_eq!(
        ctx.session.declared.model_id, None,
        "a provided writer replaces the whole writer"
    );
}

#[tokio::test]
async fn an_upsert_without_provenance_behaves_as_before() {
    let h = Harness::new();
    let user = "upsert-plain";
    let before = veld::chrono::Utc::now();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            json!({"user_id": user, "external_id": "ext-plain", "content": "Plain upsert."}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut keys: Vec<&str> = body
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["id", "success", "version", "was_update"], "{body}");
    assert_eq!(body["was_update"], json!(false));
    assert_eq!(body["version"], json!(1));

    let stored = the_one(&h, user, "ext-plain");
    assert_eq!(stored.agent_id, None);
    assert!(stored.created_at >= before, "created_at defaults to now");
    let ctx = stored.experience.context.as_ref().expect("context");
    assert_eq!(ctx.episode.episode_id, None);
    assert_eq!(
        ctx.session.declared,
        veld::memory::facets::DeclaredWriter::default()
    );

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            json!({"user_id": user, "external_id": "ext-plain", "content": "Plain upsert, edited."}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["was_update"], json!(true));
    assert_eq!(body["version"], json!(2));
    let stored = the_one(&h, user, "ext-plain");
    assert_eq!(stored.experience.content, "Plain upsert, edited.");
    assert_eq!(stored.agent_id, None);
}

// ── repo scoping ──

fn repo_of(memory: &veld::memory::Memory) -> Option<String> {
    memory
        .experience
        .context
        .as_ref()
        .and_then(|c| c.repo_slug())
        .map(str::to_string)
}

fn ids_in(body: &serde_json::Value) -> Vec<String> {
    body["memories"]
        .as_array()
        .expect("memories")
        .iter()
        .map(|m| m["id"].as_str().expect("id").to_string())
        .collect()
}

/// Three memories sharing a phrase: one in acme/one, one in acme/two, one in no repo.
/// Returns their ids in that order, once their embeddings have landed.
async fn remember_across_repos(h: &Harness, user: &str) -> [String; 3] {
    let writes = [
        (
            Some("acme/one"),
            "Quokka lantern protocol: the billing worker retries three times before paging.",
        ),
        (
            Some("acme/two"),
            "Quokka lantern protocol: dashboards refresh every ninety seconds once the cache warms.",
        ),
        (
            None,
            "Quokka lantern protocol: the release checklist requires a signed changelog entry.",
        ),
    ];
    let mut ids = Vec::new();
    for (repo, content) in writes {
        let mut req = json!({"user_id": user, "content": content});
        if let Some(repo) = repo {
            req["repo"] = json!(repo);
        }
        let (status, body) = json_of(h.app(), authed_post("/api/remember", req)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        ids.push(body["id"].as_str().expect("id").to_string());
    }
    wait_for_background_work(h).await;
    ids.try_into().unwrap()
}

#[tokio::test]
async fn recall_with_repo_returns_only_that_repos_memories() {
    let h = Harness::new();
    let user = "repo-recall";
    let [one, two, none] = remember_across_repos(&h, user).await;

    let stored = {
        let memory = h.mgr.get_user_memory(user).expect("memory");
        let guard = memory.read();
        guard
            .get_memory(&veld::memory::MemoryId(
                veld::uuid::Uuid::parse_str(&one).unwrap(),
            ))
            .expect("stored")
    };
    assert_eq!(repo_of(&stored).as_deref(), Some("acme/one"));

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({"user_id": user, "query": "quokka lantern protocol", "limit": 10,
                   "repo": "Acme/One", "competition_mode": "coexist"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(ids_in(&body), vec![one.clone()], "{body:#}");

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({"user_id": user, "query": "quokka lantern protocol", "limit": 10,
                   "competition_mode": "coexist"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let unscoped = ids_in(&body);
    for id in [&one, &two, &none] {
        assert!(unscoped.contains(id), "{id} missing unscoped: {body:#}");
    }
}

#[tokio::test]
async fn proactive_context_with_repo_surfaces_no_other_repo() {
    let h = Harness::new();
    let user = "repo-proactive";
    let [one, two, none] = remember_across_repos(&h, user).await;

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/proactive_context",
            json!({"user_id": user, "repo": "acme/one", "max_results": 10,
                   "context": "Remind me how the quokka lantern protocol handles billing and dashboards."}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let surfaced = ids_in(&body);
    assert!(surfaced.contains(&one), "{body:#}");
    assert!(!surfaced.contains(&two), "acme/two leaked: {body:#}");
    assert!(
        !surfaced.contains(&none),
        "unscoped memory leaked: {body:#}"
    );
}

#[tokio::test]
async fn upsert_stores_repo_and_an_update_without_one_keeps_it() {
    let h = Harness::new();
    let user = "repo-upsert";
    let upsert = |content: &str, repo: Option<&str>| {
        let mut req = json!({"user_id": user, "external_id": "ext-repo", "content": content});
        if let Some(repo) = repo {
            req["repo"] = json!(repo);
        }
        authed_post("/api/upsert", req)
    };

    let (status, body) = json_of(h.app(), upsert("First pass.", Some(" acme/one "))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        repo_of(&the_one(&h, user, "ext-repo")).as_deref(),
        Some("acme/one")
    );

    let (status, body) = json_of(h.app(), upsert("Second pass.", None)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["was_update"], json!(true), "{body}");
    assert_eq!(
        repo_of(&the_one(&h, user, "ext-repo")).as_deref(),
        Some("acme/one")
    );

    let (status, body) = json_of(h.app(), upsert("Third pass.", Some("acme/two"))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let stored = the_one(&h, user, "ext-repo");
    assert_eq!(repo_of(&stored).as_deref(), Some("acme/two"));
    let repos = stored
        .experience
        .context
        .as_ref()
        .unwrap()
        .place
        .places
        .iter()
        .filter(|p| matches!(p, veld::memory::facets::Place::Repo { .. }))
        .count();
    assert_eq!(repos, 1, "a provided repo replaces the stored one");
}

#[tokio::test]
async fn an_invalid_repo_is_rejected() {
    let h = Harness::new();
    let user = "repo-invalid";
    let too_long = "r".repeat(201);
    for bad in ["", "   ", too_long.as_str(), "acme/\u{7}one"] {
        for (uri, req) in [
            (
                "/api/remember",
                json!({"user_id": user, "content": "Scoped badly.", "repo": bad}),
            ),
            (
                "/api/upsert",
                json!({"user_id": user, "external_id": "ext-bad", "content": "Scoped badly.",
                       "repo": bad}),
            ),
            (
                "/api/recall",
                json!({"user_id": user, "query": "scoped", "repo": bad}),
            ),
            (
                "/api/proactive_context",
                json!({"user_id": user, "context": "What was scoped badly here?", "repo": bad}),
            ),
        ] {
            let (status, body) = json_of(h.app(), authed_post(uri, req)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{uri} {bad:?}: {body}");
        }
    }

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember/batch",
            json!({"user_id": user, "memories": [
                {"content": "Batch item scoped badly.", "repo": ""},
                {"content": "Batch item scoped well.", "repo": "acme/one"}
            ]}),
        ),
    )
    .await;
    assert!(status.is_success(), "{body}");
    assert_eq!(body["created"], json!(1), "{body}");
    assert_eq!(body["failed"], json!(1), "{body}");
    assert_eq!(body["errors"][0]["index"], json!(0), "{body}");
}

// ═══════════════════════════════════════════════════════════════════════
// recall.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn recall_empty_state() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({
                "user_id": "test-user",
                "query": "what do I know?"
            }),
        ),
    )
    .await;
    assert!(status.is_success(), "recall returned {status}: {body}");
}

#[tokio::test]
async fn context_summary_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/context_summary", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn proactive_context() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/proactive_context",
            json!({
                "user_id": "test-user",
                "context": "Working on handler tests."
            }),
        ),
    )
    .await;
    assert!(status.is_success());
    assert!(
        body.get("ingested_memory_id").is_none(),
        "a surfacing call stores nothing unless auto_ingest is set: {body}"
    );
}

#[tokio::test]
async fn recall_by_tags_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/recall/tags",
            json!({
                "user_id": "test-user",
                "tags": ["nonexistent"]
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn reinforce_feedback() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/reinforce",
            json!({
                "user_id": "test-user",
                "ids": [],
                "outcome": "helpful"
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// crud.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_memories_get_empty() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/list/test-user")).await;
    assert!(status.is_success());
}

#[tokio::test]
async fn list_memories_post_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/memories", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn get_memory_not_found() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/memory/nonexistent-id")).await;
    // handler may return 404 or 422 for missing memory
    assert!(
        status == StatusCode::NOT_FOUND
            || status == StatusCode::UNPROCESSABLE_ENTITY
            || status == StatusCode::BAD_REQUEST,
        "expected error status for missing memory, got {status}"
    );
}

#[tokio::test]
async fn forget_by_age() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/forget/age",
            json!({"user_id": "test-user", "days_old": 30}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn forget_by_importance() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/forget/importance",
            json!({"user_id": "test-user", "threshold": 0.1}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn forget_by_pattern() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/forget/pattern",
            json!({"user_id": "test-user", "pattern": "nonexistent"}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn forget_by_tags() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/forget/tags",
            json!({"user_id": "test-user", "tags": ["cleanup"]}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn clear_all_memories_requires_confirm() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/memories/clear",
            json!({"user_id": "test-user", "confirm": "CONFIRM"}),
        ),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// search.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn advanced_search_empty() {
    let h = Harness::new();
    // Handler requires at least one criterion; sending none → 4xx.
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/search/advanced", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(
        status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY,
        "expected 4xx for missing criteria, got {status}"
    );

    // With a criterion, should succeed even on empty state.
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/search/advanced",
            json!({"user_id": "test-user", "entity_name": "Rust"}),
        ),
    )
    .await;
    assert!(
        status.is_success(),
        "advanced_search with entity should succeed: {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// facts.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_facts_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/facts/list", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn search_facts_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/facts/search",
            json!({"user_id": "test-user", "query": "anything"}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn facts_by_entity_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/facts/by-entity",
            json!({"user_id": "test-user", "entity": "Rust"}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn facts_stats() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/facts/stats", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn facts_narratives_preview_purge_and_purge_are_routed() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post("/api/facts/narratives", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success(), "narratives: {status} {body}");
    assert_eq!(body["total_clusters"], 0);

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/facts/preview-purge",
            json!({"user_id": "test-user", "pattern": "relates to"}),
        ),
    )
    .await;
    assert!(status.is_success(), "preview-purge: {status} {body}");
    assert_eq!(body["match_bucket"], "none");

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/facts/purge",
            json!({"user_id": "test-user", "pattern": "relates to"}),
        ),
    )
    .await;
    assert!(status.is_success(), "purge: {status} {body}");
    assert_eq!(body["dry_run"], true, "purge defaults to a dry run");
    assert_eq!(body["purged"], 0);
}

// ═══════════════════════════════════════════════════════════════════════
// compression.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn compress_nonexistent_memory() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/memory/compress",
            json!({"user_id": "test-user", "id": "nonexistent"}),
        ),
    )
    .await;
    // Should fail gracefully for nonexistent memory
    assert!(
        status == StatusCode::NOT_FOUND
            || status == StatusCode::BAD_REQUEST
            || status == StatusCode::UNPROCESSABLE_ENTITY,
        "compress returned unexpected: {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// lineage.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn lineage_list_edges_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/lineage/edges", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn lineage_stats_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/lineage/stats", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// graph.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn graph_stats_fresh() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/graph/test-user/stats")).await;
    assert!(status.is_success());
}

#[tokio::test]
async fn get_all_entities_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/graph/entities/all", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn find_entity_nonexistent() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/graph/entity/find",
            json!({"user_id": "test-user", "entity_name": "NoSuchEntity"}),
        ),
    )
    .await;
    // Should return 200 with null/empty or 404
    assert!(
        status.is_success() || status == StatusCode::NOT_FOUND,
        "find_entity returned {status}"
    );
}

#[tokio::test]
async fn traverse_graph_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/graph/traverse",
            json!({
                "user_id": "test-user",
                "entity_name": "NoEntity"
            }),
        ),
    )
    .await;
    assert!(
        status.is_success() || status == StatusCode::NOT_FOUND,
        "traverse returned {status}"
    );
}

#[tokio::test]
async fn graph_data_authenticated() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/graph/data/test-user")).await;
    assert!(
        status.is_success(),
        "graph data with auth returned {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// todos.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn todos_resolve_dependencies_link_memories_and_report_what_completion_unblocks() {
    use veld::memory::Experience;

    let h = Harness::new();
    let user = "potent-todos";
    let create = |body: serde_json::Value| authed_post("/api/todos/add", body);

    let (status, body) = json_of(
        h.app(),
        create(json!({"user_id": user, "content": "Ship it", "blocked_by": ["SHO-999"]})),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "an unknown dependency: {body}"
    );

    let memory_id = {
        let memory = h.mgr.get_user_memory(user).expect("memory");
        let guard = memory.read();
        guard
            .remember(
                Experience {
                    content: "The release needs a signed changelog.".to_string(),
                    ..Default::default()
                },
                None,
            )
            .expect("remember")
    };
    let (status, body) = json_of(
        h.app(),
        create(json!({
            "user_id": user,
            "content": "Write the changelog",
            "related_memory_ids": [memory_id.0.to_string()],
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let blocker = body["todo"]["id"].as_str().expect("id").to_string();
    let linked = &body["todo"]["related_memory_ids"];
    assert!(
        linked
            .as_array()
            .is_some_and(|ids| ids.iter().any(|id| id == &json!(memory_id.0.to_string()))),
        "{body}"
    );

    let (status, body) = json_of(
        h.app(),
        create(json!({
            "user_id": user,
            "content": "Tag the release",
            "blocked_by": [blocker.clone()],
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let dependent = body["todo"]["id"].as_str().expect("id").to_string();

    let (status, body) = json_of(
        h.app(),
        authed_post("/api/todos", json!({"user_id": user, "query": "changelog"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["todos"][0]["id"].as_str(),
        Some(blocker.as_str()),
        "{body}"
    );

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{blocker}/complete"),
            json!({"user_id": user}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["unblocked"][0]["id"].as_str(),
        Some(dependent.as_str()),
        "{body}"
    );

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{dependent}/reorder"),
            json!({"user_id": user, "direction": "sideways"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{dependent}/comments"),
            json!({"user_id": user, "content": "Waiting on the changelog."}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let comment_id = body["comment"]["id"].as_str().expect("comment id");
    assert!(
        body["formatted"]
            .as_str()
            .is_some_and(|f| f.contains(comment_id)),
        "{body}"
    );
}

async fn create_todo_id(h: &Harness, user: &str, content: &str) -> String {
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": user, "content": content}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body["todo"]["id"].as_str().expect("id").to_string()
}

#[tokio::test]
async fn flat_todo_aliases_take_the_todo_id_from_the_body() {
    let h = Harness::new();
    let user = "flat-alias";

    let id = create_todo_id(&h, user, "alias update target").await;
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/update",
            json!({"user_id": user, "todo_id": id, "notes": "set via flat alias"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["todo"]["notes"], "set via flat alias");

    let id = create_todo_id(&h, user, "alias complete target").await;
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/complete",
            json!({"user_id": user, "todo_id": id}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["todo"]["status"], "done");

    let id = create_todo_id(&h, user, "alias reorder target").await;
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/reorder",
            json!({"user_id": user, "todo_id": id, "direction": "up"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let id = create_todo_id(&h, user, "alias delete target").await;
    let (status, body) = json_of(
        h.app(),
        authed_post("/api/todos/delete", json!({"user_id": user, "todo_id": id})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let gone = h
        .mgr
        .todo_store
        .get_todo(user, &veld::memory::TodoId(id.parse().unwrap()))
        .unwrap();
    assert!(gone.is_none(), "the flat delete alias deletes");

    let (status, body) = json_of(
        h.app(),
        authed_post("/api/todos/update", json!({"user_id": user, "notes": "x"})),
    )
    .await;
    assert!(
        status.is_client_error(),
        "a missing todo_id is a 4xx: {status} {body}"
    );
}

#[tokio::test]
async fn todo_responses_omit_the_embedding_that_storage_keeps() {
    let h = Harness::new();
    let user = "embedding-wire";
    let id = create_todo_id(&h, user, "semantic search target").await;
    let stored = h
        .mgr
        .todo_store
        .get_todo(user, &veld::memory::TodoId(id.parse().unwrap()))
        .unwrap()
        .expect("stored");
    assert!(stored.embedding.is_some(), "storage keeps the embedding");

    let (_, body) = json_of(h.app(), authed_post("/api/todos", json!({"user_id": user}))).await;
    assert!(body["todos"][0].get("embedding").is_none(), "{body}");
    let (_, body) = json_of(
        h.app(),
        authed_get(&format!("/api/todos/{id}?user_id={user}")),
    )
    .await;
    assert!(body["todo"].get("embedding").is_none(), "{body}");
    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/complete",
            json!({"user_id": user, "todo_id": id}),
        ),
    )
    .await;
    assert!(body["todo"].get("embedding").is_none(), "{body}");
}

#[tokio::test]
async fn a_todo_memory_link_is_written_on_both_sides_and_revoked_on_delete() {
    use veld::memory::{Experience, MemoryId, TodoId};

    let h = Harness::new();
    let user = "backlink";
    let (mem_a, mem_b) = {
        let memory = h.mgr.get_user_memory(user).unwrap();
        let guard = memory.read();
        let remember = |content: &str| {
            guard
                .remember(
                    Experience {
                        content: content.to_string(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };
        (remember("source memory A"), remember("source memory B"))
    };
    let back_links = |mid: &MemoryId| -> Vec<TodoId> {
        let memory = h.mgr.get_user_memory(user).unwrap();
        let guard = memory.read();
        guard.get_memory(mid).unwrap().related_todo_ids
    };

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": user, "content": "motivated by A",
                   "related_memory_ids": [mem_a.0.to_string()]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id_str = body["todo"]["id"].as_str().unwrap().to_string();
    let todo_id = TodoId(id_str.parse().unwrap());
    assert_eq!(back_links(&mem_a), vec![todo_id]);

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{id_str}/update"),
            json!({"user_id": user, "related_memory_ids": [mem_b.0.to_string()]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        back_links(&mem_a).is_empty(),
        "a dropped memory loses its back-link"
    );
    assert_eq!(back_links(&mem_b), vec![todo_id]);

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/delete",
            json!({"user_id": user, "todo_id": id_str}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        back_links(&mem_b).is_empty(),
        "deletion leaves no dangling id"
    );
}

#[tokio::test]
async fn todo_lifecycle() {
    let h = Harness::new();

    // Create a todo
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({
                "user_id": "test-user",
                "content": "Write handler tests",
                "priority": "high",
                "tags": ["testing"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create todo: {body}");
    let todo_id = body["todo"]["id"]
        .as_str()
        .or(body["id"].as_str())
        .or(body["todo_id"].as_str());
    assert!(todo_id.is_some(), "todo response should contain id: {body}");
    let todo_id = todo_id.unwrap().to_string();

    // List todos
    let (status, body) = json_of(
        h.app(),
        authed_post("/api/todos", json!({"user_id": "test-user"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // Should have at least 1 todo
    let todos = body["todos"].as_array().or(body.as_array());
    assert!(
        todos.map(|t| !t.is_empty()).unwrap_or(false),
        "should have todos after create: {body}"
    );

    // Complete the todo (path-style route)
    let (status, _) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/complete"),
            json!({"user_id": "test-user"}),
        ),
    )
    .await;
    assert!(status.is_success(), "complete todo returned {status}");

    // Delete the todo (DELETE with query param)
    let (status, _) = json_of(
        h.app(),
        Request::builder()
            .method(Method::DELETE)
            .uri(format!("/api/todos/{todo_id}?user_id=test-user"))
            .header("x-api-key", TEST_KEY)
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert!(status.is_success(), "delete todo returned {status}");
}

#[tokio::test]
async fn list_todos_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/todos", json!({"user_id": "fresh-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn todo_stats_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/todos/stats", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn project_lifecycle() {
    let h = Harness::new();

    // Create project
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/projects",
            json!({
                "user_id": "test-user",
                "name": "Test Project",
                "prefix": "TST"
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create project: {body}");

    // List projects
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/projects/list", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn codebase_scan_and_index_refuse_system_and_data_directories() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/projects",
            json!({"user_id": "test-user", "name": "Confined"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create project: {body}");

    let data_dir = h._dir.path().to_string_lossy().to_string();
    for route in ["scan", "index"] {
        for path in ["/etc", "/", data_dir.as_str()] {
            let (status, body) = json_of(
                h.app(),
                authed_post(
                    &format!("/api/projects/Confined/{route}"),
                    json!({"user_id": "test-user", "codebase_path": path}),
                ),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{route} {path}: {body}");
        }
    }

    let code = TempDir::new().unwrap();
    std::fs::write(code.path().join("lib.rs"), "pub fn f() {}\n").unwrap();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/projects/Confined/scan",
            json!({"user_id": "test-user", "codebase_path": code.path()}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "scan a project directory: {body}");
}

#[tokio::test]
async fn list_reminders_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/reminders", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn create_duration_reminder() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/reminders/set",
            json!({
                "user_id": "test-user",
                "content": "Check test results",
                "trigger": {"type": "duration", "after_seconds": 3600}
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// ab_testing.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_ab_tests_empty() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/api/ab/tests")).await;
    assert_eq!(status, StatusCode::OK);
    // Response is {"success":true, "tests":[...], "summary":{...}}
    assert!(
        body["tests"].is_array() || body.is_array(),
        "expected tests array: {body}"
    );
}

#[tokio::test]
async fn ab_test_lifecycle() {
    let h = Harness::new();

    // Create an A/B test
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/ab/tests",
            json!({
                "name": "recall_weights_v2",
                "description": "Test new recall weights",
                "traffic_split": 0.5
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create ab test: {body}");

    // List again
    let (status, body) = json_of(h.app(), authed_get("/api/ab/tests")).await;
    assert_eq!(status, StatusCode::OK);
    let tests = body["tests"].as_array().or(body.as_array());
    assert!(
        tests.map(|t| !t.is_empty()).unwrap_or(false),
        "should have one A/B test: {body}"
    );

    // Summary
    let (status, _) = json_of(h.app(), authed_get("/api/ab/summary")).await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// consolidation.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn verify_index_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/index/verify", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn rebuild_index_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/index/rebuild", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn create_backup() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/backup/create", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn list_backups_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/backup/list", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[cfg(feature = "storage-redb")]
fn graph_entity(name: &str) -> veld::graph_memory::EntityNode {
    veld::graph_memory::EntityNode {
        uuid: veld::uuid::Uuid::new_v4(),
        name: name.to_string(),
        labels: vec![veld::graph_memory::EntityLabel::Concept],
        created_at: chrono::Utc::now(),
        last_seen_at: chrono::Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: std::collections::HashMap::new(),
        name_embedding: None,
        salience: 0.5,
        is_proper_noun: false,
        pii_classification: veld::graph_memory::PiiClassification::Clean,
    }
}

#[cfg(feature = "storage-redb")]
#[tokio::test]
async fn redb_backup_routes_create_list_verify_purge_and_restore() {
    let h = Harness::new_redb();
    let user = "test-user";
    let kept = h
        .mgr
        .get_user_graph(user)
        .unwrap()
        .read()
        .add_entity(graph_entity("in the backup"))
        .unwrap();

    let (status, body) = json_of(
        h.app(),
        authed_post("/api/backup/create", json!({"user_id": user})),
    )
    .await;
    assert!(status.is_success());
    assert_eq!(body["success"], true, "{body}");
    assert_eq!(body["backup"]["backup_id"], 1, "{body}");

    let later = h
        .mgr
        .get_user_graph(user)
        .unwrap()
        .read()
        .add_entity(graph_entity("after the backup"))
        .unwrap();
    let (_, body) = json_of(
        h.app(),
        authed_post("/api/backup/create", json!({"user_id": user})),
    )
    .await;
    assert_eq!(body["success"], true, "{body}");

    let (status, body) = json_of(
        h.app(),
        authed_post("/api/backup/list", json!({"user_id": user})),
    )
    .await;
    assert!(status.is_success());
    assert_eq!(body["count"], 2, "{body}");
    assert_eq!(body["backups"][0]["backup_id"], 2, "newest first: {body}");

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/backup/verify",
            json!({"user_id": user, "backup_id": 1}),
        ),
    )
    .await;
    assert_eq!(
        (&body["success"], &body["is_valid"]),
        (&json!(true), &json!(true)),
        "{body}"
    );

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/backup/restore",
            json!({"user_id": user, "backup_id": 1}),
        ),
    )
    .await;
    assert!(status.is_success(), "{body}");
    assert_eq!(body["success"], true, "{body}");
    let restored = body["restored_stores"].to_string();
    assert!(restored.contains("NOT RESTORED: shared"), "{body}");

    let graph = h.mgr.get_user_graph(user).unwrap();
    let graph = graph.read();
    assert!(
        graph.get_entity(&kept).unwrap().is_some(),
        "restored graph lost the backed-up entity"
    );
    assert!(
        graph.get_entity(&later).unwrap().is_none(),
        "restore did not roll the graph back"
    );
    drop(graph);

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/backup/purge",
            json!({"user_id": user, "keep_count": 1}),
        ),
    )
    .await;
    assert!(status.is_success());
    assert_eq!(body["purged_count"], 1, "{body}");
    let (_, body) = json_of(
        h.app(),
        authed_post("/api/backup/list", json!({"user_id": user})),
    )
    .await;
    assert_eq!(body["count"], 1, "{body}");
}

/// What one engine's backup and point-in-time restore left behind.
#[cfg(all(feature = "rocksdb", feature = "storage-redb"))]
#[derive(Debug, PartialEq)]
struct RestoredState {
    contents: Vec<String>,
    kept_entity: bool,
    later_entity: bool,
    journal_ends_at_max_lsn: bool,
}

/// Three memories and an entity, a backup, a fourth memory and a second entity,
/// then a restore of the backup truncated to the middle of the journal.
#[cfg(all(feature = "rocksdb", feature = "storage-redb"))]
async fn backup_then_point_in_time_restore(h: &Harness) -> RestoredState {
    let user = "parity-user";
    for content in [
        "alpha: the river floods every spring",
        "beta: the pass closes after the first snow",
        "gamma: the dunes move east each year",
    ] {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({"user_id": user, "content": content}),
            ),
        )
        .await;
        assert!(status.is_success(), "{body}");
    }
    let kept = h
        .mgr
        .get_user_graph(user)
        .unwrap()
        .read()
        .add_entity(graph_entity("in the backup"))
        .unwrap();

    let journal = h._dir.path().join(user).join("intent.log");
    let lsns: Vec<u64> = veld::intent_log::IntentLogIter::open_for_scan(&journal)
        .expect("scan the journal")
        .map(|r| r.expect("journal frame").lsn.0)
        .collect();
    assert!(lsns.len() >= 3, "each remember journals a frame: {lsns:?}");
    let max_lsn = lsns[lsns.len() / 2];

    let (_, body) = json_of(
        h.app(),
        authed_post("/api/backup/create", json!({"user_id": user})),
    )
    .await;
    assert_eq!(body["success"], true, "{body}");
    let backup_id = body["backup"]["backup_id"].as_u64().expect("backup id");

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": user, "content": "delta: written after the backup"}),
        ),
    )
    .await;
    assert!(status.is_success(), "{body}");
    let later = h
        .mgr
        .get_user_graph(user)
        .unwrap()
        .read()
        .add_entity(graph_entity("after the backup"))
        .unwrap();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/backup/restore",
            json!({"user_id": user, "backup_id": backup_id, "max_lsn": max_lsn}),
        ),
    )
    .await;
    assert!(status.is_success(), "{body}");
    assert_eq!(body["success"], true, "{body}");

    let last_lsn = veld::intent_log::IntentLogIter::open_for_scan(&journal)
        .expect("scan the restored journal")
        .map(|r| r.expect("restored frame").lsn.0)
        .last();
    let mut contents: Vec<String> = h
        .mgr
        .get_user_memory(user)
        .unwrap()
        .read()
        .get_all_memories()
        .unwrap()
        .iter()
        .map(|m| m.experience.content.clone())
        .collect();
    contents.sort();
    let graph = h.mgr.get_user_graph(user).unwrap();
    let graph = graph.read();
    RestoredState {
        contents,
        kept_entity: graph.get_entity(&kept).unwrap().is_some(),
        later_entity: graph.get_entity(&later).unwrap().is_some(),
        journal_ends_at_max_lsn: last_lsn == Some(max_lsn),
    }
}

#[cfg(all(feature = "rocksdb", feature = "storage-redb"))]
#[tokio::test]
async fn backup_and_point_in_time_restore_leave_the_same_store_on_both_engines() {
    use veld::config::StorageBackend;
    let rocksdb =
        backup_then_point_in_time_restore(&Harness::with_backend(StorageBackend::RocksDb)).await;
    let redb =
        backup_then_point_in_time_restore(&Harness::with_backend(StorageBackend::Redb)).await;

    assert_eq!(
        rocksdb,
        RestoredState {
            contents: vec![
                "alpha: the river floods every spring".to_string(),
                "beta: the pass closes after the first snow".to_string(),
                "gamma: the dunes move east each year".to_string(),
            ],
            kept_entity: true,
            later_entity: false,
            journal_ends_at_max_lsn: true,
        },
        "the RocksDB restore is the reference"
    );
    assert_eq!(
        redb, rocksdb,
        "redb restored a different store than RocksDB"
    );
}

#[cfg(feature = "storage-redb")]
#[tokio::test]
async fn redb_verify_route_reports_a_tampered_backup_as_invalid() {
    let h = Harness::new_redb();
    let (_, body) = json_of(
        h.app(),
        authed_post("/api/backup/create", json!({"user_id": "test-user"})),
    )
    .await;
    assert_eq!(body["success"], true, "{body}");
    let dump = h
        ._dir
        .path()
        .join("backups/.redb/test-user/1/export/users/test-user.veldport");
    let len = std::fs::metadata(&dump).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&dump)
        .unwrap()
        .set_len(len - 1)
        .unwrap();
    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/backup/verify",
            json!({"user_id": "test-user", "backup_id": 1}),
        ),
    )
    .await;
    assert_eq!(body["is_valid"], false, "{body}");
    assert_eq!(body["success"], false, "{body}");
}

// ═══════════════════════════════════════════════════════════════════════
// files.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn file_stats() {
    let h = Harness::new();
    // Handler takes user_id as query param: Query<TodoQuery>
    let (status, _) = json_of(h.app(), authed_get("/api/files/stats?user_id=test-user")).await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// mif.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn add_entity_to_graph() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/graph/entity/add",
            json!({
                "user_id": "test-user",
                "name": "Rust",
                "label": "Technology"
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn add_relationship_to_graph() {
    let h = Harness::new();

    // Add two entities first
    let _ = json_of(
        h.app(),
        authed_post(
            "/api/graph/entity/add",
            json!({"user_id": "test-user", "name": "Rust", "label": "Tech"}),
        ),
    )
    .await;
    let _ = json_of(
        h.app(),
        authed_post(
            "/api/graph/entity/add",
            json!({"user_id": "test-user", "name": "ONNX", "label": "Tech"}),
        ),
    )
    .await;

    // Add relationship
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/graph/relationship/add",
            json!({
                "user_id": "test-user",
                "from_entity": "Rust",
                "to_entity": "ONNX",
                "relation_type": "USES"
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn export_mif_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/export/mif", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// visualization.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn graph_view_html() {
    let h = Harness::new();
    let resp = h.app().oneshot(noauth_get("/graph/view")).await.unwrap();
    // graph/view is public and returns HTML
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn brain_state_fresh() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/brain/test-user")).await;
    assert!(status.is_success());
}

#[tokio::test]
async fn build_visualization_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/visualization/build", json!({"user_id": "test-user"})),
    )
    .await;
    // May return 200 or 404 on fresh state
    assert!(
        status.is_success() || status == StatusCode::NOT_FOUND,
        "build visualization returned {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// integrations.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn linear_sync_no_integration() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/sync/linear", json!({"user_id": "test-user"})),
    )
    .await;
    // Without Linear API key configured, expect graceful error
    assert!(
        status == StatusCode::OK
            || status == StatusCode::BAD_REQUEST
            || status == StatusCode::UNPROCESSABLE_ENTITY,
        "linear sync returned unexpected {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// End-to-end: remember → recall cycle
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn remember_then_list() {
    let h = Harness::new();

    // Store a memory
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "e2e-user",
                "content": "Melbourne has unpredictable weather.",
                "tags": ["weather", "melbourne"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // List memories — should have at least one
    let (status, body) = json_of(h.app(), authed_get("/api/list/e2e-user")).await;
    assert_eq!(status, StatusCode::OK);
    let memories = body.as_array().or(body["memories"].as_array());
    assert!(
        memories.map(|m| !m.is_empty()).unwrap_or(false),
        "should see stored memory in list: {body}"
    );

    // User stats should show 1 memory
    let (status, body) = json_of(h.app(), authed_get("/api/users/e2e-user/stats")).await;
    assert_eq!(status, StatusCode::OK);
    let count = body["total_memories"]
        .as_u64()
        .or(body["memory_count"].as_u64());
    assert!(
        count.unwrap_or(0) >= 1,
        "stats should show at least 1 memory: {body}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Webhook signature enforcement (/webhook/linear, /webhook/github)
// ═══════════════════════════════════════════════════════════════════════

mod webhook_signature {
    use super::*;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    // Env vars are process-global; a tokio lock because the guard is held across awaits.
    static WEBHOOK_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    const ENV_KEYS: [&str; 3] = [
        "LINEAR_WEBHOOK_SECRET",
        "GITHUB_WEBHOOK_SECRET",
        "VELD_ALLOW_UNSIGNED_WEBHOOKS",
    ];

    /// Sets the webhook env for one test and restores the previous values on drop.
    struct WebhookEnv {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl WebhookEnv {
        fn set(secret: &str) -> Self {
            let saved = ENV_KEYS
                .iter()
                .map(|k| (*k, std::env::var(k).ok()))
                .collect();
            std::env::set_var("LINEAR_WEBHOOK_SECRET", secret);
            std::env::set_var("GITHUB_WEBHOOK_SECRET", secret);
            std::env::remove_var("VELD_ALLOW_UNSIGNED_WEBHOOKS");
            Self { saved }
        }
    }

    impl Drop for WebhookEnv {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    fn sign(secret: &str, body: &[u8]) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        hex::encode(mac.finalize().into_bytes())
    }

    // Non-Issue / non-issues events return "ignored" right after the signature
    // gate, so a 200 proves the gate passed without writing a memory.
    const LINEAR_BODY: &str = r#"{"action":"create","type":"Comment","data":{"id":"c1"}}"#;
    const GITHUB_BODY: &str = r#"{"zen":"ping"}"#;

    fn linear_req(signature: &str) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri("/webhook/linear")
            .header("content-type", "application/json")
            .header("x-api-key", TEST_KEY)
            .header("linear-signature", signature)
            .body(Body::from(LINEAR_BODY))
            .unwrap()
    }

    fn github_req(signature: &str) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri("/webhook/github")
            .header("content-type", "application/json")
            .header("x-api-key", TEST_KEY)
            .header("x-github-event", "ping")
            .header("x-hub-signature-256", signature)
            .body(Body::from(GITHUB_BODY))
            .unwrap()
    }

    #[tokio::test]
    async fn empty_secret_is_unconfigured_and_rejects_empty_key_forgery() {
        let _lock = WEBHOOK_ENV_LOCK.lock().await;
        let _env = WebhookEnv::set("");
        let h = Harness::new();

        let forged = sign("", LINEAR_BODY.as_bytes());
        let status = status_of(h.app(), linear_req(&forged)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "linear");

        let forged = format!("sha256={}", sign("", GITHUB_BODY.as_bytes()));
        let status = status_of(h.app(), github_req(&forged)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "github");
    }

    #[tokio::test]
    async fn malformed_signature_is_400_not_500() {
        let _lock = WEBHOOK_ENV_LOCK.lock().await;
        let _env = WebhookEnv::set("handler-webhook-secret");
        let h = Harness::new();

        let non_hex = "zz".repeat(32);
        let short = "ab".repeat(31);
        for sig in [non_hex.clone(), short.clone()] {
            let status = status_of(h.app(), linear_req(&sig)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "linear sig {sig}");
        }
        for sig in [format!("sha256={non_hex}"), format!("sha256={short}")] {
            let status = status_of(h.app(), github_req(&sig)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "github sig {sig}");
        }
    }

    #[tokio::test]
    async fn correctly_signed_body_is_accepted() {
        let _lock = WEBHOOK_ENV_LOCK.lock().await;
        let secret = "handler-webhook-secret";
        let _env = WebhookEnv::set(secret);
        let h = Harness::new();

        let (status, body) =
            json_of(h.app(), linear_req(&sign(secret, LINEAR_BODY.as_bytes()))).await;
        assert_eq!(status, StatusCode::OK, "linear: {body}");
        assert_eq!(body["status"], "ignored", "linear: {body}");

        let sig = format!("sha256={}", sign(secret, GITHUB_BODY.as_bytes()));
        let (status, body) = json_of(h.app(), github_req(&sig)).await;
        assert_eq!(status, StatusCode::OK, "github: {body}");
        assert_eq!(body["status"], "ignored", "github: {body}");
    }
}

/// The `.route("path", method(` calls inside `fn_sig`'s body, in source order.
fn routes_in(source: &str, fn_sig: &str) -> Vec<(String, String)> {
    let start = source
        .find(fn_sig)
        .unwrap_or_else(|| panic!("{fn_sig} not found"));
    let after = &source[start..];
    let open = after.find('{').expect("function body");
    let mut depth = 0i32;
    let mut end = open;
    for (i, ch) in after[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = open + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let body = &after[open..end];
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(i) = rest.find(".route(\"") {
        rest = &rest[i + ".route(\"".len()..];
        let Some(close) = rest.find('"') else { break };
        let path = rest[..close].to_string();
        let tail = rest[close + 1..].trim_start_matches([',', ' ', '\n', '\t', '\r']);
        let method: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphabetic() || *c == '_')
            .collect();
        out.push((path, method));
        rest = &rest[close..];
    }
    out
}

/// Every route the protected router mounts refuses a request that carries no API key. The MCP
/// server reaches veld only through these routes, with the key, so this covers its tools too.
/// `/webhook/*` and `/api/user_auth/*` skip the key check by design and answer through their
/// own signature and session checks, or 503 while the feature or its secret is not configured.
#[tokio::test]
async fn every_protected_route_refuses_a_request_without_an_api_key() {
    // The server assembles its app inline rather than through this harness, so both of its
    // protected-route branches (with and without rate limiting) must carry the auth layer.
    let server = include_str!("../src/server.rs");
    let layered = server
        .split(".protected_routes(metrics_public)")
        .skip(1)
        .filter(|after| {
            after
                .lines()
                .take(2)
                .any(|l| l.contains(".layer(axum::middleware::from_fn(auth::auth_middleware))"))
        })
        .count();
    assert_eq!(
        layered, 2,
        "server.rs mounts protected routes without auth_middleware"
    );

    let h = Harness::new();
    let source = include_str!("../src/handlers/router.rs");
    let routes = routes_in(source, "pub fn build_protected_routes(");
    assert!(
        routes.len() > 150,
        "the route parser found only {} routes",
        routes.len()
    );

    let mut answered = Vec::new();
    for (path, method) in &routes {
        let uri: String = path
            .split('/')
            .map(|seg| if seg.starts_with('{') { "probe" } else { seg })
            .collect::<Vec<_>>()
            .join("/");
        let method = match method.as_str() {
            "get" => Method::GET,
            "post" => Method::POST,
            "put" => Method::PUT,
            "delete" => Method::DELETE,
            "patch" => Method::PATCH,
            other => panic!("{path}: unrecognised method builder `{other}`"),
        };
        let req = Request::builder()
            .method(method.clone())
            .uri(&uri)
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let status = h.app().oneshot(req).await.unwrap().status();
        // 503: user auth disabled, or a webhook rejecting an unsigned call with no secret set.
        let own_check = path.starts_with("/api/user_auth/") || path.starts_with("/webhook/");
        let refused = matches!(status.as_u16(), 401 | 403)
            || (own_check && status == StatusCode::SERVICE_UNAVAILABLE);
        if !refused {
            answered.push(format!("{method} {uri} -> {status}"));
        }
    }
    assert!(
        answered.is_empty(),
        "{} of {} protected routes answered without an API key: {answered:#?}",
        answered.len(),
        routes.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Declared (typed) entities on remember / upsert / batch
// ═══════════════════════════════════════════════════════════════════════

const SERVE: &str = "commitwork:admin/serve.mjs";
const COMMIT: &str = "commitwork@1a2b3c4d5e6f";

fn declared_pair() -> serde_json::Value {
    json!([
        {"name": SERVE, "label": "FilePath"},
        {"name": COMMIT, "label": "Commit", "weight": 0.8}
    ])
}

/// The graph entity named exactly `name` (exact or case-insensitive tier only).
fn named_graph_entity(
    h: &Harness,
    user: &str,
    name: &str,
) -> Option<veld::graph_memory::EntityNode> {
    let graph = h.mgr.get_user_graph(user).expect("graph");
    let graph = graph.read();
    graph.find_entity_by_name_strict(name).expect("lookup")
}

fn edges_of(
    h: &Harness,
    user: &str,
    entity: &veld::uuid::Uuid,
) -> Vec<veld::graph_memory::RelationshipEdge> {
    let graph = h.mgr.get_user_graph(user).expect("graph");
    let graph = graph.read();
    graph.get_entity_relationships(entity).expect("edges")
}

fn stored_tags(h: &Harness, user: &str, id: &str) -> Vec<String> {
    let id = veld::uuid::Uuid::parse_str(id).unwrap();
    let memory = h.mgr.get_user_memory(user).expect("memory");
    let guard = memory.read();
    guard
        .get_memory(&veld::memory::MemoryId(id))
        .expect("stored")
        .experience
        .tags
        .clone()
}

#[tokio::test]
async fn declared_entities_reach_the_graph_typed_and_survive_the_derived_cut() {
    use veld::graph_memory::{EdgeSource, EntityLabel, LtpStatus};

    let h = Harness::new();
    let user = "declared-user";
    // Twelve tags alone exceed max_entities_per_memory (10) before NER and YAKE add more.
    let tags: Vec<String> = [
        "routing",
        "dispatch",
        "middleware",
        "session",
        "cookie",
        "redirect",
        "handler",
        "timeout",
        "listener",
        "gateway",
        "upstream",
        "fallback",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": user,
                "content": "Alice Moreau and Bijan Farahani moved the auth check in Kubernetes \
                            ahead of the REST gateway dispatch for SHO-412 and JIRA-77 in Lisbon.",
                "tags": tags,
                "entities": declared_pair()
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id_str = body["id"].as_str().expect("id").to_string();
    let id = veld::uuid::Uuid::parse_str(&id_str).unwrap();
    wait_for_episode_entity(&h, user, id, &SERVE.to_lowercase()).await;

    let names = episode_entity_names(&h, user, &id);
    assert!(names.contains(&COMMIT.to_lowercase()), "{names:?}");
    // The derived list is cut to 10; the two declared ones are on top of that.
    assert_eq!(names.len(), 12, "{names:?}");

    let serve = named_graph_entity(&h, user, SERVE).expect("FilePath entity");
    assert!(
        serve.labels.contains(&EntityLabel::FilePath),
        "{:?}",
        serve.labels
    );
    let commit = named_graph_entity(&h, user, COMMIT).expect("Commit entity");
    assert!(
        commit.labels.contains(&EntityLabel::Commit),
        "{:?}",
        commit.labels
    );

    // Edges are written after the episode, by the same background pass.
    let mut serve_edges = Vec::new();
    for _ in 0..300 {
        serve_edges = edges_of(&h, user, &serve.uuid);
        if serve_edges.len() >= 11 && !edges_of(&h, user, &commit.uuid).is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert_eq!(
        serve_edges.len(),
        11,
        "linked to every other entity of the memory"
    );
    for e in &serve_edges {
        assert_eq!(
            e.ltp_status,
            LtpStatus::Weekly,
            "declared edges are protected"
        );
        assert_eq!(e.created_by, EdgeSource::Explicit);
        assert!(
            (e.strength - 0.9).abs() < 1e-6,
            "default weight: {}",
            e.strength
        );
    }
    for e in edges_of(&h, user, &commit.uuid) {
        let other = if e.from_entity == commit.uuid {
            e.to_entity
        } else {
            e.from_entity
        };
        let want = if other == serve.uuid { 0.9 } else { 0.8 };
        assert!((e.strength - want).abs() < 1e-6, "{} vs {want}", e.strength);
    }

    // Stored as the memory's authored tags, leading the list, labelled declared.
    let tags = stored_tags(&h, user, &id_str);
    assert_eq!(
        tags[..2],
        [SERVE.to_string(), COMMIT.to_string()],
        "{tags:?}"
    );
    let memory = h.mgr.get_user_memory(user).expect("memory");
    let row = memory
        .read()
        .tag_source_store()
        .get(&veld::memory::MemoryId(id))
        .expect("ledger")
        .expect("row");
    assert_eq!(
        row.tags[0].source,
        veld::memory::tag_sources::TagSource::Declared
    );
    assert_eq!(
        row.tags[1].source,
        veld::memory::tag_sources::TagSource::Declared
    );
}

#[tokio::test]
async fn invalid_declared_entities_are_refused_not_filed_as_other() {
    let h = Harness::new();
    let user = "declared-invalid";
    let remember = |entities: serde_json::Value| {
        authed_post(
            "/api/remember",
            json!({"user_id": user, "content": "An entity declaration to refuse.", "entities": entities}),
        )
    };

    let (status, body) = json_of(
        h.app(),
        remember(json!([{"name": "x", "label": "Filepth"}])),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let text = body.to_string();
    for label in [
        "Filepth", "FilePath", "Commit", "Template", "Person", "Module",
    ] {
        assert!(text.contains(label), "{label} missing from {text}");
    }

    let (status, body) = json_of(
        h.app(),
        remember(json!([{"name": "x", "label": "Tool", "weight": 1.5}])),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let many: Vec<serde_json::Value> = (0..51)
        .map(|i| json!({"name": format!("tool:t{i}"), "label": "Tool"}))
        .collect();
    let (status, body) = json_of(h.app(), remember(json!(many))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            json!({"user_id": user, "external_id": "ext-bad", "content": "Upsert to refuse.",
                   "entities": [{"name": "x", "label": "Nope"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    // Nothing above was stored.
    let memory = h.mgr.get_user_memory(user).expect("memory");
    assert!(memory.read().get_all_memories().expect("all").is_empty());

    // In a batch the bad item fails and the good one is stored, as an unknown type does.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember/batch",
            json!({"user_id": user, "memories": [
                {"content": "Batch item with a valid declaration.",
                 "entities": [{"name": "tool:Edit", "label": "tool"}]},
                {"content": "Batch item with a bad weight.",
                 "entities": [{"name": "tool:Edit", "label": "Tool", "weight": -1}]}
            ]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["created"], 1, "{body}");
    assert_eq!(body["errors"][0]["index"], 1, "{body}");
    assert!(
        body["errors"][0]["error"].to_string().contains("weight"),
        "{body}"
    );
}

#[tokio::test]
async fn upsert_and_batch_carry_declared_entities() {
    use veld::graph_memory::EntityLabel;

    let h = Harness::new();
    let user = "declared-upsert";
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            json!({"user_id": user, "external_id": "ext-typed", "content": "First pass on the route table.",
                   "entities": [{"name": SERVE, "label": "FilePath"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id = veld::uuid::Uuid::parse_str(body["id"].as_str().unwrap()).unwrap();
    // Upsert builds the graph before it responds.
    assert!(episode_entity_names(&h, user, &id).contains(&SERVE.to_lowercase()));
    assert!(named_graph_entity(&h, user, SERVE)
        .unwrap()
        .labels
        .contains(&EntityLabel::FilePath));

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            json!({"user_id": user, "external_id": "ext-typed", "content": "Second pass on the route table.",
                   "entities": [{"name": COMMIT, "label": "Commit"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["was_update"], json!(true), "{body}");
    let names = episode_entity_names(&h, user, &id);
    assert!(names.contains(&COMMIT.to_lowercase()), "{names:?}");
    assert!(
        !names.contains(&SERVE.to_lowercase()),
        "an update rebuilds the episode from its own entities: {names:?}"
    );
    assert!(named_graph_entity(&h, user, COMMIT)
        .unwrap()
        .labels
        .contains(&EntityLabel::Commit));
    assert_eq!(stored_tags(&h, user, &id.to_string())[0], COMMIT);

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember/batch",
            json!({"user_id": user, "memories": [
                {"content": "Batch item about a session.",
                 "entities": [{"name": "cc:sess-42", "label": "Session"},
                              {"name": "cc:sess-42/agent-7", "label": "Agent"}]},
                {"content": "Batch item with no extraction, still typed."}
            ], "options": {"extract_entities": true}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["created"], 2, "{body}");
    let id = veld::uuid::Uuid::parse_str(body["memory_ids"][0].as_str().unwrap()).unwrap();
    wait_for_episode_entity(&h, user, id, "cc:sess-42/agent-7").await;
    assert!(named_graph_entity(&h, user, "cc:sess-42")
        .unwrap()
        .labels
        .contains(&EntityLabel::Session));
    assert!(named_graph_entity(&h, user, "cc:sess-42/agent-7")
        .unwrap()
        .labels
        .contains(&EntityLabel::Agent));

    // The no-extraction batch path carries them too.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember/batch",
            json!({"user_id": user, "memories": [
                {"content": "Batch item on the cheap path.", "tags": ["cheap"],
                 "entities": [{"name": "skill:/sauron", "label": "Tool"}]}
            ], "options": {"extract_entities": false}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id_str = body["memory_ids"][0].as_str().unwrap().to_string();
    let id = veld::uuid::Uuid::parse_str(&id_str).unwrap();
    wait_for_episode_entity(&h, user, id, "skill:/sauron").await;
    assert_eq!(stored_tags(&h, user, &id_str), ["skill:/sauron", "cheap"]);
}

#[tokio::test]
async fn a_write_without_entities_behaves_as_before() {
    let h = Harness::new();
    let content =
        "Rotated the staging TLS certificate and restarted the Envoy sidecars in Frankfurt.";
    let mut stored = Vec::new();
    for (user, body) in [
        (
            "plain-omitted",
            json!({"user_id": "plain-omitted", "content": content, "tags": ["tls"]}),
        ),
        (
            "plain-empty",
            json!({"user_id": "plain-empty", "content": content, "tags": ["tls"], "entities": []}),
        ),
    ] {
        let (status, resp) = json_of(h.app(), authed_post("/api/remember", body)).await;
        assert_eq!(status, StatusCode::OK, "{resp}");
        let mut keys: Vec<&str> = resp
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, ["id", "success"], "{resp}");
        let id_str = resp["id"].as_str().unwrap().to_string();
        let id = veld::uuid::Uuid::parse_str(&id_str).unwrap();
        wait_for_episode_entity(&h, user, id, "tls").await;
        let mut names = episode_entity_names(&h, user, &id);
        names.sort();
        stored.push((stored_tags(&h, user, &id_str), names));
    }
    assert_eq!(
        stored[0], stored[1],
        "omitted and empty must store the same thing"
    );
    assert_eq!(stored[0].0[0], "tls", "caller tags still lead");
}

#[tokio::test]
async fn generic_words_and_hubs_take_no_new_edges_but_declared_entities_do() {
    use veld::constants::{ENTITY_DEGREE_PRUNE_TARGET, MAX_ENTITY_DEGREE};
    use veld::graph_memory::{EntityLabel, EntityNode, RelationType, RelationshipEdge};

    let h = Harness::new();
    let user = "gate-user";

    // 1. Generic words do not become derived entities, however often they appear.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": user,
                   "content": "File File FILE. Code code CODE. The file has code; the code has lines. \
                               Lines of code in the file, first test, moved line, operator description.",
                   "tags": ["File", "Code", "Lines", "kestrel"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id = veld::uuid::Uuid::parse_str(body["id"].as_str().unwrap()).unwrap();
    wait_for_episode_entity(&h, user, id, "kestrel").await;
    for word in [
        "file",
        "files",
        "code",
        "first",
        "moved",
        "line",
        "lines",
        "test",
        "description",
        "operator",
    ] {
        assert!(
            named_graph_entity(&h, user, word).is_none(),
            "generic word '{word}' became an entity"
        );
    }

    // 2. A derived entity already at the cap gets no edge and causes no prune; a declared
    //    entity at the cap still links, and the cap's pruning makes room for it.
    let graph = h.mgr.get_user_graph(user).expect("graph");
    let node = |name: &str| EntityNode {
        uuid: veld::uuid::Uuid::new_v4(),
        name: name.to_string(),
        labels: vec![EntityLabel::Concept],
        created_at: veld::chrono::Utc::now(),
        last_seen_at: veld::chrono::Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: std::collections::HashMap::new(),
        name_embedding: None,
        salience: 0.5,
        is_proper_noun: true,
        pii_classification: Default::default(),
    };
    let seed_hub = |name: &str| -> veld::uuid::Uuid {
        let g = graph.read();
        let hub = g.add_entity(node(name)).unwrap();
        for i in 0..MAX_ENTITY_DEGREE {
            let peer = g.add_entity(node(&format!("{name}-peer-{i}"))).unwrap();
            let now = veld::chrono::Utc::now();
            g.add_relationship(RelationshipEdge {
                uuid: veld::uuid::Uuid::new_v4(),
                from_entity: hub,
                to_entity: peer,
                relation_type: RelationType::RelatedTo,
                strength: 0.3,
                created_at: now,
                valid_at: now,
                invalidated_at: None,
                source_episode_id: None,
                context: String::new(),
                last_activated: now,
                activation_count: 1,
                ltp_status: Default::default(),
                tier: Default::default(),
                activation_timestamps: None,
                entity_confidence: None,
                created_by: Default::default(),
                forward_strength: 0.3,
                backward_strength: 0.3,
                strengthening_sources: Default::default(),
                last_corroborated: None,
            })
            .unwrap();
        }
        hub
    };
    let lighthouse = seed_hub("lighthouse");
    let bash = seed_hub("tool:Bash");
    let edge_ids = |uuid: &veld::uuid::Uuid| -> std::collections::BTreeSet<veld::uuid::Uuid> {
        edges_of(&h, user, uuid)
            .into_iter()
            .map(|e| e.uuid)
            .collect()
    };
    let lighthouse_before = edge_ids(&lighthouse);
    assert_eq!(lighthouse_before.len(), MAX_ENTITY_DEGREE);

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": user,
                   "content": "The keeper climbed the stairs at dusk.",
                   "tags": ["lighthouse", "keeper"],
                   "entities": [{"name": "tool:Bash", "label": "Tool"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id = veld::uuid::Uuid::parse_str(body["id"].as_str().unwrap()).unwrap();
    wait_for_episode_entity(&h, user, id, "keeper").await;
    wait_for_background_work(&h).await;
    let keeper = named_graph_entity(&h, user, "keeper").expect("keeper").uuid;

    assert_eq!(
        edge_ids(&lighthouse),
        lighthouse_before,
        "a derived hub gained or lost an edge"
    );
    let bash_edges = edges_of(&h, user, &bash);
    // Its first new edge cut it back to the target; the memory's later ones stayed.
    assert!(
        (ENTITY_DEGREE_PRUNE_TARGET..MAX_ENTITY_DEGREE).contains(&bash_edges.len()),
        "the declared hub holds {} edges, outside the prune window",
        bash_edges.len()
    );
    assert!(
        bash_edges
            .iter()
            .any(|e| e.from_entity == keeper || e.to_entity == keeper),
        "the declared hub was skipped"
    );
    assert!(named_graph_entity(&h, user, "tool:Bash")
        .unwrap()
        .labels
        .contains(&EntityLabel::Tool));
}

#[tokio::test]
async fn a_declared_file_path_is_recalled_by_its_file_name() {
    let reranker = veld::embeddings::cross_encoder::default_cross_encoder();
    assert!(
        reranker.is_cached(),
        "this ranking needs the reranker, and it is not cached at {:?}; \
         fetch it with `cargo run --bin model-fetch -- --cross-encoder`",
        reranker.cache_dir()
    );
    let h = Harness::new();
    let user = "recall-declared";
    let distractors = [
        "Changed the retry budget in the payment worker from three to five attempts.",
        "observe.mjs now batches telemetry before flushing it to the collector.",
        "server.mjs was renamed during the express upgrade and nothing else changed.",
        "The nightly backup job moved to 02:00 to avoid the index rebuild window.",
        "Bumped the Postgres connection pool after the load test saturated it.",
        "What changed in the onboarding email copy: shorter subject, one call to action.",
        "The CSV export now escapes embedded quotes in the notes column.",
        "Dashboard tiles were reordered so error rate sits beside latency.",
        "The webhook signature check now rejects timestamps older than five minutes.",
        "Pinned the image digest for the scanner container in CI.",
        "Changed the cache TTL for avatars from a day to an hour.",
        "admin/routes/serve-static.mjs gained a content-type allowlist.",
        "Rewrote the changelog generator to group entries by scope.",
        "Changed the feature flag default for the new editor to off.",
        "The search page debounces keystrokes by 250 milliseconds now.",
        "Moved the docs build to a separate workflow so it no longer blocks tests.",
        "The rate limiter keys on the API token rather than the client address.",
        "Changed how the CLI prints errors: one line, then the hint.",
        "Deleted the unused geo lookup module and its fixtures.",
        "The worker pool shrinks to two threads when the queue is idle.",
    ];
    for content in distractors {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({"user_id": user, "content": content}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": user,
                   "content": "Moved the auth check ahead of route dispatch and dropped the legacy redirect.",
                   "entities": [{"name": SERVE, "label": "FilePath"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let target = body["id"].as_str().unwrap().to_string();
    let target_uuid = veld::uuid::Uuid::parse_str(&target).unwrap();
    wait_for_episode_entity(&h, user, target_uuid, &SERVE.to_lowercase()).await;
    // The embedding is deferred too; ranked before it lands, the target has no vector.
    wait_for_background_work(&h).await;

    for query in [
        "what changed in serve.mjs",
        "what changed in admin/serve.mjs",
    ] {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({"user_id": user, "query": query, "limit": 5}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let ids: Vec<&str> = body["memories"]
            .as_array()
            .expect("memories")
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids.first().copied(),
            Some(target.as_str()),
            "{query}: {:#}",
            body["memories"]
        );
    }
}
