//! Handler pipeline tests: validation edges and behaviour of the HTTP handlers
//! for remember, recall, proactive_context and CRUD, driven through the router.
//!
//! Ported from upstream shodh-memory `tests/handler_pipeline_tests.rs` (#229, with
//! the content-dedup block from #459). Upstream tests for behaviour veld does not
//! have (the robotics fields and strict mode, the minimum content length) are not
//! carried; the ledger row for #229 names them.
//!
//! Run with: `cargo test --test handler_pipeline_tests`

use std::sync::{Arc, Once};

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

use veld::{
    config::ServerConfig,
    handlers::{
        build_probe_routes, build_protected_routes, build_public_routes, MultiUserMemoryManager,
    },
};

// ═══════════════════════════════════════════════════════════════════════
// Test infrastructure (mirrors handler_tests.rs)
// ═══════════════════════════════════════════════════════════════════════

const TEST_KEY: &str = "pipeline-test-key";
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

fn authed_post(uri: &str, body: Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(bytes))
        .unwrap()
}

fn authed_delete(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::DELETE)
        .uri(uri)
        .header("x-api-key", TEST_KEY)
        .body(Body::empty())
        .unwrap()
}

// ── response helpers ──

async fn status_of(app: Router, req: Request<Body>) -> StatusCode {
    app.oneshot(req).await.unwrap().status()
}

async fn json_of(app: Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let val = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).to_string()))
    };
    (status, val)
}

/// Waits until the work `remember` hands to the background (embedding, graph, lineage)
/// has finished, so a recall sees the vectors and a delete cannot race a late write-back.
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

/// Helper: store a memory and return its ID
async fn store_memory(h: &Harness, user_id: &str, content: &str) -> String {
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": user_id,
                "content": content,
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "store_memory failed: {body}");
    body["id"]
        .as_str()
        .expect("missing id in response")
        .to_string()
}

/// Helper: store a memory with full options and return its ID
async fn store_memory_typed(
    h: &Harness,
    user_id: &str,
    content: &str,
    memory_type: &str,
    tags: Vec<&str>,
) -> String {
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": user_id,
                "content": content,
                "memory_type": memory_type,
                "tags": tags,
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "store_memory_typed failed: {body}");
    body["id"].as_str().expect("missing id").to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// REMEMBER HANDLER TESTS
// ═══════════════════════════════════════════════════════════════════════

mod remember_tests {
    use super::*;

    // ── Happy paths ──

    #[tokio::test]
    async fn remember_happy_path() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "The robot successfully navigated the corridor",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["id"].is_string());
        assert_eq!(body["success"], true);
    }

    #[tokio::test]
    async fn remember_with_all_optional_fields() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Discovered a critical performance issue in the vector search",
                    "memory_type": "Discovery",
                    "tags": ["performance", "vector-search"],
                    "emotional_valence": -0.3,
                    "emotional_arousal": 0.7,
                    "emotion": "concern",
                    "source_type": "ai_generated",
                    "credibility": 0.9,
                    "episode_id": "session-42",
                    "sequence_number": 3,
                    "importance": 0.8,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["id"].is_string());
    }

    #[tokio::test]
    async fn remember_with_agent_id() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Agent completed task analysis",
                    "agent_id": "agent-007",
                    "run_id": "run-42",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["id"].is_string());
    }

    #[tokio::test]
    async fn remember_with_parent_id() {
        let h = Harness::new();
        let parent_id = store_memory(&h, "test-user", "Parent research topic").await;
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Sub-topic of parent research",
                    "parent_id": parent_id,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["id"].is_string());
    }

    #[tokio::test]
    async fn remember_minimal_content_at_threshold() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "1234567890",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn remember_unicode_content() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "日本語テスト Unicode 中文 العربية",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn remember_different_memory_types() {
        let h = Harness::new();
        let types = [
            "Observation",
            "Decision",
            "Learning",
            "Error",
            "Discovery",
            "Pattern",
            "Context",
            "Task",
            "CodeEdit",
            "FileAccess",
            "Search",
            "Command",
            "Conversation",
        ];
        for t in types {
            let (status, body) = json_of(
                h.app(),
                authed_post(
                    "/api/remember",
                    json!({
                        "user_id": "test-user",
                        "content": format!("Memory of type {t}"),
                        "memory_type": t,
                    }),
                ),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "Failed for type: {t}");
            assert!(body["id"].is_string(), "No id for type: {t}");
        }
    }

    // ── Validation errors ──

    #[tokio::test]
    async fn remember_empty_user_id_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "",
                    "content": "test content",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_user_id_with_path_traversal_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "../etc/passwd",
                    "content": "test content",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_user_id_with_special_chars_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "user<script>alert(1)</script>",
                    "content": "test content",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_user_id_too_long_rejected() {
        let h = Harness::new();
        let long_id = "a".repeat(200);
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": long_id,
                    "content": "test content",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_valid_user_id_with_allowed_chars() {
        let h = Harness::new();
        // alphanumeric, dash, underscore, @, . are allowed
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "user-name_123@domain.com",
                    "content": "test content",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn remember_empty_content_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_whitespace_only_content_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "   \n\t  ",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_emotional_valence_out_of_range_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Docked at the north pier after the storm passed",
                    "emotional_valence": 1.5,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_emotional_valence_negative_limit() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Docked at the north pier after the storm passed",
                    "emotional_valence": -1.5,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_emotional_arousal_out_of_range_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Docked at the north pier after the storm passed",
                    "emotional_arousal": 1.5,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_emotional_arousal_negative_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Docked at the north pier after the storm passed",
                    "emotional_arousal": -0.1,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn remember_valid_emotional_boundary_values() {
        let h = Harness::new();
        // Exact boundary values should be accepted
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "boundary test",
                    "emotional_valence": -1.0,
                    "emotional_arousal": 0.0,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "boundary test 2",
                    "emotional_valence": 1.0,
                    "emotional_arousal": 1.0,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    // ── Batch remember ──

    #[tokio::test]
    async fn batch_remember_happy_path() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "First memory in batch"},
                        {"content": "Second memory in batch"},
                        {"content": "Third memory in batch"},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["created"], 3);
        assert!(body["memory_ids"].is_array());
    }

    #[tokio::test]
    async fn batch_remember_empty_batch() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["created"], 0);
    }

    #[tokio::test]
    async fn batch_remember_with_typed_memories() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "An observation about Rust memory safety", "memory_type": "Observation"},
                        {"content": "A decision to use power-law decay model", "memory_type": "Decision"},
                        {"content": "A learning about Hebbian plasticity rules", "memory_type": "Learning"},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["created"], 3);
    }

    #[tokio::test]
    async fn batch_remember_with_empty_content_item() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "Valid memory about graph algorithms"},
                        {"content": ""},
                        {"content": "Another valid memory about vector search"},
                    ],
                }),
            ),
        )
        .await;
        // Should partially succeed — valid items stored, empty one errors
        assert_eq!(status, StatusCode::OK);
        let created = body["created"].as_i64().unwrap_or(0);
        assert!(created >= 2, "Expected at least 2 created, got {created}");
    }

    #[tokio::test]
    async fn batch_remember_refuses_out_of_range_emotion_per_item() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "Docked at the north pier after the storm passed"},
                        {"content": "Refuelled at the south quay", "emotional_valence": 1.5},
                        {"content": "Waited out the fog in the harbour", "emotional_arousal": -0.1},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["created"].as_i64(), Some(1), "{body}");
        let errors = body["errors"].as_array().expect("errors array");
        let indices: Vec<i64> = errors.iter().filter_map(|e| e["index"].as_i64()).collect();
        assert_eq!(indices, vec![1, 2], "{body}");
        assert!(errors[0]["error"]
            .as_str()
            .is_some_and(|e| e.contains("emotional_valence")));
        assert!(errors[1]["error"]
            .as_str()
            .is_some_and(|e| e.contains("emotional_arousal")));
    }

    #[tokio::test]
    async fn proactive_context_refuses_out_of_range_emotion() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "Docked at the north pier after the storm passed",
                    "emotional_valence": -1.5,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn batch_remember_invalid_user_id() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "",
                    "memories": [{"content": "test"}],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // ── Upsert ──

    #[tokio::test]
    async fn upsert_creates_new_memory() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Upserted memory content",
                    "external_id": "ext-123",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["id"].is_string());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RECALL HANDLER TESTS
// ═══════════════════════════════════════════════════════════════════════

mod recall_tests {
    use super::*;

    #[tokio::test]
    async fn recall_happy_path_empty_db() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "something that does not exist",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn recall_after_remember() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The Vamana graph index uses greedy search for nearest neighbor queries",
        )
        .await;
        wait_for_background_work(&h).await;
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "Vamana graph nearest neighbor",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"]
            .as_array()
            .expect("memories should be array");
        assert!(!memories.is_empty(), "Should recall the stored memory");
    }

    #[tokio::test]
    async fn recall_with_limit() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory one about Rust ownership").await;
        store_memory(&h, "test-user", "Memory two about Rust lifetimes").await;
        store_memory(&h, "test-user", "Memory three about Rust borrowing").await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "Rust memory management",
                    "limit": 2,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.len() <= 2, "Should respect limit");
    }

    #[tokio::test]
    async fn recall_semantic_mode() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Machine learning model training pipeline").await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "ML model training",
                    "mode": "semantic",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn recall_associative_mode() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Knowledge graph with spreading activation").await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "graph traversal activation",
                    "mode": "associative",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn recall_temporal_mode() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Temporal memory test").await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "temporal test",
                    "mode": "temporal",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn recall_hybrid_mode() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Hybrid retrieval combines multiple signals",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "hybrid retrieval signals",
                    "mode": "hybrid",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn recall_empty_user_id_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "",
                    "query": "test",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn recall_by_tags() {
        let h = Harness::new();
        store_memory_typed(
            &h,
            "test-user",
            "Tagged memory about Rust",
            "Observation",
            vec!["rust", "programming"],
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall/tags",
                json!({
                    "user_id": "test-user",
                    "tags": ["rust"],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn recall_by_date() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory for date range test").await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall/date",
                json!({
                    "user_id": "test-user",
                    "start": "2020-01-01T00:00:00Z",
                    "end": "2030-01-01T00:00:00Z",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn recall_nonexistent_user_returns_empty() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "nonexistent-user-xyz",
                    "query": "anything",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PROACTIVE CONTEXT TESTS
// ═══════════════════════════════════════════════════════════════════════

mod proactive_context_tests {
    use super::*;

    #[tokio::test]
    async fn proactive_context_happy_path() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "Working on the vector search implementation",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn proactive_context_with_auto_ingest() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "Debugging the lineage inference pipeline for causal chain analysis",
                    "auto_ingest": true,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // Verify the endpoint accepts auto_ingest and returns successfully.
        // The ingested_memory_id may or may not be present depending on async timing,
        // so we just verify the response structure is valid.
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn proactive_context_with_max_results() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "test context",
                    "max_results": 3,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.len() <= 3);
    }

    #[tokio::test]
    async fn proactive_context_with_semantic_threshold() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "test context",
                    "semantic_threshold": 0.9,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn proactive_context_with_memory_types_filter() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "test context",
                    "memory_types": ["Decision", "Learning"],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn proactive_context_empty_user_id_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "",
                    "context": "test",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn proactive_context_alias_endpoint() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/context",
                json!({
                    "user_id": "test-user",
                    "context": "test via alias",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn context_summary() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/context_summary",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CRUD HANDLER TESTS
// ═══════════════════════════════════════════════════════════════════════

mod crud_tests {
    use super::*;

    #[tokio::test]
    async fn get_memory_by_id() {
        let h = Harness::new();
        let id = store_memory(&h, "test-user", "Memory to retrieve").await;
        let (status, body) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // MemoryWithHierarchy uses #[serde(flatten)] on Memory — response is flat
        // The experience.content is nested under "experience"
        assert!(body["experience"]["content"]
            .as_str()
            .unwrap()
            .contains("Memory to retrieve"));
    }

    #[tokio::test]
    async fn get_memory_not_found() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_get("/api/memory/00000000-0000-0000-0000-000000000000?user_id=test-user"),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn list_memories_empty() {
        let h = Harness::new();
        let (status, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn list_memories_after_store() {
        let h = Harness::new();
        store_memory(&h, "test-user", "First memory").await;
        store_memory(&h, "test-user", "Second memory").await;

        let (status, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.len() >= 2, "Should list at least 2 memories");
    }

    #[tokio::test]
    async fn delete_memory_by_id() {
        let h = Harness::new();
        let id = store_memory(&h, "test-user", "Memory to delete").await;
        wait_for_background_work(&h).await;
        let status = status_of(
            h.app(),
            authed_delete(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (status, _) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn forget_by_id_post() {
        let h = Harness::new();
        let id = store_memory(&h, "test-user", "Memory to forget").await;
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget",
                json!({
                    "user_id": "test-user",
                    "memory_id": id,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn forget_by_age() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Old memory to forget by age").await;
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/age",
                json!({
                    "user_id": "test-user",
                    "days_old": 0,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn forget_by_importance() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Low importance memory").await;
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/importance",
                json!({
                    "user_id": "test-user",
                    "threshold": 1.0,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn forget_by_pattern() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Pattern memory test XYZ").await;
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/pattern",
                json!({
                    "user_id": "test-user",
                    "pattern": "XYZ",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn forget_by_pattern_invalid_regex_rejected() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/pattern",
                json!({
                    "user_id": "test-user",
                    "pattern": "((",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn forget_by_pattern_too_long_rejected() {
        let h = Harness::new();
        // Handler rejects patterns > 1000 chars
        let long_pattern = "a".repeat(1001);
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/pattern",
                json!({
                    "user_id": "test-user",
                    "pattern": long_pattern,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn forget_by_tags() {
        let h = Harness::new();
        store_memory_typed(
            &h,
            "test-user",
            "Tagged to forget",
            "Observation",
            vec!["deleteme"],
        )
        .await;
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/tags",
                json!({
                    "user_id": "test-user",
                    "tags": ["deleteme"],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn forget_by_date_range() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory in date range").await;
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/date",
                json!({
                    "user_id": "test-user",
                    "start": "2020-01-01T00:00:00Z",
                    "end": "2030-01-01T00:00:00Z",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn clear_all_without_confirm_rejected() {
        let h = Harness::new();
        // ClearAllRequest.confirm is required (non-Option), so missing field → 422 from Axum
        let status = status_of(
            h.app(),
            authed_post(
                "/api/memories/clear",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert!(
            status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY,
            "Expected 400 or 422, got {status}"
        );
    }

    #[tokio::test]
    async fn clear_all_wrong_confirm_rejected() {
        let h = Harness::new();
        let status = status_of(
            h.app(),
            authed_post(
                "/api/memories/clear",
                json!({
                    "user_id": "test-user",
                    "confirm": "WRONG",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn clear_all_with_confirm() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory to clear").await;
        wait_for_background_work(&h).await;
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/memories/clear",
                json!({
                    "user_id": "test-user",
                    "confirm": "CONFIRM",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.is_empty(), "All memories should be cleared");
    }

    #[tokio::test]
    async fn list_memories_post_endpoint() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Listed via POST").await;
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/memories",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn list_memories_get_endpoint() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Listed via GET").await;
        let (status, body) = json_of(h.app(), authed_get("/api/memories?user_id=test-user")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CONSOLIDATION & INDEX TESTS
// ═══════════════════════════════════════════════════════════════════════

mod consolidation_tests {
    use super::*;

    #[tokio::test]
    async fn consolidate_empty_db() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/consolidate",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert!(
            status == StatusCode::ACCEPTED || status == StatusCode::OK,
            "Expected 200 or 202, got {status}"
        );
    }

    #[tokio::test]
    async fn verify_index_integrity() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/index/verify",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["healthy"].is_boolean() || body["status"].is_string());
    }

    #[tokio::test]
    async fn repair_index() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/index/repair",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GRAPH HANDLER TESTS
// ═══════════════════════════════════════════════════════════════════════

mod graph_tests {
    use super::*;

    #[tokio::test]
    async fn graph_stats_empty() {
        let h = Harness::new();
        let (status, body) = json_of(h.app(), authed_get("/api/graph/test-user/stats")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_object());
    }

    #[tokio::test]
    async fn graph_data_endpoint() {
        let h = Harness::new();
        let (status, _) = json_of(h.app(), authed_get("/api/graph/data/test-user")).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn graph_find_entity_not_found() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/graph/entity/find",
                json!({
                    "user_id": "test-user",
                    "entity_name": "nonexistent_entity",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.is_null(), "an unknown entity is found as null: {body}");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LINEAGE HANDLER TESTS
// ═══════════════════════════════════════════════════════════════════════

mod lineage_tests {
    use super::*;

    #[tokio::test]
    async fn lineage_list_edges_empty() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/edges",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["edges"].is_array());
    }

    #[tokio::test]
    async fn lineage_stats() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/stats",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_object());
    }

    #[tokio::test]
    async fn lineage_trace_nonexistent_memory() {
        let h = Harness::new();
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/trace",
                json!({
                    "user_id": "test-user",
                    "memory_id": "00000000-0000-0000-0000-000000000000",
                }),
            ),
        )
        .await;
        assert!(
            status == StatusCode::OK || status == StatusCode::NOT_FOUND,
            "Expected 200 or 404, got {status}"
        );
    }

    #[tokio::test]
    async fn lineage_add_manual_edge() {
        let h = Harness::new();
        let id1 = store_memory(&h, "test-user", "Cause memory for lineage testing").await;
        let id2 = store_memory(&h, "test-user", "Effect memory for lineage testing").await;

        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/link",
                json!({
                    "user_id": "test-user",
                    "from_memory_id": id1,
                    "to_memory_id": id2,
                    "relation": "Caused",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SEARCH HANDLER TESTS
// ═══════════════════════════════════════════════════════════════════════

mod search_tests {
    use super::*;

    #[tokio::test]
    async fn advanced_search_no_criteria_rejected() {
        let h = Harness::new();
        // At least one of entity_name, date range, or importance range is required
        let status = status_of(
            h.app(),
            authed_post(
                "/api/search/advanced",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn advanced_search_by_entity() {
        let h = Harness::new();
        store_memory_typed(
            &h,
            "test-user",
            "Searchable memory about Rust programming",
            "Decision",
            vec!["important"],
        )
        .await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/search/advanced",
                json!({
                    "user_id": "test-user",
                    "entity_name": "Rust",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn advanced_search_by_date_range() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory for date range search test").await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/search/advanced",
                json!({
                    "user_id": "test-user",
                    "start_date": "2020-01-01T00:00:00Z",
                    "end_date": "2030-01-01T00:00:00Z",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn advanced_search_by_importance_range() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory for importance search test").await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/search/advanced",
                json!({
                    "user_id": "test-user",
                    "min_importance": 0.0,
                    "max_importance": 1.0,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// USER MANAGEMENT TESTS
// ═══════════════════════════════════════════════════════════════════════

mod user_tests {
    use super::*;

    #[tokio::test]
    async fn list_users_empty() {
        let h = Harness::new();
        let (status, body) = json_of(h.app(), authed_get("/api/users")).await;
        assert_eq!(status, StatusCode::OK);
        // Response is a bare JSON array of strings, not wrapped in {"users": ...}
        assert!(body.is_array());
    }

    #[tokio::test]
    async fn list_users_after_remember() {
        let h = Harness::new();
        store_memory(&h, "user-alpha", "Memory from alpha user account").await;
        store_memory(&h, "user-beta", "Memory from beta user account").await;

        let (status, body) = json_of(h.app(), authed_get("/api/users")).await;
        assert_eq!(status, StatusCode::OK);
        let users = body.as_array().unwrap();
        assert!(users.len() >= 2, "Should list at least 2 users");
    }

    #[tokio::test]
    async fn user_stats() {
        let h = Harness::new();
        store_memory(&h, "stats-user", "Memory for stats").await;

        let (status, body) = json_of(h.app(), authed_get("/api/users/stats-user/stats")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_object());
    }

    #[tokio::test]
    async fn get_stats_query() {
        let h = Harness::new();
        store_memory(&h, "query-stats-user", "Memory for query stats").await;

        let (status, _) = json_of(h.app(), authed_get("/api/stats?user_id=query-stats-user")).await;
        assert_eq!(status, StatusCode::OK);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// COMPRESSION HANDLER TESTS
// ═══════════════════════════════════════════════════════════════════════

mod compression_tests {
    use super::*;

    #[tokio::test]
    async fn storage_stats() {
        let h = Harness::new();
        let (status, _) =
            json_of(h.app(), authed_get("/api/storage/stats?user_id=test-user")).await;
        assert_eq!(status, StatusCode::OK);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BACKUP HANDLER TESTS
// ═══════════════════════════════════════════════════════════════════════

mod backup_tests {
    use super::*;

    #[tokio::test]
    async fn list_backups_empty() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/backup/list",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["backups"].is_array());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CONTENT-DEDUP ENRICHMENT MERGE
//
// `remember` dedups on a hash of the content alone; a duplicate's extra
// enrichment is merged into the stored memory. handler_tests.rs covers the
// entity union reaching the stored record and its graph episode; this covers
// the identity a merge must leave alone.
// ═══════════════════════════════════════════════════════════════════════

mod dedup_enrichment_tests {
    use super::*;
    use veld::memory::{Memory, MemoryId};

    const DUP_CONTENT: &str =
        "The mooring line parted during the night watch and the tender drifted free.";

    /// Post a memory and return its id. Explicit `tags` are what the handler
    /// folds into the entity set, so they are the controllable half of "a
    /// better extractor found more this time".
    async fn remember_with(h: &Harness, body: Value) -> String {
        let (status, resp) = json_of(h.app(), authed_post("/api/remember", body)).await;
        assert_eq!(status, StatusCode::OK, "remember failed: {resp}");
        resp["id"].as_str().expect("id in response").to_string()
    }

    fn stored_memory(h: &Harness, user: &str, id: &str) -> Memory {
        let uuid = veld::uuid::Uuid::parse_str(id).expect("valid uuid");
        let system = h.mgr.get_user_memory(user).expect("user memory");
        let guard = system.read();
        guard.get_memory(&MemoryId(uuid)).expect("memory stored")
    }

    #[tokio::test]
    async fn dedup_merge_keeps_identity_and_unions_the_entity_set() {
        let h = Harness::new();
        let user = "merge-identity";

        let first = remember_with(
            &h,
            json!({
                "user_id": user,
                "content": DUP_CONTENT,
                "tags": ["mooring"],
            }),
        )
        .await;
        wait_for_background_work(&h).await;

        let before = stored_memory(&h, user, &first);
        let created_at_before = before.created_at;
        assert!(
            before.experience.entities.iter().any(|e| e == "mooring"),
            "precondition: the first write must carry its own tag, got {:?}",
            before.experience.entities
        );
        assert!(
            !before.experience.entities.iter().any(|e| e == "kestrel"),
            "precondition: the first write must NOT already know the second \
             extractor's entity, or the merge below proves nothing"
        );

        // Same bytes, richer enrichment — the re-ingest-with-a-better-extractor case.
        let second = remember_with(
            &h,
            json!({
                "user_id": user,
                "content": DUP_CONTENT,
                "tags": ["mooring", "kestrel"],
            }),
        )
        .await;

        assert_eq!(
            first, second,
            "a merge must return the EXISTING id — enrichment is not a back door \
             to minting a new memory"
        );

        let after = stored_memory(&h, user, &first);
        assert_eq!(
            after.created_at, created_at_before,
            "created_at is identity: downstream systems (watermarks, decay, \
             temporal ranking) key on it and a merge must never move it"
        );
        assert_eq!(after.version, before.version, "a merge is not a revision");
        assert!(
            after.experience.entities.iter().any(|e| e == "kestrel"),
            "the second copy's entity was dropped on the floor — got {:?}",
            after.experience.entities
        );
        assert!(
            after.experience.entities.iter().any(|e| e == "mooring"),
            "the merge must be a UNION; the first copy's entities cannot be lost"
        );

        let (_, list) = json_of(h.app(), authed_get(&format!("/api/list/{user}"))).await;
        assert_eq!(
            list["memories"].as_array().unwrap().len(),
            1,
            "dedup must still dedup"
        );
    }
}
