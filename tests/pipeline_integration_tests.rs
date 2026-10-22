//! Pipeline integration tests: remember → recall round trips through the HTTP router.
//!
//! Ported from upstream shodh-memory `tests/pipeline_integration_tests.rs` (#230, with the
//! proactive-context cases of #358 and the recall tier contract of #450). Covers content
//! fidelity, user isolation, dedup, recall modes, batch and upsert writes, graph and lineage
//! routes, proactive context, and forgetting. Upstream's `importance_override_is_respected` is
//! not ported: veld's remember takes no `importance` (upstream #176 is pending in
//! docs/upstream/LEDGER.tsv).
//!
//! Run with: `cargo test --test pipeline_integration_tests`

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
// Test infrastructure (the harness of handler_tests.rs)
// ═══════════════════════════════════════════════════════════════════════

const TEST_KEY: &str = "pipeline-integration-test-key";
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

// ── response helpers ──

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

/// Store a memory and return its ID.
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

/// Store a memory with a type and tags.
async fn store_memory_full(
    h: &Harness,
    user_id: &str,
    content: &str,
    memory_type: &str,
    tags: &[&str],
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
    assert_eq!(status, StatusCode::OK, "store_memory_full failed: {body}");
    body["id"]
        .as_str()
        .expect("missing id in response")
        .to_string()
}

/// Store a memory with an explicit creation timestamp.
async fn store_memory_full_at(
    h: &Harness,
    user_id: &str,
    content: &str,
    memory_type: &str,
    tags: &[&str],
    created_at: chrono::DateTime<chrono::Utc>,
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
                "created_at": created_at.to_rfc3339(),
                "importance": 0.9,
            }),
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "store_memory_full_at failed: {body}"
    );
    body["id"]
        .as_str()
        .expect("missing id in response")
        .to_string()
}

/// Recall memories for a user.
async fn recall(h: &Harness, user_id: &str, query: &str) -> (StatusCode, Value) {
    json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({
                "user_id": user_id,
                "query": query,
            }),
        ),
    )
    .await
}

/// Recall with a specific mode.
async fn recall_mode(h: &Harness, user_id: &str, query: &str, mode: &str) -> (StatusCode, Value) {
    json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({
                "user_id": user_id,
                "query": query,
                "mode": mode,
            }),
        ),
    )
    .await
}

/// Wait for the work a write leaves running: the deferred embedding and vector indexing, and the
/// graph, lineage and conflict passes. Upstream slept a fixed 150 or 800 ms instead.
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

fn ids_of(memories: &[Value]) -> Vec<&str> {
    memories.iter().filter_map(|m| m["id"].as_str()).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// REMEMBER → RECALL ROUNDTRIP TESTS
// ═══════════════════════════════════════════════════════════════════════

mod roundtrip_tests {
    use super::*;

    #[tokio::test]
    async fn recall_returns_stored_content() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The Ebbinghaus forgetting curve demonstrates exponential memory decay",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = recall(&h, "test-user", "Ebbinghaus forgetting curve").await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().expect("memories array");
        assert!(!memories.is_empty(), "Should find the stored memory");
        let content = memories[0]["experience"]["content"]
            .as_str()
            .expect("content field");
        assert!(
            content.contains("Ebbinghaus"),
            "Recalled content should match stored: {content}"
        );
    }

    #[tokio::test]
    async fn recall_returns_correct_id() {
        let h = Harness::new();
        let stored_id = store_memory(
            &h,
            "test-user",
            "Hebbian learning strengthens synaptic connections between co-active neurons",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "Hebbian synaptic connections").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let recalled_id = memories[0]["id"].as_str().expect("id field");
        assert_eq!(recalled_id, stored_id, "Recalled ID should match stored ID");
    }

    #[tokio::test]
    async fn recall_response_has_expected_fields() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "RocksDB uses log-structured merge trees for efficient writes",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "RocksDB merge trees").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());

        let m = &memories[0];
        assert!(m["id"].is_string(), "should have id");
        assert!(
            m["experience"]["content"].is_string(),
            "should have content"
        );
        assert!(m["importance"].is_number(), "should have importance");
        assert!(m["created_at"].is_string(), "should have created_at");
        assert!(m["score"].is_number(), "should have score");
        assert!(m["tier"].is_string(), "should have tier");
        // experience.tags is an array (may include auto-extracted entities)
        assert!(m["experience"]["tags"].is_array(), "should have tags array");
    }

    /// veld's recall score is rank position (70%) blended with access-weighted salience (30%),
    /// both in [0, 1] (`recall_memory_view` in src/handlers/recall.rs), where upstream scales the
    /// top raw score to 0.95. The bound is veld's scale.
    #[tokio::test]
    async fn recall_score_is_normalized() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Vector similarity search uses cosine distance for matching",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "vector similarity cosine").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());

        for m in memories {
            let score = m["score"].as_f64().unwrap();
            assert!(
                (0.0..=1.0).contains(&score),
                "Score should be within [0, 1], got {score}"
            );
        }
    }

    #[tokio::test]
    async fn recall_single_memory_gets_max_score() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Spreading activation propagates through semantic networks",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "spreading activation semantic networks").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "Should find the stored memory");
        let score = memories[0]["score"].as_f64().unwrap();
        assert!(
            score > 0.5,
            "Single memory should get high normalized score, got {score}"
        );
    }

    #[tokio::test]
    async fn recall_count_matches_array_length() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory alpha about graph traversal").await;
        store_memory(&h, "test-user", "Memory beta about graph algorithms").await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "graph algorithms traversal").await;
        let memories = body["memories"].as_array().unwrap();
        let count = body["count"].as_u64().unwrap() as usize;
        assert_eq!(
            count,
            memories.len(),
            "count field should match array length"
        );
    }

    #[tokio::test]
    async fn recall_respects_limit() {
        let h = Harness::new();
        for i in 0..5 {
            store_memory(
                &h,
                "test-user",
                &format!("Neural network layer {i} uses backpropagation for training"),
            )
            .await;
        }
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "neural network backpropagation",
                    "limit": 2,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            memories.len() <= 2,
            "Should respect limit=2, got {}",
            memories.len()
        );
    }

    #[tokio::test]
    async fn recall_created_at_is_rfc3339() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Temporal indexing uses RFC3339 timestamps for ordering",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "temporal indexing timestamps").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let created_at = memories[0]["created_at"].as_str().unwrap();
        assert!(
            chrono::DateTime::parse_from_rfc3339(created_at).is_ok(),
            "created_at should be RFC3339: {created_at}"
        );
    }

    /// Every recalled memory states its consolidation tier as the Debug rendering of
    /// `MemoryTier`. `Archive` is not among the live tiers: default recall excludes it.
    #[tokio::test]
    async fn recall_memories_carry_consolidation_tier() {
        const LIVE_TIERS: [&str; 3] = ["Working", "Session", "LongTerm"];

        let h = Harness::new();
        for i in 0..3 {
            store_memory(
                &h,
                "test-user",
                &format!("Consolidation ladder note {i} about tier promotion thresholds"),
            )
            .await;
        }
        wait_for_background_work(&h).await;

        let (status, body) = recall(&h, "test-user", "consolidation ladder tier promotion").await;
        assert_eq!(status, StatusCode::OK, "recall failed: {body}");

        let memories = body["memories"].as_array().expect("memories array");
        assert!(
            !memories.is_empty(),
            "seeded memories should be recallable: {body}"
        );

        for m in memories {
            let tier = m["tier"]
                .as_str()
                .unwrap_or_else(|| panic!("recalled memory has no string `tier` field: {m}"));
            assert!(
                LIVE_TIERS.contains(&tier),
                "tier {tier:?} is not one of the three live tiers {LIVE_TIERS:?}"
            );
        }

        // Fresh memories have not aged into a promotion, so the start of the ladder shows the
        // field is read from the record rather than defaulted downstream.
        assert!(
            memories
                .iter()
                .all(|m| m["tier"].as_str() == Some("Working")),
            "newly written memories should still be Working: {body}"
        );
    }

    #[tokio::test]
    async fn recall_memory_type_preserved() {
        let h = Harness::new();
        store_memory_full(
            &h,
            "test-user",
            "Decision to use power-law decay for long-term retention",
            "Decision",
            &[],
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "power-law decay decision").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let mem_type = memories[0]["experience"]["memory_type"]
            .as_str()
            .unwrap_or("");
        assert_eq!(mem_type, "Decision", "Memory type should be preserved");
    }

    #[tokio::test]
    async fn recall_user_tags_included_in_response() {
        let h = Harness::new();
        store_memory_full(
            &h,
            "test-user",
            "Rust ownership model prevents data races at compile time",
            "Learning",
            &["rust", "ownership"],
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "Rust ownership data races").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let tags = memories[0]["experience"]["tags"]
            .as_array()
            .expect("tags array");
        let tag_strings: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
        assert!(
            tag_strings.iter().any(|t| t.eq_ignore_ascii_case("rust")),
            "Should contain user tag 'rust', got: {tag_strings:?}"
        );
    }

    #[tokio::test]
    async fn recall_importance_is_valid() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Importance scoring uses salience detection with keyword analysis",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "importance scoring salience").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let importance = memories[0]["importance"].as_f64().unwrap();
        assert!(
            (0.0..=1.0).contains(&importance),
            "Importance should be 0.0-1.0, got {importance}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// UNICODE ROUNDTRIP TESTS
// ═══════════════════════════════════════════════════════════════════════

mod unicode_tests {
    use super::*;

    async fn assert_listed_verbatim(content: &str) {
        let h = Harness::new();
        store_memory(&h, "test-user", content).await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1, "{body}");
        // A list item carries `content` at the top level, not under `experience`.
        assert_eq!(
            memories[0]["content"].as_str(),
            Some(content),
            "content should survive the roundtrip"
        );
    }

    #[tokio::test]
    async fn unicode_content_roundtrip() {
        assert_listed_verbatim("日本語テスト: Hebbian学習は神経科学の基本原理です").await;
    }

    #[tokio::test]
    async fn mixed_script_content_roundtrip() {
        assert_listed_verbatim("English العربية 中文 हिन्दी mixing scripts in memory content").await;
    }

    #[tokio::test]
    async fn emoji_content_roundtrip() {
        assert_listed_verbatim("Memory with emojis 🧠💡🔬 testing unicode edge cases").await;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// USER ISOLATION TESTS
// ═══════════════════════════════════════════════════════════════════════

mod isolation_tests {
    use super::*;

    #[tokio::test]
    async fn users_cannot_see_each_others_memories() {
        let h = Harness::new();
        store_memory(
            &h,
            "alice",
            "Alice's secret project about quantum computing breakthroughs",
        )
        .await;
        store_memory(
            &h,
            "bob",
            "Bob's private notes about compiler optimization techniques",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "alice", "quantum computing").await;
        let memories = body["memories"].as_array().unwrap();
        for m in memories {
            let content = m["experience"]["content"].as_str().unwrap_or("");
            assert!(
                !content.contains("Bob"),
                "Alice should not see Bob's memories: {content}"
            );
        }

        let (_, body) = recall(&h, "bob", "compiler optimization").await;
        let memories = body["memories"].as_array().unwrap();
        for m in memories {
            let content = m["experience"]["content"].as_str().unwrap_or("");
            assert!(
                !content.contains("Alice"),
                "Bob should not see Alice's memories: {content}"
            );
        }
    }

    #[tokio::test]
    async fn user_list_is_isolated() {
        let h = Harness::new();
        store_memory(
            &h,
            "user-one",
            "First user memory about distributed systems",
        )
        .await;
        store_memory(&h, "user-two", "Second user memory about machine learning").await;

        let (_, body) = json_of(h.app(), authed_get("/api/list/user-one")).await;
        let memories = body["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1, "user-one should have exactly 1 memory");
        assert!(memories[0]["content"]
            .as_str()
            .unwrap()
            .contains("distributed systems"));

        let (_, body) = json_of(h.app(), authed_get("/api/list/user-two")).await;
        let memories = body["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1, "user-two should have exactly 1 memory");
        assert!(memories[0]["content"]
            .as_str()
            .unwrap()
            .contains("machine learning"));
    }

    #[tokio::test]
    async fn recall_empty_for_wrong_user() {
        let h = Harness::new();
        store_memory(
            &h,
            "data-owner",
            "Confidential memory about proprietary algorithm design",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = recall(&h, "intruder", "proprietary algorithm").await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.is_empty(), "Wrong user should see no memories");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CONTENT DEDUP TESTS
// ═══════════════════════════════════════════════════════════════════════

mod dedup_tests {
    use super::*;

    #[tokio::test]
    async fn identical_content_returns_same_id() {
        let h = Harness::new();
        let content = "Exact duplicate content for deduplication testing in memory system";
        let id1 = store_memory(&h, "test-user", content).await;
        let id2 = store_memory(&h, "test-user", content).await;
        assert_eq!(id1, id2, "Duplicate content should return same ID");
    }

    #[tokio::test]
    async fn similar_but_different_content_gets_different_ids() {
        let h = Harness::new();
        let id1 = store_memory(&h, "test-user", "The cat sat on the mat near the window").await;
        let id2 = store_memory(&h, "test-user", "The cat sat on the mat near the door").await;
        assert_ne!(id1, id2, "Different content should get different IDs");
    }

    #[tokio::test]
    async fn dedup_does_not_create_duplicate_in_list() {
        let h = Harness::new();
        let content = "Unique dedup verification content for list counting test";
        store_memory(&h, "test-user", content).await;
        store_memory(&h, "test-user", content).await;
        store_memory(&h, "test-user", content).await;

        let (_, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        let memories = body["memories"].as_array().unwrap();
        assert_eq!(
            memories.len(),
            1,
            "Dedup should prevent duplicates in storage"
        );
    }

    #[tokio::test]
    async fn dedup_is_per_user() {
        let h = Harness::new();
        let content = "Shared content stored by different users for isolation test";
        let id1 = store_memory(&h, "user-a", content).await;
        let id2 = store_memory(&h, "user-b", content).await;
        assert_ne!(
            id1, id2,
            "Same content for different users should get different IDs"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MULTI-MODE RECALL TESTS
// ═══════════════════════════════════════════════════════════════════════

mod mode_tests {
    use super::*;

    #[tokio::test]
    async fn semantic_mode_returns_semantically_similar() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The transformer architecture revolutionized natural language processing",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = recall_mode(
            &h,
            "test-user",
            "deep learning NLP transformers",
            "semantic",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            !memories.is_empty(),
            "Semantic mode should find semantically related memory"
        );
    }

    #[tokio::test]
    async fn similarity_mode_alias_works() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Gradient descent optimizes neural network parameters iteratively",
        )
        .await;
        wait_for_background_work(&h).await;

        // "similarity" is an alias for "semantic"
        let (status, body) = recall_mode(
            &h,
            "test-user",
            "gradient descent optimization",
            "similarity",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(!body["memories"].as_array().unwrap().is_empty(), "{body}");
    }

    #[tokio::test]
    async fn temporal_mode_returns_results() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Temporal memory created for mode testing in retrieval pipeline",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = recall_mode(&h, "test-user", "temporal retrieval", "temporal").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn hybrid_mode_is_default() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Hybrid retrieval fuses vector and keyword search signals together",
        )
        .await;
        wait_for_background_work(&h).await;

        // No mode specified → defaults to hybrid
        let (status1, body1) = recall(&h, "test-user", "hybrid retrieval fusion").await;
        // Explicit hybrid
        let (status2, body2) =
            recall_mode(&h, "test-user", "hybrid retrieval fusion", "hybrid").await;

        assert_eq!(status1, StatusCode::OK);
        assert_eq!(status2, StatusCode::OK);
        assert!(!body1["memories"].as_array().unwrap().is_empty(), "{body1}");
        assert!(!body2["memories"].as_array().unwrap().is_empty(), "{body2}");
    }

    #[tokio::test]
    async fn associative_mode_returns_results() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Knowledge graph nodes connected via associative activation spreading",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) =
            recall_mode(&h, "test-user", "knowledge graph activation", "associative").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn causal_mode_returns_results() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Causal inference determines cause and effect relationships in data",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = recall_mode(&h, "test-user", "causal inference", "causal").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn unknown_mode_falls_back_to_hybrid() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Fallback mode testing with unknown retrieval strategy selection",
        )
        .await;
        wait_for_background_work(&h).await;

        // Unknown mode string → falls back to Hybrid (not an error)
        let (status, body) =
            recall_mode(&h, "test-user", "fallback mode testing", "nonexistent_mode").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BATCH REMEMBER → RECALL TESTS
// ═══════════════════════════════════════════════════════════════════════

mod batch_tests {
    use super::*;

    #[tokio::test]
    async fn batch_memories_are_recallable() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "Batch item one about recurrent neural networks"},
                        {"content": "Batch item two about convolutional neural networks"},
                        {"content": "Batch item three about generative adversarial networks"},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["created"], 3);
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "neural networks").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "Should recall at least 1 batch item");
    }

    #[tokio::test]
    async fn batch_memory_ids_are_valid() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "Batch ID test memory alpha about reinforcement learning"},
                        {"content": "Batch ID test memory beta about multi-agent systems"},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let ids = body["memory_ids"].as_array().unwrap();
        assert_eq!(ids.len(), 2);

        for id in ids {
            let id_str = id.as_str().unwrap();
            let (status, _) = json_of(
                h.app(),
                authed_get(&format!("/api/memory/{id_str}?user_id=test-user")),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::OK,
                "Batch memory {id_str} should be retrievable"
            );
        }
    }

    #[tokio::test]
    async fn batch_partial_failure_still_stores_valid() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "Valid batch memory about knowledge representation"},
                        {"content": ""},
                        {"content": "Another valid batch memory about ontology design"},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let created = body["created"].as_u64().unwrap();
        let failed = body["failed"].as_u64().unwrap();
        assert!(created >= 2, "At least 2 valid items should be stored");
        assert!(failed >= 1, "Empty content item should fail");
        assert!(body["errors"].is_array());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// UPSERT → RECALL TESTS
// ═══════════════════════════════════════════════════════════════════════

mod upsert_tests {
    use super::*;

    #[tokio::test]
    async fn upsert_creates_recallable_memory() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Upserted memory about attention mechanism in transformers",
                    "external_id": "ext-attn-001",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["was_update"], json!(false));
        assert_eq!(body["version"], 1);
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "attention mechanism transformers").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "Upserted memory should be recallable");
    }

    #[tokio::test]
    async fn upsert_update_changes_content() {
        let h = Harness::new();
        let (_, body) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Initial version of the documentation about SPANN index",
                    "external_id": "doc-spann",
                }),
            ),
        )
        .await;
        let id = body["id"].as_str().unwrap().to_string();
        assert_eq!(body["was_update"], json!(false));

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Updated version of the SPANN index documentation with PQ compression details",
                    "external_id": "doc-spann",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["was_update"], json!(true));
        assert_eq!(body["version"], 2);
        assert_eq!(body["id"].as_str().unwrap(), id);

        let (status, body) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let content = body["experience"]["content"].as_str().unwrap();
        assert!(
            content.contains("PQ compression"),
            "Content should be updated: {content}"
        );
    }

    #[tokio::test]
    async fn upsert_different_external_ids_create_separate_memories() {
        let h = Harness::new();
        let (_, body1) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "First document about Vamana graph construction algorithm",
                    "external_id": "doc-vamana",
                }),
            ),
        )
        .await;
        let (_, body2) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Second document about SPANN disk-based index structure",
                    "external_id": "doc-spann-2",
                }),
            ),
        )
        .await;
        assert_ne!(
            body1["id"].as_str().unwrap(),
            body2["id"].as_str().unwrap(),
            "Different external_ids should create different memories"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TAG & DATE FILTERING TESTS
// ═══════════════════════════════════════════════════════════════════════

mod filter_tests {
    use super::*;

    #[tokio::test]
    async fn recall_by_tags_returns_matching() {
        let h = Harness::new();
        store_memory_full(
            &h,
            "test-user",
            "Important architecture decision about database schema",
            "Decision",
            &["architecture", "database"],
        )
        .await;
        store_memory_full(
            &h,
            "test-user",
            "Routine observation about build times being slow",
            "Observation",
            &["build", "performance"],
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall/tags",
                json!({
                    "user_id": "test-user",
                    "tags": ["architecture"],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            !memories.is_empty(),
            "Should find the architecture-tagged memory: {body}"
        );
        for m in memories {
            let tags: Vec<&str> = m["experience"]["tags"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|t| t.as_str())
                .collect();
            assert!(
                tags.iter().any(|t| t.eq_ignore_ascii_case("architecture")),
                "Should contain the 'architecture' tag, got: {tags:?}"
            );
        }
    }

    #[tokio::test]
    async fn recall_by_date_range_returns_recent() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Memory stored today for date range filter testing",
        )
        .await;

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
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "Should find memory within date range");
    }

    #[tokio::test]
    async fn recall_by_narrow_date_range_excludes() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Memory that should not match narrow past date range",
        )
        .await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall/date",
                json!({
                    "user_id": "test-user",
                    "start": "2000-01-01T00:00:00Z",
                    "end": "2001-01-01T00:00:00Z",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            memories.is_empty(),
            "Should not find memory outside date range"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MULTIPLE MEMORIES RANKING TESTS
// ═══════════════════════════════════════════════════════════════════════

mod ranking_tests {
    use super::*;

    #[tokio::test]
    async fn scores_are_ordered_descending() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Rust programming language memory safety guarantees",
        )
        .await;
        store_memory(
            &h,
            "test-user",
            "Python dynamic typing flexible scripting capabilities",
        )
        .await;
        store_memory(
            &h,
            "test-user",
            "Rust borrow checker prevents data races at compile time",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "Rust memory safety borrow checker").await;
        let scores: Vec<f64> = body["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["score"].as_f64().unwrap_or(0.0))
            .collect();
        assert!(
            scores.windows(2).all(|pair| pair[0] >= pair[1]),
            "Scores should be descending: {scores:?}"
        );
    }

    #[tokio::test]
    async fn relevant_memory_scores_higher_than_irrelevant() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The Vamana algorithm builds a navigable small-world graph for ANN search",
        )
        .await;
        store_memory(
            &h,
            "test-user",
            "My grocery list includes milk, eggs, and bread for the weekend",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "Vamana graph ANN search").await;
        let memories = body["memories"].as_array().unwrap();
        if memories.len() >= 2 {
            let top_content = memories[0]["experience"]["content"].as_str().unwrap_or("");
            assert!(
                top_content.contains("Vamana"),
                "Top result should be the relevant memory, got: {top_content}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GRAPH INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════

mod graph_integration_tests {
    use super::*;

    #[tokio::test]
    async fn remember_populates_graph() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Google released TensorFlow as an open source machine learning framework",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(h.app(), authed_get("/api/graph/test-user/stats")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_object(), "{body}");
    }

    #[tokio::test]
    async fn graph_data_available_after_remember() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Microsoft Azure provides cloud computing services worldwide",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, _) = json_of(h.app(), authed_get("/api/graph/data/test-user")).await;
        assert_eq!(status, StatusCode::OK);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LINEAGE INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════

mod lineage_integration_tests {
    use super::*;

    async fn link(h: &Harness, from: &str, to: &str, relation: &str) -> StatusCode {
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/link",
                json!({
                    "user_id": "test-user",
                    "from_memory_id": from,
                    "to_memory_id": to,
                    "relation": relation,
                }),
            ),
        )
        .await;
        status
    }

    #[tokio::test]
    async fn manual_lineage_edge_persists() {
        let h = Harness::new();
        let id1 = store_memory(
            &h,
            "test-user",
            "Identified performance bottleneck in the query pipeline",
        )
        .await;
        let id2 = store_memory(
            &h,
            "test-user",
            "Resolved the performance bottleneck by adding index caching",
        )
        .await;
        wait_for_background_work(&h).await;

        assert_eq!(link(&h, &id1, &id2, "ResolvedBy").await, StatusCode::OK);

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
        let edges = body["edges"].as_array().unwrap();
        assert!(!edges.is_empty(), "Lineage edge should persist");
    }

    /// Upstream asserted only that the trace body parsed. A forward trace from the chain's head
    /// is what reaches its tail, so that is what this checks.
    #[tokio::test]
    async fn lineage_trace_follows_chain() {
        let h = Harness::new();
        let id1 = store_memory(
            &h,
            "test-user",
            "Discovered bug in the serialization layer causing data loss",
        )
        .await;
        let id2 = store_memory(
            &h,
            "test-user",
            "Fixed serialization bug by adding format version tags",
        )
        .await;
        let id3 = store_memory(
            &h,
            "test-user",
            "Deployed serialization fix to production environment successfully",
        )
        .await;
        wait_for_background_work(&h).await;

        assert_eq!(link(&h, &id1, &id2, "Caused").await, StatusCode::OK);
        assert_eq!(link(&h, &id2, &id3, "Caused").await, StatusCode::OK);

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/trace",
                json!({
                    "user_id": "test-user",
                    "memory_id": id1,
                    "direction": "forward",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let path: Vec<&str> = body["path"]
            .as_array()
            .expect("trace path")
            .iter()
            .filter_map(|id| id.as_str())
            .collect();
        assert!(
            path.contains(&id2.as_str()) && path.contains(&id3.as_str()),
            "a forward trace from the head should reach both links of the chain: {body}"
        );
    }

    #[tokio::test]
    async fn all_lineage_relation_types_accepted() {
        let h = Harness::new();
        let relations = [
            "Caused",
            "ResolvedBy",
            "InformedBy",
            "SupersededBy",
            "TriggeredBy",
            "BranchedFrom",
            "RelatedTo",
        ];

        for (i, relation) in relations.iter().enumerate() {
            let id1 = store_memory(
                &h,
                "test-user",
                &format!("Source memory number {i} for lineage relation testing"),
            )
            .await;
            let id2 = store_memory(
                &h,
                "test-user",
                &format!("Target memory number {i} for lineage relation testing"),
            )
            .await;

            assert_eq!(
                link(&h, &id1, &id2, relation).await,
                StatusCode::OK,
                "Relation type '{relation}' should be accepted"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PROACTIVE CONTEXT → RECALL INTEGRATION
// ═══════════════════════════════════════════════════════════════════════

mod proactive_integration_tests {
    use super::*;

    #[tokio::test]
    async fn proactive_context_returns_relevant_memories() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The deployment pipeline uses Docker containers for isolation",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "Working on the Docker deployment pipeline configuration",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // Whether the memory surfaces depends on embedding similarity and the threshold; the
        // response shape is what is pinned here.
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn proactive_context_keeps_explicit_memory_that_overlaps_context() {
        let h = Harness::new();
        let content = "Workflow preference: when running automated validation, use the existing local runner path and avoid duplicate remote automation runs. Do not create temporary branches or extra remote jobs unless explicitly requested.";
        let memory_id = store_memory_full(
            &h,
            "test-user",
            content,
            "Decision",
            &["workflow", "validation"],
        )
        .await;
        wait_for_background_work(&h).await;

        let query =
            "run automated validation using the local runner without duplicate remote automation";
        let (status, recall_body) = recall(&h, "test-user", query).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            ids_of(recall_body["memories"].as_array().unwrap()).contains(&memory_id.as_str()),
            "explicit recall should find the stored memory: {recall_body}"
        );

        let (status, proactive_body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": query,
                    "max_results": 10,
                    "semantic_threshold": 0.1,
                    "auto_ingest": false,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            ids_of(proactive_body["memories"].as_array().unwrap()).contains(&memory_id.as_str()),
            "proactive_context should not drop explicit memories as context echoes: {proactive_body}"
        );
    }

    #[tokio::test]
    async fn proactive_context_drops_an_auto_captured_echo_of_the_context() {
        let h = Harness::new();
        let query =
            "run automated validation using the local runner without duplicate remote automation";
        let echo_id =
            store_memory_full(&h, "test-user", query, "Conversation", &["auto-captured"]).await;
        let explicit_id = store_memory_full(
            &h,
            "test-user",
            &format!("Decision: {query}, agreed with the release team"),
            "Decision",
            &["workflow"],
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": query,
                    "max_results": 10,
                    "semantic_threshold": 0.0,
                    "auto_ingest": false,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let surfaced = ids_of(body["memories"].as_array().unwrap());
        assert!(
            surfaced.contains(&explicit_id.as_str()),
            "the explicit memory surfaces: {body}"
        );
        assert!(
            !surfaced.contains(&echo_id.as_str()),
            "the auto-captured echo of the context is dropped: {body}"
        );
    }

    #[tokio::test]
    async fn proactive_context_threshold_uses_normalized_score_scale() {
        let h = Harness::new();
        let content = "Workflow preference: proactive context threshold calibration should surface durable old validation memories when recall ranks them as the best candidate.";
        let created_at = chrono::Utc::now() - chrono::Duration::days(30);
        let memory_id = store_memory_full_at(
            &h,
            "test-user",
            content,
            "Decision",
            &["workflow", "threshold", "validation"],
            created_at,
        )
        .await;
        wait_for_background_work(&h).await;

        let query = "proactive context threshold calibration validation memories";
        for (threshold, what) in [
            (
                0.0,
                "sanity check: zero threshold should surface the stored memory",
            ),
            (
                0.0001,
                "tiny positive threshold should use normalized score scale",
            ),
        ] {
            let (status, body) = json_of(
                h.app(),
                authed_post(
                    "/api/proactive_context",
                    json!({
                        "user_id": "test-user",
                        "context": query,
                        "max_results": 10,
                        "semantic_threshold": threshold,
                        "auto_ingest": false,
                    }),
                ),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert!(
                ids_of(body["memories"].as_array().unwrap()).contains(&memory_id.as_str()),
                "{what}: {body}"
            );
        }
    }

    #[tokio::test]
    async fn proactive_context_respects_max_results() {
        let h = Harness::new();
        for i in 0..5 {
            store_memory(
                &h,
                "test-user",
                &format!("Kubernetes pod {i} runs microservice for API gateway"),
            )
            .await;
        }
        wait_for_background_work(&h).await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "Kubernetes microservices",
                    "max_results": 2,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            memories.len() <= 2,
            "Should respect max_results=2, got {}",
            memories.len()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DELETE → RECALL VERIFICATION
// ═══════════════════════════════════════════════════════════════════════

mod delete_tests {
    use super::*;

    #[tokio::test]
    async fn deleted_memory_not_recalled() {
        let h = Harness::new();
        let content = "Temporary memory that will be deleted from the system";
        let id = store_memory(&h, "test-user", content).await;
        wait_for_background_work(&h).await;

        let (status, _) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

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

        let (status, _) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (status, body) = recall(&h, "test-user", content).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            !ids_of(body["memories"].as_array().unwrap()).contains(&id.as_str()),
            "a forgotten memory must not be recalled: {body}"
        );
    }

    #[tokio::test]
    async fn forget_by_tags_removes_from_recall() {
        let h = Harness::new();
        let ephemeral = store_memory_full(
            &h,
            "test-user",
            "Ephemeral memory tagged for deletion testing purposes",
            "Observation",
            &["ephemeral", "delete-me"],
        )
        .await;
        let permanent = store_memory(
            &h,
            "test-user",
            "Permanent memory that should survive tag-based deletion",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/tags",
                json!({
                    "user_id": "test-user",
                    "tags": ["ephemeral"],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // A list item carries `content` and `id` at the top level; upstream read
        // `experience.content` here, which a list item does not have.
        let (_, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        let listed = body["memories"].as_array().unwrap();
        assert_eq!(ids_of(listed), vec![permanent.as_str()], "{body}");

        let (_, body) = recall(&h, "test-user", "ephemeral memory tagged for deletion").await;
        assert!(
            !ids_of(body["memories"].as_array().unwrap()).contains(&ephemeral.as_str()),
            "a memory forgotten by tag must not be recalled: {body}"
        );
    }

    #[tokio::test]
    async fn clear_all_removes_everything() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory one to be cleared completely").await;
        store_memory(&h, "test-user", "Memory two to be cleared completely").await;
        store_memory(&h, "test-user", "Memory three to be cleared completely").await;
        // The clear is checked against settled writes: a deferred embedding still in flight
        // is a separate race from what this pins.
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
        assert!(
            memories.is_empty(),
            "All memories should be cleared: {body}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RECALL RESPONSE SHAPE EDGE CASES
// ═══════════════════════════════════════════════════════════════════════

mod response_shape_tests {
    use super::*;

    #[tokio::test]
    async fn empty_recall_has_correct_shape() {
        let h = Harness::new();
        let (status, body) = recall(&h, "test-user", "nothing stored yet").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
        assert_eq!(body["count"], 0);
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.is_empty());
    }

    #[tokio::test]
    async fn recall_optional_fields_absent_when_empty() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Memory for testing optional response fields presence",
        )
        .await;
        wait_for_background_work(&h).await;

        let (status, body) = recall(&h, "test-user", "optional response fields").await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body["retrieval_stats"].is_null(),
            "retrieval_stats should be null"
        );
        // A fresh store has no facts, todos, reminders or lineage: each list is skipped when
        // empty, so it is absent or, at most, an empty array.
        for field in ["facts", "todos", "triggered_reminders", "lineage"] {
            assert!(
                body.get(field)
                    .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty)),
                "{field} should be absent or empty: {body}"
            );
        }
    }

    #[tokio::test]
    async fn recall_tier_field_is_valid_string() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Memory for tier field validation in recall response",
        )
        .await;
        wait_for_background_work(&h).await;

        let (_, body) = recall(&h, "test-user", "tier field validation").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "{body}");
        let tier = memories[0]["tier"].as_str().unwrap();
        let valid_tiers = ["Working", "Session", "LongTerm"];
        assert!(
            valid_tiers.contains(&tier),
            "Tier should be one of {valid_tiers:?}, got '{tier}'"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EMOTIONAL CONTEXT TESTS
// ═══════════════════════════════════════════════════════════════════════

mod importance_tests {
    use super::*;

    #[tokio::test]
    async fn emotional_context_stored_and_retrievable() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Frustrating debugging session with race condition in async code",
                    "emotional_valence": -0.7,
                    "emotional_arousal": 0.8,
                    "emotion": "frustration",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let id = body["id"].as_str().unwrap();

        let (status, body) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["experience"]["content"]
            .as_str()
            .unwrap()
            .contains("Frustrating"));
    }
}
