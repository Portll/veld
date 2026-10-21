use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{json, Value};

use crate::handlers::admin::{ADMIN_API_KEY_ENV, ADMIN_ENV_LOCK};
use crate::handlers::test_helpers::{send, TestHarness, TEST_API_KEY};
use crate::storage::{PrimaryMemoryStore, RocksDbPrimaryMemoryStore};

const ADMIN_KEY: &str = "test-admin-secret-for-migration";

fn request(uri: &str, body: Value, api_key: bool, admin_key: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json");
    if api_key {
        builder = builder.header("x-api-key", TEST_API_KEY);
    }
    if let Some(k) = admin_key {
        builder = builder.header("x-admin-api-key", k);
    }
    builder.body(Body::from(body.to_string())).unwrap()
}

fn seeded_harness() -> TestHarness {
    let harness = TestHarness::new();
    let store =
        RocksDbPrimaryMemoryStore::open(&harness.manager.base_path.join("una"), None).unwrap();
    let now = chrono::Utc::now();
    let m = crate::memory::types::Memory::from_legacy(
        crate::memory::types::MemoryId(uuid::Uuid::new_v4()),
        crate::memory::types::Experience {
            content: "over http".into(),
            ..Default::default()
        },
        0.5,
        0,
        now,
        now,
        false,
        crate::memory::types::MemoryTier::LongTerm,
        Vec::new(),
        1.0,
        None,
        None,
        None,
        None,
        0.0,
        None,
        None,
        1,
        Vec::new(),
        Vec::new(),
    );
    store.put_memory(&m).unwrap();
    store.flush().unwrap();
    harness
}

const ROUTES: [&str; 3] = [
    "/api/admin/storage/export",
    "/api/admin/storage/import",
    "/api/admin/storage/migrate",
];

#[tokio::test]
async fn every_route_is_503_when_no_admin_key_is_configured() {
    let _env = ADMIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var(ADMIN_API_KEY_ENV);
    let harness = TestHarness::new();
    for uri in ROUTES {
        let (status, body) = send(
            harness.router(),
            request(uri, json!({}), true, Some("anything")),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}: {body}");
        assert_eq!(body["error"], "admin_key_not_configured");
    }
}

#[tokio::test]
async fn a_regular_api_key_alone_is_refused_on_every_route() {
    let _env = ADMIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(ADMIN_API_KEY_ENV, ADMIN_KEY);
    let harness = seeded_harness();
    let body = json!({"from": "rocksdb", "to": "redb", "user": "una", "dry_run": true});
    for uri in ROUTES {
        let (status, _) = send(harness.router(), request(uri, body.clone(), true, None)).await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "{uri} without the admin key"
        );
        let (status, _) = send(
            harness.router(),
            request(uri, body.clone(), true, Some("wrong")),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "{uri} with a wrong admin key"
        );
    }
    std::env::remove_var(ADMIN_API_KEY_ENV);
}

// Migrates into redb, which a rocksdb-only build does not have.
#[cfg(feature = "storage-redb")]
#[tokio::test]
async fn a_dry_run_over_http_reports_counts_on_both_backends() {
    let _env = ADMIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(ADMIN_API_KEY_ENV, ADMIN_KEY);
    let harness = seeded_harness();
    let (status, body) = send(
        harness.router(),
        request(
            "/api/admin/storage/migrate",
            json!({"from": "rocksdb", "to": "redb", "user": "una", "dry_run": true}),
            true,
            Some(ADMIN_KEY),
        ),
    )
    .await;
    std::env::remove_var(ADMIN_API_KEY_ENV);
    assert_eq!(status, StatusCode::OK, "{body}");
    let unit = &body["report"]["units"][0];
    assert_eq!(unit["source"]["records"], 1, "{body}");
    assert_eq!(unit["target"]["records"], 0, "{body}");
    assert!(!harness.manager.base_path.join("una/storage.redb").exists());
}

#[tokio::test]
async fn writing_into_the_backend_being_served_is_refused() {
    let _env = ADMIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(ADMIN_API_KEY_ENV, ADMIN_KEY);
    let harness = seeded_harness();
    let serving = harness
        .manager
        .server_config
        .effective_storage_backend
        .to_string();
    let (status, body) = send(
        harness.router(),
        request(
            "/api/admin/storage/import",
            json!({"name": "x", "backend": serving}),
            true,
            Some(ADMIN_KEY),
        ),
    )
    .await;
    std::env::remove_var(ADMIN_API_KEY_ENV);
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
}

#[tokio::test]
async fn an_export_name_cannot_leave_the_exports_directory() {
    let _env = ADMIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(ADMIN_API_KEY_ENV, ADMIN_KEY);
    let harness = seeded_harness();
    let (status, body) = send(
        harness.router(),
        request(
            "/api/admin/storage/export",
            json!({"user": "una", "name": "../escape"}),
            true,
            Some(ADMIN_KEY),
        ),
    )
    .await;
    std::env::remove_var(ADMIN_API_KEY_ENV);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(!harness.manager.base_path.join("escape").exists());
}

#[tokio::test]
async fn an_export_failure_is_an_error_response_never_a_success() {
    let _env = ADMIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(ADMIN_API_KEY_ENV, ADMIN_KEY);
    let harness = seeded_harness();
    crate::storage::migrate::plant_undecodable_rocks_memory(&harness.manager.base_path.join("una"));
    let (status, body) = send(
        harness.router(),
        request(
            "/api/admin/storage/export",
            json!({"user": "una", "name": "snap", "backend": "rocksdb"}),
            true,
            Some(ADMIN_KEY),
        ),
    )
    .await;
    std::env::remove_var(ADMIN_API_KEY_ENV);
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert!(
        body["detail"]
            .as_str()
            .unwrap()
            .contains("failed to decode"),
        "{body}"
    );
}
