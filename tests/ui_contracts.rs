//! The /api/ui/* contracts, driven through the real router and auth middleware: the user is bound
//! from configuration, writes record their tags' sources, every recall — including one that finds
//! nothing — reaches the recall log, and every response carries the envelope the GUI renders.

use std::sync::{Arc, Once};
use std::time::Duration;

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

const TEST_KEY: &str = "ui-contracts-test-key";
const UI_USER: &str = "ui-contracts-user";
static ENV_INIT: Once = Once::new();

fn init_env() {
    ENV_INIT.call_once(|| unsafe {
        std::env::set_var("VELD_API_KEYS", TEST_KEY);
        std::env::set_var("VELD_LAZY_CROSS_ENCODER", "1");
        std::env::set_var("VELD_UI_USER_ID", UI_USER);
    });
}

fn app() -> (Router, TempDir) {
    init_env();
    let dir = TempDir::new().unwrap();
    let cfg = ServerConfig {
        storage_path: dir.path().to_path_buf(),
        backup_enabled: false,
        ..ServerConfig::default()
    };
    let mgr = Arc::new(MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg).unwrap());
    let protected = build_protected_routes(mgr.clone(), false)
        .layer(axum::middleware::from_fn(veld::auth::auth_middleware));
    let router = Router::new()
        .merge(build_probe_routes(mgr.clone()))
        .merge(build_public_routes(mgr, false))
        .merge(protected);
    (router, dir)
}

fn get(uri: &str, key: Option<&str>) -> Request<Body> {
    let mut b = Request::builder().method(Method::GET).uri(uri);
    if let Some(k) = key {
        b = b.header("x-api-key", k);
    }
    b.body(Body::empty()).unwrap()
}

fn post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

async fn call(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

fn assert_envelope(v: &Value, route: &str) {
    assert_eq!(v["schema"], 1, "{route}: {v}");
    assert_eq!(v["user_id"], UI_USER, "{route}");
    assert_eq!(v["binding"], "ui-user-env", "{route}");
    assert!(v["measured_at"].is_string(), "{route}");
    assert!(v["witness"].is_string(), "{route}");
}

fn tag<'a>(report: &'a Value, name: &str) -> Option<&'a Value> {
    report["tags"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["tag"] == name)
}

#[tokio::test]
async fn ui_routes_bind_the_user_record_sources_and_log_every_recall() {
    let (app, _dir) = app();

    assert_eq!(
        call(&app, get("/api/ui/tags", None)).await.0,
        StatusCode::UNAUTHORIZED
    );
    let (status, _) = call(
        &app,
        get("/api/ui/tags?user_id=someone-else", Some(TEST_KEY)),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "another user is refused");

    let (status, body) = call(
        &app,
        post(
            "/api/remember",
            json!({
                "user_id": UI_USER,
                "content": "Decided to keep the tag ledger beside the record because bincode is positional.",
                "tags": ["project:veld", "scope:claude-code", "hook"],
                "memory_type": "Decision",
            }),
        ),
    )
    .await;
    assert!(status.is_success(), "remember: {status} {body}");

    let (status, v) = call(&app, get("/api/ui/tags", Some(TEST_KEY))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_envelope(&v, "tags");
    let data = &v["data"];
    assert_eq!(data["state"], "measured", "{data}");
    let report = &data["value"];
    assert_eq!(report["population"]["memories"], 1);
    assert_eq!(report["population"]["with_tag_sources"], 1);
    let project = tag(report, "project:veld").expect("an authored tag is listed");
    assert_eq!(project["origin"]["authored"], 1);
    assert_eq!(project["sources"]["caller"], 1);
    assert!(
        tag(report, "scope:claude-code").is_none(),
        "plumbing is hidden by default"
    );
    assert_eq!(report["hidden"]["plumbing"]["tags"], 2);

    let (_, all) = call(
        &app,
        get("/api/ui/tags?include_hidden=true", Some(TEST_KEY)),
    )
    .await;
    assert_eq!(
        tag(&all["data"]["value"], "hook").unwrap()["class"],
        "plumbing"
    );

    for (external, tags) in [("ext-1", vec!["first"]), ("ext-1", vec!["second"])] {
        let (status, body) = call(
            &app,
            post(
                "/api/upsert",
                json!({"user_id": UI_USER, "external_id": external, "content": "upserted", "tags": tags}),
            ),
        )
        .await;
        assert!(status.is_success(), "upsert: {status} {body}");
    }
    let (_, v) = call(&app, get("/api/ui/tags", Some(TEST_KEY))).await;
    let report = &v["data"]["value"];
    assert_eq!(report["population"]["undetermined"], 0, "{report}");
    assert!(tag(report, "second").is_some());
    assert!(
        tag(report, "first").is_none(),
        "an upsert replaces the tags and their row"
    );

    let (status, body) = call(
        &app,
        post(
            "/api/recall",
            json!({"user_id": UI_USER, "query": "zebra quantum harpsichord", "episode_id": "transcript-ui-1", "limit": 5}),
        ),
    )
    .await;
    assert!(status.is_success(), "recall: {status} {body}");

    // The event is written off the response path.
    let mut log = Value::Null;
    for _ in 0..50 {
        let (status, v) = call(&app, get("/api/ui/recall-log", Some(TEST_KEY))).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        if v["data"]["value"]["total"].as_u64().unwrap_or(0) >= 1 {
            log = v;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_envelope(&log, "recall-log");
    let report = &log["data"]["value"];
    assert_eq!(report["total"], 1, "{report}");
    let event = &report["events"][0];
    assert_eq!(event["endpoint"], "/api/recall");
    assert_eq!(event["episode_id"], "transcript-ui-1");
    assert_eq!(report["attributed"], 1);
    assert_eq!(
        report["misses"].as_u64().unwrap(),
        u64::from(event["result_count"] == 0)
    );

    let (status, v) = call(&app, get("/api/ui/knows", Some(TEST_KEY))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_envelope(&v, "knows");
    let rulings = &v["data"]["rulings"];
    assert_eq!(rulings["state"], "measured", "{rulings}");
    assert_eq!(rulings["value"]["count"], 1);
    assert_eq!(
        rulings["value"]["items"][0]["authored_tags"],
        json!(["project:veld", "scope:claude-code", "hook"])
    );
    assert_eq!(v["data"]["contradictions"]["state"], "undetermined");
    assert_eq!(v["data"]["suspected_unknowns"]["state"], "unmeasured");
}
