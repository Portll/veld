//! Remember latency under each BM25 commit interval (`bm25_commit_interval_ms`).
//!
//! A measurement, not a check, so it is ignored by default:
//!
//! ```text
//! scripts/test-local.sh --test bm25_commit_latency -- --ignored --nocapture
//! ```
//!
//! Each interval gets a fresh tenant. After a warm-up, it times sequential
//! remembers and then remembers from four concurrent callers (which queue on the
//! tenant's journal mutex, where an inline commit delays everyone behind it),
//! and finally checks that a recall finds the last memory written.

use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

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
    handlers::{build_protected_routes, MultiUserMemoryManager},
};

const TEST_KEY: &str = "bm25-commit-latency-key";
const USER: &str = "latency-user";
const WARM_UP: usize = 20;
const SEQUENTIAL: usize = 200;
const CALLERS: usize = 4;
const PER_CALLER: usize = 50;

static ENV_INIT: Once = Once::new();

fn init_env() {
    ENV_INIT.call_once(|| {
        // SAFETY: called once, before this binary's only test starts its runtime work.
        unsafe {
            std::env::set_var("VELD_API_KEYS", TEST_KEY);
            std::env::set_var("VELD_LAZY_CROSS_ENCODER", "1");
        }
    });
}

fn app(interval_ms: u64) -> (Router, TempDir) {
    init_env();
    let dir = TempDir::new().expect("create temp dir");
    let cfg = ServerConfig {
        storage_path: dir.path().to_path_buf(),
        backup_enabled: false,
        bm25_commit_interval_ms: interval_ms,
        ..ServerConfig::default()
    };
    let mgr = MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg)
        .expect("create MultiUserMemoryManager");
    let router = build_protected_routes(Arc::new(mgr), false)
        .layer(axum::middleware::from_fn(veld::auth::auth_middleware));
    (router, dir)
}

fn post(uri: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn content(caller: usize, i: usize) -> String {
    const SUBJECTS: [&str; 6] = [
        "the deploy pipeline",
        "the billing service",
        "Priya's onboarding notes",
        "the Lisbon offsite",
        "the storage migration",
        "the on-call rota",
    ];
    const FACTS: [&str; 5] = [
        "moved to Thursday after the vendor call",
        "needs a second reviewer before it ships",
        "was blocked on a missing API token",
        "now lives in the shared drive under ops",
        "caused the latency spike on Monday",
    ];
    format!(
        "Note {caller}-{i}: {} {} (ticket {}).",
        SUBJECTS[(i + caller) % SUBJECTS.len()],
        FACTS[(i * 7 + caller) % FACTS.len()],
        1000 + caller * 1000 + i
    )
}

async fn remember(app: &Router, text: String) -> Duration {
    let started = Instant::now();
    let resp = app
        .clone()
        .oneshot(post(
            "/api/remember",
            json!({ "user_id": USER, "content": text }),
        ))
        .await
        .unwrap();
    let elapsed = started.elapsed();
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        status,
        StatusCode::OK,
        "remember failed: {}",
        String::from_utf8_lossy(&body)
    );
    elapsed
}

fn summary(label: &str, mut samples: Vec<Duration>) -> String {
    samples.sort();
    let pick = |q: f64| {
        let rank = ((samples.len() as f64 * q).ceil() as usize).clamp(1, samples.len());
        samples[rank - 1].as_secs_f64() * 1000.0
    };
    format!(
        "{label:<11} n={:<4} p50={:>8.2}ms p99={:>8.2}ms max={:>8.2}ms",
        samples.len(),
        pick(0.50),
        pick(0.99),
        pick(1.0)
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "measurement; run with --ignored --nocapture"]
async fn remember_latency_by_bm25_commit_interval() {
    for interval_ms in [0u64, 1000, 5000] {
        let (app, _dir) = app(interval_ms);

        for i in 0..WARM_UP {
            remember(&app, content(99, i)).await;
        }

        let mut sequential = Vec::with_capacity(SEQUENTIAL);
        for i in 0..SEQUENTIAL {
            sequential.push(remember(&app, content(0, i)).await);
        }

        let mut callers = Vec::with_capacity(CALLERS);
        for caller in 1..=CALLERS {
            let app = app.clone();
            callers.push(tokio::spawn(async move {
                let mut samples = Vec::with_capacity(PER_CALLER);
                for i in 0..PER_CALLER {
                    samples.push(remember(&app, content(caller, i)).await);
                }
                samples
            }));
        }
        let mut concurrent = Vec::with_capacity(CALLERS * PER_CALLER);
        for caller in callers {
            concurrent.extend(caller.await.unwrap());
        }

        let last = content(CALLERS, PER_CALLER - 1);
        let resp = app
            .clone()
            .oneshot(post(
                "/api/recall",
                json!({ "user_id": USER, "query": last, "limit": 5 }),
            ))
            .await
            .unwrap();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let found = String::from_utf8_lossy(&body).contains(&last);

        println!("bm25_commit_interval_ms={interval_ms}");
        println!("  {}", summary("sequential", sequential));
        println!("  {}", summary("4 callers", concurrent));
        println!("  recall finds the last write: {found}");
        assert!(found, "recall must see a memory written just before it");
    }
}
