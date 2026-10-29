//! What a follower serves, and `GET /api/replication/role`.
//!
//! A follower holds the leader's memory records and nothing else (the follower
//! applies only memory frames; see `crate::replication`), and only those frames may
//! write to it. [`guard`] is the one place that decides, by route, what a client
//! request does on a follower:
//!
//! - [`Access::Write`] changes state replication owns, or state the follower
//!   would keep apart from its leader: refused with 503 `READ_ONLY_REPLICA`.
//! - [`Access::NotReplicated`] reads a store replication does not carry, which on
//!   a follower is empty or stale: refused with 503 `NOT_REPLICATED`.
//! - [`Access::Read`] reads memory records or this process's own runtime state.
//! - [`Access::Local`] writes only this instance's own operational state (its
//!   logins, keyring, backups, vector index and the replication controls), which no
//!   leader owns.
//!
//! Every protected route is listed in [`ROUTES`]; a test holds the list to the
//! routes `build_protected_routes` mounts, so a new route cannot go unclassified.
//! On a leader or a standalone instance the guard passes every request through.

use axum::{
    extract::{MatchedPath, Request, State},
    http::{header::RETRY_AFTER, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use super::router::AppState;
use crate::auth::ApiKeyVerified;
use crate::replication::{LeaderRef, RoleReport};

pub const READ_ONLY_REPLICA: &str = "READ_ONLY_REPLICA";
pub const NOT_REPLICATED: &str = "NOT_REPLICATED";
/// Seconds this follower was behind its leader when it last reached it; see
/// `FollowStatus::lag_seconds`. Absent until the follower has caught up once.
pub const LAG_HEADER: &str = "x-veld-replica-lag-seconds";
/// Seconds since this follower last reached its leader; see
/// `FollowStatus::leader_contact_age_seconds`. Grows while the leader is unreachable.
pub const CONTACT_AGE_HEADER: &str = "x-veld-replica-contact-age-seconds";
/// A follower stays one until an operator unfollows it, so a refused client is told
/// to wait rather than retry at once.
pub const RETRY_AFTER_SECS: u64 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    NotReplicated,
    Local,
}

use Access::{Local, NotReplicated, Read, Write};

/// Every route `build_protected_routes` mounts, by method and path template.
pub const ROUTES: &[(&str, &str, Access)] = &[
    ("GET", "/metrics", Read),
    // User auth: this instance's own logins, sessions and keyring.
    ("POST", "/api/user_auth/2fa/enroll", Local),
    ("POST", "/api/user_auth/2fa/confirm", Local),
    ("POST", "/api/user_auth/logout", Local),
    ("POST", "/api/user_auth/admin/promote", Local),
    ("POST", "/api/user_auth/admin/demote", Local),
    ("POST", "/api/user_auth/sessions/revoke_all", Local),
    ("POST", "/api/user_auth/sessions/revoke_mine", Local),
    ("GET", "/api/user_auth/me", Local),
    ("POST", "/api/user_auth/password", Local),
    ("POST", "/api/user_auth/recovery_codes", Local),
    ("POST", "/api/user_auth/sessions/revoke", Local),
    ("GET", "/api/user_auth/admin/users", Local),
    ("POST", "/api/user_auth/admin/users/disable", Local),
    ("POST", "/api/user_auth/admin/users/enable", Local),
    ("POST", "/api/user_auth/admin/users/reset_password", Local),
    ("POST", "/api/user_auth/admin/users/reset_2fa", Local),
    ("POST", "/api/user_auth/admin/users/unlock", Local),
    ("GET", "/api/user_auth/admin/sessions", Local),
    ("POST", "/api/user_auth/admin/sessions/revoke", Local),
    ("GET", "/api/user_auth/admin/audit", Local),
    ("GET", "/api/user_auth/admin/keys/status", Local),
    ("POST", "/api/user_auth/admin/keys/init", Local),
    ("POST", "/api/user_auth/admin/keys/recovery_codes", Local),
    ("POST", "/api/user_auth/admin/keys/revoke", Local),
    ("POST", "/api/user_auth/admin/keys/rotate_passphrase", Local),
    ("POST", "/api/user_auth/admin/keys/rotate_data_keys", Local),
    (
        "POST",
        "/api/user_auth/admin/keys/rotate_master_keys",
        Local,
    ),
    ("POST", "/api/user_auth/admin/keys/factors", Local),
    ("POST", "/api/user_auth/admin/keys/passkey", Local),
    ("GET", "/api/user_auth/admin/keys/passkey/{ticket}", Local),
    ("POST", "/api/user_auth/register", Local),
    ("POST", "/api/user_auth/login", Local),
    ("POST", "/api/user_auth/recover", Local),
    ("GET", "/api/user_auth/keys/passkey/{ticket}", Local),
    ("POST", "/api/user_auth/keys/passkey/{ticket}/result", Local),
    ("ANY", "/api/user_auth/{*path}", Local),
    ("GET", "/api/health/ready", Read),
    ("GET", "/api/health/index", Read),
    // Claude session status: posted by hooks, held in memory, never replicated.
    ("GET", "/api/context/sse", NotReplicated),
    ("GET", "/api/context/status", NotReplicated),
    ("POST", "/api/context/status", Write),
    ("GET", "/api/context_status", NotReplicated),
    ("POST", "/api/context_status", Write),
    ("POST", "/webhook/linear", Write),
    ("POST", "/webhook/github", Write),
    ("POST", "/api/remember", Write),
    ("POST", "/api/remember/batch", Write),
    ("POST", "/api/batch_remember", Write),
    ("POST", "/api/upsert", Write),
    ("POST", "/api/ingest", Write),
    ("POST", "/api/seed", Write),
    ("GET", "/api/ui/tags", Read),
    ("GET", "/api/ui/recall-log", Read),
    ("GET", "/api/ui/knows", Read),
    // Recall's usage writes are off on a follower (`MultiUserMemoryManager::recall_readonly`).
    ("POST", "/api/recall", Read),
    ("POST", "/api/recall/tracked", Read),
    ("POST", "/api/recall/tags", Read),
    ("POST", "/api/recall/by-tags", Read),
    ("POST", "/api/recall/date", Read),
    ("POST", "/api/context_summary", Read),
    ("POST", "/api/proactive_context", Read),
    ("POST", "/api/context", Read),
    ("POST", "/api/relevant", Read),
    ("POST", "/api/reinforce", Write),
    ("POST", "/api/prompt/gen", Read),
    ("POST", "/api/prompt/generate", Read),
    ("POST", "/api/entity/resolve", NotReplicated),
    ("POST", "/api/entity/attribute", Write),
    ("POST", "/api/entity/merge", Write),
    ("POST", "/api/entity/alias", Write),
    ("GET", "/api/memory/by-external-id", Read),
    ("GET", "/api/memory/{memory_id}", Read),
    ("GET", "/api/memories/{memory_id}", Read),
    ("GET", "/api/memory/{memory_id}/health", Read),
    ("PUT", "/api/memory/{memory_id}", Write),
    ("DELETE", "/api/memory/{memory_id}", Write),
    ("DELETE", "/api/forget/{memory_id}", Write),
    ("GET", "/api/list/{user_id}", Read),
    ("POST", "/api/memories", Read),
    ("GET", "/api/memories", Read),
    ("GET", "/api/memories/count", Read),
    ("POST", "/api/memories/bulk", Write),
    ("POST", "/api/memories/clear", Write),
    ("POST", "/api/anchor", Write),
    // Render applies Obsidian edits back to the store.
    ("POST", "/api/wiki/render", Write),
    ("POST", "/api/file_answer", Write),
    ("POST", "/api/memory/merge", Write),
    ("POST", "/api/memory/split", Write),
    ("POST", "/api/memory/tier", Write),
    ("POST", "/api/forget", Write),
    ("POST", "/api/forget/age", Write),
    ("POST", "/api/forget/importance", Write),
    ("POST", "/api/forget/pattern", Write),
    ("POST", "/api/forget/tags", Write),
    ("POST", "/api/forget/date", Write),
    ("GET", "/api/users", Read),
    ("GET", "/api/users/{user_id}/stats", Read),
    ("DELETE", "/api/users/{user_id}", Write),
    ("GET", "/api/stats", Read),
    ("POST", "/api/memory/compress", Write),
    ("POST", "/api/memory/decompress", Write),
    ("GET", "/api/storage/stats", Read),
    ("POST", "/api/admin/storage/export", Read),
    ("POST", "/api/admin/storage/import", Write),
    ("POST", "/api/admin/storage/migrate", Write),
    ("GET", "/api/admin/replication/status", Read),
    ("GET", "/api/admin/replication/records", Read),
    // Unfollowing is how an operator promotes a follower.
    ("POST", "/api/admin/replication/follow", Local),
    ("POST", "/api/admin/replication/unfollow", Local),
    ("POST", "/api/search/advanced", Read),
    ("POST", "/api/storage/uncompressed", Read),
    ("POST", "/api/index/verify", Read),
    // Both write the vector index and its id mappings from the stored records, never a
    // record: the index is this instance's own projection. Reembed rewrites records.
    ("POST", "/api/index/repair", Local),
    ("POST", "/api/index/rebuild", Local),
    ("POST", "/api/index/reembed", Write),
    ("GET", "/api/embeddings/backlog", Read),
    ("POST", "/api/embeddings/backlog/drain", Write),
    ("POST", "/api/storage/cleanup", Write),
    ("POST", "/api/storage/migrate", Write),
    ("POST", "/api/consolidate", Write),
    ("POST", "/api/consolidation/sleep", Write),
    // Consolidation runs on the leader; its events are not carried.
    ("POST", "/api/consolidation/report", NotReplicated),
    ("GET", "/api/consolidation/events", NotReplicated),
    // A backup reads this instance's store and changes no record.
    ("POST", "/api/backup/create", Local),
    ("POST", "/api/backup/list", Read),
    ("POST", "/api/backups", Read),
    ("POST", "/api/backup/verify", Read),
    ("POST", "/api/backup/purge", Local),
    ("POST", "/api/backups/purge", Local),
    ("POST", "/api/backup/restore", Write),
    ("POST", "/api/facts/list", NotReplicated),
    ("POST", "/api/facts/search", NotReplicated),
    ("POST", "/api/facts/by-entity", NotReplicated),
    ("POST", "/api/facts/stats", NotReplicated),
    ("POST", "/api/facts/narratives", NotReplicated),
    ("POST", "/api/facts/preview-purge", NotReplicated),
    ("POST", "/api/facts/purge", Write),
    ("POST", "/api/facts/temporal", NotReplicated),
    ("POST", "/api/facts/temporal/search", NotReplicated),
    ("GET", "/api/context/blocks", NotReplicated),
    ("GET", "/api/context/blocks/{key}", NotReplicated),
    ("PUT", "/api/context/blocks/{key}", Write),
    ("DELETE", "/api/context/blocks/{key}", Write),
    ("POST", "/api/sleep_time/enqueue", Write),
    ("GET", "/api/sleep_time/status", Read),
    ("GET", "/api/sleep_time/status/{user_id}", Read),
    ("POST", "/api/sleep_time/lock_block", Write),
    ("POST", "/api/lineage/trace", NotReplicated),
    ("POST", "/api/lineage/edges", NotReplicated),
    ("POST", "/api/lineage/confirm", Write),
    ("POST", "/api/lineage/reject", Write),
    ("POST", "/api/lineage/link", Write),
    ("POST", "/api/lineage/stats", NotReplicated),
    ("POST", "/api/lineage/branches", NotReplicated),
    ("POST", "/api/lineage/branch", Write),
    // The knowledge graph and tag provenance are not journaled.
    ("GET", "/api/graph/{user_id}/stats", NotReplicated),
    ("GET", "/api/graph/{user_id}/universe", NotReplicated),
    ("DELETE", "/api/graph/{user_id}/clear", Write),
    ("POST", "/api/graph/{user_id}/rebuild", Write),
    ("POST", "/api/graph/entity/find", NotReplicated),
    ("DELETE", "/api/graph/entity/delete", Write),
    ("POST", "/api/graph/entities/all", NotReplicated),
    ("POST", "/api/graph/entities/frequency", NotReplicated),
    ("POST", "/api/graph/tags/stamped-cores", NotReplicated),
    ("POST", "/api/graph/relationship/invalidate", Write),
    ("POST", "/api/graph/traverse", NotReplicated),
    ("POST", "/api/graph/episode/get", NotReplicated),
    ("POST", "/api/gap/analyze", Read),
    ("POST", "/api/gap/stats", Read),
    ("POST", "/api/gap/voronoi", Read),
    ("POST", "/api/gap/persistence", Read),
    ("POST", "/api/gap/mapper", Read),
    ("POST", "/api/graph/entity/add", Write),
    ("POST", "/api/graph/relationship/add", Write),
    ("GET", "/api/brain/{user_id}", Read),
    ("GET", "/api/graph/data/{user_id}", NotReplicated),
    ("GET", "/api/visualization/{user_id}/stats", Read),
    ("GET", "/api/visualization/{user_id}/dot", Read),
    ("POST", "/api/visualization/build", Read),
    ("POST", "/api/todos", NotReplicated),
    ("POST", "/api/todos/list", NotReplicated),
    ("POST", "/api/todos/add", Write),
    ("POST", "/api/todos/update", Write),
    ("POST", "/api/todos/complete", Write),
    ("POST", "/api/todos/delete", Write),
    ("POST", "/api/todos/reorder", Write),
    ("POST", "/api/todos/due", NotReplicated),
    ("GET", "/api/todos/{todo_id}", NotReplicated),
    ("DELETE", "/api/todos/{todo_id}", Write),
    ("POST", "/api/todos/{todo_id}/update", Write),
    ("POST", "/api/todos/{todo_id}/complete", Write),
    ("POST", "/api/todos/{todo_id}/reorder", Write),
    ("GET", "/api/todos/{todo_id}/subtasks", NotReplicated),
    ("GET", "/api/todos/{todo_id}/dependents", NotReplicated),
    (
        "GET",
        "/api/todos/{todo_id}/dependency_chain",
        NotReplicated,
    ),
    ("POST", "/api/todos/ready", NotReplicated),
    ("GET", "/api/todos/{todo_id}/comments", NotReplicated),
    ("POST", "/api/todos/{todo_id}/comments", Write),
    ("PUT", "/api/todos/{todo_id}/comments/{comment_id}", Write),
    (
        "POST",
        "/api/todos/{todo_id}/comments/{comment_id}/update",
        Write,
    ),
    (
        "DELETE",
        "/api/todos/{todo_id}/comments/{comment_id}",
        Write,
    ),
    ("POST", "/api/todos/stats", NotReplicated),
    ("POST", "/api/projects", Write),
    ("POST", "/api/projects/list", NotReplicated),
    ("POST", "/api/projects/add", Write),
    ("GET", "/api/projects/{project_id}", NotReplicated),
    ("DELETE", "/api/projects/{project_id}", Write),
    ("POST", "/api/projects/{project_id}/update", Write),
    ("POST", "/api/projects/{project_id}/delete", Write),
    ("POST", "/api/projects/{project_id}/files", NotReplicated),
    ("POST", "/api/projects/{project_id}/scan", NotReplicated),
    ("POST", "/api/projects/{project_id}/index", Write),
    (
        "POST",
        "/api/projects/{project_id}/files/search",
        NotReplicated,
    ),
    ("GET", "/api/files/stats", NotReplicated),
    ("POST", "/api/reminders", NotReplicated),
    ("POST", "/api/reminders/set", Write),
    ("POST", "/api/remind", Write),
    ("POST", "/api/reminders/due", NotReplicated),
    ("POST", "/api/reminders/check", NotReplicated),
    ("POST", "/api/reminders/context", NotReplicated),
    ("POST", "/api/reminders/{reminder_id}/dismiss", Write),
    ("POST", "/api/reminders/{reminder_id}/delete", Write),
    ("POST", "/api/sessions", NotReplicated),
    ("GET", "/api/sessions/stats", NotReplicated),
    ("POST", "/api/sessions/end", Write),
    ("GET", "/api/sessions/{session_id}", NotReplicated),
    ("GET", "/api/ab/tests", NotReplicated),
    ("POST", "/api/ab/tests", Write),
    ("GET", "/api/ab/tests/{test_id}", NotReplicated),
    ("DELETE", "/api/ab/tests/{test_id}", Write),
    ("POST", "/api/ab/tests/{test_id}/start", Write),
    ("POST", "/api/ab/tests/{test_id}/pause", Write),
    ("POST", "/api/ab/tests/{test_id}/resume", Write),
    ("POST", "/api/ab/tests/{test_id}/complete", Write),
    ("GET", "/api/ab/tests/{test_id}/analyze", NotReplicated),
    ("POST", "/api/ab/tests/{test_id}/impression", Write),
    ("POST", "/api/ab/tests/{test_id}/click", Write),
    ("POST", "/api/ab/tests/{test_id}/feedback", Write),
    ("GET", "/api/ab/summary", NotReplicated),
    ("POST", "/api/spine/dimensions", Write),
    ("POST", "/api/sleight/dimensions", Write),
    // The dials this process recalls with; tuning them would set it apart from its leader.
    ("GET", "/api/dials", Read),
    ("PUT", "/api/dials", Write),
    ("GET", "/api/dials/schema", Read),
    ("POST", "/api/external/dimensions", Write),
    ("POST", "/api/wintermute/dimensions", Write),
    ("POST", "/api/sync/linear", Write),
    ("POST", "/api/sync/github", Write),
    ("GET", "/api/context/monitor", Read),
    ("GET", "/api/events/sse", Read),
    ("GET", "/api/events", Read),
    // The streaming WebSocket ingests memories.
    ("GET", "/api/stream", Write),
    ("POST", "/api/search/multimodal", Read),
    ("POST", "/api/search/robotics", Read),
    ("POST", "/api/datasets", Write),
    ("GET", "/api/datasets", NotReplicated),
    ("GET", "/api/datasets/{name}", NotReplicated),
    ("DELETE", "/api/datasets/{name}", Write),
    ("POST", "/api/datasets/{name}/rows", Write),
    ("POST", "/api/datasets/{name}/query", NotReplicated),
    ("POST", "/api/datasets/{name}/link", Write),
    ("POST", "/api/query/plan", Read),
    ("POST", "/api/query/execute", Read),
    ("POST", "/api/export/mif", Read),
    ("POST", "/api/import/mif", Write),
    ("GET", "/api/mif/adapters", Read),
    ("GET", "/api/replication/role", Read),
];

/// How `method` on the route template `path` is treated on a follower. A route the
/// table does not name is a read when it is a GET and a write otherwise.
pub fn access(method: &Method, path: &str) -> Access {
    let method = if *method == Method::HEAD {
        "GET"
    } else {
        method.as_str()
    };
    let named = |m: &str| {
        ROUTES
            .iter()
            .find(|(rm, rp, _)| *rm == m && *rp == path)
            .map(|(_, _, access)| *access)
    };
    named(method)
        .or_else(|| named("ANY"))
        .unwrap_or(if method == "GET" { Read } else { Write })
}

/// A 503 that tells the client to go to the leader. The leader is named only to a
/// caller whose API key was verified: webhooks reach this layer without one.
fn refusal(code: &str, message: &str, leader: &LeaderRef, name_leader: bool) -> Response {
    let mut body = json!({ "code": code, "message": message });
    if name_leader {
        body["leader"] = json!({ "instance_id": leader.instance_id, "url": leader.url });
    }
    let mut response = (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    response
        .headers_mut()
        .insert(RETRY_AFTER, HeaderValue::from(RETRY_AFTER_SECS));
    response
}

/// Applied to every protected route. On a follower it refuses writes and reads of
/// unreplicated stores, and stamps [`LAG_HEADER`] and [`CONTACT_AGE_HEADER`] on
/// authenticated responses.
pub async fn guard(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let Some(leader) = state.replication.leader() else {
        return next.run(request).await;
    };
    let verified = request.extensions().get::<ApiKeyVerified>().is_some();
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|path| access(request.method(), path.as_str()));
    let mut response = match route {
        Some(Write) => refusal(
            READ_ONLY_REPLICA,
            "this instance is a read-only replica; send writes to its leader",
            &leader,
            verified,
        ),
        Some(NotReplicated) => refusal(
            NOT_REPLICATED,
            "replication does not carry what this route reads; read it from the leader",
            &leader,
            verified,
        ),
        Some(Read | Local) | None => next.run(request).await,
    };
    if verified {
        let headers = response.headers_mut();
        if let Some(lag) = leader.lag_seconds {
            headers.insert(LAG_HEADER, HeaderValue::from(lag));
        }
        if let Some(age) = leader.contact_age_seconds {
            headers.insert(CONTACT_AGE_HEADER, HeaderValue::from(age));
        }
    }
    response
}

/// `GET /api/replication/role`: which role this instance plays. Identity-level, so
/// any valid API key reads it, whatever tenant the key is bound to.
pub async fn role(State(state): State<AppState>) -> Json<RoleReport> {
    Json(state.replication.role())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::{HeaderMap, Request as HttpRequest};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;
    use crate::handlers::test_helpers::{
        delete, get, get_unauthenticated, post_json, put_json, TestHarness, TEST_API_KEY,
    };
    use crate::intent_log::{payload, IntentPayload};
    use crate::memory::types::{Experience, Memory, MemoryId};

    const LEADER: &str = "http://127.0.0.1:9";
    const USER: &str = "replica-user";

    /// The `(METHOD, path)` of every `.route(…)` in `build_protected_routes`, read from
    /// the source as the public-router guard reads it: axum does not expose its table.
    fn mounted_routes() -> BTreeSet<(String, String)> {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/handlers/router.rs"
        ))
        .expect("read router.rs");
        let start = source
            .find("pub fn build_protected_routes(")
            .expect("build_protected_routes in router.rs");
        let end = start
            + source[start..]
                .find("\npub fn build_router(")
                .expect("build_router follows it");
        let body: String = source[start..end]
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");

        let mut routes = BTreeSet::new();
        let mut rest = body.as_str();
        while let Some(at) = rest.find(".route(") {
            rest = &rest[at + ".route(".len()..];
            let call = &rest[..rest.find(".route(").unwrap_or(rest.len())];
            let path = call
                .trim_start()
                .strip_prefix('"')
                .and_then(|tail| tail.split('"').next())
                .unwrap_or_else(|| panic!("a .route( without a literal path: {call:.60}"));
            let before = routes.len();
            for method in ["get", "post", "put", "delete", "patch", "any"] {
                let token = format!("{method}(");
                for (i, _) in call.match_indices(&token) {
                    let prior = call[..i].chars().next_back();
                    if !prior.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                        routes.insert((method.to_uppercase(), path.to_string()));
                    }
                }
            }
            assert!(routes.len() > before, "no method found for {path}");
        }
        routes
    }

    #[test]
    fn every_protected_route_is_classified_exactly_once() {
        let mounted = mounted_routes();
        let classified: BTreeSet<(String, String)> = ROUTES
            .iter()
            .map(|(m, p, _)| (m.to_string(), p.to_string()))
            .collect();
        assert_eq!(classified.len(), ROUTES.len(), "a route is listed twice");
        let unclassified: Vec<_> = mounted.difference(&classified).collect();
        assert!(
            unclassified.is_empty(),
            "routes with no replica access in ROUTES: {unclassified:?}"
        );
        let stale: Vec<_> = classified.difference(&mounted).collect();
        assert!(
            stale.is_empty(),
            "ROUTES names routes not mounted: {stale:?}"
        );
    }

    #[test]
    fn an_unlisted_route_fails_closed_for_anything_but_a_get() {
        assert_eq!(access(&Method::POST, "/api/remember"), Write);
        assert_eq!(access(&Method::POST, "/api/no-such-route"), Write);
        assert_eq!(access(&Method::GET, "/api/no-such-route"), Read);
        assert_eq!(access(&Method::HEAD, "/api/todos/{todo_id}"), NotReplicated);
        assert_eq!(access(&Method::PATCH, "/api/user_auth/{*path}"), Local);
    }

    fn follow(harness: &TestHarness) {
        let state = harness.state();
        state
            .replication
            .follow(&state.base_path, LEADER, "admin-key", Some("leader-key"))
            .unwrap();
    }

    async fn call(
        harness: &TestHarness,
        request: HttpRequest<Body>,
    ) -> (StatusCode, HeaderMap, serde_json::Value) {
        let response = harness.router().oneshot(request).await.unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, headers, body)
    }

    fn remember_directly(harness: &TestHarness, content: &str) -> MemoryId {
        let memory = harness.state().get_user_memory(USER).unwrap();
        let id = memory
            .read()
            .remember(
                Experience {
                    content: content.to_string(),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        id
    }

    #[tokio::test]
    async fn a_follower_refuses_a_remember_and_an_upsert() {
        let harness = TestHarness::new();
        follow(&harness);
        let writes = [
            post_json(
                "/api/remember",
                &json!({"user_id": USER, "content": "the ferry leaves at nine"}),
            ),
            post_json(
                "/api/upsert",
                &json!({"user_id": USER, "external_id": "ferry", "content": "the ferry leaves at ten"}),
            ),
        ];
        for request in writes {
            let (status, headers, body) = call(&harness, request).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
            assert_eq!(body["code"], READ_ONLY_REPLICA, "{body}");
            assert_eq!(body["leader"]["url"], LEADER, "{body}");
            assert!(
                body["leader"].get("instance_id").is_some(),
                "the leader's id is named, null until a poll has read it: {body}"
            );
            assert_eq!(
                headers.get(RETRY_AFTER).unwrap(),
                &RETRY_AFTER_SECS.to_string()
            );
        }
        let memory = harness.state().get_user_memory(USER).unwrap();
        assert_eq!(memory.read().stored_count(), 0, "nothing reached the store");
    }

    /// What any write would move: the records stored and the journal's next LSN.
    fn footprint(harness: &TestHarness) -> (usize, u64) {
        let state = harness.state();
        let records = state
            .get_user_memory(USER)
            .unwrap()
            .read()
            .get_all_memories()
            .unwrap()
            .len();
        let head = state
            .get_user_journaled_writer(USER)
            .unwrap()
            .lock()
            .log()
            .next_lsn()
            .0;
        (records, head)
    }

    async fn assert_refused_as_a_write(harness: &TestHarness, request: HttpRequest<Body>) {
        let what = format!("{} {}", request.method(), request.uri());
        let (status, headers, body) = call(harness, request).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{what}: {body}");
        assert_eq!(body["code"], READ_ONLY_REPLICA, "{what}: {body}");
        assert!(headers.get(RETRY_AFTER).is_some(), "{what}");
    }

    #[tokio::test]
    async fn a_client_write_lands_once_the_follower_is_unfollowed() {
        let harness = TestHarness::new();
        let remember = || {
            post_json(
                "/api/remember",
                &json!({"user_id": USER, "content": "the slipway reopens on Monday"}),
            )
        };
        follow(&harness);
        assert_refused_as_a_write(&harness, remember()).await;
        let before = footprint(&harness);

        harness.state().replication.unfollow();
        let (status, headers, body) = call(&harness, remember()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(headers.get(LAG_HEADER).is_none() && headers.get(CONTACT_AGE_HEADER).is_none());
        let after = footprint(&harness);
        assert_eq!(after.0, before.0 + 1, "the memory was stored");
        assert!(after.1 > before.1, "and journaled");
    }

    #[tokio::test]
    async fn a_follower_refuses_put_and_delete_writes() {
        let harness = TestHarness::new();
        let id = remember_directly(&harness, "the tug berths at pier four").0;
        follow(&harness);
        let before = footprint(&harness);
        let requests = [
            put_json(
                &format!("/api/memory/{id}?user_id={USER}"),
                &json!({"user_id": USER, "content": "the tug berths at pier five"}),
            ),
            delete(&format!("/api/memory/{id}?user_id={USER}")),
            delete(&format!("/api/forget/{id}?user_id={USER}")),
            delete(&format!("/api/users/{USER}")),
            put_json("/api/dials", &json!({"rerank_k": 5})),
            put_json(
                &format!("/api/context/blocks/persona?user_id={USER}"),
                &json!({"user_id": USER, "content": "prefers short answers"}),
            ),
            delete(&format!("/api/context/blocks/persona?user_id={USER}")),
        ];
        for request in requests {
            assert_refused_as_a_write(&harness, request).await;
        }
        assert_eq!(footprint(&harness), before);
        let memory = harness.state().get_user_memory(USER).unwrap();
        let kept = memory.read().get_memory(&MemoryId(id)).unwrap();
        assert_eq!(kept.experience.content, "the tug berths at pier four");
    }

    #[tokio::test]
    async fn a_follower_refuses_a_mif_import_and_a_backup_restore() {
        let harness = TestHarness::new();
        remember_directly(&harness, "the dredger works the channel at low tide");
        follow(&harness);
        let before = footprint(&harness);
        let (status, _, exported) = call(
            &harness,
            post_json("/api/export/mif", &json!({"user_id": USER})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "an export is a read: {exported}");
        assert_eq!(
            exported["not_replicated"],
            json!(["todos", "projects", "reminders"]),
            "the export says which of its stores are this instance's own"
        );

        assert_refused_as_a_write(
            &harness,
            post_json(
                "/api/import/mif",
                &json!({"user_id": USER, "skip_duplicates": false, "data": exported}),
            ),
        )
        .await;
        assert_refused_as_a_write(
            &harness,
            post_json(
                "/api/backup/restore",
                &json!({"user_id": USER, "backup_id": 1}),
            ),
        )
        .await;
        assert_eq!(footprint(&harness), before);
    }

    /// Every route the table lets through on a follower, with a request each handler
    /// can take, leaves the records and the journal as they were.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn on_a_follower_no_read_or_local_route_moves_a_record_or_the_journal() {
        use crate::handlers::admin::{ADMIN_API_KEY_ENV, ADMIN_API_KEY_HEADER, ADMIN_ENV_LOCK};
        use axum::http::header::CONTENT_TYPE;
        const ADMIN_KEY: &str = "replica-sweep-admin-key";
        let _env = ADMIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(ADMIN_API_KEY_ENV, ADMIN_KEY);

        let harness = TestHarness::new();
        let id = remember_directly(&harness, "the harbour crane is serviced every spring").0;
        remember_directly(&harness, "the harbour master keeps the crane log");
        follow(&harness);
        let before = footprint(&harness);
        let now = chrono::Utc::now();
        let shared = json!({
            "user_id": USER,
            "query": "harbour crane",
            "query_text": "harbour crane",
            "context": "who services the harbour crane",
            "goal": "brief the crane crew",
            "tags": ["harbour"],
            "start": (now - chrono::Duration::days(1)).to_rfc3339(),
            "end": now.to_rfc3339(),
            "memory_id": id,
            "limit": 5,
        });
        let body_for = |path: &str| -> serde_json::Value {
            let extra = match path {
                "/api/admin/storage/export" => return json!({"user": USER, "name": "sweep"}),
                "/api/memories" => json!({"tags": "harbour"}),
                "/api/storage/uncompressed" => json!({"days_old": 0}),
                "/api/backup/verify" => json!({"backup_id": 1}),
                "/api/backup/purge" | "/api/backups/purge" => json!({"keep_count": 5}),
                "/api/search/multimodal" | "/api/search/robotics" => json!({"mode": "hybrid"}),
                "/api/search/advanced" => json!({"entity_name": "harbour"}),
                _ => json!({}),
            };
            let mut body = shared.clone();
            for (key, value) in extra.as_object().unwrap() {
                body[key] = value.clone();
            }
            body
        };

        // Requests this harness cannot make succeed; each still passes the guard. No
        // user-auth store and no sleep-time orchestrator are set up, the monitor is a
        // WebSocket upgrade, the planner takes its own query language, and no memory
        // carries the external id asked for. The admin export opens the engine's files
        // beside the live store, which redb, one open per database, refuses.
        let serves_redb = harness.state().server_config().effective_storage_backend
            == crate::config::StorageBackend::Redb;
        let unanswerable = |template: &str| {
            ["/api/user_auth/", "/api/sleep_time/", "/api/query/"]
                .iter()
                .any(|prefix| template.starts_with(prefix))
                || matches!(
                    template,
                    "/api/context/monitor" | "/api/memory/by-external-id"
                )
                || (serves_redb && template == "/api/admin/storage/export")
        };
        // Bound to the tenant as a tenant key would be: the UI routes need a binding.
        let router = harness
            .router()
            .layer(axum::Extension(crate::auth::AuthenticatedUser {
                user_id: USER.to_string(),
            }));
        let mut unexpected = Vec::new();
        for (method, template, access) in ROUTES {
            // Following and unfollowing change the very role under test.
            let role_control = template.starts_with("/api/admin/replication/") && *method == "POST";
            if !matches!(access, Read | Local) || role_control {
                continue;
            }
            let method = if *method == "ANY" { "GET" } else { *method };
            let path = template
                .replace("{memory_id}", &id.to_string())
                .replace("{user_id}", USER)
                .replace("{ticket}", "no-such-ticket")
                .replace("{*path}", "no-such-path");
            let request = HttpRequest::builder()
                .method(method)
                .uri(format!("{path}?user_id={USER}&external_id=crane-log"))
                .header("x-api-key", TEST_API_KEY)
                .header(ADMIN_API_KEY_HEADER, ADMIN_KEY)
                .header(CONTENT_TYPE, "application/json")
                .body(if method == "GET" {
                    Body::empty()
                } else {
                    Body::from(body_for(template).to_string())
                })
                .unwrap();
            let response = router.clone().oneshot(request).await.unwrap();
            let status = response.status();
            // An event stream never ends; its headers are the whole answer.
            let streams = response
                .headers()
                .get(CONTENT_TYPE)
                .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"));
            let code = if streams {
                None
            } else {
                let bytes =
                    tokio::time::timeout(Duration::from_secs(60), response.into_body().collect())
                        .await
                        .unwrap_or_else(|_| panic!("{method} {template} did not answer"))
                        .unwrap()
                        .to_bytes();
                serde_json::from_slice::<serde_json::Value>(&bytes)
                    .ok()
                    .and_then(|b| b["code"].as_str().map(str::to_string))
            };
            assert!(
                !matches!(code.as_deref(), Some(READ_ONLY_REPLICA | NOT_REPLICATED)),
                "{method} {template} is {access:?} but was refused"
            );
            assert_eq!(
                footprint(&harness),
                before,
                "{method} {template} ({status}) wrote on a follower"
            );
            if !status.is_success() && !unanswerable(template) {
                unexpected.push(format!("{method} {template} {status}"));
            }
        }
        std::env::remove_var(ADMIN_API_KEY_ENV);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(
            footprint(&harness),
            before,
            "a write landed after its reply"
        );
        assert!(harness.state().replication.is_following());
        assert!(
            unexpected.is_empty(),
            "routes the sweep's requests did not exercise: {unexpected:#?}"
        );
    }

    #[tokio::test]
    async fn a_webhook_refused_on_a_follower_does_not_learn_the_leader() {
        let harness = TestHarness::new();
        follow(&harness);
        let request = HttpRequest::builder()
            .method(Method::POST)
            .uri("/webhook/github")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let (status, headers, body) = call(&harness, request).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert_eq!(body["code"], READ_ONLY_REPLICA);
        assert!(body.get("leader").is_none(), "{body}");
        assert!(headers.get(LAG_HEADER).is_none());
    }

    #[tokio::test]
    async fn the_replication_apply_path_still_writes_on_a_follower() {
        let harness = TestHarness::new();
        follow(&harness);
        let memory = Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            Experience {
                content: "the lock keeper works the night shift".into(),
                ..Default::default()
            },
            0.5,
            None,
            None,
            None,
            None,
        );
        let frame = IntentPayload::Remember {
            user_id: USER.into(),
            memory_id: memory.id.0.to_string(),
            memory_bincode: bincode::serde::encode_to_vec(&memory, bincode::config::standard())
                .unwrap(),
            schema_version: Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        let state = harness.state();
        let journaled = tokio::task::spawn_blocking(move || {
            crate::replication::apply_replicated_page(&state, USER, vec![(0, frame)])
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(journaled, 1);

        let (status, _, body) = call(
            &harness,
            get(&format!("/api/memory/{}?user_id={USER}", memory.id.0)),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body["experience"]["content"]
                .as_str()
                .or(body["content"].as_str()),
            Some("the lock keeper works the night shift"),
            "{body}"
        );
    }

    /// Everything recall can write.
    #[derive(Debug, PartialEq)]
    struct Usage {
        retrievals: usize,
        /// Id, access count, last access and importance bits of every memory.
        records: Vec<(String, u32, i64, u32)>,
        /// Entities and relationships, co-retrieval edges among them.
        graph: (usize, usize),
        journal_head: u64,
    }

    fn usage(harness: &TestHarness) -> Usage {
        let state = harness.state();
        let memory = state.get_user_memory(USER).unwrap();
        let guard = memory.read();
        let mut records: Vec<_> = guard
            .get_all_memories()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m.id.0.to_string(),
                    m.access_count(),
                    m.last_accessed().timestamp_micros(),
                    m.importance().to_bits(),
                )
            })
            .collect();
        records.sort();
        let graph = state.get_user_graph_stats(USER).unwrap();
        let journal = state
            .get_user_journaled_writer(USER)
            .unwrap()
            .lock()
            .log()
            .next_lsn()
            .0;
        Usage {
            retrievals: guard.stats().total_retrievals,
            records,
            graph: (graph.entity_count, graph.relationship_count),
            journal_head: journal,
        }
    }

    #[tokio::test]
    async fn recall_on_a_follower_makes_no_usage_writes() {
        let harness = TestHarness::new();
        for content in [
            "the harbour crane is serviced every spring",
            "the harbour master keeps the crane log",
            "the crane operator starts at dawn",
        ] {
            remember_directly(&harness, content);
        }
        follow(&harness);
        let before = usage(&harness);
        for _ in 0..2 {
            let (status, _, body) = call(
                &harness,
                post_json(
                    "/api/recall",
                    &json!({"user_id": USER, "query": "harbour crane"}),
                ),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let (status, _, body) = call(
                &harness,
                post_json(
                    "/api/proactive_context",
                    &json!({
                        "user_id": USER,
                        "context": "who looks after the harbour crane",
                        "previous_response": "The harbour master keeps the crane log and the operator starts at dawn, so ask either of them about servicing.",
                        "auto_ingest": true,
                    }),
                ),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert!(body["ingested_memory_id"].is_null(), "{body}");
            assert_eq!(
                body["not_replicated"],
                json!([
                    "relevant_todos",
                    "due_reminders",
                    "context_reminders",
                    "relevant_facts"
                ]),
                "{body}"
            );
        }
        // Long enough for any fire-and-forget write a handler spawned to land.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(usage(&harness), before, "recall on a follower wrote");

        harness.state().replication.unfollow();
        let (status, _, body) = call(
            &harness,
            post_json(
                "/api/recall",
                &json!({"user_id": USER, "query": "harbour crane"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(
            usage(&harness).retrievals > before.retrievals,
            "the same recall on a standalone instance counts itself, so the check above can see a write"
        );
    }

    fn records(harness: &TestHarness) -> Vec<Vec<u8>> {
        let memory = harness.state().get_user_memory(USER).unwrap();
        let guard = memory.read();
        let mut all: Vec<Vec<u8>> = guard
            .get_all_memories()
            .unwrap()
            .iter()
            .map(|m| {
                bincode::serde::encode_to_vec(m.as_ref(), bincode::config::standard()).unwrap()
            })
            .collect();
        all.sort();
        all
    }

    #[tokio::test]
    async fn a_followers_maintenance_changes_no_record() {
        let harness = TestHarness::new();
        for content in [
            "the tide gauge reads high on Thursdays",
            "the tide gauge was recalibrated in March",
        ] {
            remember_directly(&harness, content);
        }
        follow(&harness);
        let state = harness.state();
        let before = (records(&harness), usage(&harness));
        let processed = tokio::task::spawn_blocking(move || {
            (0..7)
                .map(|_| state.run_maintenance_all_users())
                .sum::<usize>()
        })
        .await
        .unwrap();
        assert_eq!(
            processed, 0,
            "heavy and light cycles both skipped their writes"
        );
        assert_eq!((records(&harness), usage(&harness)), before);

        harness.state().replication.unfollow();
        let state = harness.state();
        let processed = tokio::task::spawn_blocking(move || state.run_maintenance_all_users())
            .await
            .unwrap();
        assert!(
            processed > 0,
            "a standalone instance's maintenance does run"
        );
    }

    /// Forgets the leader sends leave deleted vectors in the follower's index, and the
    /// maintenance pass that compacts a leader's does not run on a follower.
    #[tokio::test]
    async fn a_followers_heavy_cycle_compacts_its_vector_index() {
        let harness = TestHarness::new();
        let ids: Vec<MemoryId> = [
            "the breakwater lamp is red",
            "the breakwater lamp was replaced",
            "the breakwater is closed in storms",
        ]
        .into_iter()
        .map(|content| remember_directly(&harness, content))
        .collect();
        follow(&harness);
        let forgets: Vec<(u64, IntentPayload)> = ids[..2]
            .iter()
            .enumerate()
            .map(|(lsn, id)| {
                let frame = IntentPayload::Forget {
                    user_id: USER.into(),
                    memory_id: id.0.to_string(),
                    schema_version: Some(payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
                };
                (lsn as u64, frame)
            })
            .collect();
        let state = harness.state();
        tokio::task::spawn_blocking(move || {
            crate::replication::apply_replicated_page(&state, USER, forgets).unwrap();
            state.run_maintenance_all_users()
        })
        .await
        .unwrap();

        let memory = harness.state().get_user_memory(USER).unwrap();
        let health = memory.read().index_health();
        assert_eq!(health.deleted_count, 0, "{health:?}");
        assert_eq!(health.total_vectors, 1, "{health:?}");
        assert_eq!(memory.read().stored_count(), 1);
    }

    #[tokio::test]
    async fn unreplicated_stores_answer_not_replicated_on_a_follower() {
        let harness = TestHarness::new();
        let reads = || {
            [
                post_json("/api/todos/list", &json!({"user_id": USER})),
                post_json("/api/reminders", &json!({"user_id": USER})),
                post_json("/api/projects/list", &json!({"user_id": USER})),
                get(&format!("/api/context/blocks?user_id={USER}")),
            ]
        };
        for request in reads() {
            let uri = request.uri().to_string();
            let (status, _, body) = call(&harness, request).await;
            assert_eq!(
                status,
                StatusCode::OK,
                "{uri} on a standalone instance: {body}"
            );
        }
        follow(&harness);
        for request in reads() {
            let uri = request.uri().to_string();
            let (status, headers, body) = call(&harness, request).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}: {body}");
            assert_eq!(body["code"], NOT_REPLICATED, "{uri}: {body}");
            assert_eq!(body["leader"]["url"], LEADER, "{uri}: {body}");
            assert!(headers.get(RETRY_AFTER).is_some(), "{uri}");
        }
    }

    #[tokio::test]
    async fn the_role_route_needs_an_api_key_and_no_admin_key() {
        let harness = TestHarness::new();
        let (status, _, body) = call(&harness, get("/api/replication/role")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body,
            json!({"role": "standalone", "instance_id": crate::replication::instance_id()})
        );

        follow(&harness);
        let (status, _, body) = call(&harness, get("/api/replication/role")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body,
            json!({
                "role": "follower",
                "instance_id": crate::replication::instance_id(),
                "leader_url": LEADER,
            }),
            "lag and the leader's id are absent until a poll has measured them"
        );

        let (status, _, body) = call(&harness, get_unauthenticated("/api/replication/role")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    }

    #[tokio::test]
    async fn an_authenticated_response_from_a_follower_carries_its_lag() {
        let harness = TestHarness::new();
        let (_, headers, _) = call(&harness, get("/api/replication/role")).await;
        assert!(
            headers.get(LAG_HEADER).is_none(),
            "not on a standalone instance"
        );

        follow(&harness);
        let (_, headers, _) = call(&harness, get("/api/replication/role")).await;
        assert!(
            headers.get(CONTACT_AGE_HEADER).is_none(),
            "not before the leader has been reached"
        );
        let replication = &harness.state().replication;
        let generation = replication.status().generation;
        let leader_id = uuid::Uuid::new_v4();
        let asked = chrono::Utc::now() - chrono::Duration::seconds(120);
        let reached = asked + chrono::Duration::seconds(3);
        replication.record_pass(generation, Some(leader_id), &[], asked, reached);
        replication.note_leader_contact(generation, reached);
        let (status, headers, body) = call(&harness, get("/api/replication/role")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers.get(LAG_HEADER).unwrap(), "3");
        let age: u64 = headers[CONTACT_AGE_HEADER]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            (117..=127).contains(&age),
            "the leader was last reached about 117 s ago: {age}"
        );
        assert_eq!(body["leader_instance_id"], leader_id.to_string());
        assert_eq!(body["lag_seconds"], 3, "the lag at the last catch-up");
        let body_age = body["leader_contact_age_seconds"].as_u64().unwrap();
        assert!(body_age.abs_diff(age) <= 1, "header {age}, body {body_age}");
        assert_eq!(body["lag_frames"], serde_json::Value::Null);

        let (status, headers, body) = call(
            &harness,
            post_json("/api/remember", &json!({"user_id": USER, "content": "x"})),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            headers.get(LAG_HEADER).unwrap(),
            "3",
            "refusals carry it too"
        );
        assert_eq!(body["leader"]["instance_id"], leader_id.to_string());
    }

    #[tokio::test]
    async fn health_ready_publishes_no_topology_on_a_follower() {
        let harness = TestHarness::new();
        let ready = |harness: &TestHarness| {
            let router = harness
                .router()
                .merge(crate::handlers::build_probe_routes(harness.state()));
            async move {
                let response = router
                    .oneshot(
                        HttpRequest::builder()
                            .uri("/health/ready")
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let status = response.status();
                let headers = response.headers().clone();
                let bytes = response.into_body().collect().await.unwrap().to_bytes();
                let mut body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                body.as_object_mut().unwrap().remove("timestamp");
                (status, headers, body)
            }
        };
        let standalone = ready(&harness).await;
        follow(&harness);
        let follower = ready(&harness).await;
        assert_eq!(follower.0, standalone.0);
        assert_eq!(follower.2, standalone.2, "the probe answers the same");
        assert!(follower.1.get(LAG_HEADER).is_none());
        let text = follower.2.to_string();
        assert!(
            !text.contains("127.0.0.1:9") && !text.contains("follow"),
            "{text}"
        );
    }

    type Socket = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    /// The router on a loopback socket: `oneshot` cannot upgrade a WebSocket.
    async fn serve(harness: &TestHarness) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/api/stream", listener.local_addr().unwrap());
        let app = harness.router();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, server)
    }

    fn stream_request(url: &str) -> tokio_tungstenite::tungstenite::handshake::client::Request {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        let mut request = url.into_client_request().unwrap();
        request
            .headers_mut()
            .insert("x-api-key", TEST_API_KEY.parse().unwrap());
        request
    }

    async fn next_frame(socket: &mut Socket) -> tokio_tungstenite::tungstenite::Message {
        use futures::StreamExt;
        use tokio_tungstenite::tungstenite::Message;
        loop {
            let frame = tokio::time::timeout(Duration::from_secs(30), socket.next())
                .await
                .expect("a frame within 30 s")
                .expect("the socket is open")
                .expect("a frame");
            if !matches!(frame, Message::Ping(_) | Message::Pong(_)) {
                return frame;
            }
        }
    }

    async fn exchange(socket: &mut Socket, message: serde_json::Value) -> serde_json::Value {
        use futures::SinkExt;
        use tokio_tungstenite::tungstenite::Message;
        socket
            .send(Message::Text(message.to_string().into()))
            .await
            .unwrap();
        match next_frame(socket).await {
            Message::Text(text) => serde_json::from_str(text.as_str()).unwrap(),
            other => panic!("expected a JSON frame, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_follower_refuses_the_stream_upgrade() {
        let harness = TestHarness::new();
        follow(&harness);
        let (url, server) = serve(&harness).await;
        match tokio_tungstenite::connect_async(stream_request(&url)).await {
            Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                assert!(response.headers().get(RETRY_AFTER).is_some());
            }
            other => panic!("the upgrade was not refused: {other:?}"),
        }
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_open_stream_stops_ingesting_once_its_instance_follows() {
        use tokio_tungstenite::tungstenite::Message;
        let harness = TestHarness::new();
        let (url, server) = serve(&harness).await;
        let (mut socket, _) = tokio_tungstenite::connect_async(stream_request(&url))
            .await
            .unwrap();
        let content = |text: &str| json!({"type": "content", "content": text, "importance": 0.9});

        let ack = exchange(&mut socket, json!({"user_id": USER})).await;
        assert_eq!(ack["type"], "ack", "{ack}");
        exchange(&mut socket, content("the pier lights come on at dusk")).await;
        let stored = exchange(&mut socket, json!({"type": "flush"})).await;
        assert_eq!(stored["memories_created"], 1, "{stored}");
        let memory = harness.state().get_user_memory(USER).unwrap();
        assert_eq!(memory.read().stored_count(), 1);

        follow(&harness);
        let refused = exchange(&mut socket, content("the pier lights failed")).await;
        assert_eq!(refused["type"], "error", "{refused}");
        assert_eq!(refused["code"], READ_ONLY_REPLICA, "{refused}");
        assert_eq!(refused["fatal"], true, "{refused}");
        match next_frame(&mut socket).await {
            Message::Close(Some(frame)) => {
                assert_eq!(u16::from(frame.code), 1013, "try again later");
                assert_eq!(frame.reason.as_str(), READ_ONLY_REPLICA);
            }
            other => panic!("expected the socket to close, got {other:?}"),
        }
        assert_eq!(
            memory.read().stored_count(),
            1,
            "nothing ingested after the follow"
        );
        server.abort();
    }

    /// The zenoh auto-topics hold one session for the life of the process, so a
    /// follow refuses its messages and keeps it: an unfollow lets it ingest again.
    #[tokio::test]
    async fn a_stream_session_outlives_a_follow() {
        use crate::streaming::{ExtractionResult, StreamMessage};
        let harness = TestHarness::new();
        let memory = harness.state().get_user_memory(USER).unwrap();
        let extractor = harness.manager.streaming_extractor();
        let session = extractor
            .create_session(serde_json::from_value(json!({"user_id": USER})).unwrap())
            .await
            .unwrap();
        let content = || -> StreamMessage {
            serde_json::from_value(
                json!({"type": "content", "content": "the ferry ramp was resurfaced", "importance": 0.9}),
            )
            .unwrap()
        };

        follow(&harness);
        for message in [content(), StreamMessage::Flush] {
            match extractor
                .process_message(&session, message, memory.clone(), harness.state())
                .await
            {
                ExtractionResult::Error { code, fatal, .. } => {
                    assert_eq!(code, READ_ONLY_REPLICA);
                    assert!(fatal);
                }
                other => panic!("a follower took a stream message: {other:?}"),
            }
        }
        assert_eq!(memory.read().stored_count(), 0);

        harness.state().replication.unfollow();
        let mut created = 0;
        for message in [content(), StreamMessage::Flush] {
            if let ExtractionResult::Extraction {
                memories_created, ..
            } = extractor
                .process_message(&session, message, memory.clone(), harness.state())
                .await
            {
                created += memories_created;
            }
        }
        assert_eq!(created, 1, "the same session ingests again");
        assert_eq!(memory.read().stored_count(), 1);
    }

    /// A feed request and a status poll from one follower, as the leader's admin
    /// status and role route then report them.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn a_leader_shows_who_follows_it_and_how_far_behind() {
        use crate::handlers::admin::{ADMIN_API_KEY_ENV, ADMIN_API_KEY_HEADER, ADMIN_ENV_LOCK};
        use crate::replication::{FOLLOWER_URL_HEADER, INSTANCE_ID_HEADER};
        const ADMIN_KEY: &str = "replica-test-admin-key";
        let _env = ADMIN_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(ADMIN_API_KEY_ENV, ADMIN_KEY);

        let harness = TestHarness::new();
        for content in [
            "the swing bridge opens on the hour",
            "the swing bridge was greased",
        ] {
            let (status, _, body) = call(
                &harness,
                post_json(
                    "/api/remember",
                    &json!({"user_id": USER, "content": content}),
                ),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
        }
        let follower = uuid::Uuid::new_v4();
        let peer: std::net::SocketAddr = "10.4.0.9:40112".parse().unwrap();
        let admin_get = |uri: &str, named: bool| {
            let mut request = get(uri);
            let headers = request.headers_mut();
            headers.insert(ADMIN_API_KEY_HEADER, ADMIN_KEY.parse().unwrap());
            if named {
                headers.insert(INSTANCE_ID_HEADER, follower.to_string().parse().unwrap());
                headers.insert(
                    FOLLOWER_URL_HEADER,
                    "https://replica.example:3443".parse().unwrap(),
                );
            }
            request
                .extensions_mut()
                .insert(axum::extract::ConnectInfo(peer));
            request
        };

        let records = |query: &str| {
            admin_get(
                &format!("/api/admin/replication/records?user_id={USER}&{query}"),
                true,
            )
        };
        let (status, _, page) = call(&harness, records("limit=1")).await;
        assert_eq!(status, StatusCode::OK, "{page}");
        let head = page["head_lsn"].as_u64().unwrap();
        assert!(head >= 2, "two remembers journaled: {page}");
        assert_eq!(page["more"], true, "{page}");
        let (status, _, body) =
            call(&harness, admin_get("/api/admin/replication/status", false)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body["followers"][0]["users"][USER],
            json!({"requested_lsn": 0, "leader_head_lsn": head, "lag_frames": head - 1}),
            "a page the limit cut to one frame leaves the rest unsent"
        );

        let (status, _, page) = call(&harness, records("after_lsn=0")).await;
        assert_eq!(status, StatusCode::OK, "{page}");
        assert_eq!(page["more"], false, "{page}");
        let head = page["head_lsn"].as_u64().unwrap();
        let (status, _, polled) =
            call(&harness, admin_get("/api/admin/replication/status", true)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            polled.get("followers").is_none(),
            "a follower's own poll is not sent the table: {polled}"
        );

        let (status, _, body) =
            call(&harness, admin_get("/api/admin/replication/status", false)).await;
        std::env::remove_var(ADMIN_API_KEY_ENV);
        assert_eq!(status, StatusCode::OK, "{body}");
        let followers = body["followers"].as_array().expect("followers");
        assert_eq!(
            followers.len(),
            1,
            "a poll that names no follower is not one: {body}"
        );
        let one = &followers[0];
        assert_eq!(one["instance_id"], follower.to_string());
        assert_eq!(one["url"], "https://replica.example:3443/");
        assert_eq!(one["remote_addr"], "10.4.0.9:40112");
        assert!(
            one["first_seen"].is_string() && one["last_poll"].is_string(),
            "{one}"
        );
        assert_eq!(
            one["users"][USER],
            json!({"requested_lsn": 1, "leader_head_lsn": head, "lag_frames": 0}),
            "it asked from lsn 1, after the frame at 0, and the reply carried the rest"
        );

        let (status, _, role) = call(&harness, get("/api/replication/role")).await;
        assert_eq!(status, StatusCode::OK, "{role}");
        assert_eq!(
            role,
            json!({
                "role": "leader",
                "instance_id": crate::replication::instance_id(),
                "followers_count": 1,
                "max_lag_frames": 0,
            })
        );
    }
}
