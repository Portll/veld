//! `/api/admin/replication/*` — the leader's feed and the follower's controls.
//!
//! Admin-key gated like the storage migration routes: the feed carries every
//! tenant's memories in the clear, which the per-tenant API key must not reach.
//!
//! Each route answers its typed body with 200, or a hand-built `{"error", "detail"}`
//! refusal; the monitor reads those keys, which [`crate::errors::AppError`]'s body lacks.

use std::collections::BTreeMap;
use std::net::SocketAddr;

use axum::{
    body::Bytes,
    extract::{ConnectInfo, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::admin::require_admin_key;
use super::router::AppState;
use crate::replication;

fn respond(status: StatusCode, body: serde_json::Value) -> Response {
    (status, Json(body)).into_response()
}

fn bad_request(detail: impl std::fmt::Display) -> Response {
    respond(
        StatusCode::BAD_REQUEST,
        json!({"error": "bad_request", "detail": detail.to_string()}),
    )
}

fn failed(e: anyhow::Error) -> Response {
    respond(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": "replication_failed", "detail": format!("{e:#}")}),
    )
}

fn would_loop(detail: String) -> Response {
    respond(
        StatusCode::CONFLICT,
        json!({"error": "would_loop", "detail": detail}),
    )
}

fn own_base_url(state: &AppState) -> String {
    let cfg = state.server_config();
    let host = match cfg.host.as_str() {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
        other => other,
    };
    format!("http://{host}:{}", cfg.port)
}

/// `GET /api/admin/replication/status`: both roles of this instance.
#[derive(Debug, Serialize)]
pub struct ReplicationStatusResponse {
    /// This instance's own address, with a wildcard bind host read as `127.0.0.1`.
    pub base_url: String,
    pub instance_id: uuid::Uuid,
    /// The tenants this instance can feed.
    pub users: Vec<replication::UserLogInfo>,
    /// What this instance tails; null when it follows no leader.
    pub following: Option<replication::FollowStatus>,
    /// Who polled this instance lately. Absent from a follower's own poll, which names
    /// itself and reads none of them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub followers: Option<Vec<replication::FollowerView>>,
}

/// `POST /api/admin/replication/follow`: the follow state once tailing has started.
#[derive(Debug, Serialize)]
pub struct FollowResponse {
    pub following: replication::FollowStatus,
}

/// `POST /api/admin/replication/unfollow`: `following` is always null; `users` keeps each
/// tenant's progress as it stood when tailing stopped.
#[derive(Debug, Serialize)]
pub struct UnfollowResponse {
    pub following: Option<replication::FollowStatus>,
    pub users: BTreeMap<String, replication::UserProgress>,
}

/// The peer of a poll as the socket gives it; absent only off a socket.
type Peer = Option<Extension<ConnectInfo<SocketAddr>>>;

/// Record a poll from a follower that names itself, once the admin key has passed.
fn note_poll(
    state: &AppState,
    headers: &HeaderMap,
    peer: Peer,
    tenant: Option<(&str, replication::FollowerTenant)>,
) {
    if let Some(identity) = replication::PollIdentity::from_headers(headers) {
        let remote_addr = peer.map(|Extension(ConnectInfo(addr))| addr);
        state.replication.note_poll(identity, remote_addr, tenant);
    }
}

/// Both roles in one reply: `users` is what this instance can feed, `followers` who
/// polled it lately, `following` what it tails. A follower's poll, which names
/// itself, gets no `followers`: it reads none of them, and asks every interval.
#[allow(
    clippy::result_large_err,
    reason = "the refusal is the Response the route sends, built once per request"
)]
pub async fn status(
    State(state): State<AppState>,
    headers: HeaderMap,
    peer: Peer,
) -> Result<(StatusCode, Json<ReplicationStatusResponse>), Response> {
    require_admin_key(&headers)?;
    let follower_poll = headers.contains_key(replication::INSTANCE_ID_HEADER);
    note_poll(&state, &headers, peer, None);
    let users = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || replication::user_logs(&state)).await
    };
    let users = users.map_err(|e| failed(anyhow::anyhow!("listing logs: {e}")))?;
    let following = state.replication.status();
    let body = ReplicationStatusResponse {
        base_url: own_base_url(&state),
        instance_id: replication::instance_id(),
        users,
        following: following.target.is_some().then_some(following),
        followers: (!follower_poll).then(|| state.replication.followers()),
    };
    Ok((StatusCode::OK, Json(body)))
}

#[derive(Debug, Deserialize)]
pub struct RecordsQuery {
    pub user_id: String,
    #[serde(default, deserialize_with = "empty_as_none")]
    pub after_lsn: Option<u64>,
    pub limit: Option<usize>,
}

fn empty_as_none<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    let raw: Option<String> = Option::deserialize(d)?;
    match raw.as_deref().map(str::trim) {
        None | Some("") => Ok(None),
        Some(s) => s.parse().map(Some).map_err(serde::de::Error::custom),
    }
}

#[allow(
    clippy::result_large_err,
    reason = "the refusal is the Response the route sends, built once per request"
)]
pub async fn records(
    State(state): State<AppState>,
    headers: HeaderMap,
    peer: Peer,
    Query(q): Query<RecordsQuery>,
) -> Result<(StatusCode, Json<replication::FeedPage>), Response> {
    require_admin_key(&headers)?;
    if let Err(e) = crate::validation::validate_user_id(&q.user_id) {
        note_poll(&state, &headers, peer, None);
        return Err(bad_request(e));
    }
    if !replication::has_log(&state, &q.user_id) {
        note_poll(&state, &headers, peer, None);
        return Err(respond(
            StatusCode::NOT_FOUND,
            json!({"error": "no_such_log", "detail": format!("no intent log for tenant '{}'", q.user_id)}),
        ));
    }
    state.note_feed_cursor(&q.user_id, q.after_lsn);
    let limit = q.limit.unwrap_or(replication::FEED_LIMIT).clamp(1, 5000);
    let after_lsn = q.after_lsn;
    let page = {
        let state = state.clone();
        let user_id = q.user_id.clone();
        tokio::task::spawn_blocking(move || replication::feed(&state, &user_id, after_lsn, limit))
            .await
    };
    let lag = match &page {
        Ok(Ok(page)) => Some(replication::FollowerTenant::new(
            after_lsn,
            page.head_lsn,
            page.records.len() as u64,
        )),
        _ => None,
    };
    note_poll(
        &state,
        &headers,
        peer,
        lag.map(|lag| (q.user_id.as_str(), lag)),
    );
    match page {
        Ok(Ok(page)) => Ok((StatusCode::OK, Json(page))),
        Ok(Err(e)) => Err(failed(e)),
        Err(e) => Err(failed(anyhow::anyhow!("feed task: {e}"))),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FollowRequest {
    pub source: String,
    pub admin_key: String,
    /// The leader's API key; this server's own when absent.
    #[serde(default)]
    pub api_key: Option<String>,
}

impl std::fmt::Debug for FollowRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FollowRequest")
            .field("source", &self.source)
            .field("admin_key", &replication::REDACTED)
            .field(
                "api_key",
                &self.api_key.as_ref().map(|_| replication::REDACTED),
            )
            .finish()
    }
}

/// Start tailing `source`. Refused when the source is this instance, by its address
/// or by the instance id its status reports, or when the source reports following
/// this instance: either would echo frames forever.
#[allow(
    clippy::result_large_err,
    reason = "the refusal is the Response the route sends, built once per request"
)]
pub async fn follow(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, Json<FollowResponse>), Response> {
    require_admin_key(&headers)?;
    let req: FollowRequest = serde_json::from_slice(&body).map_err(bad_request)?;
    let source = replication::normalize_source(&req.source).map_err(bad_request)?;
    let me = own_base_url(&state);
    let cfg = state.server_config();
    if source == me || replication::is_same_instance(&source, &cfg.host, cfg.port) {
        return Err(would_loop(format!("{source} is this instance")));
    }
    let api_key = match req
        .api_key
        .clone()
        .filter(|k| !k.trim().is_empty())
        .or_else(|| replication::default_api_key_for(&source))
    {
        Some(k) => k,
        None => {
            return Err(bad_request(
                "pass the leader's api_key: this server's own key is only offered to a leader on this host",
            ))
        }
    };
    let probe = replication::leader_client(replication::PROBE_TIMEOUT).map_err(failed)?;
    let leader = match replication::probe_leader(probe, &source, &req.admin_key, &api_key).await {
        Ok(leader) => leader,
        Err(e) => {
            return Err(respond(
                StatusCode::BAD_GATEWAY,
                json!({"error": "source_unreachable", "detail": format!("{e:#}")}),
            ));
        }
    };
    match replication::follow_loop(&leader, replication::instance_id(), &me) {
        Some(replication::FollowLoop::Itself) => {
            return Err(would_loop(format!("{source} is this instance")));
        }
        Some(replication::FollowLoop::Mutual) => {
            return Err(would_loop(format!(
                "{source} already follows {me}; frames would echo"
            )));
        }
        None => {}
    }
    match state
        .replication
        .follow(&state.base_path, &source, &req.admin_key, Some(&api_key))
    {
        Ok(following) => Ok((StatusCode::OK, Json(FollowResponse { following }))),
        Err(e) => Err(failed(e)),
    }
}

/// Stop tailing the leader; the reply carries each tenant's progress as it stood.
#[allow(
    clippy::result_large_err,
    reason = "the refusal is the Response the route sends, built once per request"
)]
pub async fn unfollow(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<(StatusCode, Json<UnfollowResponse>), Response> {
    require_admin_key(&headers)?;
    let status = state.replication.unfollow();
    Ok((
        StatusCode::OK,
        Json(UnfollowResponse {
            following: None,
            users: status.users,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each typed reply serialises to the body its handler once built with `json!`.

    #[test]
    fn status_answers_the_keys_and_values_it_built_by_hand() {
        let users = vec![replication::UserLogInfo {
            user_id: "u1".to_string(),
            log_bytes: 42,
        }];
        let reply = |followers| ReplicationStatusResponse {
            base_url: "http://127.0.0.1:3030".to_string(),
            instance_id: uuid::Uuid::nil(),
            users: users.clone(),
            following: None,
            followers,
        };
        let by_hand = json!({
            "base_url": "http://127.0.0.1:3030",
            "instance_id": uuid::Uuid::nil(),
            "users": [{"user_id": "u1", "log_bytes": 42}],
            "following": null,
        });
        let mut leader_view = by_hand.clone();
        leader_view["followers"] = json!([]);
        assert_eq!(json!(reply(Some(Vec::new()))), leader_view);
        // A follower's own poll is sent no `followers` key at all.
        assert_eq!(json!(reply(None)), by_hand);
    }

    #[test]
    fn follow_and_unfollow_answer_the_keys_and_values_they_built_by_hand() {
        let following = replication::FollowStatus::default();
        assert_eq!(
            json!(FollowResponse {
                following: following.clone()
            }),
            json!({ "following": following }),
        );
        let users = BTreeMap::from([("u1".to_string(), replication::UserProgress::default())]);
        assert_eq!(
            json!(UnfollowResponse {
                following: None,
                users: users.clone(),
            }),
            json!({"following": null, "users": users}),
        );
    }

    #[tokio::test]
    async fn a_refusal_keeps_its_status_and_error_detail_body() {
        let refusal = would_loop("http://127.0.0.1:3030 is this instance".to_string());
        assert_eq!(refusal.status(), StatusCode::CONFLICT);
        let bytes = axum::body::to_bytes(refusal.into_body(), usize::MAX)
            .await
            .expect("a buffered body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("a JSON body");
        assert_eq!(
            body,
            json!({"error": "would_loop", "detail": "http://127.0.0.1:3030 is this instance"}),
        );
    }
}
