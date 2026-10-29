//! `/api/admin/replication/*` — the leader's feed and the follower's controls.
//!
//! Admin-key gated like the storage migration routes: the feed carries every
//! tenant's memories in the clear, which the per-tenant API key must not reach.

use std::net::SocketAddr;

use axum::{
    body::Bytes,
    extract::{ConnectInfo, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde::Deserialize;
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
pub async fn status(State(state): State<AppState>, headers: HeaderMap, peer: Peer) -> Response {
    if let Err(refusal) = require_admin_key(&headers) {
        return refusal;
    }
    let follower_poll = headers.contains_key(replication::INSTANCE_ID_HEADER);
    note_poll(&state, &headers, peer, None);
    let users = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || replication::user_logs(&state)).await
    };
    let users = match users {
        Ok(u) => u,
        Err(e) => return failed(anyhow::anyhow!("listing logs: {e}")),
    };
    let following = state.replication.status();
    let mut body = json!({
        "base_url": own_base_url(&state),
        "instance_id": replication::instance_id(),
        "users": users,
        "following": following.target.is_some().then_some(following),
    });
    if !follower_poll {
        body["followers"] = json!(state.replication.followers());
    }
    respond(StatusCode::OK, body)
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

pub async fn records(
    State(state): State<AppState>,
    headers: HeaderMap,
    peer: Peer,
    Query(q): Query<RecordsQuery>,
) -> Response {
    if let Err(refusal) = require_admin_key(&headers) {
        return refusal;
    }
    if let Err(e) = crate::validation::validate_user_id(&q.user_id) {
        note_poll(&state, &headers, peer, None);
        return bad_request(e);
    }
    if !replication::has_log(&state, &q.user_id) {
        note_poll(&state, &headers, peer, None);
        return respond(
            StatusCode::NOT_FOUND,
            json!({"error": "no_such_log", "detail": format!("no intent log for tenant '{}'", q.user_id)}),
        );
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
        Ok(Ok(page)) => respond(StatusCode::OK, json!(page)),
        Ok(Err(e)) => failed(e),
        Err(e) => failed(anyhow::anyhow!("feed task: {e}")),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FollowRequest {
    pub source: String,
    pub admin_key: String,
    /// The leader's API key; this server's own when absent.
    #[serde(default)]
    pub api_key: Option<String>,
}

/// Start tailing `source`. Refused when the source is this instance, by its address
/// or by the instance id its status reports, or when the source reports following
/// this instance: either would echo frames forever.
pub async fn follow(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(refusal) = require_admin_key(&headers) {
        return refusal;
    }
    let req: FollowRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return bad_request(e),
    };
    let source = match replication::normalize_source(&req.source) {
        Ok(s) => s,
        Err(e) => return bad_request(e),
    };
    let me = own_base_url(&state);
    let cfg = state.server_config();
    if source == me || replication::is_same_instance(&source, &cfg.host, cfg.port) {
        return would_loop(format!("{source} is this instance"));
    }
    let api_key = match req
        .api_key
        .clone()
        .filter(|k| !k.trim().is_empty())
        .or_else(|| replication::default_api_key_for(&source))
    {
        Some(k) => k,
        None => {
            return bad_request(
                "pass the leader's api_key: this server's own key is only offered to a leader on this host",
            )
        }
    };
    let probe = match replication::leader_client(replication::PROBE_TIMEOUT) {
        Ok(c) => c,
        Err(e) => return failed(e),
    };
    let leader = match replication::probe_leader(probe, &source, &req.admin_key, &api_key).await {
        Ok(leader) => leader,
        Err(e) => {
            return respond(
                StatusCode::BAD_GATEWAY,
                json!({"error": "source_unreachable", "detail": format!("{e:#}")}),
            );
        }
    };
    match replication::follow_loop(&leader, replication::instance_id(), &me) {
        Some(replication::FollowLoop::Itself) => {
            return would_loop(format!("{source} is this instance"));
        }
        Some(replication::FollowLoop::Mutual) => {
            return would_loop(format!("{source} already follows {me}; frames would echo"));
        }
        None => {}
    }
    match state
        .replication
        .follow(&state.base_path, &source, &req.admin_key, Some(&api_key))
    {
        Ok(status) => respond(StatusCode::OK, json!({"following": status})),
        Err(e) => failed(e),
    }
}

pub async fn unfollow(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(refusal) = require_admin_key(&headers) {
        return refusal;
    }
    let status = state.replication.unfollow();
    respond(
        StatusCode::OK,
        json!({"following": null, "users": status.users}),
    )
}
