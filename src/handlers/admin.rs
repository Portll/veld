//! Admin endpoints for operational recovery.
//!
//! These endpoints are gated on a SEPARATE env var (`VELD_ADMIN_API_KEY`) from
//! the regular API key surface (`VELD_API_KEYS`/`VELD_DEV_API_KEY`). The
//! separation is intentional: a leaked regular key must NOT grant rate-limit
//! reset capability, since reset is a force-multiplier for brute-force /
//! credential-stuffing attacks against any other endpoint.
//!
//! Authentication uses `subtle::ConstantTimeEq` for constant-time comparison
//! (independent of Veld's own `constant_time_compare` helper, per PR2 spec).

use std::net::SocketAddr;

use axum::{
    extract::{ConnectInfo, Extension},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use subtle::ConstantTimeEq;

use crate::config::env_var;
use crate::rate_limit_governance::ResetHandle;

/// Header name carrying the admin API key.
pub const ADMIN_API_KEY_HEADER: &str = "X-Admin-API-Key";

/// Env var holding the admin key. Distinct from `VELD_API_KEYS` /
/// `VELD_DEV_API_KEY` / `SHODH_API_KEYS` — admin actions are NOT authorized by
/// regular API keys.
pub const ADMIN_API_KEY_ENV: &str = "VELD_ADMIN_API_KEY";

/// `POST /api/admin/reset-rate-limit`
///
/// Recovers from a stuck rate-limiter bucket by atomically swapping the
/// underlying `Arc<RateLimiter>` for a fresh one. After this call returns 204,
/// every peer's burst capacity is restored to full.
///
/// ## Auth
///
/// - `VELD_ADMIN_API_KEY` env var unset → 503 `{"error":"admin_key_not_configured"}`
/// - `X-Admin-API-Key` header missing or wrong → 401 `{"error":"unauthorized"}`
/// - Valid → 204 No Content (with WARN log including caller IP and key prefix)
///
/// ## Why a separate key (and not regular API key)
///
/// If an attacker brute-forces a regular API key and gains read/write access
/// to memories, they should NOT also be able to reset the rate limit and
/// re-attempt brute-force on a freshly-reset bucket. Splitting the keys means
/// a compromised regular key still hits the GCRA wall. The admin key has a
/// strictly smaller blast radius — it can only reset the rate limiter, nothing
/// else.
pub async fn reset_rate_limit(
    handle: Option<Extension<ResetHandle>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Response {
    // 0) Rate limiting must be enabled — otherwise there is no Extension to
    //    extract and nothing to reset.
    let Some(Extension(handle)) = handle else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "rate_limiting_disabled"})),
        )
            .into_response();
    };

    // 1) Configured?
    let configured_key = match env_var(ADMIN_API_KEY_ENV, "SHODH_ADMIN_API_KEY") {
        Ok(k) if !k.trim().is_empty() => k,
        _ => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "admin_key_not_configured"})),
            )
                .into_response();
        }
    };

    // 2) Header present?
    let provided = match headers.get(ADMIN_API_KEY_HEADER).and_then(|v| v.to_str().ok()) {
        Some(s) if !s.is_empty() => s,
        _ => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "unauthorized"})),
            )
                .into_response();
        }
    };

    // 3) Constant-time compare via `subtle`. Equal-length normalisation: if
    //    lengths differ, force a no-match WITHOUT short-circuiting (the
    //    `bool::from(... .ct_eq(...))` would already early-return on length
    //    mismatch in `subtle`'s slice impl, so we add a length check + dummy
    //    compare to keep timing flat).
    let configured_bytes = configured_key.as_bytes();
    let provided_bytes = provided.as_bytes();
    let len_match = configured_bytes.len() == provided_bytes.len();
    // Always compare a fixed-length buffer so the compare cost doesn't leak
    // length information. If lengths differ, compare provided to itself
    // (always equal), then explicitly fail via `len_match`.
    let bytes_eq = if len_match {
        bool::from(configured_bytes.ct_eq(provided_bytes))
    } else {
        let _ = bool::from(provided_bytes.ct_eq(provided_bytes));
        false
    };

    if !(len_match && bytes_eq) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    // 4) Authenticated — perform the reset and audit-log.
    handle.reset();
    let key_prefix: String = configured_key.chars().take(8).collect();
    tracing::warn!(
        caller_ip = %peer.ip(),
        key_prefix = %key_prefix,
        "rate-limit state reset by admin endpoint"
    );

    StatusCode::NO_CONTENT.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::ConnectInfo;
    use axum::http::{HeaderName, HeaderValue, Method, Request};
    use axum::routing::post;
    use axum::Router;
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::Mutex;
    use tower::ServiceExt;

    /// Process-global lock so tests don't trip over each other's env mutation.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn build_app(handle: ResetHandle) -> Router {
        Router::new()
            .route("/api/admin/reset-rate-limit", post(reset_rate_limit))
            .layer(Extension(handle))
    }

    fn req(key: Option<&str>) -> Request<Body> {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri("/api/admin/reset-rate-limit");
        if let Some(k) = key {
            builder = builder.header(
                HeaderName::from_static("x-admin-api-key"),
                HeaderValue::from_str(k).unwrap(),
            );
        }
        let mut r = builder.body(Body::empty()).unwrap();
        r.extensions_mut().insert(ConnectInfo(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            54321,
        )));
        r
    }

    #[tokio::test]
    async fn returns_503_when_admin_key_not_configured() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var(ADMIN_API_KEY_ENV);
        std::env::remove_var("SHODH_ADMIN_API_KEY");

        let app = build_app(ResetHandle::new(10, 5));
        let resp = app.oneshot(req(Some("anything"))).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn returns_401_when_header_missing() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(ADMIN_API_KEY_ENV, "test-admin-secret-12345");

        let app = build_app(ResetHandle::new(10, 5));
        let resp = app.oneshot(req(None)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        std::env::remove_var(ADMIN_API_KEY_ENV);
    }

    #[tokio::test]
    async fn returns_401_when_header_wrong() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(ADMIN_API_KEY_ENV, "test-admin-secret-12345");

        let app = build_app(ResetHandle::new(10, 5));
        let resp = app.oneshot(req(Some("not-the-secret"))).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        std::env::remove_var(ADMIN_API_KEY_ENV);
    }

    #[tokio::test]
    async fn returns_204_with_valid_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(ADMIN_API_KEY_ENV, "test-admin-secret-12345");

        let handle = ResetHandle::new(10, 5);
        let app = build_app(handle.clone());
        let resp = app
            .oneshot(req(Some("test-admin-secret-12345")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        std::env::remove_var(ADMIN_API_KEY_ENV);
    }

    #[tokio::test]
    async fn returns_401_for_length_mismatch() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(ADMIN_API_KEY_ENV, "test-admin-secret-12345");

        let app = build_app(ResetHandle::new(10, 5));
        // Shorter than configured.
        let resp = app.oneshot(req(Some("short"))).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        std::env::remove_var(ADMIN_API_KEY_ENV);
    }
}
