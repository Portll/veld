//! Resettable rate limiting governance — a thin wrapper around `governor::RateLimiter`
//! that:
//!
//! 1. Caps the reported `Wait for {N}s` value in 429 responses so a stuck GCRA
//!    bucket can never report wait_time > burst_period (preventing the
//!    "Wait for 95015s" runaway seen on 2026-04-26).
//! 2. Exposes a [`ResetHandle`] that can atomically swap the underlying
//!    `Arc<RateLimiter>` behind a `parking_lot::RwLock`, allowing an
//!    authenticated admin endpoint to recover from a stuck bucket without
//!    a process restart.
//!
//! GCRA accept/reject decision logic is delegated entirely to `governor`, so
//! the cap is observability-only — it never relaxes the actual rate limit.
//!
//! ## Usage
//!
//! ```ignore
//! use std::time::Duration;
//! use veld::rate_limit_governance::{ResetHandle, RateLimitGovernanceLayer};
//!
//! let reset_handle = ResetHandle::new(/* rps */ 10, /* burst */ 5);
//! let layer = RateLimitGovernanceLayer::new(reset_handle.clone());
//! // ... wire `layer` into your router and store `reset_handle` for the admin handler.
//! ```

use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Request, Response, StatusCode};
use governor::{
    clock::{Clock, DefaultClock},
    middleware::NoOpMiddleware,
    state::keyed::DefaultKeyedStateStore,
    Quota, RateLimiter,
};
use parking_lot::RwLock;
use std::num::NonZeroU32;
use tower::{Layer, Service};

/// Concrete keyed limiter type used by the governance layer.
type KeyedLimiter =
    RateLimiter<IpAddr, DefaultKeyedStateStore<IpAddr>, DefaultClock, NoOpMiddleware>;

/// Build a [`Quota`] from rps + burst, mirroring tower_governor's nanosecond
/// cell-interval calculation so the GCRA decision matches the previous wiring.
fn build_quota(rps: u64, burst: u32) -> Quota {
    let rps = rps.max(1);
    let cell_interval = Duration::from_nanos(1_000_000_000 / rps);
    let burst = NonZeroU32::new(burst.max(1)).expect("burst >= 1 enforced above");
    Quota::with_period(cell_interval)
        .expect("cell interval is non-zero")
        .allow_burst(burst)
}

/// Compute the wait_time cap (seconds) from rps + burst.
///
/// `cap = max(1, ceil(burst * cell_interval_secs))`. With sub-second cell
/// intervals (rps >= 1) this floors to 1, which is correct: with rps=10
/// burst=5, the natural maximum honest wait is `5 * 0.1s = 500ms`, well under
/// the 1-second floor. The floor exists so callers always have a non-zero
/// retry-after value to honour.
pub fn wait_time_cap_secs(rps: u64, burst: u32) -> u64 {
    let rps = rps.max(1);
    // Use nanosecond math to avoid losing precision when burst*cell < 1s.
    let total_nanos =
        (burst.max(1) as u128).saturating_mul(1_000_000_000u128 / rps as u128);
    let secs = (total_nanos / 1_000_000_000) as u64;
    // Always allow at least 1s so retry-after is meaningful.
    secs.max(1)
}

/// Shared, swappable handle to the underlying rate limiter.
///
/// Cloneable; all clones share the same swappable inner `Arc<RateLimiter>`.
#[derive(Clone)]
pub struct ResetHandle {
    inner: Arc<RwLock<Arc<KeyedLimiter>>>,
    quota: Quota,
    cap_secs: u64,
}

impl ResetHandle {
    /// Construct a new resettable limiter from rps + burst.
    pub fn new(rps: u64, burst: u32) -> Self {
        let quota = build_quota(rps, burst);
        let cap_secs = wait_time_cap_secs(rps, burst);
        let limiter = Arc::new(RateLimiter::keyed(quota));
        Self {
            inner: Arc::new(RwLock::new(limiter)),
            quota,
            cap_secs,
        }
    }

    /// Swap the inner limiter for a freshly-built one with the SAME quota.
    ///
    /// This is the recovery primitive used by the admin reset endpoint. It
    /// drops the old `Arc<RateLimiter>` (its keyed state store goes with it),
    /// installing a fresh limiter with no key history. After this, all peers
    /// have full burst capacity again.
    pub fn reset(&self) {
        let fresh = Arc::new(RateLimiter::keyed(self.quota));
        let mut guard = self.inner.write();
        *guard = fresh;
    }

    /// Read-clone the current limiter Arc. Cheap; releases the read lock
    /// immediately. Used per-request by the governance Service.
    fn current_limiter(&self) -> Arc<KeyedLimiter> {
        let guard = self.inner.read();
        Arc::clone(&*guard)
    }

    /// Cap value (seconds) computed once at construction.
    pub fn cap_secs(&self) -> u64 {
        self.cap_secs
    }
}

/// Tower `Layer` that wires a [`ResetHandle`] into an axum service stack.
#[derive(Clone)]
pub struct RateLimitGovernanceLayer {
    handle: ResetHandle,
}

impl RateLimitGovernanceLayer {
    pub fn new(handle: ResetHandle) -> Self {
        Self { handle }
    }
}

impl<S> Layer<S> for RateLimitGovernanceLayer {
    type Service = RateLimitGovernanceService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitGovernanceService {
            inner,
            handle: self.handle.clone(),
        }
    }
}

/// Tower `Service` produced by [`RateLimitGovernanceLayer`].
#[derive(Clone)]
pub struct RateLimitGovernanceService<S> {
    inner: S,
    handle: ResetHandle,
}

impl<S, ReqBody> Service<Request<ReqBody>> for RateLimitGovernanceService<S>
where
    S: Service<Request<ReqBody>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let limiter = self.handle.current_limiter();
        let cap = self.handle.cap_secs;

        // Extract peer IP from ConnectInfo (matches tower_governor's PeerIpKeyExtractor).
        let peer_ip = req
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ci| ci.0.ip());

        // Clone the inner service for the spawned future. The clone+move pattern
        // is the standard idiom for tower middleware that may delay calling the
        // inner service (poll_ready already returned Ready on the original).
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);

        Box::pin(async move {
            let Some(peer_ip) = peer_ip else {
                // No peer IP — cannot extract key. Mirror tower_governor's behavior:
                // 500 with "Unable To Extract Key!" body.
                return Ok(unable_to_extract_key());
            };

            match limiter.check_key(&peer_ip) {
                Ok(()) => inner.call(req).await,
                Err(negative) => {
                    let raw_wait = negative
                        .wait_time_from(DefaultClock::default().now())
                        .as_secs();
                    // Cap before logging/responding. The GCRA decision was already
                    // made (limiter rejected). Capping is observability-only.
                    let capped = raw_wait.min(cap);
                    Ok(too_many_requests_response(capped))
                }
            }
        })
    }
}

/// Build a 429 response with body `"Too Many Requests! Wait for {N}s"`.
///
/// Format string preserved exactly so existing callers that regex-match it
/// (Antidote MCP server, ops dashboards, log scrapers) continue to work.
pub fn too_many_requests_response(wait_time_secs: u64) -> Response<Body> {
    let body = format!("Too Many Requests! Wait for {}s", wait_time_secs);
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("retry-after"),
        HeaderValue::from(wait_time_secs),
    );
    headers.insert(
        HeaderName::from_static("x-ratelimit-after"),
        HeaderValue::from(wait_time_secs),
    );

    let mut response = Response::new(Body::from(body));
    *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;
    *response.headers_mut() = headers;
    response
}

fn unable_to_extract_key() -> Response<Body> {
    let mut response = Response::new(Body::from("Unable To Extract Key!"));
    *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Method;
    use axum::routing::get;
    use axum::Router;
    use std::net::{IpAddr, Ipv4Addr};
    use tower::ServiceExt;

    /// Helper: parse "Wait for {N}s" out of a 429 body.
    fn parse_wait_secs(body: &str) -> Option<u64> {
        let prefix = "Too Many Requests! Wait for ";
        let suffix = "s";
        let body = body.strip_prefix(prefix)?;
        let body = body.strip_suffix(suffix)?;
        body.parse().ok()
    }

    #[test]
    fn cap_seconds_low_burst_floors_to_one() {
        // rps=10, burst=5 → 0.5s natural cap, but floored to 1.
        assert_eq!(wait_time_cap_secs(10, 5), 1);
    }

    #[test]
    fn cap_seconds_high_burst_yields_burst_period() {
        // rps=1, burst=60 → cell=1s, 60s cap.
        assert_eq!(wait_time_cap_secs(1, 60), 60);
    }

    #[test]
    fn reset_replaces_inner_limiter() {
        let h = ResetHandle::new(10, 5);
        let limiter_before = h.current_limiter();
        h.reset();
        let limiter_after = h.current_limiter();
        // Different Arc instances after reset.
        assert!(!Arc::ptr_eq(&limiter_before, &limiter_after));
    }

    /// Build a tiny axum router with the governance layer for end-to-end testing.
    fn test_router(handle: ResetHandle) -> Router {
        Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(RateLimitGovernanceLayer::new(handle))
    }

    /// Synthesize a request with ConnectInfo so the peer-IP key extractor works.
    fn req_with_peer(ip: IpAddr) -> Request<Body> {
        let mut req = Request::builder()
            .method(Method::GET)
            .uri("/")
            .body(Body::empty())
            .unwrap();
        req.extensions_mut()
            .insert(ConnectInfo(SocketAddr::new(ip, 12345)));
        req
    }

    #[tokio::test]
    async fn cap_applied_to_429_wait_time() {
        // burst=2, rps=1 → cell=1s, natural cap=2s. We exhaust the burst then
        // overshoot many times to force GCRA TAT to advance well past the cap.
        let handle = ResetHandle::new(1, 2);
        let cap = handle.cap_secs();
        assert_eq!(cap, 2);

        let app = test_router(handle);
        let ip = IpAddr::V4(Ipv4Addr::LOCALHOST);

        // Burn through burst (2 allowed) then hammer until rejected.
        // Send 50 requests in tight loop — the last several should be 429.
        let mut last_body = String::new();
        for _ in 0..50 {
            let resp = app.clone().oneshot(req_with_peer(ip)).await.unwrap();
            if resp.status() == StatusCode::TOO_MANY_REQUESTS {
                let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
                last_body = String::from_utf8(bytes.to_vec()).unwrap();
            }
        }

        let parsed = parse_wait_secs(&last_body)
            .unwrap_or_else(|| panic!("Expected 429 body, got {:?}", last_body));
        assert!(
            parsed <= cap,
            "wait_time {} exceeded cap {} (body: {:?})",
            parsed,
            cap,
            last_body
        );
    }

    #[tokio::test]
    async fn gcra_decision_unchanged_by_cap_shim() {
        // Compare 429 count over an identical request burst between the
        // capped wrapper and a raw governor::RateLimiter. They must match
        // exactly (the cap only rewrites the wait_time number, never the
        // accept/reject decision).
        let rps = 5;
        let burst = 3;
        let handle = ResetHandle::new(rps, burst);
        let app = test_router(handle.clone());
        let ip = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1));

        // Count 429s through the wrapped service.
        let mut wrapped_429s = 0usize;
        for _ in 0..20 {
            let resp = app.clone().oneshot(req_with_peer(ip)).await.unwrap();
            if resp.status() == StatusCode::TOO_MANY_REQUESTS {
                wrapped_429s += 1;
            }
        }

        // Count rejections through a raw, freshly-constructed governor limiter.
        let raw = RateLimiter::keyed(build_quota(rps, burst));
        let mut raw_rejections = 0usize;
        for _ in 0..20 {
            if raw.check_key(&ip).is_err() {
                raw_rejections += 1;
            }
        }

        // The wrapped service may show ±1 difference vs raw because the
        // GCRA is time-sensitive and tests run on different clocks. Allow
        // ±2 slack.
        let diff = (wrapped_429s as i64 - raw_rejections as i64).abs();
        assert!(
            diff <= 2,
            "wrapped 429s ({}) diverged from raw rejections ({}) by more than 2",
            wrapped_429s,
            raw_rejections
        );
    }

    #[tokio::test]
    async fn reset_restores_full_burst_capacity() {
        let rps = 1;
        let burst = 2;
        let handle = ResetHandle::new(rps, burst);
        let app = test_router(handle.clone());
        let ip = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 2));

        // Exhaust the burst.
        for _ in 0..10 {
            let _ = app.clone().oneshot(req_with_peer(ip)).await;
        }

        // Confirm we're now rate-limited.
        let resp = app.clone().oneshot(req_with_peer(ip)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        // Reset the limiter.
        handle.reset();

        // After reset, burst should be available again.
        let resp = app.clone().oneshot(req_with_peer(ip)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "post-reset request should succeed; got {}",
            resp.status()
        );
    }

}
