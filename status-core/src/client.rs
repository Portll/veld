//! HTTP polling client that keeps a [`StatusSnapshot`] fresh and an SSE subscriber
//! that appends to its activity tail.
//!
//! Spawn one [`StatusClient`] per process; it owns its own background tasks and
//! exposes a cheap [`Arc<RwLock<StatusSnapshot>>`] for the UI to read from.

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use parking_lot::RwLock;
use reqwest::{Client, StatusCode};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::dto::{
    ContextSessionDto, GraphStatsDto, HealthDto, MemoryStatsDto, TodoStatsDto, TodoStatsWrapper,
};
use crate::snapshot::{
    ContextSession, GraphStats, ReachState, ServerHealth, StatusSnapshot, TierStats, TodoStats,
};
use crate::sse::run_sse_loop;
use crate::{Result, StatusError};

/// Construction parameters for [`StatusClient::spawn`].
#[derive(Debug, Clone)]
pub struct StatusClientConfig {
    /// Base server URL, e.g. `http://127.0.0.1:3030`. Trailing slashes are stripped.
    pub base_url: String,
    /// API key passed as `X-API-Key` for protected endpoints. May be empty for
    /// health-only operation.
    pub api_key: String,
    /// Veld user_id to scope per-user stats against.
    pub user_id: String,
    /// How often to re-poll the slow endpoints (stats, graph, todos). 1–5s is reasonable.
    pub refresh_interval: Duration,
    /// Per-request HTTP timeout.
    pub request_timeout: Duration,
}

impl StatusClientConfig {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        user_id: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            user_id: user_id.into(),
            refresh_interval: Duration::from_secs(2),
            request_timeout: Duration::from_secs(5),
        }
    }

    pub fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.refresh_interval = interval;
        self
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

/// Background poller + SSE subscriber. Drop the handle to stop both tasks
/// (see the `Drop` impl below — it aborts both, so dropping a removed fleet
/// member stops it polling a dead instance).
pub struct StatusClient {
    snapshot: Arc<RwLock<StatusSnapshot>>,
    poll_task: JoinHandle<()>,
    sse_task: JoinHandle<()>,
}

impl StatusClient {
    /// Spawn the background tasks. Returns immediately; the snapshot is populated
    /// asynchronously as the first probe completes.
    pub fn spawn(config: StatusClientConfig) -> Result<Self> {
        let base = normalize_base_url(&config.base_url)?;

        let initial = StatusSnapshot {
            base_url: base.clone(),
            user_id: config.user_id.clone(),
            ..StatusSnapshot::default()
        };
        let snapshot = Arc::new(RwLock::new(initial));

        let http = http_client(config.request_timeout)?;

        let poll_task = tokio::spawn(run_poll_loop(
            http.clone(),
            base.clone(),
            config.api_key.clone(),
            config.user_id.clone(),
            config.refresh_interval,
            Arc::clone(&snapshot),
        ));

        let sse_task = tokio::spawn(run_sse_loop(
            http,
            base,
            config.api_key,
            config.user_id,
            Arc::clone(&snapshot),
        ));

        Ok(Self {
            snapshot,
            poll_task,
            sse_task,
        })
    }

    /// Cheap handle to the shared snapshot. Clone for each consumer.
    pub fn snapshot(&self) -> Arc<RwLock<StatusSnapshot>> {
        Arc::clone(&self.snapshot)
    }

    /// Probe `GET /api/users` once and return the list. Useful for tools that want
    /// to pick a user before constructing the client.
    pub async fn list_users(
        base_url: &str,
        api_key: &str,
        timeout: Duration,
    ) -> Result<Vec<String>> {
        let base = normalize_base_url(base_url)?;
        let http = http_client(timeout)?;
        let url = format!("{}/api/users", base);
        let resp = http.get(&url).header("X-API-Key", api_key).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(StatusError::BadStatus {
                status: status.as_u16(),
                url,
                body,
            });
        }
        Ok(resp.json::<Vec<String>>().await?)
    }

    /// Fetch a user's cognitive dials (`GET /api/dials`). Returns the raw JSON object
    /// so callers (the monitor) can render/edit without depending on veld's
    /// `CognitiveDials` type.
    pub async fn get_dials(
        base_url: &str,
        api_key: &str,
        user_id: &str,
        timeout: Duration,
    ) -> Result<serde_json::Value> {
        let base = normalize_base_url(base_url)?;
        let http = http_client(timeout)?;
        let url = format!("{}/api/dials", base);
        let resp = http
            .get(&url)
            .query(&[("user_id", user_id)])
            .header("X-API-Key", api_key)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(StatusError::BadStatus {
                status: status.as_u16(),
                url,
                body,
            });
        }
        Ok(resp.json::<serde_json::Value>().await?)
    }

    /// Replace a user's cognitive dials (`PUT /api/dials`). `dials` is the full dials
    /// object (as returned by [`get_dials`](Self::get_dials)). Returns the resulting,
    /// server-sanitized dials.
    pub async fn set_dials(
        base_url: &str,
        api_key: &str,
        user_id: &str,
        dials: serde_json::Value,
        timeout: Duration,
    ) -> Result<serde_json::Value> {
        let base = normalize_base_url(base_url)?;
        let http = http_client(timeout)?;
        let url = format!("{}/api/dials", base);
        let payload = serde_json::json!({ "user_id": user_id, "dials": dials });
        let resp = http
            .put(&url)
            .header("X-API-Key", api_key)
            .json(&payload)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(StatusError::BadStatus {
                status: status.as_u16(),
                url,
                body,
            });
        }
        Ok(resp.json::<serde_json::Value>().await?)
    }

    /// Kick off a project seed (`POST /api/seed`). The server scans `project_path`
    /// (validated server-side to live under the home dir or /tmp) and creates
    /// foundational memories. Returns the server's JSON response.
    pub async fn seed_project(
        base_url: &str,
        api_key: &str,
        user_id: &str,
        project_path: &str,
        timeout: Duration,
    ) -> Result<serde_json::Value> {
        let base = normalize_base_url(base_url)?;
        let http = http_client(timeout)?;
        let url = format!("{}/api/seed", base);
        let payload = serde_json::json!({ "user_id": user_id, "project_path": project_path });
        let resp = http
            .post(&url)
            .header("X-API-Key", api_key)
            .json(&payload)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(StatusError::BadStatus {
                status: status.as_u16(),
                url,
                body,
            });
        }
        Ok(resp.json::<serde_json::Value>().await?)
    }

    /// One call to the user-auth API (`/api/user_auth/*`, key management included),
    /// carrying the session token and, for a change, the step-up code. Returns the
    /// status with the JSON body whatever the status, so the caller can show the
    /// server's reason. Nothing outside that prefix is reachable this way.
    pub async fn security_request(
        base_url: &str,
        method: &str,
        path: &str,
        session: Option<&str>,
        step_up: Option<&str>,
        body: Option<serde_json::Value>,
        timeout: Duration,
    ) -> Result<(u16, serde_json::Value)> {
        let base = normalize_base_url(base_url)?;
        if !path.starts_with("/api/user_auth/") || path.contains("..") {
            return Err(StatusError::InvalidBaseUrl(
                path.to_string(),
                "not a /api/user_auth/ path".to_string(),
            ));
        }
        let http = http_client(timeout)?;
        let url = format!("{base}{path}");
        let mut request = match method {
            "GET" => http.get(&url),
            "POST" => http.post(&url),
            other => {
                return Err(StatusError::InvalidBaseUrl(
                    other.to_string(),
                    "method must be GET or POST".to_string(),
                ))
            }
        };
        if let Some(token) = session.filter(|t| !t.is_empty()) {
            request = request.bearer_auth(token);
        }
        if let Some(code) = step_up.filter(|c| !c.is_empty()) {
            request = request.header("X-Veld-Step-Up", code);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        let resp = request.send().await?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        let json = if text.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text))
        };
        Ok((status, json))
    }

    /// One call to the data-management surface: backups, MIF export/import, the
    /// admin storage migration routes, and `/health`. Authenticates with the API key;
    /// `admin_key` rides as `X-Admin-API-Key` for the `/api/admin/storage/` routes.
    /// Returns the status with the JSON body whatever the status. Nothing outside the
    /// listed prefixes is reachable this way.
    pub async fn data_request(
        base_url: &str,
        api_key: &str,
        method: &str,
        path: &str,
        admin_key: Option<&str>,
        body: Option<serde_json::Value>,
        timeout: Duration,
    ) -> Result<(u16, serde_json::Value)> {
        const ALLOWED: &[&str] = &[
            "/health",
            "/api/backup/",
            "/api/export/mif",
            "/api/import/mif",
            "/api/mif/adapters",
            "/api/admin/storage/",
            "/api/admin/replication/",
        ];
        let base = normalize_base_url(base_url)?;
        if path.contains("..") || !ALLOWED.iter().any(|p| path.starts_with(p)) {
            return Err(StatusError::InvalidBaseUrl(
                path.to_string(),
                "not a data-management path".to_string(),
            ));
        }
        let http = http_client(timeout)?;
        let url = format!("{base}{path}");
        let mut request = match method {
            "GET" => http.get(&url),
            "POST" => http.post(&url),
            other => {
                return Err(StatusError::InvalidBaseUrl(
                    other.to_string(),
                    "method must be GET or POST".to_string(),
                ))
            }
        };
        request = request.header("X-API-Key", api_key);
        if let Some(key) = admin_key.filter(|k| !k.is_empty()) {
            request = request.header("X-Admin-API-Key", key);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        let resp = request.send().await?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        let json = if text.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text))
        };
        Ok((status, json))
    }
}

impl Drop for StatusClient {
    fn drop(&mut self) {
        self.poll_task.abort();
        self.sse_task.abort();
    }
}

/// Path to a PEM bundle of extra trust roots, added to the built-in webpki roots,
/// for a server whose certificate a private CA signed.
pub const ENV_CA_CERT: &str = "VELD_CA_CERT";

/// The HTTP client every request in this crate goes through: the built-in webpki
/// roots plus any in [`ENV_CA_CERT`], read on each call. A bundle that cannot be
/// read or holds no certificate is a [`StatusError::CaBundle`], never skipped.
fn http_client(timeout: Duration) -> Result<Client> {
    http_client_with(std::env::var(ENV_CA_CERT).ok().as_deref(), timeout)
}

fn http_client_with(ca_bundle: Option<&str>, timeout: Duration) -> Result<Client> {
    let builder = Client::builder()
        .timeout(timeout)
        .user_agent(concat!("veld-status-core/", env!("CARGO_PKG_VERSION")));
    let Some(path) = ca_bundle.map(str::trim).filter(|p| !p.is_empty()) else {
        return Ok(builder.build()?);
    };
    extra_roots(path)?
        .into_iter()
        .fold(builder, |b, cert| b.add_root_certificate(cert))
        .build()
        .map_err(|e| ca_bundle_error(path, format!("refused by the TLS client: {}", chain(&e))))
}

fn extra_roots(path: &str) -> Result<Vec<reqwest::Certificate>> {
    let pem = std::fs::read(path)
        .map_err(|e| ca_bundle_error(path, format!("cannot read the file: {e}")))?;
    let certs = reqwest::Certificate::from_pem_bundle(&pem).map_err(|e| {
        ca_bundle_error(path, format!("not a PEM certificate bundle: {}", chain(&e)))
    })?;
    if certs.is_empty() {
        return Err(ca_bundle_error(
            path,
            "no PEM certificate in the file".into(),
        ));
    }
    Ok(certs)
}

fn ca_bundle_error(path: &str, reason: String) -> StatusError {
    StatusError::CaBundle {
        path: path.to_string(),
        reason,
    }
}

/// `err` and its sources joined: reqwest's own message for a builder fault is only
/// "builder error", with the cause one level down.
fn chain(err: &dyn std::error::Error) -> String {
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        source = cause.source();
    }
    out
}

fn normalize_base_url(raw: &str) -> Result<String> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(StatusError::InvalidBaseUrl(
            raw.to_string(),
            "empty".to_string(),
        ));
    }
    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return Err(StatusError::InvalidBaseUrl(
            raw.to_string(),
            "must start with http:// or https://".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

async fn run_poll_loop(
    http: Client,
    base: String,
    api_key: String,
    user_id: String,
    interval: Duration,
    snapshot: Arc<RwLock<StatusSnapshot>>,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        ticker.tick().await;
        poll_once(&http, &base, &api_key, &user_id, &snapshot).await;
    }
}

async fn poll_once(
    http: &Client,
    base: &str,
    api_key: &str,
    user_id: &str,
    snapshot: &Arc<RwLock<StatusSnapshot>>,
) {
    let health = probe_health(http, base).await;
    let memory = fetch_memory_stats(http, base, api_key, user_id).await;
    let graph = fetch_graph_stats(http, base, api_key, user_id).await;
    let todos = fetch_todo_stats(http, base, api_key, user_id).await;
    let sessions = fetch_context_sessions(http, base).await;

    let mut guard = snapshot.write();
    let prior_first_seen = guard.server.first_seen;
    guard.server = match health {
        Ok(mut server) => {
            server.first_seen = prior_first_seen.or(Some(Utc::now()));
            server
        }
        Err(err) => ServerHealth {
            state: ReachState::Unreachable(err.to_string()),
            last_checked: Some(Utc::now()),
            first_seen: prior_first_seen,
            ..ServerHealth::default()
        },
    };
    if let Ok(memory) = memory {
        guard.memory = memory;
    }
    if let Ok(graph) = graph {
        guard.graph = graph;
    }
    if let Ok(todos) = todos {
        guard.todos = todos;
    }
    if let Ok(sessions) = sessions {
        guard.sessions = sessions;
    }
    guard.last_full_refresh = Some(Utc::now());
}

async fn probe_health(http: &Client, base: &str) -> Result<ServerHealth> {
    let url = format!("{}/health", base);
    let start = Instant::now();
    let resp = http.get(&url).send().await?;
    let rtt_u128 = start.elapsed().as_millis();
    let rtt = if rtt_u128 > u32::MAX as u128 {
        u32::MAX
    } else {
        rtt_u128 as u32
    };
    let status = resp.status();
    let now = Utc::now();

    if status == StatusCode::OK {
        let body: HealthDto = resp.json().await?;
        Ok(ServerHealth {
            state: ReachState::Reachable,
            rtt_ms: Some(rtt),
            version: opt_string(body.version),
            build: opt_string(body.build),
            built_at: opt_string(body.built_at),
            effective_storage_backend: opt_string(body.effective_storage_backend),
            users_count: Some(body.users_count),
            users_in_cache: Some(body.users_in_cache),
            last_checked: Some(now),
            first_seen: None,
        })
    } else {
        let detail = status.canonical_reason().unwrap_or("error").to_string();
        Ok(ServerHealth {
            state: ReachState::Unhealthy(format!("{} {}", status.as_u16(), detail)),
            rtt_ms: Some(rtt),
            last_checked: Some(now),
            first_seen: None,
            ..ServerHealth::default()
        })
    }
}

async fn fetch_memory_stats(
    http: &Client,
    base: &str,
    api_key: &str,
    user_id: &str,
) -> Result<TierStats> {
    let url = format!("{}/api/users/{}/stats", base, user_id);
    let dto: MemoryStatsDto = get_json(http, &url, api_key).await?;
    let total = dto.total_memories as u64;
    let vector_index = dto.vector_index_count as u64;
    Ok(TierStats {
        working: dto.working_memory_count as u64,
        session: dto.session_memory_count as u64,
        long_term: dto.long_term_memory_count as u64,
        total,
        vector_index,
        total_retrievals: dto.total_retrievals as u64,
        index_healthy: vector_index >= total,
    })
}

async fn fetch_graph_stats(
    http: &Client,
    base: &str,
    api_key: &str,
    user_id: &str,
) -> Result<GraphStats> {
    let url = format!("{}/api/graph/{}/stats", base, user_id);
    let dto: GraphStatsDto = get_json(http, &url, api_key).await?;
    Ok(GraphStats {
        entities: dto.entity_count as u64,
        relationships: dto.relationship_count as u64,
    })
}

async fn fetch_todo_stats(
    http: &Client,
    base: &str,
    api_key: &str,
    user_id: &str,
) -> Result<TodoStats> {
    let url = format!("{}/api/todos/stats", base);
    let body = serde_json::json!({ "user_id": user_id });
    let resp = http
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(StatusError::BadStatus {
            status: status.as_u16(),
            url,
            body,
        });
    }
    let wrapper: TodoStatsWrapper = resp.json().await?;
    Ok(from_todo_dto(wrapper.stats))
}

fn from_todo_dto(dto: TodoStatsDto) -> TodoStats {
    TodoStats {
        total: dto.total,
        backlog: dto.backlog,
        todo: dto.todo,
        in_progress: dto.in_progress,
        blocked: dto.blocked,
        done: dto.done,
        overdue: dto.overdue,
    }
}

async fn fetch_context_sessions(http: &Client, base: &str) -> Result<Vec<ContextSession>> {
    let url = format!("{}/api/context_status", base);
    let resp = http.get(&url).send().await?;
    if !resp.status().is_success() {
        return Ok(Vec::new());
    }
    let dtos: Vec<ContextSessionDto> = resp.json().await.unwrap_or_default();
    Ok(dtos
        .into_iter()
        .map(|d| ContextSession {
            session_id: d.session_id.unwrap_or_default(),
            tokens_used: d.tokens_used,
            tokens_budget: d.tokens_budget,
            percent_used: d.percent_used,
            current_task: d.current_task,
            model: d.model,
            updated_at: d.updated_at,
        })
        .collect())
}

async fn get_json<T: serde::de::DeserializeOwned>(
    http: &Client,
    url: &str,
    api_key: &str,
) -> Result<T> {
    let resp = http.get(url).header("X-API-Key", api_key).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        warn!(%status, url, "non-2xx response in status poller");
        return Err(StatusError::BadStatus {
            status: status.as_u16(),
            url: url.to_string(),
            body,
        });
    }
    let json = resp.json::<T>().await?;
    debug!(url, "polled");
    Ok(json)
}

fn opt_string(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT: Duration = Duration::from_secs(5);

    fn fixture(name: &str) -> String {
        format!(
            "{}/../tests/fixtures/tls/{name}",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn refusal(bundle: &str) -> String {
        let err = http_client_with(Some(bundle), TIMEOUT).expect_err("the bundle is refused");
        assert!(matches!(err, StatusError::CaBundle { .. }), "{err}");
        err.to_string()
    }

    #[test]
    fn no_bundle_gives_a_plain_client() {
        http_client_with(None, TIMEOUT).unwrap();
        http_client_with(Some("  "), TIMEOUT).unwrap();
    }

    #[test]
    fn a_bundle_adds_every_certificate_in_it() {
        let bundle = fixture("test-ca-bundle.pem");
        assert_eq!(extra_roots(&bundle).unwrap().len(), 2);
        http_client_with(Some(&bundle), TIMEOUT).unwrap();
    }

    #[test]
    fn an_unreadable_bundle_names_the_variable_and_the_path() {
        let missing = fixture("no-such-bundle.pem");
        let err = refusal(&missing);
        assert!(
            err.starts_with(&format!("{ENV_CA_CERT}={missing}: cannot read")),
            "{err}"
        );
    }

    #[test]
    fn a_bad_pem_names_the_variable_and_the_path() {
        let corrupt = fixture("corrupt-cert.pem");
        let err = refusal(&corrupt);
        assert!(
            err.starts_with(&format!(
                "{ENV_CA_CERT}={corrupt}: not a PEM certificate bundle"
            )),
            "{err}"
        );

        let not_pem = format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"));
        let err = refusal(&not_pem);
        assert_eq!(
            err,
            format!("{ENV_CA_CERT}={not_pem}: no PEM certificate in the file")
        );
    }
}
