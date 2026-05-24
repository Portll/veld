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
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, user_id: impl Into<String>) -> Self {
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

/// Background poller + SSE subscriber. Drop the handle to stop both tasks.
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

        let http = Client::builder()
            .timeout(config.request_timeout)
            .user_agent(concat!("veld-status-core/", env!("CARGO_PKG_VERSION")))
            .build()?;

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
    pub async fn list_users(base_url: &str, api_key: &str, timeout: Duration) -> Result<Vec<String>> {
        let base = normalize_base_url(base_url)?;
        let http = Client::builder().timeout(timeout).build()?;
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
}

impl Drop for StatusClient {
    fn drop(&mut self) {
        self.poll_task.abort();
        self.sse_task.abort();
    }
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
