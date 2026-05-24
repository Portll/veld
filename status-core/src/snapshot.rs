//! The shared snapshot type that both the TUI and the GUI render from.
//!
//! Every field is plain data — no locks, no IO. The owning [`crate::StatusClient`]
//! mutates the snapshot under a write lock, and consumers take a read lock just
//! long enough to clone what they need.

use std::collections::VecDeque;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Whether the server is currently reachable, and if not, how the last attempt failed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum ReachState {
    /// The most recent health probe succeeded.
    Reachable,
    /// The probe completed but returned a non-2xx status (server alive, unhealthy).
    Unhealthy(String),
    /// Network failure — could not connect or the request timed out.
    Unreachable(String),
    /// No probe has run yet.
    #[default]
    Unknown,
}

impl ReachState {
    pub fn is_reachable(&self) -> bool {
        matches!(self, ReachState::Reachable)
    }
}

/// Health attributes scraped from `GET /health`, plus the round-trip time of the probe.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerHealth {
    pub state: ReachState,
    pub rtt_ms: Option<u32>,
    pub version: Option<String>,
    pub build: Option<String>,
    pub built_at: Option<String>,
    pub effective_storage_backend: Option<String>,
    pub users_count: Option<usize>,
    pub users_in_cache: Option<usize>,
    pub last_checked: Option<DateTime<Utc>>,
    /// Set on first successful probe and never reset. The UI computes uptime as
    /// `now - first_seen` — accurate only for the duration of this client process,
    /// which is what "uptime since I started watching" actually means to a user.
    pub first_seen: Option<DateTime<Utc>>,
}

impl ServerHealth {
    pub fn uptime_secs(&self) -> Option<u64> {
        let first = self.first_seen?;
        let now = Utc::now();
        let delta = (now - first).num_seconds();
        if delta < 0 {
            None
        } else {
            Some(delta as u64)
        }
    }
}

/// Memory tier counts from `GET /api/users/{user}/stats`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierStats {
    pub working: u64,
    pub session: u64,
    pub long_term: u64,
    pub total: u64,
    pub vector_index: u64,
    pub total_retrievals: u64,
    /// True when the vector index covers every stored memory.
    pub index_healthy: bool,
}

/// Knowledge-graph counts from `GET /api/graph/{user}/stats`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    pub entities: u64,
    pub relationships: u64,
}

/// Todo counts from `POST /api/todos/stats`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TodoStats {
    pub total: u32,
    pub backlog: u32,
    pub todo: u32,
    pub in_progress: u32,
    pub blocked: u32,
    pub done: u32,
    pub overdue: u32,
}

/// One active Claude Code session, as reported by `GET /api/context_status`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextSession {
    pub session_id: String,
    pub tokens_used: u64,
    pub tokens_budget: u64,
    pub percent_used: u8,
    pub model: Option<String>,
    pub current_task: Option<String>,
    pub updated_at: Option<DateTime<Utc>>,
}

/// One line of the activity tail. Built from SSE `MemoryEvent`s, stripped of
/// graph-specific fields the status surface does not render.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEntry {
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    pub memory_type: Option<String>,
    pub preview: Option<String>,
}

/// All data the status UIs render. Cheap to clone field-by-field; do not clone whole.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatusSnapshot {
    pub server: ServerHealth,
    pub memory: TierStats,
    pub graph: GraphStats,
    pub todos: TodoStats,
    pub sessions: Vec<ContextSession>,
    pub recent: VecDeque<ActivityEntry>,
    pub base_url: String,
    pub user_id: String,
    pub last_full_refresh: Option<DateTime<Utc>>,
}

impl StatusSnapshot {
    /// Cap on the in-memory activity tail. SSE writers must enforce this.
    pub const MAX_ACTIVITY: usize = 200;

    pub fn push_activity(&mut self, entry: ActivityEntry) {
        if self.recent.len() >= Self::MAX_ACTIVITY {
            self.recent.pop_back();
        }
        self.recent.push_front(entry);
    }
}
