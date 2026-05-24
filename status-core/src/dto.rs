//! Wire DTOs matching the veld server's response shapes.
//!
//! Modeled directly off the existing `tui/src/stream.rs` definitions to guarantee
//! compatibility with what the server actually emits, including all `serde(default)`
//! tolerance for fields that older builds may omit.

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct HealthDto {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub build: String,
    #[serde(default)]
    pub built_at: String,
    #[serde(default)]
    pub effective_storage_backend: String,
    #[serde(default)]
    pub users_count: usize,
    #[serde(default)]
    pub users_in_cache: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MemoryStatsDto {
    #[serde(default)]
    pub total_memories: usize,
    #[serde(default)]
    pub working_memory_count: usize,
    #[serde(default)]
    pub session_memory_count: usize,
    #[serde(default)]
    pub long_term_memory_count: usize,
    #[serde(default)]
    pub vector_index_count: usize,
    #[serde(default)]
    pub total_retrievals: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GraphStatsDto {
    #[serde(default)]
    pub entity_count: usize,
    #[serde(default)]
    pub relationship_count: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TodoStatsWrapper {
    pub stats: TodoStatsDto,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TodoStatsDto {
    #[serde(default)]
    pub total: u32,
    #[serde(default)]
    pub backlog: u32,
    #[serde(default)]
    pub todo: u32,
    #[serde(default)]
    pub in_progress: u32,
    #[serde(default)]
    pub blocked: u32,
    #[serde(default)]
    pub done: u32,
    #[serde(default)]
    pub overdue: u32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ContextSessionDto {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub tokens_used: u64,
    #[serde(default)]
    pub tokens_budget: u64,
    #[serde(default)]
    pub percent_used: u8,
    #[serde(default)]
    pub current_task: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,
}

/// SSE event payload — fields match `tui::types::MemoryEvent`. Most are unused by
/// the status surface; we keep only what the activity tail renders.
#[derive(Debug, Deserialize)]
pub(crate) struct MemoryEventDto {
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub content_preview: Option<String>,
}
