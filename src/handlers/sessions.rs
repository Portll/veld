//! Session Management Handlers
//!
//! Handlers for user session tracking and management.

use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{
    Session, SessionEvent, SessionId, SessionStatus, SessionStoreStats, SessionSummary,
};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

fn default_sessions_limit() -> usize {
    10
}

fn default_end_reason() -> String {
    "user_ended".to_string()
}

/// Request for listing sessions
#[derive(Debug, Deserialize)]
pub struct ListSessionsRequest {
    pub user_id: String,
    #[serde(default = "default_sessions_limit")]
    pub limit: usize,
}

/// Response for listing sessions
#[derive(Debug, Serialize)]
pub struct ListSessionsResponse {
    pub success: bool,
    pub sessions: Vec<SessionSummary>,
    pub count: usize,
}

/// Request for getting a specific session
#[derive(Debug, Deserialize)]
pub struct GetSessionRequest {
    pub user_id: String,
}

/// Response for getting a session
#[derive(Debug, Serialize)]
pub struct GetSessionResponse {
    pub success: bool,
    pub session: Option<Session>,
}

/// Request for ending a session
#[derive(Debug, Deserialize)]
pub struct EndSessionRequest {
    pub user_id: String,
    #[serde(default = "default_end_reason")]
    pub reason: String,
}

/// Response for ending a session
#[derive(Debug, Serialize)]
pub struct EndSessionResponse {
    pub success: bool,
    pub session: Option<Session>,
}

/// Response for session store stats
#[derive(Debug, Serialize)]
pub struct SessionStoreStatsResponse {
    pub success: bool,
    pub stats: SessionStoreStats,
}

/// POST /api/sessions - List sessions for a user
pub async fn list_sessions(
    State(state): State<AppState>,
    Json(req): Json<ListSessionsRequest>,
) -> Result<Json<ListSessionsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let sessions = state
        .session_store
        .get_user_sessions(&req.user_id, req.limit);
    let count = sessions.len();

    Ok(Json(ListSessionsResponse {
        success: true,
        sessions,
        count,
    }))
}

/// GET /api/sessions/{session_id} - Get a specific session
pub async fn get_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Query(req): Query<GetSessionRequest>,
) -> Result<Json<GetSessionResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let uuid = uuid::Uuid::parse_str(&session_id).map_err(|e| AppError::InvalidInput {
        field: "session_id".to_string(),
        reason: format!("Invalid UUID: {e}"),
    })?;
    let sid = SessionId(uuid);
    let session = state.session_store.get_session(&sid);

    Ok(Json(GetSessionResponse {
        success: session.is_some(),
        session,
    }))
}

/// POST /api/sessions/end - End the current/active session for a user
pub async fn end_session(
    State(state): State<AppState>,
    Json(req): Json<EndSessionRequest>,
) -> Result<Json<EndSessionResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let sessions = state.session_store.get_user_sessions(&req.user_id, 1);
    let active_session = sessions
        .into_iter()
        .find(|s| matches!(s.status, SessionStatus::Active));

    if let Some(summary) = active_session {
        let session = state.session_store.end_session(&summary.id, &req.reason);

        // Auto-populate session_summary context block from the ended session
        if let Some(ref ended) = session {
            populate_session_summary_block(&state, &req.user_id, ended);
        }

        Ok(Json(EndSessionResponse {
            success: session.is_some(),
            session,
        }))
    } else {
        Ok(Json(EndSessionResponse {
            success: false,
            session: None,
        }))
    }
}

/// Build and write a `session_summary` context block from a completed session.
///
/// Collects key entities mentioned, memory types created, and aggregate stats
/// from the session timeline, then writes a compact summary (max 500 chars)
/// to the user's `session_summary` context block.
fn populate_session_summary_block(
    state: &AppState,
    user_id: &str,
    session: &Session,
) {
    use std::collections::{BTreeSet, HashSet};

    // Collect entities and topics from timeline events
    let mut entities: BTreeSet<String> = BTreeSet::new();
    let mut memory_types: HashSet<String> = HashSet::new();

    for event in &session.timeline {
        if let SessionEvent::MemoryCreated {
            entities: ents,
            memory_type,
            ..
        } = event
        {
            for e in ents {
                entities.insert(e.clone());
            }
            memory_types.insert(memory_type.clone());
        }
    }

    let stats = &session.stats;
    let duration_mins = session.duration().num_minutes();

    // Build summary parts
    let mut parts: Vec<String> = Vec::new();

    // Duration and date
    parts.push(format!(
        "Session ended ({} min, {})",
        duration_mins,
        session.temporal.short_label()
    ));

    // Memory stats
    if stats.memories_created > 0 {
        let types_str = if memory_types.is_empty() {
            String::new()
        } else {
            let types_vec: Vec<&str> = memory_types.iter().map(|s| s.as_str()).collect();
            format!(" [{}]", types_vec.join(", "))
        };
        parts.push(format!(
            "{} memories created{}",
            stats.memories_created, types_str
        ));
    }

    if stats.memories_surfaced > 0 {
        parts.push(format!(
            "{} surfaced, {} used (hit rate {:.0}%)",
            stats.memories_surfaced,
            stats.memories_used,
            stats.memory_hit_rate * 100.0
        ));
    }

    // Todo stats
    if stats.todos_created > 0 || stats.todos_completed > 0 {
        parts.push(format!(
            "todos: {} created, {} completed",
            stats.todos_created, stats.todos_completed
        ));
    }

    // Key entities (top 10 to keep summary compact)
    if !entities.is_empty() {
        let entity_list: Vec<&str> = entities.iter().map(|s| s.as_str()).take(10).collect();
        let suffix = if entities.len() > 10 {
            format!(" (+{} more)", entities.len() - 10)
        } else {
            String::new()
        };
        parts.push(format!("entities: {}{}", entity_list.join(", "), suffix));
    }

    // Query volume
    if stats.queries_count > 0 {
        parts.push(format!("{} queries", stats.queries_count));
    }

    let mut summary = parts.join(". ");

    // Enforce 500-char max
    if summary.len() > 500 {
        summary.truncate(497);
        summary.push_str("...");
    }

    if let Err(e) = state
        .context_block_store
        .set(user_id, "session_summary", &summary, None)
    {
        tracing::warn!(
            user_id = user_id,
            error = %e,
            "Failed to write session_summary context block"
        );
    } else {
        tracing::debug!(
            user_id = user_id,
            summary_len = summary.len(),
            "Session summary context block updated"
        );
    }
}

/// GET /api/sessions/stats - Get overall session store statistics
pub async fn get_session_stats(
    State(state): State<AppState>,
) -> Result<Json<SessionStoreStatsResponse>, AppError> {
    let stats = state.session_store.stats();

    Ok(Json(SessionStoreStatsResponse {
        success: true,
        stats,
    }))
}
