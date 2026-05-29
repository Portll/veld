//! HTTP endpoints for sleep-time / observational memory.
//!
//! Routes mounted under `/api/sleep_time/*`:
//!
//! | Method | Path                                    | Purpose                                     |
//! |--------|-----------------------------------------|---------------------------------------------|
//! | POST   | /api/sleep_time/enqueue                 | Enqueue a trigger for a user/mode pair      |
//! | GET    | /api/sleep_time/status                  | Orchestrator-wide status snapshot           |
//! | GET    | /api/sleep_time/status/:user_id         | Per-user pending + budget snapshot          |
//! | POST   | /api/sleep_time/lock_block              | Toggle a ContextBlock lock (R14 + R22)      |
//!
//! All handlers return `503 Service Unavailable` when sleep-time is not
//! enabled on the running server (no orchestrator installed). The handler
//! signatures match Axum 0.8's `State<AppState>` extraction pattern used
//! throughout Veld's existing routes.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::handlers::state::MultiUserMemoryManager;
use crate::memory::sleep_time::{
    types::{SleepMode, SleepTimeTrigger},
    OrchestratorStatus, SleepTimeOrchestrator,
};

pub type AppState = Arc<MultiUserMemoryManager>;

// =============================================================================
// Request / response shapes
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct EnqueueRequest {
    pub user_id: String,
    pub mode: String,    // "nrem" | "rem"
    pub trigger: String, // "idle" | "session_close" | "maintenance_heavy_cycle" | "manual"
}

#[derive(Debug, Serialize)]
pub struct EnqueueResponse {
    pub accepted: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OrchestratorStatusResponse {
    pub enabled: bool,
    pub num_workers: usize,
    pub queue_pending_total: usize,
    pub distinct_users_in_queue: usize,
}

impl From<OrchestratorStatus> for OrchestratorStatusResponse {
    fn from(s: OrchestratorStatus) -> Self {
        Self {
            enabled: s.enabled,
            num_workers: s.num_workers,
            queue_pending_total: s.queue_pending_total,
            distinct_users_in_queue: s.distinct_users_in_queue,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct UserStatusResponse {
    pub user_id: String,
    pub pending_count: usize,
    pub tokens_this_hour: u32,
    pub calls_today: u32,
    pub locked_blocks: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct LockBlockRequest {
    pub user_id: String,
    pub block_key: String,
    pub locked: bool,
}

#[derive(Debug, Serialize)]
pub struct LockBlockResponse {
    pub user_id: String,
    pub block_key: String,
    pub locked: bool,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// =============================================================================
// Handlers
// =============================================================================

fn orchestrator_or_503(state: &AppState) -> Result<Arc<SleepTimeOrchestrator>, impl IntoResponse> {
    let guard = state.sleep_time_orchestrator.read();
    match guard.as_ref() {
        Some(orch) => Ok(orch.clone()),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "sleep-time not enabled on this server".to_string(),
            }),
        )),
    }
}

fn parse_mode(s: &str) -> Result<SleepMode, String> {
    match s {
        "nrem" => Ok(SleepMode::Nrem),
        "rem" => Ok(SleepMode::Rem),
        other => Err(format!("unknown sleep mode `{other}` (expected nrem|rem)")),
    }
}

fn parse_trigger(s: &str) -> Result<SleepTimeTrigger, String> {
    match s {
        "idle" => Ok(SleepTimeTrigger::Idle),
        "session_close" => Ok(SleepTimeTrigger::SessionClose),
        "maintenance_heavy_cycle" => Ok(SleepTimeTrigger::MaintenanceHeavyCycle),
        "manual" => Ok(SleepTimeTrigger::Manual),
        other => Err(format!(
            "unknown sleep trigger `{other}` (expected idle|session_close|maintenance_heavy_cycle|manual)"
        )),
    }
}

/// `POST /api/sleep_time/enqueue`
pub async fn enqueue(
    State(state): State<AppState>,
    Json(body): Json<EnqueueRequest>,
) -> impl IntoResponse {
    let orch = match orchestrator_or_503(&state) {
        Ok(o) => o,
        Err(resp) => return resp.into_response(),
    };
    let mode = match parse_mode(&body.mode) {
        Ok(m) => m,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })).into_response(),
    };
    let trigger = match parse_trigger(&body.trigger) {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })).into_response(),
    };

    match orch.enqueue(&body.user_id, mode, trigger) {
        Ok(true) => (
            StatusCode::OK,
            Json(EnqueueResponse {
                accepted: true,
                reason: None,
            }),
        )
            .into_response(),
        Ok(false) => (
            StatusCode::OK,
            Json(EnqueueResponse {
                accepted: false,
                reason: Some("debounced".to_string()),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("enqueue failed: {e}"),
            }),
        )
            .into_response(),
    }
}

/// `GET /api/sleep_time/status`
pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let orch = match orchestrator_or_503(&state) {
        Ok(o) => o,
        Err(resp) => return resp.into_response(),
    };
    match orch.status() {
        Ok(s) => (
            StatusCode::OK,
            Json(OrchestratorStatusResponse::from(s)),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("status failed: {e}"),
            }),
        )
            .into_response(),
    }
}

/// `GET /api/sleep_time/status/:user_id`
pub async fn user_status(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    let orch = match orchestrator_or_503(&state) {
        Ok(o) => o,
        Err(resp) => return resp.into_response(),
    };
    match orch.user_status(&user_id) {
        Ok(s) => (
            StatusCode::OK,
            Json(UserStatusResponse {
                user_id: s.user_id,
                pending_count: s.pending_count,
                tokens_this_hour: s.budget.tokens_this_hour,
                calls_today: s.budget.calls_today,
                locked_blocks: s.budget.locked_blocks,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("user_status failed: {e}"),
            }),
        )
            .into_response(),
    }
}

/// `POST /api/sleep_time/lock_block`
pub async fn lock_block(
    State(state): State<AppState>,
    Json(body): Json<LockBlockRequest>,
) -> impl IntoResponse {
    let orch = match orchestrator_or_503(&state) {
        Ok(o) => o,
        Err(resp) => return resp.into_response(),
    };
    match orch.set_block_lock(&body.user_id, &body.block_key, body.locked) {
        Ok(()) => (
            StatusCode::OK,
            Json(LockBlockResponse {
                user_id: body.user_id,
                block_key: body.block_key,
                locked: body.locked,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("lock_block failed: {e}"),
            }),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mode_known() {
        assert_eq!(parse_mode("nrem").unwrap(), SleepMode::Nrem);
        assert_eq!(parse_mode("rem").unwrap(), SleepMode::Rem);
    }

    #[test]
    fn parse_mode_unknown() {
        assert!(parse_mode("light").is_err());
    }

    #[test]
    fn parse_trigger_known() {
        assert_eq!(parse_trigger("idle").unwrap(), SleepTimeTrigger::Idle);
        assert_eq!(
            parse_trigger("session_close").unwrap(),
            SleepTimeTrigger::SessionClose
        );
        assert_eq!(
            parse_trigger("maintenance_heavy_cycle").unwrap(),
            SleepTimeTrigger::MaintenanceHeavyCycle
        );
        assert_eq!(parse_trigger("manual").unwrap(), SleepTimeTrigger::Manual);
    }

    #[test]
    fn parse_trigger_unknown() {
        assert!(parse_trigger("explode").is_err());
    }
}
