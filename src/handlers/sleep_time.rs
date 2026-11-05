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
    extract::{Extension, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::auth::AuthenticatedUser;
use crate::errors::AppError;
use crate::handlers::state::MultiUserMemoryManager;
use crate::handlers::utils::resolve_request_user_id;
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

/// Why a sleep-time route refused: a hand-built `{"error": …}` body with its status (no
/// orchestrator, a bad mode or trigger, an orchestrator failure), or the shared [`AppError`]
/// body when the caller may not act for the requested user.
#[derive(Debug)]
pub enum SleepTimeRefusal {
    Status(StatusCode, ErrorResponse),
    App(AppError),
}

impl IntoResponse for SleepTimeRefusal {
    fn into_response(self) -> Response {
        match self {
            Self::Status(status, body) => (status, Json(body)).into_response(),
            Self::App(e) => e.into_response(),
        }
    }
}

impl From<AppError> for SleepTimeRefusal {
    fn from(e: AppError) -> Self {
        Self::App(e)
    }
}

impl SleepTimeRefusal {
    fn status(status: StatusCode, error: String) -> Self {
        Self::Status(status, ErrorResponse { error })
    }
}

// =============================================================================
// Handlers
// =============================================================================

fn orchestrator_or_503(state: &AppState) -> Result<Arc<SleepTimeOrchestrator>, SleepTimeRefusal> {
    let guard = state.sleep_time_orchestrator.read();
    match guard.as_ref() {
        Some(orch) => Ok(orch.clone()),
        None => Err(SleepTimeRefusal::status(
            StatusCode::SERVICE_UNAVAILABLE,
            "sleep-time not enabled on this server".to_string(),
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
) -> Result<(StatusCode, Json<EnqueueResponse>), SleepTimeRefusal> {
    let orch = orchestrator_or_503(&state)?;
    let mode =
        parse_mode(&body.mode).map_err(|e| SleepTimeRefusal::status(StatusCode::BAD_REQUEST, e))?;
    let trigger = parse_trigger(&body.trigger)
        .map_err(|e| SleepTimeRefusal::status(StatusCode::BAD_REQUEST, e))?;

    match orch.enqueue(&body.user_id, mode, trigger) {
        Ok(true) => Ok((
            StatusCode::OK,
            Json(EnqueueResponse {
                accepted: true,
                reason: None,
            }),
        )),
        Ok(false) => Ok((
            StatusCode::OK,
            Json(EnqueueResponse {
                accepted: false,
                reason: Some("debounced".to_string()),
            }),
        )),
        Err(e) => Err(SleepTimeRefusal::status(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("enqueue failed: {e}"),
        )),
    }
}

/// `GET /api/sleep_time/status`
pub async fn status(
    State(state): State<AppState>,
) -> Result<(StatusCode, Json<OrchestratorStatusResponse>), SleepTimeRefusal> {
    let orch = orchestrator_or_503(&state)?;
    match orch.status() {
        Ok(s) => Ok((StatusCode::OK, Json(OrchestratorStatusResponse::from(s)))),
        Err(e) => Err(SleepTimeRefusal::status(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("status failed: {e}"),
        )),
    }
}

/// `GET /api/sleep_time/status/:user_id`
pub async fn user_status(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<(StatusCode, Json<UserStatusResponse>), SleepTimeRefusal> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    let orch = orchestrator_or_503(&state)?;
    match orch.user_status(&user_id) {
        Ok(s) => Ok((
            StatusCode::OK,
            Json(UserStatusResponse {
                user_id: s.user_id,
                pending_count: s.pending_count,
                tokens_this_hour: s.budget.tokens_this_hour,
                calls_today: s.budget.calls_today,
                locked_blocks: s.budget.locked_blocks,
            }),
        )),
        Err(e) => Err(SleepTimeRefusal::status(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("user_status failed: {e}"),
        )),
    }
}

/// `POST /api/sleep_time/lock_block`
pub async fn lock_block(
    State(state): State<AppState>,
    Json(body): Json<LockBlockRequest>,
) -> Result<(StatusCode, Json<LockBlockResponse>), SleepTimeRefusal> {
    let orch = orchestrator_or_503(&state)?;
    match orch.set_block_lock(&body.user_id, &body.block_key, body.locked) {
        Ok(()) => Ok((
            StatusCode::OK,
            Json(LockBlockResponse {
                user_id: body.user_id,
                block_key: body.block_key,
                locked: body.locked,
            }),
        )),
        Err(e) => Err(SleepTimeRefusal::status(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("lock_block failed: {e}"),
        )),
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

    #[tokio::test]
    async fn a_status_refusal_answers_its_code_with_an_error_body() {
        let refusal = SleepTimeRefusal::status(
            StatusCode::SERVICE_UNAVAILABLE,
            "sleep-time not enabled on this server".to_string(),
        )
        .into_response();
        assert_eq!(refusal.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(refusal.into_body(), usize::MAX)
            .await
            .expect("a buffered body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("a JSON body");
        assert_eq!(
            body,
            serde_json::json!({"error": "sleep-time not enabled on this server"})
        );
    }
}
