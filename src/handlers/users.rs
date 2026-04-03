//! User Management Handlers
//!
//! Handlers for user-related operations including stats, deletion (GDPR), and listing.

use axum::{
    extract::{Extension, Path, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::MemoryStats;
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// GET /api/users/{user_id}/stats - Get user statistics
pub async fn get_user_stats(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<MemoryStats>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let stats = state.get_stats(&user_id).map_err(AppError::Internal)?;
    Ok(Json(stats))
}

/// Query parameters for stats endpoint
#[derive(Debug, Deserialize)]
pub struct StatsQuery {
    pub user_id: Option<String>,
}

/// GET /api/stats - OpenAPI spec compatible stats endpoint
pub async fn get_stats_query(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(query): Query<StatsQuery>,
) -> Result<Json<MemoryStats>, AppError> {
    let user_id = resolve_request_user_id(
        query.user_id.as_deref(),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let stats = state
        .get_stats(&user_id)
        .map_err(AppError::Internal)?;
    Ok(Json(stats))
}

/// Response for user deletion
#[derive(Debug, Serialize)]
pub struct DeleteUserResponse {
    pub success: bool,
    pub user_id: String,
    pub message: String,
}

/// DELETE /api/users/{user_id} - Delete user data (GDPR compliance)
pub async fn delete_user(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<DeleteUserResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    state.forget_user(&user_id).map_err(AppError::Internal)?;

    Ok(Json(DeleteUserResponse {
        success: true,
        user_id,
        message: "User data deleted successfully".to_string(),
    }))
}

/// GET /api/users - List all users
pub async fn list_users(State(state): State<AppState>) -> Json<Vec<String>> {
    Json(state.list_users())
}
