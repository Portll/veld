//! User Management Handlers
//!
//! Handlers for user-related operations including stats, deletion (GDPR), and listing.

use axum::{
    extract::{Extension, Path, Query, State},
    http::HeaderMap,
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
    let caller_id = authenticated_user
        .as_ref()
        .map(|e| e.user_id.as_str())
        .unwrap_or("<unauthenticated>");
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    tracing::warn!(
        audit = "delete_user",
        caller = %caller_id,
        target_user = %user_id,
        "audit: delete_user requested by authenticated caller"
    );

    state.forget_user(&user_id).map_err(AppError::Internal)?;

    Ok(Json(DeleteUserResponse {
        success: true,
        user_id,
        message: "User data deleted successfully".to_string(),
    }))
}

/// GET /api/users - List all users
///
/// This endpoint enumerates all tenant IDs. Access is restricted to requests
/// presenting the designated admin API key (`VELD_ADMIN_KEY` env var). If
/// `VELD_ADMIN_KEY` is not configured the endpoint is accessible to any
/// authenticated caller (development mode).
pub async fn list_users(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
) -> Result<Json<Vec<String>>, AppError> {
    // Admin gate: if VELD_ADMIN_KEY is set, the request must supply that exact key.
    if let Ok(admin_key) = std::env::var("VELD_ADMIN_KEY") {
        let provided_key = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if provided_key != admin_key.as_str() {
            return Err(AppError::InvalidInput {
                field: "authorization".to_string(),
                reason: "Admin access required to enumerate tenants".to_string(),
            });
        }
    }

    let caller_id = authenticated_user
        .as_ref()
        .map(|e| e.user_id.as_str())
        .unwrap_or("<unauthenticated>");

    tracing::warn!(
        audit = "list_users",
        caller = %caller_id,
        "audit: list_users enumeration by authenticated caller"
    );

    Ok(Json(state.list_users()))
}
