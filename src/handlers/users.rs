//! User Management Handlers
//!
//! Handlers for user-related operations including stats, deletion (GDPR), and listing.

use axum::{
    extract::{Extension, Path, Query, State},
    http::HeaderMap,
    response::{Json, Response},
};
use serde::{Deserialize, Serialize};

use super::admin::require_admin_key;
use super::state::MultiUserMemoryManager;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::MemoryStats;
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// [`MultiUserMemoryManager::get_stats`] on the blocking pool: it opens the tenant and its graph
/// when they are not cached.
async fn stats_off_worker(state: &AppState, user_id: String) -> Result<MemoryStats, AppError> {
    let state = state.clone();
    tokio::task::spawn_blocking(move || state.get_stats(&user_id))
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("stats task failed: {e}")))?
        .map_err(AppError::Internal)
}

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
    Ok(Json(stats_off_worker(&state, user_id).await?))
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
    Ok(Json(stats_off_worker(&state, user_id).await?))
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
/// A key bound to one tenant (multi-tenant mode) must also present the admin key
/// (`X-Admin-API-Key`, checked as every admin route checks it): enumerating other
/// tenants is outside its binding. An unbound key already reaches every tenant by
/// `user_id`, so it may list them.
pub async fn list_users(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
) -> Result<Json<Vec<String>>, Response> {
    if authenticated_user.is_some() {
        require_admin_key(&headers)?;
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
