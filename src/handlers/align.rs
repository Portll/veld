//! Alignment performance-flags API (GET/PUT) — the runtime tuning surface for the v2
//! alignment engine.
//!
//! `GET /api/align?user_id=…` returns a user's current [`AlignmentConfig`];
//! `PUT /api/align` replaces it (full-replace REST — client GETs, edits, PUTs the whole
//! struct). Values are sanitized server-side and capability-validated: a config the build
//! can't honor (e.g. a gradient objective or adapter with no training runtime, or vec2vec
//! when not compiled) is rejected `400`. Both routes live on the **protected** router and
//! bind to `user_id` (per the `public_router_has_no_per_user_handlers` test).
//!
//! Mirrors [`super::dials`] — the same env-seed → KV-overlay → runtime-set substrate.

use axum::{
    extract::{Query, State},
    response::Json,
};
use serde::Deserialize;

use super::state::MultiUserMemoryManager;
use crate::embeddings::AlignmentConfig;
use crate::errors::{AppError, ValidationErrorExt};
use crate::validation;

/// Application state type alias.
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Query for `GET /api/align`.
#[derive(Debug, Deserialize)]
pub struct AlignQuery {
    pub user_id: String,
}

/// Body for `PUT /api/align` (full replace).
#[derive(Debug, Deserialize)]
pub struct SetAlignRequest {
    pub user_id: String,
    pub config: AlignmentConfig,
}

/// `GET /api/align?user_id=…` — return the user's current alignment config.
pub async fn get_align(
    State(state): State<AppState>,
    Query(query): Query<AlignQuery>,
) -> Result<Json<AlignmentConfig>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;
    let memory = state
        .get_user_earth(&query.user_id)
        .map_err(AppError::Internal)?;
    let config = memory.read().alignment_config();
    Ok(Json(config))
}

/// `PUT /api/align` — replace the user's alignment config (sanitized + capability-
/// validated + `updated_at` stamped). Returns the resulting, sanitized config so the
/// caller sees clamped values; a config the build can't honor is rejected `400`.
pub async fn set_align(
    State(state): State<AppState>,
    Json(req): Json<SetAlignRequest>,
) -> Result<Json<AlignmentConfig>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;
    let result = {
        let guard = memory.read();
        guard
            .set_alignment_config(req.config)
            .map_err(|reason| AppError::InvalidInput {
                field: "config".to_string(),
                reason,
            })?
    };
    tracing::info!(
        user_id = %req.user_id,
        v2_enabled = result.v2_enabled,
        method = ?result.method,
        "Alignment config updated"
    );
    Ok(Json(result))
}
