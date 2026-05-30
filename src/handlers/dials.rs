//! Cognitive Dials API — the runtime tuning surface (GET/PUT).
//!
//! `GET /api/dials?user_id=…` returns a user's current [`CognitiveDials`];
//! `PUT /api/dials` replaces them (full-replace REST semantics — the client GETs,
//! edits, and PUTs the whole struct). Values are sanitized server-side
//! ([`CognitiveDials::sanitized`]) and `updated_at` is stamped. Both routes live on
//! the **protected** router and bind to `user_id` (per the
//! `public_router_has_no_per_user_handlers` test); persistence of the new value is
//! wired separately (KV store).
//!
//! This is the manual/operator counterpart to the automated tuning Sleight pushes via
//! [`super::external_dimensions`] — both drive the same substrate.

use axum::{
    extract::{Query, State},
    response::Json,
};
use serde::Deserialize;

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::CognitiveDials;
use crate::validation;

/// Application state type alias.
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Query for `GET /api/dials`.
#[derive(Debug, Deserialize)]
pub struct DialsQuery {
    pub user_id: String,
}

/// Body for `PUT /api/dials` (full replace).
#[derive(Debug, Deserialize)]
pub struct SetDialsRequest {
    pub user_id: String,
    pub dials: CognitiveDials,
}

/// `GET /api/dials?user_id=…` — return the user's current cognitive dials.
pub async fn get_dials(
    State(state): State<AppState>,
    Query(query): Query<DialsQuery>,
) -> Result<Json<CognitiveDials>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;
    let memory = state
        .get_user_earth(&query.user_id)
        .map_err(AppError::Internal)?;
    let dials = memory.read().cognitive_dials();
    Ok(Json(dials))
}

/// `PUT /api/dials` — replace the user's cognitive dials (sanitized + `updated_at`
/// stamped). Returns the resulting, sanitized dials so the caller sees clamped values.
pub async fn set_dials(
    State(state): State<AppState>,
    Json(req): Json<SetDialsRequest>,
) -> Result<Json<CognitiveDials>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;
    let guard = memory.read();
    guard.set_cognitive_dials(req.dials);
    let result = guard.cognitive_dials();
    tracing::info!(
        user_id = %req.user_id,
        reconsolidation_enabled = result.reconsolidation_enabled,
        "Cognitive dials updated"
    );
    Ok(Json(result))
}
