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
    extract::{Extension, Query, State},
    response::Json,
};
use serde::Deserialize;

use super::state::MultiUserMemoryManager;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
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
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(query): Query<DialsQuery>,
) -> Result<Json<CognitiveDials>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&query.user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let memory = state
        .get_user_memory_async(&user_id)
        .await
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
        .get_user_memory_async(&req.user_id)
        .await
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

/// `GET /api/dials/schema` — default / recommended / absolute ranges for the recall
/// sliders, so the monitor can render each slider (track = absolute, shaded band =
/// recommended, tick = default) without hard-coding any bounds.
pub async fn get_dials_schema() -> Json<serde_json::Value> {
    let sliders: Vec<serde_json::Value> = crate::memory::dials::RECALL_SLIDER_SPECS
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "group": s.group,
                "label": s.label,
                "default": s.default,
                "recommended": [s.rec_lo, s.rec_hi],
                "absolute": [s.abs_lo, s.abs_hi],
                "step": s.step,
            })
        })
        .collect();
    Json(serde_json::json!({ "sliders": sliders }))
}
