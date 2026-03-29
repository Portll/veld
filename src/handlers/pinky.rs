//! Pinky Dimension Push Handlers
//!
//! Accepts dimension scores from the Pinky evaluation engine via HTTP POST.
//! Pinky computes topological health metrics (density, coherence, closure,
//! confidence, isotropy) from gap analysis + Voronoi decomposition on shodh's
//! graph API. These scores modulate retrieval scoring in Layer 5.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::types::PinkyDimensionScores;
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Request body for POST /api/pinky/dimensions
#[derive(Debug, Deserialize)]
pub struct PinkyDimensionsRequest {
    pub user_id: String,
    /// Entity density in the relevant region (0.0 = sparse, 1.0 = saturated)
    pub density: f32,
    /// Semantic coherence of neighbors (0.0 = unrelated, 1.0 = tight cluster)
    pub coherence: f32,
    /// Fraction of potential triangles closed (0.0 = all open, 1.0 = fully closed)
    pub closure: f32,
    /// Average edge confidence in the region (0.0 = weak, 1.0 = strong)
    pub confidence: f32,
    /// Directional balance of knowledge (0.0 = anisotropic, 1.0 = isotropic)
    pub isotropy: f32,
}

/// Response body for POST /api/pinky/dimensions
#[derive(Debug, Serialize)]
pub struct PinkyDimensionsResponse {
    pub success: bool,
}

/// POST /api/pinky/dimensions
///
/// Accept dimension scores from Pinky and store them on the user's MemorySystem.
/// These scores are consumed during retrieval (Layer 5) as a global quality
/// multiplier via `pinky_aggregate_score()`.
pub async fn push_dimensions(
    State(state): State<AppState>,
    Json(req): Json<PinkyDimensionsRequest>,
) -> Result<Json<PinkyDimensionsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Validate score ranges (all must be 0.0..=1.0)
    for (name, value) in [
        ("density", req.density),
        ("coherence", req.coherence),
        ("closure", req.closure),
        ("confidence", req.confidence),
        ("isotropy", req.isotropy),
    ] {
        if !(0.0..=1.0).contains(&value) {
            return Err(AppError::InvalidInput {
                field: name.to_string(),
                reason: format!("must be between 0.0 and 1.0, got {value}"),
            });
        }
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    let memory_guard = memory.read();

    let scores = PinkyDimensionScores {
        density: req.density,
        coherence: req.coherence,
        closure: req.closure,
        confidence: req.confidence,
        isotropy: req.isotropy,
        computed_at: Some(chrono::Utc::now()),
    };

    memory_guard.set_pinky_scores(scores);

    tracing::info!(
        user_id = %req.user_id,
        density = req.density,
        coherence = req.coherence,
        closure = req.closure,
        confidence = req.confidence,
        isotropy = req.isotropy,
        "Pinky dimension scores updated"
    );

    Ok(Json(PinkyDimensionsResponse { success: true }))
}
