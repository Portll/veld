//! Embedding spaces of one tenant's corpus, and the maps between them.
//!
//! `GET /api/spaces` lists every space the corpus holds and every installed map.
//! `POST /api/spaces/align` fits the map from one space into another on the memories that hold
//! vectors in both, and installs it. Both are tenant-bound: a `user_id` that does not match the
//! API key's binding is refused.

use std::sync::Arc;

use axum::{
    extract::{Extension, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::embeddings::alignment_spaces::BridgeReport;
use crate::embeddings::space::{RegisteredAlignment, SpaceId, SpaceSummary};
use crate::errors::{AppError, ValidationErrorExt};
use crate::validation;

pub type AppState = Arc<MultiUserMemoryManager>;

#[derive(Debug, Deserialize)]
pub struct SpacesQuery {
    pub user_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpacesResponse {
    pub user_id: String,
    pub index_space: SpaceId,
    pub spaces: Vec<SpaceSummary>,
    pub alignments: Vec<RegisteredAlignment>,
    /// The corpus's space migration, finished or under way, when one was ever recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub migration: Option<crate::memory::space_migration::SpaceMigration>,
}

/// GET /api/spaces?user_id= — the tenant's spaces and maps.
pub async fn list_spaces(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(q): Query<SpacesQuery>,
) -> Result<Json<SpacesResponse>, AppError> {
    let user_id = resolve_request_user_id(
        q.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let (index_space, spaces, alignments, migration) = tokio::task::spawn_blocking(move || {
        let guard = memory.read();
        anyhow::Ok((
            guard.index_space(),
            guard.embedding_spaces()?,
            guard.alignments(),
            guard.space_migration(),
        ))
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("spaces task failed: {e}")))?
    .map_err(AppError::Internal)?;
    Ok(Json(SpacesResponse {
        user_id,
        index_space,
        spaces,
        alignments,
        migration,
    }))
}

#[derive(Debug, Deserialize)]
pub struct RetireRequest {
    pub user_id: Option<String>,
    pub space: String,
}

/// POST /api/spaces/retire — drop a space's kept vectors, index and maps. Refused while any
/// memory's only vector is in it.
pub async fn retire_space(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Json(req): Json<RetireRequest>,
) -> Result<Json<crate::memory::retrieval::RetireReport>, AppError> {
    let user_id = resolve_request_user_id(
        req.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let space = parse_space("space", &req.space)?;
    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let manager = state.clone();
    let report = tokio::task::spawn_blocking(move || {
        let report = memory.read().retire_space(&space)?;
        super::remember::journal_stored(
            &manager,
            &memory,
            &user_id,
            &report.rewritten,
            super::remember::SnapshotKind::Update,
        );
        Ok::<_, anyhow::Error>(report)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("retire task failed: {e}")))?
    .map_err(|e| AppError::InvalidInput {
        field: "space".to_string(),
        reason: format!("{e:#}"),
    })?;
    Ok(Json(report))
}

#[derive(Debug, Deserialize)]
pub struct AlignRequest {
    pub user_id: Option<String>,
    /// The space queries or foreign vectors arrive in.
    pub from: String,
    /// The space they are searched in.
    pub to: String,
}

fn parse_space(field: &str, raw: &str) -> Result<SpaceId, AppError> {
    SpaceId::parse(raw).ok_or_else(|| AppError::InvalidInput {
        field: field.to_string(),
        reason: "must be <model>@<dimension>".to_string(),
    })
}

/// POST /api/spaces/align — fit and install the map from `from` into `to`. The fit reads every
/// memory once and runs off the async runtime.
pub async fn align_spaces(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Json(req): Json<AlignRequest>,
) -> Result<Json<BridgeReport>, AppError> {
    let user_id = resolve_request_user_id(
        req.user_id.as_deref(),
        authenticated_user.as_ref().map(|e| &e.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let from = parse_space("from", &req.from)?;
    let to = parse_space("to", &req.to)?;
    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let report = tokio::task::spawn_blocking(move || memory.read().fit_alignment(&from, &to))
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("alignment fit task failed: {e}")))?
        .map_err(|e| AppError::InvalidInput {
            field: "from".to_string(),
            reason: format!("{e:#}"),
        })?;
    Ok(Json(report))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    #[test]
    fn both_space_routes_are_mounted_behind_auth() {
        let src = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/handlers/router.rs"),
        )
        .expect("router.rs must be readable");
        let protected_at = src
            .find("pub fn build_protected_routes")
            .expect("build_protected_routes must exist");
        for handler in [
            "get(spaces::list_spaces)",
            "post(spaces::align_spaces)",
            "post(spaces::retire_space)",
        ] {
            let at = src.find(handler).expect("route must be mounted");
            assert!(at > protected_at, "{handler} must be behind auth");
        }
    }
}
