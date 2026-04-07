//! Context Block API Handlers
//!
//! CRUD endpoints for agent-editable context blocks (Letta-style mutable state).

use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

// =============================================================================
// REQUEST / RESPONSE TYPES
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct BlockQuery {
    pub user_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SetBlockRequest {
    pub user_id: String,
    pub content: String,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct BlockResponse {
    pub key: String,
    pub content: String,
    pub max_tokens: usize,
    pub updated_at: String,
    pub version: u32,
}

#[derive(Debug, Serialize)]
pub struct BlockListResponse {
    pub blocks: Vec<BlockResponse>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct BlockDeleteResponse {
    pub deleted: bool,
}

fn block_to_response(block: &crate::memory::ContextBlock) -> BlockResponse {
    BlockResponse {
        key: block.key.clone(),
        content: block.content.clone(),
        max_tokens: block.max_tokens,
        updated_at: block.updated_at.to_rfc3339(),
        version: block.version,
    }
}

// =============================================================================
// HANDLERS
// =============================================================================

/// GET /api/context/blocks?user_id=X — list all context blocks for a user
#[tracing::instrument(skip(state), fields(user_id = %query.user_id))]
pub async fn list_context_blocks(
    State(state): State<AppState>,
    Query(query): Query<BlockQuery>,
) -> Result<Json<BlockListResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    let store = state.context_block_store.clone();
    let user_id = query.user_id.clone();

    let blocks = tokio::task::spawn_blocking(move || store.list(&user_id))
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?;

    let entries: Vec<BlockResponse> = blocks.iter().map(block_to_response).collect();
    let total = entries.len();
    Ok(Json(BlockListResponse {
        blocks: entries,
        total,
    }))
}

/// GET /api/context/blocks/{key}?user_id=X — get a single context block
#[tracing::instrument(skip(state), fields(user_id = %query.user_id, key = %key))]
pub async fn get_context_block(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Query(query): Query<BlockQuery>,
) -> Result<Json<BlockResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    let store = state.context_block_store.clone();
    let user_id = query.user_id.clone();
    let block_key = key.clone();

    let block = tokio::task::spawn_blocking(move || store.get(&user_id, &block_key))
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?;

    match block {
        Some(b) => Ok(Json(block_to_response(&b))),
        None => Err(AppError::ContextBlockNotFound(key)),
    }
}

/// PUT /api/context/blocks/{key} — create or update a context block
#[tracing::instrument(skip(state, req), fields(user_id = %req.user_id, key = %key))]
pub async fn set_context_block(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Json(req): Json<SetBlockRequest>,
) -> Result<Json<BlockResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let store = state.context_block_store.clone();
    let user_id = req.user_id.clone();
    let content = req.content.clone();
    let max_tokens = req.max_tokens;
    let block_key = key;

    let block =
        tokio::task::spawn_blocking(move || store.set(&user_id, &block_key, &content, max_tokens))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
            .map_err(AppError::Internal)?;

    Ok(Json(block_to_response(&block)))
}

/// DELETE /api/context/blocks/{key}?user_id=X — delete a context block
#[tracing::instrument(skip(state), fields(user_id = %query.user_id, key = %key))]
pub async fn delete_context_block(
    State(state): State<AppState>,
    Path(key): Path<String>,
    Query(query): Query<BlockQuery>,
) -> Result<Json<BlockDeleteResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    let store = state.context_block_store.clone();
    let user_id = query.user_id.clone();
    let block_key = key;

    let deleted = tokio::task::spawn_blocking(move || store.delete(&user_id, &block_key))
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?;

    Ok(Json(BlockDeleteResponse { deleted }))
}
