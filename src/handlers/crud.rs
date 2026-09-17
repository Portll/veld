//! CRUD Handlers for Memory Operations
//!
//! Create, Read, Update, Delete operations for individual memories
//! and bulk delete operations (forget by age, importance, tags, etc.)

use axum::{
    extract::{Extension, Path, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use super::state::MultiUserMemoryManager;
use super::types::MemoryEvent;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{self, ExperienceType, Memory};
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// GET MEMORY RESPONSE (with hierarchy)
// =============================================================================

/// Response for GET /api/memories/{memory_id} - includes hierarchy context
#[derive(Debug, Serialize)]
pub struct MemoryWithHierarchy {
    /// The memory itself (flattened)
    #[serde(flatten)]
    pub memory: Memory,
    /// Children memory IDs (if any)
    pub children_ids: Vec<String>,
    /// Number of children
    pub children_count: usize,
    /// True when `experience.content` is the complete stored text. False when the record is
    /// compressed and `?decompress=1` was not asked for, or when the record is a legacy one whose
    /// body was replaced at storage time and cannot be restored; `content_note` says which.
    /// Measured 2026-09-02: `?full=true` and `?decompress=true` were silently ignored and a
    /// 410-byte preview came back as if it were a 19,396-byte document.
    pub full_content: bool,
    /// Whether the full content is recoverable at all (false only for legacy lossy records).
    pub lossless: bool,
    /// Why `full_content` is false, when it is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_note: Option<String>,
}

/// `?decompress=1|true` (alias `?full=`) asks GET /api/memory/{id} for the restored content.
fn wants_full_content(params: &HashMap<String, String>) -> bool {
    ["decompress", "full"]
        .iter()
        .filter_map(|k| params.get(*k))
        .any(|v| matches!(v.as_str(), "1" | "true"))
}

// =============================================================================
// LIST MEMORIES TYPES
// =============================================================================

/// Hard ceiling on `limit` for the memory-listing endpoints (`/api/list/{user_id}`,
/// `/api/memories`). Any `limit` is clamped to this to bound per-request memory and scan cost;
/// large exports page through with `offset`. Ported from shodh-memory 00c42ec (#407): the
/// previous cap of 1000 with no offset made every record past the first 1000 unreachable
/// through this API, and the response gave no sign of it.
const MAX_LIST_LIMIT: usize = 10_000;

/// Ceiling on the content preview in `ListMemoryItem::content`. `content_truncated` and
/// `content_length` say when the cut happened, so a caller cannot mistake the preview for the
/// record (shodh-memory 9233dcd).
const LIST_CONTENT_PREVIEW_CHARS: usize = 500;

/// Query parameters for listing memories
#[derive(Debug, Deserialize)]
pub struct ListQuery {
    /// Max items to return. Defaults to 100, clamped to [`MAX_LIST_LIMIT`].
    pub limit: Option<usize>,
    /// Items to skip before collecting `limit` results. Defaults to 0. Combine with `total`
    /// in the response to page past `MAX_LIST_LIMIT`.
    pub offset: Option<usize>,
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    /// Text search query - filters by content or tags (case-insensitive)
    pub query: Option<String>,
}

/// List response - simplified memory list
#[derive(Debug, Serialize)]
pub struct ListResponse {
    pub memories: Vec<ListMemoryItem>,
    /// Total matching the filters, independent of `limit`/`offset`: compare
    /// `memories.len() + offset` against it to know whether pages remain.
    pub total: usize,
}

/// Request for POST /api/memories - list memories with user_id in body
#[derive(Debug, Deserialize)]
pub struct ListMemoriesRequest {
    pub user_id: String,
    /// Max items to return. Defaults to 100, clamped to [`MAX_LIST_LIMIT`].
    pub limit: Option<usize>,
    /// Items to skip before collecting `limit` results. Defaults to 0.
    pub offset: Option<usize>,
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    pub query: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListMemoryItem {
    pub id: String,
    /// Content preview, cut at [`LIST_CONTENT_PREVIEW_CHARS`]. Check `content_truncated`
    /// before treating it as the record; GET /api/memory/{id} returns the whole text.
    pub content: String,
    /// True when `content` is shorter than the stored content.
    pub content_truncated: bool,
    /// Character length of the full stored content.
    pub content_length: usize,
    pub memory_type: String,
    pub importance: f32,
    pub tags: Vec<String>,
    pub created_at: String,
    pub tier: String,
}

/// One list row from a memory, with the preview cut declared rather than silent.
fn list_item(m: &crate::memory::Memory) -> ListMemoryItem {
    let content_length = m.experience.content.chars().count();
    ListMemoryItem {
        id: m.id.0.to_string(),
        content: m
            .experience
            .content
            .chars()
            .take(LIST_CONTENT_PREVIEW_CHARS)
            .collect(),
        content_truncated: content_length > LIST_CONTENT_PREVIEW_CHARS,
        content_length,
        memory_type: format!("{:?}", m.experience.experience_type),
        importance: m.importance(),
        tags: m.experience.entities.clone(),
        created_at: m.created_at.to_rfc3339(),
        tier: format!("{:?}", m.tier),
    }
}

// =============================================================================
// UPDATE/DELETE RESPONSE TYPES
// =============================================================================

/// Request for updating memory content
#[derive(Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    pub user_id: String,
    pub content: String,
    pub embeddings: Option<Vec<f32>>,
}

/// Response for memory update operations
#[derive(Debug, Serialize)]
pub struct UpdateMemoryResponse {
    pub success: bool,
    pub id: String,
    pub message: String,
}

/// Response for memory delete operations
#[derive(Debug, Serialize)]
pub struct DeleteMemoryResponse {
    pub success: bool,
    pub id: String,
    pub message: String,
}

// =============================================================================
// FORGET REQUEST TYPES (local - not in shared types.rs)
// =============================================================================

/// Forget a single memory by ID (POST body variant)
#[derive(Debug, Deserialize)]
pub struct ForgetByIdRequest {
    pub user_id: String,
    pub memory_id: String,
}

/// Forget memories by age
#[derive(Debug, Deserialize)]
pub struct ForgetByAgeRequest {
    pub user_id: String,
    pub days_old: u32,
}

/// Forget memories by importance threshold
#[derive(Debug, Deserialize)]
pub struct ForgetByImportanceRequest {
    pub user_id: String,
    pub threshold: f32,
}

/// Forget memories matching a pattern
#[derive(Debug, Deserialize)]
pub struct ForgetByPatternRequest {
    pub user_id: String,
    pub pattern: String,
}

/// Forget memories by tags
#[derive(Debug, Deserialize)]
pub struct ForgetByTagsRequest {
    pub user_id: String,
    /// Tags to match for deletion (deletes memories matching ANY of these tags)
    pub tags: Vec<String>,
}

/// Forget memories by date range
#[derive(Debug, Deserialize)]
pub struct ForgetByDateRequest {
    pub user_id: String,
    /// Start of date range (inclusive) - ISO 8601 format
    pub start: chrono::DateTime<chrono::Utc>,
    /// End of date range (inclusive) - ISO 8601 format
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Bulk delete memories by filters
#[derive(Debug, Deserialize)]
pub struct BulkDeleteRequest {
    pub user_id: String,
    /// Delete memories matching ANY of these tags
    pub tags: Option<Vec<String>>,
    /// Delete memories of this type
    pub memory_type: Option<String>,
    /// Delete memories created after this timestamp
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    /// Delete memories created before this timestamp
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
}

/// Clear ALL memories for a user (GDPR compliance)
#[derive(Debug, Deserialize)]
pub struct ClearAllRequest {
    pub user_id: String,
    /// Safety confirmation - must be "CONFIRM" to proceed
    pub confirm: String,
}

/// PATCH endpoint for partial memory updates
#[derive(Debug, Deserialize)]
pub struct PatchMemoryRequest {
    pub user_id: String,
    /// New content (optional)
    pub content: Option<String>,
    /// New/additional tags (optional)
    pub tags: Option<Vec<String>>,
    /// New memory type (optional)
    pub memory_type: Option<String>,
}

// =============================================================================
// GET MEMORY HANDLER
// =============================================================================

/// GET /api/memories/{memory_id} - Get specific memory by ID
/// Returns memory with hierarchy context (parent_id in memory, children_ids in response)
#[tracing::instrument(skip(state))]
pub async fn get_memory(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(memory_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<MemoryWithHierarchy>, AppError> {
    let user_id = resolve_request_user_id(
        params.get("user_id").map(String::as_str),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;

    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let memory_guard = memory.read();

    let shared_memory = resolve_memory(&memory_guard, &memory_id)?;
    let memory_obj = (*shared_memory).clone();
    let resolved_id = shared_memory.id.clone();

    // The response says what its content IS. A compressed record is returned as stored unless
    // the caller asks for the restored body; a legacy lossy record is never passed off as full.
    let lossless = memory_guard.memory_is_lossless(&memory_obj);
    let (memory_obj, full_content, content_note) = if !memory_obj.compressed {
        (memory_obj, true, None)
    } else if !lossless {
        (
            memory_obj,
            false,
            Some(
                "written by the pre-fix lossy compressor: the body was replaced by a summary at \
                 storage time and cannot be restored"
                    .to_string(),
            ),
        )
    } else if wants_full_content(&params) {
        match memory_guard.decompress_memory(&memory_obj) {
            Ok(restored) => (restored, true, None),
            Err(e) => (
                memory_obj,
                false,
                Some(format!("decompression failed: {e}")),
            ),
        }
    } else {
        (
            memory_obj,
            false,
            Some("compressed; pass ?decompress=1 for the restored content".to_string()),
        )
    };

    // Fetch children for hierarchy context
    let children = memory_guard
        .get_memory_children(&resolved_id)
        .unwrap_or_default();

    let children_ids: Vec<String> = children.iter().map(|c| c.id.0.to_string()).collect();
    let children_count = children_ids.len();

    Ok(Json(MemoryWithHierarchy {
        memory: memory_obj,
        children_ids,
        children_count,
        full_content,
        lossless,
        content_note,
    }))
}

// =============================================================================
// LIST MEMORIES HANDLER
// =============================================================================

/// GET /api/list/{user_id} - List all memories for a user
/// Query params: ?limit=100&type=Decision
#[tracing::instrument(skip(state), fields(user_id = %user_id))]
pub async fn list_memories(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<Json<ListResponse>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let all_memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_all_memories()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Filter by type if specified
    let mut filtered: Vec<_> = if let Some(ref type_filter) = query.memory_type {
        let type_lower = type_filter.to_lowercase();
        all_memories
            .into_iter()
            .filter(|m| format!("{:?}", m.experience.experience_type).to_lowercase() == type_lower)
            .collect()
    } else {
        all_memories
    };

    // Filter by text query if specified (search in content and tags)
    if let Some(ref text_query) = query.query {
        let query_lower = text_query.to_lowercase();
        filtered.retain(|m| {
            // Check content
            if m.experience.content.to_lowercase().contains(&query_lower) {
                return true;
            }
            // Check tags/entities
            for tag in &m.experience.entities {
                if tag.to_lowercase().contains(&query_lower) {
                    return true;
                }
            }
            false
        });
    }

    let total = filtered.len();
    let limit = query.limit.unwrap_or(100).min(MAX_LIST_LIMIT);
    let offset = query.offset.unwrap_or(0);

    let memories: Vec<ListMemoryItem> = filtered
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|m| list_item(&m))
        .collect();

    Ok(Json(ListResponse { memories, total }))
}

/// POST /api/memories - List memories (user_id in body)
/// Alternative to GET /api/list/{user_id} for clients that prefer POST
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn list_memories_post(
    State(state): State<AppState>,
    Json(req): Json<ListMemoriesRequest>,
) -> Result<Json<ListResponse>, AppError> {
    list_memories_inner(state, req).await
}

/// Query parameters for GET /api/memories
#[derive(Debug, Deserialize)]
pub struct ListMemoriesQuery {
    pub user_id: String,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    pub query: Option<String>,
}

/// GET /api/memories?user_id=...&limit=... - List memories via query params
/// Cloudflare Worker compatibility alias for POST /api/memories
#[tracing::instrument(skip(state), fields(user_id = %params.user_id))]
pub async fn list_memories_get(
    State(state): State<AppState>,
    Query(params): Query<ListMemoriesQuery>,
) -> Result<Json<ListResponse>, AppError> {
    let req = ListMemoriesRequest {
        user_id: params.user_id,
        limit: params.limit,
        offset: params.offset,
        memory_type: params.memory_type,
        query: params.query,
    };
    list_memories_inner(state, req).await
}

/// Shared implementation for both POST and GET list_memories
async fn list_memories_inner(
    state: AppState,
    req: ListMemoriesRequest,
) -> Result<Json<ListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let all_memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_all_memories()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Filter by type if specified
    let mut filtered: Vec<_> = if let Some(ref type_filter) = req.memory_type {
        let type_lower = type_filter.to_lowercase();
        all_memories
            .into_iter()
            .filter(|m| format!("{:?}", m.experience.experience_type).to_lowercase() == type_lower)
            .collect()
    } else {
        all_memories
    };

    // Filter by text query if specified (search in content and tags)
    if let Some(ref text_query) = req.query {
        let query_lower = text_query.to_lowercase();
        filtered.retain(|m| {
            // Check content
            if m.experience.content.to_lowercase().contains(&query_lower) {
                return true;
            }
            // Check tags/entities
            for tag in &m.experience.entities {
                if tag.to_lowercase().contains(&query_lower) {
                    return true;
                }
            }
            false
        });
    }

    let total = filtered.len();
    let limit = req.limit.unwrap_or(100).min(MAX_LIST_LIMIT);
    let offset = req.offset.unwrap_or(0);

    let memories: Vec<ListMemoryItem> = filtered
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|m| list_item(&m))
        .collect();

    Ok(Json(ListResponse { memories, total }))
}

// =============================================================================
// UPDATE MEMORY HANDLER
// =============================================================================

/// PUT /api/memories/{memory_id} - Update memory content
#[tracing::instrument(skip(state), fields(memory_id = %memory_id))]
pub async fn update_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<Json<UpdateMemoryResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    if let Some(ref emb) = req.embeddings {
        validation::validate_embeddings(emb)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let shared_memory = resolve_memory(&memory_guard, &memory_id)?;
    let mut current_memory = (*shared_memory).clone();
    let resolved_id_str = current_memory.id.0.to_string();

    let content_preview: String = req.content.chars().take(50).collect();

    current_memory.experience.content = req.content;
    if let Some(emb) = req.embeddings {
        current_memory.experience.embeddings = Some(emb);
    } else {
        // Clear embeddings so they're regenerated by the vector index
        current_memory.experience.embeddings = None;
    }

    // Update in-place instead of creating a duplicate via remember()
    memory_guard
        .update_memory(&current_memory)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "UPDATE",
        &resolved_id_str,
        &format!("Updated memory content: {content_preview}"),
    );

    Ok(Json(UpdateMemoryResponse {
        success: true,
        id: resolved_id_str,
        message: "Memory updated successfully".to_string(),
    }))
}

// =============================================================================
// DELETE MEMORY HANDLER
// =============================================================================

/// DELETE /api/memories/{memory_id} - Delete specific memory
#[tracing::instrument(skip(state), fields(memory_id = %memory_id))]
pub async fn delete_memory(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(memory_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<DeleteMemoryResponse>, AppError> {
    // SEC: resolve user_id against the authenticated tenant binding so a
    // multi-tenant caller cannot delete another tenant's memory via ?user_id=.
    let user_id = resolve_request_user_id(
        params.get("user_id").map(String::as_str),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;

    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let memory_guard = memory.read();

    let shared_memory = resolve_memory(&memory_guard, &memory_id)?;
    let resolved_id = shared_memory.id.clone();
    let resolved_id_str = resolved_id.0.to_string();

    memory_guard
        .forget(memory::ForgetCriteria::ById(resolved_id))
        .map_err(AppError::Internal)?;

    // W5: journal the delete. See `handlers::remember` for the full
    // rationale — RocksDB already holds the canonical removal; the
    // journal frame is what lets the SQLite projection (and future
    // projections) converge via replay.
    let journal_payload = crate::intent_log::IntentPayload::Forget {
        user_id: user_id.to_string(),
        memory_id: resolved_id_str.clone(),
        schema_version: Some(crate::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
    };
    if let Err(e) = state.journal_and_apply(&user_id, &journal_payload) {
        tracing::warn!(
            user_id = %user_id,
            memory_id = %resolved_id_str,
            error = %e,
            "intent-log journal_and_apply (Forget) failed (RocksDB delete already durable)",
        );
    }

    state.log_event(&user_id, "DELETE", &resolved_id_str, "Memory deleted");

    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.to_string(),
        memory_id: Some(resolved_id_str.clone()),
        content_preview: None,
        memory_type: None,
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    Ok(Json(DeleteMemoryResponse {
        success: true,
        id: resolved_id_str,
        message: "Memory deleted successfully".to_string(),
    }))
}

// =============================================================================
// FORGET BY ID (POST BODY VARIANT)
// =============================================================================

/// POST /api/forget - Delete a single memory by ID from JSON body
///
/// Convenience endpoint matching the POST pattern of other forget endpoints
/// (/api/forget/age, /api/forget/tags, etc.). Delegates to the same logic as
/// DELETE /api/forget/{memory_id}.
#[tracing::instrument(skip(state), fields(memory_id = %req.memory_id))]
pub async fn forget_by_id(
    State(state): State<AppState>,
    Json(req): Json<ForgetByIdRequest>,
) -> Result<Json<DeleteMemoryResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    let memory_guard = memory.read();

    let shared_memory = resolve_memory(&memory_guard, &req.memory_id)?;
    let resolved_id = shared_memory.id.clone();
    let resolved_id_str = resolved_id.0.to_string();

    memory_guard
        .forget(memory::ForgetCriteria::ById(resolved_id))
        .map_err(AppError::Internal)?;

    // W5: journal the delete (see `delete_memory` for rationale).
    let journal_payload = crate::intent_log::IntentPayload::Forget {
        user_id: req.user_id.clone(),
        memory_id: resolved_id_str.clone(),
        schema_version: Some(crate::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
    };
    if let Err(e) = state.journal_and_apply(&req.user_id, &journal_payload) {
        tracing::warn!(
            user_id = %req.user_id,
            memory_id = %resolved_id_str,
            error = %e,
            "intent-log journal_and_apply (Forget) failed (RocksDB delete already durable)",
        );
    }

    state.log_event(&req.user_id, "DELETE", &resolved_id_str, "Memory deleted");

    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(resolved_id_str.clone()),
        content_preview: None,
        memory_type: None,
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    Ok(Json(DeleteMemoryResponse {
        success: true,
        id: resolved_id_str,
        message: "Memory deleted successfully".to_string(),
    }))
}

// =============================================================================
// PATCH MEMORY HANDLER
// =============================================================================

/// PATCH /api/memories/{memory_id} - Partial memory update
#[tracing::instrument(skip(state), fields(memory_id = %memory_id))]
pub async fn patch_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<PatchMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let shared_memory = resolve_memory(&memory_guard, &memory_id)?;
    let mut current_memory = (*shared_memory).clone();
    let resolved_id_str = current_memory.id.0.to_string();
    let mut changes = Vec::new();

    // Update content if provided
    if let Some(ref new_content) = req.content {
        validation::validate_content(new_content, false).map_validation_err("content")?;
        current_memory.experience.content = new_content.clone();
        current_memory.experience.embeddings = None;
        changes.push("content");
    }

    // Update tags if provided (add to existing entities)
    if let Some(ref new_tags) = req.tags {
        for tag in new_tags {
            if !current_memory.experience.entities.contains(tag) {
                current_memory.experience.entities.push(tag.clone());
            }
        }
        changes.push("tags");
    }

    // Update type if provided
    if let Some(ref type_str) = req.memory_type {
        current_memory.experience.experience_type = parse_experience_type(type_str)?;
        changes.push("type");
    }

    if changes.is_empty() {
        return Err(AppError::InvalidInput {
            field: "body".to_string(),
            reason: "No fields to update provided".to_string(),
        });
    }

    // Update in-place instead of creating a duplicate via remember()
    memory_guard
        .update_memory(&current_memory)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "PATCH",
        &resolved_id_str,
        &format!("Updated fields: {}", changes.join(", ")),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "id": resolved_id_str,
        "updated_fields": changes
    })))
}

// =============================================================================
// FORGET BY AGE HANDLER
// =============================================================================

/// C5 (W-017): journal a `Forget` frame for every canonical-store key a
/// bulk forget actually removed. The RocksDB removal is already durable;
/// these frames keep the intent log — and every projection / PITR replay
/// built on it — from resurrecting deleted memories. Per-projection apply
/// errors are swallowed (the frame is durable; replay retries on restart).
fn journal_bulk_forget(
    state: &AppState,
    user_id: &str,
    removed: impl IntoIterator<Item = memory::MemoryId>,
) {
    for id in removed {
        let payload = crate::intent_log::IntentPayload::Forget {
            user_id: user_id.to_string(),
            memory_id: id.0.to_string(),
            schema_version: Some(crate::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
        };
        if let Err(e) = state.journal_and_apply(user_id, &payload) {
            tracing::warn!(
                user_id = %user_id,
                memory_id = %id.0,
                error = %e,
                "intent-log journal_and_apply (bulk Forget) failed (RocksDB delete already durable)",
            );
        }
    }
}

/// POST /api/forget/age - Forget memories older than N days
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_age(
    State(state): State<AppState>,
    Json(req): Json<ForgetByAgeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (count, removed) = memory_guard
        .forget_collecting_removed(memory::ForgetCriteria::OlderThan(req.days_old))
        .map_err(AppError::Internal)?;
    drop(memory_guard);
    journal_bulk_forget(&state, &req.user_id, removed);

    state.log_event(
        &req.user_id,
        "FORGET_BY_AGE",
        &format!("{} days", req.days_old),
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "criteria": format!("older than {} days", req.days_old)
    })))
}

// =============================================================================
// FORGET BY IMPORTANCE HANDLER
// =============================================================================

/// POST /api/forget/importance - Forget memories below importance threshold
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_importance(
    State(state): State<AppState>,
    Json(req): Json<ForgetByImportanceRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.threshold < 0.0 || req.threshold > 1.0 {
        return Err(AppError::InvalidInput {
            field: "threshold".to_string(),
            reason: "Must be between 0.0 and 1.0".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (count, removed) = memory_guard
        .forget_collecting_removed(memory::ForgetCriteria::LowImportance(req.threshold))
        .map_err(AppError::Internal)?;
    drop(memory_guard);
    journal_bulk_forget(&state, &req.user_id, removed);

    state.log_event(
        &req.user_id,
        "FORGET_BY_IMPORTANCE",
        &format!("threshold {}", req.threshold),
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "criteria": format!("importance < {}", req.threshold)
    })))
}

// =============================================================================
// FORGET BY PATTERN HANDLER
// =============================================================================

/// POST /api/forget/pattern - Forget memories matching a pattern
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_pattern(
    State(state): State<AppState>,
    Json(req): Json<ForgetByPatternRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Validate regex pattern to prevent ReDoS attacks
    if req.pattern.len() > 1000 {
        return Err(AppError::InvalidInput {
            field: "pattern".to_string(),
            reason: "pattern too long (max 1000 chars)".to_string(),
        });
    }
    if regex::Regex::new(&req.pattern).is_err() {
        return Err(AppError::InvalidInput {
            field: "pattern".to_string(),
            reason: "invalid regex pattern".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (count, removed) = memory_guard
        .forget_collecting_removed(memory::ForgetCriteria::Pattern(req.pattern.clone()))
        .map_err(AppError::Internal)?;
    drop(memory_guard);
    journal_bulk_forget(&state, &req.user_id, removed);

    state.log_event(
        &req.user_id,
        "FORGET_BY_PATTERN",
        &req.pattern,
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "pattern": req.pattern
    })))
}

// =============================================================================
// FORGET BY TAGS HANDLER
// =============================================================================

/// POST /api/forget/tags - Forget memories matching any of the provided tags
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_tags(
    State(state): State<AppState>,
    Json(req): Json<ForgetByTagsRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.tags.is_empty() {
        return Err(AppError::InvalidInput {
            field: "tags".to_string(),
            reason: "At least one tag must be provided".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let (deleted_count, removed) = memory_guard
        .forget_collecting_removed(memory::ForgetCriteria::ByTags(req.tags.clone()))
        .map_err(AppError::Internal)?;
    drop(memory_guard);
    journal_bulk_forget(&state, &req.user_id, removed);

    info!(
        "🏷️ Forget by tags: user={}, tags={:?}, deleted={}",
        req.user_id, req.tags, deleted_count
    );

    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(format!("tags: {:?}", req.tags)),
        memory_type: None,
        importance: None,
        count: Some(deleted_count),
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "tags": req.tags
    })))
}

// =============================================================================
// FORGET BY DATE HANDLER
// =============================================================================

/// POST /api/forget/date - Forget memories within a date range
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_date(
    State(state): State<AppState>,
    Json(req): Json<ForgetByDateRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.end < req.start {
        return Err(AppError::InvalidInput {
            field: "end".to_string(),
            reason: "End date must be after start date".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let (deleted_count, removed) = memory_guard
        .forget_collecting_removed(memory::ForgetCriteria::ByDateRange {
            start: req.start,
            end: req.end,
        })
        .map_err(AppError::Internal)?;
    drop(memory_guard);
    journal_bulk_forget(&state, &req.user_id, removed);

    info!(
        "📅 Forget by date: user={}, start={}, end={}, deleted={}",
        req.user_id, req.start, req.end, deleted_count
    );

    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(format!(
            "{} to {}",
            req.start.format("%Y-%m-%d"),
            req.end.format("%Y-%m-%d")
        )),
        memory_type: None,
        importance: None,
        count: Some(deleted_count),
        entities: None,
        results: None,
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "start": req.start.to_rfc3339(),
        "end": req.end.to_rfc3339()
    })))
}

// =============================================================================
// BULK DELETE HANDLER
// =============================================================================

/// POST /api/bulk_delete - Bulk delete memories by filters
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn bulk_delete_memories(
    State(state): State<AppState>,
    Json(req): Json<BulkDeleteRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let mut total_count = 0;

    // Delete by tags if specified
    if let Some(ref tags) = req.tags {
        if !tags.is_empty() {
            let count = memory_guard
                .forget(memory::ForgetCriteria::ByTags(tags.clone()))
                .map_err(AppError::Internal)?;
            total_count += count;
        }
    }

    // Delete by type if specified
    if let Some(ref type_str) = req.memory_type {
        let exp_type = parse_experience_type(type_str)?;
        let count = memory_guard
            .forget(memory::ForgetCriteria::ByType(exp_type))
            .map_err(AppError::Internal)?;
        total_count += count;
    }

    // Delete by date range if specified
    if req.created_after.is_some() || req.created_before.is_some() {
        let start = req
            .created_after
            .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC);
        let end = req.created_before.unwrap_or(chrono::Utc::now());
        let count = memory_guard
            .forget(memory::ForgetCriteria::ByDateRange { start, end })
            .map_err(AppError::Internal)?;
        total_count += count;
    }

    state.log_event(
        &req.user_id,
        "BULK_DELETE",
        "multiple",
        &format!("Deleted {total_count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": total_count
    })))
}

// =============================================================================
// CLEAR ALL HANDLER (GDPR)
// =============================================================================

/// POST /api/clear_all - Clear ALL memories for a user (GDPR compliance)
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn clear_all_memories(
    State(state): State<AppState>,
    Json(req): Json<ClearAllRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Safety check - require explicit confirmation
    if req.confirm != "CONFIRM" {
        return Err(AppError::InvalidInput {
            field: "confirm".to_string(),
            reason: "Must provide confirm: \"CONFIRM\" to clear all memories".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::All)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "CLEAR_ALL",
        "GDPR",
        &format!("GDPR erasure: deleted {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": count,
        "message": "All memories have been permanently deleted (GDPR erasure)"
    })))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Resolve a memory ID (full UUID or 8+ char hex prefix) to a concrete Memory.
///
/// Used by get/update/delete/patch handlers. Validates the input format,
/// then searches across all memory tiers via prefix matching.
fn resolve_memory(
    memory_guard: &memory::MemorySystem,
    memory_id_str: &str,
) -> Result<memory::SharedMemory, AppError> {
    validation::validate_memory_id_or_prefix(memory_id_str)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    memory_guard
        .find_memory_by_prefix(memory_id_str)
        .map_err(|e| {
            let msg = e.to_string();
            if msg.starts_with("Ambiguous") {
                // Parse count from error message "...matches N memories"
                let count = msg
                    .rsplit("matches ")
                    .next()
                    .and_then(|s| s.split(' ').next())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                AppError::AmbiguousMemoryId {
                    prefix: memory_id_str.to_string(),
                    count,
                }
            } else {
                AppError::Internal(e)
            }
        })?
        .ok_or_else(|| AppError::MemoryNotFound(memory_id_str.to_string()))
}

// =============================================================================
// MEMORY HEALTH ENDPOINT (FIX-02)
// =============================================================================

/// Memory health response — exposes internal state Claude can't normally see
#[derive(Debug, Serialize)]
pub struct MemoryHealthResponse {
    pub memory_id: String,
    pub importance: f32,
    pub tier: String,
    pub access_count: u32,
    pub created_at: String,
    pub last_accessed: Option<String>,
    /// Feedback momentum EMA (-1.0 to +1.0)
    pub feedback_momentum: Option<f32>,
    /// Number of feedback signals received
    pub feedback_signal_count: Option<u32>,
    /// Edge count from this memory's entities
    pub edge_count: usize,
    /// Health classification: "healthy", "at_risk", "degraded"
    pub health_status: String,
    /// Estimated days until memory falls below retrieval threshold
    pub estimated_days_to_decay: Option<f32>,
}

/// GET /api/memory/{memory_id}/health — Expose memory health metrics
///
/// Returns importance, tier, feedback momentum, edge count, and decay estimate.
/// FIX-02: Breaks the one-way teaching relationship by giving Claude visibility
/// into which memories are fragile, consolidated, or degraded.
pub async fn get_memory_health(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(memory_id_str): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<MemoryHealthResponse>, AppError> {
    // SEC: resolve user_id against the tenant binding (see delete_memory) so a
    // multi-tenant caller cannot read another tenant's memory health via ?user_id=.
    let user_id = resolve_request_user_id(
        params.get("user_id").map(String::as_str),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let mem = {
        let guard = memory.read();
        resolve_memory(&guard, &memory_id_str)?
    };

    let importance = mem.importance();
    let tier = format!("{:?}", mem.tier);
    let access_count = mem.access_count();
    let created_at = mem.created_at.to_rfc3339();
    let last_accessed = Some(mem.last_accessed().to_rfc3339());

    // FIX-02 wiring: Look up feedback momentum from FeedbackStore
    let memory_id_typed = crate::memory::types::MemoryId(
        uuid::Uuid::parse_str(&memory_id_str)
            .map_err(|_| AppError::InvalidMemoryId(memory_id_str.clone()))?,
    );
    let (feedback_momentum, feedback_signal_count) = {
        let store = state.feedback_store.read();
        match store.get_momentum(&memory_id_typed) {
            Some(momentum) => (Some(momentum.ema), Some(momentum.signal_count)),
            None => (None, None),
        }
    };

    // Calculate health status (incorporate feedback momentum)
    let health_status = if importance >= 0.5 && feedback_momentum.unwrap_or(0.0) >= -0.2 {
        "healthy"
    } else if importance >= 0.2 {
        "at_risk"
    } else {
        "degraded"
    }
    .to_string();

    // Estimate days to decay below retrieval threshold (importance 0.05)
    let estimated_days = if importance > 0.05 {
        let ratio = importance / 0.05;
        Some(ratio.powf(2.0))
    } else {
        None
    };

    Ok(Json(MemoryHealthResponse {
        memory_id: memory_id_str,
        importance,
        tier,
        access_count,
        created_at,
        last_accessed,
        feedback_momentum,
        feedback_signal_count,
        edge_count: mem.entity_refs.len(),
        health_status,
        estimated_days_to_decay: estimated_days,
    }))
}

// =============================================================================
// ANCHOR ENDPOINT
// =============================================================================

/// Request to anchor or unanchor a memory
#[derive(Debug, Deserialize)]
pub struct AnchorRequest {
    pub user_id: String,
    pub memory_id: String,
    /// true to anchor, false to unanchor
    #[serde(default = "default_anchor_true")]
    pub anchor: bool,
}

fn default_anchor_true() -> bool {
    true
}

/// Response for anchor operations
#[derive(Debug, Serialize)]
pub struct AnchorResponse {
    pub success: bool,
    pub memory_id: String,
    pub anchored: bool,
}

/// POST /api/anchor — Mark a memory as compaction-resistant.
///
/// Anchored memories maintain a minimum importance floor (`ANCHOR_IMPORTANCE_FLOOR`)
/// and are skipped during compression. Use this for critical facts that must persist.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, memory_id = %req.memory_id))]
pub async fn anchor_memory(
    State(state): State<AppState>,
    Json(req): Json<AnchorRequest>,
) -> Result<Json<AnchorResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let uuid = uuid::Uuid::parse_str(&req.memory_id).map_err(|_| AppError::InvalidInput {
        field: "memory_id".to_string(),
        reason: "Invalid UUID format".to_string(),
    })?;
    let memory_id = memory::MemoryId(uuid);

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let anchor_value = req.anchor;
    let mid = memory_id.clone();
    let memory_sys_for_anchor = memory_sys.clone();
    let final_importance = tokio::task::spawn_blocking(move || {
        let guard = memory_sys_for_anchor.read();
        let mem = guard.get_memory(&mid)?;
        mem.set_anchored(anchor_value);
        guard.update_memory(&mem)?;
        // Sample the post-anchor importance so the projection can mirror
        // it without round-tripping the full Memory blob.
        Ok::<f32, anyhow::Error>(mem.importance())
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    // W5: journal the anchor as an `Anchor` payload so the SQLite
    // projection updates its `importance` column without rewriting the
    // memory blob. See `handlers::remember` for the broader contract.
    let journal_payload = crate::intent_log::IntentPayload::Anchor {
        user_id: req.user_id.clone(),
        memory_id: req.memory_id.clone(),
        importance: final_importance,
        schema_version: Some(crate::intent_log::payload::CURRENT_PAYLOAD_SCHEMA_VERSION),
    };
    if let Err(e) = state.journal_and_apply(&req.user_id, &journal_payload) {
        tracing::warn!(
            user_id = %req.user_id,
            memory_id = %req.memory_id,
            error = %e,
            "intent-log journal_and_apply (Anchor) failed (RocksDB anchor already durable)",
        );
    }

    state.emit_event(MemoryEvent {
        event_type: if req.anchor {
            "ANCHOR".to_string()
        } else {
            "UNANCHOR".to_string()
        },
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(req.memory_id.clone()),
        content_preview: None,
        memory_type: None,
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    Ok(Json(AnchorResponse {
        success: true,
        memory_id: req.memory_id,
        anchored: req.anchor,
    }))
}

/// POST /api/wiki/render — materialise / refresh the wiki for a user (W-018).
///
/// Authed: the user is resolved against the tenant binding via
/// `resolve_request_user_id`, so this never trusts a bare `?user_id=` — it
/// lives on the protected router, satisfying the
/// `public_router_has_no_per_user_handlers` contract. Returns 503 when wiki
/// render is disabled (`VELD_WIKI_RENDER` off). When enabled, opening the
/// journaled writer replays the intent log into the wiki projection on first
/// open (full backfill); subsequent live writes keep it current. Reports the
/// output directory and the on-disk page count.
#[tracing::instrument(skip_all)]
pub async fn wiki_render(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let user_id = resolve_request_user_id(
        params.get("user_id").map(String::as_str),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    if !crate::wiki::wiki_render_enabled() {
        return Err(AppError::ServiceUnavailable(
            "wiki render is disabled (set VELD_WIKI_RENDER=1 to enable)".to_string(),
        ));
    }

    // (b) Start the live edit-back watcher for this user (no-op unless
    // VELD_WIKI_WATCH is on). The Arc is the shared AppState the watcher
    // thread calls back into.
    state.ensure_wiki_watcher(state.clone(), &user_id);

    let (target_dir, pages, edits_applied) =
        state.render_wiki(&user_id).map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "success": true,
        "enabled": true,
        "target_dir": target_dir,
        "pages": pages,
        "edits_applied": edits_applied,
    })))
}

/// Request body for the tier-move endpoint.
#[derive(Debug, Deserialize)]
pub struct TierMoveRequest {
    pub user_id: String,
    pub memory_id: String,
    /// One of: "working", "session", "longterm" / "long_term", "archive"
    pub target_tier: String,
}

/// Response for tier-move operations.
#[derive(Debug, Serialize)]
pub struct TierMoveResponse {
    pub success: bool,
    pub memory_id: String,
    pub previous_tier: String,
    pub current_tier: String,
}

fn parse_memory_tier(s: &str) -> Result<memory::MemoryTier, AppError> {
    match s.to_lowercase().replace('_', "").as_str() {
        "working" | "hot" => Ok(memory::MemoryTier::Working),
        "session" => Ok(memory::MemoryTier::Session),
        "longterm" | "long" => Ok(memory::MemoryTier::LongTerm),
        "archive" | "cold" => Ok(memory::MemoryTier::Archive),
        other => Err(AppError::InvalidInput {
            field: "target_tier".to_string(),
            reason: format!(
                "Unknown tier '{other}' — expected working / session / longterm / archive"
            ),
        }),
    }
}

fn tier_to_str(tier: memory::MemoryTier) -> &'static str {
    match tier {
        memory::MemoryTier::Working => "working",
        memory::MemoryTier::Session => "session",
        memory::MemoryTier::LongTerm => "longterm",
        memory::MemoryTier::Archive => "archive",
    }
}

/// POST /api/memory/tier — Agent-initiated tier move.
///
/// Lets the agent explicitly promote a memory back to the working tier
/// (hot context) or push it down to archive. Differs from automatic
/// promotion/demotion (which only moves one step) by jumping directly to
/// the target tier.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, memory_id = %req.memory_id, target = %req.target_tier))]
pub async fn move_memory_tier(
    State(state): State<AppState>,
    Json(req): Json<TierMoveRequest>,
) -> Result<Json<TierMoveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let uuid = uuid::Uuid::parse_str(&req.memory_id).map_err(|_| AppError::InvalidInput {
        field: "memory_id".to_string(),
        reason: "Invalid UUID format".to_string(),
    })?;
    let memory_id = memory::MemoryId(uuid);
    let target_tier = parse_memory_tier(&req.target_tier)?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let mid = memory_id.clone();
    let previous_tier = tokio::task::spawn_blocking(move || {
        let guard = memory_sys.read();
        guard.move_to_tier(&mid, target_tier)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    state.emit_event(MemoryEvent {
        event_type: "TIER_MOVE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(req.memory_id.clone()),
        content_preview: None,
        memory_type: Some(tier_to_str(target_tier).to_string()),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    Ok(Json(TierMoveResponse {
        success: true,
        memory_id: req.memory_id,
        previous_tier: tier_to_str(previous_tier).to_string(),
        current_tier: tier_to_str(target_tier).to_string(),
    }))
}

fn parse_experience_type(type_str: &str) -> Result<ExperienceType, AppError> {
    match type_str.to_lowercase().as_str() {
        "observation" => Ok(ExperienceType::Observation),
        "decision" => Ok(ExperienceType::Decision),
        "learning" => Ok(ExperienceType::Learning),
        "error" => Ok(ExperienceType::Error),
        "discovery" => Ok(ExperienceType::Discovery),
        "pattern" => Ok(ExperienceType::Pattern),
        "context" => Ok(ExperienceType::Context),
        "task" => Ok(ExperienceType::Task),
        "codeedit" | "code_edit" => Ok(ExperienceType::CodeEdit),
        "fileaccess" | "file_access" => Ok(ExperienceType::FileAccess),
        "search" => Ok(ExperienceType::Search),
        "command" => Ok(ExperienceType::Command),
        "conversation" => Ok(ExperienceType::Conversation),
        "intention" => Ok(ExperienceType::Intention),
        _ => Err(AppError::InvalidInput {
            field: "memory_type".to_string(),
            reason: format!("Invalid memory type: {type_str}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::handlers::test_helpers::{self, TestHarness};
    use crate::memory::Experience;
    use axum::http::StatusCode;

    fn remember_n(harness: &TestHarness, user_id: &str, contents: &[String]) {
        let memory = harness.manager.get_user_memory(user_id).unwrap();
        let guard = memory.read();
        for c in contents {
            guard
                .remember(
                    Experience {
                        content: c.clone(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
        }
    }

    /// Ported from shodh-memory 9233dcd: the list path used to cut content at 500 chars with
    /// no signal it was incomplete, so a preview read as the record.
    #[tokio::test]
    async fn list_memories_marks_truncated_content() {
        let harness = TestHarness::new();
        let user_id = "truncation-user";
        let long_content = "a".repeat(600);
        let short_content = "b".repeat(42);
        remember_n(
            &harness,
            user_id,
            &[long_content.clone(), short_content.clone()],
        );

        let req = test_helpers::get(&format!("/api/list/{user_id}?limit=10"));
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);

        let memories = body["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 2);

        let long_item = memories
            .iter()
            .find(|m| m["content_truncated"] == true)
            .expect("expected one truncated item");
        assert_eq!(long_item["content_length"], 600u64);
        assert_eq!(long_item["content"].as_str().unwrap().chars().count(), 500);

        let short_item = memories
            .iter()
            .find(|m| m["content_truncated"] == false)
            .expect("expected one non-truncated item");
        assert_eq!(short_item["content_length"], 42u64);
        assert_eq!(short_item["content"], short_content);
    }

    /// Ported from shodh-memory 00c42ec (#407): without `offset`, everything past `limit` was
    /// unreachable through this API and `total` was the only sign it existed.
    #[tokio::test]
    async fn list_memories_pages_with_offset_and_total() {
        let harness = TestHarness::new();
        let user_id = "paging-user";
        let contents: Vec<String> = (0..5).map(|i| format!("record number {i}")).collect();
        remember_n(&harness, user_id, &contents);

        let page = |offset: usize| {
            test_helpers::get(&format!("/api/list/{user_id}?limit=2&offset={offset}"))
        };

        let (status, p0) = test_helpers::send(harness.router(), page(0)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(p0["total"], 5u64);
        assert_eq!(p0["memories"].as_array().unwrap().len(), 2);

        let (_, p1) = test_helpers::send(harness.router(), page(2)).await;
        let (_, p2) = test_helpers::send(harness.router(), page(4)).await;
        assert_eq!(p1["memories"].as_array().unwrap().len(), 2);
        assert_eq!(p2["memories"].as_array().unwrap().len(), 1);

        // Three pages cover the five records exactly once.
        let mut seen: Vec<String> = [&p0, &p1, &p2]
            .iter()
            .flat_map(|p| p["memories"].as_array().unwrap().iter())
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 5, "pages must partition the result set");

        // Past the end is an empty page, not an error.
        let (status, p3) = test_helpers::send(harness.router(), page(6)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(p3["memories"].as_array().unwrap().len(), 0);
        assert_eq!(p3["total"], 5u64);
    }

    /// GET /api/memory/{id} says what its content is. An uncompressed record is full and
    /// lossless whether or not ?decompress is passed, and carries no note; the flag is parsed
    /// under both spellings.
    #[tokio::test]
    async fn get_memory_reports_full_content_and_honours_the_decompress_flag() {
        let harness = TestHarness::new();
        let user_id = "get-full-user";
        let content = "z".repeat(1200);
        let id = {
            let memory = harness.manager.get_user_memory(user_id).unwrap();
            let guard = memory.read();
            guard
                .remember(
                    Experience {
                        content: content.clone(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap()
        };

        for suffix in ["", "&decompress=1", "&full=true"] {
            let req = test_helpers::get(&format!("/api/memory/{}?user_id={user_id}{suffix}", id.0));
            let (status, body) = test_helpers::send(harness.router(), req).await;
            assert_eq!(status, StatusCode::OK, "{suffix}");
            assert_eq!(body["full_content"], true, "{suffix}");
            assert_eq!(body["lossless"], true, "{suffix}");
            assert!(
                body.get("content_note").is_none(),
                "{suffix}: {:?}",
                body["content_note"]
            );
            assert_eq!(
                body["experience"]["content"], content,
                "{suffix}: the whole body, not a preview"
            );
        }

        assert!(super::wants_full_content(
            &[("decompress".to_string(), "true".to_string())]
                .into_iter()
                .collect()
        ));
        assert!(!super::wants_full_content(
            &[("decompress".to_string(), "0".to_string())]
                .into_iter()
                .collect()
        ));
    }
}
