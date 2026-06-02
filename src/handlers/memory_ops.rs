//! Identity-shifting memory operations: merge and split (plan J3).
//!
//! Both reuse the standard `remember` / `forget_by_id` handlers in-process,
//! so entity extraction, intent-log journaling, and every projection (SQLite,
//! Vamana, BM25, and the wiki render) apply unchanged — there is no second,
//! drift-prone write path. Lineage is recorded as tags (`merged-from:<id>`,
//! `split-from:<id>`, `split-part:<n>`) rather than a new `Memory` field, so
//! these land additively with no serialization migration.
//!
//! **W-021:** split operates on *character-index* ranges, never byte offsets,
//! so multibyte content can never panic or be cut mid-codepoint — the bug the
//! synthesis flagged in the plan's `content_range: 0..100` byte design.
//!
//! Anti-autophagy (W-022 sibling): a `derived` memory cannot be merged — a
//! filed answer must be regenerated, not folded into primary material.

use axum::{extract::State, response::Json};
use serde::Deserialize;

use super::crud::{forget_by_id, ForgetByIdRequest};
use super::recall::AppState;
use super::remember::{remember, RememberRequest, DERIVED_TAG};
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::types::MemoryId;
use crate::validation;

/// Character-safe substring by char index (W-021). Returns `None` for an
/// empty or out-of-bounds range instead of panicking — `content.chars()`
/// never slices mid-codepoint.
fn char_slice(content: &str, start: usize, end: usize) -> Option<String> {
    if start >= end {
        return None;
    }
    let chars: Vec<char> = content.chars().collect();
    if end > chars.len() {
        return None;
    }
    Some(chars[start..end].iter().collect())
}

/// Request to merge several memories into one.
#[derive(Debug, Deserialize)]
pub struct MergeRequest {
    pub user_id: String,
    pub ids: Vec<String>,
    /// Separator placed between merged bodies (default `\n\n---\n\n`).
    #[serde(default)]
    pub separator: Option<String>,
    /// Experience type for the merged memory (default `Context`).
    #[serde(default)]
    pub memory_type: Option<String>,
}

/// POST /api/memory/merge — merge N memories into one (J3).
///
/// Concatenates the source bodies, unions their tags, records each source as
/// a `merged-from:<id>` lineage tag, creates the merged memory via the normal
/// `remember` pipeline, then deletes the originals via `forget_by_id`. Refuses
/// to merge a `derived` memory (anti-autophagy).
#[tracing::instrument(skip_all)]
pub async fn merge_memories(
    State(state): State<AppState>,
    Json(req): Json<MergeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    if req.ids.len() < 2 {
        return Err(AppError::InvalidInput {
            field: "ids".to_string(),
            reason: "merge needs at least 2 memory ids".to_string(),
        });
    }

    let (content, tags) = {
        let earth = state
            .get_user_earth(&req.user_id)
            .map_err(AppError::Internal)?;
        let guard = earth.read();
        let mut bodies = Vec::with_capacity(req.ids.len());
        let mut tagset: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for id in &req.ids {
            let uuid = uuid::Uuid::parse_str(id.trim()).map_err(|_| AppError::InvalidInput {
                field: "ids".to_string(),
                reason: format!("invalid memory id {id}"),
            })?;
            let mem = guard
                .get_memory(&MemoryId(uuid))
                .map_err(|_| AppError::InvalidInput {
                    field: "ids".to_string(),
                    reason: format!("memory {id} not found"),
                })?;
            if mem.experience.tags.iter().any(|t| t == DERIVED_TAG) {
                return Err(AppError::InvalidInput {
                    field: "ids".to_string(),
                    reason: format!(
                        "cannot merge derived memory {id} (anti-autophagy): regenerate it instead"
                    ),
                });
            }
            bodies.push(mem.experience.content.clone());
            for t in &mem.experience.tags {
                tagset.insert(t.clone());
            }
            tagset.insert(format!("merged-from:{id}"));
        }
        let sep = req.separator.unwrap_or_else(|| "\n\n---\n\n".to_string());
        (bodies.join(&sep), tagset.into_iter().collect::<Vec<_>>())
    };

    let memory_type = req.memory_type.unwrap_or_else(|| "Context".to_string());
    let remember_req: RememberRequest = serde_json::from_value(serde_json::json!({
        "user_id": req.user_id,
        "content": content,
        "tags": tags,
        "memory_type": memory_type,
    }))
    .map_err(|e| AppError::Internal(anyhow::anyhow!("merge: build remember request: {e}")))?;

    let resp = remember(State(state.clone()), Json(remember_req)).await?;
    let merged_id = resp.0.id;

    for id in &req.ids {
        let freq = ForgetByIdRequest {
            user_id: req.user_id.clone(),
            memory_id: id.clone(),
        };
        let _ = forget_by_id(State(state.clone()), Json(freq)).await?;
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "merged_id": merged_id,
        "source_ids": req.ids,
    })))
}

/// Request to split one memory into several.
#[derive(Debug, Deserialize)]
pub struct SplitRequest {
    pub user_id: String,
    pub id: String,
    /// `[start, end)` ranges as **character** indices into the content
    /// (W-021 — never byte offsets).
    pub ranges: Vec<(usize, usize)>,
    /// Experience type for each child (default `Context`).
    #[serde(default)]
    pub memory_type: Option<String>,
}

/// POST /api/memory/split — split one memory into N children (J3, W-021).
///
/// Each range carves a child body by character index (multibyte-safe). Each
/// child inherits the parent's tags plus `split-from:<parent>` and
/// `split-part:<n>` lineage tags, is created via the normal `remember`
/// pipeline, and the parent is deleted via `forget_by_id`.
#[tracing::instrument(skip_all)]
pub async fn split_memory(
    State(state): State<AppState>,
    Json(req): Json<SplitRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    if req.ranges.is_empty() {
        return Err(AppError::InvalidInput {
            field: "ranges".to_string(),
            reason: "split needs at least one range".to_string(),
        });
    }
    let uuid = uuid::Uuid::parse_str(req.id.trim()).map_err(|_| AppError::InvalidInput {
        field: "id".to_string(),
        reason: "invalid memory id".to_string(),
    })?;

    let (content, parent_tags) = {
        let earth = state
            .get_user_earth(&req.user_id)
            .map_err(AppError::Internal)?;
        let guard = earth.read();
        let mem = guard
            .get_memory(&MemoryId(uuid))
            .map_err(|_| AppError::InvalidInput {
                field: "id".to_string(),
                reason: format!("memory {} not found", req.id),
            })?;
        (mem.experience.content.clone(), mem.experience.tags.clone())
    };

    let memory_type = req.memory_type.unwrap_or_else(|| "Context".to_string());
    let mut child_ids = Vec::with_capacity(req.ranges.len());
    for (i, (start, end)) in req.ranges.iter().enumerate() {
        let piece = char_slice(&content, *start, *end).ok_or_else(|| AppError::InvalidInput {
            field: "ranges".to_string(),
            reason: format!(
                "range [{start},{end}) is empty or out of bounds (content has {} chars)",
                content.chars().count()
            ),
        })?;
        let mut tags = parent_tags.clone();
        tags.push(format!("split-from:{}", req.id));
        tags.push(format!("split-part:{}", i + 1));
        let remember_req: RememberRequest = serde_json::from_value(serde_json::json!({
            "user_id": req.user_id,
            "content": piece,
            "tags": tags,
            "memory_type": memory_type,
        }))
        .map_err(|e| AppError::Internal(anyhow::anyhow!("split: build remember request: {e}")))?;
        let resp = remember(State(state.clone()), Json(remember_req)).await?;
        child_ids.push(resp.0.id);
    }

    let freq = ForgetByIdRequest {
        user_id: req.user_id.clone(),
        memory_id: req.id.clone(),
    };
    let _ = forget_by_id(State(state.clone()), Json(freq)).await?;

    Ok(Json(serde_json::json!({
        "success": true,
        "parent_id": req.id,
        "child_ids": child_ids,
    })))
}

#[cfg(test)]
mod tests {
    use super::char_slice;

    #[test]
    fn char_slice_is_multibyte_safe() {
        // Byte-offset slicing of this string would panic mid-codepoint;
        // char-index slicing (W-021) does not.
        let s = "café 日本語";
        assert_eq!(char_slice(s, 0, 4).as_deref(), Some("café"));
        assert_eq!(char_slice(s, 5, 8).as_deref(), Some("日本語"));
        assert_eq!(char_slice(s, 0, 100), None); // out of bounds -> None, no panic
        assert_eq!(char_slice(s, 3, 3), None); // empty range -> None
        assert_eq!(char_slice(s, 4, 2), None); // inverted -> None
    }
}
