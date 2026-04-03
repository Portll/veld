//! Facts API Handlers
//!
//! Handlers for semantic facts extracted from episodic memories.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{self, SemanticFact, TemporalFact};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

fn facts_default_limit() -> usize {
    50
}

/// Request for listing facts
#[derive(Debug, Deserialize)]
pub struct FactsListRequest {
    pub user_id: String,
    #[serde(default = "facts_default_limit")]
    pub limit: usize,
}

/// Request for searching facts
#[derive(Debug, Deserialize)]
pub struct FactsSearchRequest {
    pub user_id: String,
    pub query: String,
    #[serde(default = "facts_default_limit")]
    pub limit: usize,
}

/// Request for facts by entity
#[derive(Debug, Deserialize)]
pub struct FactsByEntityRequest {
    pub user_id: String,
    pub entity: String,
    #[serde(default = "facts_default_limit")]
    pub limit: usize,
}

/// Response containing facts
#[derive(Debug, Serialize)]
pub struct FactsResponse {
    pub facts: Vec<SemanticFact>,
    pub total: usize,
}

/// POST /api/facts/list - List semantic facts for a user
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn list_facts(
    State(state): State<AppState>,
    Json(req): Json<FactsListRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_facts(&user_id, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/search - Search facts by keyword
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
pub async fn search_facts(
    State(state): State<AppState>,
    Json(req): Json<FactsSearchRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let query = req.query.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.search_facts(&user_id, &query, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/by-entity - Get facts related to an entity
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, entity = %req.entity))]
pub async fn facts_by_entity(
    State(state): State<AppState>,
    Json(req): Json<FactsByEntityRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let entity = req.entity.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_facts_by_entity(&user_id, &entity, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/stats - Get statistics about stored facts
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn get_facts_stats(
    State(state): State<AppState>,
    Json(req): Json<FactsListRequest>,
) -> Result<Json<memory::FactStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();

    let stats = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_fact_stats(&user_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

// =============================================================================
// TEMPORAL FACTS
// =============================================================================

/// Request for listing/filtering temporal facts
#[derive(Debug, Deserialize)]
pub struct TemporalFactsRequest {
    pub user_id: String,
    pub entity: Option<String>,
    pub event: Option<String>,
    #[serde(default = "facts_default_limit")]
    pub limit: usize,
    /// When true, include facts that have been invalidated by contradiction detection.
    /// Default: false (only return currently-valid facts).
    #[serde(default)]
    pub include_expired: bool,
}

/// Request for searching temporal facts
#[derive(Debug, Deserialize)]
pub struct TemporalFactsSearchRequest {
    pub user_id: String,
    pub query: String,
    #[serde(default = "facts_default_limit")]
    pub limit: usize,
    /// When true, include facts that have been invalidated by contradiction detection.
    #[serde(default)]
    pub include_expired: bool,
}

/// A single temporal fact in the API response
#[derive(Debug, Serialize)]
pub struct TemporalFactEntry {
    pub entity: String,
    pub event: String,
    pub event_type: String,
    pub timestamp: String,
    pub source_memory_id: String,
    pub confidence: f32,
    pub source_text: String,
    /// When this fact became true (ISO 8601)
    pub valid_from: Option<String>,
    /// When this fact was invalidated (ISO 8601). None means still valid.
    pub valid_until: Option<String>,
    /// Whether this fact is currently valid
    pub is_valid: bool,
}

/// Response containing temporal facts
#[derive(Debug, Serialize)]
pub struct TemporalFactsResponse {
    pub facts: Vec<TemporalFactEntry>,
    pub total: usize,
}

fn temporal_fact_to_entry(fact: &TemporalFact) -> TemporalFactEntry {
    let is_valid = fact.valid_until.is_none()
        || fact.valid_until.is_some_and(|until| until >= chrono::Utc::now());
    TemporalFactEntry {
        entity: fact.entity.clone(),
        event: fact.event.clone(),
        event_type: format!("{:?}", fact.event_type),
        timestamp: fact.resolved_time.to_sortable_string(),
        source_memory_id: fact.source_memory_id.0.to_string(),
        confidence: fact.confidence,
        source_text: fact.source_text.clone(),
        valid_from: fact.valid_from.map(|t| t.to_rfc3339()),
        valid_until: fact.valid_until.map(|t| t.to_rfc3339()),
        is_valid,
    }
}

/// POST /api/facts/temporal - List temporal facts with optional entity/event filters
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn list_temporal_facts(
    State(state): State<AppState>,
    Json(req): Json<TemporalFactsRequest>,
) -> Result<Json<TemporalFactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let entity = req.entity.clone();
    let event = req.event.clone();
    let limit = req.limit;
    let include_expired = req.include_expired;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        match (entity.as_deref(), event.as_deref()) {
            (Some(entity), Some(event)) => {
                let keywords: Vec<&str> = event.split_whitespace().collect();
                memory_guard.find_temporal_facts_filtered(
                    &user_id,
                    entity,
                    &keywords,
                    None,
                    include_expired,
                )
            }
            (Some(entity), None) => memory_guard.find_temporal_facts_by_entity_filtered(
                &user_id,
                entity,
                limit,
                include_expired,
            ),
            (None, Some(event)) => memory_guard.find_temporal_facts_by_event_filtered(
                &user_id,
                event,
                limit,
                include_expired,
            ),
            (None, None) => {
                memory_guard.list_temporal_facts_filtered(&user_id, limit, include_expired)
            }
        }
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let entries: Vec<TemporalFactEntry> = facts.iter().map(temporal_fact_to_entry).collect();
    let total = entries.len();
    Ok(Json(TemporalFactsResponse {
        facts: entries,
        total,
    }))
}

/// POST /api/facts/temporal/search - Semantic search across temporal facts
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
pub async fn search_temporal_facts(
    State(state): State<AppState>,
    Json(req): Json<TemporalFactsSearchRequest>,
) -> Result<Json<TemporalFactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let query = req.query.clone();
    let limit = req.limit;
    let include_expired = req.include_expired;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        // List all temporal facts then filter by query relevance
        let all_facts =
            memory_guard.list_temporal_facts_filtered(&user_id, 1000, include_expired)?;
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut matched: Vec<memory::TemporalFact> = all_facts
            .into_iter()
            .filter(|fact| {
                let entity_lower = fact.entity.to_lowercase();
                let event_lower = fact.event.to_lowercase();
                let source_lower = fact.source_text.to_lowercase();
                let time_str = fact.resolved_time.to_sortable_string().to_lowercase();

                // Match if any keyword appears in entity, event, source text, or timestamp
                keywords.iter().any(|kw| {
                    entity_lower.contains(kw)
                        || event_lower.contains(kw)
                        || source_lower.contains(kw)
                        || time_str.contains(kw)
                })
            })
            .collect();

        // Sort by relevance: count keyword hits
        matched.sort_by(|a, b| {
            let score = |fact: &memory::TemporalFact| -> usize {
                let combined = format!(
                    "{} {} {}",
                    fact.entity.to_lowercase(),
                    fact.event.to_lowercase(),
                    fact.source_text.to_lowercase()
                );
                keywords
                    .iter()
                    .filter(|kw| combined.contains(*kw))
                    .count()
            };
            score(b).cmp(&score(a))
        });

        matched.truncate(limit);
        Ok(matched)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let entries: Vec<TemporalFactEntry> = facts.iter().map(temporal_fact_to_entry).collect();
    let total = entries.len();
    Ok(Json(TemporalFactsResponse {
        facts: entries,
        total,
    }))
}
