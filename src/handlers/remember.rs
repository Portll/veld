//! Remember Handlers - Memory Storage Operations
//!
//! Core handlers for storing memories: remember, batch_remember, upsert.

use std::collections::HashSet;

use axum::{extract::State, response::Json};

use super::health::AppState;
use super::types::MemoryEvent;
use crate::earth::SharedEarth;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{
    types::{
        ChangeType, ContextId, EmotionalContext, EpisodeContext, NerEntityRecord, RichContext,
        SourceContext, SourceType,
    },
    Experience, ExperienceType, SessionEvent,
};
use crate::metrics;
use crate::validation;

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// Remember request - store a new memory
#[derive(Debug, serde::Deserialize)]
pub struct RememberRequest {
    pub user_id: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, alias = "experience_type")]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub external_id: Option<String>,
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub emotional_valence: Option<f32>,
    #[serde(default)]
    pub emotional_arousal: Option<f32>,
    #[serde(default)]
    pub emotion: Option<String>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub credibility: Option<f32>,
    #[serde(default)]
    pub episode_id: Option<String>,
    #[serde(default)]
    pub sequence_number: Option<u32>,
    #[serde(default)]
    pub preceding_memory_id: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub parent_agent_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    /// Parent memory ID for hierarchical organization
    /// Use this to create memory trees (e.g., "71-research" -> "algebraic" -> "21×27≡-1")
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Optional filename for format detection (e.g. "notes.md", "data.json").
    /// When provided, the content is run through the ingest text-extraction
    /// pipeline before storage.
    #[serde(default)]
    pub filename: Option<String>,
    /// Semantic intent tags from Claude (FIX-10)
    /// Merged into entity tags with "intent:" prefix for importance boost
    #[serde(default)]
    pub intent_tags: Vec<String>,
}

/// Remember response
#[derive(Debug, serde::Serialize)]
pub struct RememberResponse {
    pub id: String,
    pub success: bool,
}

/// Batch remember request
#[derive(Debug, serde::Deserialize)]
pub struct BatchRememberRequest {
    pub user_id: String,
    pub memories: Vec<BatchMemoryItem>,
    #[serde(default)]
    pub options: BatchRememberOptions,
}

/// Options for batch remember
#[derive(Debug, serde::Deserialize, Clone, Default)]
pub struct BatchRememberOptions {
    #[serde(default = "default_true")]
    pub extract_entities: bool,
    #[serde(default = "default_true")]
    pub create_edges: bool,
}

fn default_true() -> bool {
    true
}

/// Single item in batch remember
#[derive(Debug, serde::Deserialize, Clone)]
pub struct BatchMemoryItem {
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, alias = "experience_type")]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub emotional_valence: Option<f32>,
    #[serde(default)]
    pub emotional_arousal: Option<f32>,
    #[serde(default)]
    pub emotion: Option<String>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub credibility: Option<f32>,
    #[serde(default)]
    pub episode_id: Option<String>,
    #[serde(default)]
    pub sequence_number: Option<u32>,
    #[serde(default)]
    pub preceding_memory_id: Option<String>,
    /// Parent memory ID for hierarchical organization
    #[serde(default)]
    pub parent_id: Option<String>,
}

/// Error detail for batch item
#[derive(Debug, serde::Serialize)]
pub struct BatchErrorItem {
    pub index: usize,
    pub error: String,
}

/// Batch remember response
#[derive(Debug, serde::Serialize)]
pub struct BatchRememberResponse {
    pub created: usize,
    pub failed: usize,
    pub memory_ids: Vec<String>,
    pub errors: Vec<BatchErrorItem>,
}

/// Upsert request - create or update memory
#[derive(Debug, serde::Deserialize)]
pub struct UpsertRequest {
    pub user_id: String,
    pub external_id: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, alias = "experience_type")]
    pub memory_type: Option<String>,
    #[serde(default = "default_change_type")]
    pub change_type: String,
    #[serde(default)]
    pub changed_by: Option<String>,
    #[serde(default)]
    pub change_reason: Option<String>,
}

fn default_change_type() -> String {
    "content_updated".to_string()
}

/// Upsert response
#[derive(Debug, serde::Serialize)]
pub struct UpsertResponse {
    pub id: String,
    pub success: bool,
    pub was_update: bool,
    pub version: u32,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Parse memory type from string
pub fn parse_experience_type(s: Option<&String>) -> ExperienceType {
    s.and_then(|s| match s.to_lowercase().as_str() {
        "observation" => Some(ExperienceType::Observation),
        "decision" => Some(ExperienceType::Decision),
        "learning" => Some(ExperienceType::Learning),
        "error" => Some(ExperienceType::Error),
        "discovery" => Some(ExperienceType::Discovery),
        "pattern" => Some(ExperienceType::Pattern),
        "context" => Some(ExperienceType::Context),
        "task" => Some(ExperienceType::Task),
        "codeedit" | "code_edit" => Some(ExperienceType::CodeEdit),
        "fileaccess" | "file_access" => Some(ExperienceType::FileAccess),
        "search" => Some(ExperienceType::Search),
        "command" => Some(ExperienceType::Command),
        "conversation" => Some(ExperienceType::Conversation),
        "intention" => Some(ExperienceType::Intention),
        _ => None,
    })
    .unwrap_or(ExperienceType::Observation)
}

/// Parse source type from string
pub fn parse_source_type(s: Option<&String>) -> SourceType {
    s.map(|s| match s.to_lowercase().as_str() {
        "user" => SourceType::User,
        "system" => SourceType::System,
        "api" | "external_api" => SourceType::ExternalApi,
        "file" => SourceType::File,
        "web" => SourceType::Web,
        "ai_generated" | "ai" => SourceType::AiGenerated,
        "inferred" => SourceType::Inferred,
        _ => SourceType::Unknown,
    })
    .unwrap_or(SourceType::Unknown)
}

/// Build RichContext from request fields
#[allow(clippy::too_many_arguments)]
pub fn build_rich_context(
    emotional_valence: Option<f32>,
    emotional_arousal: Option<f32>,
    emotion: Option<String>,
    source_type: Option<String>,
    credibility: Option<f32>,
    episode_id: Option<String>,
    sequence_number: Option<u32>,
    preceding_memory_id: Option<String>,
) -> Option<RichContext> {
    let has_context = emotional_valence.is_some()
        || emotional_arousal.is_some()
        || emotion.is_some()
        || source_type.is_some()
        || credibility.is_some()
        || episode_id.is_some()
        || sequence_number.is_some()
        || preceding_memory_id.is_some();

    if !has_context {
        return None;
    }

    let emotional = EmotionalContext {
        valence: emotional_valence.unwrap_or(0.0),
        arousal: emotional_arousal.unwrap_or(0.0),
        dominant_emotion: emotion,
        confidence: if emotional_valence.is_some() || emotional_arousal.is_some() {
            0.8
        } else {
            0.0
        },
        ..Default::default()
    };

    let source = SourceContext {
        source_type: parse_source_type(source_type.as_ref()),
        credibility: credibility.unwrap_or(0.8),
        ..Default::default()
    };

    let episode = EpisodeContext {
        episode_id,
        sequence_number,
        preceding_memory_id,
        is_episode_start: sequence_number == Some(1),
        ..Default::default()
    };

    let now = chrono::Utc::now();
    Some(RichContext {
        id: ContextId(uuid::Uuid::new_v4()),
        emotional,
        source,
        episode,
        repository: Default::default(),
        conversation: Default::default(),
        user: Default::default(),
        project: Default::default(),
        temporal: Default::default(),
        semantic: Default::default(),
        code: Default::default(),
        document: Default::default(),
        environment: Default::default(),
        parent: None,
        embeddings: None,
        decay_rate: 1.0,
        created_at: now,
        updated_at: now,
    })
}

// =============================================================================
// HANDLERS
// =============================================================================

/// Remember a single memory
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn remember(
    State(state): State<AppState>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    // ── Ingest pipeline: extract clean text from structured formats ──────
    let (processed_content, ingest_tags) = if req.filename.is_some() || {
        // Auto-detect non-plaintext content even without filename
        let fmt = crate::ingest::detect_format(None, &req.content);
        fmt != crate::ingest::InputFormat::PlainText
    } {
        let format =
            crate::ingest::detect_format(req.filename.as_deref(), &req.content);
        if format != crate::ingest::InputFormat::PlainText {
            match crate::ingest::extract_text(&req.content, format) {
                Ok(extracted) => {
                    let mut tags = extracted.metadata.entities_hint;
                    tags.push(format!("format:{}", extracted.metadata.format));
                    (extracted.text, tags)
                }
                Err(e) => {
                    tracing::debug!("Ingest extraction failed, using raw content: {}", e);
                    (req.content.clone(), vec![])
                }
            }
        } else {
            (req.content.clone(), vec![])
        }
    } else {
        (req.content.clone(), vec![])
    };

    let experience_type = parse_experience_type(req.memory_type.as_ref());

    // PERF: Run NER and YAKE extraction in parallel using spawn_blocking
    // Both are CPU-bound and independent - parallelization reduces latency by ~40%
    let ner = state.get_neural_ner();
    let yake = state.get_keyword_extractor();
    let content_for_ner = processed_content.clone();
    let content_for_yake = processed_content.clone();

    let (ner_result, yake_result) = tokio::join!(
        // NER extraction (named entities: Person, Org, Location, Misc)
        // Preserve full entity records for downstream graph insertion with proper labels
        tokio::task::spawn_blocking(move || {
            match ner.extract(&content_for_ner) {
                Ok(entities) => entities
                    .into_iter()
                    .map(|e| NerEntityRecord {
                        text: e.text,
                        entity_type: e.entity_type.as_str().to_string(),
                        confidence: e.confidence,
                        start_char: Some(e.start),
                        end_char: Some(e.end),
                    })
                    .collect::<Vec<NerEntityRecord>>(),
                Err(e) => {
                    tracing::debug!("NER extraction failed: {}", e);
                    Vec::new()
                }
            }
        }),
        // YAKE extraction (keywords: common nouns, verbs, etc.)
        // Captures important terms like "sunrise", "painting", "lake"
        tokio::task::spawn_blocking(move || yake.extract_texts(&content_for_yake))
    );

    let ner_entities = match ner_result {
        Ok(entities) => entities,
        Err(e) => {
            if e.is_panic() {
                tracing::error!("NER extraction task panicked: {:?}", e);
            } else {
                tracing::debug!("NER extraction task cancelled: {:?}", e);
            }
            Vec::new()
        }
    };
    let extracted_keywords = match yake_result {
        Ok(keywords) => keywords,
        Err(e) => {
            if e.is_panic() {
                tracing::error!("YAKE extraction task panicked: {:?}", e);
            } else {
                tracing::debug!("YAKE extraction task cancelled: {:?}", e);
            }
            Vec::new()
        }
    };

    let mut merged_entities: Vec<String> = req.tags.clone();
    // FIX-10: Merge semantic intent tags with "intent:" prefix for importance boost
    for tag in &req.intent_tags {
        merged_entities.push(format!("intent:{tag}"));
    }
    let mut seen: HashSet<String> = merged_entities.iter().map(|t| t.to_lowercase()).collect();
    // Merge ingest-derived entity hints (JSON keys, CSV columns, code symbols)
    for tag in &ingest_tags {
        if seen.insert(tag.to_lowercase()) {
            merged_entities.push(tag.clone());
        }
    }
    for record in &ner_entities {
        if seen.insert(record.text.to_lowercase()) {
            merged_entities.push(record.text.clone());
        }
    }
    for keyword in extracted_keywords {
        if seen.insert(keyword.to_lowercase()) {
            merged_entities.push(keyword);
        }
    }

    // Cold-start entity extraction: when the graph is sparse and NER found few entities,
    // use aggressive heuristics (proper nouns, CamelCase, acronyms, file paths, versions)
    let graph_entity_count = state
        .get_user_graph(&req.user_id)
        .ok()
        .map(|g| g.read().get_stats().map(|s| s.entity_count).unwrap_or(0))
        .unwrap_or(0);
    if graph_entity_count < crate::constants::ENTITY_COLD_START_THRESHOLD
        && ner_entities.len() < 2
    {
        let cold_entities =
            crate::embeddings::ner::cold_start_extract_entities(&req.content, &merged_entities);
        if !cold_entities.is_empty() {
            tracing::debug!(
                "Cold-start extraction found {} additional entities (graph has {})",
                cold_entities.len(),
                graph_entity_count
            );
            for e in cold_entities {
                if seen.insert(e.text.to_lowercase()) {
                    merged_entities.push(e.text);
                }
            }
        }
    }

    if merged_entities.len() > validation::MAX_ENTITIES_PER_MEMORY {
        tracing::debug!(
            count = merged_entities.len(),
            max = validation::MAX_ENTITIES_PER_MEMORY,
            "Capping entities to maximum allowed"
        );
        merged_entities.truncate(validation::MAX_ENTITIES_PER_MEMORY);
    }

    let experience_type_str = format!("{:?}", experience_type);

    let context = build_rich_context(
        req.emotional_valence,
        req.emotional_arousal,
        req.emotion.clone(),
        req.source_type.clone(),
        req.credibility,
        req.episode_id.clone(),
        req.sequence_number,
        req.preceding_memory_id.clone(),
    );

    let experience = Experience {
        content: processed_content.clone(),
        experience_type,
        entities: merged_entities.clone(),
        tags: merged_entities,
        context,
        ner_entities,
        ..Default::default()
    };

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_id = {
        let memory = memory.clone();
        let exp_clone = experience.clone();
        let created_at = req.created_at;
        let agent_id = req.agent_id.clone();
        let run_id = req.run_id.clone();
        let external_id = req.external_id.clone();

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            // Fast path: persist + BM25 only (~10ms), embedding deferred to background
            if agent_id.is_some() || run_id.is_some() {
                memory_guard.remember_with_agent_deferred(
                    exp_clone, created_at, agent_id, run_id, external_id,
                )
            } else {
                memory_guard.remember_deferred(exp_clone, created_at, external_id)
            }
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Background: complete embedding + vector indexing.
    // The memory is already searchable via BM25; vector search follows in ~150ms.
    {
        let memory_for_bg = memory.clone();
        let mid = memory_id.clone();
        tokio::task::spawn_blocking(move || {
            let guard = memory_for_bg.read();
            if let Err(e) = guard.embed_and_index(&mid) {
                tracing::warn!(memory_id = %mid.0, "Background embed_and_index failed: {e}");
            }
        });
    }

    // Record metrics + session + broadcast BEFORE returning response (fast, <1ms)
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    let session_id = state.session_store().get_or_create_session(&req.user_id);
    state.session_store().add_event(
        &session_id,
        SessionEvent::MemoryCreated {
            timestamp: chrono::Utc::now(),
            memory_id: memory_id.0.to_string(),
            memory_type: experience_type_str.clone(),
            content_preview: req.content.chars().take(100).collect(),
            entities: req.tags.clone(),
        },
    );

    state.emit_event(MemoryEvent {
        event_type: "CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(req.content.chars().take(500).collect()),
        memory_type: Some(experience_type_str),
        importance: None,
        count: None,
        entities: if req.tags.is_empty() {
            None
        } else {
            Some(req.tags.clone())
        },
        results: None,
    });

    // IDEMPOTENCY FIX (issue #109): Return response IMMEDIATELY after persist.
    // The 4 post-processing tasks below are all non-fatal (log errors and continue)
    // and their results are never included in the response. Running them synchronously
    // caused 5-15s handler latency, exceeding the MCP client's 10s timeout and
    // triggering retries that created duplicate memories (31% duplication rate).
    // Now fire-and-forget: response returns in <200ms, post-tasks run in background.
    let response_id = memory_id.0.to_string();
    {
        let state = state.clone();
        let memory = memory.clone();
        let user_id = req.user_id.clone();
        let content = req.content.clone();
        let experience = experience.clone();
        let parent_id = req.parent_id.clone();
        let created_at = req.created_at;

        tokio::spawn(async move {
            // Task 1: Build episodic graph (entities + episode + relationships)
            {
                let state = state.clone();
                let user_id = user_id.clone();
                let experience = experience.clone();
                let mid = memory_id.clone();
                if let Err(e) = tokio::task::spawn_blocking(move || {
                    state.process_experience_into_graph(&user_id, &experience, &mid)
                })
                .await
                .unwrap_or_else(|e| Err(anyhow::anyhow!("Graph task panicked: {e}")))
                {
                    tracing::debug!("Graph processing failed (non-fatal): {}", e);
                }
            }

            // Task 2: Set parent_id for hierarchical organization
            if let Some(ref parent_id_str) = parent_id {
                let resolved_parent = if let Ok(parent_uuid) = uuid::Uuid::parse_str(parent_id_str)
                {
                    Some(crate::memory::MemoryId(parent_uuid))
                } else {
                    let mem = memory.clone();
                    let prefix = parent_id_str.clone();
                    match tokio::task::spawn_blocking(move || {
                        let guard = mem.read();
                        guard
                            .find_memory_by_prefix(&prefix)
                            .ok()
                            .flatten()
                            .map(|m| m.id.clone())
                    })
                    .await
                    {
                        Ok(result) => result,
                        Err(e) => {
                            tracing::debug!("Parent resolve panicked (non-fatal): {e}");
                            None
                        }
                    }
                };

                if let Some(resolved) = resolved_parent {
                    let mem = memory.clone();
                    let mid = memory_id.clone();
                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        let guard = mem.read();
                        guard.set_memory_parent(&mid, Some(resolved))
                    })
                    .await
                    {
                        tracing::debug!("Parent set task panicked (non-fatal): {e}");
                    }
                } else {
                    tracing::warn!("Could not resolve parent_id: {}", parent_id_str);
                }
            }

            // Task 3: Extract and store temporal facts
            {
                let mem = memory.clone();
                let uid = user_id.clone();
                let cnt = content.clone();
                let ents = experience.entities.clone();
                let ts = created_at.unwrap_or_else(chrono::Utc::now);
                let mid = memory_id.clone();

                if let Err(e) = tokio::task::spawn_blocking(move || {
                    let guard = mem.read();
                    guard.store_temporal_facts_for_memory(&uid, &mid, &cnt, &ents, ts)
                })
                .await
                {
                    tracing::debug!("Temporal fact extraction panicked (non-fatal): {e}");
                }
            }

            // Task 4: Infer causal lineage (runs after graph processing)
            {
                let graph_arc = state.get_user_graph(&user_id).ok();
                let memory_arc = memory.clone();
                let uid = user_id.clone();
                let mid = memory_id.clone();

                if let Err(e) = tokio::task::spawn_blocking(move || {
                    let Some(graph_arc) = graph_arc else {
                        return;
                    };
                    let graph = graph_arc.read();
                    let memory_guard = memory_arc.read();

                    let episode = match graph.get_episode(&mid.0) {
                        Ok(Some(ep)) => ep,
                        _ => return,
                    };

                    if episode.entity_refs.is_empty() {
                        return;
                    }

                    let mut candidate_ids = std::collections::HashSet::new();
                    let cutoff = chrono::Utc::now() - chrono::Duration::days(7);

                    for entity_uuid in &episode.entity_refs {
                        if let Ok(episodes) = graph.get_episodes_by_entity(entity_uuid) {
                            for ep in &episodes {
                                if ep.created_at >= cutoff {
                                    candidate_ids.insert(crate::memory::MemoryId(ep.uuid));
                                }
                            }
                        }
                    }

                    candidate_ids.remove(&mid);
                    let candidate_ids: Vec<_> = candidate_ids.into_iter().take(20).collect();
                    if candidate_ids.is_empty() {
                        return;
                    }

                    let candidates: Vec<_> = candidate_ids
                        .iter()
                        .filter_map(|id| memory_guard.get_memory(id).ok())
                        .collect();

                    let Ok(new_memory) = memory_guard.get_memory(&mid) else {
                        return;
                    };

                    match memory_guard.infer_lineage_for_memory(&uid, &new_memory, &candidates) {
                        Ok(edges) if !edges.is_empty() => {
                            tracing::info!(
                                user_id = %uid,
                                memory_id = %mid.0,
                                edges = edges.len(),
                                relations = ?edges.iter().map(|e| format!("{:?}", e.relation)).collect::<Vec<_>>(),
                                "Lineage inference: {} causal edges detected",
                                edges.len()
                            );
                        }
                        Ok(_) => {
                            tracing::debug!(
                                "Lineage inference: no causal edges for {} (checked {} candidates)",
                                mid.0,
                                candidates.len()
                            );
                        }
                        Err(e) => {
                            tracing::debug!("Lineage inference failed (non-fatal): {}", e);
                        }
                    }
                })
                .await
                {
                    tracing::debug!("Lineage inference panicked (non-fatal): {e}");
                }
            }

            // Task 5: Conflict detection — detect contradictions with existing memories (A3)
            {
                let state = state.clone();
                let memory_arc = memory.clone();
                let uid = user_id.clone();
                let mid = memory_id.clone();

                if let Err(e) = tokio::task::spawn_blocking(move || {
                    detect_and_resolve_conflicts(&state, &memory_arc, &uid, &mid)
                })
                .await
                .unwrap_or_else(|e| Err(anyhow::anyhow!("Conflict detection panicked: {e}")))
                {
                    tracing::debug!("Conflict detection failed (non-fatal): {}", e);
                }
            }
        });
    }

    Ok(Json(RememberResponse {
        id: response_id,
        success: true,
    }))
}

/// Detect and resolve contradictions between a new memory and existing memories (A3)
///
/// For each focal entity in the new memory, queries the graph for other memories
/// sharing that entity. If a candidate has high cosine similarity AND contains
/// a semantic contradiction, creates a SupersededBy edge and decays the old
/// memory's importance.
///
/// Reference: Bai et al. (2026) §4.2.3 — H-MEM weight regulation, WISE side-memory
fn detect_and_resolve_conflicts(
    state: &AppState,
    memory_arc: &SharedEarth,
    user_id: &str,
    new_memory_id: &crate::memory::MemoryId,
) -> anyhow::Result<()> {
    use crate::constants::{
        CONFLICT_COSINE_THRESHOLD, CONFLICT_IMPORTANCE_DECAY, CONFLICT_MAX_CANDIDATES,
        CONFLICT_MAX_FOCAL_ENTITIES,
    };
    use crate::memory::temporal_facts::{detect_semantic_opposition, text_similarity};
    use crate::similarity::cosine_similarity;

    let memory_guard = memory_arc.read();
    let new_memory = memory_guard.get_memory(new_memory_id)?;

    let new_embedding = match &new_memory.experience.embeddings {
        Some(emb) => emb.clone(),
        None => return Ok(()), // No embedding yet — skip conflict detection
    };

    let graph_arc = match state.get_user_graph(user_id) {
        Ok(g) => g,
        Err(_) => return Ok(()),
    };
    let graph = graph_arc.read();

    // Get entity UUIDs from the new memory's entity_refs
    let focal_entities: Vec<uuid::Uuid> = new_memory
        .entity_refs
        .iter()
        .take(CONFLICT_MAX_FOCAL_ENTITIES)
        .map(|e| e.entity_id)
        .collect();

    if focal_entities.is_empty() {
        return Ok(());
    }

    // Collect candidate memory IDs from graph episodes sharing focal entities
    let mut candidate_ids = std::collections::HashSet::new();
    for entity_uuid in &focal_entities {
        if let Ok(episodes) = graph.get_episodes_by_entity(entity_uuid) {
            for ep in episodes.iter().take(CONFLICT_MAX_CANDIDATES) {
                let mid = crate::memory::MemoryId(ep.uuid);
                if mid != *new_memory_id {
                    candidate_ids.insert(mid);
                }
            }
        }
    }

    if candidate_ids.is_empty() {
        return Ok(());
    }

    let new_content = &new_memory.experience.content;
    let mut conflicts_resolved = 0usize;

    for candidate_id in candidate_ids.iter().take(CONFLICT_MAX_CANDIDATES) {
        let mut candidate = match memory_guard.get_memory(candidate_id) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Check cosine similarity — only high-overlap memories can contradict
        let candidate_embedding = match &candidate.experience.embeddings {
            Some(emb) => emb,
            None => continue,
        };

        let sim = cosine_similarity(&new_embedding, candidate_embedding);
        if sim < CONFLICT_COSINE_THRESHOLD {
            continue;
        }

        // Check for text-level similarity + semantic opposition
        let text_sim = text_similarity(new_content, &candidate.experience.content);
        if text_sim < 0.5 {
            continue; // Not similar enough at text level to be contradictory
        }

        if !detect_semantic_opposition(new_content, &candidate.experience.content) {
            continue;
        }

        // Contradiction detected — create SupersededBy edge and decay old importance
        tracing::info!(
            user_id = %user_id,
            new_memory = %new_memory_id.0,
            old_memory = %candidate_id.0,
            cosine = format!("{:.3}", sim),
            "Conflict detected: creating SupersededBy edge"
        );

        // Decay old memory importance, set temporal expiry, and persist
        let old_importance = candidate.importance();
        candidate.set_importance(old_importance * CONFLICT_IMPORTANCE_DECAY);
        candidate.set_valid_until(chrono::Utc::now());
        memory_guard.update_memory(&candidate)?;

        // Create SupersededBy edge in graph
        let now = chrono::Utc::now();
        // Find entity pairs to link
        let old_entities: Vec<uuid::Uuid> = candidate.entity_refs.iter().map(|e| e.entity_id).collect();
        if let (Some(&new_ent), Some(&old_ent)) = (focal_entities.first(), old_entities.first()) {
            let edge = crate::graph_memory::RelationshipEdge {
                uuid: uuid::Uuid::new_v4(),
                from_entity: new_ent,
                to_entity: old_ent,
                relation_type: crate::graph_memory::RelationType::SupersededBy,
                strength: sim, // Use similarity as edge strength
                created_at: now,
                valid_at: now,
                invalidated_at: None,
                source_episode_id: Some(new_memory_id.0),
                context: format!(
                    "conflict-resolution: cosine={:.3}, text_sim={:.3}",
                    sim, text_sim
                ),
                last_activated: now,
                activation_count: 1,
                ltp_status: Default::default(),
                tier: Default::default(),
                activation_timestamps: None,
                entity_confidence: None,
                created_by: crate::graph_memory::EdgeSource::CoOccurrence,
                forward_strength: sim,
                backward_strength: sim,
            };
            let _ = graph.add_relationship(edge);
        }

        conflicts_resolved += 1;
    }

    if conflicts_resolved > 0 {
        tracing::info!(
            user_id = %user_id,
            memory_id = %new_memory_id.0,
            conflicts_resolved,
            "Conflict detection complete"
        );
    }

    Ok(())
}

/// Batch remember - store multiple memories at once
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, batch_size = req.memories.len()))]
pub async fn batch_remember(
    State(state): State<AppState>,
    Json(req): Json<BatchRememberRequest>,
) -> Result<Json<BatchRememberResponse>, AppError> {
    let op_start = std::time::Instant::now();
    let batch_size = req.memories.len();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.memories.is_empty() {
        return Ok(Json(BatchRememberResponse {
            created: 0,
            failed: 0,
            memory_ids: vec![],
            errors: vec![],
        }));
    }

    if req.memories.len() > 1000 {
        return Err(AppError::InvalidInput {
            field: "memories".to_string(),
            reason: "Batch size exceeds 1000 limit".to_string(),
        });
    }

    // Pre-validate all items
    let mut validation_errors: Vec<BatchErrorItem> = Vec::new();
    let mut valid_items: Vec<(usize, BatchMemoryItem)> = Vec::new();

    let mut seen_content: HashSet<u64> = HashSet::new();
    for (index, item) in req.memories.into_iter().enumerate() {
        if let Err(e) = validation::validate_content(&item.content, false) {
            validation_errors.push(BatchErrorItem {
                index,
                error: e.to_string(),
            });
            continue;
        }
        // Deduplicate within the batch: skip items with identical content
        let content_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            item.content.hash(&mut hasher);
            hasher.finish()
        };
        if !seen_content.insert(content_hash) {
            tracing::debug!(
                batch_index = index,
                "Skipping duplicate content in batch (same content already queued)"
            );
            validation_errors.push(BatchErrorItem {
                index,
                error: "Duplicate content within batch".to_string(),
            });
            continue;
        }
        valid_items.push((index, item));
    }

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let extract_entities = req.options.extract_entities;
    let neural_ner = state.get_neural_ner();
    let keyword_extractor = state.get_keyword_extractor();

    let create_edges = req.options.create_edges;

    // Build experiences
    let mut experiences_with_index: Vec<(
        usize,
        Experience,
        Option<chrono::DateTime<chrono::Utc>>,
    )> = Vec::with_capacity(valid_items.len());
    let mut parent_ids: std::collections::HashMap<usize, String> =
        std::collections::HashMap::new();

    for (index, item) in valid_items {
        if let Some(ref pid) = item.parent_id {
            parent_ids.insert(index, pid.clone());
        }
        let experience_type = parse_experience_type(item.memory_type.as_ref());

        let (merged_entities, ner_records) = if extract_entities {
            // NER for named entities (Person, Org, Location, Misc)
            let ner_records: Vec<NerEntityRecord> = match neural_ner.extract(&item.content) {
                Ok(entities) => entities
                    .into_iter()
                    .map(|e| NerEntityRecord {
                        text: e.text,
                        entity_type: e.entity_type.as_str().to_string(),
                        confidence: e.confidence,
                        start_char: Some(e.start),
                        end_char: Some(e.end),
                    })
                    .collect(),
                Err(e) => {
                    tracing::debug!("NER extraction failed for batch item {}: {}", index, e);
                    Vec::new()
                }
            };

            // YAKE for common nouns, verbs, concepts
            let extracted_keywords: Vec<String> = keyword_extractor.extract_texts(&item.content);

            let mut merged: Vec<String> = item.tags.clone();
            let mut seen: HashSet<String> = merged.iter().map(|t| t.to_lowercase()).collect();
            for record in &ner_records {
                if seen.insert(record.text.to_lowercase()) {
                    merged.push(record.text.clone());
                }
            }
            for keyword in extracted_keywords {
                if seen.insert(keyword.to_lowercase()) {
                    merged.push(keyword);
                }
            }
            if merged.len() > validation::MAX_ENTITIES_PER_MEMORY {
                tracing::debug!(
                    batch_index = index,
                    count = merged.len(),
                    max = validation::MAX_ENTITIES_PER_MEMORY,
                    "Capping entities to maximum allowed in batch item"
                );
                merged.truncate(validation::MAX_ENTITIES_PER_MEMORY);
            }
            (merged, ner_records)
        } else {
            (item.tags.clone(), Vec::new())
        };

        let context = build_rich_context(
            item.emotional_valence,
            item.emotional_arousal,
            item.emotion.clone(),
            item.source_type.clone(),
            item.credibility,
            item.episode_id.clone(),
            item.sequence_number,
            item.preceding_memory_id.clone(),
        );

        let experience = Experience {
            content: item.content,
            experience_type,
            entities: merged_entities.clone(),
            tags: merged_entities,
            context,
            ner_entities: ner_records,
            ..Default::default()
        };

        experiences_with_index.push((index, experience, item.created_at));
    }

    // Store memories
    let (memory_results, storage_errors) = {
        let memory = memory.clone();
        let experiences = experiences_with_index;
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let mut results: Vec<(usize, String, Experience)> =
                Vec::with_capacity(experiences.len());
            let mut errors: Vec<BatchErrorItem> = Vec::new();

            for (index, experience, created_at) in experiences {
                match memory_guard.remember(experience.clone(), created_at) {
                    Ok(id) => {
                        results.push((index, id.0.to_string(), experience));
                    }
                    Err(e) => {
                        errors.push(BatchErrorItem {
                            index,
                            error: e.to_string(),
                        });
                    }
                }
            }
            (results, errors)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    let memory_ids: Vec<String> = memory_results.iter().map(|(_, id, _)| id.clone()).collect();
    let created = memory_ids.len();

    let mut all_errors = validation_errors;
    all_errors.extend(storage_errors);
    all_errors.sort_by_key(|e| e.index);
    let failed = all_errors.len();

    // Build episodic graph and run post-processing in background (non-blocking).
    // Mirrors the single remember handler's 5-task pipeline to ensure batch items
    // get the same treatment: graph, parent_id, temporal facts, lineage, conflicts.
    {
        let state = state.clone();
        let user_id = req.user_id.clone();
        let memory = memory.clone();
        let memory_results_bg = memory_results;

        tokio::spawn(async move {
            // Task 1: Build episodic graph for each stored memory
            let mut batch_memory_ids: Vec<crate::memory::MemoryId> = Vec::new();
            for (_, id_str, experience) in &memory_results_bg {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    let memory_id = crate::memory::MemoryId(uuid);
                    let state = state.clone();
                    let uid = user_id.clone();
                    let exp = experience.clone();
                    let mid = memory_id.clone();
                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        state.process_experience_into_graph(&uid, &exp, &mid)
                    })
                    .await
                    .unwrap_or_else(|e| Err(anyhow::anyhow!("Graph task panicked: {e}")))
                    {
                        tracing::debug!(
                            "Graph processing failed for {} (non-fatal): {}",
                            id_str,
                            e
                        );
                    }
                    batch_memory_ids.push(memory_id);
                }
            }

            // Task 2: Set parent_id for hierarchical organization
            for (index, id_str, _) in &memory_results_bg {
                if let Some(parent_id_str) = parent_ids.get(index) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        let mid = crate::memory::MemoryId(uuid);
                        let resolved_parent =
                            if let Ok(parent_uuid) = uuid::Uuid::parse_str(parent_id_str) {
                                Some(crate::memory::MemoryId(parent_uuid))
                            } else {
                                let mem = memory.clone();
                                let prefix = parent_id_str.clone();
                                match tokio::task::spawn_blocking(move || {
                                    let guard = mem.read();
                                    guard
                                        .find_memory_by_prefix(&prefix)
                                        .ok()
                                        .flatten()
                                        .map(|m| m.id.clone())
                                })
                                .await
                                {
                                    Ok(result) => result,
                                    Err(e) => {
                                        tracing::debug!(
                                            "Parent resolve panicked (non-fatal): {e}"
                                        );
                                        None
                                    }
                                }
                            };

                        if let Some(resolved) = resolved_parent {
                            let mem = memory.clone();
                            if let Err(e) = tokio::task::spawn_blocking(move || {
                                let guard = mem.read();
                                guard.set_memory_parent(&mid, Some(resolved))
                            })
                            .await
                            {
                                tracing::debug!("Parent set task panicked (non-fatal): {e}");
                            }
                        } else {
                            tracing::warn!(
                                "Could not resolve parent_id: {}",
                                parent_id_str
                            );
                        }
                    }
                }
            }

            // Task 3: Extract and store temporal facts for each memory
            for (_, id_str, experience) in &memory_results_bg {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    let mem = memory.clone();
                    let uid = user_id.clone();
                    let cnt = experience.content.clone();
                    let ents = experience.entities.clone();
                    let ts = chrono::Utc::now();
                    let mid = crate::memory::MemoryId(uuid);

                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        let guard = mem.read();
                        guard.store_temporal_facts_for_memory(&uid, &mid, &cnt, &ents, ts)
                    })
                    .await
                    {
                        tracing::debug!(
                            "Temporal fact extraction panicked for {} (non-fatal): {e}",
                            id_str
                        );
                    }
                }
            }

            // Task 4: Infer causal lineage for each memory
            for (_, id_str, _) in &memory_results_bg {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    let graph_arc = state.get_user_graph(&user_id).ok();
                    let memory_arc = memory.clone();
                    let uid = user_id.clone();
                    let mid = crate::memory::MemoryId(uuid);

                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        let Some(graph_arc) = graph_arc else {
                            return;
                        };
                        let graph = graph_arc.read();
                        let memory_guard = memory_arc.read();

                        let episode = match graph.get_episode(&mid.0) {
                            Ok(Some(ep)) => ep,
                            _ => return,
                        };

                        if episode.entity_refs.is_empty() {
                            return;
                        }

                        let mut candidate_ids = std::collections::HashSet::new();
                        let cutoff = chrono::Utc::now() - chrono::Duration::days(7);

                        for entity_uuid in &episode.entity_refs {
                            if let Ok(episodes) = graph.get_episodes_by_entity(entity_uuid) {
                                for ep in &episodes {
                                    if ep.created_at >= cutoff {
                                        candidate_ids
                                            .insert(crate::memory::MemoryId(ep.uuid));
                                    }
                                }
                            }
                        }

                        candidate_ids.remove(&mid);
                        let candidate_ids: Vec<_> =
                            candidate_ids.into_iter().take(20).collect();
                        if candidate_ids.is_empty() {
                            return;
                        }

                        let candidates: Vec<_> = candidate_ids
                            .iter()
                            .filter_map(|id| memory_guard.get_memory(id).ok())
                            .collect();

                        let Ok(new_memory) = memory_guard.get_memory(&mid) else {
                            return;
                        };

                        match memory_guard.infer_lineage_for_memory(
                            &uid,
                            &new_memory,
                            &candidates,
                        ) {
                            Ok(edges) if !edges.is_empty() => {
                                tracing::info!(
                                    user_id = %uid,
                                    memory_id = %mid.0,
                                    edges = edges.len(),
                                    "Batch lineage: {} causal edges detected",
                                    edges.len()
                                );
                            }
                            Ok(_) => {}
                            Err(e) => {
                                tracing::debug!(
                                    "Lineage inference failed (non-fatal): {}",
                                    e
                                );
                            }
                        }
                    })
                    .await
                    {
                        tracing::debug!(
                            "Lineage inference panicked for {} (non-fatal): {e}",
                            id_str
                        );
                    }
                }
            }

            // Task 5: Conflict detection for each memory
            for (_, id_str, _) in &memory_results_bg {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    let state = state.clone();
                    let memory_arc = memory.clone();
                    let uid = user_id.clone();
                    let mid = crate::memory::MemoryId(uuid);

                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        detect_and_resolve_conflicts(&state, &memory_arc, &uid, &mid)
                    })
                    .await
                    .unwrap_or_else(|e| {
                        Err(anyhow::anyhow!("Conflict detection panicked: {e}"))
                    })
                    {
                        tracing::debug!(
                            "Conflict detection failed for {} (non-fatal): {}",
                            id_str,
                            e
                        );
                    }
                }
            }

            // Task 6: Create inter-batch association edges when create_edges is true.
            // Memories submitted in the same batch share temporal+topical context,
            // so we create co-occurrence edges between their episodic nodes.
            if create_edges && batch_memory_ids.len() > 1 {
                if let Ok(graph_arc) = state.get_user_graph(&user_id) {
                    let graph_guard = graph_arc.read();
                    let now = chrono::Utc::now();

                    // Collect entity_refs per batch memory from their episodes
                    let mut episode_entities: Vec<(uuid::Uuid, Vec<uuid::Uuid>)> = Vec::new();
                    for mid in &batch_memory_ids {
                        if let Ok(Some(ep)) = graph_guard.get_episode(&mid.0) {
                            episode_entities.push((mid.0, ep.entity_refs.clone()));
                        }
                    }

                    // Create edges between entity pairs across batch items
                    for i in 0..episode_entities.len() {
                        for j in (i + 1)..episode_entities.len() {
                            let (ep_i, refs_i) = &episode_entities[i];
                            let (_, refs_j) = &episode_entities[j];

                            for &ent_i in refs_i {
                                for &ent_j in refs_j {
                                    if ent_i == ent_j {
                                        continue; // Same entity, skip self-loop
                                    }
                                    let init_weight =
                                        crate::graph_memory::EdgeTier::L1Working
                                            .initial_weight();
                                    let edge = crate::graph_memory::RelationshipEdge {
                                        uuid: uuid::Uuid::new_v4(),
                                        from_entity: ent_i,
                                        to_entity: ent_j,
                                        relation_type:
                                            crate::graph_memory::RelationType::RelatedTo,
                                        strength: init_weight,
                                        created_at: now,
                                        valid_at: now,
                                        invalidated_at: None,
                                        source_episode_id: Some(*ep_i),
                                        context: "batch-association".to_string(),
                                        last_activated: now,
                                        activation_count: 1,
                                        ltp_status: Default::default(),
                                        tier: Default::default(),
                                        activation_timestamps: None,
                                        entity_confidence: None,
                                        created_by:
                                            crate::graph_memory::EdgeSource::CoOccurrence,
                                        forward_strength: init_weight,
                                        backward_strength: init_weight,
                                    };
                                    if let Err(e) = graph_guard.add_relationship(edge) {
                                        tracing::debug!(
                                            "Failed to add batch association edge: {}",
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }

                    tracing::debug!(
                        user_id = %user_id,
                        batch_size = batch_memory_ids.len(),
                        "Inter-batch association edges created"
                    );
                }
            }
        });
    }

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::BATCH_STORE_DURATION.observe(duration);
    metrics::BATCH_STORE_SIZE.observe(batch_size as f64);
    for _ in 0..created {
        metrics::MEMORY_STORE_TOTAL
            .with_label_values(&["success"])
            .inc();
    }
    for _ in 0..failed {
        metrics::MEMORY_STORE_TOTAL
            .with_label_values(&["error"])
            .inc();
    }

    Ok(Json(BatchRememberResponse {
        created,
        failed,
        memory_ids,
        errors: all_errors,
    }))
}

/// Upsert memory - create or update with external ID linking
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, external_id = %req.external_id))]
pub async fn upsert_memory(
    State(state): State<AppState>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    if req.external_id.is_empty() {
        return Err(AppError::InvalidInput {
            field: "external_id".to_string(),
            reason: "external_id is required for upsert".to_string(),
        });
    }

    let experience_type = parse_experience_type(req.memory_type.as_ref());

    let change_type = match req.change_type.to_lowercase().as_str() {
        "created" => ChangeType::Created,
        "content_updated" => ChangeType::ContentUpdated,
        "status_changed" => ChangeType::StatusChanged,
        "tags_updated" => ChangeType::TagsUpdated,
        "importance_adjusted" => ChangeType::ImportanceAdjusted,
        _ => ChangeType::ContentUpdated,
    };

    // Extract entities via NER (preserve full records for graph label propagation)
    let ner_entities: Vec<NerEntityRecord> = match state.get_neural_ner().extract(&req.content) {
        Ok(entities) => entities
            .into_iter()
            .map(|e| NerEntityRecord {
                text: e.text,
                entity_type: e.entity_type.as_str().to_string(),
                confidence: e.confidence,
                start_char: Some(e.start),
                end_char: Some(e.end),
            })
            .collect(),
        Err(e) => {
            tracing::debug!("NER extraction failed in upsert: {}", e);
            Vec::new()
        }
    };

    // Extract keywords via YAKE for common nouns, verbs, concepts
    let extracted_keywords: Vec<String> = state.get_keyword_extractor().extract_texts(&req.content);

    let mut merged_entities: Vec<String> = req.tags.clone();
    let mut seen: HashSet<String> = merged_entities.iter().map(|t| t.to_lowercase()).collect();
    for record in &ner_entities {
        if seen.insert(record.text.to_lowercase()) {
            merged_entities.push(record.text.clone());
        }
    }
    for keyword in extracted_keywords {
        if seen.insert(keyword.to_lowercase()) {
            merged_entities.push(keyword);
        }
    }
    if merged_entities.len() > validation::MAX_ENTITIES_PER_MEMORY {
        tracing::debug!(
            count = merged_entities.len(),
            max = validation::MAX_ENTITIES_PER_MEMORY,
            "Capping entities to maximum allowed in upsert"
        );
        merged_entities.truncate(validation::MAX_ENTITIES_PER_MEMORY);
    }

    let experience = Experience {
        content: req.content.clone(),
        experience_type,
        entities: merged_entities.clone(),
        tags: merged_entities,
        ner_entities,
        ..Default::default()
    };

    let memory_system = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let external_id = req.external_id.clone();
    let changed_by = req.changed_by.clone();
    let change_reason = req.change_reason.clone();

    let (memory_id, was_update) = {
        let memory = memory_system.clone();
        let exp = experience.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.upsert(external_id, exp, change_type, changed_by, change_reason)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    let version = {
        let memory = memory_system.clone();
        let mid = memory_id.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard
                .get_memory(&mid)
                .map(|m| m.version)
                .unwrap_or(1)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // Build episodic graph for multi-hop retrieval
    // On updates, clean up the old episode's edges/entities first to prevent
    // stale graph data from accumulating (entity_episodes index, orphan edges).
    if was_update {
        if let Ok(graph) = state.get_user_graph(&req.user_id) {
            let graph_guard = graph.read();
            match graph_guard.delete_episode(&memory_id.0) {
                Ok(true) => {
                    tracing::debug!(
                        "Cleaned up old episode {} before graph rebuild",
                        &memory_id.0.to_string()[..8]
                    );
                }
                Ok(false) => {} // No prior episode existed
                Err(e) => {
                    tracing::debug!(
                        "Old episode cleanup failed for {} (non-fatal): {}",
                        &memory_id.0.to_string()[..8],
                        e
                    );
                }
            }
        }
    }
    if let Err(e) = state.process_experience_into_graph(&req.user_id, &experience, &memory_id) {
        tracing::debug!("Graph processing failed (non-fatal): {}", e);
    }

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&[if was_update {
            "upsert_update"
        } else {
            "upsert_create"
        }])
        .inc();

    // Broadcast event
    state.emit_event(MemoryEvent {
        event_type: if was_update {
            "UPDATE".to_string()
        } else {
            "CREATE".to_string()
        },
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(req.content.chars().take(500).collect()),
        memory_type: req.memory_type.clone(),
        importance: None,
        count: None,
        entities: if req.tags.is_empty() {
            None
        } else {
            Some(req.tags.clone())
        },
        results: None,
    });

    Ok(Json(UpsertResponse {
        id: memory_id.0.to_string(),
        success: true,
        was_update,
        version,
    }))
}
