//! Ingest Handler - Multi-format text extraction and memory storage
//!
//! Convenience endpoint that auto-detects input format, extracts clean text,
//! and stores the result as a memory. Combines format detection + extraction +
//! remember in a single call.

use axum::{extract::State, response::Json};

use super::health::AppState;
use super::remember::parse_experience_type;
use super::types::MemoryEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::ingest;
use crate::memory::{
    types::NerEntityRecord,
    Experience, SessionEvent,
};
use crate::metrics;
use crate::validation;
use std::collections::HashSet;

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// Ingest request - auto-detect format, extract text, store as memory
#[derive(Debug, serde::Deserialize)]
pub struct IngestRequest {
    pub user_id: String,
    pub content: String,
    /// Optional filename for format detection (e.g. "notes.md", "data.csv")
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub importance: Option<f32>,
    #[serde(default, alias = "experience_type")]
    pub memory_type: Option<String>,
}

/// Ingest response - remember response plus extraction metadata
#[derive(Debug, serde::Serialize)]
pub struct IngestResponse {
    pub id: String,
    pub success: bool,
    pub format_detected: String,
    pub metadata: IngestMetadata,
}

/// Extraction metadata returned alongside the stored memory
#[derive(Debug, serde::Serialize)]
pub struct IngestMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub headings: Vec<String>,
    pub entities_hint: Vec<String>,
    pub line_count: usize,
    pub word_count: usize,
}

// =============================================================================
// HANDLER
// =============================================================================

/// Ingest content: auto-detect format, extract text, store as memory.
///
/// `POST /api/ingest`
///
/// This is a convenience endpoint that combines:
/// 1. Format detection (from filename extension or content sniffing)
/// 2. Text extraction (Markdown stripping, JSON flattening, CSV parsing, etc.)
/// 3. Memory storage (same as `/api/remember`)
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn ingest(
    State(state): State<AppState>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    // ── Format detection and extraction ─────────────────────────────────
    let format = ingest::detect_format(req.filename.as_deref(), &req.content);
    let format_label = format.as_str().to_string();

    let (processed_content, extraction_metadata) = if format != ingest::InputFormat::PlainText {
        match ingest::extract_text(&req.content, format) {
            Ok(extracted) => {
                let meta = IngestMetadata {
                    title: extracted.metadata.title.clone(),
                    headings: extracted.metadata.headings.clone(),
                    entities_hint: extracted.metadata.entities_hint.clone(),
                    line_count: extracted.metadata.line_count,
                    word_count: extracted.metadata.word_count,
                };
                (extracted.text, meta)
            }
            Err(e) => {
                tracing::debug!("Ingest extraction failed, using raw content: {}", e);
                let meta = IngestMetadata {
                    title: None,
                    headings: vec![],
                    entities_hint: vec![],
                    line_count: req.content.lines().count(),
                    word_count: req.content.split_whitespace().count(),
                };
                (req.content.clone(), meta)
            }
        }
    } else {
        let meta = IngestMetadata {
            title: None,
            headings: vec![],
            entities_hint: vec![],
            line_count: req.content.lines().count(),
            word_count: req.content.split_whitespace().count(),
        };
        (req.content.clone(), meta)
    };

    // ── NER + YAKE extraction in parallel ───────────────────────────────
    let experience_type = parse_experience_type(req.memory_type.as_ref());

    let ner = state.get_neural_ner();
    let yake = state.get_keyword_extractor();
    let content_for_ner = processed_content.clone();
    let content_for_yake = processed_content.clone();

    let (ner_result, yake_result) = tokio::join!(
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
                    tracing::debug!("NER extraction failed in ingest: {}", e);
                    Vec::new()
                }
            }
        }),
        tokio::task::spawn_blocking(move || yake.extract_texts(&content_for_yake))
    );

    let ner_entities = ner_result.unwrap_or_default();
    let extracted_keywords = yake_result.unwrap_or_default();

    // ── Merge all entity sources ────────────────────────────────────────
    let mut merged_entities: Vec<String> = req.tags.clone();
    let mut seen: HashSet<String> = merged_entities.iter().map(|t| t.to_lowercase()).collect();

    // Add format tag
    if format != ingest::InputFormat::PlainText {
        let format_tag = format!("format:{}", format_label);
        if seen.insert(format_tag.to_lowercase()) {
            merged_entities.push(format_tag);
        }
    }

    // Ingest-derived entities
    for entity in &extraction_metadata.entities_hint {
        if seen.insert(entity.to_lowercase()) {
            merged_entities.push(entity.clone());
        }
    }

    // NER entities
    for record in &ner_entities {
        if seen.insert(record.text.to_lowercase()) {
            merged_entities.push(record.text.clone());
        }
    }

    // YAKE keywords
    for keyword in extracted_keywords {
        if seen.insert(keyword.to_lowercase()) {
            merged_entities.push(keyword);
        }
    }

    if merged_entities.len() > validation::MAX_ENTITIES_PER_MEMORY {
        merged_entities.truncate(validation::MAX_ENTITIES_PER_MEMORY);
    }

    // ── Build and store experience ──────────────────────────────────────
    let experience_type_str = format!("{:?}", experience_type);

    let experience = Experience {
        content: processed_content,
        experience_type,
        entities: merged_entities.clone(),
        tags: merged_entities,
        ner_entities,
        ..Default::default()
    };

    let memory = state
        .get_user_earth(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_id = {
        let memory = memory.clone();
        let exp_clone = experience.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.remember(exp_clone, None)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // ── Metrics + events ────────────────────────────────────────────────
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
        content_preview: Some(req.content.chars().take(100).collect()),
        memory_type: Some(experience_type_str),
        importance: req.importance,
        count: None,
        entities: Some(req.tags.clone()),
        results: None,
    });

    // Background graph processing
    {
        let state = state.clone();
        let user_id = req.user_id.clone();
        let experience = experience.clone();
        let mid = memory_id.clone();
        tokio::spawn(async move {
            if let Err(e) = tokio::task::spawn_blocking(move || {
                state.process_experience_into_graph(&user_id, &experience, &mid)
            })
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("Graph task panicked: {e}")))
            {
                tracing::debug!("Ingest graph processing failed (non-fatal): {}", e);
            }
        });
    }

    Ok(Json(IngestResponse {
        id: memory_id.0.to_string(),
        success: true,
        format_detected: format_label,
        metadata: extraction_metadata,
    }))
}
