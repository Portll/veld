//! A/B Testing Handlers
//!
//! Handlers for A/B test management, metrics recording, and analysis.

use axum::{
    extract::{Path, State},
    response::Json,
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::ab_testing;
use crate::errors::{AppError, ValidationErrorExt};
use crate::relevance;
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

fn default_traffic_split() -> f32 {
    0.5
}

fn default_min_impressions() -> u64 {
    100
}

/// Request to create a new A/B test
#[derive(Debug, Deserialize)]
pub struct CreateABTestRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub control_weights: Option<relevance::LearnedSignalWeights>,
    #[serde(default)]
    pub treatment_weights: Option<relevance::LearnedSignalWeights>,
    #[serde(default = "default_traffic_split")]
    pub traffic_split: f32,
    #[serde(default = "default_min_impressions")]
    pub min_impressions: u64,
    #[serde(default)]
    pub max_duration_hours: Option<u64>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Response for A/B test operations
#[derive(Debug, Serialize)]
pub struct ABTestResponse {
    pub success: bool,
    pub test_id: Option<String>,
    pub message: String,
}

/// Request to record an impression
#[derive(Debug, Deserialize)]
pub struct RecordImpressionRequest {
    pub user_id: String,
    #[serde(default)]
    pub relevance_score: Option<f64>,
    #[serde(default)]
    pub latency_us: Option<u64>,
}

/// Request to record a click
#[derive(Debug, Deserialize)]
pub struct RecordClickRequest {
    pub user_id: String,
    pub memory_id: uuid::Uuid,
}

/// Request to record feedback
#[derive(Debug, Deserialize)]
pub struct RecordFeedbackRequest {
    pub user_id: String,
    pub positive: bool,
}

/// Reply of `GET /api/ab/tests`.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestListResponse {
    pub success: bool,
    pub tests: Vec<AbTestListItem>,
    pub summary: ab_testing::ABTestManagerSummary,
}

/// One test of a `GET /api/ab/tests` reply.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestListItem {
    pub id: String,
    pub name: String,
    pub description: String,
    /// `Draft`, `Running`, `Paused`, `Completed` or `Archived`.
    pub status: String,
    pub traffic_split: f32,
    pub control_impressions: u64,
    pub treatment_impressions: u64,
    /// RFC 3339.
    pub created_at: String,
}

/// GET /api/ab/tests - List all A/B tests
pub async fn list_ab_tests(
    State(state): State<AppState>,
) -> Result<Json<AbTestListResponse>, AppError> {
    let tests = state.ab_test_manager.list_tests();
    let summary = state.ab_test_manager.summary();

    Ok(Json(AbTestListResponse {
        success: true,
        tests: tests
            .iter()
            .map(|t| AbTestListItem {
                id: t.id.clone(),
                name: t.config.name.clone(),
                description: t.config.description.clone(),
                status: format!("{:?}", t.status),
                traffic_split: t.config.traffic_split,
                control_impressions: t.control_metrics.impressions,
                treatment_impressions: t.treatment_metrics.impressions,
                created_at: t.created_at.to_rfc3339(),
            })
            .collect(),
        summary,
    }))
}

/// POST /api/ab/tests - Create a new A/B test
pub async fn create_ab_test(
    State(state): State<AppState>,
    Json(req): Json<CreateABTestRequest>,
) -> Result<Json<ABTestResponse>, AppError> {
    let mut builder = ab_testing::ABTest::builder(&req.name)
        .with_traffic_split(req.traffic_split)
        .with_min_impressions(req.min_impressions);

    if let Some(desc) = req.description {
        builder = builder.with_description(&desc);
    }

    if let Some(control) = req.control_weights {
        builder = builder.with_control(control);
    }

    if let Some(treatment) = req.treatment_weights {
        builder = builder.with_treatment(treatment);
    }

    if let Some(hours) = req.max_duration_hours {
        builder = builder.with_max_duration_hours(hours);
    }

    if !req.tags.is_empty() {
        builder = builder.with_tags(req.tags);
    }

    let test = builder.build();

    match state.ab_test_manager.create_test(test) {
        Ok(test_id) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "A/B test created successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to create test: {}", e),
        })),
    }
}

/// Reply of `GET /api/ab/tests/{test_id}`: `test` when it exists, else `message`.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestDetailResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test: Option<AbTestDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// A test as `GET /api/ab/tests/{test_id}` reports it.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestDetail {
    pub id: String,
    pub name: String,
    pub description: String,
    /// `Draft`, `Running`, `Paused`, `Completed` or `Archived`.
    pub status: String,
    pub traffic_split: f32,
    pub min_impressions: u64,
    pub max_duration_hours: Option<u64>,
    pub control_weights: relevance::LearnedSignalWeights,
    pub treatment_weights: relevance::LearnedSignalWeights,
    pub control_metrics: AbVariantMetrics,
    pub treatment_metrics: AbVariantMetrics,
    /// RFC 3339.
    pub created_at: String,
    /// RFC 3339.
    pub started_at: Option<String>,
    /// RFC 3339.
    pub completed_at: Option<String>,
}

/// One arm's counters, with its click-through rate.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbVariantMetrics {
    pub impressions: u64,
    pub clicks: u64,
    /// `clicks / impressions`, 0 when there were no impressions.
    pub ctr: f64,
    pub positive_feedback: u64,
    pub negative_feedback: u64,
}

impl AbVariantMetrics {
    fn of(metrics: &ab_testing::VariantMetrics) -> Self {
        let ctr = if metrics.impressions > 0 {
            metrics.clicks as f64 / metrics.impressions as f64
        } else {
            0.0
        };
        Self {
            impressions: metrics.impressions,
            clicks: metrics.clicks,
            ctr,
            positive_feedback: metrics.positive_feedback,
            negative_feedback: metrics.negative_feedback,
        }
    }
}

/// GET /api/ab/tests/{test_id} - Get a specific A/B test
pub async fn get_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<AbTestDetailResponse>, AppError> {
    match state.ab_test_manager.get_test(&test_id) {
        Some(test) => Ok(Json(AbTestDetailResponse {
            success: true,
            test: Some(AbTestDetail {
                id: test.id.clone(),
                name: test.config.name.clone(),
                description: test.config.description.clone(),
                status: format!("{:?}", test.status),
                traffic_split: test.config.traffic_split,
                min_impressions: test.config.min_impressions,
                max_duration_hours: test.config.max_duration_hours,
                control_weights: test.config.control_weights.clone(),
                treatment_weights: test.config.treatment_weights.clone(),
                control_metrics: AbVariantMetrics::of(&test.control_metrics),
                treatment_metrics: AbVariantMetrics::of(&test.treatment_metrics),
                created_at: test.created_at.to_rfc3339(),
                started_at: test.started_at.map(|t| t.to_rfc3339()),
                completed_at: test.completed_at.map(|t| t.to_rfc3339()),
            }),
            message: None,
        })),
        None => Ok(Json(AbTestDetailResponse {
            success: false,
            test: None,
            message: Some(format!("Test not found: {}", test_id)),
        })),
    }
}

/// DELETE /api/ab/tests/{test_id} - Delete an A/B test
pub async fn delete_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.delete_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test deleted successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to delete test: {}", e),
        })),
    }
}

/// POST /api/ab/tests/{test_id}/start - Start an A/B test
pub async fn start_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.start_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test started successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to start test: {}", e),
        })),
    }
}

/// POST /api/ab/tests/{test_id}/pause - Pause an A/B test
pub async fn pause_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.pause_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test paused successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to pause test: {}", e),
        })),
    }
}

/// POST /api/ab/tests/{test_id}/resume - Resume a paused A/B test
pub async fn resume_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.resume_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test resumed successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to resume test: {}", e),
        })),
    }
}

/// Reply of `POST /api/ab/tests/{test_id}/complete`: `test_id` and `results` on success,
/// else `message`.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestCompleteResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<AbTestAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// The statistics of one test, as `analyze` and `complete` report them.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestAnalysis {
    pub is_significant: bool,
    pub confidence_level: f64,
    pub chi_squared: f64,
    pub p_value: f64,
    /// `Control` or `Treatment`; absent while neither arm is ahead significantly.
    pub winner: Option<String>,
    pub relative_improvement: f64,
    pub control_ctr: f64,
    pub treatment_ctr: f64,
    pub confidence_interval: (f64, f64),
    pub recommendations: Vec<String>,
}

impl AbTestAnalysis {
    fn of(results: ab_testing::ABTestResults) -> Self {
        Self {
            is_significant: results.is_significant,
            confidence_level: results.confidence_level,
            chi_squared: results.chi_squared,
            p_value: results.p_value,
            winner: results.winner.map(|w| format!("{:?}", w)),
            relative_improvement: results.relative_improvement,
            control_ctr: results.control_ctr,
            treatment_ctr: results.treatment_ctr,
            confidence_interval: results.confidence_interval,
            recommendations: results.recommendations,
        }
    }
}

/// POST /api/ab/tests/{test_id}/complete - Complete an A/B test and get results
pub async fn complete_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<AbTestCompleteResponse>, AppError> {
    match state.ab_test_manager.complete_test(&test_id) {
        Ok(results) => Ok(Json(AbTestCompleteResponse {
            success: true,
            test_id: Some(test_id),
            results: Some(AbTestAnalysis::of(results)),
            message: None,
        })),
        Err(e) => Ok(Json(AbTestCompleteResponse {
            success: false,
            test_id: None,
            results: None,
            message: Some(format!("Failed to complete test: {}", e)),
        })),
    }
}

/// Reply of `GET /api/ab/tests/{test_id}/analyze`: `test_id` and `analysis` on success,
/// else `message`.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbTestAnalysisResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis: Option<AbTestAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// GET /api/ab/tests/{test_id}/analyze - Analyze an A/B test without completing it
pub async fn analyze_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<AbTestAnalysisResponse>, AppError> {
    match state.ab_test_manager.analyze_test(&test_id) {
        Ok(results) => Ok(Json(AbTestAnalysisResponse {
            success: true,
            test_id: Some(test_id),
            analysis: Some(AbTestAnalysis::of(results)),
            message: None,
        })),
        Err(e) => Ok(Json(AbTestAnalysisResponse {
            success: false,
            test_id: None,
            analysis: None,
            message: Some(format!("Failed to analyze test: {}", e)),
        })),
    }
}

/// Reply of `POST /api/ab/tests/{test_id}/impression`: `variant` on success (`null` when
/// the user has no assignment), else `message`.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbImpressionResponse {
    pub success: bool,
    /// `Control` or `Treatment`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// POST /api/ab/tests/{test_id}/impression - Record an impression for an A/B test
pub async fn record_ab_impression(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordImpressionRequest>,
) -> Result<Json<AbImpressionResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let relevance_score = req.relevance_score.unwrap_or(0.0);
    let latency_us = req.latency_us.unwrap_or(0);

    match state.ab_test_manager.record_impression(
        &test_id,
        &req.user_id,
        relevance_score,
        latency_us,
    ) {
        Ok(()) => {
            let variant = state
                .ab_test_manager
                .get_variant(&test_id, &req.user_id)
                .ok();
            Ok(Json(AbImpressionResponse {
                success: true,
                variant: Some(variant.map(|v| format!("{:?}", v))),
                message: None,
            }))
        }
        Err(e) => Ok(Json(AbImpressionResponse {
            success: false,
            variant: None,
            message: Some(format!("Failed to record impression: {}", e)),
        })),
    }
}

/// Reply of `POST /api/ab/tests/{test_id}/click` and `.../feedback`: `message` only on
/// failure.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbRecordResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// POST /api/ab/tests/{test_id}/click - Record a click for an A/B test
pub async fn record_ab_click(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordClickRequest>,
) -> Result<Json<AbRecordResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .ab_test_manager
        .record_click(&test_id, &req.user_id, req.memory_id)
    {
        Ok(()) => Ok(Json(AbRecordResponse {
            success: true,
            message: None,
        })),
        Err(e) => Ok(Json(AbRecordResponse {
            success: false,
            message: Some(format!("Failed to record click: {}", e)),
        })),
    }
}

/// POST /api/ab/tests/{test_id}/feedback - Record feedback for an A/B test
pub async fn record_ab_feedback(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordFeedbackRequest>,
) -> Result<Json<AbRecordResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .ab_test_manager
        .record_feedback(&test_id, &req.user_id, req.positive)
    {
        Ok(()) => Ok(Json(AbRecordResponse {
            success: true,
            message: None,
        })),
        Err(e) => Ok(Json(AbRecordResponse {
            success: false,
            message: Some(format!("Failed to record feedback: {}", e)),
        })),
    }
}

/// Reply of `GET /api/ab/summary`.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbSummaryResponse {
    pub success: bool,
    pub summary: ab_testing::ABTestManagerSummary,
    /// Tests this call archived for running past `max_duration_hours`.
    pub expired_tests: Vec<String>,
}

/// GET /api/ab/summary - Get summary of all A/B tests
pub async fn get_ab_summary(
    State(state): State<AppState>,
) -> Result<Json<AbSummaryResponse>, AppError> {
    let summary = state.ab_test_manager.summary();
    let expired = state.ab_test_manager.check_expired_tests();

    Ok(Json(AbSummaryResponse {
        success: true,
        summary,
        expired_tests: expired,
    }))
}
