//! Structural Graph Analysis Handlers
//!
//! REST API endpoints exposing structural analysis primitives on the knowledge graph:
//! gap detection, Voronoi cell analysis, persistent homology, and Mapper topology.
//! These return raw structural data — interpretation is left to consumers.

use axum::{extract::State, http::StatusCode, response::Json};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::sync::watch;

use super::state::MultiUserMemoryManager;
use super::types::MemoryEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::gap_topology::{GapDetectionConfig, GapDetectionResult, GapDetector};
use crate::memory::mapper::{compute_mapper, MapperConfig, MapperFilter};
use crate::memory::persistence::{self, PersistenceConfig};
use crate::memory::slow_store::SlowStore;
use crate::memory::voronoi::{VoronoiAnalyzer, VoronoiConfig};
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// PARAMETER BOUNDS (prevent CPU DoS from adversarial inputs)
// =============================================================================

const MIN_STEP_SIZE: f32 = 0.01;
const MAX_STEP_SIZE: f32 = 1.0;
const MAX_GAPS_PER_TYPE: usize = 500;
const MIN_PERSISTENCE: f32 = 0.001;
const MAX_K: usize = 100;
const MAX_NUM_INTERVALS: usize = 200;
const MIN_OVERLAP: f32 = 0.0;
const MAX_OVERLAP: f32 = 0.9;
const MAX_ENTITIES: usize = 2000;
/// How long a gap detection result is served as fresh.
const GAP_FRESH_SECS: i64 = 300;
/// Detection results kept at most, the oldest dropped first.
const GAP_RESULTS_KEPT: usize = 64;

fn validate_range_f32(value: f32, min: f32, max: f32, name: &str) -> Result<f32, AppError> {
    if !value.is_finite() {
        return Err(AppError::InvalidInput {
            field: name.to_string(),
            reason: format!("{name} must be a finite number"),
        });
    }
    Ok(value.clamp(min, max))
}

fn validate_range_usize(value: usize, max: usize, _name: &str) -> Result<usize, AppError> {
    Ok(value.min(max))
}

// =============================================================================
// GAP DETECTION
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct GapAnalysisRequest {
    pub user_id: String,
    #[serde(default = "default_min_strength")]
    pub min_edge_strength: f32,
    #[serde(default = "default_max_gaps")]
    pub max_gaps_per_type: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Minimum embedding similarity to flag a U-shape (default 0.3)
    #[serde(default)]
    pub min_embedding_similarity: Option<f32>,
    /// Minimum spokes for star gap detection (default 3)
    #[serde(default)]
    pub star_min_spokes: Option<usize>,
    /// Minimum cluster size for orbit detection (default 3)
    #[serde(default)]
    pub orbit_min_cluster_size: Option<usize>,
}

fn default_limit() -> usize {
    50
}

fn default_min_strength() -> f32 {
    0.2
}
fn default_max_gaps() -> usize {
    50
}

#[derive(Debug, Serialize)]
pub struct GapAnalysisResponse {
    pub gaps: Vec<GapSummary>,
    pub total_count: usize,
    pub type_counts: std::collections::HashMap<String, usize>,
    pub duration_ms: u64,
    /// `fresh`: computed within the last five minutes. `stale`: an older result,
    /// served while a new run goes on (or after it failed). `running`: no result
    /// yet, the run goes on (status 202).
    pub status: &'static str,
    /// When the result was computed, RFC 3339; absent while `running`.
    pub computed_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GapSummary {
    pub id: String,
    pub gap_type: String,
    pub confidence: f32,
    pub embedding_similarity: Option<f32>,
    pub impact_score: f32,
    pub entity_names: Vec<String>,
    pub shape: ShapeSummary,
}

#[derive(Debug, Serialize)]
pub struct ShapeSummary {
    pub node_count: usize,
    pub existing_edges: usize,
    pub missing_edges: usize,
    pub sparsity: f32,
}

/// What a detection depends on; offset and limit only page its result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GapKey {
    user_id: String,
    min_strength: u32,
    max_gaps: usize,
    min_embedding_similarity: Option<u32>,
    star_min_spokes: Option<usize>,
    orbit_min_cluster_size: Option<usize>,
}

#[derive(Clone)]
struct Detected {
    result: Arc<GapDetectionResult>,
    at: chrono::DateTime<chrono::Utc>,
}

impl Detected {
    fn is_fresh(&self) -> bool {
        chrono::Utc::now() - self.at < chrono::Duration::seconds(GAP_FRESH_SECS)
    }
}

/// A run's end: its result, or why it failed.
type Outcome = Option<Result<Detected, String>>;

#[derive(Default)]
struct GapEntry {
    last: Option<Detected>,
    running: Option<watch::Receiver<Outcome>>,
}

/// Detection results by user and parameters, and the run under way for each.
static GAP_RUNS: LazyLock<Mutex<HashMap<GapKey, GapEntry>>> = LazyLock::new(Default::default);

enum Served {
    /// Computed within the freshness window.
    Fresh(Detected),
    /// An older result, while a new run goes on or after it failed.
    Stale(Detected),
    /// No result yet; the run goes on.
    Running,
}

/// The detection for `key`: a fresh result at once; otherwise the end of a run,
/// started here unless one is under way, if it comes within `wait`; otherwise
/// the last result while the run goes on. One run per key at a time, and a run
/// finishes and is kept whether or not anyone is still waiting for it.
async fn detection(
    key: GapKey,
    wait: Duration,
    job: impl FnOnce() -> anyhow::Result<GapDetectionResult> + Send + 'static,
) -> Result<Served, AppError> {
    let (mut run, last) = {
        let mut runs = GAP_RUNS.lock();
        let entry = runs.entry(key.clone()).or_default();
        if let Some(last) = entry.last.as_ref().filter(|d| d.is_fresh()) {
            return Ok(Served::Fresh(last.clone()));
        }
        // A run whose task died without an outcome has dropped its sender; start another.
        let run = match entry.running.as_ref().filter(|r| r.has_changed().is_ok()) {
            Some(run) => run.clone(),
            None => {
                let run = start(key, job);
                entry.running = Some(run.clone());
                run
            }
        };
        (run, entry.last.clone())
    };
    let ended = match tokio::time::timeout(wait, run.wait_for(Option::is_some)).await {
        Ok(Ok(outcome)) => Option::clone(&outcome),
        Ok(Err(_)) => Some(Err("the gap detection run ended without a result".into())),
        Err(_) => None,
    };
    Ok(match ended {
        Some(Ok(detected)) => Served::Fresh(detected),
        Some(Err(error)) => match last {
            Some(last) => Served::Stale(last),
            None => return Err(AppError::Internal(anyhow::anyhow!(error))),
        },
        None => last.map_or(Served::Running, Served::Stale),
    })
}

/// Runs `job` off the async runtime and keeps its result under `key`.
fn start(
    key: GapKey,
    job: impl FnOnce() -> anyhow::Result<GapDetectionResult> + Send + 'static,
) -> watch::Receiver<Outcome> {
    let (done, run) = watch::channel(None);
    tokio::spawn(async move {
        let outcome = match tokio::task::spawn_blocking(job).await {
            Ok(Ok(result)) => Ok(Detected {
                result: Arc::new(result),
                at: chrono::Utc::now(),
            }),
            Ok(Err(e)) => Err(format!("{e:#}")),
            Err(e) => Err(format!("gap detection task failed: {e}")),
        };
        if let Err(e) = &outcome {
            tracing::warn!(user_id = %key.user_id, error = %e, "gap detection failed");
        }
        {
            let mut runs = GAP_RUNS.lock();
            let entry = runs.entry(key).or_default();
            entry.running = None;
            if let Ok(detected) = &outcome {
                entry.last = Some(detected.clone());
            }
            prune(&mut runs);
        }
        done.send_replace(Some(outcome));
    });
    run
}

/// Drops results more than an hour old, then the oldest beyond the cap; a key
/// with a run under way is kept.
fn prune(runs: &mut HashMap<GapKey, GapEntry>) {
    let now = chrono::Utc::now();
    runs.retain(|_, e| {
        e.running.is_some()
            || e.last
                .as_ref()
                .is_some_and(|d| now - d.at < chrono::Duration::hours(1))
    });
    let excess = runs.len().saturating_sub(GAP_RESULTS_KEPT);
    if excess > 0 {
        let mut idle: Vec<(GapKey, chrono::DateTime<chrono::Utc>)> = runs
            .iter()
            .filter(|(_, e)| e.running.is_none())
            .filter_map(|(k, e)| e.last.as_ref().map(|d| (k.clone(), d.at)))
            .collect();
        idle.sort_by_key(|(_, at)| *at);
        for (k, _) in idle.into_iter().take(excess) {
            runs.remove(&k);
        }
    }
}

/// One page of a detection result.
fn summarize(
    detected: Option<&Detected>,
    status: &'static str,
    offset: usize,
    limit: usize,
) -> GapAnalysisResponse {
    let Some(detected) = detected else {
        return GapAnalysisResponse {
            gaps: Vec::new(),
            total_count: 0,
            type_counts: HashMap::new(),
            duration_ms: 0,
            status,
            computed_at: None,
        };
    };
    let result = &detected.result;
    GapAnalysisResponse {
        gaps: result
            .gaps
            .iter()
            .skip(offset)
            .take(limit)
            .map(|g| GapSummary {
                id: g.id.clone(),
                gap_type: g.gap_type.as_str().to_string(),
                confidence: g.confidence,
                embedding_similarity: g.embedding_similarity,
                impact_score: g.impact_score,
                entity_names: g.entities.iter().map(|e| e.name.clone()).collect(),
                shape: ShapeSummary {
                    node_count: g.shape.node_count,
                    existing_edges: g.shape.existing_edges,
                    missing_edges: g.shape.missing_edges,
                    sparsity: g.shape.sparsity,
                },
            })
            .collect(),
        total_count: result.gaps.len(),
        type_counts: result.type_counts.clone(),
        duration_ms: result.duration_ms,
        status,
        computed_at: Some(detected.at.to_rfc3339()),
    }
}

/// Detect structural gaps in the knowledge graph.
///
/// POST /api/gap/analyze
///
/// Detection runs in the background, one run per user and parameters, and its
/// result is kept. A request gets a result from the last five minutes at once;
/// otherwise it waits for a run until shortly before the route's timeout, then
/// gets the last result marked `stale`, or 202 with `running` when there is none.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
#[allow(clippy::manual_clamp)]
pub async fn analyze_gaps(
    State(state): State<AppState>,
    Json(req): Json<GapAnalysisRequest>,
) -> Result<(StatusCode, Json<GapAnalysisResponse>), AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let min_strength = validate_range_f32(req.min_edge_strength, 0.0, 1.0, "min_edge_strength")?;
    let max_gaps = validate_range_usize(
        req.max_gaps_per_type,
        MAX_GAPS_PER_TYPE,
        "max_gaps_per_type",
    )?;
    let offset = req.offset;
    let limit = validate_range_usize(req.limit.max(1), MAX_GAPS_PER_TYPE, "limit")?;

    let key = GapKey {
        user_id: req.user_id.clone(),
        min_strength: min_strength.to_bits(),
        max_gaps,
        min_embedding_similarity: req.min_embedding_similarity.map(f32::to_bits),
        star_min_spokes: req.star_min_spokes,
        orbit_min_cluster_size: req.orbit_min_cluster_size,
    };
    // Answer a few seconds inside the route's timeout rather than let it cut the request off.
    let wait = Duration::from_secs(
        state
            .server_config()
            .gap_request_timeout_secs
            .saturating_sub(5)
            .max(1),
    );
    let job_state = state.clone();
    let user_id = req.user_id.clone();
    let (min_embedding_similarity, star_min_spokes, orbit_min_cluster_size) = (
        req.min_embedding_similarity,
        req.star_min_spokes,
        req.orbit_min_cluster_size,
    );
    let served = detection(key, wait, move || {
        let store = get_or_create_slow_store(&job_state, &user_id)?;
        sync_graph_to_slow_store(&job_state, &user_id, &store)?;

        let mut config = GapDetectionConfig {
            min_edge_strength: min_strength,
            max_gaps_per_type: max_gaps,
            ..Default::default()
        };
        if let Some(v) = min_embedding_similarity {
            config.min_embedding_similarity = v.clamp(0.0, 1.0);
        }
        if let Some(v) = star_min_spokes {
            config.star_min_spokes = v.max(2).min(20);
        }
        if let Some(v) = orbit_min_cluster_size {
            config.orbit_min_cluster_size = v.max(2).min(50);
        }

        let result = GapDetector::detect(store.as_ref(), &config)?;
        job_state.emit_event(MemoryEvent {
            event_type: "gap_analysis_complete".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: user_id.clone(),
            memory_id: None,
            content_preview: None,
            memory_type: None,
            importance: None,
            count: Some(result.gaps.len()),
            entities: None,
            results: Some(serde_json::json!({
                "type": "gap_analysis_complete",
                "gap_count": result.gaps.len(),
                "duration_ms": result.duration_ms,
            })),
        });
        Ok(result)
    })
    .await?;

    Ok(match served {
        Served::Fresh(d) => (
            StatusCode::OK,
            Json(summarize(Some(&d), "fresh", offset, limit)),
        ),
        Served::Stale(d) => (
            StatusCode::OK,
            Json(summarize(Some(&d), "stale", offset, limit)),
        ),
        Served::Running => (
            StatusCode::ACCEPTED,
            Json(summarize(None, "running", offset, limit)),
        ),
    })
}

// =============================================================================
// VORONOI ANALYSIS
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct VoronoiAnalysisRequest {
    pub user_id: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_voronoi_limit")]
    pub limit: usize,
}

fn default_voronoi_limit() -> usize {
    10
}

fn default_k() -> usize {
    8
}

#[derive(Debug, Serialize)]
pub struct VoronoiAnalysisResponse {
    pub entity_count: usize,
    pub voids_found: usize,
    pub avg_anisotropy: f32,
    pub total_count: usize,
    pub most_isolated: Vec<IsolatedEntity>,
    pub voids: Vec<VoidSummary>,
}

#[derive(Debug, Serialize)]
pub struct IsolatedEntity {
    pub name: String,
    pub isolation: f32,
    pub anisotropy: f32,
}

#[derive(Debug, Serialize)]
pub struct VoidSummary {
    pub boundary_entities: Vec<String>,
    pub radius: f32,
    pub confidence: f32,
}

/// Run Voronoi cell analysis on the knowledge graph embedding space.
///
/// POST /api/gap/voronoi
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn voronoi_analysis(
    State(state): State<AppState>,
    Json(req): Json<VoronoiAnalysisRequest>,
) -> Result<Json<VoronoiAnalysisResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let k = validate_range_usize(req.k.max(1), MAX_K, "k")?;

    let user_id = req.user_id.clone();
    let state_clone = state.clone();
    let offset = req.offset;
    let limit = req.limit.max(1);

    let result =
        tokio::task::spawn_blocking(move || -> Result<VoronoiAnalysisResponse, anyhow::Error> {
            let store = get_or_create_slow_store(&state_clone, &user_id)?;
            sync_graph_to_slow_store(&state_clone, &user_id, &store)?;

            let config = VoronoiConfig {
                k,
                ..Default::default()
            };
            let analysis = VoronoiAnalyzer::analyze(&store, &config)?;

            let mut most_isolated: Vec<IsolatedEntity> = analysis
                .cells
                .iter()
                .map(|c| IsolatedEntity {
                    name: c.name.clone(),
                    isolation: c.isolation,
                    anisotropy: c.anisotropy,
                })
                .collect();
            most_isolated.sort_by(|a, b| {
                b.isolation
                    .partial_cmp(&a.isolation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let total_count = most_isolated.len();
            most_isolated = most_isolated.into_iter().skip(offset).take(limit).collect();

            let voids: Vec<VoidSummary> = analysis
                .voids
                .iter()
                .map(|v| VoidSummary {
                    boundary_entities: v.boundary_entities.iter().map(|(_, n)| n.clone()).collect(),
                    radius: v.radius,
                    confidence: v.confidence,
                })
                .collect();

            Ok(VoronoiAnalysisResponse {
                entity_count: analysis.stats.entity_count,
                voids_found: analysis.voids.len(),
                avg_anisotropy: analysis.stats.avg_anisotropy,
                total_count,
                most_isolated,
                voids,
            })
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Voronoi analysis task failed: {e}")))?
        .map_err(AppError::Internal)?;

    Ok(Json(result))
}

// =============================================================================
// PERSISTENCE (PERSISTENT HOMOLOGY)
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct PersistenceRequest {
    pub user_id: String,
    #[serde(default = "default_persistence_step")]
    pub step_size: f32,
    #[serde(default = "default_min_persistence")]
    pub min_persistence: f32,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_persistence_step() -> f32 {
    0.1
}
fn default_min_persistence() -> f32 {
    0.15
}

#[derive(Debug, Serialize)]
pub struct PersistenceResponse {
    pub total_count: usize,
    pub pairs: Vec<PersistencePairSummary>,
    pub betti_curves: Vec<BettiCurveSummary>,
    pub stats: PersistenceStatsSummary,
}

#[derive(Debug, Serialize)]
pub struct PersistencePairSummary {
    pub dimension: usize,
    pub dimension_name: String,
    pub birth: f32,
    pub death: Option<f32>,
    pub persistence: f32,
    pub birth_entities: Vec<String>,
    pub sandwich_lower: f32,
    pub sandwich_upper: f32,
}

#[derive(Debug, Serialize)]
pub struct BettiCurveSummary {
    pub epsilon: f32,
    pub beta_0: usize,
    pub beta_1: usize,
    pub beta_2: usize,
    pub simplex_count: usize,
}

#[derive(Debug, Serialize)]
pub struct PersistenceStatsSummary {
    pub entity_count: usize,
    pub filtration_levels: usize,
    pub total_pairs: usize,
    pub persistent_features: usize,
    pub noise_features: usize,
    pub duration_ms: u64,
}

/// Compute persistent homology via Rips filtration.
///
/// POST /api/gap/persistence
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn persistence_analysis(
    State(state): State<AppState>,
    Json(req): Json<PersistenceRequest>,
) -> Result<Json<PersistenceResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let step_size = validate_range_f32(req.step_size, MIN_STEP_SIZE, MAX_STEP_SIZE, "step_size")?;
    let min_persistence = validate_range_f32(
        req.min_persistence,
        MIN_PERSISTENCE,
        MAX_STEP_SIZE,
        "min_persistence",
    )?;

    let user_id = req.user_id.clone();
    let state_clone = state.clone();
    let offset = req.offset;
    let limit = req.limit.max(1);

    let result =
        tokio::task::spawn_blocking(move || -> Result<PersistenceResponse, anyhow::Error> {
            let store = get_or_create_slow_store(&state_clone, &user_id)?;
            sync_graph_to_slow_store(&state_clone, &user_id, &store)?;

            let config = PersistenceConfig {
                step_size,
                min_persistence,
                max_entities: MAX_ENTITIES,
                ..Default::default()
            };

            let diagram = persistence::compute_persistence(&store, &config)?;

            let dim_name = |d: usize| match d {
                0 => "component",
                1 => "loop",
                2 => "void",
                _ => "higher",
            };

            let total_count = diagram.pairs.len();
            let pairs: Vec<PersistencePairSummary> = diagram
                .pairs
                .iter()
                .map(|p| {
                    let entity_names: Vec<String> = p
                        .birth_simplex
                        .iter()
                        .filter_map(|&idx| diagram.vertex_names.get(idx).cloned())
                        .collect();

                    PersistencePairSummary {
                        dimension: p.dimension,
                        dimension_name: dim_name(p.dimension).to_string(),
                        birth: p.birth,
                        death: p.death,
                        persistence: if p.persistence.is_infinite() {
                            -1.0
                        } else {
                            p.persistence
                        },
                        birth_entities: entity_names,
                        sandwich_lower: p.birth,
                        sandwich_upper: (p.birth * 2.0).min(1.0),
                    }
                })
                .skip(offset)
                .take(limit)
                .collect();

            let betti_curves: Vec<BettiCurveSummary> = diagram
                .betti_curves
                .iter()
                .map(|b| BettiCurveSummary {
                    epsilon: b.epsilon,
                    beta_0: b.beta_0,
                    beta_1: b.beta_1,
                    beta_2: b.beta_2,
                    simplex_count: b.simplex_count,
                })
                .collect();

            Ok(PersistenceResponse {
                total_count,
                pairs,
                betti_curves,
                stats: PersistenceStatsSummary {
                    entity_count: diagram.stats.entity_count,
                    filtration_levels: diagram.stats.filtration_levels,
                    total_pairs: diagram.stats.total_pairs,
                    persistent_features: diagram.stats.persistent_features,
                    noise_features: diagram.stats.noise_features,
                    duration_ms: diagram.stats.duration_ms,
                },
            })
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Persistence analysis failed: {e}")))?
        .map_err(AppError::Internal)?;

    Ok(Json(result))
}

// =============================================================================
// MAPPER
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct MapperRequest {
    pub user_id: String,
    #[serde(default = "default_mapper_filter")]
    pub filter: String,
    #[serde(default = "default_num_intervals")]
    pub num_intervals: usize,
    #[serde(default = "default_overlap")]
    pub overlap: f32,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_mapper_filter() -> String {
    "centroid_distance".to_string()
}
fn default_num_intervals() -> usize {
    10
}
fn default_overlap() -> f32 {
    0.3
}

#[derive(Debug, Serialize)]
pub struct MapperResponse {
    pub total_count: usize,
    pub nodes: Vec<MapperNodeSummary>,
    pub edges: Vec<MapperEdgeSummary>,
    pub num_components: usize,
    pub num_loops: usize,
    pub flare_count: usize,
    pub branch_count: usize,
    pub filter: String,
    pub stats: MapperStatsSummary,
}

#[derive(Debug, Serialize)]
pub struct MapperNodeSummary {
    pub id: usize,
    pub member_names: Vec<String>,
    pub size: usize,
    pub avg_filter_value: f32,
}

#[derive(Debug, Serialize)]
pub struct MapperEdgeSummary {
    pub from: usize,
    pub to: usize,
    pub weight: usize,
}

#[derive(Debug, Serialize)]
pub struct MapperStatsSummary {
    pub entity_count: usize,
    pub interval_count: usize,
    pub cluster_count: usize,
    pub edge_count: usize,
    pub duration_ms: u64,
}

/// Run Mapper topological analysis on the knowledge graph.
///
/// POST /api/gap/mapper
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn mapper_analysis(
    State(state): State<AppState>,
    Json(req): Json<MapperRequest>,
) -> Result<Json<MapperResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let num_intervals =
        validate_range_usize(req.num_intervals.max(1), MAX_NUM_INTERVALS, "num_intervals")?;
    let overlap = validate_range_f32(req.overlap, MIN_OVERLAP, MAX_OVERLAP, "overlap")?;

    let user_id = req.user_id.clone();
    let filter_str = req.filter.clone();
    let state_clone = state.clone();
    let offset = req.offset;
    let limit = req.limit.max(1);

    let result = tokio::task::spawn_blocking(move || -> Result<MapperResponse, anyhow::Error> {
        let store = get_or_create_slow_store(&state_clone, &user_id)?;
        sync_graph_to_slow_store(&state_clone, &user_id, &store)?;

        let filter = match filter_str.as_str() {
            "embedding_pc1" => MapperFilter::EmbeddingPC1,
            "density" => MapperFilter::Density,
            "eccentricity" => MapperFilter::Eccentricity,
            "neighbor_distance" => MapperFilter::NeighborDistance,
            _ => MapperFilter::CentroidDistance,
        };

        let config = MapperConfig {
            filter,
            num_intervals,
            overlap,
            ..Default::default()
        };

        let graph = compute_mapper(&store, &config)?;

        let total_count = graph.nodes.len();
        let nodes: Vec<MapperNodeSummary> = graph
            .nodes
            .iter()
            .skip(offset)
            .take(limit)
            .map(|n| MapperNodeSummary {
                id: n.id,
                member_names: n.member_names.clone(),
                size: n.size,
                avg_filter_value: n.avg_filter_value,
            })
            .collect();

        let edges: Vec<MapperEdgeSummary> = graph
            .edges
            .iter()
            .map(|e| MapperEdgeSummary {
                from: e.from,
                to: e.to,
                weight: e.weight,
            })
            .collect();

        Ok(MapperResponse {
            total_count,
            nodes,
            edges,
            num_components: graph.num_components,
            num_loops: graph.num_loops,
            flare_count: graph.flare_count,
            branch_count: graph.branch_count,
            filter: graph.filter,
            stats: MapperStatsSummary {
                entity_count: graph.stats.entity_count,
                interval_count: graph.stats.interval_count,
                cluster_count: graph.stats.cluster_count,
                edge_count: graph.stats.edge_count,
                duration_ms: graph.stats.duration_ms,
            },
        })
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Mapper analysis task failed: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(result))
}

// =============================================================================
// STATS (lightweight — no detection, just counts)
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct GapStatsRequest {
    pub user_id: String,
}

#[derive(Debug, Serialize)]
pub struct GapStatsResponse {
    pub entity_count: usize,
    pub edge_count: usize,
}

/// Get lightweight graph stats without running detection. Serves the counts
/// the last sync left in the slow store, up to `graph_mirror_sync_secs` stale; rebuilds
/// only when the store has never been synced, so a fresh store does not report
/// zero until another handler happens to sync it.
///
/// POST /api/gap/stats
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn gap_stats(
    State(state): State<AppState>,
    Json(req): Json<GapStatsRequest>,
) -> Result<Json<GapStatsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let user_id = req.user_id.clone();
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<GapStatsResponse, anyhow::Error> {
        let store = get_or_create_slow_store(&state_clone, &user_id)?;
        if store.never_synced() {
            sync_graph_to_slow_store(&state_clone, &user_id, &store)?;
        }
        Ok(GapStatsResponse {
            entity_count: store.entity_count()?,
            edge_count: store.edge_count()?,
        })
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Gap stats task failed: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(result))
}

// =============================================================================
// HELPERS
// =============================================================================

/// Get the cached SlowStore for a user (creates on first access).
fn get_or_create_slow_store(
    state: &MultiUserMemoryManager,
    user_id: &str,
) -> Result<std::sync::Arc<SlowStore>, anyhow::Error> {
    state
        .get_user_slow_store(user_id)
        .map_err(|e| anyhow::anyhow!("Failed to get slow store for user {user_id}: {e}"))
}

/// Sync graph data from RocksDB (GraphMemory) into SQLite (SlowStore).
/// Serves the existing mirror when a rebuild is already in flight or the last
/// one finished less than the configured `graph_mirror_sync_secs` ago.
fn sync_graph_to_slow_store(
    state: &MultiUserMemoryManager,
    user_id: &str,
    store: &SlowStore,
) -> Result<(), anyhow::Error> {
    let Some(sync) = store.begin_sync(state.server_config().graph_mirror_sync_ttl()) else {
        tracing::debug!("SlowStore sync skipped: rebuild in flight or within TTL");
        return Ok(());
    };

    let graph = state
        .get_user_graph(user_id)
        .map_err(|e| anyhow::anyhow!("Failed to get graph for user {user_id}: {e}"))?;
    // Hold the graph read lock only for the snapshot, not the SQLite writes.
    let (entities, edges) = {
        let graph_guard = graph.read();
        (
            graph_guard.get_all_entities()?,
            graph_guard.get_all_relationships()?,
        )
    };

    sync.rebuild(&entities, &edges)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn key(user: &str) -> GapKey {
        GapKey {
            user_id: user.to_string(),
            min_strength: 0.2f32.to_bits(),
            max_gaps: 50,
            min_embedding_similarity: None,
            star_min_spokes: None,
            orbit_min_cluster_size: None,
        }
    }

    fn result(duration_ms: u64) -> GapDetectionResult {
        GapDetectionResult {
            gaps: Vec::new(),
            type_counts: HashMap::new(),
            duration_ms,
        }
    }

    fn counted(
        runs: &Arc<AtomicUsize>,
        took: Duration,
        duration_ms: u64,
    ) -> impl FnOnce() -> anyhow::Result<GapDetectionResult> + Send + 'static {
        let runs = Arc::clone(runs);
        move || {
            runs.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(took);
            Ok(result(duration_ms))
        }
    }

    #[tokio::test]
    async fn concurrent_requests_share_one_run_and_a_fresh_result_needs_none() {
        let runs = Arc::new(AtomicUsize::new(0));
        let wait = Duration::from_secs(5);
        let took = Duration::from_millis(200);
        let (a, b) = tokio::join!(
            detection(key("gap-shared"), wait, counted(&runs, took, 7)),
            detection(key("gap-shared"), wait, counted(&runs, took, 7)),
        );
        let (Served::Fresh(a), Served::Fresh(b)) = (a.unwrap(), b.unwrap()) else {
            panic!("both requests get the run's result");
        };
        assert_eq!(a.at, b.at);
        assert_eq!(runs.load(Ordering::SeqCst), 1);

        let again = detection(key("gap-shared"), wait, counted(&runs, took, 7))
            .await
            .unwrap();
        assert!(matches!(again, Served::Fresh(d) if d.at == a.at));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_run_outlasting_the_wait_answers_running_and_its_result_is_kept() {
        let runs = Arc::new(AtomicUsize::new(0));
        let short = Duration::from_millis(50);
        let first = detection(
            key("gap-slow"),
            short,
            counted(&runs, Duration::from_millis(300), 1),
        )
        .await
        .unwrap();
        assert!(matches!(first, Served::Running));

        tokio::time::sleep(Duration::from_millis(700)).await;
        let later = detection(key("gap-slow"), short, counted(&runs, Duration::ZERO, 2))
            .await
            .unwrap();
        assert!(matches!(later, Served::Fresh(d) if d.result.duration_ms == 1));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failed_first_run_is_an_error_and_a_failed_refresh_serves_the_last_result() {
        let k = key("gap-fails");
        let wait = Duration::from_secs(5);
        assert!(detection(k.clone(), wait, || anyhow::bail!("no graph"))
            .await
            .is_err());

        GAP_RUNS.lock().entry(k.clone()).or_default().last = Some(Detected {
            result: Arc::new(result(3)),
            at: chrono::Utc::now() - chrono::Duration::seconds(GAP_FRESH_SECS + 1),
        });
        let served = detection(k, wait, || anyhow::bail!("no graph"))
            .await
            .unwrap();
        assert!(matches!(served, Served::Stale(d) if d.result.duration_ms == 3));
    }
}
