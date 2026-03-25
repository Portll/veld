//! Structural Graph Analysis Handlers
//!
//! REST API endpoints exposing structural analysis primitives on the knowledge graph:
//! gap detection, Voronoi cell analysis, persistent homology, and Mapper topology.
//! These return raw structural data — interpretation is left to consumers.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::gap_topology::{GapDetectionConfig, GapDetector};
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
const MAX_ENTITIES: usize = 5000;
const SYNC_TTL_SECS: u64 = 30;

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

/// Detect structural gaps in the knowledge graph.
///
/// POST /api/gap/analyze
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn analyze_gaps(
    State(state): State<AppState>,
    Json(req): Json<GapAnalysisRequest>,
) -> Result<Json<GapAnalysisResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let min_strength = validate_range_f32(req.min_edge_strength, 0.0, 1.0, "min_edge_strength")?;
    let max_gaps =
        validate_range_usize(req.max_gaps_per_type, MAX_GAPS_PER_TYPE, "max_gaps_per_type")?;

    let user_id = req.user_id.clone();
    let state_clone = state.clone();

    let offset = req.offset;
    let limit = validate_range_usize(req.limit.max(1), MAX_GAPS_PER_TYPE, "limit")?;

    let result = tokio::task::spawn_blocking(move || -> Result<GapAnalysisResponse, anyhow::Error> {
        let store = get_or_create_slow_store(&state_clone, &user_id)?;
        sync_graph_to_slow_store(&state_clone, &user_id, &store)?;

        let config = GapDetectionConfig {
            min_edge_strength: min_strength,
            max_gaps_per_type: max_gaps,
            ..Default::default()
        };

        let result = GapDetector::detect(&store, &config)?;
        let total_count = result.gaps.len();

        let gaps: Vec<GapSummary> = result
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
            .collect();

        Ok(GapAnalysisResponse {
            gaps,
            total_count,
            type_counts: result.type_counts,
            duration_ms: result.duration_ms,
        })
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Gap analysis task failed: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(result))
}

// =============================================================================
// VORONOI ANALYSIS
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct VoronoiAnalysisRequest {
    pub user_id: String,
    #[serde(default = "default_k")]
    pub k: usize,
}

fn default_k() -> usize {
    8
}

#[derive(Debug, Serialize)]
pub struct VoronoiAnalysisResponse {
    pub entity_count: usize,
    pub voids_found: usize,
    pub avg_anisotropy: f32,
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

    let result = tokio::task::spawn_blocking(move || -> Result<VoronoiAnalysisResponse, anyhow::Error> {
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
        most_isolated.truncate(10);

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
}

fn default_persistence_step() -> f32 {
    0.1
}
fn default_min_persistence() -> f32 {
    0.15
}

#[derive(Debug, Serialize)]
pub struct PersistenceResponse {
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
    let min_persistence =
        validate_range_f32(req.min_persistence, MIN_PERSISTENCE, MAX_STEP_SIZE, "min_persistence")?;

    let user_id = req.user_id.clone();
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<PersistenceResponse, anyhow::Error> {
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

        let nodes: Vec<MapperNodeSummary> = graph
            .nodes
            .iter()
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

/// Get lightweight graph stats without running detection.
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
        sync_graph_to_slow_store(&state_clone, &user_id, &store)?;
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

/// Get or create a SlowStore for a user.
fn get_or_create_slow_store(
    state: &MultiUserMemoryManager,
    user_id: &str,
) -> Result<SlowStore, anyhow::Error> {
    let user_path = state.base_path.join(user_id);
    std::fs::create_dir_all(&user_path)?;
    let db_path = user_path.join("slow_store.db");
    SlowStore::open(&db_path)
}

/// Sync graph data from RocksDB (GraphMemory) into SQLite (SlowStore).
/// Skips sync if the last sync was less than SYNC_TTL_SECS ago.
fn sync_graph_to_slow_store(
    state: &MultiUserMemoryManager,
    user_id: &str,
    store: &SlowStore,
) -> Result<(), anyhow::Error> {
    if !store.should_sync(SYNC_TTL_SECS) {
        tracing::debug!("SlowStore sync skipped (TTL)");
        return Ok(());
    }

    let graph = state
        .get_user_graph(user_id)
        .map_err(|e| anyhow::anyhow!("Failed to get graph for user {user_id}: {e}"))?;
    let graph_guard = graph.read();

    let entities = graph_guard.get_all_entities()?;
    let edges = graph_guard.get_all_relationships()?;

    store.sync_from_graph(&entities, &edges)?;
    Ok(())
}
