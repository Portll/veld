//! Gap Analysis and Thought Surfacing Handlers
//!
//! REST API endpoints for gap topology detection, golden feature generation,
//! and thought surfacing. These enable the knowledge graph to reason about
//! its own structural gaps — what it doesn't know and what shape the missing
//! information has.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::gap_topology::{GapDetectionConfig, GapScope};
use crate::memory::mapper::{compute_mapper, MapperConfig, MapperFilter};
use crate::memory::persistence::{self, PersistenceConfig};
use crate::memory::slow_store::SlowStore;
use crate::memory::thoughts::{ThoughtEngine, ThoughtReport};
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
// REQUEST / RESPONSE TYPES
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct GapAnalysisRequest {
    pub user_id: String,
    #[serde(default = "default_scope")]
    pub scope: String,
    #[serde(default = "default_min_strength")]
    pub min_edge_strength: f32,
    #[serde(default = "default_max_gaps")]
    pub max_gaps_per_type: usize,
}

fn default_scope() -> String {
    "content".to_string()
}
fn default_min_strength() -> f32 {
    0.2
}
fn default_max_gaps() -> usize {
    50
}

#[derive(Debug, Serialize)]
pub struct GapAnalysisResponse {
    pub thoughts: Vec<ThoughtSummary>,
    pub stats: ThoughtStatsResponse,
}

#[derive(Debug, Serialize)]
pub struct ThoughtSummary {
    pub id: String,
    pub kind: String,
    pub scope: String,
    pub confidence: f32,
    pub description: String,
    pub hypothesis: Option<String>,
    pub impact_score: f32,
    pub entity_names: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ThoughtStatsResponse {
    pub gaps_detected: usize,
    pub golden_features_generated: usize,
    pub thoughts_generated: usize,
    pub fractal_patterns_found: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct GetThoughtsRequest {
    pub user_id: String,
    #[serde(default = "default_thought_limit")]
    pub limit: usize,
}

fn default_thought_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct GetThoughtsResponse {
    pub thoughts: Vec<ThoughtSummary>,
}

#[derive(Debug, Deserialize)]
pub struct DismissThoughtRequest {
    pub user_id: String,
    pub thought_id: String,
}

#[derive(Debug, Serialize)]
pub struct DismissThoughtResponse {
    pub dismissed: bool,
}

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
    pub planet_x_found: usize,
    pub avg_anisotropy: f32,
    pub most_isolated: Vec<IsolatedEntity>,
    pub voids: Vec<VoidSummary>,
    pub planet_x: Vec<PlanetXSummary>,
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
    pub implied_topic: String,
}

#[derive(Debug, Serialize)]
pub struct PlanetXSummary {
    pub evidence_entities: Vec<String>,
    pub convergence_count: usize,
    pub confidence: f32,
    pub description: String,
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

// =============================================================================
// HANDLERS
// =============================================================================

/// Run full gap analysis: sync graph → detect gaps → generate thoughts.
///
/// POST /api/gap/analyze
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn analyze_gaps(
    State(state): State<AppState>,
    Json(req): Json<GapAnalysisRequest>,
) -> Result<Json<GapAnalysisResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let min_strength = validate_range_f32(req.min_edge_strength, 0.0, 1.0, "min_edge_strength")?;
    let max_gaps = validate_range_usize(req.max_gaps_per_type, MAX_GAPS_PER_TYPE, "max_gaps_per_type")?;

    let user_id = req.user_id.clone();
    let scope = match req.scope.as_str() {
        "codebase" => GapScope::Codebase,
        "schema" => GapScope::Schema,
        _ => GapScope::Content,
    };
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<ThoughtReport, anyhow::Error> {
        // Get or create slow store
        let store = get_or_create_slow_store(&state_clone, &user_id)?;

        // Sync graph data into SQLite
        sync_graph_to_slow_store(&state_clone, &user_id, &store)?;

        // Run gap detection + thought generation
        let config = GapDetectionConfig {
            min_edge_strength: min_strength,
            max_gaps_per_type: max_gaps,
            scope,
            ..Default::default()
        };

        ThoughtEngine::generate(&store, &config)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Gap analysis task failed: {e}")))?
    .map_err(AppError::Internal)?;

    let thoughts: Vec<ThoughtSummary> = result
        .thoughts
        .iter()
        .map(|t| ThoughtSummary {
            id: t.id.clone(),
            kind: t.kind.as_str().to_string(),
            scope: t.scope.as_str().to_string(),
            confidence: t.confidence,
            description: t.description.clone(),
            hypothesis: t.hypothesis.clone(),
            impact_score: t.impact_score,
            entity_names: t.entities.iter().map(|(_, name)| name.clone()).collect(),
        })
        .collect();

    Ok(Json(GapAnalysisResponse {
        thoughts,
        stats: ThoughtStatsResponse {
            gaps_detected: result.stats.gaps_detected,
            golden_features_generated: result.stats.golden_features_generated,
            thoughts_generated: result.stats.thoughts_generated,
            fractal_patterns_found: result.stats.fractal_patterns_found,
            duration_ms: result.stats.duration_ms,
        },
    }))
}

/// Get previously generated thoughts.
///
/// POST /api/gap/thoughts
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn get_thoughts(
    State(state): State<AppState>,
    Json(req): Json<GetThoughtsRequest>,
) -> Result<Json<GetThoughtsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let user_id = req.user_id.clone();
    let limit = req.limit;
    let state_clone = state.clone();

    let thoughts = tokio::task::spawn_blocking(move || -> Result<Vec<ThoughtSummary>, anyhow::Error> {
        let store = get_or_create_slow_store(&state_clone, &user_id)?;
        let active = ThoughtEngine::get_active_thoughts(&store, limit)?;

        Ok(active
            .into_iter()
            .map(|t| ThoughtSummary {
                id: t.id,
                kind: t.kind.as_str().to_string(),
                scope: t.scope.as_str().to_string(),
                confidence: t.confidence,
                description: t.description,
                hypothesis: t.hypothesis,
                impact_score: t.impact_score,
                entity_names: t.entities.into_iter().map(|(_, name)| name).collect(),
            })
            .collect())
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Get thoughts task failed: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(GetThoughtsResponse { thoughts }))
}

/// Dismiss a thought (mark as not useful).
///
/// POST /api/gap/dismiss
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn dismiss_thought(
    State(state): State<AppState>,
    Json(req): Json<DismissThoughtRequest>,
) -> Result<Json<DismissThoughtResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let user_id = req.user_id.clone();
    let thought_id = req.thought_id.clone();
    let state_clone = state.clone();

    tokio::task::spawn_blocking(move || -> Result<(), anyhow::Error> {
        let store = get_or_create_slow_store(&state_clone, &user_id)?;
        store.dismiss_thought(&thought_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Dismiss task failed: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(DismissThoughtResponse { dismissed: true }))
}

/// Run Voronoi analysis on the knowledge graph embedding space.
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

        // Build response
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
                implied_topic: v.implied_topic.clone(),
            })
            .collect();

        let planet_x: Vec<PlanetXSummary> = analysis
            .planet_x_candidates
            .iter()
            .map(|px| PlanetXSummary {
                evidence_entities: px.evidence_entities.iter().map(|(_, n)| n.clone()).collect(),
                convergence_count: px.convergence_count,
                confidence: px.confidence,
                description: px.predicted_description.clone(),
            })
            .collect();

        Ok(VoronoiAnalysisResponse {
            entity_count: analysis.stats.entity_count,
            voids_found: analysis.voids.len(),
            planet_x_found: analysis.planet_x_candidates.len(),
            avg_anisotropy: analysis.stats.avg_anisotropy,
            most_isolated,
            voids,
            planet_x,
        })
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Voronoi analysis task failed: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(result))
}

// =============================================================================
// PERSISTENCE
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
    let min_persistence = validate_range_f32(req.min_persistence, MIN_PERSISTENCE, MAX_STEP_SIZE, "min_persistence")?;

    let user_id = req.user_id.clone();
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<PersistenceResponse, anyhow::Error> {
        let store = get_or_create_slow_store(&state_clone, &user_id)?;
        sync_graph_to_slow_store(&state_clone, &user_id, &store)?;

        let config = PersistenceConfig {
            step_size,
            min_persistence,
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

/// Run Mapper topological analysis on the knowledge graph.
///
/// POST /api/gap/mapper
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn mapper_analysis(
    State(state): State<AppState>,
    Json(req): Json<MapperRequest>,
) -> Result<Json<MapperResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    let num_intervals = validate_range_usize(req.num_intervals.max(1), MAX_NUM_INTERVALS, "num_intervals")?;
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
// HELPERS
// =============================================================================

/// Get or create a SlowStore for a user.
///
/// The slow store lives at `{base_path}/{user_id}/slow_store.db`.
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
fn sync_graph_to_slow_store(
    state: &MultiUserMemoryManager,
    user_id: &str,
    store: &SlowStore,
) -> Result<(), anyhow::Error> {
    let graph = state
        .get_user_graph(user_id)
        .map_err(|e| anyhow::anyhow!("Failed to get graph for user {user_id}: {e}"))?;
    let graph_guard = graph.read();

    let entities = graph_guard.get_all_entities()?;
    let edges = graph_guard.get_all_relationships()?;

    store.sync_from_graph(&entities, &edges)?;
    Ok(())
}
