//! Visualization Handlers
//!
//! Handlers for brain state visualization and memory graph visualization.
//! Includes live browser-based graph visualization with SSE updates.

use axum::{
    extract::{Extension, Path, Query, State},
    response::{Html, IntoResponse, Json},
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use super::utils::resolve_request_user_id;
use crate::auth::AuthenticatedUser;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::GraphStats as VisualizationStats;
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// Brain state response with memories organized by tier
#[derive(Debug, Serialize)]
pub struct BrainStateResponse {
    pub working_memory: Vec<MemoryNeuron>,
    pub session_memory: Vec<MemoryNeuron>,
    pub longterm_memory: Vec<MemoryNeuron>,
    pub stats: BrainStats,
}

/// Individual memory neuron for visualization
#[derive(Debug, Serialize)]
pub struct MemoryNeuron {
    pub id: String,
    pub content_preview: String,
    pub activation: f32,
    pub importance: f32,
    pub tier: String,
    pub access_count: u32,
    pub created_at: String,
}

/// Brain statistics
#[derive(Debug, Serialize)]
pub struct BrainStats {
    pub total_memories: usize,
    pub working_count: usize,
    pub session_count: usize,
    pub longterm_count: usize,
    pub avg_activation: f32,
    pub avg_importance: f32,
}

/// GET /api/brain/{user_id} - Get brain state visualization
pub async fn get_brain_state(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<BrainStateResponse>, AppError> {
    // SEC: the path names a user; a tenant-bound key may only name its own.
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let mut working_memory = Vec::new();
    let mut session_memory = Vec::new();
    let mut longterm_memory = Vec::new();
    let mut total_activation = 0.0f32;
    let mut total_importance = 0.0f32;

    // Get working memory
    for mem in memory_guard.get_working_memories() {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "working".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        working_memory.push(neuron);
    }

    // Get session memory
    for mem in memory_guard.get_session_memories() {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "session".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        session_memory.push(neuron);
    }

    // Get longterm memory sample
    let longterm_sample = memory_guard.get_longterm_memories(50).unwrap_or_default();
    for mem in longterm_sample {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "longterm".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        longterm_memory.push(neuron);
    }

    let total_count = working_memory.len() + session_memory.len() + longterm_memory.len();
    let stats = BrainStats {
        total_memories: total_count,
        working_count: working_memory.len(),
        session_count: session_memory.len(),
        longterm_count: longterm_memory.len(),
        avg_activation: if total_count > 0 {
            total_activation / total_count as f32
        } else {
            0.0
        },
        avg_importance: if total_count > 0 {
            total_importance / total_count as f32
        } else {
            0.0
        },
    };

    Ok(Json(BrainStateResponse {
        working_memory,
        session_memory,
        longterm_memory,
        stats,
    }))
}

/// GET /api/visualization/{user_id}/stats - Get visualization statistics
pub async fn get_visualization_stats(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<VisualizationStats>, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard.get_visualization_stats();

    Ok(Json(stats))
}

/// GET /api/visualization/{user_id}/dot - Export graph as DOT format
pub async fn get_visualization_dot(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<String, AppError> {
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let dot = memory_guard.export_visualization_dot();

    Ok(dot)
}

/// Request to build visualization
#[derive(Debug, Deserialize)]
pub struct BuildVisualizationRequest {
    pub user_id: String,
}

/// POST /api/visualization/build - Build visualization graph
pub async fn build_visualization(
    State(state): State<AppState>,
    Json(req): Json<BuildVisualizationRequest>,
) -> Result<Json<VisualizationStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard
        .build_visualization_graph()
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// Query parameters for graph view
#[derive(Debug, Deserialize)]
pub struct GraphViewParams {
    pub user_id: Option<String>,
}

/// Graph node for d3.js visualization
#[derive(Debug, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String, // "memory", "entity"
    pub tier: String,      // "L1", "L2", "L3" or memory tier
    pub strength: f32,
    pub size: f32,
}

/// Graph edge for d3.js visualization
#[derive(Debug, Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub edge_type: String,
    pub tier: String, // "L1", "L2", "L3"
    pub strength: f32,
    /// Distinct sources whose strengthening this edge rests on.
    pub distinct_sources: usize,
    /// When a source new to this edge last strengthened it.
    pub last_corroborated: Option<chrono::DateTime<chrono::Utc>>,
}

/// Graph data response for d3.js
#[derive(Debug, Serialize)]
pub struct GraphDataResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub stats: GraphDataStats,
}

/// Graph statistics
#[derive(Debug, Serialize)]
pub struct GraphDataStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub l1_edges: usize,
    pub l2_edges: usize,
    pub l3_edges: usize,
}

/// GET /graph/view - Serve interactive graph visualization HTML
pub async fn graph_view(Query(params): Query<GraphViewParams>) -> impl IntoResponse {
    let user_id = params.user_id.unwrap_or_else(|| "default".to_string());
    let nonce = fresh_csp_nonce();
    (
        [(
            "Content-Security-Policy",
            format!(
                "default-src 'none'; script-src 'nonce-{nonce}' \
                 https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; \
                 style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; \
                 base-uri 'none'; form-action 'none'"
            ),
        )],
        Html(generate_graph_html(&user_id, &nonce)),
    )
}

/// Per-response nonce; 128 bits from the OS RNG.
fn fresh_csp_nonce() -> String {
    use base64::Engine;
    let mut bytes = [0u8; 16];
    rand::Rng::fill(&mut rand::thread_rng(), &mut bytes[..]);
    base64::engine::general_purpose::STANDARD_NO_PAD.encode(bytes)
}

/// GET /api/graph/data/{user_id} - Get graph data as JSON for d3.js
pub async fn get_graph_data(
    State(state): State<AppState>,
    authenticated_user: Option<Extension<AuthenticatedUser>>,
    Path(user_id): Path<String>,
) -> Result<Json<GraphDataResponse>, AppError> {
    // SEC: the path names a user; a tenant-bound key may only name its own.
    let user_id = resolve_request_user_id(
        Some(&user_id),
        authenticated_user.as_ref().map(|extension| &extension.0),
    )?;
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let graph = memory_guard
        .graph_memory()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Graph memory not initialized")))?;
    let graph_guard = graph.read();

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut l1_count = 0;
    let mut l2_count = 0;
    let mut l3_count = 0;

    // Get entities as nodes
    if let Ok(entities) = graph_guard.get_all_entities() {
        for entity in entities.iter().take(200) {
            let tier_label = entity
                .labels
                .first()
                .map(|l| l.as_str().to_string())
                .unwrap_or_else(|| "entity".to_string());
            nodes.push(GraphNode {
                id: entity.uuid.to_string(),
                label: entity.name.clone(),
                node_type: "entity".to_string(),
                tier: tier_label,
                strength: 1.0,
                size: 10.0,
            });
        }
    }

    // Get relationships as edges - sample from each tier for visibility
    if let Ok(relationships) = graph_guard.get_all_relationships() {
        use crate::graph_memory::EdgeTier;

        // Separate by tier for proportional sampling
        let l1_edges: Vec<_> = relationships
            .iter()
            .filter(|r| matches!(r.tier, EdgeTier::L1Working))
            .take(200)
            .collect();
        let l2_edges: Vec<_> = relationships
            .iter()
            .filter(|r| matches!(r.tier, EdgeTier::L2Episodic))
            .take(200)
            .collect();
        let l3_edges: Vec<_> = relationships
            .iter()
            .filter(|r| matches!(r.tier, EdgeTier::L3Semantic))
            .take(200)
            .collect();

        // Add edges from each tier
        for rel in l1_edges
            .iter()
            .chain(l2_edges.iter())
            .chain(l3_edges.iter())
        {
            let tier_str = match rel.tier {
                EdgeTier::L1Working => {
                    l1_count += 1;
                    "L1"
                }
                EdgeTier::L2Episodic => {
                    l2_count += 1;
                    "L2"
                }
                EdgeTier::L3Semantic => {
                    l3_count += 1;
                    "L3"
                }
            };

            edges.push(GraphEdge {
                source: rel.from_entity.to_string(),
                target: rel.to_entity.to_string(),
                edge_type: rel.relation_type.as_str().to_string(),
                tier: tier_str.to_string(),
                strength: rel.effective_strength(),
                distinct_sources: rel.distinct_sources(),
                last_corroborated: rel.last_corroborated,
            });
        }
    }

    // Add memory nodes and connect them to their entities
    let memories = memory_guard.get_longterm_memories(100).unwrap_or_default();
    let entity_ids: std::collections::HashSet<String> =
        nodes.iter().map(|n| n.id.clone()).collect();

    for mem in memories {
        let mem_id = mem.id.0.to_string();
        nodes.push(GraphNode {
            id: mem_id.clone(),
            label: mem.experience.content.chars().take(30).collect::<String>() + "...",
            node_type: "memory".to_string(),
            tier: "longterm".to_string(),
            strength: mem.importance(),
            size: 6.0 + mem.importance() * 8.0,
        });

        // Connect memory to its entities
        for entity_id in mem.entity_ids() {
            let entity_id_str = entity_id.to_string();
            if entity_ids.contains(&entity_id_str) {
                edges.push(GraphEdge {
                    source: mem_id.clone(),
                    target: entity_id_str,
                    edge_type: "mentions".to_string(),
                    tier: "L2".to_string(),
                    strength: 0.5,
                    distinct_sources: 0,
                    last_corroborated: None,
                });
            }
        }
    }

    Ok(Json(GraphDataResponse {
        stats: GraphDataStats {
            total_nodes: nodes.len(),
            total_edges: edges.len(),
            l1_edges: l1_count,
            l2_edges: l2_count,
            l3_edges: l3_count,
        },
        nodes,
        edges,
    }))
}

/// Generate the HTML page for graph visualization (includes 2D/3D toggle)
fn generate_graph_html(user_id: &str, nonce: &str) -> String {
    let html = include_str!("graph_view.html");
    // The value lands inside an inline <script>; a literal `</script>` in it would close the
    // element and render whatever follows as markup on this public page.
    let user_id_json = serde_json::to_string(user_id)
        .unwrap_or_else(|_| "\"default\"".to_string())
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    html.replace("{{USER_ID_JSON}}", &user_id_json)
        .replace("{{CSP_NONCE}}", nonce)
}

#[cfg(test)]
mod tests {
    use super::generate_graph_html;

    #[test]
    fn a_user_id_cannot_close_the_inline_script() {
        let html = generate_graph_html("</script><meta http-equiv=refresh content=0>&", "n");
        assert!(!html.contains("</script><meta"));
        assert!(html.contains("\\u003c/script\\u003e\\u003cmeta"));
    }
}
